use anyhow::{Context, Result, anyhow};
use futures_util::StreamExt;
use goida_bridge::{Mode, http_sink, parser::SampleJson};
use portpicker::pick_unused_port;
use serde_json::Value;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;
use tokio::net::TcpStream;
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, watch};
use tokio::time::sleep;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async, tungstenite::Message};

type TestWebSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;

fn python_command() -> Command {
    if let Ok(python) = env::var("PYTHON") {
        return Command::new(python);
    }

    if let Ok(virtual_env) = env::var("VIRTUAL_ENV") {
        let mut candidate = PathBuf::from(&virtual_env);
        if cfg!(windows) {
            candidate.push("Scripts");
            candidate.push("python.exe");
        } else {
            candidate.push("bin");
            candidate.push("python");
        }

        if candidate.exists() {
            return Command::new(candidate);
        }
    }

    if cfg!(windows) {
        Command::new("python")
    } else {
        Command::new("python3")
    }
}

struct BackendServer {
    child: Child,
    port: u16,
}

impl BackendServer {
    async fn spawn() -> Result<Self> {
        let port = pick_unused_port().ok_or_else(|| anyhow!("no free port available"))?;
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or_else(|| anyhow!("unable to find project root"))?
            .to_path_buf();

        ensure_backend_dependencies(&project_root).await?;

        let mut cmd = python_command();
        cmd.args([
            "-m",
            "uvicorn",
            "backend.app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--log-level",
            "warning",
        ]);
        cmd.current_dir(&project_root);
        cmd.env("PYTHONPATH", &project_root);
        cmd.stdout(std::process::Stdio::null());
        cmd.stderr(std::process::Stdio::piped());

        let mut child = cmd.spawn().context("failed to spawn backend server")?;

        let client = reqwest::Client::new();
        let base_url = format!("http://127.0.0.1:{port}/");
        let deadline = Instant::now() + Duration::from_secs(15);

        loop {
            if Instant::now() > deadline {
                if let Some(mut stderr) = child.stderr.take() {
                    let mut buf = Vec::new();
                    let _ = stderr.read_to_end(&mut buf).await;
                    let output = String::from_utf8_lossy(&buf).to_string();
                    let _ = child.start_kill();
                    anyhow::bail!("backend did not start in time: {output}");
                }
                let _ = child.start_kill();
                anyhow::bail!("backend did not start in time");
            }

            if let Some(status) = child.try_wait().context("failed to poll backend child")? {
                if !status.success() {
                    if let Some(mut stderr) = child.stderr.take() {
                        let mut buf = Vec::new();
                        let _ = stderr.read_to_end(&mut buf).await;
                        let output = String::from_utf8_lossy(&buf).to_string();
                        anyhow::bail!("backend exited early: {output}");
                    }
                    anyhow::bail!("backend exited early with status {status}");
                }
            }

            match client.get(&base_url).send().await {
                Ok(resp) if resp.status().is_success() => break,
                _ => sleep(Duration::from_millis(100)).await,
            }
        }

        Ok(Self { child, port })
    }

    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    fn ingest_url(&self) -> String {
        format!("{}/api/ingest", self.base_url())
    }

    fn recordings_url(&self) -> String {
        format!("{}/api/recordings/start", self.base_url())
    }

    fn ws_url(&self, mode: &str) -> String {
        format!("ws://127.0.0.1:{}/ws/{}", self.port, mode)
    }

    async fn shutdown(mut self) -> Result<()> {
        if self.child.try_wait()?.is_none() {
            let _ = self.child.start_kill();
            let _ = tokio::time::timeout(Duration::from_secs(5), self.child.wait()).await;
        }
        Ok(())
    }
}

impl Drop for BackendServer {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

async fn ensure_backend_dependencies(project_root: &Path) -> Result<()> {
    let mut check_cmd = python_command();
    check_cmd.args(["-c", "import fastapi, uvicorn"]);
    check_cmd.env("PYTHONPATH", project_root);
    check_cmd.stdout(Stdio::null());
    check_cmd.stderr(Stdio::null());

    if check_cmd.status().await?.success() {
        return Ok(());
    }

    let mut pip_check_cmd = python_command();
    pip_check_cmd.args(["-m", "pip", "--version"]);
    pip_check_cmd.env("PYTHONPATH", project_root);
    pip_check_cmd.stdout(Stdio::null());
    pip_check_cmd.stderr(Stdio::null());

    if !pip_check_cmd.status().await?.success() {
        let mut ensure_cmd = python_command();
        ensure_cmd.args(["-m", "ensurepip", "--upgrade"]);
        ensure_cmd.env("PYTHONPATH", project_root);

        let status = ensure_cmd
            .status()
            .await
            .context("failed to bootstrap pip with ensurepip")?;

        if !status.success() {
            anyhow::bail!("python ensurepip failed with status {status}");
        }
    }

    let mut install_cmd = python_command();
    install_cmd.args([
        "-m",
        "pip",
        "install",
        "--quiet",
        "--disable-pip-version-check",
        "-r",
        "backend/requirements.txt",
    ]);
    install_cmd.current_dir(project_root);

    let status = install_cmd
        .status()
        .await
        .context("failed to run pip to install backend requirements")?;

    if status.success() {
        Ok(())
    } else {
        anyhow::bail!("pip install backend dependencies failed with status {status}");
    }
}

fn make_sample(t_ms: u128, adc: u16, mv: f32) -> SampleJson {
    SampleJson {
        t_ms,
        ts_unix_ms: 1_700_000_000_000,
        adc: Some(adc),
        lead_off: false,
        hp: None,
        mv: Some(mv),
    }
}

async fn expect_batch(ws: &mut TestWebSocket, expected: &[f64]) -> Result<()> {
    let msg = tokio::time::timeout(Duration::from_secs(5), ws.next())
        .await
        .context("websocket timed out")?
        .context("websocket closed")?;

    let frame = msg.map_err(|e| anyhow!("websocket receive failed: {e}"))?;
    let text = match frame {
        Message::Text(t) => t,
        Message::Binary(b) => String::from_utf8(b).context("binary frame was not valid UTF-8")?,
        other => anyhow::bail!("unexpected websocket frame: {other:?}"),
    };

    let payload: Value = serde_json::from_str(&text).context("invalid JSON payload")?;
    let samples = payload
        .get("samples")
        .and_then(|v| v.as_array())
        .context("payload missing samples array")?;

    assert_eq!(samples.len(), expected.len(), "unexpected samples length");
    for (value, expected_value) in samples.iter().zip(expected.iter()) {
        let actual = value
            .as_f64()
            .or_else(|| value.as_i64().map(|v| v as f64))
            .context("sample was not numeric")?;
        assert!(
            (actual - expected_value).abs() < 1e-3,
            "sample mismatch: {actual} vs {expected_value}"
        );
    }

    let mode = payload
        .get("mode")
        .and_then(|v| v.as_str())
        .context("payload missing mode")?;
    assert_eq!(mode, "ecg");
    Ok(())
}

async fn start_recording(server: &BackendServer) -> Result<()> {
    let client = reqwest::Client::new();
    client
        .post(server.recordings_url())
        .send()
        .await
        .context("failed to call /api/recordings/start")?
        .error_for_status()
        .context("backend rejected recording start")?;
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn streaming_happy_path() -> Result<()> {
    let server = BackendServer::spawn().await?;
    start_recording(&server).await?;

    let (mut ws, _) = connect_async(server.ws_url("ecg")).await?;

    let (tx, rx) = mpsc::channel(16);
    let (_mode_tx, mode_rx) = watch::channel(Mode::Ecg);
    let ingest_url = server.ingest_url();
    let sink_handle = tokio::spawn(http_sink::run_http_sink(ingest_url, rx, mode_rx, 2, 50));

    tx.send(make_sample(1, 100, 1.23)).await?;
    tx.send(make_sample(2, 200, 2.34)).await?;

    expect_batch(&mut ws, &[1.23, 2.34]).await?;

    drop(tx);
    sink_handle.await?;
    ws.close(None).await.ok();
    server.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn streaming_survives_abrupt_disconnect() -> Result<()> {
    let server = BackendServer::spawn().await?;
    start_recording(&server).await?;

    let (ws_stale, _) = connect_async(server.ws_url("ecg")).await?;
    drop(ws_stale);
    sleep(Duration::from_millis(50)).await;

    let (mut ws_active, _) = connect_async(server.ws_url("ecg")).await?;

    let (tx, rx) = mpsc::channel(16);
    let (_mode_tx, mode_rx) = watch::channel(Mode::Ecg);
    let sink_handle = tokio::spawn(http_sink::run_http_sink(
        server.ingest_url(),
        rx,
        mode_rx,
        1,
        50,
    ));

    tx.send(make_sample(10, 300, 3.21)).await?;
    expect_batch(&mut ws_active, &[3.21]).await?;

    tx.send(make_sample(11, 301, 4.56)).await?;
    expect_batch(&mut ws_active, &[4.56]).await?;

    drop(tx);
    sink_handle.await?;
    ws_active.close(None).await.ok();
    server.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn streaming_recovers_after_ingest_failure() -> Result<()> {
    let server = BackendServer::spawn().await?;

    let (tx, rx) = mpsc::channel(16);
    let (_mode_tx, mode_rx) = watch::channel(Mode::Ecg);
    let sink_handle = tokio::spawn(http_sink::run_http_sink(
        server.ingest_url(),
        rx,
        mode_rx,
        2,
        50,
    ));

    tx.send(make_sample(50, 512, 5.43)).await?;
    sleep(Duration::from_millis(200)).await;

    start_recording(&server).await?;
    let (mut ws, _) = connect_async(server.ws_url("ecg")).await?;

    tx.send(make_sample(60, 400, 6.78)).await?;
    tx.send(make_sample(61, 401, 7.89)).await?;

    expect_batch(&mut ws, &[6.78, 7.89]).await?;

    drop(tx);
    sink_handle.await?;
    ws.close(None).await.ok();
    server.shutdown().await?;
    Ok(())
}
