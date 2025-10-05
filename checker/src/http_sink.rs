use crate::Mode;
use crate::parser::SampleJson;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::sync::{mpsc, watch};
use tracing::{debug, error, info, warn};

#[derive(Serialize)]
struct HttpSample {
    t_ms: i64,
    ts_unix_ms: i64,
    adc: i64,
    #[serde(skip_serializing_if = "is_false")]
    lead_off: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    hp: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mv: Option<f32>,
}

#[inline]
fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Serialize)]
struct IngestPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    recording_id: Option<String>,
    samples: Vec<HttpSample>,
}

#[derive(Deserialize)]
struct ModePayload {
    mode: String,
}

pub async fn run_http_sink(
    http_url: String,
    mut rx: mpsc::Receiver<SampleJson>,
    mut mode_rx: watch::Receiver<Mode>,
    batch_size: usize,
    batch_interval_ms: u64,
) {
    let client = reqwest::Client::builder()
        .user_agent(concat!(
            env!("CARGO_PKG_NAME"),
            "/",
            env!("CARGO_PKG_VERSION")
        ))
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(10))
        .pool_idle_timeout(Some(Duration::from_secs(30)))
        .pool_max_idle_per_host(1)
        .no_proxy()
        .build()
        .unwrap_or_else(|err| {
            warn!("Falling back to default HTTP client (builder error: {err})");
            reqwest::Client::new()
        });
    let mut buffer: Vec<HttpSample> = Vec::new();

    let mut zero_count: usize = 0;
    let zero_threshold: usize = 5;
    let mut current_mode = *mode_rx.borrow();

    let mut interval = tokio::time::interval(Duration::from_millis(batch_interval_ms));

    loop {
        tokio::select! {
            maybe_sample = rx.recv() => {
                match maybe_sample {
                    Some(sample) => {
                        if current_mode == Mode::Ecg {
                            match sample.adc {
                                Some(v) if v == 0 => zero_count += 1,
                                Some(_) | None => zero_count = 0,
                            }
                        } else {
                            zero_count = 0;
                        }

                        let mut lead = sample.lead_off;
                        if current_mode == Mode::Ecg && zero_count >= zero_threshold {
                            lead = true;
                        }

                        let t_ms_i64 = sample.t_ms as i64;
                        let ts_unix_ms_i64 = sample.ts_unix_ms as i64;
                        let adc_val = sample.adc.unwrap_or(0) as i64;

                        buffer.push(HttpSample {
                            t_ms: t_ms_i64,
                            ts_unix_ms: ts_unix_ms_i64,
                            adc: adc_val,
                            lead_off: lead,
                            hp: sample.hp,
                            mv: sample.mv,
                        });

                        if buffer.len() >= batch_size {
                            flush_buffer(&client, &http_url, &mut buffer).await;
                        }
                    }
                    None => {
                        if !buffer.is_empty() {
                            flush_buffer(&client, &http_url, &mut buffer).await;
                        }
                        return;
                    }
                }
            }
            _ = interval.tick() => {
                if !buffer.is_empty() {
                    flush_buffer(&client, &http_url, &mut buffer).await;
                }
            }
            changed = mode_rx.changed() => {
                if changed.is_ok() {
                    current_mode = *mode_rx.borrow();
                    zero_count = 0;
                }
            }
        }
    }
}

async fn flush_buffer(client: &reqwest::Client, url: &str, buffer: &mut Vec<HttpSample>) {
    if buffer.is_empty() {
        return;
    }
    let payload = IngestPayload {
        recording_id: None,
        samples: std::mem::take(buffer),
    };

    match client.post(url).json(&payload).send().await {
        Ok(resp) => {
            if !resp.status().is_success() {
                warn!("HTTP ingest {} returned status {}", url, resp.status());
            }
        }
        Err(e) => {
            error!(
                target: "goida_bridge::http_sink",
                url = %url,
                error = %e,
                debug_error = ?e,
                is_connect = e.is_connect(),
                is_timeout = e.is_timeout(),
                "Failed to send HTTP ingest"
            );
        }
    }
}

#[allow(dead_code)]
pub async fn run_control_poll(
    control_url: String,
    poll_interval_ms: u64,
    mode_tx: watch::Sender<Mode>,
    cmd_tx: mpsc::UnboundedSender<Mode>,
) {
    let client = reqwest::Client::builder()
        .user_agent(concat!(
            env!("CARGO_PKG_NAME"),
            "/",
            env!("CARGO_PKG_VERSION")
        ))
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(10))
        .pool_idle_timeout(Some(Duration::from_secs(30)))
        .pool_max_idle_per_host(1)
        .no_proxy()
        .build()
        .unwrap_or_else(|err| {
            warn!("Falling back to default HTTP client for control poll (builder error: {err})");
            reqwest::Client::new()
        });
    let mut interval = tokio::time::interval(Duration::from_millis(poll_interval_ms));
    let mut current_mode = *mode_tx.borrow();

    loop {
        interval.tick().await;
        match client.get(&control_url).send().await {
            Ok(resp) => {
                if resp.status().is_success() {
                    match resp.json::<ModePayload>().await {
                        Ok(payload) => {
                            if let Some(mode) = Mode::from_runtime_token(&payload.mode) {
                                if mode != current_mode {
                                    info!(
                                        mode = mode.label(),
                                        "Received backend mode update via control poll"
                                    );
                                    current_mode = mode;
                                    if mode_tx.send(mode).is_err() {
                                        debug!("Mode watch channel closed; stopping control poll");
                                        return;
                                    }
                                    if cmd_tx.send(mode).is_err() {
                                        debug!("Command channel closed; stopping control poll");
                                        return;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Failed to parse control response JSON: {}", e);
                        }
                    }
                } else {
                    warn!(
                        "Control poll {} returned status {}",
                        control_url,
                        resp.status()
                    );
                }
            }
            Err(e) => {
                warn!("Error fetching control URL {}: {}", control_url, e);
            }
        }
    }
}
