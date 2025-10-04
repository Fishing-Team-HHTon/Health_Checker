use anyhow::{Context, Result};
use clap::{ArgAction, Parser, ValueEnum};
use std::{
    fs::File,
    io::Write,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::io::{self, AsyncBufReadExt, AsyncReadExt, AsyncWriteExt};
use tokio::sync::{mpsc, watch};
use tokio_serial::SerialPortBuilderExt;
use tracing::{debug, error, info, trace, warn};
use tracing_subscriber::EnvFilter;

use serialport::{ClearBuffer, SerialPort};

mod filters;
mod parser;
mod serial;

use filters::{HighPass1, MovingAvg};
use parser::{parse_line, Incoming, SampleJson};
use serial::SerialFramer;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum OutFmt {
    Ndjson,
    Csv,
}

#[derive(Copy, Clone, Debug, ValueEnum, Eq, PartialEq)]
enum Mode {
    Ecg,
    Ppg,
    Resp,
    Emg,
    None,
}

impl Mode {
    fn command(self) -> Option<&'static [u8]> {
        match self {
            Mode::Ecg => Some(b"eE\r\n"),
            Mode::Ppg => Some(b"pP\r\n"),
            Mode::Resp => Some(b"rR\r\n"),
            Mode::Emg => Some(b"mM\r\n"),
            Mode::None => None,
        }
    }

    fn from_runtime_token(token: &str) -> Option<Self> {
        let lower = token.trim().to_ascii_lowercase();
        match lower.as_str() {
            "e" | "ecg" => Some(Mode::Ecg),
            "p" | "ppg" => Some(Mode::Ppg),
            "r" | "resp" => Some(Mode::Resp),
            "m" | "emg" => Some(Mode::Emg),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Mode::Ecg => "ECG",
            Mode::Ppg => "PPG",
            Mode::Resp => "RESP",
            Mode::Emg => "EMG",
            Mode::None => "NONE",
        }
    }
}

const DEFAULT_BAUD: u32 = 115_200;

#[derive(Parser, Debug)]
#[command(version, about = "ECG middleware for Arduino AD8232")]
struct Opts {
    #[arg(short = 'p', long = "port")]
    port: String,

    #[arg(long, default_value_t = DEFAULT_BAUD)]
    baud: u32,

    #[arg(long = "fs", default_value_t = 100.0)]
    fs_hz: f32,

    #[arg(long = "vref", default_value_t = 5.0)]
    vref: f32,

    #[arg(long = "hp", action = ArgAction::SetTrue)]
    high_pass: bool,

    #[arg(long = "ma", default_value_t = 0usize)]
    moving_avg_len: usize,

    #[arg(long = "format", value_enum, default_value_t = OutFmt::Ndjson)]
    format: OutFmt,

    #[arg(long = "csv")]
    csv_path: Option<String>,

    #[arg(long = "mode", value_enum, default_value_t = Mode::Ecg)]
    mode: Mode,

    #[arg(long = "echo-mode-lines", action = ArgAction::SetTrue)]
    echo_mode_lines: bool,

    #[arg(long = "max-line", default_value_t = 2048usize)]
    max_line: usize,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env().add_directive("ecg_bridge=info".parse().unwrap()),
        )
        .init();

    let opts = Opts::parse();
    debug!(?opts, "CLI options parsed");
    info!(
        "Opening port {} @ {} baud… (default {DEFAULT_BAUD}; override with --baud)",
        opts.port, opts.baud
    );
    info!(mode = ?opts.mode, "Selected Arduino mode auto-switch");

    if let Ok(ports) = serialport::available_ports() {
        if ports.is_empty() {
            warn!("serialport: портов не найдено. Убедись, что Arduino подключён.");
        } else {
            let list = ports
                .into_iter()
                .map(|p| p.port_name)
                .collect::<Vec<_>>()
                .join(", ");
            info!("Доступные порты (подсказка): {list}");
        }
    }

    let builder = tokio_serial::new(&opts.port, opts.baud);
    #[cfg(not(windows))]
    let builder = builder.timeout(Duration::from_millis(2000));
    let mut port = builder
        .open_native_async()
        .with_context(|| format!("Не удалось открыть порт {}", opts.port))?;

    let _ = port.write_data_terminal_ready(true);
    let _ = port.write_request_to_send(true);
    let _ = port.clear(ClearBuffer::All);

    tokio::time::sleep(Duration::from_millis(800)).await;

    if let Some(cmd) = opts.mode.command() {
        info!(mode = ?opts.mode, "Sending mode select command to Arduino");
        if let Err(e) = AsyncWriteExt::write_all(&mut port, cmd).await {
            error!("Не удалось отправить команду переключения при старте: {e}");
        } else if let Err(e) = AsyncWriteExt::flush(&mut port).await {
            error!("Не удалось сбросить буфер порта после команды старта: {e}");
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    let (mut reader, writer) = tokio::io::split(port);

    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<Mode>();
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let writer_task = tokio::spawn(async move {
        let mut writer = writer;
        while let Some(mode) = cmd_rx.recv().await {
            match mode.command() {
                Some(cmd) => {
                    info!(
                        mode = mode.label(),
                        "Switching Arduino mode via stdin command"
                    );
                    if let Err(e) = writer.write_all(cmd).await {
                        error!("Не удалось отправить команду переключения: {e}");
                        break;
                    }
                    if let Err(e) = writer.flush().await {
                        error!("Не удалось сбросить буфер порта после команды: {e}");
                        break;
                    }
                }
                None => {
                    warn!(
                        mode = mode.label(),
                        "Команда режима без управляющей последовательности — пропускаем"
                    );
                }
            }
        }
        debug!("stdin command channel closed; writer task exiting");
    });

    let stdin_task = {
        let cmd_tx = cmd_tx.clone();
        let mut shutdown_rx = shutdown_rx.clone();
        tokio::spawn(async move {
            let stdin = io::stdin();
            let mut lines = io::BufReader::new(stdin).lines();
            loop {
                tokio::select! {
                    changed = shutdown_rx.changed() => {
                        match changed {
                            Ok(_) => {
                                if *shutdown_rx.borrow() {
                                    debug!("stdin reader received shutdown signal");
                                    break;
                                }
                            }
                            Err(_) => {
                                debug!("stdin reader shutdown channel closed");
                                break;
                            }
                        }
                    }
                    line = lines.next_line() => {
                        match line {
                            Ok(Some(line)) => {
                                let trimmed = line.trim();
                                if trimmed.is_empty() {
                                    continue;
                                }
                                if let Some(mode) = Mode::from_runtime_token(trimmed) {
                                    info!(mode = mode.label(), "Прочитали команду режима из stdin");
                                    if cmd_tx.send(mode).is_err() {
                                        debug!("Writer task dropped; stopping stdin reader");
                                        break;
                                    }
                                } else {
                                    warn!(command = %trimmed, "Неизвестная команда stdin — игнорируем");
                                }
                            }
                            Ok(None) => {
                                debug!("stdin EOF reached; stopping mode switch task");
                                break;
                            }
                            Err(e) => {
                                error!("Ошибка чтения stdin: {e}");
                                break;
                            }
                        }
                    }
                }
            }
        })
    };

    // Фильтры (если нужны)
    let mut hp = if opts.high_pass {
        debug!(fs = opts.fs_hz, fc = 0.5, "Enabling high-pass filter");
        Some(HighPass1::new(opts.fs_hz, 0.5))
    } else {
        None
    };
    let mut ma = if opts.moving_avg_len > 0 {
        debug!(len = opts.moving_avg_len, "Enabling moving average filter");
        Some(MovingAvg::new(opts.moving_avg_len))
    } else {
        None
    };

    // CSV (если задан путь)
    let mut csv_file: Option<File> = if let Some(path) = &opts.csv_path {
        let mut f =
            File::create(path).with_context(|| format!("Не удалось создать CSV: {}", path))?;
        writeln!(f, "t_ms,ts_unix_ms,lead_off,adc,hp,mv").ok();
        Some(f)
    } else {
        None
    };

    let t0 = Instant::now();
    let mut chunk = vec![0u8; 256];
    let mut framer = SerialFramer::new(opts.max_line);

    loop {
        let n = match reader.read(&mut chunk).await {
            Ok(0) => {
                warn!("EOF от COM-порта (0 байт).");
                break;
            }
            Ok(n) => n,
            Err(e) => {
                if let Some(code) = e.raw_os_error() {
                    if code == 995 {
                        warn!("Получили os error 995 (операция прервана); повторяем чтение…");
                        tokio::time::sleep(Duration::from_millis(150)).await;
                        continue;
                    }
                }
                error!("Ошибка чтения из порта: {e}");
                break;
            }
        };

        trace!(bytes_read = n, "Read chunk from serial port");

        let tokens = framer.push_chunk(&chunk[..n]);

        for token in tokens {
            let s_trim = token.trim();
            trace!(token = %s_trim, "Processing framed token");

            let looks_ok =
                s_trim == "!" || s_trim.starts_with("[MODE]") || looks_like_number_str(s_trim);
            if !looks_ok {
                if opts.echo_mode_lines {
                    eprintln!("[IGNORED] {s_trim}");
                }
                continue;
            }

            let now_unix_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as i128)
                .unwrap_or(0);
            let t_ms = t0.elapsed().as_millis();

            match parse_line(s_trim) {
                Incoming::ModeLine(m) => {
                    if opts.echo_mode_lines {
                        eprintln!("[ARDUINO] {m}");
                    }
                }
                Incoming::LeadOff => {
                    let rec = SampleJson {
                        t_ms,
                        ts_unix_ms: now_unix_ms,
                        adc: None,
                        lead_off: true,
                        hp: None,
                        mv: None,
                    };
                    output(&opts, &mut csv_file, rec)?;
                }
                Incoming::Sample(adc) => {
                    let mut x_adc = (adc as f32) - 511.5;
                    if let Some(hp_) = hp.as_mut() {
                        x_adc = hp_.step(x_adc);
                    }
                    if let Some(ma_) = ma.as_mut() {
                        x_adc = ma_.step(x_adc);
                    }
                    let mv = x_adc * (opts.vref * 1000.0 / 1023.0);
                    let rec = SampleJson {
                        t_ms,
                        ts_unix_ms: now_unix_ms,
                        adc: Some(adc),
                        lead_off: false,
                        hp: Some(x_adc),
                        mv: Some(mv),
                    };
                    output(&opts, &mut csv_file, rec)?;
                }
                Incoming::Unknown(_) => { /* ignore */ }
            }
        }
    }

    drop(cmd_tx);
    let _ = shutdown_tx.send(true);

    match stdin_task.await {
        Ok(()) => {}
        Err(e) if e.is_cancelled() => debug!("stdin task cancelled"),
        Err(e) => error!("stdin task join error: {e}"),
    }

    match writer_task.await {
        Ok(()) => {}
        Err(e) if e.is_cancelled() => debug!("writer task cancelled"),
        Err(e) => error!("writer task join error: {e}"),
    }

    Ok(())
}

fn looks_like_number_str(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    let bytes = s.as_bytes();
    if matches!(bytes[0], b'+' | b'-') {
        return bytes.len() > 1 && bytes[1..].iter().all(u8::is_ascii_digit);
    }
    bytes.iter().all(u8::is_ascii_digit)
}

#[cfg(test)]
mod tests {
    use super::{looks_like_number_str, output, Mode, Opts, OutFmt, DEFAULT_BAUD};
    use crate::parser::SampleJson;
    use clap::Parser;
    use std::fs::{remove_file, File};
    use std::time::{SystemTime, UNIX_EPOCH};

    macro_rules! mode_letter_tests {
        ($($name:ident: $input:expr => $expected:expr),+ $(,)?) => {
            $(
                #[test]
                fn $name() {
                    assert_eq!(Mode::from_runtime_token($input), Some($expected));
                }
            )+
        };
    }

    mode_letter_tests! {
        mode_from_lower_e: "e" => Mode::Ecg,
        mode_from_upper_e: "E" => Mode::Ecg,
        mode_from_lower_p: "p" => Mode::Ppg,
        mode_from_upper_p: "P" => Mode::Ppg,
        mode_from_lower_r: "r" => Mode::Resp,
        mode_from_upper_r: "R" => Mode::Resp,
        mode_from_lower_m: "m" => Mode::Emg,
        mode_from_upper_m: "M" => Mode::Emg,
        mode_from_word_ecg: "ecg" => Mode::Ecg,
        mode_from_word_ppg: "ppg" => Mode::Ppg,
        mode_from_word_resp: "resp" => Mode::Resp,
        mode_from_word_emg: "emg" => Mode::Emg,
        mode_from_mixed_case_ecg: "EcG" => Mode::Ecg,
        mode_from_mixed_case_ppg: "pPg" => Mode::Ppg,
        mode_from_mixed_case_resp: "ReSp" => Mode::Resp,
        mode_from_mixed_case_emg: "eMg" => Mode::Emg,
        mode_from_trimmed_letter_e: "   e   " => Mode::Ecg,
        mode_from_trimmed_letter_p: "\tp\t" => Mode::Ppg,
        mode_from_trimmed_letter_r: "  r" => Mode::Resp,
        mode_from_trimmed_letter_m: "m  " => Mode::Emg,
        mode_from_trimmed_word_ecg: "  ecg  " => Mode::Ecg,
        mode_from_trimmed_word_ppg: "\tppg " => Mode::Ppg,
        mode_from_trimmed_word_resp: " resp" => Mode::Resp,
        mode_from_trimmed_word_emg: " emg " => Mode::Emg,
    }

    #[test]
    fn numeric_detector_accepts_signed() {
        assert!(looks_like_number_str("+512"));
        assert!(looks_like_number_str("-12"));
        assert!(!looks_like_number_str(""));
        assert!(!looks_like_number_str("[MODE]"));
    }

    #[test]
    fn numeric_detector_accepts_zero_variants() {
        assert!(looks_like_number_str("0"));
        assert!(looks_like_number_str("0000"));
        assert!(looks_like_number_str("+0"));
        assert!(looks_like_number_str("-0"));
    }

    #[test]
    fn numeric_detector_rejects_sign_only() {
        assert!(!looks_like_number_str("+"));
        assert!(!looks_like_number_str("-"));
    }

    #[test]
    fn numeric_detector_handles_whitespace() {
        assert!(looks_like_number_str("  123  "));
        assert!(looks_like_number_str("\t-42\n"));
        assert!(!looks_like_number_str("   "));
    }

    #[test]
    fn numeric_detector_rejects_alpha_suffix() {
        assert!(!looks_like_number_str("123a"));
        assert!(!looks_like_number_str("1 2 3"));
        assert!(!looks_like_number_str("--1"));
    }

    #[test]
    fn mode_from_runtime_token_rejects_unknowns() {
        for token in ["", "q", " ec", " eeg", "mode"] {
            assert!(
                Mode::from_runtime_token(token).is_none(),
                "token {token} was not rejected"
            );
        }
    }

    #[test]
    fn mode_command_bytes_match_expected() {
        assert_eq!(Mode::Ecg.command(), Some(b"eE\r\n".as_slice()));
        assert_eq!(Mode::Ppg.command(), Some(b"pP\r\n".as_slice()));
        assert_eq!(Mode::Resp.command(), Some(b"rR\r\n".as_slice()));
        assert_eq!(Mode::Emg.command(), Some(b"mM\r\n".as_slice()));
        assert!(Mode::None.command().is_none());
    }

    #[test]
    fn mode_labels_are_human_readable() {
        assert_eq!(Mode::Ecg.label(), "ECG");
        assert_eq!(Mode::Ppg.label(), "PPG");
        assert_eq!(Mode::Resp.label(), "RESP");
        assert_eq!(Mode::Emg.label(), "EMG");
        assert_eq!(Mode::None.label(), "NONE");
    }

    #[test]
    fn mode_runtime_token_prefers_single_letter() {
        assert_eq!(Mode::from_runtime_token("E"), Some(Mode::Ecg));
        assert_eq!(Mode::from_runtime_token("P"), Some(Mode::Ppg));
        assert_eq!(Mode::from_runtime_token("R"), Some(Mode::Resp));
        assert_eq!(Mode::from_runtime_token("M"), Some(Mode::Emg));
    }

    #[test]
    fn runtime_token_is_case_insensitive() {
        let tokens = [
            ("EcG", Mode::Ecg),
            ("PPG", Mode::Ppg),
            ("Resp", Mode::Resp),
            ("EMG", Mode::Emg),
        ];
        for (token, mode) in tokens {
            assert_eq!(Mode::from_runtime_token(token), Some(mode));
        }
    }

    #[test]
    fn runtime_token_trims_leading_and_trailing_whitespace() {
        let tokens = [
            ("  e\n", Mode::Ecg),
            ("\tP ", Mode::Ppg),
            ("\rresp\r", Mode::Resp),
            ("  EMG  ", Mode::Emg),
        ];
        for (token, mode) in tokens {
            assert_eq!(Mode::from_runtime_token(token), Some(mode));
        }
    }

    #[test]
    fn runtime_token_rejects_partial_words() {
        for token in ["ec", "pp", "re", "em"] {
            assert!(Mode::from_runtime_token(token).is_none());
        }
    }

    #[test]
    fn runtime_token_rejects_numeric_input() {
        for token in ["1", "42", "-7"] {
            assert!(Mode::from_runtime_token(token).is_none());
        }
    }

    #[test]
    fn default_baud_matches_arduino_sketch() {
        let opts = Opts::parse_from(["ecg_bridge", "--port", "COM4"]);
        assert_eq!(opts.baud, DEFAULT_BAUD);
    }

    #[test]
    fn runtime_token_accepts_words() {
        assert_eq!(Mode::from_runtime_token("PPG"), Some(Mode::Ppg));
        assert_eq!(Mode::from_runtime_token("resp"), Some(Mode::Resp));
        assert!(Mode::from_runtime_token("foobar").is_none());
    }

    #[test]
    fn opts_parse_high_pass_flag() {
        let opts = Opts::parse_from(["ecg_bridge", "--port", "COM4", "--hp"]);
        assert!(opts.high_pass);
    }

    #[test]
    fn opts_parse_echo_mode_lines_flag() {
        let opts = Opts::parse_from(["ecg_bridge", "--port", "COM4", "--echo-mode-lines"]);
        assert!(opts.echo_mode_lines);
    }

    #[test]
    fn opts_parse_custom_frequency_and_vref() {
        let opts = Opts::parse_from([
            "ecg_bridge",
            "--port",
            "COM4",
            "--fs",
            "250",
            "--vref",
            "3.3",
        ]);
        assert_eq!(opts.fs_hz, 250.0);
        assert_eq!(opts.vref, 3.3);
    }

    #[test]
    fn opts_parse_moving_average_length() {
        let opts = Opts::parse_from(["ecg_bridge", "--port", "COM4", "--ma", "8"]);
        assert_eq!(opts.moving_avg_len, 8);
    }

    #[test]
    fn opts_parse_format_csv() {
        let opts = Opts::parse_from(["ecg_bridge", "--port", "COM4", "--format", "csv"]);
        assert!(matches!(opts.format, OutFmt::Csv));
    }

    #[test]
    fn opts_parse_mode_none() {
        let opts = Opts::parse_from(["ecg_bridge", "--port", "COM4", "--mode", "none"]);
        assert_eq!(opts.mode, Mode::None);
    }

    #[test]
    fn opts_parse_mode_variants_by_letter() {
        let opts_ecg = Opts::parse_from(["ecg_bridge", "--port", "COM4", "--mode", "ecg"]);
        assert_eq!(opts_ecg.mode, Mode::Ecg);
        let opts_ppg = Opts::parse_from(["ecg_bridge", "--port", "COM4", "--mode", "ppg"]);
        assert_eq!(opts_ppg.mode, Mode::Ppg);
        let opts_resp = Opts::parse_from(["ecg_bridge", "--port", "COM4", "--mode", "resp"]);
        assert_eq!(opts_resp.mode, Mode::Resp);
        let opts_emg = Opts::parse_from(["ecg_bridge", "--port", "COM4", "--mode", "emg"]);
        assert_eq!(opts_emg.mode, Mode::Emg);
    }

    #[test]
    fn opts_parse_custom_csv_path() {
        let opts = Opts::parse_from(["ecg_bridge", "--port", "COM4", "--csv", "output.csv"]);
        assert_eq!(opts.csv_path.as_deref(), Some("output.csv"));
    }

    #[test]
    fn opts_parse_custom_max_line() {
        let opts = Opts::parse_from(["ecg_bridge", "--port", "COM4", "--max-line", "1024"]);
        assert_eq!(opts.max_line, 1024);
    }

    #[test]
    fn opts_parse_custom_baud() {
        let opts = Opts::parse_from(["ecg_bridge", "--port", "COM4", "--baud", "57600"]);
        assert_eq!(opts.baud, 57_600);
    }

    #[test]
    fn opts_parse_combined_flags() {
        let opts = Opts::parse_from([
            "ecg_bridge",
            "--port",
            "COM4",
            "--hp",
            "--echo-mode-lines",
            "--mode",
            "ppg",
        ]);
        assert!(opts.high_pass);
        assert!(opts.echo_mode_lines);
        assert_eq!(opts.mode, Mode::Ppg);
    }

    #[test]
    fn output_writes_json_samples() {
        let mut opts = Opts::parse_from(["ecg_bridge", "--port", "COM4"]);
        opts.format = OutFmt::Ndjson;
        let mut csv_file = None;
        let rec = SampleJson {
            t_ms: 1,
            ts_unix_ms: 2,
            adc: Some(512),
            lead_off: false,
            hp: Some(1.23),
            mv: Some(3.21),
        };
        output(&opts, &mut csv_file, rec).expect("output should succeed");
    }

    #[test]
    fn output_writes_csv_with_file() {
        let csv_path = {
            let unique = format!(
                "ecg_bridge_test_{}_{}.csv",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            );
            std::env::temp_dir().join(unique)
        };
        let mut opts = Opts::parse_from(["ecg_bridge", "--port", "COM4"]);
        opts.format = OutFmt::Csv;
        opts.csv_path = Some(csv_path.to_string_lossy().into_owned());
        let mut csv_file = Some(File::create(csv_path).unwrap());
        let rec = SampleJson {
            t_ms: 10,
            ts_unix_ms: 20,
            adc: Some(256),
            lead_off: false,
            hp: Some(0.5),
            mv: Some(0.25),
        };
        output(&opts, &mut csv_file, rec).expect("output should succeed");
        if let Some(path) = opts.csv_path.as_ref() {
            let _ = remove_file(path);
        }
    }

    #[test]
    fn output_writes_lead_off_rows() {
        let opts = Opts::parse_from(["ecg_bridge", "--port", "COM4", "--format", "csv"]);
        let mut csv_file = None;
        let rec = SampleJson {
            t_ms: 5,
            ts_unix_ms: 6,
            adc: None,
            lead_off: true,
            hp: None,
            mv: None,
        };
        output(&opts, &mut csv_file, rec).expect("output should succeed");
    }

    #[test]
    fn mode_command_sequences_are_unique() {
        let mut commands = std::collections::HashSet::new();
        for mode in [Mode::Ecg, Mode::Ppg, Mode::Resp, Mode::Emg] {
            commands.insert(mode.command().unwrap().to_vec());
        }
        assert_eq!(commands.len(), 4);
    }

    #[test]
    fn runtime_token_handles_mixed_whitespace() {
        assert_eq!(Mode::from_runtime_token("\t e\r"), Some(Mode::Ecg));
        assert_eq!(Mode::from_runtime_token("\np\n"), Some(Mode::Ppg));
        assert_eq!(Mode::from_runtime_token("\rr\t"), Some(Mode::Resp));
        assert_eq!(Mode::from_runtime_token(" m \n"), Some(Mode::Emg));
    }

    #[test]
    fn mode_none_has_no_command_and_label_none() {
        assert!(Mode::None.command().is_none());
        assert_eq!(Mode::None.label(), "NONE");
    }
}

fn output(opts: &Opts, csv_file: &mut Option<File>, rec: SampleJson) -> Result<()> {
    match opts.format {
        OutFmt::Ndjson => {
            let line = serde_json::to_string(&rec)?;
            println!("{line}");
        }
        OutFmt::Csv => {
            if let Some(adc) = rec.adc {
                println!(
                    "{},{},{},{},{},{}",
                    rec.t_ms,
                    rec.ts_unix_ms,
                    rec.lead_off as u8,
                    adc,
                    rec.hp.map(|v| v.to_string()).unwrap_or_default(),
                    rec.mv.map(|v| v.to_string()).unwrap_or_default()
                );
            } else {
                println!("{},{},{},,,", rec.t_ms, rec.ts_unix_ms, rec.lead_off as u8);
            }
        }
    }

    if let Some(f) = csv_file.as_mut() {
        if let Some(adc) = rec.adc {
            writeln!(
                f,
                "{},{},{},{},{},{}",
                rec.t_ms,
                rec.ts_unix_ms,
                rec.lead_off as u8,
                adc,
                rec.hp.map(|v| v.to_string()).unwrap_or_default(),
                rec.mv.map(|v| v.to_string()).unwrap_or_default()
            )
            .ok();
        } else {
            writeln!(
                f,
                "{},{},{},,,",
                rec.t_ms, rec.ts_unix_ms, rec.lead_off as u8
            )
            .ok();
        }
    }

    Ok(())
}
