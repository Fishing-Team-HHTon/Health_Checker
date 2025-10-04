use anyhow::{Context, Result};
use clap::{ArgAction, Parser, ValueEnum};
use std::{
    fs::File,
    io::Write,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
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

    #[arg(long = "send-e", action = ArgAction::SetTrue, default_value_t = true)]
    send_e: bool,

    #[arg(long = "echo-mode-lines", action = ArgAction::SetTrue)]
    echo_mode_lines: bool,

    #[arg(long = "max-line", default_value_t = 2048usize)]
    max_line: usize,
}

#[tokio::main]
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

    if opts.send_e {
        info!("Sending 'e' to select ECG mode…");
        let _ = AsyncWriteExt::write_all(&mut port, b"e\n").await;
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

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
        let n = match port.read(&mut chunk).await {
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
    use super::{looks_like_number_str, Opts, DEFAULT_BAUD};
    use clap::Parser;

    #[test]
    fn numeric_detector_accepts_signed() {
        assert!(looks_like_number_str("+512"));
        assert!(looks_like_number_str("-12"));
        assert!(!looks_like_number_str(""));
        assert!(!looks_like_number_str("[MODE]"));
    }

    #[test]
    fn default_baud_matches_arduino_sketch() {
        let opts = Opts::parse_from(["ecg_bridge", "--port", "COM4"]);
        assert_eq!(opts.baud, DEFAULT_BAUD);
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
