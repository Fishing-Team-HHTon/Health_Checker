use clap::ValueEnum;

#[derive(Copy, Clone, Debug, ValueEnum, Eq, PartialEq)]
pub enum Mode {
    Ecg,
    Ppg,
    Resp,
    Emg,
    None,
}

impl Mode {
    pub fn command(self) -> Option<&'static [u8]> {
        match self {
            Mode::Ecg => Some(b"eE\r\n"),
            Mode::Ppg => Some(b"pP\r\n"),
            Mode::Resp => Some(b"rR\r\n"),
            Mode::Emg => Some(b"mM\r\n"),
            Mode::None => None,
        }
    }

    pub fn api_token(self) -> Option<&'static str> {
        match self {
            Mode::Ecg => Some("ecg"),
            Mode::Ppg => Some("ppg"),
            Mode::Resp => Some("resp"),
            Mode::Emg => Some("emg"),
            Mode::None => None,
        }
    }

    pub fn from_runtime_token(token: &str) -> Option<Self> {
        let lower = token.trim().to_ascii_lowercase();
        match lower.as_str() {
            "e" | "ecg" => Some(Mode::Ecg),
            "p" | "ppg" => Some(Mode::Ppg),
            "r" | "resp" => Some(Mode::Resp),
            "m" | "emg" => Some(Mode::Emg),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Mode::Ecg => "ECG",
            Mode::Ppg => "PPG",
            Mode::Resp => "RESP",
            Mode::Emg => "EMG",
            Mode::None => "NONE",
        }
    }
}
