use serde::Serialize;
use tracing::{debug, trace};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Incoming {
    ModeLine(String), 
    LeadOff,          
    Sample(u16),      
    Unknown(String),  
}

pub fn parse_line(line: &str) -> Incoming {
    let s = line.trim();
    trace!(raw = line, trimmed = s, "Parsing line");
    let incoming = if s.is_empty() {
        Incoming::Unknown(s.to_string())
    } else if s == "!" {
        Incoming::LeadOff
    } else if s.starts_with("[MODE]") {
        Incoming::ModeLine(s.to_string())
    } else {
        match s.parse::<u16>() {
            Ok(v) => Incoming::Sample(v),
            Err(_) => Incoming::Unknown(s.to_string()),
        }
    };
    debug!(?incoming, "Classified incoming line");
    incoming
}

#[derive(Debug, Serialize, Clone)]
pub struct SampleJson {
    /// Монотонное время, мс с запуска процесса
    pub t_ms: u128,
    /// Unix-время, мс (прибл. момент приёма строки)
    pub ts_unix_ms: i128,
    /// Сырой АЦП 0..1023 (если lead_off=false)
    pub adc: Option<u16>,
    /// Признак отрыва электродов (при "!")
    pub lead_off: bool,
    /// Значение после фильтра ВЧ (если включён). Ед. — "ADC отсчёты"
    pub hp: Option<f32>,
    /// Тот же отсчёт в милливольтах (относительно Vref/2)
    pub mv: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mode_line() {
        assert_eq!(
            parse_line("[MODE] ECG (AD8232)"),
            Incoming::ModeLine("[MODE] ECG (AD8232)".to_string())
        );
    }

    #[test]
    fn parses_lead_off() {
        assert_eq!(parse_line("!"), Incoming::LeadOff);
    }

    #[test]
    fn parses_numeric_sample() {
        assert_eq!(parse_line("512"), Incoming::Sample(512));
    }

    #[test]
    fn rejects_garbage() {
        assert_eq!(parse_line("foo"), Incoming::Unknown("foo".to_string()));
    }
}
