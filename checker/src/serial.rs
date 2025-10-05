use tracing::{debug, trace, warn};

pub struct SerialFramer {
    pending: Vec<u8>,
    max_line: usize,
    backlog_limit: usize,
}

impl SerialFramer {
    pub fn new(max_line: usize) -> Self {
        let backlog_limit = max_line.saturating_mul(8).max(max_line);
        debug!(max_line, backlog_limit, "Initialized SerialFramer");
        Self {
            pending: Vec::with_capacity(max_line.min(256)),
            max_line,
            backlog_limit,
        }
    }

    #[allow(dead_code)]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn push_chunk(&mut self, chunk: &[u8]) -> Vec<String> {
        trace!(chunk_len = chunk.len(), "Pushing chunk into SerialFramer");
        self.pending.extend_from_slice(chunk);
        let mut tokens = Vec::new();

        if self.pending.len() > self.max_line && self.find_delimiter().is_none() {
            warn!(
                pending_len = self.pending.len(),
                max_line = self.max_line,
                "Dropping oversized buffer without delimiters"
            );
            self.pending.clear();
            return tokens;
        }

        while let Some(token) = self.drain_next_token() {
            if token.len() > self.max_line {
                warn!(
                    token_len = token.len(),
                    max_line = self.max_line,
                    "Dropping oversized token"
                );
                continue;
            }

            match String::from_utf8(token) {
                Ok(s) => {
                    if !s.trim().is_empty() {
                        trace!(token = %s, "Framed token");
                        tokens.push(s);
                    }
                }
                Err(err) => {
                    warn!(error = %err, "Dropping non-UTF8 token");
                }
            }
        }

        if self.pending.len() > self.backlog_limit {
            warn!(
                pending_len = self.pending.len(),
                backlog_limit = self.backlog_limit,
                "Clearing backlog without delimiters"
            );
            self.pending.clear();
        }

        tokens
    }

    fn find_delimiter(&self) -> Option<usize> {
        for (idx, &b) in self.pending.iter().enumerate() {
            if matches!(b, b'\n' | b'\r' | b',' | b';') {
                return Some(idx);
            }
            if matches!(b, b' ' | b'\t') {
                if idx == 0 || looks_like_number(&self.pending[..idx]) {
                    return Some(idx);
                }
            }
        }
        None
    }

    fn drain_next_token(&mut self) -> Option<Vec<u8>> {
        let pos = self.find_delimiter()?;
        let mut token = self.pending.drain(..=pos).collect::<Vec<_>>();
        trim_token(&mut token);
        if token.is_empty() {
            if self.pending.is_empty() {
                trace!("Only delimiters drained, buffer empty");
                return None;
            }
            trace!("Skipping empty token after delimiter");
            return self.drain_next_token();
        }
        Some(token)
    }
}

fn trim_token(token: &mut Vec<u8>) {
    let start = token
        .iter()
        .position(|b| !is_delimiter_byte(*b))
        .unwrap_or(token.len());
    token.drain(..start);
    if let Some(end) = token.iter().rposition(|b| !is_delimiter_byte(*b)) {
        token.truncate(end + 1);
    } else {
        token.clear();
    }
}

fn is_delimiter_byte(b: u8) -> bool {
    matches!(b, b'\n' | b'\r' | b',' | b';' | b' ' | b'\t')
}

fn looks_like_number(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    let (first, rest) = bytes.split_first().unwrap();
    if matches!(first, b'+' | b'-') {
        return !rest.is_empty() && rest.iter().all(u8::is_ascii_digit);
    }
    bytes.iter().all(u8::is_ascii_digit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_crlf_lines() {
        let mut framer = SerialFramer::new(64);
        let tokens = framer.push_chunk(b"512\r\n513\r\n");
        assert_eq!(tokens, vec!["512".to_string(), "513".to_string()]);
    }

    #[test]
    fn splits_space_separated_tokens() {
        let mut framer = SerialFramer::new(64);
        let mut tokens = framer.push_chunk(b"512 513");
        assert_eq!(tokens, vec!["512".to_string()]);
        tokens.extend(framer.push_chunk(b" 514\n"));
        assert_eq!(
            tokens,
            vec!["512".to_string(), "513".to_string(), "514".to_string()]
        );
    }

    #[test]
    fn drops_oversized_without_delimiters() {
        let mut framer = SerialFramer::new(8);
        let tokens = framer.push_chunk(&vec![b'1'; 9]);
        assert!(tokens.is_empty());
        assert_eq!(framer.pending_len(), 0);
    }

    #[test]
    fn drops_non_utf8_token() {
        let mut framer = SerialFramer::new(8);
        let data = vec![0xFF, b'\n'];
        let tokens = framer.push_chunk(&data);
        assert!(tokens.is_empty());
        assert_eq!(framer.pending_len(), 0);
    }

    #[test]
    fn enforces_token_limit() {
        let mut framer = SerialFramer::new(4);
        let tokens = framer.push_chunk(b"12345\n123\n");
        assert_eq!(tokens, vec!["123".to_string()]);
        assert_eq!(framer.pending_len(), 0);
    }

    #[test]
    fn handles_tokens_split_across_chunks() {
        let mut framer = SerialFramer::new(64);
        let mut tokens = framer.push_chunk(b"1");
        assert!(tokens.is_empty());
        tokens.extend(framer.push_chunk(b"23\n45"));
        assert_eq!(tokens, vec!["123".to_string()]);
        tokens.extend(framer.push_chunk(b"6\n"));
        assert_eq!(tokens, vec!["123".to_string(), "456".to_string()]);
    }

    #[test]
    fn handles_semicolon_delimiter() {
        let mut framer = SerialFramer::new(32);
        let tokens = framer.push_chunk(b"512;513;514\n");
        assert_eq!(
            tokens,
            vec!["512".to_string(), "513".to_string(), "514".to_string()]
        );
    }

    #[test]
    fn ignores_leading_delimiters() {
        let mut framer = SerialFramer::new(32);
        let tokens = framer.push_chunk(b"\n\r512\n");
        assert_eq!(tokens, vec!["512".to_string()]);
    }

    #[test]
    fn handles_negative_numbers() {
        let mut framer = SerialFramer::new(32);
        let tokens = framer.push_chunk(b"-1\n-2\n");
        assert_eq!(tokens, vec!["-1".to_string(), "-2".to_string()]);
    }

    #[test]
    fn handles_positive_numbers_with_plus_sign() {
        let mut framer = SerialFramer::new(32);
        let tokens = framer.push_chunk(b"+1 +2 +3\n");
        assert_eq!(
            tokens,
            vec!["+1".to_string(), "+2".to_string(), "+3".to_string()]
        );
    }

    #[test]
    fn clears_backlog_when_limit_exceeded() {
        let mut framer = SerialFramer::new(4);
        let big = vec![b'1'; 40];
        let tokens = framer.push_chunk(&big);
        assert!(tokens.is_empty());
        assert_eq!(framer.pending_len(), 0);
    }

    #[test]
    fn whitespace_delimiters_with_numbers_are_detected() {
        let mut framer = SerialFramer::new(16);
        let tokens = framer.push_chunk(b"123 456\t789\n");
        assert_eq!(
            tokens,
            vec!["123".to_string(), "456".to_string(), "789".to_string()]
        );
    }

    #[test]
    fn trailing_space_is_trimmed() {
        let mut framer = SerialFramer::new(16);
        let tokens = framer.push_chunk(b"123 \n");
        assert_eq!(tokens, vec!["123".to_string()]);
    }
}
