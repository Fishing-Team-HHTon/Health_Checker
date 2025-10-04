use tracing::{debug, trace};

/// Простой 1-го порядка высокочастотный фильтр (удаление DC/дрейфа).
/// Формула: y[n] = a*(y[n-1] + x[n] - x[n-1]),
/// где a = RC/(RC + dt). Для fc≈0.5 Гц при fs=100 Гц получаем a≈0.969.
pub struct HighPass1 {
    a: f32,
    prev_x: f32,
    prev_y: f32,
    inited: bool,
}

impl HighPass1 {
    pub fn new(fs_hz: f32, fc_hz: f32) -> Self {
        let rc = 1.0_f32 / (2.0 * std::f32::consts::PI * fc_hz);
        let dt = 1.0 / fs_hz;
        let a = rc / (rc + dt);
        let instance = Self {
            a,
            prev_x: 0.0,
            prev_y: 0.0,
            inited: false,
        };
        debug!(fs_hz, fc_hz, a, "Created HighPass1 filter");
        instance
    }

    pub fn step(&mut self, x: f32) -> f32 {
        if !self.inited {
            self.prev_x = x;
            self.prev_y = 0.0;
            self.inited = true;
        }
        let y = self.a * (self.prev_y + x - self.prev_x);
        self.prev_x = x;
        self.prev_y = y;
        trace!(input = x, output = y, a = self.a, "HighPass1 step");
        y
    }
}

pub struct MovingAvg {
    buf: Vec<f32>,
    sum: f32,
    idx: usize,
    filled: bool,
}

impl MovingAvg {
    pub fn new(len: usize) -> Self {
        assert!(len > 0);
        let instance = Self {
            buf: vec![0.0; len],
            sum: 0.0,
            idx: 0,
            filled: false,
        };
        debug!(len, "Created MovingAvg filter");
        instance
    }

    pub fn step(&mut self, x: f32) -> f32 {
        let old = self.buf[self.idx];
        self.sum -= old;
        self.buf[self.idx] = x;
        self.sum += x;
        self.idx = (self.idx + 1) % self.buf.len();
        if self.idx == 0 {
            self.filled = true;
        }
        let n = if self.filled {
            self.buf.len()
        } else {
            self.idx
        };
        if n == 0 {
            trace!(
                input = x,
                output = x,
                window = n,
                "MovingAvg step (not filled)"
            );
            x
        } else {
            let result = self.sum / (n as f32);
            trace!(input = x, output = result, window = n, "MovingAvg step");
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{HighPass1, MovingAvg};

    #[test]
    fn high_pass_attacks_dc_offset() {
        let mut hp = HighPass1::new(100.0, 0.5);
        let mut last = 0.0;
        for _ in 0..10 {
            last = hp.step(1.0);
        }
        assert!(last.abs() < 1e-2);
    }

    #[test]
    fn moving_average_smooths_noise() {
        let mut ma = MovingAvg::new(4);
        let inputs = [1.0, 3.0, 5.0, 7.0];
        let mut out = 0.0;
        for &x in &inputs {
            out = ma.step(x);
        }
        assert!((out - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn high_pass_initializes_on_first_step() {
        let mut hp = HighPass1::new(200.0, 0.5);
        let first = hp.step(2.0);
        let second = hp.step(2.0);
        assert_eq!(first, 0.0);
        assert!(second.abs() < 1e-6);
    }

    #[test]
    fn high_pass_tracks_changing_signal() {
        let mut hp = HighPass1::new(100.0, 0.5);
        let mut last = 0.0;
        for x in [0.0, 1.0, 2.0, 3.0, 4.0] {
            last = hp.step(x);
        }
        assert!(last.is_finite());
        assert!(last > 0.0);
        assert!(last < 5.0);
    }

    #[test]
    fn moving_average_handles_single_element_window() {
        let mut ma = MovingAvg::new(1);
        assert!((ma.step(5.0) - 5.0).abs() < f32::EPSILON);
        assert!((ma.step(7.0) - 7.0).abs() < f32::EPSILON);
    }

    #[test]
    #[should_panic]
    fn moving_average_requires_positive_length() {
        let _ = MovingAvg::new(0);
    }

    #[test]
    fn moving_average_wraps_buffer_correctly() {
        let mut ma = MovingAvg::new(3);
        ma.step(1.0);
        ma.step(2.0);
        ma.step(3.0);
        let out = ma.step(4.0);
        assert!((out - (2.0 + 3.0 + 4.0) / 3.0).abs() < 1e-6);
    }

    #[test]
    fn moving_average_reports_partial_window_average() {
        let mut ma = MovingAvg::new(4);
        assert!((ma.step(1.0) - 1.0).abs() < f32::EPSILON);
        assert!((ma.step(2.0) - 1.5).abs() < f32::EPSILON);
    }
}
