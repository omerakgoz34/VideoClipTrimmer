/// `90` → `01:30.000`. Hours only when the clip is at least an hour.
pub fn format_time(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "00:00.000".to_string();
    }
    let total_ms = (seconds * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    if h > 0 {
        format!("{h}:{m:02}:{s:02}.{ms:03}")
    } else {
        format!("{m:02}:{s:02}.{ms:03}")
    }
}

/// Accept `90`, `90.5`, `1:30`, `1:30.25`, or `1:02:03.5`.
pub fn parse_time(text: &str) -> Option<f64> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    if !text.contains(':') {
        return text
            .parse::<f64>()
            .ok()
            .filter(|n| n.is_finite() && *n >= 0.0);
    }
    let mut parts: Vec<&str> = text.split(':').collect();
    if parts.len() < 2 || parts.len() > 3 {
        return None;
    }
    let seconds: f64 = parts.pop()?.parse().ok()?;
    let minutes: f64 = parts.pop()?.parse().ok()?;
    let hours: f64 = match parts.pop() {
        Some(h) => h.parse().ok()?,
        None => 0.0,
    };
    if hours < 0.0 || minutes < 0.0 || seconds < 0.0 {
        return None;
    }
    Some(hours * 3600.0 + minutes * 60.0 + seconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_under_an_hour() {
        let text = format_time(90.5);
        assert_eq!(text, "01:30.500");
        assert!((parse_time(&text).unwrap() - 90.5).abs() < 0.001);
    }

    #[test]
    fn parses_loose_input() {
        assert!((parse_time("90").unwrap() - 90.0).abs() < 0.001);
        assert!((parse_time("1:02:03").unwrap() - 3723.0).abs() < 0.001);
        assert!((parse_time("00:00:02.00").unwrap() - 2.0).abs() < 0.001);
        assert!(parse_time("nope").is_none());
        assert!(parse_time("").is_none());
    }

    #[test]
    fn hours_get_their_own_field() {
        assert_eq!(format_time(3723.25), "1:02:03.250");
    }
}
