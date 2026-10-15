//! Every ffmpeg command lives in this file.
//!
//! The app shells out to the `ffmpeg` binary. It does not link libav.
//! Change a flag here and the whole app picks it up.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use crate::timeutil::parse_time;

/// How an export is encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodeMode {
    /// Re-encode so clips with different sizes or codecs can be joined.
    Compatible,
    /// Stream copy. Fast, but a cut starts at the previous keyframe,
    /// and a join only works when the clips already match.
    FastCopy,
}

#[derive(Clone, Debug)]
pub struct MediaInfo {
    pub duration: f64,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub has_audio: bool,
    pub video_codec: String,
    pub audio_codec: String,
}

/// One range of one file, ready to export.
#[derive(Clone, Debug)]
pub struct Segment {
    pub path: PathBuf,
    pub start: f64,
    pub end: f64,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub has_audio: bool,
}

#[derive(Debug)]
pub enum ExportError {
    Cancelled,
    Message(String),
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportError::Cancelled => write!(f, "Cancelled."),
            ExportError::Message(message) => write!(f, "{message}"),
        }
    }
}

/// Progress and log callbacks for a running export. `cancel` is checked between steps.
pub struct Hooks<'a> {
    pub progress: &'a dyn Fn(f32),
    pub log: &'a dyn Fn(String),
    pub cancel: &'a AtomicBool,
}

pub struct RgbaFrame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// `FFMPEG` env var, then a copy sitting next to the executable, then `PATH`.
pub fn ffmpeg_program() -> String {
    if let Ok(path) = std::env::var("FFMPEG") {
        let path = path.trim();
        if !path.is_empty() {
            return path.to_string();
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let name = if cfg!(windows) {
                "ffmpeg.exe"
            } else {
                "ffmpeg"
            };
            let sibling = dir.join(name);
            if sibling.is_file() {
                return sibling.to_string_lossy().into_owned();
            }
        }
    }
    "ffmpeg".to_string()
}

pub fn check_ffmpeg() -> Result<String, String> {
    let program = ffmpeg_program();
    let output = Command::new(&program).arg("-version").output().map_err(|err| {
        format!(
            "ffmpeg was not found ({program}). Install it and make sure it is on PATH, or set the FFMPEG variable.\n{err}"
        )
    })?;
    if !output.status.success() {
        return Err(format!("`{program} -version` failed."));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Ok(text.lines().next().unwrap_or("ffmpeg").trim().to_string())
}

pub fn probe(path: &Path) -> Result<MediaInfo, String> {
    let output = Command::new(ffmpeg_program())
        .args(["-hide_banner", "-i"])
        .arg(path)
        .output()
        .map_err(|err| format!("Could not start ffmpeg: {err}"))?;
    // ffmpeg exits with an error when `-i` is used alone. The info we need is still on stderr.
    let text = String::from_utf8_lossy(&output.stderr);
    parse_probe(&text).map_err(|err| {
        if path.is_file() {
            format!("{err}\n{}", tail(&text, 8))
        } else {
            format!("Could not open {}.", path.display())
        }
    })
}

pub fn grab_frame(path: &Path, time: f64) -> Result<RgbaFrame, String> {
    let args = [
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        &format!("{time:.3}"),
        "-i",
    ];
    let mut command = Command::new(ffmpeg_program());
    command
        .args(args)
        .arg(path)
        .args([
            "-frames:v",
            "1",
            // Fit inside a 960 box. Preview only — the export keeps full resolution.
            "-vf",
            "scale=960:960:force_original_aspect_ratio=decrease",
            "-c:v",
            "mjpeg",
            "-q:v",
            "4",
            "-f",
            "image2pipe",
            "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = run_to_completion(command)?;
    if output.stdout.is_empty() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "ffmpeg did not return a preview frame.\n{}",
            tail(&err, 6)
        ));
    }
    let image = image::load_from_memory(&output.stdout)
        .map_err(|err| format!("Could not decode the preview frame: {err}"))?;
    let rgba = image.to_rgba8();
    Ok(RgbaFrame {
        width: rgba.width(),
        height: rgba.height(),
        pixels: rgba.into_raw(),
    })
}

pub fn export(
    segments: &[Segment],
    dest: &Path,
    mode: EncodeMode,
    hooks: &Hooks<'_>,
) -> Result<(), ExportError> {
    if segments.is_empty() {
        return Err(ExportError::Message(
            "Add at least one clip before exporting.".into(),
        ));
    }
    for segment in segments {
        if segment.end - segment.start < 0.04 {
            return Err(ExportError::Message(
                "A clip is too short. Each range needs a little time in it.".into(),
            ));
        }
    }

    if segments.len() == 1 {
        (hooks.progress)(0.0);
        encode_segment(&segments[0], dest, mode, None, hooks, 0.0, 1.0)?;
        (hooks.progress)(1.0);
        return Ok(());
    }

    let temp = TempDir::new()?;
    let mut parts = Vec::with_capacity(segments.len());
    let encode_share = 0.92;
    for (index, segment) in segments.iter().enumerate() {
        check_cancel(hooks)?;
        let part = temp.0.join(format!("part-{index:03}.mp4"));
        let start = encode_share * (index as f32) / segments.len() as f32;
        let end = encode_share * (index as f32 + 1.0) / segments.len() as f32;
        // Joined clips have to share a size. The first clip decides it.
        let target = Some(output_target(&segments[0]));
        encode_segment(segment, &part, mode, target, hooks, start, end)?;
        parts.push(part);
    }

    check_cancel(hooks)?;
    let list_path = temp.0.join("list.txt");
    let list = parts
        .iter()
        .map(|path| concat_line(path))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&list_path, list + "\n")
        .map_err(|err| ExportError::Message(format!("Could not write the concat list: {err}")))?;

    let total: f64 = segments.iter().map(|s| s.end - s.start).sum();
    let args = vec![
        "-y".into(),
        "-hide_banner".into(),
        "-nostats".into(),
        "-progress".into(),
        temp.0.join("progress.txt").display().to_string(),
        "-loglevel".into(),
        "error".into(),
        "-f".into(),
        "concat".into(),
        "-safe".into(),
        "0".into(),
        "-i".into(),
        list_path.display().to_string(),
        "-c".into(),
        "copy".into(),
        "-movflags".into(),
        "+faststart".into(),
        dest.display().to_string(),
    ];
    run_with_progress(
        &args,
        &temp.0.join("progress.txt"),
        total.max(0.1),
        hooks,
        0.92,
        1.0,
    )?;
    (hooks.progress)(1.0);
    Ok(())
}

fn encode_segment(
    segment: &Segment,
    dest: &Path,
    mode: EncodeMode,
    target: Option<Target>,
    hooks: &Hooks<'_>,
    progress_from: f32,
    progress_to: f32,
) -> Result<(), ExportError> {
    let progress_path = dest.with_extension("progress");
    let args = match mode {
        EncodeMode::FastCopy => fast_copy_args(segment, dest, &progress_path),
        EncodeMode::Compatible => {
            let target = target.unwrap_or_else(|| output_target(segment));
            compatible_args(segment, dest, &target, &progress_path)
        }
    };
    let duration = (segment.end - segment.start).max(0.05);
    run_with_progress(
        &args,
        &progress_path,
        duration,
        hooks,
        progress_from,
        progress_to,
    )?;
    let _ = std::fs::remove_file(&progress_path);
    Ok(())
}

fn fast_copy_args(segment: &Segment, dest: &Path, progress: &Path) -> Vec<String> {
    let duration = segment.end - segment.start;
    vec![
        "-y".into(),
        "-hide_banner".into(),
        "-nostats".into(),
        "-progress".into(),
        progress.display().to_string(),
        "-loglevel".into(),
        "error".into(),
        // `-ss` before `-i` seeks quickly. With `-c copy` the cut snaps to a keyframe.
        "-ss".into(),
        format!("{:.3}", segment.start),
        "-t".into(),
        format!("{:.3}", duration),
        "-i".into(),
        segment.path.display().to_string(),
        "-map".into(),
        "0".into(),
        "-c".into(),
        "copy".into(),
        "-avoid_negative_ts".into(),
        "make_zero".into(),
        "-movflags".into(),
        "+faststart".into(),
        dest.display().to_string(),
    ]
}

fn compatible_args(
    segment: &Segment,
    dest: &Path,
    target: &Target,
    progress: &Path,
) -> Vec<String> {
    let duration = segment.end - segment.start;
    let filter = format!(
        "scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={fps}",
        w = target.width,
        h = target.height,
        fps = format_fps(target.fps),
    );
    let mut args = vec![
        "-y".into(),
        "-hide_banner".into(),
        "-nostats".into(),
        "-progress".into(),
        progress.display().to_string(),
        "-loglevel".into(),
        "error".into(),
        // `-ss` before `-i` is fast. Re-encoding still starts near the requested frame,
        // because ffmpeg decodes back to the previous keyframe.
        "-ss".into(),
        format!("{:.3}", segment.start),
        "-t".into(),
        format!("{:.3}", duration),
        "-i".into(),
        segment.path.display().to_string(),
    ];

    if segment.has_audio {
        args.extend([
            "-map".into(),
            "0:v:0".into(),
            "-map".into(),
            "0:a:0".into(),
            "-vf".into(),
            filter,
            "-c:v".into(),
            "libx264".into(),
            "-preset".into(),
            "veryfast".into(),
            "-crf".into(),
            "20".into(),
            "-pix_fmt".into(),
            "yuv420p".into(),
            "-c:a".into(),
            "aac".into(),
            "-b:a".into(),
            "160k".into(),
            "-ar".into(),
            "48000".into(),
            "-ac".into(),
            "2".into(),
        ]);
    } else {
        // A silent track keeps a join working when some clips have audio and some do not.
        args.extend([
            "-f".into(),
            "lavfi".into(),
            "-t".into(),
            format!("{:.3}", duration),
            "-i".into(),
            "anullsrc=r=48000:cl=stereo".into(),
            "-map".into(),
            "0:v:0".into(),
            "-map".into(),
            "1:a:0".into(),
            "-vf".into(),
            filter,
            "-c:v".into(),
            "libx264".into(),
            "-preset".into(),
            "veryfast".into(),
            "-crf".into(),
            "20".into(),
            "-pix_fmt".into(),
            "yuv420p".into(),
            "-c:a".into(),
            "aac".into(),
            "-b:a".into(),
            "160k".into(),
            "-shortest".into(),
        ]);
    }
    args.extend([
        "-movflags".into(),
        "+faststart".into(),
        dest.display().to_string(),
    ]);
    args
}

struct Target {
    width: u32,
    height: u32,
    fps: f64,
}

fn output_target(segment: &Segment) -> Target {
    Target {
        width: even_dim(segment.width),
        height: even_dim(segment.height),
        fps: if segment.fps.is_finite() && (1.0..=240.0).contains(&segment.fps) {
            segment.fps
        } else {
            30.0
        },
    }
}

fn even_dim(value: u32) -> u32 {
    let value = value.max(2);
    if value.is_multiple_of(2) {
        value
    } else {
        value - 1
    }
}

fn format_fps(fps: f64) -> String {
    if (fps - fps.round()).abs() < 0.01 {
        format!("{}", fps.round() as i64)
    } else {
        format!("{fps:.3}")
    }
}

fn concat_line(path: &Path) -> String {
    let text = path.to_string_lossy().replace('\\', "/");
    let escaped = text.replace('\'', r"'\''");
    format!("file '{escaped}'")
}

fn run_with_progress(
    args: &[String],
    progress_path: &Path,
    duration: f64,
    hooks: &Hooks<'_>,
    progress_from: f32,
    progress_to: f32,
) -> Result<(), ExportError> {
    check_cancel(hooks)?;
    let _ = std::fs::write(progress_path, b"");
    (hooks.log)(shell_join(args));

    let mut child = Command::new(ffmpeg_program())
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| ExportError::Message(format!("Could not start ffmpeg: {err}")))?;

    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| ExportError::Message("Could not read ffmpeg's error output.".into()))?;
    let stderr_handle = thread::spawn(move || {
        let mut buf = Vec::new();
        let mut reader = std::io::BufReader::new(stderr);
        let _ = reader.read_to_end(&mut buf);
        if buf.len() > 16 * 1024 {
            buf = buf[buf.len() - 16 * 1024..].to_vec();
        }
        buf
    });

    let status = loop {
        if hooks.cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stderr_handle.join();
            return Err(ExportError::Cancelled);
        }
        if let Some(fraction) = read_progress(progress_path) {
            let along = (fraction / duration).clamp(0.0, 1.0) as f32;
            (hooks.progress)(progress_from + (progress_to - progress_from) * along);
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => thread::sleep(Duration::from_millis(100)),
            Err(err) => {
                let _ = child.kill();
                let _ = stderr_handle.join();
                return Err(ExportError::Message(format!("ffmpeg wait failed: {err}")));
            }
        }
    };

    let stderr = stderr_handle.join().unwrap_or_default();
    if !status.success() {
        let text = String::from_utf8_lossy(&stderr);
        return Err(ExportError::Message(format!(
            "ffmpeg failed (status {status}).\n{}",
            tail(&text, 12)
        )));
    }
    Ok(())
}

fn read_progress(path: &Path) -> Option<f64> {
    let file = std::fs::File::open(path).ok()?;
    let mut file = file;
    let len = file.metadata().ok()?.len();
    if len > 2048 {
        file.seek(SeekFrom::End(-2048)).ok()?;
    }
    let mut text = String::new();
    file.read_to_string(&mut text).ok()?;
    last_out_time(&text)
}

fn last_out_time(text: &str) -> Option<f64> {
    let mut found = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("out_time=") {
            if rest != "N/A" {
                if let Some(time) = parse_time(rest) {
                    found = Some(time);
                }
            }
        }
    }
    found
}

fn check_cancel(hooks: &Hooks<'_>) -> Result<(), ExportError> {
    if hooks.cancel.load(Ordering::Relaxed) {
        Err(ExportError::Cancelled)
    } else {
        Ok(())
    }
}

fn run_to_completion(mut command: Command) -> Result<std::process::Output, String> {
    let mut child = command
        .spawn()
        .map_err(|err| format!("Could not start ffmpeg: {err}"))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let out_handle = thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut pipe) = stdout {
            let _ = pipe.read_to_end(&mut buf);
        }
        buf
    });
    let err_handle = thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut pipe) = stderr {
            let _ = pipe.read_to_end(&mut buf);
        }
        buf
    });
    let status = child
        .wait()
        .map_err(|err| format!("ffmpeg wait failed: {err}"))?;
    let stdout = out_handle.join().unwrap_or_default();
    let stderr = err_handle.join().unwrap_or_default();
    if !status.success() {
        let err = String::from_utf8_lossy(&stderr);
        return Err(format!("ffmpeg failed.\n{}", tail(&err, 8)));
    }
    Ok(std::process::Output {
        status,
        stdout,
        stderr,
    })
}

fn shell_join(args: &[String]) -> String {
    let mut line = String::from("ffmpeg");
    for arg in args {
        line.push(' ');
        if arg.chars().any(|c| c.is_whitespace() || c == '\'') {
            line.push('\'');
            line.push_str(&arg.replace('\'', r"'\''"));
            line.push('\'');
        } else {
            line.push_str(arg);
        }
    }
    line
}

fn tail(text: &str, lines: usize) -> String {
    let kept: Vec<&str> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    if kept.is_empty() {
        return "(no ffmpeg output)".to_string();
    }
    kept[kept.len().saturating_sub(lines)..].join("\n")
}

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Result<Self, ExportError> {
        let millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("videocliptrimmer-{}-{millis}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|err| {
            ExportError::Message(format!("Could not create a temp folder: {err}"))
        })?;
        Ok(TempDir(dir))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn parse_probe(text: &str) -> Result<MediaInfo, String> {
    let duration = text
        .lines()
        .find(|line| line.contains("Duration:"))
        .and_then(|line| {
            let rest = line.split("Duration:").nth(1)?;
            let stamp = rest.split(',').next()?.trim();
            if stamp == "N/A" {
                None
            } else {
                parse_time(stamp)
            }
        })
        .filter(|d| *d > 0.0)
        .ok_or_else(|| "Could not read a duration. Is this a video file?".to_string())?;

    let mut video_codec = String::new();
    let mut width = 0u32;
    let mut height = 0u32;
    let mut fps = 0.0;
    let mut has_audio = false;
    let mut audio_codec = String::new();

    for line in text.lines() {
        if line.contains("Video:") && !line.contains("attached pic") && width == 0 {
            video_codec = codec_name(line, "Video:");
            if let Some((w, h)) = find_wxh(line) {
                width = w;
                height = h;
            }
            if let Some(value) = find_fps(line) {
                fps = value;
            }
        }
        if line.contains("Audio:") && !has_audio {
            has_audio = true;
            audio_codec = codec_name(line, "Audio:");
        }
    }

    if width < 2 || height < 2 {
        return Err("Could not read the video size.".into());
    }

    // Phone videos are often stored sideways, with a rotation tag. Swap so the
    // export size matches what you see.
    if let Some(angle) = find_rotation(text) {
        let turned = angle.abs() % 180.0;
        if (turned - 90.0).abs() < 1.0 {
            std::mem::swap(&mut width, &mut height);
        }
    }

    if fps <= 0.0 {
        fps = 30.0;
    }

    Ok(MediaInfo {
        duration,
        width,
        height,
        fps,
        has_audio,
        video_codec,
        audio_codec,
    })
}

fn codec_name(line: &str, key: &str) -> String {
    let Some(rest) = line.split(key).nth(1) else {
        return String::new();
    };
    let name: String = rest
        .trim()
        .chars()
        .take_while(|c| !c.is_whitespace() && *c != '(')
        .collect();
    name
}

fn find_wxh(line: &str) -> Option<(u32, u32)> {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b'x' {
                let width: u32 = line[start..i].parse().ok()?;
                let hstart = i + 1;
                let mut j = hstart;
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                if j > hstart {
                    if let Ok(height) = line[hstart..j].parse::<u32>() {
                        if width >= 16 && height >= 16 {
                            return Some((width, height));
                        }
                    }
                }
            }
        } else {
            i += 1;
        }
    }
    None
}

fn find_fps(line: &str) -> Option<f64> {
    for part in line.split(',') {
        let part = part.trim();
        if let Some(num) = part.strip_suffix(" fps") {
            if let Ok(value) = num.trim().parse::<f64>() {
                if value > 0.0 && value < 1000.0 {
                    return Some(value);
                }
            }
        }
    }
    None
}

fn find_rotation(text: &str) -> Option<f64> {
    for line in text.lines() {
        if let Some(rest) = line.split("rotation of ").nth(1) {
            let num: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || matches!(*c, '-' | '+' | '.'))
                .collect();
            if let Ok(angle) = num.parse::<f64>() {
                return Some(angle);
            }
        }
    }
    None
}

/// Used by the UI thread. The grab itself runs on a worker so the window stays live.
pub fn spawn_grab(path: PathBuf, time: f64, seq: u64) -> mpsc::Receiver<Grabbed> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let result = grab_frame(&path, time);
        let _ = tx.send(Grabbed { seq, time, result });
    });
    rx
}

pub struct Grabbed {
    pub seq: u64,
    pub time: f64,
    pub result: Result<RgbaFrame, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'src.mp4':
  Duration: 00:00:02.00, start: 0.000000, bitrate: 145 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1920x1080 [SAR 1:1 DAR 16:9], 63 kb/s, 25 fps, 25 tbr, 12800 tbn (default)
  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, mono, fltp, 69 kb/s (default)
    displaymatrix: rotation of -90.00 degrees
"#;

    #[test]
    fn parses_ffmpeg_banner() {
        let info = parse_probe(SAMPLE).unwrap();
        assert!((info.duration - 2.0).abs() < 0.001);
        // Rotated, so the display size is portrait.
        assert_eq!(info.width, 1080);
        assert_eq!(info.height, 1920);
        assert!((info.fps - 25.0).abs() < 0.01);
        assert!(info.has_audio);
        assert_eq!(info.video_codec, "h264");
        assert_eq!(info.audio_codec, "aac");
    }

    #[test]
    fn progress_line_uses_out_time_not_the_microsecond_field() {
        let text = "out_time_ms=520000\nout_time=00:00:00.520000\nprogress=end\n";
        assert!((last_out_time(text).unwrap() - 0.52).abs() < 0.001);
    }

    #[test]
    fn fast_copy_keeps_the_streams() {
        let segment = Segment {
            path: PathBuf::from("clip.mp4"),
            start: 1.0,
            end: 2.5,
            width: 100,
            height: 80,
            fps: 30.0,
            has_audio: true,
        };
        let args = fast_copy_args(&segment, Path::new("out.mp4"), Path::new("p.txt"));
        assert!(args.windows(2).any(|w| w == ["-c", "copy"]));
        assert!(args.iter().any(|a| a == "-avoid_negative_ts"));
        assert!(args.iter().any(|a| a == "1.500"));
    }

    #[test]
    fn concat_line_escapes_quotes() {
        let line = concat_line(Path::new("/tmp/it's.mp4"));
        assert!(line.starts_with("file '"));
        assert!(line.contains(r"'\''"));
    }

    fn ffmpeg_ready() -> bool {
        check_ffmpeg().is_ok()
    }

    #[test]
    fn cuts_and_joins_a_real_file() {
        if !ffmpeg_ready() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("videocliptrimmer-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.mp4");
        let status = Command::new(ffmpeg_program())
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=160x120:rate=25:duration=2",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration=2",
                "-c:v",
                "libx264",
                "-g",
                "10",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
            ])
            .arg(&src)
            .status()
            .unwrap();
        assert!(status.success());

        let info = probe(&src).unwrap();
        assert!((info.duration - 2.0).abs() < 0.15);
        assert_eq!((info.width, info.height), (160, 120));
        assert!(info.has_audio);

        let frame = grab_frame(&src, 0.4).unwrap();
        assert!(frame.width > 0 && frame.height > 0);
        assert_eq!(
            frame.pixels.len(),
            frame.width as usize * frame.height as usize * 4
        );

        let segment = |start: f64, end: f64| Segment {
            path: src.clone(),
            start,
            end,
            width: info.width,
            height: info.height,
            fps: info.fps,
            has_audio: info.has_audio,
        };
        let dest = dir.join("joined.mp4");
        let cancel = AtomicBool::new(false);
        let hooks = Hooks {
            progress: &|_| {},
            log: &|_| {},
            cancel: &cancel,
        };
        export(
            &[segment(0.0, 0.5), segment(1.0, 1.5)],
            &dest,
            EncodeMode::Compatible,
            &hooks,
        )
        .unwrap();
        let joined = probe(&dest).unwrap();
        assert!(
            (joined.duration - 1.0).abs() < 0.2,
            "joined duration {}",
            joined.duration
        );
        assert_eq!((joined.width, joined.height), (160, 120));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
