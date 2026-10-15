//! Every ffmpeg command lives in this file.
//!
//! The app shells out to the `ffmpeg` binary. It does not link libav.
//! Change a flag here and the whole app picks it up.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

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

/// Which hardware encoder to use. `Off` is libx264 on the CPU.
///
/// Preview uses the matching hardware decoder while this is not `Off`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HwAccel {
    Off,
    Nvidia,
    Amd,
    Intel,
}

/// Quality and audio rate used when `EncodeMode::Compatible` re-encodes.
///
/// `video_crf` is 0–51, same scale as x264. Lower keeps more of the picture.
/// `audio_bitrate` is an ffmpeg rate such as `192k`.
#[derive(Clone, Debug)]
pub struct EncodeSettings {
    pub video_crf: i32,
    pub audio_bitrate: String,
    pub hw: HwAccel,
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
    /// Pixel format without the extra tags, for example `yuv420p`.
    pub pix_fmt: String,
    /// `tv` (limited), `pc` (full), or empty when ffmpeg did not say.
    pub color_range: String,
    /// ffmpeg color names. Empty when the file did not say.
    pub color_space: String,
    pub color_primaries: String,
    pub color_transfer: String,
    /// Bits per second, when the probe printed a stream bitrate.
    pub video_bitrate: Option<u64>,
    pub audio_bitrate: Option<u64>,
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
    pub pix_fmt: String,
    pub color_range: String,
    pub color_space: String,
    pub color_primaries: String,
    pub color_transfer: String,
}

impl MediaInfo {
    pub fn segment(&self, path: PathBuf, start: f64, end: f64) -> Segment {
        Segment {
            path,
            start,
            end,
            width: self.width,
            height: self.height,
            fps: self.fps,
            has_audio: self.has_audio,
            pix_fmt: self.pix_fmt.clone(),
            color_range: self.color_range.clone(),
            color_space: self.color_space.clone(),
            color_primaries: self.color_primaries.clone(),
            color_transfer: self.color_transfer.clone(),
        }
    }
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

pub struct GrabOutcome {
    pub frame: RgbaFrame,
    pub warning: Option<String>,
}

pub fn grab_frame(path: &Path, time: f64, hw: HwAccel) -> Result<GrabOutcome, String> {
    if hw == HwAccel::Off {
        let frame = grab_frame_once(path, time, HwAccel::Off)?;
        return Ok(GrabOutcome {
            frame,
            warning: None,
        });
    }
    match grab_frame_once(path, time, hw) {
        Ok(frame) => Ok(GrabOutcome {
            frame,
            warning: None,
        }),
        Err(err) => match grab_frame_once(path, time, HwAccel::Off) {
            Ok(frame) => Ok(GrabOutcome {
                frame,
                warning: Some(format!(
                    "Hardware decode failed. Using the CPU decoder instead.\n{err}"
                )),
            }),
            Err(soft) => Err(format!("{err}\nSoftware decode failed as well.\n{soft}")),
        },
    }
}

fn grab_frame_once(path: &Path, time: f64, hw: HwAccel) -> Result<RgbaFrame, String> {
    let mut command = Command::new(ffmpeg_program());
    command
        .args(["-hide_banner", "-loglevel", "error", "-nostdin"])
        .args(hw_decode_args(hw))
        .args(["-ss", &format!("{time:.3}"), "-i"])
        .arg(path)
        .args([
            "-map",
            "0:v:0",
            "-an",
            "-frames:v",
            "1",
            // Fit inside a 960 box. Preview only — the export keeps full resolution.
            "-vf",
            "scale=960:960:force_original_aspect_ratio=decrease:flags=fast_bilinear",
            "-c:v",
            "mjpeg",
            "-q:v",
            "5",
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

/// One open ffmpeg process that decodes forward. Playback reads this instead of
/// starting a new ffmpeg for every frame.
pub struct PlaySession {
    pub path: PathBuf,
    pub hw: HwAccel,
    rx: mpsc::Receiver<PlayEvent>,
    stop: Arc<AtomicBool>,
    child: Arc<Mutex<Option<Child>>>,
}

pub enum PlayEvent {
    Warning(String),
    Frame { time: f64, frame: RgbaFrame },
    Ended,
    Failed(String),
}

pub struct PlayRequest {
    pub path: PathBuf,
    pub start: f64,
    pub fps: f64,
    pub width: u32,
    pub height: u32,
    pub hw: HwAccel,
}

impl PlaySession {
    pub fn try_recv(&self) -> Result<PlayEvent, mpsc::TryRecvError> {
        self.rx.try_recv()
    }
}

impl Drop for PlaySession {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Ok(mut slot) = self.child.lock() {
            if let Some(child) = slot.as_mut() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

pub fn spawn_play(request: PlayRequest) -> PlaySession {
    let stop = Arc::new(AtomicBool::new(false));
    let child = Arc::new(Mutex::new(None));
    let (tx, rx) = mpsc::sync_channel(2);
    let stop_worker = Arc::clone(&stop);
    let child_worker = Arc::clone(&child);
    let path = request.path.clone();
    let hw = request.hw;
    thread::spawn(move || play_worker(request, stop_worker, child_worker, tx));
    PlaySession {
        path,
        hw,
        rx,
        stop,
        child,
    }
}

fn play_worker(
    request: PlayRequest,
    stop: Arc<AtomicBool>,
    child: Arc<Mutex<Option<Child>>>,
    tx: mpsc::SyncSender<PlayEvent>,
) {
    let hw = request.hw;
    match play_decode(&request, hw, &stop, &child, &tx) {
        Ok(()) => {}
        Err(err) if hw != HwAccel::Off && !stop.load(Ordering::Relaxed) => {
            let _ = tx.try_send(PlayEvent::Warning(
                "Hardware decode failed. Using the CPU decoder instead.".into(),
            ));
            if let Err(soft) = play_decode(&request, HwAccel::Off, &stop, &child, &tx) {
                let _ = tx.try_send(PlayEvent::Failed(format!(
                    "{err}\nSoftware decode failed as well.\n{soft}"
                )));
            }
        }
        Err(err) => {
            let _ = tx.try_send(PlayEvent::Failed(err));
        }
    }
}

fn play_decode(
    request: &PlayRequest,
    hw: HwAccel,
    stop: &AtomicBool,
    child_slot: &Mutex<Option<Child>>,
    tx: &mpsc::SyncSender<PlayEvent>,
) -> Result<(), String> {
    let fps = if request.fps.is_finite() && (1.0..240.0).contains(&request.fps) {
        request.fps
    } else {
        30.0
    };
    let (width, height) = preview_size(request.width, request.height);
    if let Ok(mut slot) = child_slot.lock() {
        if let Some(child) = slot.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        *slot = None;
    }
    let scale = format!("scale={width}:{height}:flags=fast_bilinear");
    let mut command = Command::new(ffmpeg_program());
    command
        .args(["-hide_banner", "-loglevel", "error", "-nostdin"])
        .args(hw_decode_args(hw))
        .args(["-ss", &format!("{:.3}", request.start), "-i"])
        .arg(&request.path)
        .args([
            "-map", "0:v:0", "-an", "-vf", &scale, "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|err| format!("Could not start ffmpeg: {err}"))?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| "ffmpeg did not return a preview frame.".to_string())?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| "ffmpeg did not return a preview frame.".to_string())?;
    {
        let mut slot = child_slot.lock().unwrap();
        *slot = Some(child);
    }
    let errors = Arc::new(Mutex::new(Vec::new()));
    let errors_worker = Arc::clone(&errors);
    thread::spawn(move || {
        let _ = std::io::copy(&mut stderr, &mut Writer(&errors_worker));
    });
    if stop.load(Ordering::Relaxed) {
        return Ok(());
    }

    let frame_bytes = width as usize * height as usize * 4;
    let mut pixels = vec![0u8; frame_bytes];
    let mut index = 0u64;
    let clock = Instant::now();
    loop {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        if stdout.read_exact(&mut pixels).is_err() {
            if index == 0 {
                let text = String::from_utf8_lossy(&errors.lock().unwrap()).to_string();
                return Err(if text.trim().is_empty() {
                    "ffmpeg did not return a preview frame.".into()
                } else {
                    tail(&text, 8)
                });
            }
            let _ = tx.try_send(PlayEvent::Ended);
            return Ok(());
        }
        let time = request.start + index as f64 / fps;
        index += 1;
        let due = Duration::from_secs_f64(index as f64 / fps);
        if clock.elapsed().saturating_sub(due) > Duration::from_secs_f64(1.5 / fps) {
            // Behind realtime: drop the frame instead of queueing a slideshow.
            continue;
        }
        while clock.elapsed() + Duration::from_millis(2) < due {
            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            let left = due.saturating_sub(clock.elapsed());
            thread::sleep(left.min(Duration::from_millis(8)));
        }
        let frame = RgbaFrame {
            width,
            height,
            pixels: pixels.clone(),
        };
        match tx.try_send(PlayEvent::Frame { time, frame }) {
            Ok(()) => {}
            Err(mpsc::TrySendError::Full(_)) => {}
            Err(mpsc::TrySendError::Disconnected(_)) => return Ok(()),
        }
    }
}

/// Small writer so a background thread can drain ffmpeg's stderr without blocking it.
struct Writer<'a>(&'a Mutex<Vec<u8>>);

impl std::io::Write for Writer<'_> {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(data);
        Ok(data.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Longest side of a preview frame. Full-resolution decode is what makes CPU playback heavy.
pub fn preview_size(width: u32, height: u32) -> (u32, u32) {
    const MAX: u32 = 960;
    if width < 2 || height < 2 {
        return (640, 360);
    }
    let (w, h) = if width <= MAX && height <= MAX {
        (width, height)
    } else {
        let scale = (MAX as f64 / width as f64).min(MAX as f64 / height as f64);
        (
            (width as f64 * scale).round() as u32,
            (height as f64 * scale).round() as u32,
        )
    };
    (even_dim(w.max(2)), even_dim(h.max(2)))
}

pub fn export(
    segments: &[Segment],
    dest: &Path,
    mode: EncodeMode,
    settings: &EncodeSettings,
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
        encode_segment(&segments[0], dest, mode, None, settings, hooks, (0.0, 1.0))?;
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
        encode_segment(segment, &part, mode, target, settings, hooks, (start, end))?;
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
    settings: &EncodeSettings,
    hooks: &Hooks<'_>,
    progress: (f32, f32),
) -> Result<(), ExportError> {
    let progress_path = dest.with_extension("progress");
    let gpu = mode == EncodeMode::Compatible && settings.hw == HwAccel::Intel && cfg!(windows);
    let args = encode_args(
        segment,
        dest,
        mode,
        target.as_ref(),
        settings,
        &progress_path,
        gpu,
    )?;
    let duration = (segment.end - segment.start).max(0.05);
    match run_with_progress(
        &args,
        &progress_path,
        duration,
        hooks,
        progress.0,
        progress.1,
    ) {
        Ok(()) => {}
        Err(ExportError::Cancelled) => return Err(ExportError::Cancelled),
        Err(_) if gpu => {
            check_cancel(hooks)?;
            (hooks.log)(
                "Quick Sync could not keep frames on the GPU. Using the slower encoder.".into(),
            );
            let _ = std::fs::remove_file(dest);
            let slow = encode_args(
                segment,
                dest,
                mode,
                target.as_ref(),
                settings,
                &progress_path,
                false,
            )?;
            run_with_progress(
                &slow,
                &progress_path,
                duration,
                hooks,
                progress.0,
                progress.1,
            )?;
        }
        Err(err) => return Err(err),
    }
    let _ = std::fs::remove_file(&progress_path);
    Ok(())
}

fn encode_args(
    segment: &Segment,
    dest: &Path,
    mode: EncodeMode,
    target: Option<&Target>,
    settings: &EncodeSettings,
    progress_path: &Path,
    gpu: bool,
) -> Result<Vec<String>, ExportError> {
    Ok(match mode {
        EncodeMode::FastCopy => fast_copy_args(segment, dest, progress_path),
        EncodeMode::Compatible => {
            let crf = settings.video_crf.clamp(0, 51);
            let audio_bitrate = normalize_bitrate(&settings.audio_bitrate)
                .map_err(|err| ExportError::Message(format!("Audio bitrate: {err}")))?;
            let owned;
            let target = match target {
                Some(target) => target,
                None => {
                    owned = output_target(segment);
                    &owned
                }
            };
            compatible_args(
                segment,
                dest,
                target,
                progress_path,
                crf,
                &audio_bitrate,
                settings.hw,
                gpu,
            )
        }
    })
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

#[allow(clippy::too_many_arguments)]
fn compatible_args(
    segment: &Segment,
    dest: &Path,
    target: &Target,
    progress: &Path,
    video_crf: i32,
    audio_bitrate: &str,
    hw: HwAccel,
    gpu: bool,
) -> Vec<String> {
    let duration = segment.end - segment.start;
    let mut args = vec![
        "-y".into(),
        "-hide_banner".into(),
        "-nostats".into(),
        "-progress".into(),
        progress.display().to_string(),
        "-loglevel".into(),
        "error".into(),
    ];
    // `gpu` keeps Windows Intel frames on the encode engine. The other path
    // copies every frame back to memory, which only keeps the decode unit busy.
    let gpu = gpu && hw == HwAccel::Intel && cfg!(windows);
    if gpu {
        args.extend(intel_gpu_input_args());
    } else {
        args.extend(export_input_args(hw));
    }
    args.extend([
        // `-ss` before `-i` is fast. Re-encoding still starts near the requested frame,
        // because ffmpeg decodes back to the previous keyframe.
        "-ss".into(),
        format!("{:.3}", segment.start),
        "-t".into(),
        format!("{:.3}", duration),
        "-i".into(),
        segment.path.display().to_string(),
    ]);

    if segment.has_audio {
        args.extend(["-map".into(), "0:v:0".into(), "-map".into(), "0:a:0".into()]);
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
        ]);
    }

    // Leave size, frame rate, and pixel format alone when they already match.
    // A scale or fps filter resamples the picture even if the numbers look the same.
    let filter = if gpu {
        Some(intel_gpu_filter(segment, target))
    } else {
        video_filter(segment, target, hw)
    };
    if let Some(filter) = filter {
        args.extend(["-vf".into(), filter]);
    }

    if gpu {
        args.extend(intel_fast_encoder_args(segment, video_crf));
    } else {
        args.extend(video_encoder_args(segment, video_crf, hw));
    }
    args.extend([
        "-c:a".into(),
        "aac".into(),
        "-b:a".into(),
        audio_bitrate.to_string(),
        "-ar".into(),
        "48000".into(),
        "-ac".into(),
        "2".into(),
    ]);
    if !segment.has_audio {
        args.push("-shortest".into());
    }
    args.extend([
        "-movflags".into(),
        "+faststart".into(),
        dest.display().to_string(),
    ]);
    args
}

/// D3D11 surfaces mapped into Quick Sync. One decoder thread, or the fixed
/// surface pool overflows (`Static surface pool size exceeded`).
fn intel_gpu_input_args() -> Vec<String> {
    vec![
        "-init_hw_device".into(),
        "d3d11va=intel,vendor=0x8086".into(),
        "-hwaccel".into(),
        "d3d11va".into(),
        "-hwaccel_device".into(),
        "intel".into(),
        "-hwaccel_output_format".into(),
        "d3d11".into(),
        "-threads".into(),
        "1".into(),
        "-extra_hw_frames".into(),
        "128".into(),
    ]
}

fn intel_gpu_filter(segment: &Segment, target: &Target) -> String {
    let mut chain = String::from("hwmap=derive_device=qsv,format=qsv");
    let size_differs = segment.width != target.width || segment.height != target.height;
    let fps_differs = (segment.fps - target.fps).abs() > 0.01;
    if size_differs || fps_differs {
        let mut vpp = format!("vpp_qsv=w={}:h={}", target.width, target.height);
        if fps_differs {
            vpp.push_str(&format!(":framerate={}", format_fps(target.fps)));
        }
        chain.push(',');
        chain.push_str(&vpp);
    }
    chain
}

fn intel_fast_encoder_args(segment: &Segment, crf: i32) -> Vec<String> {
    let quality = crf.clamp(0, 51).to_string();
    let mut args = vec![
        "-c:v".into(),
        "h264_qsv".into(),
        // veryfast + low_power is Intel's VDENC path, the one HandBrake uses.
        "-preset".into(),
        "veryfast".into(),
        "-low_power".into(),
        "1".into(),
        "-async_depth".into(),
        "4".into(),
        "-look_ahead".into(),
        "0".into(),
        "-global_quality".into(),
        quality,
    ];
    args.extend(color_args(segment));
    args
}

/// Filters only when something about the picture must change to match `target`.
fn video_filter(segment: &Segment, target: &Target, hw: HwAccel) -> Option<String> {
    let mut parts = Vec::new();
    let size_differs = segment.width != target.width || segment.height != target.height;
    if size_differs {
        let mut scale = format!(
            "scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos",
            w = target.width,
            h = target.height,
        );
        // Same range in and out, so a resize does not wash the color out or crush it.
        if segment.color_range == "tv" || segment.color_range == "pc" {
            let range = &segment.color_range;
            scale.push_str(&format!(":in_range={range}:out_range={range}"));
        }
        parts.push(scale);
        parts.push(format!(
            "pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black",
            w = target.width,
            h = target.height,
        ));
        parts.push("setsar=1".into());
    }
    if (segment.fps - target.fps).abs() > 0.01 {
        parts.push(format!("fps={}", format_fps(target.fps)));
    }

    let pix = output_pix_fmt(segment, hw);
    if pix != segment.pix_fmt {
        // yuvj* is full-range JPEG style. Convert the levels, don't just rename them.
        if segment.pix_fmt.starts_with("yuvj")
            && (segment.color_range == "pc" || segment.color_range.is_empty())
        {
            parts.push("scale=in_range=pc:out_range=pc".into());
        }
        parts.push(format!("format={pix}"));
    }
    if let Some(upload) = upload_filter(hw) {
        parts.push(upload.to_string());
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(","))
    }
}

fn video_encoder_args(segment: &Segment, crf: i32, hw: HwAccel) -> Vec<String> {
    let quality = crf.clamp(0, 51).to_string();
    let mut args = Vec::new();
    match hw {
        HwAccel::Off => {
            args.extend([
                "-c:v".into(),
                "libx264".into(),
                // `medium` keeps more detail than `veryfast` at the same CRF.
                "-preset".into(),
                "medium".into(),
                "-crf".into(),
                quality,
            ]);
        }
        HwAccel::Nvidia => {
            // HandBrake's quality mode: CQ with no bitrate cap, spatial AQ on.
            // `p4` is its balanced preset. `-b:v 0` stops the default bitrate
            // from overriding the quality number.
            args.extend([
                "-c:v".into(),
                "h264_nvenc".into(),
                "-preset".into(),
                "p4".into(),
                "-tune".into(),
                "hq".into(),
                "-rc".into(),
                "vbr".into(),
                "-cq".into(),
                quality,
                "-b:v".into(),
                "0".into(),
                "-spatial-aq".into(),
                "1".into(),
                "-profile:v".into(),
                "high".into(),
            ]);
        }
        HwAccel::Amd => args.extend(amd_encoder_args(&quality)),
        HwAccel::Intel => {
            // `global_quality` is Intel's CRF-style control. Lower keeps more detail.
            args.extend([
                "-c:v".into(),
                "h264_qsv".into(),
                "-preset".into(),
                "medium".into(),
                "-global_quality".into(),
                quality,
            ]);
        }
    }
    let pix = output_pix_fmt(segment, hw);
    if (hw == HwAccel::Intel && cfg!(windows))
        || hw == HwAccel::Nvidia
        || (hw == HwAccel::Amd && cfg!(windows))
    {
        // NVENC and AMF take nv12, the same layout HandBrake feeds them.
        // yuv420p makes ffmpeg insert a conversion the encoder may reject.
        args.extend(["-pix_fmt".into(), "nv12".into()]);
    } else if !uses_hw_upload(hw) {
        args.extend(["-pix_fmt".into(), pix]);
    }
    args.extend(color_args(segment));
    args
}

fn amd_encoder_args(crf: &str) -> Vec<String> {
    if cfg!(windows) {
        // CQP is HandBrake's AMD quality mode. `balanced` is its default speed.
        vec![
            "-c:v".into(),
            "h264_amf".into(),
            "-usage".into(),
            "transcoding".into(),
            "-quality".into(),
            "balanced".into(),
            "-rc".into(),
            "cqp".into(),
            "-qp_i".into(),
            crf.to_string(),
            "-qp_p".into(),
            crf.to_string(),
            "-qp_b".into(),
            crf.to_string(),
            "-profile:v".into(),
            "high".into(),
        ]
    } else if cfg!(target_os = "macos") {
        // Macs have no AMF encoder. VideoToolbox is the hardware encoder there.
        // Its quality is 1–100 and higher is better, the opposite of CRF.
        let value: i32 = crf.parse().unwrap_or(18);
        vec![
            "-c:v".into(),
            "h264_videotoolbox".into(),
            "-q:v".into(),
            videotoolbox_quality(value).to_string(),
            "-profile:v".into(),
            "high".into(),
        ]
    } else {
        vec![
            "-c:v".into(),
            "h264_vaapi".into(),
            "-rc_mode".into(),
            "CQP".into(),
            "-qp".into(),
            crf.to_string(),
        ]
    }
}

/// VideoToolbox quality 1–100 from a CRF value. CRF 0 → 100, CRF 51 → 1.
fn videotoolbox_quality(crf: i32) -> i32 {
    let crf = crf.clamp(0, 51);
    (((51 - crf) * 100) / 51).clamp(1, 100)
}

/// Flags that go before `-i` while playing or grabbing one frame.
///
/// Intel on Windows uses D3D11 here, not `-hwaccel qsv`. The QSV decoder
/// fails on a lot of Intel laptops (`Error number -1145393733`) even when
/// other apps decode on the same GPU.
fn hw_decode_args(hw: HwAccel) -> Vec<String> {
    match hw {
        HwAccel::Off => Vec::new(),
        // CUDA is NVDEC. HandBrake decodes there and keeps NV12.
        // `nvdec` is only an alias, and some builds do not accept it.
        HwAccel::Nvidia => vec![
            "-hwaccel".into(),
            "cuda".into(),
            "-hwaccel_output_format".into(),
            "nv12".into(),
        ],
        HwAccel::Intel if cfg!(windows) => vec![
            "-hwaccel".into(),
            "d3d11va".into(),
            "-hwaccel_output_format".into(),
            "nv12".into(),
        ],
        HwAccel::Intel => vec![
            "-hwaccel".into(),
            "qsv".into(),
            "-hwaccel_output_format".into(),
            "nv12".into(),
        ],
        // D3D11VA on the AMD adapter. The AMF decoder itself fails to open.
        // 0x1002 is AMD's vendor id, so a second GPU is not picked by mistake.
        HwAccel::Amd if cfg!(windows) => vec![
            "-init_hw_device".into(),
            "d3d11va=amd,vendor=0x1002".into(),
            "-hwaccel".into(),
            "d3d11va".into(),
            "-hwaccel_device".into(),
            "amd".into(),
            "-hwaccel_output_format".into(),
            "nv12".into(),
        ],
        HwAccel::Amd if cfg!(target_os = "macos") => {
            vec!["-hwaccel".into(), "videotoolbox".into()]
        }
        HwAccel::Amd => vec![
            "-vaapi_device".into(),
            vaapi_device(),
            "-hwaccel".into(),
            "vaapi".into(),
        ],
    }
}

/// Decode flags for export. Windows Intel matches playback: D3D11 decode,
/// then normal nv12 frames. Keeping the frames as D3D11 surfaces overflowed
/// the pool (`Static surface pool size exceeded`).
fn export_input_args(hw: HwAccel) -> Vec<String> {
    if hw == HwAccel::Intel && cfg!(windows) {
        return hw_decode_args(HwAccel::Intel);
    }
    if hw == HwAccel::Intel {
        return vec![
            "-init_hw_device".into(),
            "qsv=hw".into(),
            "-filter_hw_device".into(),
            "hw".into(),
            "-hwaccel".into(),
            "qsv".into(),
            "-hwaccel_device".into(),
            "hw".into(),
            "-hwaccel_output_format".into(),
            "nv12".into(),
        ];
    }
    let mut args = hw_device_init(hw);
    // VAAPI already puts its device in the decode flags.
    if hw == HwAccel::Amd && !(cfg!(windows) || cfg!(target_os = "macos")) {
        args.clear();
    }
    args.extend(hw_decode_args(hw));
    args
}

fn hw_device_init(hw: HwAccel) -> Vec<String> {
    match hw {
        // Windows Intel does not use this. Export maps D3D11 frames instead.
        HwAccel::Intel if cfg!(windows) => Vec::new(),
        HwAccel::Intel => vec![
            "-init_hw_device".into(),
            "qsv=hw".into(),
            "-filter_hw_device".into(),
            "hw".into(),
        ],
        HwAccel::Amd if !(cfg!(windows) || cfg!(target_os = "macos")) => {
            vec!["-vaapi_device".into(), vaapi_device()]
        }
        _ => Vec::new(),
    }
}

fn uses_hw_upload(hw: HwAccel) -> bool {
    hw == HwAccel::Intel || hw_needs_upload(hw)
}

fn upload_filter(hw: HwAccel) -> Option<&'static str> {
    if hw == HwAccel::Intel && cfg!(windows) {
        return None;
    }
    if hw == HwAccel::Intel {
        // `format=qsv` stops ffmpeg inserting auto_scale in front of h264_qsv.
        return Some("format=nv12,hwupload=extra_hw_frames=64,format=qsv");
    }
    if hw_needs_upload(hw) {
        return Some("format=nv12,hwupload");
    }
    None
}

fn hw_needs_upload(hw: HwAccel) -> bool {
    // VAAPI encode wants the frame uploaded. The other encoders accept CPU frames.
    hw == HwAccel::Amd && !(cfg!(windows) || cfg!(target_os = "macos"))
}

fn vaapi_device() -> String {
    match std::env::var("VAAPI_DEVICE") {
        Ok(path) if !path.trim().is_empty() => path,
        _ => "/dev/dri/renderD128".to_string(),
    }
}

pub fn hw_name(hw: HwAccel) -> &'static str {
    match hw {
        HwAccel::Off => "off",
        HwAccel::Nvidia => "nvidia",
        HwAccel::Amd => "amd",
        HwAccel::Intel => "intel",
    }
}

pub fn parse_hw_name(text: &str) -> Option<HwAccel> {
    match text.trim().to_ascii_lowercase().as_str() {
        "off" => Some(HwAccel::Off),
        "nvidia" => Some(HwAccel::Nvidia),
        "amd" => Some(HwAccel::Amd),
        "intel" => Some(HwAccel::Intel),
        _ => None,
    }
}

/// First hardware pair that can both encode and decode here.
/// Tries NVIDIA, then AMD, then Intel. `Off` when none of them work.
pub fn detect_hw() -> HwAccel {
    for hw in [HwAccel::Nvidia, HwAccel::Amd, HwAccel::Intel] {
        if hw_pair_works(hw) {
            return hw;
        }
    }
    HwAccel::Off
}

pub fn spawn_hw_detect() -> mpsc::Receiver<HwAccel> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(detect_hw());
    });
    rx
}

fn hw_pair_works(hw: HwAccel) -> bool {
    let dir = std::env::temp_dir().join(format!(
        "videocliptrimmer-hw-{}-{}",
        std::process::id(),
        hw_name(hw)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    if std::fs::create_dir_all(&dir).is_err() {
        return false;
    }
    let sample = dir.join("sample.mp4");
    let ok = encoder_accepts(hw, &sample) && decoder_accepts(hw, &sample);
    let _ = std::fs::remove_dir_all(&dir);
    ok
}

fn encoder_accepts(hw: HwAccel, dest: &Path) -> bool {
    if hw == HwAccel::Intel && cfg!(windows) {
        return intel_windows_encoder_accepts(dest);
    }
    let mut cmd = Command::new(ffmpeg_program());
    cmd.args(["-hide_banner", "-loglevel", "error", "-y"]);
    cmd.args(hw_device_init(hw));
    cmd.args(["-f", "lavfi", "-i", "color=c=black:s=64x64:r=25:d=0.08"]);
    if let Some(filter) = upload_filter(hw) {
        cmd.args(["-vf", filter]);
    }
    cmd.args(video_encoder_args(&probe_segment(), 23, hw));
    cmd.arg(dest);
    command_ok(cmd, Duration::from_secs(5)) && dest.is_file()
}

/// Lavfi frames are not D3D11 surfaces, so the Windows probe transcodes a tiny file.
fn intel_windows_encoder_accepts(dest: &Path) -> bool {
    let tiny = dest.with_extension("src.mp4");
    let mut make = Command::new(ffmpeg_program());
    make.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x64:r=25:d=0.08",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
    ])
    .arg(&tiny);
    if !command_ok(make, Duration::from_secs(5)) {
        return false;
    }
    let segment = probe_segment();
    let target = output_target(&segment);
    let fast = intel_probe_encode(&tiny, dest, &segment, &target, true);
    let ok = fast || intel_probe_encode(&tiny, dest, &segment, &target, false);
    let _ = std::fs::remove_file(tiny);
    ok
}

fn intel_probe_encode(
    src: &Path,
    dest: &Path,
    segment: &Segment,
    target: &Target,
    gpu: bool,
) -> bool {
    let mut cmd = Command::new(ffmpeg_program());
    cmd.args(["-hide_banner", "-loglevel", "error", "-y"]);
    if gpu {
        cmd.args(intel_gpu_input_args());
    } else {
        cmd.args(export_input_args(HwAccel::Intel));
    }
    cmd.arg("-i").arg(src);
    let filter = if gpu {
        Some(intel_gpu_filter(segment, target))
    } else {
        video_filter(segment, target, HwAccel::Intel)
    };
    if let Some(filter) = filter {
        cmd.args(["-vf", &filter]);
    }
    if gpu {
        cmd.args(intel_fast_encoder_args(segment, 23));
    } else {
        cmd.args(video_encoder_args(segment, 23, HwAccel::Intel));
    }
    cmd.arg(dest);
    command_ok(cmd, Duration::from_secs(8)) && dest.is_file()
}

fn decoder_accepts(hw: HwAccel, src: &Path) -> bool {
    let mut cmd = Command::new(ffmpeg_program());
    cmd.args(["-hide_banner", "-loglevel", "error"]);
    cmd.args(hw_decode_args(hw));
    cmd.arg("-i").arg(src);
    cmd.args(["-frames:v", "1", "-f", "null", "-"]);
    command_ok(cmd, Duration::from_secs(5))
}

fn probe_segment() -> Segment {
    Segment {
        path: PathBuf::new(),
        start: 0.0,
        end: 0.1,
        width: 64,
        height: 64,
        fps: 25.0,
        has_audio: false,
        pix_fmt: "yuv420p".into(),
        color_range: String::new(),
        color_space: String::new(),
        color_primaries: String::new(),
        color_transfer: String::new(),
    }
}

fn command_ok(mut cmd: Command, timeout: Duration) -> bool {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(_) => return false,
    };
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return status.success(),
            Ok(None) if started.elapsed() > timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return false;
            }
            Ok(None) => thread::sleep(Duration::from_millis(40)),
            Err(_) => return false,
        }
    }
}

/// Pixel format written to the file. Hardware H.264 encoders stay on `yuv420p`.
/// libx264 keeps the source format when that format is already a normal yuv one.
fn output_pix_fmt(segment: &Segment, hw: HwAccel) -> String {
    if hw != HwAccel::Off {
        return "yuv420p".to_string();
    }
    match segment.pix_fmt.as_str() {
        "yuv420p" | "yuv422p" | "yuv444p" | "yuv420p10le" | "yuv422p10le" | "yuv444p10le" => {
            segment.pix_fmt.clone()
        }
        "yuvj420p" | "yuvj422p" | "yuvj444p" => "yuv420p".to_string(),
        _ => "yuv420p".to_string(),
    }
}

fn color_args(segment: &Segment) -> Vec<String> {
    let mut args = Vec::new();
    if segment.color_range == "tv" || segment.color_range == "pc" {
        args.extend(["-color_range".into(), segment.color_range.clone()]);
    }
    if !segment.color_space.is_empty() {
        args.extend(["-colorspace".into(), segment.color_space.clone()]);
    }
    if !segment.color_primaries.is_empty() {
        args.extend(["-color_primaries".into(), segment.color_primaries.clone()]);
    }
    if !segment.color_transfer.is_empty() {
        args.extend(["-color_trc".into(), segment.color_transfer.clone()]);
    }
    args
}

/// `8000k` or `8M`, or a plain number of kilobits (`192` → `192k`).
///
/// `M` stays uppercase. ffmpeg treats `m` as milli, which is not a useful bitrate.
pub fn normalize_bitrate(text: &str) -> Result<String, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("enter a bitrate, for example 8000k.".into());
    }
    let (number, suffix) = match text.as_bytes().last().map(|c| c.to_ascii_lowercase()) {
        Some(b'k') => (&text[..text.len() - 1], "k"),
        Some(b'm') => (&text[..text.len() - 1], "M"),
        _ => (text, "k"),
    };
    match number.parse::<f64>() {
        Ok(value) if value.is_finite() && value > 0.0 => {
            if value.fract() == 0.0 {
                Ok(format!("{}{suffix}", value as u64))
            } else {
                Ok(format!("{value}{suffix}"))
            }
        }
        _ => Err(format!("'{text}' is not a bitrate. Use 8000k or 8M.")),
    }
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
        let dir =
            std::env::temp_dir().join(format!("videocliptrimmer-{}-{millis}", std::process::id()));
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
    let mut pix_fmt = String::new();
    let mut color_range = String::new();
    let mut color_space = String::new();
    let mut color_primaries = String::new();
    let mut color_transfer = String::new();
    let mut video_bitrate = None;
    let mut has_audio = false;
    let mut audio_codec = String::new();
    let mut audio_bitrate = None;

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
            if let Some(parsed) = parse_pix_fmt(line) {
                pix_fmt = parsed.pix_fmt;
                color_range = parsed.range;
                color_space = parsed.space;
                color_primaries = parsed.primaries;
                color_transfer = parsed.transfer;
            }
            video_bitrate = first_bitrate_bps(line);
        }
        if line.contains("Audio:") && !has_audio {
            has_audio = true;
            audio_codec = codec_name(line, "Audio:");
            audio_bitrate = first_bitrate_bps(line);
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
        pix_fmt,
        color_range,
        color_space,
        color_primaries,
        color_transfer,
        video_bitrate,
        audio_bitrate,
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

struct PixFmt {
    pix_fmt: String,
    range: String,
    space: String,
    primaries: String,
    transfer: String,
}

/// `yuv420p(tv, bt709)` → pixel format plus the color tags written in that parenthesis.
fn parse_pix_fmt(line: &str) -> Option<PixFmt> {
    let token = pix_fmt_token(line)?;
    let name = token.split('(').next().unwrap_or(token).trim();
    if name.is_empty() {
        return None;
    }
    let mut range = String::new();
    let mut space = String::new();
    if let Some(inside) = token
        .split_once('(')
        .and_then(|(_, rest)| rest.split(')').next())
    {
        for item in inside.split(',') {
            let item = item.trim();
            match item {
                "tv" | "mpeg" | "limited" => range = "tv".to_string(),
                "pc" | "jpeg" | "full" => range = "pc".to_string(),
                "progressive" | "top" | "bottom" | "unknown" | "" => {}
                other => {
                    if space.is_empty() {
                        space = other.to_string();
                    }
                }
            }
        }
    }
    // yuvj420p is full-range JPEG. Keep that so a later convert does not shift levels.
    if name.starts_with("yuvj") && range.is_empty() {
        range = "pc".to_string();
    }
    let (primaries, transfer, space) = match space.as_str() {
        "bt709" => ("bt709".into(), "bt709".into(), "bt709".into()),
        "bt470bg" => ("bt470bg".into(), "bt470bg".into(), "bt470bg".into()),
        "smpte170m" => ("smpte170m".into(), "smpte170m".into(), "smpte170m".into()),
        "bt2020" | "bt2020nc" => ("bt2020".into(), String::new(), "bt2020nc".into()),
        "" => (String::new(), String::new(), String::new()),
        other => (String::new(), String::new(), other.to_string()),
    };
    Some(PixFmt {
        pix_fmt: name.to_string(),
        range,
        space,
        primaries,
        transfer,
    })
}

fn pix_fmt_token(line: &str) -> Option<&str> {
    let start = ["yuv", "nv12", "nv21", "p010le", "gbr", "rgb", "gray"]
        .iter()
        .filter_map(|marker| line.find(marker))
        .min()?;
    let rest = &line[start..];
    let mut depth = 0i32;
    for (index, ch) in rest.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                if depth > 0 {
                    depth -= 1;
                }
                if depth == 0 {
                    return Some(&rest[..=index]);
                }
            }
            ',' if depth == 0 => return Some(rest[..index].trim()),
            _ => {}
        }
    }
    let end = rest
        .find(|ch: char| ch.is_whitespace())
        .unwrap_or(rest.len());
    Some(rest[..end].trim())
}

fn first_bitrate_bps(line: &str) -> Option<u64> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    for pair in parts.windows(2) {
        let unit = pair[1].trim_matches(|c| c == ',' || c == ')');
        let number = pair[0].trim_matches(|c| c == ',' || c == '(');
        let factor = match unit {
            "kb/s" => 1_000.0,
            "mb/s" => 1_000_000.0,
            _ => continue,
        };
        let value: f64 = number.parse().ok()?;
        if value > 0.0 {
            return Some((value * factor).round() as u64);
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
pub fn spawn_grab(path: PathBuf, time: f64, seq: u64, hw: HwAccel) -> mpsc::Receiver<Grabbed> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let grabbed = match grab_frame(&path, time, hw) {
            Ok(outcome) => Grabbed {
                seq,
                time,
                result: Ok(outcome.frame),
                warning: outcome.warning,
            },
            Err(err) => Grabbed {
                seq,
                time,
                result: Err(err),
                warning: None,
            },
        };
        let _ = tx.send(grabbed);
    });
    rx
}

pub struct Grabbed {
    pub seq: u64,
    pub time: f64,
    pub result: Result<RgbaFrame, String>,
    pub warning: Option<String>,
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
        assert_eq!(info.pix_fmt, "yuv420p");
        assert_eq!(info.video_bitrate, Some(63_000));
        assert_eq!(info.audio_bitrate, Some(69_000));
    }

    #[test]
    fn reads_color_tags_from_the_pixel_format() {
        let line = "Stream #0:0: Video: h264, yuv420p(tv, bt709), 1280x720, 4500 kb/s, 30 fps";
        let parsed = parse_pix_fmt(line).unwrap();
        assert_eq!(parsed.pix_fmt, "yuv420p");
        assert_eq!(parsed.range, "tv");
        assert_eq!(parsed.space, "bt709");
        assert_eq!(parsed.primaries, "bt709");
        assert_eq!(parsed.transfer, "bt709");
        assert_eq!(first_bitrate_bps(line), Some(4_500_000));
    }

    #[test]
    fn yuvj_is_full_range() {
        let parsed = parse_pix_fmt("Video: mjpeg, yuvj420p, 640x360, 30 fps").unwrap();
        assert_eq!(parsed.pix_fmt, "yuvj420p");
        assert_eq!(parsed.range, "pc");
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
            pix_fmt: "yuv420p".into(),
            color_range: "tv".into(),
            color_space: "bt709".into(),
            color_primaries: "bt709".into(),
            color_transfer: "bt709".into(),
        };
        let args = fast_copy_args(&segment, Path::new("out.mp4"), Path::new("p.txt"));
        assert!(args.windows(2).any(|w| w == ["-c", "copy"]));
        assert!(args.iter().any(|a| a == "-avoid_negative_ts"));
        assert!(args.iter().any(|a| a == "1.500"));
    }

    #[test]
    fn same_picture_is_not_scaled_or_recolored() {
        let segment = Segment {
            path: PathBuf::from("clip.mp4"),
            start: 0.0,
            end: 1.0,
            width: 1920,
            height: 1080,
            fps: 25.0,
            has_audio: true,
            pix_fmt: "yuv420p".into(),
            color_range: "tv".into(),
            color_space: "bt709".into(),
            color_primaries: "bt709".into(),
            color_transfer: "bt709".into(),
        };
        let target = output_target(&segment);
        assert!(video_filter(&segment, &target, HwAccel::Off).is_none());
        let args = compatible_args(
            &segment,
            Path::new("out.mp4"),
            &target,
            Path::new("p.txt"),
            18,
            "192k",
            HwAccel::Off,
            false,
        );
        assert!(!args.iter().any(|arg| arg == "-vf" || arg == "-b:v"));
        assert!(args.windows(2).any(|pair| pair == ["-crf", "18"]));
        assert!(args.windows(2).any(|pair| pair == ["-b:a", "192k"]));
        assert!(args.windows(2).any(|pair| pair == ["-color_range", "tv"]));
        assert!(args.windows(2).any(|pair| pair == ["-colorspace", "bt709"]));
        assert!(args.windows(2).any(|pair| pair == ["-pix_fmt", "yuv420p"]));

        let nvidia = video_encoder_args(&segment, 18, HwAccel::Nvidia);
        assert!(nvidia.windows(2).any(|pair| pair == ["-cq", "18"]));
        assert!(nvidia.windows(2).any(|pair| pair == ["-b:v", "0"]));
        assert!(nvidia.windows(2).any(|pair| pair == ["-pix_fmt", "nv12"]));
        let intel = video_encoder_args(&segment, 18, HwAccel::Intel);
        assert!(intel
            .windows(2)
            .any(|pair| pair == ["-global_quality", "18"]));
        assert!(!intel.iter().any(|arg| arg == "-pix_fmt"));
        assert_eq!(
            video_filter(&segment, &target, HwAccel::Intel).as_deref(),
            Some("format=nv12,hwupload=extra_hw_frames=64,format=qsv")
        );
        let qsv = hw_decode_args(HwAccel::Intel);
        assert!(qsv.windows(2).any(|pair| pair == ["-hwaccel", "qsv"]));
        assert!(qsv
            .windows(2)
            .any(|pair| pair == ["-hwaccel_output_format", "nv12"]));
        assert!(!qsv.iter().any(|arg| arg.contains("init_hw_device")));
        let export = export_input_args(HwAccel::Intel);
        assert!(export.iter().any(|arg| arg.contains("qsv=hw")));
        assert!(export
            .windows(2)
            .any(|pair| pair == ["-hwaccel_device", "hw"]));
        assert_eq!(preview_size(1920, 1080), (960, 540));
        assert_eq!(preview_size(160, 120), (160, 120));
        assert_eq!(videotoolbox_quality(0), 100);
        assert_eq!(videotoolbox_quality(18), 64);
        assert_eq!(videotoolbox_quality(51), 1);
        assert_eq!(parse_hw_name("NVIDIA"), Some(HwAccel::Nvidia));
        assert_eq!(hw_name(HwAccel::Intel), "intel");
    }

    #[test]
    fn nvidia_decode_is_requested_when_that_encoder_is_on() {
        let args = hw_decode_args(HwAccel::Nvidia);
        assert!(args.windows(2).any(|pair| pair == ["-hwaccel", "cuda"]));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["-hwaccel_output_format", "nv12"]));
        assert_eq!(normalize_bitrate("8M").unwrap(), "8M");
        assert_eq!(normalize_bitrate("192").unwrap(), "192k");
        assert!(normalize_bitrate("nope").is_err());
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
        let dir =
            std::env::temp_dir().join(format!("videocliptrimmer-test-{}", std::process::id()));
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

        let frame = grab_frame(&src, 0.4, HwAccel::Off).unwrap().frame;
        assert!(frame.width > 0 && frame.height > 0);
        assert_eq!(
            frame.pixels.len(),
            frame.width as usize * frame.height as usize * 4
        );

        let session = spawn_play(PlayRequest {
            path: src.clone(),
            start: 0.0,
            fps: info.fps,
            width: info.width,
            height: info.height,
            hw: HwAccel::Off,
        });
        let started = std::time::Instant::now();
        let mut played = false;
        while started.elapsed() < std::time::Duration::from_secs(8) {
            match session.try_recv() {
                Ok(PlayEvent::Frame { frame, .. }) => {
                    assert!(frame.width > 0 && frame.height > 0);
                    played = true;
                    break;
                }
                Ok(PlayEvent::Failed(err)) => panic!("playback failed: {err}"),
                Ok(_) => {}
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(20)),
            }
        }
        drop(session);
        assert!(played, "playback did not produce a frame");

        let segment = |start: f64, end: f64| info.segment(src.clone(), start, end);
        let dest = dir.join("joined.mp4");
        let cancel = AtomicBool::new(false);
        let hooks = Hooks {
            progress: &|_| {},
            log: &|_| {},
            cancel: &cancel,
        };
        let settings = EncodeSettings {
            video_crf: 18,
            audio_bitrate: "160k".into(),
            hw: HwAccel::Off,
        };
        export(
            &[segment(0.0, 0.5), segment(1.0, 1.5)],
            &dest,
            EncodeMode::Compatible,
            &settings,
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
