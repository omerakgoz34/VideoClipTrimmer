//! The window layout. State lives on `App`; drawing happens in `App::ui`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

use crate::ffmpeg::{
    self, EncodeMode, EncodeSettings, ExportError, Grabbed, Hooks, HwAccel, MediaInfo, PlayEvent,
    PlayRequest, PlaySession, Segment, check_ffmpeg, export, hw_name, parse_hw_name, spawn_grab,
    spawn_hw_detect, spawn_play,
};
use crate::mpv::Player;
use crate::timeutil::{format_time, parse_time};

const MIN_RANGE: f64 = 0.05;
const VIDEO_EXT: &[&str] = &[
    "mp4", "mov", "mkv", "webm", "avi", "m4v", "mpg", "mpeg", "wmv", "flv", "ts", "mts", "m2ts",
];

pub struct App {
    ffmpeg: Result<String, String>,
    media: Option<Loaded>,
    playhead: f64,
    in_point: f64,
    out_point: f64,
    in_text: String,
    out_text: String,
    in_edit: bool,
    out_edit: bool,
    playing: bool,
    loop_range: bool,
    clips: Vec<Clip>,
    selected: Option<usize>,
    mode: EncodeMode,
    video_crf: i32,
    /// One CRF for Off, NVIDIA, AMD, and Intel, in that order.
    crf: [i32; 4],
    audio_bitrate: String,
    fps: String,
    hw: HwAccel,
    hw_touched: bool,
    hw_detect: Option<Receiver<HwAccel>>,
    hw_warned: bool,
    /// True after this playback session gave up on the hardware decoder.
    play_using_cpu: bool,
    path_text: String,
    notice: Notice,
    log: String,
    clear_armed: bool,
    preview: Option<egui::TextureHandle>,
    shown_time: Option<f64>,
    wanted_time: Option<f64>,
    grab: Option<Receiver<Grabbed>>,
    grab_seq: u64,
    play: Option<PlaySession>,
    /// Real player. When this is open, the preview is mpv, not grabbed frames.
    film: Option<Player>,
    mpv_ok: bool,
    mpv_warned: bool,
    settings_open: bool,
    job: Option<Job>,
    themed: bool,
    /// Shown once at startup when ffmpeg is not installed.
    warn_ffmpeg: bool,
    /// Shown once at startup when the mpv library is not installed.
    warn_mpv: bool,
}

struct Loaded {
    path: PathBuf,
    info: MediaInfo,
}

struct Clip {
    label: String,
    path: PathBuf,
    start: f64,
    end: f64,
    info: MediaInfo,
}

struct Job {
    rx: Receiver<JobMsg>,
    cancel: Arc<AtomicBool>,
    progress: f32,
    dest: PathBuf,
}

enum JobMsg {
    Progress(f32),
    Log(String),
    Finished(Result<(), ExportError>),
}

enum Notice {
    None,
    Info(String),
    Error(String),
}

impl App {
    pub fn new() -> Self {
        let ffmpeg = check_ffmpeg();
        let warn_ffmpeg = ffmpeg.is_err();
        let mpv_ok = crate::mpv::library_present();
        let saved = load_settings();
        let hw = saved.hw.unwrap_or(HwAccel::Off);
        // A saved choice means this is not the first launch. Don't probe again.
        let hw_touched = saved.hw.is_some();
        let hw_detect = if !hw_touched && ffmpeg.is_ok() {
            Some(spawn_hw_detect())
        } else {
            None
        };
        let crf = saved.crf;
        let video_crf = crf[hw_index(hw)];
        Self {
            ffmpeg,
            media: None,
            playhead: 0.0,
            in_point: 0.0,
            out_point: 0.0,
            in_text: format_time(0.0),
            out_text: format_time(0.0),
            in_edit: false,
            out_edit: false,
            playing: false,
            loop_range: false,
            clips: Vec::new(),
            selected: None,
            mode: saved.mode,
            video_crf,
            crf,
            audio_bitrate: saved.audio,
            fps: saved.fps,
            hw,
            hw_touched,
            hw_detect,
            hw_warned: false,
            play_using_cpu: false,
            path_text: String::new(),
            notice: Notice::None,
            log: String::new(),
            clear_armed: false,
            preview: None,
            shown_time: None,
            wanted_time: None,
            grab: None,
            grab_seq: 0,
            play: None,
            film: None,
            mpv_ok,
            mpv_warned: false,
            settings_open: false,
            job: None,
            themed: false,
            warn_ffmpeg,
            warn_mpv: !mpv_ok,
        }
    }

    pub fn wants_redraw(&self) -> bool {
        self.playing
            || self.grab.is_some()
            || self.job.is_some()
            || self.wanted_time.is_some()
            || self.hw_detect.is_some()
            || self.film.as_ref().is_some_and(Player::waiting)
    }

    /// How long to wait before the next frame. Idle windows wait on the OS instead.
    pub fn redraw_wait(&self) -> Duration {
        if self.playing && self.film.is_some() {
            // mpv keeps the clock. We just sample it often enough to follow.
            Duration::from_millis(8)
        } else if self.playing {
            let fps = self
                .media
                .as_ref()
                .map(|media| media.info.fps)
                .filter(|fps| fps.is_finite() && *fps > 1.0)
                .unwrap_or(30.0)
                .clamp(24.0, 60.0);
            Duration::from_secs_f64(1.0 / fps)
        } else if self.wants_redraw() {
            // A job or a single frame is in flight. Polling faster than this just burns a core.
            Duration::from_millis(16)
        } else {
            Duration::ZERO
        }
    }

    pub fn open_paths(&mut self, paths: Vec<PathBuf>) {
        if paths.is_empty() {
            return;
        }
        self.clear_armed = false;
        if paths.len() == 1 {
            self.selected = None;
            self.load_file(paths.into_iter().next().unwrap(), None);
            return;
        }
        // Several files: put each whole file on the list, and show the first.
        let mut first = None;
        for path in paths {
            match ffmpeg::probe(&path) {
                Ok(info) => {
                    let index = self.clips.len();
                    self.clips.push(Clip::whole(path, info));
                    if first.is_none() {
                        first = Some(index);
                    }
                }
                Err(err) => self.fail(err),
            }
        }
        if let Some(index) = first {
            self.select_clip(index);
            self.info(format!(
                "Added {} clips.",
                self.clips.len().saturating_sub(index)
            ));
        }
    }

    pub fn ui(&mut self, ctx: &egui::Context) {
        self.ensure_theme(ctx);
        self.poll_hw_detect();
        self.handle_keys(ctx);
        self.pump_preview(ctx);
        self.poll_job();
        self.sync_time_fields();

        egui::TopBottomPanel::top("title").show(ctx, |ui| self.title_bar(ui));

        egui::TopBottomPanel::bottom("controls").show(ctx, |ui| self.controls(ui));

        egui::SidePanel::right("clips")
            .resizable(true)
            .default_width(300.0)
            .width_range(240.0..=440.0)
            .show(ctx, |ui| self.clip_list(ui));

        egui::CentralPanel::default().show(ctx, |ui| self.stage(ui));
        self.settings_popup(ctx);
        self.ffmpeg_popup(ctx);
        self.mpv_popup(ctx);
    }

    fn ensure_theme(&mut self, ctx: &egui::Context) {
        if self.themed {
            return;
        }
        self.themed = true;
        let mut style = (*ctx.style()).clone();
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(10.0, 5.0);
        style.text_styles.insert(
            egui::TextStyle::Body,
            egui::FontId::new(14.5, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Button,
            egui::FontId::new(14.5, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Small,
            egui::FontId::new(12.0, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Heading,
            egui::FontId::new(20.0, egui::FontFamily::Proportional),
        );
        ctx.set_style(style);

        let mut visuals = egui::Visuals::dark();
        let ink = rgb(0xec, 0xea, 0xe4);
        visuals.panel_fill = rgb(0x16, 0x18, 0x1c);
        visuals.window_fill = rgb(0x16, 0x18, 0x1c);
        visuals.extreme_bg_color = rgb(0x10, 0x11, 0x14);
        visuals.faint_bg_color = rgb(0x22, 0x25, 0x2a);
        visuals.widgets.noninteractive.bg_fill = rgb(0x16, 0x18, 0x1c);
        visuals.widgets.noninteractive.fg_stroke.color = ink;
        visuals.widgets.inactive.bg_fill = rgb(0x2c, 0x31, 0x38);
        visuals.widgets.inactive.fg_stroke.color = ink;
        visuals.widgets.hovered.bg_fill = rgb(0x3a, 0x42, 0x4a);
        visuals.widgets.hovered.fg_stroke.color = egui::Color32::WHITE;
        visuals.widgets.active.bg_fill = rgb(0x3d, 0x6a, 0x74);
        visuals.widgets.active.fg_stroke.color = egui::Color32::WHITE;
        visuals.selection.bg_fill = rgb(0x3d, 0x6a, 0x74);
        visuals.hyperlink_color = rgb(0x8f, 0xb8, 0xc4);
        visuals.override_text_color = Some(ink);
        ctx.set_visuals(visuals);
    }

    fn title_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_centered(|ui| {
            ui.heading("VideoClipTrimmer");
            ui.label(muted(crate::VERSION));
            ui.add_space(8.0);
            if let Some(media) = &self.media {
                ui.label(file_label(&media.path));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                match &self.ffmpeg {
                    Ok(version) => {
                        ui.label(muted(&ffmpeg_label(version)))
                            .on_hover_text(version);
                    }
                    Err(_) => {
                        ui.colored_label(rgb(0xe0, 0x7a, 0x5f), FFMPEG_MISSING);
                    }
                }
                if !self.mpv_ok {
                    ui.colored_label(rgb(0xe0, 0x7a, 0x5f), MPV_MISSING);
                }
            });
        });
    }

    fn ffmpeg_popup(&mut self, ctx: &egui::Context) {
        if !self.warn_ffmpeg {
            return;
        }
        let mut close = false;
        egui::Modal::new(egui::Id::new("ffmpeg-missing")).show(ctx, |ui| {
            ui.set_min_width(360.0);
            ui.vertical_centered(|ui| {
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(FFMPEG_MISSING).color(rgb(0xe0, 0x7a, 0x5f)),
                    )
                    .halign(egui::Align::Center),
                );
                ui.add_space(6.0);
                ui.add(
                    egui::Label::new("Install ffmpeg, then open the app again.")
                        .halign(egui::Align::Center),
                );
                ui.add_space(8.0);
                if ui.button("OK").clicked() {
                    close = true;
                }
            });
        });
        if close {
            self.warn_ffmpeg = false;
        }
    }

    fn mpv_popup(&mut self, ctx: &egui::Context) {
        if !self.warn_mpv || self.warn_ffmpeg {
            return;
        }
        let mut close = false;
        egui::Modal::new(egui::Id::new("mpv-missing")).show(ctx, |ui| {
            ui.set_min_width(360.0);
            ui.vertical_centered(|ui| {
                ui.add(
                    egui::Label::new(egui::RichText::new(MPV_MISSING).color(rgb(0xe0, 0x7a, 0x5f)))
                        .halign(egui::Align::Center),
                );
                ui.add_space(6.0);
                ui.add(
                    egui::Label::new("Put libmpv next to the app, then open it again.")
                        .halign(egui::Align::Center),
                );
                ui.add_space(8.0);
                if ui.button("OK").clicked() {
                    close = true;
                }
            });
        });
        if close {
            self.warn_mpv = false;
        }
    }

    fn stage(&mut self, ui: &mut egui::Ui) {
        if let Some(media) = &self.media {
            ui.label(summary(&media.info));
            ui.add_space(4.0);
        }

        ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
            self.transport(ui);
            let size = ui.available_size();
            let frame = egui::Frame::NONE
                .fill(rgb(0x10, 0x11, 0x14))
                .inner_margin(8.0)
                .corner_radius(6.0);
            frame.show(ui, |ui| {
                let inner = (size - egui::vec2(16.0, 16.0)).max(egui::vec2(40.0, 80.0));
                ui.set_min_size(inner);
                ui.set_max_size(inner);
                if self.media.is_none() {
                    self.empty_state(ui);
                    return;
                }
                let clicked = if let Some(tex) = &self.preview {
                    paint_fitted(ui, tex).clicked()
                } else {
                    let available = ui.available_size();
                    let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click());
                    ui.painter().text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "Reading a frame…",
                        egui::TextStyle::Body.resolve(ui.style()),
                        ui.visuals().text_color(),
                    );
                    response.clicked()
                };
                if clicked {
                    self.toggle_play();
                }
            });
        });
    }

    fn empty_state(&mut self, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.add_space(28.0);
            ui.label("Open a video, mark the part you want, and add it to the list.");
            ui.label("Export one range, or join the list into a single file.");
            ui.add_space(12.0);
            if icon_button(ui, true, Icon::Folder, "Open videos…", true)
                .on_hover_text("You can also drop files onto this window.")
                .clicked()
            {
                self.browse_open();
            }
            ui.add_space(8.0);
            ui.label(muted("Or drop files onto this window."));
        });
    }

    fn transport(&mut self, ui: &mut egui::Ui) {
        let has_media = self.media.is_some();
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            // Remember the row's width so it stays centered on the video,
            // not on the wider window under the clip list.
            let id = ui.id().with("transport-width");
            let known = ui.data(|data| data.get_temp::<f32>(id).unwrap_or(0.0));
            let spare = (ui.available_width() - known).max(0.0);
            ui.add_space(spare / 2.0);
            let start = ui.cursor().min.x;

            if icon_button(ui, has_media, Icon::Back, "1s", false).clicked() {
                self.nudge(-1.0);
            }
            let (icon, play) = if self.playing {
                (Icon::Pause, "Pause")
            } else {
                (Icon::Play, "Play")
            };
            if icon_button(ui, has_media, icon, play, false).clicked() {
                self.toggle_play();
            }
            if icon_button(ui, has_media, Icon::Forward, "1s", false).clicked() {
                self.nudge(1.0);
            }
            ui.label(
                egui::RichText::new(format!(
                    "{}    /    {}",
                    format_time(self.playhead),
                    format_time(self.duration())
                ))
                .size(16.0),
            );
            ui.checkbox(&mut self.loop_range, "Loop the range");

            let width = ui.cursor().min.x - start;
            ui.data_mut(|data| data.insert_temp(id, width));
        });
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        let has_media = self.media.is_some();
        let busy = self.job.is_some();
        ui.add_space(4.0);

        if let Some(time) = timeline(
            ui,
            self.duration(),
            self.playhead,
            self.in_point,
            self.out_point,
        ) {
            self.seek(time);
            self.clear_armed = false;
        }

        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if icon_button(ui, true, Icon::Settings, "Settings", false).clicked() {
                    self.settings_open = true;
                }
                ui.label(self.encode_summary());
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    self.range_controls(ui, has_media);
                });
            });
        });

        ui.horizontal_wrapped(|ui| {
            if icon_button(ui, has_media, Icon::Plus, "Add to list", true).clicked() {
                self.add_current_range();
            }
            if icon_button(
                ui,
                has_media && self.selected.is_some(),
                Icon::Update,
                "Update selected",
                false,
            )
            .clicked()
            {
                self.update_selected();
            }
            if icon_button(
                ui,
                has_media && !busy && self.ffmpeg.is_ok(),
                Icon::Export,
                "Export this range…",
                true,
            )
            .clicked()
            {
                self.export_current();
            }
        });

        self.status_line(ui);
        ui.label(muted(
            "Space play   ·   I / O set in and out   ·   arrows seek 1s, Shift 5s",
        ));
        ui.add_space(2.0);
    }

    fn range_controls(&mut self, ui: &mut egui::Ui, has_media: bool) {
        ui.label(
            egui::RichText::new("In")
                .strong()
                .color(rgb(0x8f, 0xb8, 0xc4)),
        );
        let in_edit = ui.add_enabled(
            has_media,
            egui::TextEdit::singleline(&mut self.in_text).desired_width(96.0),
        );
        if in_edit.changed()
            && let Some(time) = parse_time(&self.in_text)
        {
            self.in_point = time;
            self.clamp_range();
        }
        self.in_edit = in_edit.has_focus();
        if icon_button(ui, has_media, Icon::Mark, "Set", false).clicked() {
            self.in_point = self.playhead;
            self.clamp_range();
            self.write_time_fields();
        }
        if icon_button(ui, has_media, Icon::Go, "Go", false).clicked() {
            self.seek(self.in_point);
        }

        ui.add_space(16.0);
        ui.label(
            egui::RichText::new("Out")
                .strong()
                .color(rgb(0x8f, 0xb8, 0xc4)),
        );
        let out_edit = ui.add_enabled(
            has_media,
            egui::TextEdit::singleline(&mut self.out_text).desired_width(96.0),
        );
        if out_edit.changed()
            && let Some(time) = parse_time(&self.out_text)
        {
            self.out_point = time;
            self.clamp_range();
        }
        self.out_edit = out_edit.has_focus();
        if icon_button(ui, has_media, Icon::Mark, "Set", false).clicked() {
            self.out_point = self.playhead;
            self.clamp_range();
            self.write_time_fields();
        }
        if icon_button(ui, has_media, Icon::Go, "Go", false).clicked() {
            self.seek(self.out_point);
        }
    }

    fn encode_summary(&self) -> egui::RichText {
        let mode = match self.mode {
            EncodeMode::FastCopy => "Fast-Copy".to_string(),
            EncodeMode::Compatible => format!("Re-Encode · CRF {}", self.video_crf),
        };
        let text = if self.hw_detect.is_some() {
            format!("{mode} · checking hardware…")
        } else {
            format!("{mode} · {}", hw_label(self.hw))
        };
        muted(&text)
    }

    fn settings_popup(&mut self, ctx: &egui::Context) {
        if !self.settings_open {
            return;
        }
        let mut close = false;
        let response = egui::Modal::new(egui::Id::new("encode-settings")).show(ctx, |ui| {
            ui.set_min_width(440.0);
            ui.heading("Settings");
            ui.add_space(8.0);

            ui.label(egui::RichText::new("Export").strong());
            let previous_mode = self.mode;
            ui.horizontal(|ui| {
                ui.radio_value(&mut self.mode, EncodeMode::FastCopy, "Fast-Copy");
                ui.radio_value(&mut self.mode, EncodeMode::Compatible, "Re-Encode");
            });
            if self.mode != previous_mode {
                self.persist();
            }
            ui.label(muted(match self.mode {
                EncodeMode::FastCopy => {
                    "Keeps the original quality and exports faster but the cut can miss the exact frame."
                }
                EncodeMode::Compatible => {
                    "Cuts on the exact frame but picture is rebuilt so it is lower quality and much slower. Lower CRF value means higher quality."
                }
            }));

            if self.mode == EncodeMode::Compatible {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label("Video quality");
                    ui.spacing_mut().slider_width = 180.0;
                    let quality =
                        ui.add(egui::Slider::new(&mut self.video_crf, 0..=51).text("CRF"));
                    ui.label(muted(&format!("for {}", hw_label(self.hw))));
                    if quality.changed() {
                        self.video_crf = self.video_crf.clamp(0, 51);
                        self.crf[hw_index(self.hw)] = self.video_crf;
                        self.persist();
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Audio bitrate");
                    let audio = ui.add(
                        egui::TextEdit::singleline(&mut self.audio_bitrate)
                            .desired_width(80.0)
                            .hint_text("auto"),
                    );
                    if audio.changed() {
                        self.persist();
                    }
                    ui.label(muted("'auto' uses the source. 128k, 256k, 320k etc."));
                });
                ui.horizontal(|ui| {
                    ui.label("Frame rate");
                    let fps = ui.add(
                        egui::TextEdit::singleline(&mut self.fps)
                            .desired_width(80.0)
                            .hint_text("auto"),
                    );
                    if fps.changed() {
                        self.persist();
                    }
                    ui.label(muted("'auto' uses the source. 24, 30, 60 etc."));
                });
            }

            ui.add_space(12.0);
            ui.label(egui::RichText::new("Hardware").strong());
            ui.label(muted(
                "Plays the preview, and encodes when Re-Encode is selected.",
            ));
            let previous = self.hw;
            ui.horizontal_wrapped(|ui| {
                ui.radio_value(&mut self.hw, HwAccel::Off, "Off");
                ui.radio_value(&mut self.hw, HwAccel::Nvidia, "NVIDIA");
                ui.radio_value(&mut self.hw, HwAccel::Amd, amd_button_label());
                ui.radio_value(&mut self.hw, HwAccel::Intel, "Intel");
                if self.hw_detect.is_some() {
                    ui.label(muted("checking…"));
                }
            });
            if self.hw != previous {
                self.apply_hw(previous);
            }

            ui.add_space(14.0);
            ui.vertical_centered(|ui| {
                if ui.button("Close").clicked() {
                    close = true;
                }
            });
        });
        if close || response.should_close() {
            self.settings_open = false;
        }
    }

    fn apply_hw(&mut self, previous: HwAccel) {
        if self.hw == previous {
            return;
        }
        self.hw_warned = false;
        self.clear_hw_fallback_notice();
        self.hw_touched = true;
        self.play = None;
        self.crf[hw_index(previous)] = self.video_crf.clamp(0, 51);
        self.video_crf = self.crf[hw_index(self.hw)];
        self.persist();
        if self.media.is_some() {
            self.wanted_time = Some(self.playhead);
        }
    }

    fn persist(&self) {
        save_settings(self.hw, self.crf, self.mode, &self.audio_bitrate, &self.fps);
    }

    fn status_line(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            match &self.notice {
                Notice::None => {
                    ui.label(muted("Ready."));
                }
                Notice::Info(text) => {
                    ui.label(text);
                }
                Notice::Error(text) => {
                    ui.colored_label(rgb(0xe0, 0x7a, 0x5f), text);
                }
            }
            if let Some(job) = &self.job {
                let progress = job.progress;
                ui.add(
                    egui::ProgressBar::new(progress)
                        .show_percentage()
                        .desired_width(180.0),
                );
                if ui.button("Cancel").clicked() {
                    job.cancel.store(true, Ordering::Relaxed);
                }
            }
        });
        if !self.log.is_empty() {
            egui::CollapsingHeader::new("ffmpeg details")
                .default_open(matches!(self.notice, Notice::Error(_)))
                .show(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .max_height(90.0)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new(&self.log).small().monospace());
                        });
                });
        }
    }

    fn clip_list(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Clips");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let total: f64 = self.clips.iter().map(|c| c.end - c.start).sum();
                ui.label(muted(&format_time(total)));
            });
        });
        ui.label(muted("The list is joined from top to bottom."));
        ui.add_space(4.0);
        if icon_button(ui, true, Icon::Plus, "Add videos…", true).clicked() {
            self.add_videos();
        }
        ui.add_space(4.0);

        ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
            self.path_opener(ui);
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if self.clips.is_empty() {
                    icon_button(ui, false, Icon::Clear, "Clear", false);
                } else if self.clear_armed {
                    if icon_button(ui, true, Icon::Clear, "Click again to clear", false).clicked() {
                        self.clips.clear();
                        self.selected = None;
                        self.clear_armed = false;
                    }
                } else if icon_button(ui, true, Icon::Clear, "Clear", false).clicked() {
                    self.clear_armed = true;
                }
                let busy = self.job.is_some();
                if icon_button(
                    ui,
                    !self.clips.is_empty() && !busy && self.ffmpeg.is_ok(),
                    Icon::Export,
                    "Export list…",
                    true,
                )
                .clicked()
                {
                    self.export_list();
                }
            });
            ui.horizontal(|ui| {
                let has = self.selected.is_some();
                if icon_button(ui, has, Icon::Up, "Up", false).clicked() {
                    self.move_selected(-1);
                }
                if icon_button(ui, has, Icon::Down, "Down", false).clicked() {
                    self.move_selected(1);
                }
                if icon_button(ui, has, Icon::Remove, "Remove", false).clicked() {
                    self.remove_selected();
                }
            });
            ui.add_space(6.0);

            let mut select = None;
            let list_h = ui.available_height().max(40.0);
            egui::ScrollArea::vertical()
                .max_height(list_h)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if self.clips.is_empty() {
                        ui.label(muted("Nothing here yet. Set in and out, then Add to list."));
                        return;
                    }
                    for (index, clip) in self.clips.iter().enumerate() {
                        let selected = self.selected == Some(index);
                        let title = format!("{}.  {}", index + 1, clip.label);
                        let meta = format!(
                            "{}  –  {}     {}",
                            format_time(clip.start),
                            format_time(clip.end),
                            format_time(clip.end - clip.start)
                        );
                        let mut button = egui::Button::new(format!("{title}\n{meta}"))
                            .min_size(egui::vec2(ui.available_width(), 44.0));
                        if selected {
                            button = button.fill(rgb(0x3d, 0x6a, 0x74));
                        }
                        if ui
                            .add(button)
                            .on_hover_text(clip.path.display().to_string())
                            .clicked()
                        {
                            select = Some(index);
                        }
                        ui.add_space(2.0);
                    }
                });
            if let Some(index) = select {
                self.select_clip(index);
            }
        });
    }

    fn path_opener(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Open a path").show(ui, |ui| {
            ui.label(muted("Use this if the file dialog is not available."));
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.path_text)
                        .desired_width(ui.available_width() - 70.0)
                        .hint_text("/path/to/video.mp4"),
                );
                if icon_button(ui, true, Icon::Folder, "Load", false).clicked() {
                    let path = PathBuf::from(self.path_text.trim());
                    if path.as_os_str().is_empty() {
                        return;
                    }
                    self.selected = None;
                    self.load_file(path, None);
                }
            });
        });
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        if ctx.wants_keyboard_input() || self.media.is_none() {
            return;
        }
        let (space, set_in, set_out, left, right, shift) = ctx.input(|input| {
            (
                input.key_pressed(egui::Key::Space),
                input.key_pressed(egui::Key::I),
                input.key_pressed(egui::Key::O),
                input.key_pressed(egui::Key::ArrowLeft),
                input.key_pressed(egui::Key::ArrowRight),
                input.modifiers.shift,
            )
        });
        if space {
            self.toggle_play();
        }
        if set_in {
            self.in_point = self.playhead;
            self.clamp_range();
            self.write_time_fields();
        }
        if set_out {
            self.out_point = self.playhead;
            self.clamp_range();
            self.write_time_fields();
        }
        let step = if shift { 5.0 } else { 1.0 };
        if left {
            self.nudge(-step);
        }
        if right {
            self.nudge(step);
        }
    }

    fn pump_film(&mut self, ctx: &egui::Context) {
        let (frame, time) = {
            let Some(film) = self.film.as_mut() else {
                return;
            };
            film.poll()
        };
        let showed = frame.is_some();
        let clock = time;
        if let Some(pixels) = frame {
            let (width, height) = self
                .film
                .as_ref()
                .map(|film| (film.width(), film.height()))
                .unwrap_or((0, 0));
            if width > 0 && height > 0 {
                self.present_frame(ctx, &pixels, width, height, clock.unwrap_or(self.playhead));
            }
        }
        let Some(time) = clock else {
            return;
        };
        if !self.playing {
            // A seek finished on this frame. Sit the cursor on mpv's clock.
            if showed {
                self.playhead = self.clamp_time(time);
            }
            return;
        }
        if time + 0.0005 >= self.out_point {
            if self.loop_range {
                self.playhead = self.in_point;
                if let Some(film) = self.film.as_mut() {
                    film.seek(self.in_point);
                }
            } else {
                self.playing = false;
                self.playhead = self.clamp_time(self.out_point);
                if let Some(film) = self.film.as_mut() {
                    film.set_paused(true);
                }
            }
            return;
        }
        self.playhead = self.clamp_time(time);
    }

    fn clamp_time(&self, time: f64) -> f64 {
        let duration = self.duration();
        if duration > 0.0 {
            time.clamp(0.0, duration)
        } else {
            time.max(0.0)
        }
    }

    fn open_film(&mut self) {
        let Some(media) = self.media.as_ref() else {
            return;
        };
        let path = media.path.clone();
        let (width, height) = ffmpeg::preview_size(media.info.width, media.info.height);
        let hw = self.hw != HwAccel::Off;
        let playhead = self.playhead;
        self.film = None;
        self.play = None;
        match Player::open(&path, width, height, hw) {
            Ok(mut player) => {
                player.seek(playhead);
                self.film = Some(player);
                self.wanted_time = None;
            }
            Err(err) => {
                self.film = None;
                self.wanted_time = Some(playhead);
                if !self.mpv_ok || err.contains("not installed") {
                    self.mpv_ok = false;
                    self.warn_mpv = true;
                } else if !self.mpv_warned {
                    self.mpv_warned = true;
                    self.fail(err);
                }
            }
        }
    }

    fn pump_preview(&mut self, ctx: &egui::Context) {
        let hw_on = self.hw != HwAccel::Off;
        if self.film.as_ref().is_some_and(|film| film.hw_on() != hw_on) {
            self.open_film();
        }
        if self.film.is_some() {
            self.pump_film(ctx);
            return;
        }
        if self.playing {
            self.pump_play(ctx);
            return;
        }
        // A seek needs a new frame. Pause keeps the decoder so play continues
        // on this frame instead of jumping back to the previous keyframe.
        if self.wanted_time.is_some() {
            self.play = None;
        } else if let Some(play) = &self.play {
            play.set_paused(true);
        }

        let finished = self.grab.as_ref().and_then(|rx| rx.try_recv().ok());
        if finished.is_some() {
            self.grab = None;
        }
        if let Some(Grabbed {
            seq,
            time,
            result,
            warning,
        }) = finished
            && seq == self.grab_seq
        {
            match result {
                Ok(frame) => {
                    self.present_frame(ctx, &frame.pixels, frame.width, frame.height, time);
                    if !self.playing {
                        let duration = self.duration();
                        self.playhead = if duration > 0.0 {
                            time.clamp(0.0, duration)
                        } else {
                            time.max(0.0)
                        };
                    }
                }
                Err(err) => self.fail(format!("Preview: {err}")),
            }
            if let Some(warning) = warning {
                self.note_hw_fallback(&warning);
            } else if self.hw != HwAccel::Off {
                self.clear_hw_fallback_notice();
            }
        }

        let Some(wanted) = self.wanted_time else {
            return;
        };
        if self.grab.is_some() {
            return;
        }
        if self
            .shown_time
            .is_some_and(|shown| (shown - wanted).abs() < 0.04)
        {
            self.wanted_time = None;
            return;
        }
        let Some(path) = self.media.as_ref().map(|m| m.path.clone()) else {
            return;
        };
        let duration = self.duration();
        let time = if duration > 0.05 {
            wanted.clamp(0.0, duration - 0.04)
        } else {
            0.0
        };
        self.grab_seq += 1;
        self.grab = Some(spawn_grab(path, time, self.grab_seq, self.hw));
        self.wanted_time = None;
    }

    fn pump_play(&mut self, ctx: &egui::Context) {
        let Some(media) = self.media.as_ref() else {
            self.playing = false;
            self.play = None;
            return;
        };
        let path = media.path.clone();
        let width = media.info.width;
        let height = media.info.height;
        let hw = self.hw;
        let restart = self
            .play
            .as_ref()
            .map(|play| play.path != path || play.hw != hw)
            .unwrap_or(true);
        if restart {
            self.play_using_cpu = false;
            self.play = Some(spawn_play(PlayRequest {
                path,
                start: self.playhead,
                width,
                height,
                hw,
            }));
        } else if let Some(play) = &self.play {
            play.set_paused(false);
        }
        let events = {
            let Some(play) = &self.play else {
                return;
            };
            // One frame per refresh. Draining the queue showed a later
            // timestamp than the picture that stayed on screen.
            match play.try_recv() {
                Ok(event) => vec![event],
                Err(_) => Vec::new(),
            }
        };
        let mut frame = None;
        let mut ended = false;
        let mut failed = None;
        for event in events {
            match event {
                PlayEvent::Warning(text) => {
                    self.play_using_cpu = true;
                    self.note_hw_fallback(&text);
                }
                PlayEvent::Frame { time, frame: image } => frame = Some((time, image)),
                PlayEvent::Ended => ended = true,
                PlayEvent::Failed(err) => failed = Some(err),
            }
        }
        if let Some(err) = failed {
            self.play = None;
            self.playing = false;
            self.fail(format!("Preview: {err}"));
            return;
        }
        if let Some((time, image)) = frame {
            if !self.play_using_cpu {
                self.clear_hw_fallback_notice();
            }
            // The next picture is past the out point. Keep the frame that is
            // already on screen, and leave the cursor on that frame.
            if time > self.out_point + 0.0005 {
                self.play = None;
                if self.loop_range {
                    self.playhead = self.in_point;
                    self.shown_time = None;
                } else {
                    self.playing = false;
                }
                return;
            }
            self.present_frame(ctx, &image.pixels, image.width, image.height, time);
            self.playhead = time;
            return;
        }
        if ended {
            self.play = None;
            if self.loop_range {
                self.playhead = self.in_point;
                self.shown_time = None;
            } else {
                self.playing = false;
            }
        }
    }

    fn present_frame(
        &mut self,
        ctx: &egui::Context,
        pixels: &[u8],
        width: u32,
        height: u32,
        time: f64,
    ) {
        let image =
            egui::ColorImage::from_rgba_unmultiplied([width as usize, height as usize], pixels);
        let options = egui::TextureOptions::LINEAR;
        if let Some(tex) = &mut self.preview {
            tex.set(image, options);
        } else {
            self.preview = Some(ctx.load_texture("preview", image, options));
        }
        self.shown_time = Some(time);
    }

    fn note_hw_fallback(&mut self, warning: &str) {
        if self.hw_warned {
            return;
        }
        self.hw_warned = true;
        let line = warning.lines().next().unwrap_or(warning).to_string();
        self.info(line);
        self.push_log(warning.to_string());
    }

    fn clear_hw_fallback_notice(&mut self) {
        let showing = matches!(
            &self.notice,
            Notice::Info(text) if text.starts_with("Hardware decode failed")
        );
        if showing {
            self.notice = Notice::None;
            self.hw_warned = false;
        }
    }

    fn poll_job(&mut self) {
        let Some(job) = &self.job else {
            return;
        };
        let mut msgs = Vec::new();
        while let Ok(msg) = job.rx.try_recv() {
            msgs.push(msg);
        }
        let mut finished = None;
        for msg in msgs {
            match msg {
                JobMsg::Progress(value) => {
                    if let Some(job) = &mut self.job {
                        job.progress = value;
                    }
                }
                JobMsg::Log(line) => self.push_log(line),
                JobMsg::Finished(result) => finished = Some(result),
            }
        }
        let Some(result) = finished else {
            return;
        };
        let dest = self.job.take().map(|job| job.dest);
        match result {
            Ok(()) => {
                let where_ = dest
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "the file".into());
                self.info(format!("Saved {where_}"));
            }
            Err(ExportError::Cancelled) => self.info("Export cancelled.".into()),
            Err(ExportError::Message(message)) => self.fail(message),
        }
    }

    fn browse_open(&mut self) {
        self.clear_armed = false;
        let Some(paths) = pick_videos("Open video") else {
            return;
        };
        self.open_paths(paths);
    }

    fn add_videos(&mut self) {
        self.clear_armed = false;
        let Some(paths) = pick_videos("Add videos") else {
            return;
        };
        let start = self.clips.len();
        let mut failed = false;
        for path in paths {
            match ffmpeg::probe(&path) {
                Ok(info) => self.clips.push(Clip::whole(path, info)),
                Err(err) => {
                    failed = true;
                    self.fail(err);
                }
            }
        }
        let added = self.clips.len() - start;
        if added == 0 {
            return;
        }
        if self.media.is_none() {
            self.select_clip(start);
        }
        if !failed {
            let noun = if added == 1 { "clip" } else { "clips" };
            self.info(format!("Added {added} {noun}."));
        }
    }

    fn load_file(&mut self, path: PathBuf, keep: Option<(f64, f64)>) {
        match ffmpeg::probe(&path) {
            Ok(info) => {
                let duration = info.duration;
                self.media = Some(Loaded { path, info });
                self.preview = None;
                self.shown_time = None;
                if let Some((start, end)) = keep {
                    self.in_point = start;
                    self.out_point = end;
                    self.playhead = start;
                } else {
                    self.in_point = 0.0;
                    self.out_point = duration;
                    self.playhead = 0.0;
                }
                self.clamp_range();
                self.write_time_fields();
                self.playing = false;
                self.play = None;
                self.film = None;
                self.notice = Notice::None;
                self.open_film();
            }
            Err(err) => self.fail(err),
        }
    }

    fn add_current_range(&mut self) {
        let Some(media) = &self.media else {
            return;
        };
        self.clips.push(Clip {
            label: file_label(&media.path),
            path: media.path.clone(),
            start: self.in_point,
            end: self.out_point,
            info: media.info.clone(),
        });
        self.selected = Some(self.clips.len() - 1);
        self.clear_armed = false;
        self.info(format!("Added clip {}.", self.clips.len()));
    }

    fn update_selected(&mut self) {
        let Some(index) = self.selected else {
            return;
        };
        let Some(media) = &self.media else {
            return;
        };
        if let Some(clip) = self.clips.get_mut(index) {
            clip.path = media.path.clone();
            clip.label = file_label(&media.path);
            clip.start = self.in_point;
            clip.end = self.out_point;
            clip.info = media.info.clone();
            self.info(format!("Updated clip {}.", index + 1));
        }
    }

    fn select_clip(&mut self, index: usize) {
        let Some(clip) = self.clips.get(index) else {
            return;
        };
        let path = clip.path.clone();
        let range = (clip.start, clip.end);
        self.selected = Some(index);
        self.clear_armed = false;
        let same = self.media.as_ref().is_some_and(|m| m.path == path);
        if same {
            self.in_point = range.0;
            self.out_point = range.1;
            self.playhead = range.0;
            self.clamp_range();
            self.write_time_fields();
            self.wanted_time = Some(self.playhead);
            self.playing = false;
            self.play = None;
        } else {
            self.load_file(path, Some(range));
            self.selected = Some(index);
        }
    }

    fn move_selected(&mut self, delta: isize) {
        let Some(index) = self.selected else {
            return;
        };
        let next = index as isize + delta;
        if next < 0 || next as usize >= self.clips.len() {
            return;
        }
        let next = next as usize;
        self.clips.swap(index, next);
        self.selected = Some(next);
        self.clear_armed = false;
    }

    fn remove_selected(&mut self) {
        let Some(index) = self.selected else {
            return;
        };
        self.clips.remove(index);
        self.selected = if self.clips.is_empty() {
            None
        } else {
            Some(index.min(self.clips.len() - 1))
        };
        self.clear_armed = false;
    }

    fn export_current(&mut self) {
        let Some(media) = &self.media else {
            return;
        };
        let segment = media
            .info
            .segment(media.path.clone(), self.in_point, self.out_point);
        let suggested = suggested_name(&file_label(&media.path), "-cut");
        self.start_export(vec![segment], suggested);
    }

    fn export_list(&mut self) {
        let segments: Vec<_> = self
            .clips
            .iter()
            .map(|clip| clip.info.segment(clip.path.clone(), clip.start, clip.end))
            .collect();
        self.start_export(segments, "videocliptrimmer.mp4".into());
    }

    fn start_export(&mut self, segments: Vec<Segment>, suggested: String) {
        if self.job.is_some() {
            return;
        }
        if self.ffmpeg.is_err() {
            self.fail("Install ffmpeg before exporting.".into());
            return;
        }
        let Some(dest) = rfd::FileDialog::new()
            .set_title("Export video")
            .add_filter("MP4", &["mp4"])
            .set_file_name(&suggested)
            .save_file()
        else {
            return;
        };
        let dest = ensure_mp4(dest);
        let job_dest = dest.clone();
        let mode = self.mode;
        let settings = EncodeSettings {
            video_crf: self.video_crf,
            audio_bitrate: self.audio_bitrate.clone(),
            fps: self.fps.clone(),
            hw: self.hw,
        };
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = Arc::clone(&cancel);
        let tx_progress = tx.clone();
        let tx_log = tx.clone();
        thread::spawn(move || {
            let hooks = Hooks {
                progress: &|value| {
                    let _ = tx_progress.send(JobMsg::Progress(value));
                },
                log: &|line| {
                    let _ = tx_log.send(JobMsg::Log(line));
                },
                cancel: &cancel_worker,
            };
            let result = export(&segments, &dest, mode, &settings, &hooks);
            let _ = tx.send(JobMsg::Finished(result));
        });
        let message = format!("Exporting to {}…", job_dest.display());
        self.job = Some(Job {
            rx,
            cancel,
            progress: 0.0,
            dest: job_dest,
        });
        self.log.clear();
        self.info(message);
        self.clear_armed = false;
    }

    fn toggle_play(&mut self) {
        if self.media.is_none() {
            return;
        }
        if self.playing {
            self.playing = false;
            if let Some(film) = &mut self.film {
                film.set_paused(true);
            } else if let Some(play) = &self.play {
                play.set_paused(true);
            }
            return;
        }
        if self.playhead >= self.out_point - 0.05 {
            self.playhead = self.in_point;
            self.play = None;
            self.shown_time = None;
        }
        self.playing = true;
        self.wanted_time = None;
        if let Some(film) = &mut self.film {
            film.seek(self.playhead);
            film.set_paused(false);
        } else if let Some(play) = &self.play {
            play.set_paused(false);
        }
    }

    fn nudge(&mut self, delta: f64) {
        self.playing = false;
        self.play = None;
        let duration = self.duration();
        self.playhead = (self.playhead + delta).clamp(0.0, duration);
        if let Some(film) = &mut self.film {
            film.set_paused(true);
            film.seek(self.playhead);
            self.wanted_time = None;
        } else {
            self.wanted_time = Some(self.playhead);
        }
    }

    fn seek(&mut self, time: f64) {
        self.playing = false;
        self.play = None;
        self.playhead = time.clamp(0.0, self.duration());
        if let Some(film) = &mut self.film {
            film.set_paused(true);
            film.seek(self.playhead);
            self.wanted_time = None;
        } else {
            self.wanted_time = Some(self.playhead);
        }
    }

    fn poll_hw_detect(&mut self) {
        let message = {
            let Some(rx) = &self.hw_detect else {
                return;
            };
            rx.try_recv()
        };
        match message {
            Ok(hw) => {
                self.hw_detect = None;
                if self.hw_touched {
                    return;
                }
                self.crf[hw_index(self.hw)] = self.video_crf.clamp(0, 51);
                self.hw = hw;
                self.video_crf = self.crf[hw_index(hw)];
                self.hw_touched = true;
                self.persist();
                if hw != HwAccel::Off {
                    self.info(format!("Using {} hardware.", hw_label(hw)));
                }
                if self.media.is_some() {
                    self.open_film();
                }
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                self.hw_detect = None;
                if !self.hw_touched {
                    self.hw_touched = true;
                    self.crf[hw_index(self.hw)] = self.video_crf.clamp(0, 51);
                    self.persist();
                }
            }
        }
    }

    fn duration(&self) -> f64 {
        self.media.as_ref().map(|m| m.info.duration).unwrap_or(0.0)
    }

    fn clamp_range(&mut self) {
        let duration = self.duration();
        if duration <= 0.0 {
            return;
        }
        let min_len = MIN_RANGE.min(duration);
        self.in_point = self.in_point.clamp(0.0, (duration - min_len).max(0.0));
        self.out_point = self.out_point.clamp(min_len, duration);
        if self.out_point < self.in_point + min_len {
            self.out_point = (self.in_point + min_len).min(duration);
        }
        self.playhead = self.playhead.clamp(0.0, duration);
    }

    fn write_time_fields(&mut self) {
        self.in_text = format_time(self.in_point);
        self.out_text = format_time(self.out_point);
    }

    fn sync_time_fields(&mut self) {
        if !self.in_edit {
            self.in_text = format_time(self.in_point);
        }
        if !self.out_edit {
            self.out_text = format_time(self.out_point);
        }
    }

    fn info(&mut self, text: String) {
        self.notice = Notice::Info(text);
    }

    fn fail(&mut self, text: String) {
        let mut lines = text.lines();
        let first = lines.next().unwrap_or("Something went wrong.").to_string();
        let rest: Vec<&str> = lines.collect();
        if !rest.is_empty() {
            self.push_log(format!("{first}\n{}", rest.join("\n")));
        }
        self.notice = Notice::Error(first);
    }

    fn push_log(&mut self, line: String) {
        if !self.log.is_empty() {
            self.log.push('\n');
        }
        self.log.push_str(&line);
        if self.log.len() > 8000 {
            let mut idx = self.log.len() - 6000;
            while !self.log.is_char_boundary(idx) {
                idx += 1;
            }
            self.log = self.log[idx..].to_string();
        }
    }
}

impl Clip {
    fn whole(path: PathBuf, info: MediaInfo) -> Self {
        Self {
            label: file_label(&path),
            start: 0.0,
            end: info.duration,
            path,
            info,
        }
    }
}

fn timeline(
    ui: &mut egui::Ui,
    duration: f64,
    playhead: f64,
    in_point: f64,
    out_point: f64,
) -> Option<f64> {
    let width = ui.available_width();
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(width, 36.0), egui::Sense::click_and_drag());
    let painter = ui.painter();
    painter.rect_filled(rect, 5.0, rgb(0x10, 0x11, 0x14));
    if duration > 0.0 {
        let x_at =
            |time: f64| rect.left() + rect.width() * (time / duration).clamp(0.0, 1.0) as f32;
        let range = egui::Rect::from_min_max(
            egui::pos2(x_at(in_point), rect.top()),
            egui::pos2(x_at(out_point), rect.bottom()),
        );
        painter.rect_filled(range, 5.0, rgb(0x3d, 0x6a, 0x74));
        let edge = egui::Stroke::new(2.0_f32, rgb(0xd8, 0xd0, 0xc4));
        painter.vline(x_at(in_point), rect.y_range(), edge);
        painter.vline(x_at(out_point), rect.y_range(), edge);
        let x = x_at(playhead);
        painter.vline(
            x,
            rect.y_range(),
            egui::Stroke::new(2.0_f32, rgb(0xec, 0xea, 0xe4)),
        );
    }
    if (response.clicked() || response.dragged())
        && let Some(pos) = response.interact_pointer_pos()
        && duration > 0.0
        && rect.width() > 0.0
    {
        let fraction = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
        return Some(fraction as f64 * duration);
    }
    None
}

fn paint_fitted(ui: &mut egui::Ui, tex: &egui::TextureHandle) -> egui::Response {
    let available = ui.available_size();
    let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click());
    let image = tex.size_vec2();
    if image.x < 1.0 || image.y < 1.0 || rect.width() < 1.0 || rect.height() < 1.0 {
        return response;
    }
    let scale = (rect.width() / image.x).min(rect.height() / image.y);
    let size = image * scale;
    let image_rect = egui::Rect::from_center_size(rect.center(), size);
    let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
    ui.painter()
        .image(tex.id(), image_rect, uv, egui::Color32::WHITE);
    response.on_hover_cursor(egui::CursorIcon::PointingHand)
}

#[derive(Clone, Copy)]
enum Icon {
    Play,
    Pause,
    Back,
    Forward,
    Mark,
    Go,
    Plus,
    Update,
    Export,
    Up,
    Down,
    Remove,
    Clear,
    Folder,
    Settings,
}

fn icon_button(
    ui: &mut egui::Ui,
    enabled: bool,
    icon: Icon,
    label: &str,
    accent: bool,
) -> egui::Response {
    let mark = match icon {
        Icon::Play => "▶",
        Icon::Pause => "⏸",
        Icon::Back => "⏪",
        Icon::Forward => "⏩",
        Icon::Mark => "📍",
        Icon::Go => "➡",
        Icon::Plus => "➕",
        Icon::Update => "🔁",
        Icon::Export => "💾",
        Icon::Up => "⬆",
        Icon::Down => "⬇",
        Icon::Remove => "🗑",
        Icon::Clear => "🧹",
        Icon::Folder => "📂",
        Icon::Settings => "⚙",
    };
    let mut button = egui::Button::new(format!("{mark}  {label}"));
    if accent {
        button = button.fill(rgb(0x3d, 0x6a, 0x74));
    }
    ui.add_enabled(enabled, button)
}

fn muted(text: &str) -> egui::RichText {
    egui::RichText::new(text)
        .color(rgb(0xc4, 0xc0, 0xb6))
        .size(13.0)
}

fn rgb(r: u8, g: u8, b: u8) -> egui::Color32 {
    egui::Color32::from_rgb(r, g, b)
}

const FFMPEG_MISSING: &str = "No ffmpeg found on the system!";
const MPV_MISSING: &str = "No mpv library found on the system!";

fn pick_videos(title: &str) -> Option<Vec<PathBuf>> {
    rfd::FileDialog::new()
        .set_title(title)
        .add_filter("Video", VIDEO_EXT)
        .add_filter("All files", &["*"])
        .pick_files()
}

fn ffmpeg_label(version: &str) -> String {
    let rest = version
        .trim()
        .strip_prefix("ffmpeg version ")
        .unwrap_or(version.trim());
    let number = rest.split_whitespace().next().unwrap_or(rest);
    format!("ffmpeg {number}")
}

fn amd_button_label() -> &'static str {
    if cfg!(target_os = "macos") {
        "Apple"
    } else {
        "AMD"
    }
}

fn hw_label(hw: HwAccel) -> &'static str {
    match hw {
        HwAccel::Off => "CPU",
        HwAccel::Nvidia => "NVIDIA",
        HwAccel::Amd => amd_button_label(),
        HwAccel::Intel => "Intel",
    }
}

/// Remembers the hardware choice, a CRF for each encoder, the export mode, and the auto rates.
fn settings_path() -> Option<PathBuf> {
    let dir = if cfg!(windows) {
        PathBuf::from(std::env::var_os("APPDATA")?).join("VideoClipTrimmer")
    } else if cfg!(target_os = "macos") {
        PathBuf::from(std::env::var_os("HOME")?)
            .join("Library")
            .join("Application Support")
            .join("VideoClipTrimmer")
    } else {
        let home = PathBuf::from(std::env::var_os("HOME")?);
        match std::env::var_os("XDG_CONFIG_HOME") {
            Some(path) if !path.is_empty() => PathBuf::from(path),
            _ => home.join(".config"),
        }
        .join("videocliptrimmer")
    };
    Some(dir.join("settings.txt"))
}

const DEFAULT_CRF: i32 = 18;

struct SavedSettings {
    hw: Option<HwAccel>,
    crf: [i32; 4],
    mode: EncodeMode,
    audio: String,
    fps: String,
}

fn hw_index(hw: HwAccel) -> usize {
    match hw {
        HwAccel::Off => 0,
        HwAccel::Nvidia => 1,
        HwAccel::Amd => 2,
        HwAccel::Intel => 3,
    }
}

fn crf_key(hw: HwAccel) -> &'static str {
    match hw {
        HwAccel::Off => "crf_off",
        HwAccel::Nvidia => "crf_nvidia",
        HwAccel::Amd => "crf_amd",
        HwAccel::Intel => "crf_intel",
    }
}

fn load_settings() -> SavedSettings {
    let text = settings_path()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .unwrap_or_default();
    parse_settings(&text)
}

fn parse_settings(text: &str) -> SavedSettings {
    let mut saved = SavedSettings {
        hw: None,
        crf: [DEFAULT_CRF; 4],
        mode: EncodeMode::FastCopy,
        audio: "auto".into(),
        fps: "auto".into(),
    };
    for line in text.lines() {
        let Some((key, value)) = line.trim().split_once('=') else {
            continue;
        };
        let value = value.trim();
        if key == "hw" {
            saved.hw = parse_hw_name(value);
            continue;
        }
        if key == "mode" {
            if let Some(mode) = parse_mode_name(value) {
                saved.mode = mode;
            }
            continue;
        }
        if key == "audio" {
            if !value.is_empty() {
                saved.audio = value.to_string();
            }
            continue;
        }
        if key == "fps" {
            if !value.is_empty() {
                saved.fps = value.to_string();
            }
            continue;
        }
        let Some(hw) = parse_crf_key(key) else {
            continue;
        };
        if let Ok(crf) = value.parse::<i32>() {
            saved.crf[hw_index(hw)] = crf.clamp(0, 51);
        }
    }
    saved
}

fn parse_crf_key(key: &str) -> Option<HwAccel> {
    match key {
        "crf_off" => Some(HwAccel::Off),
        "crf_nvidia" => Some(HwAccel::Nvidia),
        "crf_amd" => Some(HwAccel::Amd),
        "crf_intel" => Some(HwAccel::Intel),
        _ => None,
    }
}

fn parse_mode_name(text: &str) -> Option<EncodeMode> {
    match text.trim().to_ascii_lowercase().as_str() {
        "fast" | "fastcopy" | "fast-copy" => Some(EncodeMode::FastCopy),
        "reencode" | "re-encode" | "compatible" => Some(EncodeMode::Compatible),
        _ => None,
    }
}

fn mode_name(mode: EncodeMode) -> &'static str {
    match mode {
        EncodeMode::FastCopy => "fast",
        EncodeMode::Compatible => "reencode",
    }
}

fn save_settings(hw: HwAccel, crf: [i32; 4], mode: EncodeMode, audio: &str, fps: &str) {
    let Some(path) = settings_path() else {
        return;
    };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut text = format!(
        "hw={}\nmode={}\naudio={}\nfps={}\n",
        hw_name(hw),
        mode_name(mode),
        audio.trim(),
        fps.trim()
    );
    for choice in [HwAccel::Off, HwAccel::Nvidia, HwAccel::Amd, HwAccel::Intel] {
        text.push_str(&format!("{}={}\n", crf_key(choice), crf[hw_index(choice)]));
    }
    let _ = std::fs::write(path, text);
}

fn file_label(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

fn summary(info: &MediaInfo) -> String {
    let fps = if (info.fps - info.fps.round()).abs() < 0.05 {
        format!("{}", info.fps.round() as i64)
    } else {
        format!("{:.2}", info.fps)
    };
    let audio = if info.has_audio {
        info.audio_codec.as_str()
    } else {
        "no audio"
    };
    format!(
        "{}×{}   ·   {fps} fps   ·   {}   ·   {audio}   ·   {}",
        info.width,
        info.height,
        info.video_codec,
        format_time(info.duration)
    )
}

fn suggested_name(label: &str, suffix: &str) -> String {
    let stem = Path::new(label)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "clip".into());
    format!("{stem}{suffix}.mp4")
}

fn ensure_mp4(path: PathBuf) -> PathBuf {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("mp4") => path,
        _ => path.with_extension("mp4"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_settings_file_still_loads() {
        let saved = parse_settings("hw=nvidia\n");
        assert_eq!(saved.hw, Some(HwAccel::Nvidia));
        assert_eq!(saved.crf, [DEFAULT_CRF; 4]);
        assert_eq!(saved.mode, EncodeMode::FastCopy);
        assert_eq!(saved.audio, "auto");
        assert_eq!(saved.fps, "auto");
    }

    #[test]
    fn reads_saved_audio_and_frame_rate() {
        let saved = parse_settings("audio=256k\nfps=60\n");
        assert_eq!(saved.audio, "256k");
        assert_eq!(saved.fps, "60");
    }

    #[test]
    fn reads_the_saved_export_mode() {
        let saved = parse_settings("mode=reencode\n");
        assert_eq!(saved.mode, EncodeMode::Compatible);
        let saved = parse_settings("mode=fast\n");
        assert_eq!(saved.mode, EncodeMode::FastCopy);
    }

    #[test]
    fn reads_a_crf_for_each_encoder() {
        let saved =
            parse_settings("hw=intel\ncrf_off=18\ncrf_nvidia=20\ncrf_amd=16\ncrf_intel=23\n");
        assert_eq!(saved.hw, Some(HwAccel::Intel));
        assert_eq!(saved.crf[hw_index(HwAccel::Off)], 18);
        assert_eq!(saved.crf[hw_index(HwAccel::Nvidia)], 20);
        assert_eq!(saved.crf[hw_index(HwAccel::Amd)], 16);
        assert_eq!(saved.crf[hw_index(HwAccel::Intel)], 23);
    }

    #[test]
    fn clamps_a_crf_and_ignores_junk() {
        let saved = parse_settings("crf_intel=nope\ncrf_amd=80\ncrf_nvidia=-4\n");
        assert_eq!(saved.crf[hw_index(HwAccel::Intel)], DEFAULT_CRF);
        assert_eq!(saved.crf[hw_index(HwAccel::Amd)], 51);
        assert_eq!(saved.crf[hw_index(HwAccel::Nvidia)], 0);
    }
}
