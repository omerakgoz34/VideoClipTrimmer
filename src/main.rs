//! VideoClipTrimmer — a small desktop window for cutting and joining video.
//!
//! The window is drawn with egui. The pixels come from miniquad (a real OS
//! window, OpenGL, no browser). Playback is mpv when it is installed.
//! ffmpeg does the cutting and joining.

// Release builds are a GUI program. No console window, and no console output.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod drop_files;
mod ffmpeg;
mod mpv;
mod timeutil;
mod ui;

use miniquad as mq;

/// From the `version` field in Cargo.toml, set when the app is built.
pub const VERSION: &str = concat!("v", env!("CARGO_PKG_VERSION"));

fn main() {
    let platform = mq::conf::Platform {
        blocking_event_loop: false,
        linux_wm_class: "videocliptrimmer",
        ..Default::default()
    };

    let conf = mq::conf::Conf {
        window_title: format!("VideoClipTrimmer {VERSION}"),
        window_width: 1125,
        window_height: 720,
        high_dpi: true,
        fullscreen: false,
        window_resizable: true,
        icon: Some(app_icon()),
        platform,
        ..Default::default()
    };

    mq::start(conf, || Box::new(Stage::new()));
}

struct Stage {
    egui_mq: egui_miniquad::EguiMq,
    mq_ctx: Box<dyn mq::RenderingBackend>,
    app: ui::App,
}

impl Stage {
    fn new() -> Self {
        let mut mq_ctx = mq::window::new_rendering_backend();
        let egui_mq = egui_miniquad::EguiMq::new(&mut *mq_ctx);
        Self {
            egui_mq,
            mq_ctx,
            app: ui::App::new(),
        }
    }

    fn dropped_paths() -> Vec<std::path::PathBuf> {
        let count = mq::window::dropped_file_count();
        let mut paths = Vec::with_capacity(count);
        for index in 0..count {
            if let Some(path) = mq::window::dropped_file_path(index) {
                paths.push(path);
            }
        }
        paths
    }
}

impl mq::EventHandler for Stage {
    fn update(&mut self) {}

    fn draw(&mut self) {
        drop_files::install();
        let dropped = drop_files::take();
        if !dropped.is_empty() {
            self.app.open_paths(dropped);
        }

        let Stage {
            egui_mq,
            mq_ctx,
            app,
        } = self;
        mq_ctx.begin_default_pass(mq::PassAction::clear_color(0.086, 0.094, 0.110, 1.0));
        mq_ctx.end_render_pass();

        egui_mq.run(&mut **mq_ctx, |_backend, ctx| {
            app.ui(ctx);
        });
        egui_mq.draw(&mut **mq_ctx);
        mq_ctx.commit_frame();

        if app.wants_redraw() {
            let wait = app.redraw_wait();
            if !wait.is_zero() {
                std::thread::sleep(wait);
            }
            mq::window::schedule_update();
        }
    }

    fn mouse_motion_event(&mut self, x: f32, y: f32) {
        self.egui_mq.mouse_motion_event(x, y);
        mq::window::schedule_update();
    }

    fn mouse_wheel_event(&mut self, dx: f32, dy: f32) {
        self.egui_mq.mouse_wheel_event(dx, dy);
        mq::window::schedule_update();
    }

    fn mouse_button_down_event(&mut self, button: mq::MouseButton, x: f32, y: f32) {
        self.egui_mq.mouse_button_down_event(button, x, y);
        mq::window::schedule_update();
    }

    fn mouse_button_up_event(&mut self, button: mq::MouseButton, x: f32, y: f32) {
        self.egui_mq.mouse_button_up_event(button, x, y);
        mq::window::schedule_update();
    }

    fn char_event(&mut self, character: char, _mods: mq::KeyMods, _repeat: bool) {
        self.egui_mq.char_event(character);
        mq::window::schedule_update();
    }

    fn key_down_event(&mut self, keycode: mq::KeyCode, mods: mq::KeyMods, _repeat: bool) {
        self.egui_mq.key_down_event(keycode, mods);
        mq::window::schedule_update();
    }

    fn key_up_event(&mut self, keycode: mq::KeyCode, mods: mq::KeyMods) {
        self.egui_mq.key_up_event(keycode, mods);
        mq::window::schedule_update();
    }

    fn files_dropped_event(&mut self) {
        let paths = Self::dropped_paths();
        if !paths.is_empty() {
            self.app.open_paths(paths);
        }
        mq::window::schedule_update();
    }
}

fn app_icon() -> mq::conf::Icon {
    mq::conf::Icon {
        small: raster::<{ 16 * 16 * 4 }>(16),
        medium: raster::<{ 32 * 32 * 4 }>(32),
        big: raster::<{ 64 * 64 * 4 }>(64),
    }
}

/// A dark tile with a steel frame and an orange cut down the middle.
fn raster<const N: usize>(size: usize) -> [u8; N] {
    assert_eq!(N, size * size * 4);
    let mut pixels = [0u8; N];
    let margin = (size / 16).max(1);
    let left = size * 2 / 10;
    let right = size * 8 / 10;
    let top = size * 3 / 10;
    let bottom = size * 7 / 10;
    let mid = size / 2;
    let gap = (size / 16).max(1);
    for y in 0..size {
        for x in 0..size {
            let i = (y * size + x) * 4;
            let (r, g, b) = if x < margin || y < margin || x + margin >= size || y + margin >= size
            {
                (0x6f, 0x9e, 0xab)
            } else if x >= left
                && x < right
                && y >= top
                && y < bottom
                && (x as i32 - mid as i32).abs() <= gap as i32
            {
                (0xe0, 0x7a, 0x5f)
            } else if x >= left && x < right && y >= top && y < bottom {
                (0x2a, 0x2e, 0x34)
            } else {
                (0x16, 0x18, 0x1c)
            };
            pixels[i] = r;
            pixels[i + 1] = g;
            pixels[i + 2] = b;
            pixels[i + 3] = 255;
        }
    }
    pixels
}
