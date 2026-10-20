// Release builds are a GUI program. No console window, and no console output.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod drop_files;
mod ffmpeg;
mod mpv;
mod timeutil;
mod ui;
mod video_view;

use miniquad as mq;

pub const VERSION: &str = concat!("v", env!("CARGO_PKG_VERSION"));

fn main() {
    let platform = mq::conf::Platform {
        swap_interval: Some(1),
        // macOS paces frames with a display link, and only while this is false.
        // Other systems block until the next draw, then the swap waits for vsync.
        blocking_event_loop: !cfg!(target_os = "macos"),
        linux_wm_class: "videocliptrimmer",
        ..Default::default()
    };

    let conf = mq::conf::Conf {
        window_title: "VideoClipTrimmer".to_string(),
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
        let obscured = window_is_obscured();
        self.app.note_obscured(obscured);
        if obscured {
            return;
        }
        pace_to_display();
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
            mq::window::schedule_update();
        }
    }

    fn resize_event(&mut self, _width: f32, _height: f32) {
        mq::window::schedule_update();
    }

    fn window_minimized_event(&mut self) {
        mq::window::schedule_update();
    }

    fn window_restored_event(&mut self) {
        mq::window::schedule_update();
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

/// True when this window is minimized, cloaked, or otherwise not on screen.
fn window_is_obscured() -> bool {
    #[cfg(windows)]
    {
        win_window_is_obscured()
    }
    #[cfg(not(windows))]
    {
        false
    }
}

#[cfg(windows)]
#[allow(clippy::upper_case_acronyms)]
fn win_window_is_obscured() -> bool {
    use std::ffi::c_void;

    type HWND = *mut c_void;
    const DWMWA_CLOAKED: u32 = 14;

    #[link(name = "user32")]
    unsafe extern "system" {
        fn FindWindowW(class: *const u16, title: *const u16) -> HWND;
        fn IsIconic(hwnd: HWND) -> i32;
        fn IsWindowVisible(hwnd: HWND) -> i32;
    }
    #[link(name = "dwmapi")]
    unsafe extern "system" {
        fn DwmGetWindowAttribute(hwnd: HWND, attr: u32, value: *mut u32, size: u32) -> i32;
    }

    unsafe {
        let class: Vec<u16> = "MINIQUADAPP".encode_utf16().chain(Some(0)).collect();
        let hwnd = FindWindowW(class.as_ptr(), std::ptr::null());
        if hwnd.is_null() {
            return false;
        }
        if IsIconic(hwnd) != 0 || IsWindowVisible(hwnd) == 0 {
            return true;
        }
        let mut cloaked = 0u32;
        let ok = DwmGetWindowAttribute(
            hwnd,
            DWMWA_CLOAKED,
            &mut cloaked,
            std::mem::size_of::<u32>() as u32,
        );
        ok == 0 && cloaked != 0
    }
}

/// Holds the next frame back when it would be faster than the monitor.
/// Vsync already waits; this only fills the gap if the present returned early.
fn pace_to_display() {
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    let now = Instant::now();
    let mut last = LAST.lock().unwrap_or_else(|err| err.into_inner());
    let Some(previous) = *last else {
        *last = Some(now);
        return;
    };
    let Some(hz) = display_hz() else {
        *last = Some(now);
        return;
    };
    let frame = Duration::from_secs_f64(1.0 / hz);
    let elapsed = now.saturating_duration_since(previous);
    if elapsed + Duration::from_millis(2) >= frame {
        *last = Some(now);
        return;
    }
    drop(last);
    std::thread::sleep(frame - elapsed);
    *LAST.lock().unwrap_or_else(|err| err.into_inner()) = Some(Instant::now());
}

/// Refresh rate of the monitor the window is on, in Hz.
/// `None` when the system does not say. Vsync still applies.
fn display_hz() -> Option<f64> {
    #[cfg(windows)]
    {
        win_display_hz()
    }
    #[cfg(not(windows))]
    {
        None
    }
}

#[cfg(windows)]
fn win_display_hz() -> Option<f64> {
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static CACHED: AtomicU32 = AtomicU32::new(0);
    static CHECKED_MS: AtomicU64 = AtomicU64::new(0);

    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|time| time.as_millis() as u64)
        .unwrap_or(0);
    let checked = CHECKED_MS.load(Ordering::Relaxed);
    if now_ms.saturating_sub(checked) < 500 {
        let hz = CACHED.load(Ordering::Relaxed);
        return (hz >= 24).then_some(f64::from(hz));
    }
    CHECKED_MS.store(now_ms, Ordering::Relaxed);
    let hz = query_window_hz().unwrap_or(0);
    CACHED.store(hz, Ordering::Relaxed);
    (hz >= 24).then_some(f64::from(hz))
}

#[cfg(windows)]
#[allow(clippy::upper_case_acronyms)]
fn query_window_hz() -> Option<u32> {
    use std::ffi::c_void;

    type HWND = *mut c_void;
    type HDC = *mut c_void;
    type HMONITOR = *mut c_void;

    const MONITOR_DEFAULTTONEAREST: u32 = 2;
    const ENUM_CURRENT_SETTINGS: u32 = 0xFFFF_FFFF;
    const VREFRESH: i32 = 116;

    #[repr(C)]
    struct MonitorInfoEx {
        size: u32,
        monitor: [i32; 4],
        work: [i32; 4],
        flags: u32,
        device: [u16; 32],
    }

    /// `DEVMODEW` is 220 bytes. The refresh rate sits at byte 184.
    #[repr(C, align(8))]
    struct DevMode([u8; 220]);

    #[link(name = "user32")]
    unsafe extern "system" {
        fn FindWindowW(class: *const u16, title: *const u16) -> HWND;
        fn GetDC(hwnd: HWND) -> HDC;
        fn ReleaseDC(hwnd: HWND, dc: HDC) -> i32;
        fn GetDeviceCaps(dc: HDC, index: i32) -> i32;
        fn MonitorFromWindow(hwnd: HWND, flags: u32) -> HMONITOR;
        fn GetMonitorInfoW(monitor: HMONITOR, info: *mut MonitorInfoEx) -> i32;
        fn EnumDisplaySettingsW(device: *const u16, mode: u32, devmode: *mut DevMode) -> i32;
    }

    unsafe {
        let class: Vec<u16> = "MINIQUADAPP".encode_utf16().chain(Some(0)).collect();
        let hwnd = FindWindowW(class.as_ptr(), std::ptr::null());
        if hwnd.is_null() {
            return None;
        }
        let monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
        if !monitor.is_null() {
            let mut info = std::mem::zeroed::<MonitorInfoEx>();
            info.size = std::mem::size_of::<MonitorInfoEx>() as u32;
            if GetMonitorInfoW(monitor, &mut info) != 0 {
                let mut mode = std::mem::zeroed::<DevMode>();
                mode.0[68..70].copy_from_slice(&220u16.to_ne_bytes());
                if EnumDisplaySettingsW(info.device.as_ptr(), ENUM_CURRENT_SETTINGS, &mut mode) != 0
                {
                    let hz = u32::from_ne_bytes(mode.0[184..188].try_into().unwrap());
                    if (24..=500).contains(&hz) {
                        return Some(hz);
                    }
                }
            }
        }
        let dc = GetDC(hwnd);
        if dc.is_null() {
            return None;
        }
        let hz = GetDeviceCaps(dc, VREFRESH);
        ReleaseDC(hwnd, dc);
        (24..=500).contains(&hz).then_some(hz as u32)
    }
}

fn app_icon() -> mq::conf::Icon {
    mq::conf::Icon {
        small: raster::<{ 16 * 16 * 4 }>(16),
        medium: raster::<{ 32 * 32 * 4 }>(32),
        big: raster::<{ 64 * 64 * 4 }>(64),
    }
}

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
