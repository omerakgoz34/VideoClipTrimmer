//! Real playback through libmpv, loaded at run time.
//!
//! mpv draws the video itself, in a child of this window. The app's OpenGL
//! context is left alone, so the interface is not flipped or overwritten.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::Path;
use std::ptr;

use crate::video_view::VideoView;

#[repr(C)]
struct MpvEvent {
    event_id: c_int,
    error: c_int,
    reply_userdata: u64,
    data: *mut c_void,
}

const FILE_LOADED: c_int = 8;
const FORMAT_INT64: c_int = 4;
const FORMAT_DOUBLE: c_int = 5;

struct Api {
    _lib: libloading::Library,
    create: unsafe extern "C" fn() -> *mut c_void,
    initialize: unsafe extern "C" fn(*mut c_void) -> c_int,
    set_option: unsafe extern "C" fn(*mut c_void, *const c_char, c_int, *mut c_void) -> c_int,
    set_option_string: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> c_int,
    command: unsafe extern "C" fn(*mut c_void, *const *const c_char) -> c_int,
    set_property_string: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> c_int,
    get_property: unsafe extern "C" fn(*mut c_void, *const c_char, c_int, *mut c_void) -> c_int,
    wait_event: unsafe extern "C" fn(*mut c_void, f64) -> *mut MpvEvent,
    terminate: unsafe extern "C" fn(*mut c_void),
    error_string: unsafe extern "C" fn(c_int) -> *const c_char,
}

pub struct Player {
    api: Api,
    handle: *mut c_void,
    view: VideoView,
    hw: bool,
    waiting_frame: bool,
    pending_seek: Option<f64>,
    shown: bool,
    placed: Option<(i32, i32, i32, i32)>,
}

impl Player {
    pub fn open(path: &Path, hw: bool) -> Result<Self, String> {
        let api = load_api()?;
        let view = VideoView::attach()?;
        let handle = unsafe { (api.create)() };
        if handle.is_null() {
            view.destroy();
            return Err("mpv could not start.".into());
        }
        let mut player = Self {
            api,
            handle,
            view,
            hw,
            waiting_frame: true,
            pending_seek: Some(0.0),
            shown: false,
            placed: None,
        };
        player.configure()?;
        player.load(path)?;
        Ok(player)
    }

    pub fn hw_on(&self) -> bool {
        self.hw
    }

    pub fn embedded(&self) -> bool {
        true
    }

    pub fn waiting(&self) -> bool {
        self.waiting_frame
    }

    pub fn set_paused(&mut self, paused: bool) {
        let value = if paused { "yes" } else { "no" };
        self.set_string("pause", value);
    }

    pub fn seek(&mut self, seconds: f64) {
        let seconds = if seconds.is_finite() {
            seconds.max(0.0)
        } else {
            0.0
        };
        self.pending_seek = Some(seconds);
        self.waiting_frame = true;
        self.seek_now(seconds);
    }

    /// Move the video to this rectangle, in window pixels.
    pub fn place(&mut self, x: i32, y: i32, w: i32, h: i32) {
        let next = (x, y, w.max(1), h.max(1));
        if self.shown && self.placed == Some(next) {
            return;
        }
        self.view.place(next.0, next.1, next.2, next.3);
        self.placed = Some(next);
        self.shown = true;
    }

    pub fn hide(&mut self) {
        if !self.shown {
            return;
        }
        self.view.hide();
        self.shown = false;
        self.placed = None;
    }

    /// Drain mpv and read its clock. The picture updates on its own.
    pub fn poll(&mut self) -> Option<f64> {
        self.drain();
        let time = self.time();
        if time.is_some() {
            self.waiting_frame = false;
        }
        time
    }

    pub fn time(&self) -> Option<f64> {
        let name = CString::new("time-pos").ok()?;
        let mut value = 0.0f64;
        let rc = unsafe {
            (self.api.get_property)(
                self.handle,
                name.as_ptr(),
                FORMAT_DOUBLE,
                (&mut value as *mut f64).cast(),
            )
        };
        (rc >= 0 && value.is_finite()).then_some(value)
    }

    fn configure(&mut self) -> Result<(), String> {
        let mut wid = self.view.id();
        self.set_int("wid", &mut wid)?;
        // Preview only. The fast profile uses bilinear scaling and skips the
        // heavy shaders. Older mpv builds may not have the profile name.
        if self.option("profile", "fast").is_err() {
            let _ = self.option("scale", "bilinear");
            let _ = self.option("dscale", "bilinear");
            let _ = self.option("cscale", "bilinear");
            let _ = self.option("dither", "no");
            let _ = self.option("correct-downscaling", "no");
            let _ = self.option("linear-downscaling", "no");
            let _ = self.option("sigmoid-upscaling", "no");
        }
        // Keep the preview cheap. One blit-style frame per video frame, synced
        // to audio, instead of extra shader passes or uncapped presents.
        let _ = self.option("gpu-dumb-mode", "yes");
        let _ = self.option("deband", "no");
        let _ = self.option("interpolation", "no");
        let _ = self.option("video-sync", "audio");
        let _ = self.option("framedrop", "vo");
        let _ = self.option("d3d11-sync-interval", "1");
        // Flip mode leaves a green line along the bottom of an embedded window.
        let _ = self.option("d3d11-flip", "no");
        let _ = self.option("video-latency-hacks", "yes");
        // mpv draws with its own Direct3D device. It must not use this
        // window's OpenGL context, or the whole interface breaks.
        self.option("vo", "gpu")?;
        let _ = self.option("gpu-context", "d3d11");
        self.option("keep-open", "yes")?;
        self.option("idle", "yes")?;
        self.option("pause", "yes")?;
        self.option("force-window", "no")?;
        self.option("osc", "no")?;
        self.option("osd-level", "0")?;
        self.option("input-default-bindings", "no")?;
        let _ = self.option("input-builtin-bindings", "no");
        self.option("input-vo-keyboard", "no")?;
        self.option("ytdl", "no")?;
        self.option("hr-seek", "yes")?;
        self.option("terminal", "no")?;
        self.option("keepaspect", "yes")?;
        let hwdec = if self.hw { "auto" } else { "no" };
        self.option("hwdec", hwdec)?;
        if self.hw {
            // Use the decoder texture directly instead of copying every frame.
            let _ = self.option("d3d11va-zero-copy", "yes");
        }
        let rc = unsafe { (self.api.initialize)(self.handle) };
        if rc < 0 {
            return Err(self.explain(rc));
        }
        self.view.hook_clicks();
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<(), String> {
        let text = path.to_string_lossy();
        let path =
            CString::new(text.as_ref()).map_err(|_| "The file path is not valid.".to_string())?;
        let loadfile = CString::new("loadfile").unwrap();
        let mode = CString::new("replace").unwrap();
        let args = [loadfile.as_ptr(), path.as_ptr(), mode.as_ptr(), ptr::null()];
        let rc = unsafe { (self.api.command)(self.handle, args.as_ptr()) };
        if rc < 0 {
            return Err(self.explain(rc));
        }
        Ok(())
    }

    fn seek_now(&mut self, seconds: f64) {
        let seek = CString::new("seek").unwrap();
        let time = CString::new(format!("{seconds:.3}")).unwrap();
        let flags = CString::new("absolute+exact").unwrap();
        let args = [seek.as_ptr(), time.as_ptr(), flags.as_ptr(), ptr::null()];
        unsafe {
            (self.api.command)(self.handle, args.as_ptr());
        }
    }

    fn drain(&mut self) {
        if self.handle.is_null() {
            return;
        }
        for _ in 0..32 {
            let event = unsafe { (self.api.wait_event)(self.handle, 0.0) };
            if event.is_null() {
                break;
            }
            let id = unsafe { (*event).event_id };
            if id == 0 {
                break;
            }
            if id == FILE_LOADED
                && let Some(seconds) = self.pending_seek.take()
            {
                self.seek_now(seconds);
            }
        }
    }

    fn set_int(&self, name: &str, value: &mut i64) -> Result<(), String> {
        let name = CString::new(name).unwrap();
        let rc = unsafe {
            (self.api.set_option)(
                self.handle,
                name.as_ptr(),
                FORMAT_INT64,
                (value as *mut i64).cast(),
            )
        };
        if rc < 0 {
            Err(self.explain(rc))
        } else {
            Ok(())
        }
    }

    fn option(&self, name: &str, value: &str) -> Result<(), String> {
        let name = CString::new(name).unwrap();
        let value = CString::new(value).unwrap();
        let rc =
            unsafe { (self.api.set_option_string)(self.handle, name.as_ptr(), value.as_ptr()) };
        if rc < 0 {
            Err(self.explain(rc))
        } else {
            Ok(())
        }
    }

    fn set_string(&self, name: &str, value: &str) {
        let Ok(name) = CString::new(name) else {
            return;
        };
        let Ok(value) = CString::new(value) else {
            return;
        };
        unsafe {
            (self.api.set_property_string)(self.handle, name.as_ptr(), value.as_ptr());
        }
    }

    fn explain(&self, code: c_int) -> String {
        let text = unsafe { (self.api.error_string)(code) };
        let detail = if text.is_null() {
            "unknown error".to_string()
        } else {
            unsafe { CStr::from_ptr(text) }
                .to_string_lossy()
                .into_owned()
        };
        format!("mpv: {detail}")
    }

    fn shutdown(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                (self.api.terminate)(self.handle);
                self.handle = ptr::null_mut();
            }
        }
        self.view.destroy();
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub fn library_present() -> bool {
    open_library().is_ok()
}

fn load_api() -> Result<Api, String> {
    let lib = open_library()?;
    Ok(Api {
        create: load_sym(&lib, b"mpv_create\0")?,
        initialize: load_sym(&lib, b"mpv_initialize\0")?,
        set_option: load_sym(&lib, b"mpv_set_option\0")?,
        set_option_string: load_sym(&lib, b"mpv_set_option_string\0")?,
        command: load_sym(&lib, b"mpv_command\0")?,
        set_property_string: load_sym(&lib, b"mpv_set_property_string\0")?,
        get_property: load_sym(&lib, b"mpv_get_property\0")?,
        wait_event: load_sym(&lib, b"mpv_wait_event\0")?,
        terminate: load_sym(&lib, b"mpv_terminate_destroy\0")?,
        error_string: load_sym(&lib, b"mpv_error_string\0")?,
        _lib: lib,
    })
}

fn load_sym<T: Copy>(lib: &libloading::Library, name: &[u8]) -> Result<T, String> {
    let symbol: libloading::Symbol<T> = unsafe { lib.get(name) }
        .map_err(|_| "This mpv library is too old or incomplete.".to_string())?;
    Ok(*symbol)
}

fn open_library() -> Result<libloading::Library, String> {
    let mut tried = Vec::new();
    for path in library_names() {
        tried.push(path.display().to_string());
        if let Ok(lib) = unsafe { libloading::Library::new(&path) } {
            return Ok(lib);
        }
    }
    Err(
        "mpv is not installed. Playback needs the mpv library (libmpv) next to the app or on PATH."
            .into(),
    )
}

fn library_names() -> Vec<std::path::PathBuf> {
    let names: &[&str] = if cfg!(windows) {
        &["libmpv-2.dll", "mpv-2.dll", "mpv-1.dll"]
    } else if cfg!(target_os = "macos") {
        &["libmpv.2.dylib", "libmpv.dylib"]
    } else {
        &["libmpv.so.2", "libmpv.so.1", "libmpv.so"]
    };
    let mut paths = Vec::new();
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        for name in names {
            paths.push(dir.join(name));
        }
    }
    if let Ok(path) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path) {
            for name in names {
                paths.push(dir.join(name));
            }
        }
    }
    for name in names {
        paths.push(std::path::PathBuf::from(name));
    }
    paths
}
