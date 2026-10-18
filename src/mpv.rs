//! Real playback through libmpv, loaded at run time.
//!
//! mpv owns the clock and the audio. We only copy the frame it has already
//! decided to show, and read `time-pos` for the playhead. That is a player,
//! not a timer glued onto decoded frames.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::Path;
use std::ptr;

#[repr(C)]
struct RenderParam {
    kind: c_int,
    data: *mut c_void,
}

#[repr(C)]
struct MpvEvent {
    event_id: c_int,
    error: c_int,
    reply_userdata: u64,
    data: *mut c_void,
}

const API_TYPE: c_int = 1;
const ADVANCED_CONTROL: c_int = 10;
const SW_SIZE: c_int = 17;
const SW_FORMAT: c_int = 18;
const SW_STRIDE: c_int = 19;
const SW_POINTER: c_int = 20;
const UPDATE_FRAME: u64 = 1;
const FILE_LOADED: c_int = 8;
const FORMAT_DOUBLE: c_int = 5;

struct Api {
    _lib: libloading::Library,
    create: unsafe extern "C" fn() -> *mut c_void,
    initialize: unsafe extern "C" fn(*mut c_void) -> c_int,
    set_option_string: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> c_int,
    command: unsafe extern "C" fn(*mut c_void, *const *const c_char) -> c_int,
    set_property_string: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> c_int,
    get_property: unsafe extern "C" fn(*mut c_void, *const c_char, c_int, *mut c_void) -> c_int,
    wait_event: unsafe extern "C" fn(*mut c_void, f64) -> *mut MpvEvent,
    terminate: unsafe extern "C" fn(*mut c_void),
    error_string: unsafe extern "C" fn(c_int) -> *const c_char,
    render_create: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *mut RenderParam) -> c_int,
    render_update: unsafe extern "C" fn(*mut c_void) -> u64,
    render: unsafe extern "C" fn(*mut c_void, *mut RenderParam) -> c_int,
    render_free: unsafe extern "C" fn(*mut c_void),
}

pub struct Player {
    api: Api,
    handle: *mut c_void,
    render: *mut c_void,
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    scratch: Vec<u8>,
    stride: usize,
    hw: bool,
    waiting_frame: bool,
    pending_seek: Option<f64>,
}

impl Player {
    pub fn open(path: &Path, width: u32, height: u32, hw: bool) -> Result<Self, String> {
        let api = load_api()?;
        let width = width.max(2);
        let height = height.max(2);
        let handle = unsafe { (api.create)() };
        if handle.is_null() {
            return Err("mpv could not start.".into());
        }
        let mut player = Self {
            api,
            handle,
            render: ptr::null_mut(),
            width,
            height,
            pixels: vec![0; width as usize * height as usize * 4],
            scratch: Vec::new(),
            stride: 0,
            hw,
            waiting_frame: true,
            pending_seek: Some(0.0),
        };
        player.configure()?;
        player.load(path)?;
        Ok(player)
    }

    pub fn hw_on(&self) -> bool {
        self.hw
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
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

    /// A newly displayed frame, already in packed RGBA, plus mpv's clock.
    pub fn poll(&mut self) -> (Option<Vec<u8>>, Option<f64>) {
        self.drain();
        if !self.render.is_null() {
            let flags = unsafe { (self.api.render_update)(self.render) };
            if flags & UPDATE_FRAME != 0 {
                self.draw_frame();
                self.waiting_frame = false;
                let pixels = self.pixels.clone();
                return (Some(pixels), self.time());
            }
        }
        (None, self.time())
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
        self.option("vo", "libmpv")?;
        self.option("keep-open", "yes")?;
        self.option("idle", "yes")?;
        self.option("pause", "yes")?;
        self.option("force-window", "no")?;
        self.option("osc", "no")?;
        self.option("osd-level", "0")?;
        self.option("input-default-bindings", "no")?;
        self.option("input-vo-keyboard", "no")?;
        self.option("ytdl", "no")?;
        self.option("hr-seek", "yes")?;
        self.option("terminal", "no")?;
        let hwdec = if self.hw { "auto-safe" } else { "no" };
        self.option("hwdec", hwdec)?;
        let rc = unsafe { (self.api.initialize)(self.handle) };
        if rc < 0 {
            return Err(self.explain(rc));
        }
        let mut render = ptr::null_mut();
        let mut advanced: c_int = 1;
        let api_name = c"sw";
        let mut params = [
            RenderParam {
                kind: API_TYPE,
                data: api_name.as_ptr() as *mut c_void,
            },
            RenderParam {
                kind: ADVANCED_CONTROL,
                data: (&mut advanced as *mut c_int).cast(),
            },
            RenderParam {
                kind: 0,
                data: ptr::null_mut(),
            },
        ];
        let rc = unsafe { (self.api.render_create)(&mut render, self.handle, params.as_mut_ptr()) };
        if rc < 0 || render.is_null() {
            return Err(self.explain(rc));
        }
        self.render = render;
        let stride = (self.width as usize * 4).div_ceil(64) * 64;
        self.stride = stride;
        self.scratch = vec![0; stride * self.height as usize];
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

    fn draw_frame(&mut self) {
        let mut size = [self.width as c_int, self.height as c_int];
        let mut stride = self.stride;
        let format = c"rgb0";
        let mut params = [
            RenderParam {
                kind: SW_SIZE,
                data: size.as_mut_ptr().cast(),
            },
            RenderParam {
                kind: SW_FORMAT,
                data: format.as_ptr() as *mut c_void,
            },
            RenderParam {
                kind: SW_STRIDE,
                data: (&mut stride as *mut usize).cast(),
            },
            RenderParam {
                kind: SW_POINTER,
                data: self.scratch.as_mut_ptr().cast(),
            },
            RenderParam {
                kind: 0,
                data: ptr::null_mut(),
            },
        ];
        let rc = unsafe { (self.api.render)(self.render, params.as_mut_ptr()) };
        if rc < 0 {
            return;
        }
        let width = self.width as usize;
        let height = self.height as usize;
        for y in 0..height {
            let src = y * self.stride;
            let dst = y * width * 4;
            for x in 0..width {
                let s = src + x * 4;
                let d = dst + x * 4;
                self.pixels[d] = self.scratch[s];
                self.pixels[d + 1] = self.scratch[s + 1];
                self.pixels[d + 2] = self.scratch[s + 2];
                self.pixels[d + 3] = 255;
            }
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
}

impl Drop for Player {
    fn drop(&mut self) {
        unsafe {
            if !self.render.is_null() {
                (self.api.render_free)(self.render);
                self.render = ptr::null_mut();
            }
            if !self.handle.is_null() {
                (self.api.terminate)(self.handle);
                self.handle = ptr::null_mut();
            }
        }
    }
}

fn load_api() -> Result<Api, String> {
    let lib = open_library()?;
    Ok(Api {
        create: load_sym(&lib, b"mpv_create\0")?,
        initialize: load_sym(&lib, b"mpv_initialize\0")?,
        set_option_string: load_sym(&lib, b"mpv_set_option_string\0")?,
        command: load_sym(&lib, b"mpv_command\0")?,
        set_property_string: load_sym(&lib, b"mpv_set_property_string\0")?,
        get_property: load_sym(&lib, b"mpv_get_property\0")?,
        wait_event: load_sym(&lib, b"mpv_wait_event\0")?,
        terminate: load_sym(&lib, b"mpv_terminate_destroy\0")?,
        error_string: load_sym(&lib, b"mpv_error_string\0")?,
        render_create: load_sym(&lib, b"mpv_render_context_create\0")?,
        render_update: load_sym(&lib, b"mpv_render_context_update\0")?,
        render: load_sym(&lib, b"mpv_render_context_render\0")?,
        render_free: load_sym(&lib, b"mpv_render_context_free\0")?,
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
