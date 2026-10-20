//! Real playback through libmpv, loaded at run time.
//!
//! mpv owns the clock and the audio. Frames are drawn into an OpenGL texture
//! in this window, at the source resolution, so the CPU does not copy them.
//! If that render path is missing, frames fall back to a software buffer.

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
const OPENGL_INIT: c_int = 2;
const OPENGL_FBO: c_int = 3;
const FLIP_Y: c_int = 4;
const ADVANCED_CONTROL: c_int = 10;
const BLOCK_FOR_TARGET_TIME: c_int = 12;
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
    gpu: Option<GpuTarget>,
    soft: Option<SoftTarget>,
    hw: bool,
    waiting_frame: bool,
    pending_seek: Option<f64>,
    surface_tried: bool,
}

struct GpuTarget {
    tex: u32,
    fbo: u32,
}

struct SoftTarget {
    pixels: Vec<u8>,
    scratch: Vec<u8>,
    stride: usize,
}

/// One displayed frame. The GPU path is a texture already in the window's
/// OpenGL context, so the CPU never sees the pixels.
pub enum Shot {
    Gpu {
        id: u32,
        width: u32,
        height: u32,
    },
    Cpu {
        pixels: Vec<u8>,
        width: u32,
        height: u32,
    },
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
            gpu: None,
            soft: None,
            hw,
            waiting_frame: true,
            pending_seek: Some(0.0),
            surface_tried: false,
        };
        player.configure()?;
        player.load(path)?;
        Ok(player)
    }

    pub fn hw_on(&self) -> bool {
        self.hw
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

    /// A newly displayed frame, plus mpv's clock. `None` for the frame when
    /// the picture on screen has not changed.
    pub fn poll(&mut self) -> (Option<Shot>, Option<f64>) {
        self.drain();
        if !self.ensure_surface() {
            return (None, self.time());
        }
        let flags = unsafe { (self.api.render_update)(self.render) };
        if flags & UPDATE_FRAME == 0 && !self.waiting_frame {
            return (None, self.time());
        }
        if self.draw_frame() {
            self.waiting_frame = false;
            return (Some(self.shot()), self.time());
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
        Ok(())
    }

    fn ensure_surface(&mut self) -> bool {
        if !self.render.is_null() {
            return true;
        }
        if self.surface_tried {
            return false;
        }
        self.surface_tried = true;
        if self.start_gpu().is_ok() {
            return true;
        }
        self.start_software().is_ok()
    }

    /// Draw into an OpenGL texture in this window. No pixel readback.
    fn start_gpu(&mut self) -> Result<(), ()> {
        let init = OpenGlInit {
            get_proc_address: gl_proc_address,
            ctx: ptr::null_mut(),
        };
        let api_name = c"opengl";
        let mut advanced: c_int = 1;
        let mut params = [
            RenderParam {
                kind: API_TYPE,
                data: api_name.as_ptr() as *mut c_void,
            },
            RenderParam {
                kind: OPENGL_INIT,
                data: (&init as *const OpenGlInit).cast_mut().cast(),
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
        let mut render = ptr::null_mut();
        let rc = unsafe { (self.api.render_create)(&mut render, self.handle, params.as_mut_ptr()) };
        if rc < 0 || render.is_null() {
            return Err(());
        }
        self.render = render;
        match GpuTarget::new(self.width, self.height) {
            Some(gpu) => {
                self.gpu = Some(gpu);
                Ok(())
            }
            None => {
                unsafe { (self.api.render_free)(self.render) };
                self.render = ptr::null_mut();
                Err(())
            }
        }
    }

    fn start_software(&mut self) -> Result<(), ()> {
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
            return Err(());
        }
        self.render = render;
        let stride = (self.width as usize * 4).div_ceil(64) * 64;
        self.soft = Some(SoftTarget {
            pixels: vec![0; self.width as usize * self.height as usize * 4],
            scratch: vec![0; stride * self.height as usize],
            stride,
        });
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

    fn draw_frame(&mut self) -> bool {
        if self.gpu.is_some() {
            return self.draw_gpu();
        }
        self.draw_software()
    }

    fn draw_gpu(&mut self) -> bool {
        let Some(gpu) = &self.gpu else {
            return false;
        };
        let saved = GlSnap::take();
        let fbo = OpenGlFbo {
            fbo: gpu.fbo as c_int,
            w: self.width as c_int,
            h: self.height as c_int,
            internal_format: 0,
        };
        let mut flip: c_int = 1;
        let mut block: c_int = 0;
        let mut params = [
            RenderParam {
                kind: OPENGL_FBO,
                data: (&fbo as *const OpenGlFbo).cast_mut().cast(),
            },
            RenderParam {
                kind: FLIP_Y,
                data: (&mut flip as *mut c_int).cast(),
            },
            RenderParam {
                kind: BLOCK_FOR_TARGET_TIME,
                data: (&mut block as *mut c_int).cast(),
            },
            RenderParam {
                kind: 0,
                data: ptr::null_mut(),
            },
        ];
        let rc = unsafe { (self.api.render)(self.render, params.as_mut_ptr()) };
        saved.restore();
        rc >= 0
    }

    fn draw_software(&mut self) -> bool {
        let Some(soft) = &mut self.soft else {
            return false;
        };
        let mut size = [self.width as c_int, self.height as c_int];
        let mut stride = soft.stride;
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
                data: soft.scratch.as_mut_ptr().cast(),
            },
            RenderParam {
                kind: 0,
                data: ptr::null_mut(),
            },
        ];
        let rc = unsafe { (self.api.render)(self.render, params.as_mut_ptr()) };
        if rc < 0 {
            return false;
        }
        let width = self.width as usize;
        let height = self.height as usize;
        for y in 0..height {
            let src = y * soft.stride;
            let dst = y * width * 4;
            soft.pixels[dst..dst + width * 4].copy_from_slice(&soft.scratch[src..src + width * 4]);
            let mut x = 3;
            while x < width * 4 {
                soft.pixels[dst + x] = 255;
                x += 4;
            }
        }
        true
    }

    fn shot(&self) -> Shot {
        if let Some(gpu) = &self.gpu {
            return Shot::Gpu {
                id: gpu.tex,
                width: self.width,
                height: self.height,
            };
        }
        let soft = self.soft.as_ref().expect("software frame");
        Shot::Cpu {
            pixels: soft.pixels.clone(),
            width: self.width,
            height: self.height,
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
            if let Some(gpu) = self.gpu.take() {
                gpu.free();
            }
            if !self.handle.is_null() {
                (self.api.terminate)(self.handle);
                self.handle = ptr::null_mut();
            }
        }
    }
}

#[repr(C)]
struct OpenGlInit {
    get_proc_address: unsafe extern "C" fn(*mut c_void, *const c_char) -> *mut c_void,
    ctx: *mut c_void,
}

#[repr(C)]
struct OpenGlFbo {
    fbo: c_int,
    w: c_int,
    h: c_int,
    internal_format: c_int,
}

impl GpuTarget {
    fn new(width: u32, height: u32) -> Option<Self> {
        use miniquad::gl::{
            GL_CLAMP_TO_EDGE, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER, GL_LINEAR, GL_RGBA, GL_RGBA8,
            GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S,
            GL_TEXTURE_WRAP_T, GL_UNSIGNED_BYTE, glBindFramebuffer, glBindTexture,
            glCheckFramebufferStatus, glDeleteFramebuffers, glDeleteTextures,
            glFramebufferTexture2D, glGenFramebuffers, glGenTextures, glTexImage2D,
            glTexParameteri,
        };
        const COMPLETE: u32 = 0x8CD5;
        unsafe {
            let mut tex = 0u32;
            glGenTextures(1, &mut tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA8 as i32,
                width as i32,
                height as i32,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                ptr::null(),
            );
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR as i32);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR as i32);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE as i32);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE as i32);
            let mut fbo = 0u32;
            glGenFramebuffers(1, &mut fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
            let ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) == COMPLETE;
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glBindTexture(GL_TEXTURE_2D, 0);
            if !ok || tex == 0 || fbo == 0 {
                if fbo != 0 {
                    glDeleteFramebuffers(1, &fbo);
                }
                if tex != 0 {
                    glDeleteTextures(1, &tex);
                }
                return None;
            }
            Some(Self { tex, fbo })
        }
    }

    fn free(self) {
        unsafe {
            miniquad::gl::glDeleteFramebuffers(1, &self.fbo);
            miniquad::gl::glDeleteTextures(1, &self.tex);
        }
    }
}

struct GlSnap {
    fbo: i32,
    vao: i32,
    program: i32,
    active: i32,
    tex: i32,
    viewport: [i32; 4],
    scissor: [i32; 4],
}

impl GlSnap {
    fn take() -> Self {
        Self {
            fbo: gl_int(0x8CA6),
            vao: gl_int(0x85B5),
            program: gl_int(0x8B8D),
            active: gl_int(0x84E0),
            tex: gl_int(0x8069),
            viewport: gl_ints(0x0BA2),
            scissor: gl_ints(0x0C10),
        }
    }

    fn restore(self) {
        use miniquad::gl::{
            GL_FRAMEBUFFER, GL_TEXTURE_2D, glActiveTexture, glBindFramebuffer, glBindTexture,
            glBindVertexArray, glScissor, glUseProgram, glViewport,
        };
        unsafe {
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbo as u32);
            glBindVertexArray(self.vao as u32);
            glUseProgram(self.program as u32);
            glActiveTexture(self.active as u32);
            glBindTexture(GL_TEXTURE_2D, self.tex as u32);
            glViewport(
                self.viewport[0],
                self.viewport[1],
                self.viewport[2],
                self.viewport[3],
            );
            glScissor(
                self.scissor[0],
                self.scissor[1],
                self.scissor[2],
                self.scissor[3],
            );
        }
    }
}

fn gl_int(pname: u32) -> i32 {
    let mut value = 0;
    unsafe { miniquad::gl::glGetIntegerv(pname, &mut value) };
    value
}

fn gl_ints(pname: u32) -> [i32; 4] {
    let mut value = [0; 4];
    unsafe { miniquad::gl::glGetIntegerv(pname, value.as_mut_ptr()) };
    value
}

unsafe extern "C" fn gl_proc_address(_ctx: *mut c_void, name: *const c_char) -> *mut c_void {
    if name.is_null() {
        return ptr::null_mut();
    }
    lookup_gl(name)
}

fn lookup_gl(name: *const c_char) -> *mut c_void {
    #[cfg(windows)]
    {
        windows_gl(name)
    }
    #[cfg(not(windows))]
    {
        unix_gl(name)
    }
}

#[cfg(windows)]
fn windows_gl(name: *const c_char) -> *mut c_void {
    #[link(name = "opengl32")]
    unsafe extern "system" {
        fn wglGetProcAddress(name: *const i8) -> *mut c_void;
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleA(name: *const i8) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
    }
    unsafe {
        let found = wglGetProcAddress(name.cast());
        let raw = found as usize;
        if raw > 3 {
            return found;
        }
        let module = GetModuleHandleA(c"opengl32.dll".as_ptr().cast());
        GetProcAddress(module, name.cast())
    }
}

#[cfg(not(windows))]
fn unix_gl(name: *const c_char) -> *mut c_void {
    use std::sync::OnceLock;
    type Proc = unsafe extern "C" fn(*const c_char) -> *mut c_void;
    struct Loader {
        _libs: Vec<libloading::Library>,
        egl: Option<Proc>,
        glx: Option<Proc>,
    }
    static LOADER: OnceLock<Loader> = OnceLock::new();
    let loader = LOADER.get_or_init(|| {
        let mut libs = Vec::new();
        let mut egl = None;
        let mut glx = None;
        if let Ok(lib) = unsafe { libloading::Library::new("libEGL.so.1") } {
            egl = unsafe { lib.get(b"eglGetProcAddress\0") }
                .ok()
                .map(|sym| *sym);
            libs.push(lib);
        }
        if let Ok(lib) = unsafe { libloading::Library::new("libGL.so.1") } {
            glx = unsafe { lib.get(b"glXGetProcAddressARB\0") }
                .ok()
                .map(|sym| *sym);
            libs.push(lib);
        }
        Loader {
            _libs: libs,
            egl,
            glx,
        }
    });
    unsafe {
        if let Some(egl) = loader.egl {
            let found = egl(name);
            if !found.is_null() {
                return found;
            }
        }
        if let Some(glx) = loader.glx {
            return glx(name);
        }
    }
    ptr::null_mut()
}

pub fn library_present() -> bool {
    open_library().is_ok()
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
