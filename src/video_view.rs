pub struct VideoView {
    #[cfg(windows)]
    hwnd: *mut std::ffi::c_void,
}

impl VideoView {
    pub fn attach() -> Result<Self, String> {
        #[cfg(windows)]
        {
            win::attach()
        }
        #[cfg(not(windows))]
        {
            Err("mpv playback inside the window is only set up on Windows.".into())
        }
    }

    pub fn id(&self) -> i64 {
        #[cfg(windows)]
        {
            self.hwnd as isize as i64
        }
        #[cfg(not(windows))]
        {
            0
        }
    }

    pub fn place(&self, x: i32, y: i32, w: i32, h: i32) {
        #[cfg(windows)]
        win::place(self.hwnd, x, y, w, h);
        #[cfg(not(windows))]
        {
            let _ = (x, y, w, h);
        }
    }

    pub fn hide(&self) {
        #[cfg(windows)]
        win::hide(self.hwnd);
    }

    pub fn hook_clicks(&self) {
        #[cfg(windows)]
        win::hook_clicks(self.hwnd);
    }

    pub fn destroy(&self) {
        #[cfg(windows)]
        win::destroy(self.hwnd);
    }
}

#[cfg(windows)]
#[allow(clippy::upper_case_acronyms)]
mod win {
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, Ordering};

    type HWND = *mut c_void;
    type HINSTANCE = *mut c_void;
    type HBRUSH = *mut c_void;
    type HCURSOR = *mut c_void;
    type WPARAM = usize;
    type LPARAM = isize;
    type LRESULT = isize;
    type UINT = u32;

    const WS_CHILD: u32 = 0x4000_0000;
    const WS_CLIPSIBLINGS: u32 = 0x0400_0000;
    const WS_CLIPCHILDREN: u32 = 0x0200_0000;
    const SW_HIDE: i32 = 0;
    const WM_MOUSEACTIVATE: UINT = 0x0021;
    const MA_NOACTIVATE: LRESULT = 3;
    const SWP_NOZORDER: u32 = 0x0004;
    const SWP_NOACTIVATE: u32 = 0x0010;
    const SWP_SHOWWINDOW: u32 = 0x0040;

    static CLASS_READY: AtomicBool = AtomicBool::new(false);
    static PREV_PROC: std::sync::atomic::AtomicIsize = std::sync::atomic::AtomicIsize::new(0);
    static PROC_DEPTH: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    const GWLP_WNDPROC: i32 = -4;

    #[link(name = "user32")]
    unsafe extern "system" {
        fn FindWindowW(class: *const u16, title: *const u16) -> HWND;
        fn RegisterClassW(class: *const WndClass) -> u16;
        fn CreateWindowExW(
            ex: u32,
            class: *const u16,
            title: *const u16,
            style: u32,
            x: i32,
            y: i32,
            w: i32,
            h: i32,
            parent: HWND,
            menu: *mut c_void,
            instance: HINSTANCE,
            param: *mut c_void,
        ) -> HWND;
        fn DefWindowProcW(hwnd: HWND, msg: UINT, wparam: WPARAM, lparam: LPARAM) -> LRESULT;
        fn DestroyWindow(hwnd: HWND) -> i32;
        fn ShowWindow(hwnd: HWND, cmd: i32) -> i32;
        fn SetWindowPos(hwnd: HWND, after: HWND, x: i32, y: i32, w: i32, h: i32, flags: u32)
        -> i32;
        fn SetWindowLongPtrW(hwnd: HWND, index: i32, new: isize) -> isize;
        fn CallWindowProcW(
            prev: isize,
            hwnd: HWND,
            msg: UINT,
            wparam: WPARAM,
            lparam: LPARAM,
        ) -> LRESULT;
        fn LoadCursorW(instance: HINSTANCE, name: *const u16) -> HCURSOR;
        fn GetModuleHandleW(name: *const u16) -> HINSTANCE;
        fn CreateSolidBrush(color: u32) -> HBRUSH;
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetLastError() -> u32;
    }

    #[repr(C)]
    struct WndClass {
        style: u32,
        wnd_proc: unsafe extern "system" fn(HWND, UINT, WPARAM, LPARAM) -> LRESULT,
        cls_extra: i32,
        wnd_extra: i32,
        instance: HINSTANCE,
        icon: *mut c_void,
        cursor: HCURSOR,
        background: HBRUSH,
        menu: *const u16,
        class_name: *const u16,
    }

    pub fn attach() -> Result<super::VideoView, String> {
        unsafe {
            ensure_class()?;
            let class = wide("MINIQUADAPP");
            let parent = FindWindowW(class.as_ptr(), std::ptr::null());
            if parent.is_null() {
                return Err("The app window is not ready for playback.".into());
            }
            let view_class = wide("VideoClipView");
            let hwnd = CreateWindowExW(
                0,
                view_class.as_ptr(),
                std::ptr::null(),
                WS_CHILD | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
                0,
                0,
                1,
                1,
                parent,
                std::ptr::null_mut(),
                GetModuleHandleW(std::ptr::null()),
                std::ptr::null_mut(),
            );
            if hwnd.is_null() {
                return Err("Could not open the video view.".into());
            }
            Ok(super::VideoView { hwnd })
        }
    }

    pub fn place(hwnd: HWND, x: i32, y: i32, w: i32, h: i32) {
        unsafe {
            SetWindowPos(
                hwnd,
                std::ptr::null_mut(),
                x,
                y,
                w.max(1),
                h.max(1),
                SWP_NOZORDER | SWP_NOACTIVATE | SWP_SHOWWINDOW,
            );
        }
    }

    pub fn hide(hwnd: HWND) {
        unsafe {
            ShowWindow(hwnd, SW_HIDE);
        }
    }

    pub fn hook_clicks(hwnd: HWND) {
        unsafe {
            let prev = SetWindowLongPtrW(hwnd, GWLP_WNDPROC, click_proc as *const () as isize);
            // mpv already calls the class procedure. Pointing this at the hook
            // would call the hook again and overflow the stack.
            if prev != 0 && prev != click_proc as *const () as isize {
                PREV_PROC.store(prev, Ordering::Relaxed);
            }
        }
    }

    pub fn destroy(hwnd: HWND) {
        unsafe {
            if !hwnd.is_null() {
                DestroyWindow(hwnd);
            }
        }
    }

    unsafe fn ensure_class() -> Result<(), String> {
        use std::sync::OnceLock;
        if CLASS_READY.load(Ordering::Relaxed) {
            return Ok(());
        }
        static NAME: OnceLock<Vec<u16>> = OnceLock::new();
        unsafe {
            let name = NAME.get_or_init(|| wide("VideoClipView"));
            let cursor = LoadCursorW(std::ptr::null_mut(), 32512 as *const u16);
            let class = WndClass {
                style: 0,
                wnd_proc: class_proc,
                cls_extra: 0,
                wnd_extra: 0,
                instance: GetModuleHandleW(std::ptr::null()),
                icon: std::ptr::null_mut(),
                cursor,
                background: CreateSolidBrush(0x0014_1110),
                menu: std::ptr::null(),
                class_name: name.as_ptr(),
            };
            if RegisterClassW(&class) == 0 && GetLastError() != 1410 {
                return Err("Could not prepare the video view.".into());
            }
            CLASS_READY.store(true, Ordering::Relaxed);
        }
        Ok(())
    }

    unsafe extern "system" fn class_proc(
        hwnd: HWND,
        msg: UINT,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
    }

    unsafe extern "system" fn click_proc(
        hwnd: HWND,
        msg: UINT,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        unsafe {
            if PROC_DEPTH.fetch_add(1, Ordering::Relaxed) > 4 {
                PROC_DEPTH.fetch_sub(1, Ordering::Relaxed);
                return DefWindowProcW(hwnd, msg, wparam, lparam);
            }
            let result = if msg == WM_MOUSEACTIVATE {
                MA_NOACTIVATE
            } else {
                let prev = PREV_PROC.load(Ordering::Relaxed);
                if prev != 0 {
                    CallWindowProcW(prev, hwnd, msg, wparam, lparam)
                } else {
                    DefWindowProcW(hwnd, msg, wparam, lparam)
                }
            };
            PROC_DEPTH.fetch_sub(1, Ordering::Relaxed);
            result
        }
    }

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(Some(0)).collect()
    }
}
