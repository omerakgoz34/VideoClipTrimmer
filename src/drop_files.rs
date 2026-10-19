//! File drops.
//!
//! On Windows, miniquad accepts the drop but never tells the app, and it
//! reads the whole file into memory first. A video is too big for that, so
//! the drop dies. This module takes the path itself and leaves the file on disk.

use std::path::PathBuf;

pub fn install() {
    #[cfg(windows)]
    win::install();
}

pub fn take() -> Vec<PathBuf> {
    #[cfg(windows)]
    {
        win::take()
    }
    #[cfg(not(windows))]
    {
        Vec::new()
    }
}

#[cfg(windows)]
#[allow(clippy::upper_case_acronyms)]
mod win {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStringExt;
    use std::path::PathBuf;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};

    type HWND = *mut c_void;
    type WPARAM = usize;
    type LPARAM = isize;
    type LRESULT = isize;
    type UINT = u32;
    type DWORD = u32;
    type HDROP = *mut c_void;

    const WM_DROPFILES: UINT = 0x0233;
    const WM_COPYGLOBALDATA: UINT = 0x0049;
    const GWLP_WNDPROC: i32 = -4;
    const MSGFLT_ALLOW: DWORD = 1;

    static INSTALLED: AtomicBool = AtomicBool::new(false);
    static ORIG: AtomicIsize = AtomicIsize::new(0);
    static PATHS: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

    #[link(name = "user32")]
    unsafe extern "system" {
        fn FindWindowW(class: *const u16, title: *const u16) -> HWND;
        fn SetWindowLongPtrW(hwnd: HWND, index: i32, new: isize) -> isize;
        fn CallWindowProcW(
            prev: isize,
            hwnd: HWND,
            msg: UINT,
            wparam: WPARAM,
            lparam: LPARAM,
        ) -> LRESULT;
        fn DefWindowProcW(hwnd: HWND, msg: UINT, wparam: WPARAM, lparam: LPARAM) -> LRESULT;
        fn ChangeWindowMessageFilterEx(
            hwnd: HWND,
            message: UINT,
            action: DWORD,
            filter: *mut c_void,
        ) -> i32;
    }

    #[link(name = "shell32")]
    unsafe extern "system" {
        fn DragQueryFileW(hdrop: HDROP, index: u32, file: *mut u16, size: u32) -> u32;
        fn DragFinish(hdrop: HDROP);
    }

    pub fn install() {
        if INSTALLED.load(Ordering::Relaxed) {
            return;
        }
        unsafe {
            let class = wide("MINIQUADAPP");
            let hwnd = FindWindowW(class.as_ptr(), std::ptr::null());
            if hwnd.is_null() {
                return;
            }
            let proc = wnd_proc as *const () as isize;
            let prev = SetWindowLongPtrW(hwnd, GWLP_WNDPROC, proc);
            if prev == 0 {
                return;
            }
            ORIG.store(prev, Ordering::Relaxed);
            ChangeWindowMessageFilterEx(hwnd, WM_DROPFILES, MSGFLT_ALLOW, std::ptr::null_mut());
            ChangeWindowMessageFilterEx(
                hwnd,
                WM_COPYGLOBALDATA,
                MSGFLT_ALLOW,
                std::ptr::null_mut(),
            );
            INSTALLED.store(true, Ordering::Relaxed);
        }
    }

    pub fn take() -> Vec<PathBuf> {
        PATHS
            .lock()
            .map(|mut paths| std::mem::take(&mut *paths))
            .unwrap_or_default()
    }

    unsafe extern "system" fn wnd_proc(
        hwnd: HWND,
        msg: UINT,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        unsafe {
            if msg == WM_DROPFILES {
                let hdrop = wparam as HDROP;
                let paths = query(hdrop);
                DragFinish(hdrop);
                if let Ok(mut slot) = PATHS.lock() {
                    slot.extend(paths);
                }
                miniquad::window::schedule_update();
                return 0;
            }
            let prev = ORIG.load(Ordering::Relaxed);
            if prev == 0 {
                return DefWindowProcW(hwnd, msg, wparam, lparam);
            }
            CallWindowProcW(prev, hwnd, msg, wparam, lparam)
        }
    }

    unsafe fn query(hdrop: HDROP) -> Vec<PathBuf> {
        unsafe {
            let count = DragQueryFileW(hdrop, u32::MAX, std::ptr::null_mut(), 0);
            let mut paths = Vec::with_capacity(count as usize);
            for index in 0..count {
                let len = DragQueryFileW(hdrop, index, std::ptr::null_mut(), 0);
                if len == 0 {
                    continue;
                }
                let mut buf = vec![0u16; len as usize + 1];
                let written = DragQueryFileW(hdrop, index, buf.as_mut_ptr(), buf.len() as u32);
                if written == 0 {
                    continue;
                }
                buf.truncate(written as usize);
                let path = PathBuf::from(std::ffi::OsString::from_wide(&buf));
                if path.is_file() {
                    paths.push(path);
                }
            }
            paths
        }
    }

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(std::iter::once(0)).collect()
    }
}
