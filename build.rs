use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/icon.ico");
    println!("cargo:rerun-if-env-changed=CARGO_PKG_VERSION");

    if std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref() != Some("windows") {
        return;
    }

    let version = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".into());
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");
    let icon = rc_path(
        &Path::new("assets/icon.ico")
            .canonicalize()
            .expect("assets/icon.ico"),
    );
    let rc_path = Path::new(&out_dir).join("video-clip-trimmer.rc");
    std::fs::write(&rc_path, resource_script(&version, &icon)).unwrap();

    embed_resource::compile(&rc_path, embed_resource::NONE)
        .manifest_required()
        .expect("windows resource compile");
}

fn resource_script(version: &str, icon: &str) -> String {
    let quad = version_quad(version);
    format!(
        r#"1 ICON "{icon}"

1 VERSIONINFO
FILEVERSION {quad}
PRODUCTVERSION {quad}
FILEFLAGSMASK 0x3fL
FILEFLAGS 0x0L
FILEOS 0x40004L
FILETYPE 0x1L
FILESUBTYPE 0x0L
BEGIN
  BLOCK "StringFileInfo"
  BEGIN
    BLOCK "040904B0"
    BEGIN
      VALUE "FileDescription", "VideoClipTrimmer"
      VALUE "FileVersion", "{version}"
      VALUE "InternalName", "video-clip-trimmer"
      VALUE "OriginalFilename", "video-clip-trimmer.exe"
      VALUE "ProductName", "VideoClipTrimmer"
      VALUE "ProductVersion", "{version}"
      VALUE "LegalCopyright", "MIT"
    END
  END
  BLOCK "VarFileInfo"
  BEGIN
    VALUE "Translation", 0x409, 1200
  END
END
"#,
        icon = rc_escape(icon),
    )
}

fn rc_path(path: &Path) -> String {
    let text = path.to_string_lossy();
    let text = text.strip_prefix(r"\\?\").unwrap_or(&text);
    text.replace('\\', "/")
}

fn version_quad(version: &str) -> String {
    let mut parts = version
        .split('.')
        .filter_map(|part| part.parse::<u16>().ok())
        .collect::<Vec<_>>();
    parts.resize(4, 0);
    format!("{},{},{},{}", parts[0], parts[1], parts[2], parts[3])
}

fn rc_escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}
