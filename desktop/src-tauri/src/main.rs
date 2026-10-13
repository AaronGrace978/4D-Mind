#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tauri::{Manager, RunEvent, Url, WebviewWindow};

const STARTUP_LIMIT: Duration = Duration::from_secs(60);

struct Viewer {
    child: Mutex<Option<Child>>,
    root: Mutex<Option<PathBuf>>,
    port: Mutex<Option<u16>>,
}

#[derive(serde::Serialize)]
struct IpcResult {
    ok: bool,
    returncode: i32,
    output: String,
    content: Option<String>,
}

fn safe_inside(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let canon_root = root.canonicalize().map_err(|e| format!("Invalid root: {e}"))?;
    let candidate = canon_root.join(relative);
    let canon_file = candidate.canonicalize().map_err(|e| format!("File not found ({relative}): {e}"))?;
    if !canon_file.starts_with(&canon_root) || !canon_file.is_file() {
        return Err(format!("Path escapes repository root or is not a file: {relative}"));
    }
    Ok(canon_file)
}

#[tauri::command]
fn native_read_file(state: tauri::State<'_, Viewer>, file: String) -> Result<IpcResult, String> {
    let root = state.root.lock().unwrap().clone().ok_or("No repository root active")?;
    let path = safe_inside(&root, &file)?;
    let text = std::fs::read_to_string(&path).map_err(|e| format!("Could not read {file}: {e}"))?;
    let lines = text.lines().count();
    Ok(IpcResult {
        ok: true,
        returncode: 0,
        output: format!("[Rust IPC] Loaded {file} ({lines} lines)"),
        content: Some(text),
    })
}

#[tauri::command]
fn native_run_harness(state: tauri::State<'_, Viewer>, action: String) -> Result<IpcResult, String> {
    let root = state.root.lock().unwrap().clone().ok_or("No repository root active")?;
    let py = python();
    let (prog, args): (&str, Vec<&str>) = match action.as_str() {
        "pytest" => (&py, vec!["-m", "pytest", "-q"]),
        "git_status" => ("git", vec!["status", "-s"]),
        "git_diff" => ("git", vec!["diff", "--stat"]),
        "cargo_check" => ("cargo", vec!["check"]),
        "npm_test" => ("npm", vec!["test"]),
        other => return Err(format!("Unsupported Rust IPC harness action: {other}")),
    };
    let out = Command::new(prog)
        .args(&args)
        .current_dir(&root)
        .output()
        .map_err(|e| format!("Failed to spawn {prog}: {e}"))?;
    let code = out.status.code().unwrap_or(-1);
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    let err_text = String::from_utf8_lossy(&out.stderr);
    if !err_text.trim().is_empty() {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&err_text);
    }
    Ok(IpcResult {
        ok: out.status.success(),
        returncode: code,
        output: if text.trim().is_empty() { "(completed with no output)".into() } else { text },
        content: None,
    })
}

#[tauri::command]
fn native_patch_and_verify(
    state: tauri::State<'_, Viewer>,
    file: String,
    find_text: String,
    replace_text: String,
    auto_verify: bool,
    rollback_on_fail: bool,
) -> Result<IpcResult, String> {
    let root = state.root.lock().unwrap().clone().ok_or("No repository root active")?;
    let path = safe_inside(&root, &file)?;
    if find_text.is_empty() {
        return Err("Provide non-empty find_text".into());
    }
    let original = std::fs::read_to_string(&path).map_err(|e| format!("Read failed: {e}"))?;
    if !original.contains(&find_text) {
        return Err(format!("find_text not found in {file}"));
    }
    let updated = original.replacen(&find_text, &replace_text, 1);
    std::fs::write(&path, &updated).map_err(|e| format!("Write failed: {e}"))?;
    let mut msg = format!("[Rust IPC] Patched {file} ({} -> {} bytes).", find_text.len(), replace_text.len());
    let mut ok = true;
    let mut code = 0;
    if auto_verify {
        let verify = native_run_harness(state, "pytest".into())?;
        ok = verify.ok;
        code = verify.returncode;
        if !ok && rollback_on_fail {
            let _ = std::fs::write(&path, &original);
            msg.push_str("\n[Self-Validating 4D CI] pytest FAILED — rolled back file via native Rust IPC.");
        } else if ok {
            msg.push_str("\n[Self-Validating 4D CI] pytest PASSED.");
        } else {
            msg.push_str(&format!("\n[Self-Validating 4D CI] pytest FAILED (exit {code})."));
        }
        msg.push_str("\n\n");
        msg.push_str(&verify.output);
    }
    Ok(IpcResult {
        ok,
        returncode: code,
        output: msg,
        content: None,
    })
}

fn has_viewer(dir: &Path) -> bool {
    dir.join("tesseract").join("demo.py").is_file() && dir.join("web").join("index.html").is_file()
}

/// The checkout or bundled resource directory that holds `tesseract/` and `web/`.
fn repo_root(app: &tauri::AppHandle) -> Option<PathBuf> {
    if let Ok(root) = std::env::var("FOURD_MIND_ROOT") {
        return Some(PathBuf::from(root));
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(res_dir) = app.path().resource_dir() {
        candidates.push(res_dir.clone());
        candidates.push(res_dir.join("_up_").join("_up_"));
    }
    if let Ok(exe) = std::env::current_exe() {
        candidates.extend(exe.ancestors().map(Path::to_path_buf));
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.extend(cwd.ancestors().map(Path::to_path_buf));
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join(".."));
    candidates.into_iter().find(|dir| has_viewer(dir))
}

fn python() -> String {
    std::env::var("FOURD_MIND_PYTHON").unwrap_or_else(|_| {
        if cfg!(windows) { "python".into() } else { "python3".into() }
    })
}

fn free_port() -> std::io::Result<u16> {
    Ok(TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

/// Arguments for `python -m tesseract.demo`, after the port.
fn viewer_args() -> Vec<String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if let Ok(repo) = std::env::var("FOURD_MIND_REPO") {
        args.extend(["--repo".to_string(), repo]);
    } else if let Ok(npz) = std::env::var("FOURD_MIND_NPZ") {
        args.extend(["--npz".to_string(), npz]);
    }
    args
}

fn show_error(window: &WebviewWindow, message: &str) {
    let literal = serde_json::to_string(message).unwrap_or_else(|_| "\"error\"".into());
    let _ = window.eval(format!("window.showError({literal})"));
}

fn start_viewer(app: &tauri::AppHandle, window: WebviewWindow) -> Result<(), String> {
    let root = repo_root(app).ok_or(
        "Could not find the 4D-Mind checkout (tesseract/ and web/). Set FOURD_MIND_ROOT to its path.",
    )?;
    let port = free_port().map_err(|err| format!("No free local port: {err}"))?;
    let python = python();
    let mut child = Command::new(&python)
        .args(["-m", "tesseract.demo", "--port", &port.to_string()])
        .args(viewer_args())
        .current_dir(&root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| format!("Could not run {python}: {err}. Set FOURD_MIND_PYTHON to a Python with numpy."))?;

    let tail = Arc::new(Mutex::new(VecDeque::<String>::new()));
    if let Some(stderr) = child.stderr.take() {
        let tail = Arc::clone(&tail);
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                eprintln!("{line}");
                let mut tail = tail.lock().unwrap();
                tail.push_back(line);
                if tail.len() > 20 {
                    tail.pop_front();
                }
            }
        });
    }
    *app.state::<Viewer>().child.lock().unwrap() = Some(child);
    *app.state::<Viewer>().root.lock().unwrap() = Some(root.clone());
    *app.state::<Viewer>().port.lock().unwrap() = Some(port);

    let app = app.clone();
    std::thread::spawn(move || {
        let address = format!("127.0.0.1:{port}");
        let started = Instant::now();
        loop {
            let exited = app
                .state::<Viewer>()
                .child
                .lock()
                .unwrap()
                .as_mut()
                .and_then(|child| child.try_wait().ok().flatten());
            if let Some(status) = exited {
                std::thread::sleep(Duration::from_millis(200));
                let log: Vec<String> = tail.lock().unwrap().iter().cloned().collect();
                show_error(&window, &format!("The viewer exited ({status}).\n\n{}", log.join("\n")));
                return;
            }
            if TcpStream::connect(&address).is_ok() {
                let url = Url::parse(&format!("http://{address}/")).expect("local url");
                if let Err(err) = window.navigate(url) {
                    show_error(&window, &format!("Could not open the viewer: {err}"));
                }
                return;
            }
            if started.elapsed() > STARTUP_LIMIT {
                show_error(&window, "The viewer did not start within a minute.");
                return;
            }
            std::thread::sleep(Duration::from_millis(150));
        }
    });
    Ok(())
}

fn stop_viewer(app: &tauri::AppHandle) {
    if let Some(mut child) = app.state::<Viewer>().child.lock().unwrap().take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn main() {
    tauri::Builder::default()
        .manage(Viewer {
            child: Mutex::new(None),
            root: Mutex::new(None),
            port: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            native_read_file,
            native_run_harness,
            native_patch_and_verify
        ])
        .setup(|app| {
            let window = app.get_webview_window("main").expect("main window");
            if let Err(message) = start_viewer(app.handle(), window.clone()) {
                // The loading page defines showError once it has parsed.
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_millis(800));
                    show_error(&window, &message);
                });
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build the 4D Mind window")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                stop_viewer(app);
            }
        });
}
