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

struct Viewer(Mutex<Option<Child>>);

fn has_viewer(dir: &Path) -> bool {
    dir.join("tesseract").join("demo.py").is_file() && dir.join("web").join("index.html").is_file()
}

/// The checkout that holds `tesseract/` and `web/`.
fn repo_root() -> Option<PathBuf> {
    if let Ok(root) = std::env::var("FOURD_MIND_ROOT") {
        return Some(PathBuf::from(root));
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
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
    let root = repo_root().ok_or(
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
    *app.state::<Viewer>().0.lock().unwrap() = Some(child);

    let app = app.clone();
    std::thread::spawn(move || {
        let address = format!("127.0.0.1:{port}");
        let started = Instant::now();
        loop {
            let exited = app
                .state::<Viewer>()
                .0
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
    if let Some(mut child) = app.state::<Viewer>().0.lock().unwrap().take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn main() {
    tauri::Builder::default()
        .manage(Viewer(Mutex::new(None)))
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
