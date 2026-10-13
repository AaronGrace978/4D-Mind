#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== 4D Mind v1.2.0 — Local Installer (Linux / macOS) ==="
PYTHON="${FOURD_MIND_PYTHON:-python3}"

echo "[1/2] Installing 4D Mind Python CLI (4d-mind, 4d-mind-risk, 4d-mind-codeblock, 4d-mind-train)..."
"$PYTHON" -m pip install -e .

if command -v cargo >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
  echo "[2/2] Rust & Node detected — building native Tauri 2 desktop bundle..."
  (cd desktop && npm install && npm run build)
  echo "Desktop installers written under desktop/src-tauri/target/release/bundle/"
else
  echo "[2/2] Skipping native Tauri bundle (install Rust + Node 18+ to build desktop/ installers)."
fi

echo ""
echo "Installed! Run:"
echo "  4d-mind --repo . --port 18765"
