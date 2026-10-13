$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "=== 4D Mind v1.2.0 — Local Installer (Windows / RTX 5060 Ti Ready) ===" -ForegroundColor Cyan
$Python = if ($env:FOURD_MIND_PYTHON) { $env:FOURD_MIND_PYTHON } else { "python" }

Write-Host "[1/2] Installing 4D Mind Python CLI (4d-mind, 4d-mind-risk, 4d-mind-codeblock, 4d-mind-train)..."
& $Python -m pip install -e .

if ((Get-Command cargo -ErrorAction SilentlyContinue) -and (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "[2/2] Rust & Node detected — building native Windows Tauri 2 installer (.msi / .exe)..."
    Push-Location desktop
    npm install
    npm run build
    Pop-Location
    Write-Host "Windows installers written under desktop\src-tauri\target\release\bundle\" -ForegroundColor Green
} else {
    Write-Host "[2/2] Skipping native Tauri installer build (install Rust + Node 18+ to build .msi/.exe)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Installed! Launch on any git repo:" -ForegroundColor Green
Write-Host "  4d-mind --repo . --port 18765"
