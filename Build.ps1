# Builds the Rust API and the Next.js app with the toolchain bundled under .tools.
# Stop the running console first (the API .exe is locked while it runs).
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
# The GNU tools mis-handle paths with spaces, so reach .tools through a space-free junction.
$tools = Join-Path $env:TEMP "field-console-tools"
if (!(Test-Path $tools)) { cmd /c mklink /J "$tools" "$root\.tools" | Out-Null }
$selfContained = "$tools\rustup\toolchains\stable-x86_64-pc-windows-gnu\lib\rustlib\x86_64-pc-windows-gnu\bin\self-contained"
$env:CARGO_HOME = "$tools\cargo"
$env:RUSTUP_HOME = "$tools\rustup"
# Drop any 32-bit MinGW from PATH; it shadows the 64-bit linker tools.
$env:PATH = "$selfContained;$tools\cargo\bin;" + (($env:PATH -split ';' | Where-Object { $_ -notlike '*MinGW*' }) -join ';')
# Link with Rust's bundled linker; use w64devkit's dlltool, which ships with the assembler it needs.
$env:CARGO_ENCODED_RUSTFLAGS = "-Clink-self-contained=yes" + [char]0x1f + "-Cdlltool=$tools\w64devkit\bin\dlltool.exe"
Push-Location $root
try {
    cargo build --locked
    if ($LASTEXITCODE) { throw "cargo build failed" }
    npm.cmd --prefix web run build
    if ($LASTEXITCODE) { throw "web build failed" }
    Write-Output "Build complete. Run ./Start.ps1"
} finally { Pop-Location }
