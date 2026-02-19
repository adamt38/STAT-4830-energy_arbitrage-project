# Install uv (if missing), create .venv, and install requirements.
# Run from repo root: powershell -ExecutionPolicy Bypass -File .\script\install.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

# Check for uv
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    Write-Host "uv not found. Installing via official Astral installer..."
    Invoke-Expression (Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -UseBasicParsing).Content
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvCmd) {
        Write-Host "uv install may have completed. Close and reopen PowerShell, then re-run this script."
        exit 1
    }
}

Write-Host "Using uv: $(uv --version)"

# Create venv and install
uv venv .venv
uv pip install -r requirements.txt

Write-Host ""
Write-Host "--- Next steps ---"
Write-Host "Activate the virtual environment:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "Run the main script:"
Write-Host "  python script/gd_1d_torch.py"
Write-Host "Run tests:"
Write-Host "  pytest tests/"
