param(
    [switch]$SkipDocker,
    [switch]$SkipWsl,
    [switch]$SkipGit,
    [switch]$SkipPython
)

$ErrorActionPreference = "Stop"

function Info($Message) {
    Write-Host ""
    Write-Host "[setup] $Message" -ForegroundColor Cyan
}

function Assert-Winget {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "winget was not found. Install 'App Installer' from the Microsoft Store, then rerun this script."
    }
}

function Install-WingetPackage($Id, $Name) {
    Info "Installing $Name"
    winget install --id $Id --exact --source winget --accept-package-agreements --accept-source-agreements
}

Assert-Winget

if (-not $SkipPython) {
    Install-WingetPackage "Python.Python.3.11" "Python 3.11"
}

if (-not $SkipGit) {
    Install-WingetPackage "Git.Git" "Git"
}

if (-not $SkipDocker) {
    Install-WingetPackage "Docker.DockerDesktop" "Docker Desktop"
}

if (-not $SkipWsl) {
    Info "Installing WSL Ubuntu"
    wsl --install -d Ubuntu
}

Info "Refreshing Python dependencies for this project"
$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
} else {
    Write-Host "Python was installed, but this terminal may need to be reopened before 'python' is on PATH." -ForegroundColor Yellow
}

Info "Next steps"
Write-Host "1. Restart Windows if WSL or Docker asks you to."
Write-Host "2. Open Docker Desktop and enable the WSL 2 backend."
Write-Host "3. Open Ubuntu from the Start Menu and finish first-time username setup."
Write-Host "4. From Ubuntu, cd into this project and run:"
Write-Host "   bash scripts/setup_wsl_containerlab.sh"
Write-Host "5. Then test:"
Write-Host "   bash run.sh synthetic --samples 120 --epochs 5"
Write-Host "   bash run.sh deploy"
Write-Host "   bash run.sh live --samples 20 --interval 5 --epochs 5"
