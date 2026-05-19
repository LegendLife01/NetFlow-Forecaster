param(
    [ValidateSet("synthetic", "simulate", "live", "deploy", "destroy", "train", "visualize")]
    [string]$Mode = "synthetic",
    [int]$Samples = 720,
    [int]$Interval = 1,
    [int]$Epochs = 120,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$MlDir = Join-Path $ProjectDir "ml"
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path (Join-Path $ProjectDir "runs") "$($RunStamp)_$Mode"
$DataFile = Join-Path $RunDir "telemetry.csv"

function Log-Step($Message) {
    Write-Host ""
    Write-Host "[$(Get-Date -Format HH:mm:ss)] $Message" -ForegroundColor Cyan
}

function Get-PythonCommand {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notlike "*WindowsApps*") {
        return "python"
    }

    $cmd = Get-Command py -ErrorAction SilentlyContinue
    if ($cmd) {
        return "py"
    }

    throw "Python was not found. Run .\scripts\setup_windows.ps1 -SkipDocker -SkipWsl, then reopen this terminal."
}

function ConvertTo-WslPath($Path) {
    $result = & wsl.exe wslpath -a $Path 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $result) {
        throw "WSL is not ready. Run .\scripts\setup_windows.ps1, open Ubuntu once, then run scripts/setup_wsl_containerlab.sh inside Ubuntu."
    }
    return ($result | Select-Object -First 1)
}

function Invoke-WslBash($Command) {
    & wsl.exe bash -lc $Command
    if ($LASTEXITCODE -ne 0) {
        throw "WSL command failed: $Command"
    }
}

function Test-WindowsDockerLab {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        return $false
    }
    $names = & docker ps --format "{{.Names}}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    return ($names -match "^clab-ai-traffic-lab-").Count -gt 0
}

function Test-WslDockerLab {
    $names = & wsl.exe -d Ubuntu -- bash -lc "docker ps --format '{{.Names}}'" 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    return ($names -match "^clab-ai-traffic-lab-").Count -gt 0
}

function Invoke-ContainerLab($Action) {
    if (Get-Command containerlab -ErrorAction SilentlyContinue) {
        Push-Location (Join-Path $ProjectDir "containerlab")
        & containerlab $Action -t topology.clab.yml
        Pop-Location
        if ($LASTEXITCODE -ne 0) {
            throw "containerlab $Action failed."
        }
        return
    }

    $wslProject = ConvertTo-WslPath $ProjectDir
    Invoke-WslBash "cd '$wslProject/containerlab' && sudo containerlab $Action -t topology.clab.yml"
}

function Invoke-ModelPipeline($Python, $InputCsv, $OutputDir) {
    Log-Step "Training LSTM"
    Push-Location $MlDir
    & $Python train_model.py --data $InputCsv --epochs $Epochs --output-dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        throw "Model training failed."
    }

    Log-Step "Building dashboard"
    & $Python visualize.py --data (Join-Path $OutputDir "telemetry.csv") --output-dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        throw "Dashboard generation failed."
    }
    Pop-Location
}

$NeedsPython = $Mode -in @("synthetic", "simulate", "live", "train", "visualize")
if ($NeedsPython) {
    $Python = Get-PythonCommand
    if (-not $SkipInstall) {
        Log-Step "Installing Python dependencies"
        & $Python -m pip install -r (Join-Path $ProjectDir "requirements.txt")
    }
}

switch ($Mode) {
    "synthetic" {
        New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
        Log-Step "Generating synthetic telemetry"
        Push-Location $MlDir
        & $Python generate_data.py --hours $Samples --output $DataFile --seed 7
        Pop-Location
        Invoke-ModelPipeline $Python $DataFile $RunDir
    }

    "simulate" {
        New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
        Log-Step "Collecting simulated telemetry with collector"
        Push-Location $ProjectDir
        & $Python scripts\collect_telemetry.py --mode simulate --samples $Samples --interval $Interval --output $DataFile
        if ($LASTEXITCODE -ne 0) {
            Pop-Location
            throw "Simulated telemetry collection failed."
        }
        Pop-Location
        Invoke-ModelPipeline $Python $DataFile $RunDir
    }

    "live" {
        New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
        Log-Step "Collecting live ContainerLab telemetry"
        if (Test-WindowsDockerLab) {
            Push-Location $ProjectDir
            & $Python scripts\collect_telemetry.py --mode live --samples $Samples --interval $Interval --output $DataFile
            if ($LASTEXITCODE -ne 0) {
                Pop-Location
                throw "Live telemetry collection failed."
            }
            Pop-Location
        } elseif (Test-WslDockerLab) {
            $wslProject = ConvertTo-WslPath $ProjectDir
            $wslDataFile = ConvertTo-WslPath $DataFile
            Invoke-WslBash "cd '$wslProject' && python3 scripts/collect_telemetry.py --mode live --samples $Samples --interval $Interval --output '$wslDataFile'"
        } else {
            throw "No running ContainerLab containers found. Run .\run.ps1 deploy first."
        }
        Invoke-ModelPipeline $Python $DataFile $RunDir
    }

    "deploy" {
        Log-Step "Deploying ContainerLab topology"
        Invoke-ContainerLab "deploy"
        return
    }

    "destroy" {
        Log-Step "Destroying ContainerLab topology"
        Invoke-ContainerLab "destroy"
        return
    }

    "train" {
        Log-Step "Training LSTM from existing telemetry"
        $ExistingData = Join-Path $MlDir "telemetry.csv"
        New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
        Invoke-ModelPipeline $Python $ExistingData $RunDir
    }

    "visualize" {
        Log-Step "Building dashboard from existing artifacts"
        $LatestRun = Get-ChildItem (Join-Path $ProjectDir "runs") -Directory -ErrorAction SilentlyContinue |
            Where-Object {
                (Test-Path (Join-Path $_.FullName "predictions.csv")) -and
                (Test-Path (Join-Path $_.FullName "actuals.csv")) -and
                (Test-Path (Join-Path $_.FullName "train_losses.csv"))
            } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if (-not $LatestRun) {
            throw "No run folders with readable CSV artifacts found. Run .\run.ps1 synthetic first."
        }
        $RunDir = $LatestRun.FullName
        $DataFile = Join-Path $RunDir "telemetry.csv"
        Push-Location $MlDir
        & $Python visualize.py --data $DataFile --output-dir $RunDir
        Pop-Location
    }
}

Log-Step "Done"
Write-Host "Artifacts:"
Write-Host "  $DataFile"
Write-Host "  $(Join-Path $RunDir 'lstm_model.pth')"
Write-Host "  $(Join-Path $RunDir 'traffic_prediction_dashboard.png')"
Write-Host "Run folder:"
Write-Host "  $RunDir"
