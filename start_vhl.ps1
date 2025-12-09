# VHL Simulation Launcher
# This script starts the VHL WebGPU simulation in your browser

param(
    [switch]$WithAPI,
    [int]$Port = 8000
)

Write-Host "🌀 VHL Simulation Launcher" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Get the script's directory (repository root)
$RepoPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoPath

Write-Host "📁 Repository: $RepoPath" -ForegroundColor Green
Write-Host ""

# Check if git is available
if (Get-Command git -ErrorAction SilentlyContinue) {
    Write-Host "🔄 Pulling latest changes..." -ForegroundColor Yellow

    # Get current branch
    $CurrentBranch = git rev-parse --abbrev-ref HEAD
    Write-Host "   Branch: $CurrentBranch" -ForegroundColor Gray

    # Pull latest changes
    git pull origin $CurrentBranch

    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✓ Repository up to date" -ForegroundColor Green
    } else {
        Write-Host "   ⚠ Warning: Could not pull updates (continuing anyway)" -ForegroundColor Yellow
    }
    Write-Host ""
} else {
    Write-Host "⚠ Git not found - skipping update check" -ForegroundColor Yellow
    Write-Host ""
}

# Check if Python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Error: Python not found!" -ForegroundColor Red
    Write-Host "   Please install Python from https://www.python.org/" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

$PythonVersion = python --version
Write-Host "🐍 Python: $PythonVersion" -ForegroundColor Green
Write-Host ""

# Start Flask API if requested
if ($WithAPI) {
    Write-Host "🔧 Starting Flask API backend..." -ForegroundColor Cyan

    # Check if Flask is installed
    $FlaskInstalled = python -c "import flask" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "   Installing Flask dependencies..." -ForegroundColor Yellow
        pip install flask flask-cors pyscf
    }

    # Start Flask in background
    $FlaskJob = Start-Job -ScriptBlock {
        param($Path)
        Set-Location $Path
        python vhl_api.py
    } -ArgumentList $RepoPath

    Write-Host "   ✓ Flask API starting on http://localhost:5000" -ForegroundColor Green
    Write-Host "   Job ID: $($FlaskJob.Id)" -ForegroundColor Gray
    Write-Host ""

    # Wait a moment for Flask to start
    Start-Sleep -Seconds 2
}

# Check if port is already in use
$PortInUse = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
if ($PortInUse) {
    Write-Host "⚠ Warning: Port $Port is already in use" -ForegroundColor Yellow
    $Port = 8001
    Write-Host "   Using port $Port instead" -ForegroundColor Yellow
    Write-Host ""
}

# Start HTTP server
Write-Host "🌐 Starting HTTP server on port $Port..." -ForegroundColor Cyan

$URL = "http://localhost:$Port/vhl_webgpu.html"

# Start Python HTTP server in background
$ServerJob = Start-Job -ScriptBlock {
    param($Path, $Port)
    Set-Location $Path
    python -m http.server $Port
} -ArgumentList $RepoPath, $Port

Write-Host "   ✓ Server running at http://localhost:$Port" -ForegroundColor Green
Write-Host "   Job ID: $($ServerJob.Id)" -ForegroundColor Gray
Write-Host ""

# Wait for server to start
Start-Sleep -Seconds 2

# Open browser
Write-Host "🚀 Launching browser..." -ForegroundColor Cyan
Write-Host "   URL: $URL" -ForegroundColor Gray
Write-Host ""

Start-Process $URL

Write-Host "✅ VHL Simulation is now running!" -ForegroundColor Green
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "Controls:" -ForegroundColor Yellow
Write-Host "  • Mouse drag to rotate" -ForegroundColor White
Write-Host "  • Scroll to zoom" -ForegroundColor White
Write-Host "  • Right-click drag to pan" -ForegroundColor White
Write-Host ""
Write-Host "To stop the server:" -ForegroundColor Yellow
Write-Host "  Press Ctrl+C or close this window" -ForegroundColor White
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

# Keep script running and monitor jobs
try {
    Write-Host "📊 Server Status (Press Ctrl+C to stop)" -ForegroundColor Cyan
    Write-Host ""

    while ($true) {
        # Check if server job is still running
        $ServerState = (Get-Job -Id $ServerJob.Id).State

        if ($ServerState -ne "Running") {
            Write-Host "⚠ Server stopped unexpectedly!" -ForegroundColor Red
            break
        }

        # Show timestamp
        $Timestamp = Get-Date -Format "HH:mm:ss"
        Write-Host "`r[$Timestamp] Server running... (Ctrl+C to stop)" -NoNewline -ForegroundColor Gray

        Start-Sleep -Seconds 5
    }
} finally {
    # Cleanup on exit
    Write-Host ""
    Write-Host ""
    Write-Host "🛑 Shutting down..." -ForegroundColor Yellow

    if ($ServerJob) {
        Stop-Job -Id $ServerJob.Id -ErrorAction SilentlyContinue
        Remove-Job -Id $ServerJob.Id -ErrorAction SilentlyContinue
        Write-Host "   ✓ HTTP server stopped" -ForegroundColor Green
    }

    if ($FlaskJob) {
        Stop-Job -Id $FlaskJob.Id -ErrorAction SilentlyContinue
        Remove-Job -Id $FlaskJob.Id -ErrorAction SilentlyContinue
        Write-Host "   ✓ Flask API stopped" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "👋 Thanks for using VHL Simulation!" -ForegroundColor Cyan
}
