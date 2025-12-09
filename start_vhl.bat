@echo off
REM VHL Simulation Launcher - Windows Batch Version
REM Simple alternative to PowerShell script

echo ========================================
echo        VHL Simulation Launcher
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

echo Repository: %CD%
echo.

REM Check for git and pull updates
where git >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Pulling latest changes...
    git pull
    echo.
) else (
    echo Git not found - skipping update
    echo.
)

REM Check for Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found!
    echo Please install Python from https://www.python.org/
    echo.
    pause
    exit /b 1
)

python --version
echo.

echo Starting HTTP server on port 8000...
echo Server will run at http://localhost:8000
echo.

echo Opening browser...
timeout /t 2 /nobreak >nul
start http://localhost:8000/vhl_webgpu.html

echo.
echo ========================================
echo VHL Simulation is now running!
echo ========================================
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start Python HTTP server (blocking)
python -m http.server 8000
