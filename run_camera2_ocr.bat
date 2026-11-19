@echo off
title Panel Vision - SSP Tailgate
color 0A

echo ============================================================
echo   Panel Vision Inspection System - SSP Tailgate
echo   Vidana Consulting Pvt Ltd
echo ============================================================
echo.

rem Change to the project directory
cd /d "%~dp0"

rem Clean up any previous stale python or ffmpeg instances
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000 ^| findstr LISTENING 2^>nul') do taskkill /F /PID %%a 2>nul
taskkill /F /IM ffmpeg.exe 2>nul

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo [INFO] Starting app_vision.py with %PYTHON_EXE% (Flask server on port 5000)...
echo [INFO] Browser will open automatically in 5 seconds.
echo.

rem Start the Flask app in a new window so it keeps running
start "Panel Vision Server" "%PYTHON_EXE%" app_vision.py

rem Wait 5 seconds for Flask to initialize before opening browser
timeout /t 5 /nobreak >nul 2>nul || ping -n 6 127.0.0.1 >nul

echo [INFO] Opening browser at http://127.0.0.1:5000 ...
start "" "http://127.0.0.1:5000"

echo.
echo [INFO] Server is running. Close the "Panel Vision Server" window to stop.
echo.
pause
