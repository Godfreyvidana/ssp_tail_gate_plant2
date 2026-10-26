@echo off
title Panel Vision - SSP Tailgate
color 0A

echo ============================================================
echo   Panel Vision Inspection System - SSP Tailgate
echo   Vidana Consulting Pvt Ltd
echo ============================================================
echo.

rem Change to the project directory
cd /d "d:\SSP_TAILGATE"

echo [INFO] Starting app_vision.py (Flask server on port 5000)...
echo [INFO] Browser will open automatically in 5 seconds.
echo.

rem Start the Flask app in a new window so it keeps running
start "Panel Vision Server" python app_vision.py

rem Wait 5 seconds for Flask to initialize before opening browser
timeout /t 5 /nobreak >nul

echo [INFO] Opening browser at http://127.0.0.1:5000 ...
start "" "http://127.0.0.1:5000"

echo.
echo [INFO] Server is running. Close the "Panel Vision Server" window to stop.
echo.
pause
