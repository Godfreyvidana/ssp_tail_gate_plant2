@echo off
title Panel Vision - SSP Tailgate (Microservice Mode)
color 0A

echo ============================================================
echo   Panel Vision Inspection System - SSP Tailgate
echo   Vidana Consulting Pvt Ltd
echo   MICROSERVICE MODE: OCR on port 5001, Main App on port 5000
echo ============================================================
echo.

cd /d "%~dp0"

rem Kill stale instances
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000 ^| findstr LISTENING 2^>nul') do taskkill /F /PID %%a 2>nul
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5001 ^| findstr LISTENING 2^>nul') do taskkill /F /PID %%a 2>nul
taskkill /F /IM ffmpeg.exe 2>nul

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo [STEP 1] Starting OCR Microservice (port 5001)...
start "OCR Service - Port 5001" "%PYTHON_EXE%" ocr_service.py

echo [STEP 2] Waiting 8 seconds for PaddleOCR to initialise in OCR service...
timeout /t 8 /nobreak >nul 2>nul || ping -n 9 127.0.0.1 >nul

echo [STEP 3] Starting Main Application (port 5000)...
start "Panel Vision Server - Port 5000" "%PYTHON_EXE%" app_vision.py

echo [STEP 4] Waiting 5 seconds for Flask to start...
timeout /t 5 /nobreak >nul 2>nul || ping -n 6 127.0.0.1 >nul

echo [STEP 5] Opening browser...
start "" "http://127.0.0.1:5000"

echo.
echo [INFO] Both services are running:
echo        OCR Microservice : http://127.0.0.1:5001/health
echo        Main Application : http://127.0.0.1:5000
echo.
echo [INFO] Close both console windows to stop.
echo.
pause
