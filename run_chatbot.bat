@echo off
REM ==================== CHATBOT SERVICE WITH FULL LOGGING ====================
REM This batch file starts the chatbot backend with all detailed logs
REM All questions from users and backend processing are displayed in real-time
REM Logs show how the backend treats each question
REM All previous Python processes are cleaned up before starting

setlocal enabledelayedexpansion

cd /d "%~dp0"

REM Set window title
title Chatbot Service - Full Logging

echo.
echo ================================================================================
echo                    CHATBOT SERVICE STARTUP WITH FULL LOGGING
echo ================================================================================
echo.
echo [%date% %time%] Working directory: %CD%
echo [%date% %time%] Cleaning up previous backend processes...
echo.

REM Kill all existing Python processes to ensure clean startup
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo [OK] Previous processes cleaned up
echo.
echo [%date% %time%] Starting Chatbot Service...
echo.

REM Check if Python 3.11 is installed
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.11 is not installed
    echo [ERROR] Please install Python 3.11
    pause
    exit /b 1
)

echo [OK] Python 3.11 is available
py -3.11 --version

REM Check if backend.py exists
if not exist "backend.py" (
    echo [ERROR] backend.py not found
    pause
    exit /b 1
)

echo [OK] Backend file found: backend.py

REM Check if .env exists
if not exist ".env" (
    echo [ERROR] .env file not found
    pause
    exit /b 1
)

echo [OK] Environment file found: .env

echo.
echo ================================================================================
echo                    CHECKING DEPENDENCIES AND LAUNCHING
echo ================================================================================
echo.

REM Check if FastAPI is installed for Python 3.11
py -3.11 -m pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Installing missing dependencies from requirements.txt...
    py -3.11 -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed successfully
) else (
    echo [OK] All dependencies verified
)

echo.
echo ================================================================================
echo                    LAUNCHING CHATBOT SERVICE WITH LOGS
echo ================================================================================
echo.
echo [%date% %time%] All logs will be displayed in real-time below...
echo [%date% %time%] Press CTRL+C to stop the chatbot
echo.
echo Chatbot API URL: http://localhost:8000
echo.
echo ================================================================================
echo.

REM Set environment variables for full output
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1

REM Start the chatbot backend with uvicorn using Python 3.11 - all logs will display
REM This shows all user questions and how backend processes them
py -3.11 -m uvicorn backend:app --host 0.0.0.0 --port 8000 --reload --log-level debug --access-log

REM If backend exits, show message
echo.
echo ================================================================================
echo [%date% %time%] CHATBOT SERVICE STOPPED
echo ================================================================================
echo [INFO] Chatbot service has stopped. Check logs above for details.
echo.
echo [%date% %time%] Cleaning up processes...
taskkill /F /IM python.exe >nul 2>&1
echo [OK] All processes cleaned up
echo.
pause
