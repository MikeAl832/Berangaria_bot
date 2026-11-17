@echo off
rem --------------------------------------------------
rem  Windows launcher for Berangaria Bot
rem  Uses .venv (or venv), falls back to embedded local Qdrant
rem  when Docker Qdrant is not listening on 127.0.0.1:6333.
rem --------------------------------------------------
setlocal

cd /d "%~dp0.."

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Creating new virtual environment in ".venv"...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
)

python -c "import socket; s=socket.socket(); s.settimeout(0.5); exit(0 if s.connect_ex(('127.0.0.1', 6333)) == 0 else 1)" >nul 2>&1
if errorlevel 1 (
    if "%QDRANT_PATH%"=="" set "QDRANT_PATH=qdrant_local"
    echo [INFO] Docker Qdrant on 127.0.0.1:6333 is offline — using local embedded Qdrant in %QDRANT_PATH%
)

python -m berangaria
pause
