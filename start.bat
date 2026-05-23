@echo off
rem --------------------------------------------------
rem  Windows batch script to launch the project
rem  - creates/activates a virtual environment
rem  - installs dependencies from requirements.txt
rem  - starts the main Python module (main.py)
rem --------------------------------------------------
setlocal

rem --- Activate / create venv ---------------------------------
if exist venv\Scripts\activate.bat (
    echo Activating existing virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Creating new virtual environment in "venv"...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment. Exiting.
        exit /b 1
    )
    call venv\Scripts\activate.bat
)

rem --- Install dependencies ---------------------------------
echo Installing required Python packages...
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo Failed to install dependencies. Exiting.
    exit /b 1
)

rem --- Run the main application -----------------------------
echo Starting the main application...
python -m main

rem --------------------------------------------------
rem  End of script
rem --------------------------------------------------
pause
