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
echo [INFO] Installing dependencies from requirements.txt...

python -m pip install --upgrade pip

pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo [OK] Dependencies installed/updated successfully.

rem --- Run the main application -----------------------------
echo Starting the main application...
python -m main

rem --------------------------------------------------
rem  End of script
rem --------------------------------------------------
pause
