@echo off
REM ============================================================
REM  Burn Polygon Editor - launcher
REM
REM  Edit the two lines below once, then just double-click this
REM  file any time you want to open the editor.
REM ============================================================

REM --- 1) Path to your Python virtual environment folder ---
SET VENV_PATH=D:\NahidW\Coding\seg_env

REM --- 2) Path to your dataset root (the folder containing PAT01, PAT02, ...) ---
SET DATASET_PATH=D:\NahidW\Dataset\face_burn_dataset

REM --- (optional) change the port if 5050 is already used by something else ---
SET PORT=5050

REM ============================================================
REM  You shouldn't need to edit anything below this line.
REM ============================================================

cd /d "%~dp0"

SET PYEXE=%VENV_PATH%\Scripts\python.exe

if not exist "%PYEXE%" (
    echo.
    echo Could not find python.exe inside your virtual environment at:
    echo   %PYEXE%
    echo Check the VENV_PATH line at the top of this file.
    pause
    exit /b 1
)

echo Checking required packages...
"%PYEXE%" -m pip install --quiet flask pillow

echo.
echo Starting Burn Polygon Editor...
echo Dataset: %DATASET_PATH%
echo A browser tab will open automatically. Close this window to stop the server.
echo.

"%PYEXE%" run_with_picker.py --dataset "%DATASET_PATH%" --port %PORT%

pause