@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%~dp0src"

if /I "%CD:~0,2%"=="C:" (
  echo ERROR: This research package must be unpacked on a non-C drive.
  echo Move the whole folder to D:, E:, or another data drive and run again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  call "SETUP_ENV.bat"
  if errorlevel 1 (
    echo Environment setup failed.
    pause
    exit /b 1
  )
)

echo Preparing portable paths and verifying ground-motion files...
".venv\Scripts\python.exe" "prepare_portable_runtime.py" --root "%~dp0"
if errorlevel 1 (
  echo Portable-path preparation failed.
  pause
  exit /b 1
)

echo Running fail-closed scientific preflight...
".venv\Scripts\python.exe" "preflight_portable.py" --root "%~dp0"
if errorlevel 1 (
  echo Preflight is BLOCKED. Read runtime\preflight.json.
  pause
  exit /b 1
)

echo Opening dashboard at http://127.0.0.1:8765
start "" "http://127.0.0.1:8765"
".venv\Scripts\python.exe" "server_dashboard.py" --host 127.0.0.1 --port 8765
exit /b %errorlevel%
