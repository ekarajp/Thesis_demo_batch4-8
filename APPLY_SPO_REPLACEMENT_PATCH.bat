@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%~dp0;%~dp0src"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Existing server Python environment was not found.
  echo Run START_UI.bat once before applying this patch.
  pause
  exit /b 1
)

echo Installing checkpoint-safe SPO quarantine and replacement policy...
".venv\Scripts\python.exe" "install_spo_replacement_patch.py"
if errorlevel 1 (
  echo.
  echo PATCH FAILED. Keep the server paused and read the message above.
  pause
  exit /b 1
)

echo.
echo PATCH PASSED. Open START_UI.bat and press Start / Resume.
pause
exit /b 0
