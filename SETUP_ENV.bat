@echo off
setlocal
cd /d "%~dp0"
set "PIP_CACHE_DIR=%~dp0runtime_storage\pip_cache"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%~dp0src"

echo [1/4] Checking 64-bit CPython 3.10...
py -3.10 -c "import struct,sys; assert sys.version_info[:2]==(3,10) and struct.calcsize('P')*8==64" >nul 2>&1
if errorlevel 1 (
  echo ERROR: Install 64-bit CPython 3.10 and the Windows Python launcher first.
  echo Required version: Python 3.10.x 64-bit.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [2/4] Creating project-local environment...
  py -3.10 -m venv ".venv"
  if errorlevel 1 exit /b 1
) else (
  echo [2/4] Existing project-local environment found.
)

echo [3/4] Installing locked dependencies...
if exist "wheelhouse" (
  ".venv\Scripts\python.exe" -m pip install --no-index --find-links "wheelhouse" setuptools==75.8.2 wheel==0.45.1
  if errorlevel 1 exit /b 1
  ".venv\Scripts\python.exe" -m pip install --no-index --find-links "wheelhouse" -r "requirements-lock.txt"
) else (
  ".venv\Scripts\python.exe" -m pip install --no-cache-dir setuptools==75.8.2 wheel==0.45.1
  if errorlevel 1 exit /b 1
  ".venv\Scripts\python.exe" -m pip install --no-cache-dir -r "requirements-lock.txt"
)
if errorlevel 1 exit /b 1

echo [4/4] Verifying RC Fragility imports...
".venv\Scripts\python.exe" -c "import fragility_poc, numpy, scipy, sklearn, openseespy.opensees; print('RC Fragility environment import: PASS')"
if errorlevel 1 exit /b 1

echo Environment setup completed successfully.
exit /b 0
