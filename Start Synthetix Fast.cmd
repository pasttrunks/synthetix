@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Synthetix - Fast Baseline mode
echo  Faster, but frequently misses unfamiliar AI writing
echo ============================================================

set "PYCMD="
where py >nul 2>nul
if not errorlevel 1 set "PYCMD=py -3"
if not defined PYCMD (
    where python >nul 2>nul
    if not errorlevel 1 set "PYCMD=python"
)
if not defined PYCMD (
    for %%P in (
        "%LocalAppData%\Programs\Python\Python313\python.exe"
        "%LocalAppData%\Programs\Python\Python312\python.exe"
        "%LocalAppData%\Programs\Python\Python311\python.exe"
        "%LocalAppData%\Programs\Python\Python310\python.exe"
        "%ProgramFiles%\Python313\python.exe"
        "%ProgramFiles%\Python312\python.exe"
        "%ProgramFiles%\Python311\python.exe"
        "%ProgramFiles%\Python310\python.exe"
        "%ProgramFiles(x86)%\Python313-32\python.exe"
        "%ProgramFiles(x86)%\Python312-32\python.exe"
        "%ProgramFiles(x86)%\Python311-32\python.exe"
        "%ProgramFiles(x86)%\Python310-32\python.exe"
    ) do (
        if not defined PYCMD if exist "%%~P" set "PYCMD=%%~P"
    )
)
if not defined PYCMD (
    echo.
    echo ERROR: Python 3.10 or newer was not found.
    echo Install Python from https://www.python.org/downloads/ and make sure
    echo "Add python.exe to PATH" is checked, then run this launcher again.
    echo.
    pause
    exit /b 1
)

%PYCMD% launch_synthetix.py --backend hc3_roberta

set EXITCODE=%errorlevel%
echo.
if not "%EXITCODE%"=="0" (
    echo Synthetix exited with an error. See the messages above.
)
echo Close this window when done, or press any key to close now.
pause >nul
endlocal
