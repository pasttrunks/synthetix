@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Synthetix - Academic Sensitive mode
echo  Slower, stronger recall, high false-positive risk
echo ============================================================

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 launch_synthetix.py --backend desklib_academic
) else (
    python launch_synthetix.py --backend desklib_academic
)

set EXITCODE=%errorlevel%
echo.
if not "%EXITCODE%"=="0" (
    echo Synthetix exited with an error. See the messages above.
)
echo Close this window when done, or press any key to close now.
pause >nul
endlocal
