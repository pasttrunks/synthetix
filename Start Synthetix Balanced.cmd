@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Synthetix - Experimental Comparison mode
echo  Two detectors; uncertain outcome when they disagree
echo ============================================================

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 launch_synthetix.py --backend balanced_review
) else (
    python launch_synthetix.py --backend balanced_review
)

set EXITCODE=%errorlevel%
echo.
if not "%EXITCODE%"=="0" (
    echo Synthetix exited with an error. See the messages above.
)
echo Close this window when done, or press any key to close now.
pause >nul
endlocal
