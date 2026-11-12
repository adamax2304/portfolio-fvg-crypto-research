@echo off
REM ==========================================================================
REM  FVG Bot — MTF agent auto-start wrapper with watchdog restart loop
REM  Invoked by Windows on user logon via Startup folder shortcut,
REM  or manually. Loops forever: restarts the bot automatically after crash.
REM ==========================================================================

cd /d "C:\Users\adama\Documents\zeph ai projects\fvgbot"

:loop
REM Generate timestamp via PowerShell
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"`) do set TS=%%I
set LOG=multi_agent\logs\autostart_%TS%.log

REM Guard: don't start a second instance if one is already running.
powershell -NoProfile -Command "if (Get-CimInstance -Query \"SELECT ProcessId FROM Win32_Process WHERE (Name='python.exe' OR Name='py.exe') AND CommandLine LIKE '%%multi_agent%%main.py%%'\") { exit 1 } else { exit 0 }"
if errorlevel 1 (
    echo [watchdog] Bot already running at %TS% — waiting 60s... > "%LOG%"
    timeout /t 60 /nobreak >nul
    goto loop
)

echo [watchdog] Starting MTF bot at %TS% > "%LOG%"
py -3 multi_agent\main.py >> "%LOG%" 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo [watchdog] Bot exited (code=%EXIT_CODE%) at %TS%, restarting in 15s... >> "%LOG%"
timeout /t 15 /nobreak >nul
goto loop
