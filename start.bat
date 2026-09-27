@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1"
if errorlevel 1 (
  echo.
  echo Startup failed. Check the message above and logs in temp\launcher.
  pause
  exit /b 1
)
