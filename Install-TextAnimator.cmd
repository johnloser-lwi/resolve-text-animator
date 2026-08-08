@echo off
rem Double-click helper: builds, then installs (prompting for elevation).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" -Build %*
