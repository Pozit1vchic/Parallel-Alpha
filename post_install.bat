@echo off
cd /d "%~dp0"
call install_deps.bat "%~1" "%~2" "%~3" "%~4"
exit /b %errorlevel%