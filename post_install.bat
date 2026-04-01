@echo off
setlocal EnableDelayedExpansion

:: ============================================================
::  ParallelFinder — post_install.bat
::
::  Called by the Inno Setup installer after file copying.
::  Forwards all arguments to install_deps.bat verbatim.
::
::  Arguments (positional, all optional):
::    %1  TORCH_MODE   cpu | cu118 | cu128
::    %2  LIBS_PATH    path to Python libs folder (quoted)
::    %3  FAISS_MODE   cpu | gpu
::    %4  TEMP_PATH    pip temp folder (quoted, or "" for system TEMP)
::
::  Exit code mirrors install_deps.bat exit code.
:: ============================================================

:: Resolve the directory this script lives in (no trailing backslash)
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

:: Change to the application directory so relative paths inside
:: install_deps.bat resolve correctly
cd /d "!SCRIPT_DIR!"

echo [post_install] Starting dependency installation...
echo [post_install] TORCH_MODE = %~1
echo [post_install] LIBS_PATH  = %~2
echo [post_install] FAISS_MODE = %~3
echo [post_install] TEMP_PATH  = %~4
echo.

:: ── Forward to install_deps.bat ──────────────────────────────
:: We pass all 4 arguments, each individually quoted.
:: Even if an argument is empty (""), install_deps.bat handles it gracefully.
call "!SCRIPT_DIR!\install_deps.bat" %1 %2 %3 %4

:: Capture and forward the exit code
set "RESULT=!errorlevel!"

if !RESULT! neq 0 (
    echo [post_install] install_deps.bat returned error code !RESULT!
) else (
    echo [post_install] Installation completed successfully.
)

endlocal
exit /b %RESULT%