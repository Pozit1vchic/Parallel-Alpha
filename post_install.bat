@echo off
setlocal EnableDelayedExpansion

:: ============================================================
::  ParallelFinder Alpha v13 — post_install.bat
::
::  Called by the Inno Setup installer after file copying.
::  Forwards all arguments to install_deps.bat verbatim.
::
::  Arguments (positional, all optional):
::    %1  TORCH_MODE   cpu | cu118 | cu128
::    %2  LIBS_PATH    path to Python libs folder (quoted)
::    %3  FAISS_MODE   cpu | gpu
::    %4  TEMP_PATH    pip temp folder (quoted, or "" for system TEMP)
:: ============================================================

set "SCRIPT_DIR=%~dp0"
if "!SCRIPT_DIR:~-1!"=="\" set "SCRIPT_DIR=!SCRIPT_DIR:~0,-1!"

cd /d "!SCRIPT_DIR!"

echo [post_install] ParallelFinder Alpha v13
echo [post_install] Starting dependency installation...
echo [post_install] TORCH_MODE = %~1
echo [post_install] LIBS_PATH  = %~2
echo [post_install] FAISS_MODE = %~3
echo [post_install] TEMP_PATH  = %~4
echo.

:: Передаём аргументы корректно: %1 без кавычек (torch mode),
:: %2 уже в кавычках (libs path), %3 без кавычек (faiss mode),
:: %4 уже в кавычках или "" (temp path)
call "!SCRIPT_DIR!\install_deps.bat" %~1 %2 %~3 %4

:: Сохраняем код возврата ДО endlocal
set "RESULT=!errorlevel!"

if "!RESULT!"=="0" (
    echo [post_install] Installation completed successfully.
) else (
    echo [post_install] install_deps.bat returned error code !RESULT!
)

:: Передаём код возврата через endlocal
endlocal & exit /b %RESULT%