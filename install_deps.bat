@echo off
setlocal EnableDelayedExpansion

echo.
echo ==========================================
echo ParallelFinder - Dependency Installer
echo ==========================================
echo.

set "SCRIPT_DIR=%~dp0"
if "!SCRIPT_DIR:~-1!"=="\" set "SCRIPT_DIR=!SCRIPT_DIR:~0,-1!"

set "TORCH_MODE=%~1"
set "LIBS_PATH=%~2"
set "FAISS_MODE=%~3"
set "CUSTOM_TEMP=%~4"

if "!TORCH_MODE!"=="" set "TORCH_MODE=cpu"
if "!LIBS_PATH!"=="" set "LIBS_PATH=!SCRIPT_DIR!\pythonlibs"
if "!FAISS_MODE!"=="" set "FAISS_MODE=cpu"

if not "!CUSTOM_TEMP!"=="" (
    echo [*] Using custom TEMP: !CUSTOM_TEMP!
    if not exist "!CUSTOM_TEMP!\" (
        mkdir "!CUSTOM_TEMP!" 2>nul
        if errorlevel 1 (
            echo [ERROR] Failed to create TEMP folder: !CUSTOM_TEMP!
            goto :FAIL
        )
    )
    set "TEMP=!CUSTOM_TEMP!"
    set "TMP=!CUSTOM_TEMP!"
) else (
    echo [*] Using system TEMP: !TEMP!
)

echo [*] Torch mode : !TORCH_MODE!
echo [*] FAISS mode : !FAISS_MODE!
echo [*] Libs path  : !LIBS_PATH!
echo.

set "PYTHON_EXE="

where py >nul 2>&1
if !errorlevel! equ 0 (
    py -3 --version >nul 2>&1
    if !errorlevel! equ 0 set "PYTHON_EXE=py -3"
)

if "!PYTHON_EXE!"=="" (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        python --version >nul 2>&1
        if !errorlevel! equ 0 set "PYTHON_EXE=python"
    )
)

if "!PYTHON_EXE!"=="" (
    echo [ERROR] Python not found.
    goto :FAIL
)

echo [*] Using Python: !PYTHON_EXE!

if exist "!LIBS_PATH!\" (
    echo [*] Removing old dependencies folder...
    rmdir /s /q "!LIBS_PATH!" 2>nul
    if exist "!LIBS_PATH!\" (
        echo [ERROR] Failed to remove old folder: !LIBS_PATH!
        goto :FAIL
    )
)

mkdir "!LIBS_PATH!" 2>nul
if not exist "!LIBS_PATH!\" (
    echo [ERROR] Failed to create libs folder: !LIBS_PATH!
    goto :FAIL
)

:: Записываем путь один раз в начале
echo !LIBS_PATH!> "!SCRIPT_DIR!\pythonlibs_path.txt"
echo [*] Written pythonlibs_path.txt

echo.
echo [1/4] Installing Torch...

if /I "!TORCH_MODE!"=="cu128" (
    !PYTHON_EXE! -m pip install --target "!LIBS_PATH!" ^
        torch torchvision torchaudio ^
        --index-url https://download.pytorch.org/whl/cu128
) else if /I "!TORCH_MODE!"=="cu118" (
    !PYTHON_EXE! -m pip install --target "!LIBS_PATH!" ^
        torch torchvision torchaudio ^
        --index-url https://download.pytorch.org/whl/cu118
) else (
    !PYTHON_EXE! -m pip install --target "!LIBS_PATH!" ^
        torch torchvision torchaudio ^
        --index-url https://download.pytorch.org/whl/cpu
)

if errorlevel 1 (
    echo [ERROR] Torch installation failed.
    goto :FAIL
)

echo.
echo [2/4] Installing base packages...

!PYTHON_EXE! -m pip install --target "!LIBS_PATH!" ^
    numpy ^
    opencv-python ^
    pillow ^
    psutil

if errorlevel 1 (
    echo [ERROR] Base packages installation failed.
    goto :FAIL
)

echo.
echo [3/4] Installing ultralytics + tkinterdnd2...

!PYTHON_EXE! -m pip install --target "!LIBS_PATH!" ^
    ultralytics ^
    tkinterdnd2

if errorlevel 1 (
    echo [ERROR] ultralytics/tkinterdnd2 installation failed.
    goto :FAIL
)

echo.
echo [4/4] Installing optional packages...

if /I "!FAISS_MODE!"=="gpu" (
    echo [WARN] FAISS GPU selected, but Windows pip install is unstable.
    echo [WARN] Installing faiss-cpu instead.
)

!PYTHON_EXE! -m pip install --target "!LIBS_PATH!" ^
    faiss-cpu ^
    pystray

if errorlevel 1 (
    echo [WARN] Optional packages failed - continuing anyway.
)

echo.
echo ==========================================
echo Dependencies installed successfully
echo ==========================================
echo.
endlocal
exit /b 0

:FAIL
echo.
echo ==========================================
echo Installation failed
echo ==========================================
echo.
endlocal
exit /b 1