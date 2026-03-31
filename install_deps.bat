@echo off
setlocal

set MODE=%1
set LIBS_PATH=%~2

if "%MODE%"=="" set MODE=cpu
if "%LIBS_PATH%"=="" set LIBS_PATH=%~dp0pythonlibs

echo ============================
echo ParallelFinder dependency installer
echo Mode: %MODE%
echo Libs path: %LIBS_PATH%
echo ============================

if not exist "%LIBS_PATH%" mkdir "%LIBS_PATH%"

echo %LIBS_PATH%> "%~dp0pythonlibs_path.txt"

py -m pip install --upgrade pip setuptools wheel

echo Installing common packages...
py -m pip install --target "%LIBS_PATH%" ultralytics opencv-python numpy pillow psutil faiss-cpu

if /I "%MODE%"=="cu118" (
    echo Installing GPU version of torch CUDA 11.8...
    py -m pip install --target "%LIBS_PATH%" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else if /I "%MODE%"=="cu128" (
    echo Installing GPU version of torch CUDA 12.8...
    py -m pip install --target "%LIBS_PATH%" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
) else (
    echo Installing CPU version of torch...
    py -m pip install --target "%LIBS_PATH%" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

echo Done.
endlocal