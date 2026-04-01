@echo off
setlocal

set "TORCH_MODE=%~1"
set "LIBS_PATH=%~2"
set "FAISS_MODE=%~3"
set "CUSTOM_TEMP=%~4"

if "%TORCH_MODE%"=="" set "TORCH_MODE=cpu"
if "%LIBS_PATH%"=="" set "LIBS_PATH=%~dp0pythonlibs"
if "%FAISS_MODE%"=="" set "FAISS_MODE=cpu"

if not "%CUSTOM_TEMP%"=="" (
    if not exist "%CUSTOM_TEMP%" mkdir "%CUSTOM_TEMP%"
    set "TEMP=%CUSTOM_TEMP%"
    set "TMP=%CUSTOM_TEMP%"
)

echo ============================
echo ParallelFinder dependency installer
echo Torch mode: %TORCH_MODE%
echo Libs path: %LIBS_PATH%
echo FAISS mode: %FAISS_MODE%
echo Temp path: %TEMP%
echo ============================

if exist "%LIBS_PATH%" (
    echo Cleaning existing library folder...
    rmdir /s /q "%LIBS_PATH%"
)

mkdir "%LIBS_PATH%"
if errorlevel 1 (
    echo ERROR: Failed to create library folder.
    exit /b 1
)

echo %LIBS_PATH%> "%~dp0pythonlibs_path.txt"

where py >nul 2>nul
if %errorlevel%==0 (
    set "PY_CMD=py -3"
) else (
    set "PY_CMD=python"
)

%PY_CMD% -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip tools.
    exit /b 1
)

echo Installing torch first...
if /I "%TORCH_MODE%"=="cu118" (
    echo Installing GPU version of torch CUDA 11.8...
    %PY_CMD% -m pip install --target "%LIBS_PATH%" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else if /I "%TORCH_MODE%"=="cu128" (
    echo Installing GPU version of torch CUDA 12.8...
    %PY_CMD% -m pip install --target "%LIBS_PATH%" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
) else (
    echo Installing CPU version of torch...
    %PY_CMD% -m pip install --target "%LIBS_PATH%" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

if errorlevel 1 (
    echo ERROR: Torch installation failed.
    exit /b 1
)

echo Installing base dependencies...
%PY_CMD% -m pip install --target "%LIBS_PATH%" numpy pillow psutil opencv-python matplotlib pyyaml requests scipy polars packaging contourpy cycler fonttools kiwisolver pyparsing python-dateutil six charset_normalizer idna urllib3 certifi filelock typing-extensions sympy networkx jinja2 fsspec mpmath MarkupSafe
if errorlevel 1 (
    echo ERROR: Base dependency installation failed.
    exit /b 1
)

echo Installing FAISS...
if /I "%FAISS_MODE%"=="gpu" (
    echo FAISS GPU mode selected.
    echo NOTE: GPU FAISS on Windows is experimental. Installing faiss-cpu for compatibility.
    %PY_CMD% -m pip install --target "%LIBS_PATH%" faiss-cpu
) else (
    echo Installing FAISS CPU...
    %PY_CMD% -m pip install --target "%LIBS_PATH%" faiss-cpu
)

if errorlevel 1 (
    echo ERROR: FAISS installation failed.
    exit /b 1
)

echo Installing ultralytics-thop without dependencies...
%PY_CMD% -m pip install --target "%LIBS_PATH%" ultralytics-thop --no-deps
if errorlevel 1 (
    echo ERROR: ultralytics-thop installation failed.
    exit /b 1
)

echo Installing ultralytics without dependencies...
%PY_CMD% -m pip install --target "%LIBS_PATH%" ultralytics --no-deps
if errorlevel 1 (
    echo ERROR: ultralytics installation failed.
    exit /b 1
)

echo Done.
endlocal
exit /b 0