import sys
sys.path.insert(0, r"D:\PythonLibs")

libs = [
    "numpy", "cv2", "PIL", "ultralytics", "torch", "torchvision", 
    "torchaudio", "usearch", "sdtw_pytorch", "PySide6", "psutil", 
    "thop", "yaml", "matplotlib", "requests", "scipy", "polars",
    "packaging", "contourpy", "cycler", "fontTools", "kiwisolver",
    "pyparsing", "dateutil", "six", "charset_normalizer", "idna",
    "urllib3", "certifi", "filelock", "typing_extensions", "sympy",
    "networkx", "jinja2", "fsspec", "mpmath", "markupsafe", "pystray"
]

print("=" * 50)
print("ПРОВЕРКА БИБЛИОТЕК")
print("=" * 50)

found = []
missing = []
version_issues = []

# Проверка обычных библиотек
for lib in libs:
    try:
        module = __import__(lib)
        version = getattr(module, "__version__", "версия не указана")
        found.append(f"✓ {lib}: {version}")
    except ImportError:
        missing.append(f"✗ {lib}: НЕ НАЙДЕНА")

# Специальная проверка для torch с CUDA
try:
    import torch
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else "N/A"
    found.append(f"✓ torch CUDA: {cuda_available} (CUDA {cuda_version})")
    if "+cu" not in torch.__version__:
        version_issues.append(f"⚠ torch версия {torch.__version__} - возможно без CUDA!")
except:
    pass

# Специальная проверка для opencv
try:
    import cv2
    found.append(f"✓ cv2 (opencv): {cv2.__version__}")
except:
    pass

print("\n✅ НАЙДЕНЫ:")
for f in found:
    print(f"  {f}")

if version_issues:
    print("\n⚠️ ПРЕДУПРЕЖДЕНИЯ:")
    for v in version_issues:
        print(f"  {v}")

if missing:
    print("\n❌ ОТСУТСТВУЮТ:")
    for m in missing:
        print(f"  {m}")
else:
    print("\n🎉 ВСЕ БИБЛИОТЕКИ УСТАНОВЛЕНЫ!")