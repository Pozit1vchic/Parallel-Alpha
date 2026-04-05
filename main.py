#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — точка входа Parallel Finder Alpha v13.

Порядок запуска:
  1. _bootstrap_sys_path()  — добавляет pythonlibs_path.txt в sys.path (если есть)
  2. _print_startup_banner() — выводит метаданные приложения в консоль
  3. main()                 — импортирует и запускает ParallelFinderApp
"""

from __future__ import annotations

import sys
import os
import traceback
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Корень проекта — папка, где лежит этот файл.
# Добавляем в sys.path до любых локальных импортов, чтобы гарантировать
# правильное разрешение пакетов core / ui / utils даже при нестандартном cwd.
# ─────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap: pythonlibs_path.txt
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_sys_path() -> None:
    """
    Читает pythonlibs_path.txt рядом с main.py и добавляет
    указанный путь в начало sys.path.

    Файл опционален: если он отсутствует или повреждён — продолжаем
    без него, не падаем.
    """
    libs_file = _PROJECT_ROOT / "pythonlibs_path.txt"

    if not libs_file.exists():
        return

    try:
        raw = libs_file.read_text(encoding="utf-8", errors="ignore")
        # Убираем нулевые байты и пробельные символы
        libs_path = raw.strip().replace("\x00", "").strip()

        if not libs_path:
            return

        resolved = Path(libs_path).expanduser().resolve()

        if not resolved.exists():
            print(
                f"[main] ПРЕДУПРЕЖДЕНИЕ: pythonlibs_path.txt указывает "
                f"на несуществующий путь: {resolved}"
            )
            return

        path_str = str(resolved)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            print(f"[main] sys.path расширен: {resolved}")

    except OSError as exc:
        print(f"[main] Не удалось прочитать pythonlibs_path.txt: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"[main] Ошибка bootstrap sys.path: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Баннер запуска
# ─────────────────────────────────────────────────────────────────────────────

def _print_startup_banner() -> None:
    """
    Выводит в консоль метаданные приложения при запуске.
    Данные подтягиваются из utils/constants.py.
    Если импорт не удался — выводим минимальный fallback.
    """
    try:
        from utils.constants import (
            APP_DISPLAY_NAME,
            APP_BUILD_VERSION,
            APP_AUTHOR,
        )
        display_name  = APP_DISPLAY_NAME
        build_version = APP_BUILD_VERSION
        author        = APP_AUTHOR
    except ImportError:
        display_name  = "Parallel Finder Alpha v13"
        build_version = "v13.0.2.1"
        author        = "Pozit1vchic"

    try:
        import torch
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            device_info = f"CUDA  ({gpu_name})"
        else:
            device_info = "CPU   (CUDA недоступен)"
    except ImportError:
        device_info = "неизвестно (torch не установлен)"

    sep = "─" * 54
    print(f"\n{sep}")
    print(f"  {display_name}")
    print(f"  Версия  : {build_version}")
    print(f"  Автор   : {author}")
    print(f"  Устройство: {device_info}")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Главная функция
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """
    Точка входа приложения.

    Returns
    -------
    int
        0 — успешное завершение,
        1 — ошибка запуска.
    """
    # 1. Расширяем sys.path из pythonlibs_path.txt (если файл есть)
    _bootstrap_sys_path()

    # 2. Баннер в консоль
    _print_startup_banner()

    # 3. Импорт главного окна
    #    Делаем здесь (не на уровне модуля), чтобы bootstrap уже отработал
    #    и чтобы ошибка импорта давала понятное сообщение.
    try:
        from ui.main_window import ParallelFinderApp  # noqa: PLC0415
    except ImportError as exc:
        print(
            f"\n[main] КРИТИЧЕСКАЯ ОШИБКА: не удалось импортировать ParallelFinderApp.\n"
            f"  Причина : {exc}\n"
            f"  Проверьте:\n"
            f"    • наличие файла ui/main_window.py\n"
            f"    • корректность pythonlibs_path.txt\n"
            f"    • установку зависимостей (pip install -r requirements.txt)\n"
        )
        traceback.print_exc()
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"\n[main] Неожиданная ошибка при импорте: {exc}")
        traceback.print_exc()
        return 1

    # 4. Запуск
    try:
        app = ParallelFinderApp()
        app.run()
        return 0

    except KeyboardInterrupt:
        print("\n[main] Приложение остановлено пользователем (Ctrl+C).")
        return 0

    except Exception as exc:  # noqa: BLE001
        print(f"\n[main] ОШИБКА во время выполнения: {exc}")
        traceback.print_exc()
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# Точка входа
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(main())
