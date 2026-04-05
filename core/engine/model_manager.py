#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelManager — централизованное управление файлами YOLO-моделей.

Принципы:
- Все .pt файлы хранятся только в models/
- Если файл есть локально — сеть не трогаем
- Если нет — скачиваем через ultralytics
- Валидация имени по реестру
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional

# ── Пути ─────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR    = _PROJECT_ROOT / "models"

DEFAULT_MODEL_NAME = "yolo26x-pose.pt"

# ── Реестр моделей ────────────────────────────────────────────────────────────

AVAILABLE_MODELS: list[str] = [
    "yolov8n-pose.pt",
    "yolov8s-pose.pt",
    "yolov8m-pose.pt",
    "yolov8l-pose.pt",
    "yolov8x-pose.pt",
    "yolov8x-pose-p6.pt",
    "yolo11n-pose.pt",
    "yolo11s-pose.pt",
    "yolo11m-pose.pt",
    "yolo11l-pose.pt",
    "yolo11x-pose.pt",
    "yolo26n-pose.pt",
    "yolo26s-pose.pt",
    "yolo26m-pose.pt",
    "yolo26l-pose.pt",
    "yolo26x-pose.pt",
]

_AVAILABLE_SET: set[str] = set(AVAILABLE_MODELS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_cb(fn: Optional[Callable], value) -> None:
    """Безопасный вызов callback — никогда не падает."""
    if fn is not None:
        try:
            fn(value)
        except Exception:
            pass


def _find_ultralytics_cache(name: str) -> Optional[Path]:
    """
    Найти модель в кеше ultralytics.
    Сначала спрашиваем у самого ultralytics, потом угадываем пути.
    """
    # Способ 1: спросить у ultralytics напрямую
    try:
        from ultralytics.utils import SETTINGS
        weights_dir = Path(SETTINGS.get("weights_dir", ""))
        candidate   = weights_dir / name
        if candidate.is_file():
            return candidate
    except Exception:
        pass

    # Способ 2: стандартные пути кеша
    candidates = [
        Path.home() / ".cache" / "ultralytics" / name,
        Path.home() / ".cache" / "ultralytics" / "models" / name,
        # Windows
        Path.home() / "AppData" / "Roaming" / "ultralytics" / name,
        Path.home() / "AppData" / "Local"   / "ultralytics" / name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


# ── Основной класс ────────────────────────────────────────────────────────────

class ModelManager:
    """
    Управление файлами моделей.

    Публичный API:
      is_valid_name(name)    → bool
      is_local(name)         → bool
      list_local()           → list[str]
      get_model_path(name)   → Path
      prepare(name, ...)     → Path   ← главный метод
    """

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ── Валидация ─────────────────────────────────────────────────────────

    def is_valid_name(self, name: str) -> bool:
        return name in _AVAILABLE_SET

    def validate_name(self, name: str) -> str:
        """Вернуть name если валидно, иначе DEFAULT_MODEL_NAME."""
        if self.is_valid_name(name):
            return name
        print(
            f"[ModelManager] ⚠ Неизвестная модель '{name}'. "
            f"Используем: {DEFAULT_MODEL_NAME}"
        )
        return DEFAULT_MODEL_NAME

    # ── Пути ──────────────────────────────────────────────────────────────

    def get_model_path(self, name: str) -> Path:
        return self.models_dir / name

    def is_local(self, name: str) -> bool:
        return self.get_model_path(name).is_file()

    def list_local(self) -> list[str]:
        return [
            p.name for p in self.models_dir.glob("*.pt")
            if p.name in _AVAILABLE_SET
        ]

    # ── Подготовка ────────────────────────────────────────────────────────

    def prepare(
        self,
        name:        str,
        on_status:   Optional[Callable[[str],   None]] = None,
        on_progress: Optional[Callable[[float], None]] = None,
        on_source:   Optional[Callable[[bool],  None]] = None,
    ) -> Path:
        """
        Убедиться что модель есть локально.
        Если есть → сразу возвращает путь.
        Если нет  → скачивает через ultralytics.

        Raises RuntimeError если скачать не удалось.
        """
        name       = self.validate_name(name)
        local_path = self.get_model_path(name)

        # Уже есть локально
        if local_path.is_file():
            _safe_cb(on_source,   True)
            _safe_cb(on_status,   f"Локальная модель: {name}")
            _safe_cb(on_progress, 100.0)
            print(f"[ModelManager] ✓ Локально: {local_path}")
            return local_path

        # Нужно скачать
        _safe_cb(on_source,   False)
        _safe_cb(on_status,   f"Скачивается {name}…")
        _safe_cb(on_progress, 0.0)
        print(f"[ModelManager] ↓ Скачиваем {name} → {local_path}")

        try:
            from ultralytics import YOLO as _YOLO
            _safe_cb(on_progress, 10.0)

            # ultralytics скачивает в свой кеш
            _YOLO(name)
            _safe_cb(on_progress, 80.0)

            # Копируем из кеша в models/
            cached = _find_ultralytics_cache(name)
            if cached and cached.is_file():
                try:
                    shutil.copy2(cached, local_path)
                    print(f"[ModelManager] ✓ Скопировано: {cached} → {local_path}")
                except OSError as e:
                    raise RuntimeError(
                        f"Не удалось скопировать файл модели: {e}"
                    ) from e
            else:
                # ultralytics мог положить рядом с запуском
                alt = Path(name)
                if alt.is_file():
                    shutil.move(str(alt), str(local_path))
                    print(f"[ModelManager] ✓ Перемещено: {alt} → {local_path}")
                else:
                    # Не нашли файл, но модель загружена в память
                    # Продолжаем без локального кеша
                    print(
                        f"[ModelManager] ⚠ Файл не найден в кеше, "
                        f"продолжаем без кеширования."
                    )

            _safe_cb(on_progress, 100.0)
            _safe_cb(on_status,   f"Модель {name} готова")
            return local_path

        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"[ModelManager] Не удалось скачать '{name}': {exc}"
            ) from exc