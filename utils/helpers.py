#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/helpers.py — Вспомогательные функции и типы для Parallel Finder Alpha v13
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TypeAlias

import numpy as np

# ── Типы ─────────────────────────────────────────────────────────────────────

# Алиас для массивов float32 — используется в core и utils
ArrayF32: TypeAlias = np.ndarray

# Строковый тип для направления позы
Direction: TypeAlias = str

# Допустимые направления
VALID_DIRECTIONS: frozenset[str] = frozenset({
    "forward", "back", "left", "right",
    "forward-left", "forward-right",
    "back-left", "back-right",
    "unknown",
})


# ── Числа и время ─────────────────────────────────────────────────────────────

def compact_number(n: int | float) -> str:
    """Компактное представление числа: 1500 → '1.5K', 1_000_000 → '1.0M'."""
    n = float(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(int(n))


def format_time(seconds: float) -> str:
    """Форматировать секунды в HH:MM:SS."""
    s = max(0.0, float(seconds))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    ss = int(s % 60)
    return f"{h:02d}:{m:02d}:{ss:02d}"


def to_timecode(seconds: float, fps: float = 25.0) -> str:
    """Форматировать секунды в SMPTE таймкод HH:MM:SS:FF."""
    s   = max(0.0, float(seconds))
    fps = max(1.0, float(fps))
    h   = int(s // 3600)
    m   = int((s % 3600) // 60)
    ss  = int(s % 60)
    ff  = int(round((s % 1) * fps))
    ff  = min(ff, int(fps) - 1)
    return f"{h:02d}:{m:02d}:{ss:02d}:{ff:02d}"


# ── Файловые операции ────────────────────────────────────────────────────────

def get_file_hash(path: str | Path, chunk: int = 65536) -> str:
    """MD5 хэш файла (первые 64KB для скорости)."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            data = f.read(chunk)
            if data:
                h.update(data)
    except OSError:
        h.update(str(path).encode())
    return h.hexdigest()


def normalize_path(path: str) -> str:
    """Нормализовать путь (кириллица, спецсимволы, обратные слеши)."""
    try:
        return str(Path(path).resolve())
    except Exception:
        return path


def truncate_path(path: str, max_len: int = 50) -> str:
    """Сократить длинный путь для отображения."""
    if len(path) <= max_len:
        return path
    parts = Path(path).parts
    if len(parts) <= 2:
        return path[:max_len] + "…"
    return str(Path(parts[0]) / "…" / parts[-1])


# ── Направление / поза ───────────────────────────────────────────────────────

_DIRECTION_EMOJI: dict[str, str] = {
    "forward":       "👤",
    "back":          "🔙",
    "left":          "👈",
    "right":         "👉",
    "forward-left":  "↖",
    "forward-right": "↗",
    "back-left":     "↙",
    "back-right":    "↘",
    "unknown":       "❓",
}

_DIRECTION_STR_RU: dict[str, str] = {
    "forward":       "Лицом к камере",
    "back":          "Спиной к камере",
    "left":          "Смотрит влево",
    "right":         "Смотрит вправо",
    "forward-left":  "Пол-оборота влево",
    "forward-right": "Пол-оборота вправо",
    "back-left":     "Спиной-влево",
    "back-right":    "Спиной-вправо",
    "unknown":       "Неизвестно",
}

_DIRECTION_STR_EN: dict[str, str] = {
    "forward":       "Facing camera",
    "back":          "Facing away",
    "left":          "Facing left",
    "right":         "Facing right",
    "forward-left":  "Half-turn left",
    "forward-right": "Half-turn right",
    "back-left":     "Back-left",
    "back-right":    "Back-right",
    "unknown":       "Unknown",
}


def direction_to_emoji(direction: str) -> str:
    """Вернуть эмодзи для направления."""
    return _DIRECTION_EMOJI.get(direction, "❓")


def direction_to_string(direction: str, lang: str = "ru") -> str:
    """Вернуть читаемое название направления."""
    if lang == "en":
        return _DIRECTION_STR_EN.get(direction, "Unknown")
    return _DIRECTION_STR_RU.get(direction, "Неизвестно")


# ── Работа с позами ──────────────────────────────────────────────────────────

def normalize_pose(keypoints: ArrayF32) -> ArrayF32:
    """
    Нормализовать ключевые точки позы:
    - центрировать относительно центра масс
    - масштабировать к единичному размаху

    Parameters
    ----------
    keypoints : ArrayF32
        Массив формы (N, 2) или (N, 3) — x, y [, conf]

    Returns
    -------
    ArrayF32
        Нормализованные точки той же формы.
    """
    kps = np.asarray(keypoints, dtype=np.float32)
    if kps.ndim != 2 or kps.shape[1] < 2:
        return kps

    xy = kps[:, :2].copy()

    # Центрирование
    center = xy.mean(axis=0)
    xy -= center

    # Масштабирование
    scale = np.abs(xy).max()
    if scale > 1e-6:
        xy /= scale

    result = kps.copy()
    result[:, :2] = xy
    return result


# ── UI-утилиты ───────────────────────────────────────────────────────────────

def numpy_to_qpixmap(arr: ArrayF32):
    """
    Конвертировать numpy RGB-массив в QPixmap (если доступен PyQt/PySide).
    Возвращает None если Qt не установлен.
    """
    try:
        from PIL import Image
        img = Image.fromarray(arr.astype(np.uint8))
        # Попробовать PyQt5
        try:
            from PyQt5.QtGui import QPixmap, QImage
            from PyQt5.QtCore import Qt
            data = img.tobytes("raw", "RGB")
            qimg = QImage(data, img.width, img.height,
                          3 * img.width, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg)
        except ImportError:
            pass
        # Попробовать PySide6
        try:
            from PySide6.QtGui import QPixmap, QImage
            data = img.tobytes("raw", "RGB")
            qimg = QImage(data, img.width, img.height,
                          3 * img.width, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg)
        except ImportError:
            pass
    except Exception:
        pass
    return None


# ── Прочее ───────────────────────────────────────────────────────────────────

def clamp(value: float, lo: float, hi: float) -> float:
    """Ограничить значение в диапазоне [lo, hi]."""
    return max(lo, min(hi, value))


def safe_int(value, default: int = 0) -> int:
    """Безопасное преобразование в int."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value, default: float = 0.0) -> float:
    """Безопасное преобразование в float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default