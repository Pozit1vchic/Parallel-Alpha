#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ui/controllers/foolproof.py — Система защиты от некорректных действий
"""
from __future__ import annotations

import os
import gc
import time
import shutil
from pathlib import Path
from typing import Callable

import tkinter as tk
from tkinter import messagebox

from utils.locales import t

# Расширения видео
VIDEO_EXT = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv",
    ".webm", ".m4v", ".ts", ".mts", ".m2ts", ".3gp",
    ".mpeg", ".mpg", ".ogv",
}

# Не показывать снова
_suppressed: set[str] = set()


def warn_once(key: str, message: str, parent=None) -> None:
    """Показать предупреждение с возможностью 'Больше не показывать'."""
    if key in _suppressed:
        return

    dlg = tk.Toplevel(parent)
    dlg.title(t("warning"))
    dlg.configure(bg="#0f0f1a")
    dlg.resizable(False, False)
    dlg.grab_set()
    if parent:
        dlg.transient(parent)
        px = parent.winfo_rootx() + 100
        py = parent.winfo_rooty() + 100
        dlg.geometry(f"360x160+{px}+{py}")

    tk.Label(
        dlg, text=message,
        bg="#0f0f1a", fg="#e2e8f0",
        font=("Segoe UI", 10),
        wraplength=320, justify="left",
    ).pack(padx=20, pady=(16, 8))

    var_suppress = tk.BooleanVar(value=False)
    tk.Checkbutton(
        dlg,
        text=t("dont_show_again"),
        variable=var_suppress,
        bg="#0f0f1a", fg="#64748b",
        selectcolor="#1e293b",
        activebackground="#0f0f1a",
    ).pack(padx=20, anchor="w")

    def _ok():
        if var_suppress.get():
            _suppressed.add(key)
        dlg.destroy()

    tk.Button(
        dlg, text=t("ok"),
        command=_ok,
        bg="#1d4ed8", fg="#ffffff",
        font=("Segoe UI", 10, "bold"),
        relief="flat", padx=20, pady=6,
        cursor="hand2",
    ).pack(pady=(8, 16))


class FoolProof:
    """
    Централизованная проверка всех пользовательских действий.
    """

    def __init__(self, root, state):
        self._root  = root
        self._state = state

    # ═══════════════════════════════════════════════════════════
    # 1. МОДЕЛИ
    # ═══════════════════════════════════════════════════════════

    def check_model_load(
        self,
        model_path: str | None = None,
    ) -> bool:
        """True = можно грузить."""
        if getattr(self._state, "model_loading", False):
            messagebox.showwarning(
                t("warning"), t("err_model_loading"), parent=self._root
            )
            return False

        if getattr(self._state, "analysis_running", False):
            messagebox.showwarning(
                t("warning"), t("err_model_during_analysis"), parent=self._root
            )
            return False

        if model_path and not os.path.exists(model_path):
            # Может быть — ещё не скачана, не блокируем
            pass

        return True

    def check_model_file(self, path: str) -> bool:
        """Проверить что .pt файл существует и не нулевой."""
        if not os.path.exists(path):
            messagebox.showerror(
                t("error"), t("err_model_not_found"), parent=self._root
            )
            return False
        if os.path.getsize(path) < 1024:
            messagebox.showerror(
                t("error"), t("err_model_corrupt"), parent=self._root
            )
            return False
        return True

    def check_disk_space(
        self,
        path: str,
        need_bytes: int = 200 * 1024 * 1024,
    ) -> bool:
        """Проверить место на диске."""
        try:
            free = shutil.disk_usage(path).free
            if free < need_bytes:
                messagebox.showerror(
                    t("error"), t("err_no_disk_space"), parent=self._root
                )
                return False
        except Exception:
            pass
        return True

    # ═══════════════════════════════════════════════════════════
    # 2. ВИДЕО
    # ═══════════════════════════════════════════════════════════

    def validate_video_file(self, path: str) -> tuple[bool, str]:
        """
        Проверить видеофайл.
        Returns: (ok, error_message)
        """
        p = Path(path)

        # Расширение
        if p.suffix.lower() not in VIDEO_EXT:
            return False, t("err_non_video_files")

        # Существование
        if not p.exists():
            return False, t("err_file_corrupt")

        # Размер
        try:
            size = p.stat().st_size
        except PermissionError:
            return False, t("err_no_access")

        if size == 0:
            return False, t("err_file_empty")

        # Очень большой файл
        size_gb = size / (1024 ** 3)
        if size_gb > 100:
            warn_once(
                f"large_{path}", 
                t("err_file_too_large", size=int(size_gb)),
                self._root,
            )

        # Длинный путь (Windows)
        if os.name == "nt" and len(str(p)) > 260:
            warn_once(
                f"long_path_{path}",
                t("err_path_too_long", n=260),
                self._root,
            )

        # Попытка открыть
        try:
            import cv2
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                cap.release()
                return False, t("err_file_corrupt")

            fps          = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width        = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height       = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()

            # Плохое разрешение
            if width <= 0 or height <= 0:
                return False, t("err_bad_resolution")

            # Нет видео потока
            if total_frames == 0:
                return False, t("err_no_video_stream")

            # Странный FPS — исправляем
            if fps <= 0 or fps > 240:
                pass  # будет принудительно 30fps

            # Короткое
            if fps > 0 and total_frames > 0:
                duration = total_frames / fps
                if duration < 1.0:
                    return False, t("err_video_too_short")
                if duration > 86400:  # 24 часа
                    warn_once(
                        f"long_video_{path}",
                        t("err_video_too_long", hours=24),
                        self._root,
                    )

        except Exception as e:
            return False, f"{t('err_file_corrupt')}: {e}"

        return True, ""

    def validate_folder(self, folder: str) -> tuple[bool, list[str]]:
        """
        Проверить папку на видео.
        Returns: (ok, list_of_valid_paths)
        """
        p = Path(folder)
        if not p.exists():
            messagebox.showerror(
                t("error"), t("err_folder_no_access"), parent=self._root
            )
            return False, []

        try:
            video_files = [
                str(f) for f in p.rglob("*")
                if f.suffix.lower() in VIDEO_EXT
            ]
        except PermissionError:
            messagebox.showerror(
                t("error"), t("err_folder_no_access"), parent=self._root
            )
            return False, []

        if not video_files:
            messagebox.showinfo(
                t("info"), t("err_folder_empty"), parent=self._root
            )
            return False, []

        # Слишком много видео
        if len(video_files) > 100:
            answer = messagebox.askyesno(
                t("warning"),
                t("err_too_many_videos", n=len(video_files)),
                parent=self._root,
            )
            if not answer:
                return False, []

        return True, video_files

    # ═══════════════════════════════════════════════════════════
    # 3. АНАЛИЗ
    # ═══════════════════════════════════════════════════════════

    def check_start_analysis(
        self,
        video_queue: list,
        yolo_loaded: bool,
    ) -> bool:
        """True = можно начинать."""
        if not video_queue:
            messagebox.showwarning(
                t("warning"), t("err_no_video_to_start"), parent=self._root
            )
            return False

        if not yolo_loaded:
            messagebox.showwarning(
                t("warning"), t("err_model_not_loaded"), parent=self._root
            )
            return False

        if getattr(self._state, "analysis_running", False):
            return False  # уже идёт

        return True

    def check_memory(self) -> bool:
        """Предупредить если мало памяти."""
        try:
            import psutil
            free_gb = psutil.virtual_memory().available / 1e9
            if free_gb < 2.0:
                warn_once(
                    "low_memory",
                    t("err_no_memory"),
                    self._root,
                )
        except Exception:
            pass
        return True

    def check_vram(self) -> str:
        """Вернуть 'cuda' или 'cpu' в зависимости от VRAM."""
        try:
            import torch
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory
                free = torch.cuda.memory_reserved(0)
                if vram < 2 * 1024 ** 3:
                    warn_once(
                        "low_vram", t("err_no_vram"), self._root
                    )
                    return "cpu"
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def check_settings(self, settings: dict) -> bool:
        """Предупреждения о нелогичных настройках."""
        threshold = settings.get("threshold", 80)
        if threshold <= 55:
            warn_once(
                "low_thr", t("err_threshold_low"), self._root
            )
        elif threshold >= 98:
            warn_once(
                "high_thr", t("err_threshold_high"), self._root
            )

        # Все опции включены
        if (settings.get("use_mirror") and settings.get("use_body_weights")):
            warn_once(
                "all_options", t("warn_all_options_on"), self._root
            )

        scene = settings.get("scene_interval", 3)
        gap   = settings.get("match_gap", 5)
        if gap < scene:
            warn_once(
                "gap_scene",
                "Мин. разрыв меньше интервала сцены — возможны дубли",
                self._root,
            )

        return True

    # ═══════════════════════════════════════════════════════════
    # 4. ЗАКРЫТИЕ
    # ═══════════════════════════════════════════════════════════

    def check_close(self) -> bool:
        """Спросить подтверждение если идёт анализ."""
        if getattr(self._state, "analysis_running", False):
            return messagebox.askyesno(
                t("warning"),
                t("err_close_during_analysis"),
                parent=self._root,
            )
        return True

    # ═══════════════════════════════════════════════════════════
    # 5. ЭКСПОРТ
    # ═══════════════════════════════════════════════════════════

    def check_export(self, matches: list, export_path: str | None = None) -> bool:
        if not matches:
            messagebox.showwarning(
                t("warning"), t("err_no_results_export"), parent=self._root
            )
            return False

        if export_path and os.path.exists(export_path):
            return messagebox.askyesno(
                t("warning"),
                t("err_file_overwrite"),
                parent=self._root,
            )

        return True

    # ═══════════════════════════════════════════════════════════
    # 6. ФОТО
    # ═══════════════════════════════════════════════════════════

    def check_photo(self, path: str) -> bool:
        IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        if Path(path).suffix.lower() not in IMAGE_EXT:
            messagebox.showerror(
                t("error"), t("err_photo_not_image"), parent=self._root
            )
            return False
        return True

    def check_photo_has_person(self, yolo, image_path: str) -> bool:
        """Проверить что на фото есть человек."""
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                return False
            result = yolo.detect_batch([img])
            if not result or result[0] is None:
                messagebox.showwarning(
                    t("warning"), t("err_photo_no_person"), parent=self._root
                )
                return False
        except Exception:
            pass
        return True