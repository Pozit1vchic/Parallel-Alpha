#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ui/main_window.py — Parallel Finder"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import psutil
import torch
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import messagebox

from core.engine import YoloEngine

# ── Классификатор движений ────────────────────────────────────────────────────
try:
    from core.motion_classifier import (
        MotionClassifier as _MotionClassifier,
        CATEGORY_LABELS  as _CAT_LABELS,
    )
    _motion_clf     = _MotionClassifier()
    _HAS_CLASSIFIER = True
except ImportError:
    _motion_clf     = None
    _HAS_CLASSIFIER = False
    _CAT_LABELS     = {}

try:
    from core.analysis_backend import AnalysisBackend
    _HAS_BACKEND = True
except ImportError:
    _HAS_BACKEND = False

try:
    from core.project import ProjectManager
    _HAS_PROJECT = True
except ImportError:
    _HAS_PROJECT = False

try:
    from utils.constants import (
        APP_DISPLAY_NAME, APP_SHORT_VERSION, APP_BUILD_VERSION,
        APP_AUTHOR, MODELS_DIR, DEFAULT_MODEL_NAME,
        YOLO_AVAILABLE_MODELS, VIDEO_EXTENSIONS,
        is_video_file, get_model_path, is_model_local, list_local_models,
    )
except ImportError:
    APP_DISPLAY_NAME   = "Parallel Finder"
    APP_SHORT_VERSION  = "13.4.5"
    APP_BUILD_VERSION  = "v13.4.5"
    APP_AUTHOR         = "Pozit1vchic"
    MODELS_DIR         = "models"
    DEFAULT_MODEL_NAME = "yolo11x-pose.pt"
    YOLO_AVAILABLE_MODELS: list[str] = [
        "yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt",
        "yolo11l-pose.pt", "yolo11x-pose.pt",
        "yolo26n-pose.pt", "yolo26s-pose.pt", "yolo26m-pose.pt",
        "yolo26l-pose.pt", "yolo26x-pose.pt",
    ]
    VIDEO_EXTENSIONS = (".mp4",".avi",".mkv",".mov",
                        ".ts",".webm",".flv",".m4v",".wmv")

    def is_video_file(path: str) -> bool:
        return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS

    def get_model_path(name: str):
        return Path(MODELS_DIR) / name

    def is_model_local(name: str) -> bool:
        return os.path.exists(os.path.join(MODELS_DIR, name))

    def list_local_models() -> list[str]:
        if not os.path.isdir(MODELS_DIR):
            return []
        return [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]

try:
    from utils.locales import (
        SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE,
        get_translator,
    )
except ImportError:
    SUPPORTED_LANGUAGES = ["ru", "en"]
    DEFAULT_LANGUAGE    = "ru"
    def get_translator(lang="ru"):
        return lambda k, **kw: k

try:
    import pystray
    from PIL import Image as PILImage
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False

from ui.app_state import AppState
from ui.themes import THEMES
from ui.overlays.model_loading import ModelLoadingOverlay
from ui.panels.stats_bar import StatsBar
from ui.panels.source_panel import SourcePanel
from ui.panels.settings_panel import SettingsPanel
from ui.panels.preview_panel import PreviewPanel
from ui.panels.results_panel import ResultsPanel
from ui.controllers.analysis_controller import AnalysisController
from ui.controllers.model_controller import ModelController
from ui.controllers.export_controller import ExportController
from ui.controllers.navigation_controller import NavigationController
from ui.widgets.glow_button import GlowButton

warnings.filterwarnings("ignore")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


# ── Конфиг ────────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "config.json"


class ConfigManager:
    @staticmethod
    def load() -> dict:
        try:
            return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def save(data: dict) -> None:
        try:
            _CONFIG_PATH.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def get_language(cfg: dict) -> str:
        lang = cfg.get("language", {}).get("current", DEFAULT_LANGUAGE)
        return lang if lang in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE

    @staticmethod
    def set_language(lang: str) -> None:
        cfg = ConfigManager.load()
        cfg.setdefault("language", {})["current"] = lang
        ConfigManager.save(cfg)

    @staticmethod
    def get_model(cfg: dict) -> str:
        return cfg.get("models", {}).get(
            "default_model", DEFAULT_MODEL_NAME)

    @staticmethod
    def set_model(model_name: str) -> None:
        cfg = ConfigManager.load()
        cfg.setdefault("models", {})["default_model"] = model_name
        ConfigManager.save(cfg)


# ── Утилиты ───────────────────────────────────────────────────────────────────

def _fmt_hms(secs: float) -> str:
    s  = max(0.0, float(secs))
    h, rem = divmod(int(s), 3600)
    m, ss  = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{ss:02d}"


def _fmt_num(n: int) -> str:
    return f"{n / 1000:.1f}K" if n >= 1000 else str(n)


def _build_model_list() -> list[str]:
    local     = set(list_local_models())
    all_m     = list(YOLO_AVAILABLE_MODELS)
    for lm in sorted(local):
        if lm not in all_m:
            all_m.insert(0, lm)
    return ([m for m in all_m if m in local]
            + [m for m in all_m if m not in local])


# ── UI строки ─────────────────────────────────────────────────────────────────

_UI_STRINGS: dict[str, dict[str, str]] = {
    "models_btn":  {"ru": "Модели",     "en": "Models"},
    "no_model":    {"ru": "Нет модели", "en": "No model"},
    "done_found":  {"ru": "Готово. Найдено", "en": "Done. Found"},
    "repeats":     {"ru": "повторов",   "en": "repeats"},
    "no_repeats":  {
        "ru": "Повторений не найдено. Снизьте порог схожести",
        "en": "No repeats found. Lower the similarity threshold",
    },
}


def _S(key: str, lang: str) -> str:
    return _UI_STRINGS.get(key, {}).get(lang, key)


# ── Направления ───────────────────────────────────────────────────────────────

_DIRECTION_LABELS: dict[str, dict[str, str]] = {
    "forward":       {"ru": "↑ Лицом к камере",    "en": "↑ Facing camera"},
    "left":          {"ru": "← Смотрит влево",      "en": "← Facing left"},
    "right":         {"ru": "→ Смотрит вправо",     "en": "→ Facing right"},
    "back":          {"ru": "↓ Спиной к камере",    "en": "↓ Facing away"},
    "forward-right": {"ru": "↗ Пол-оборота вправо", "en": "↗ Half-turn right"},
    "forward-left":  {"ru": "↖ Пол-оборота влево",  "en": "↖ Half-turn left"},
    "back-right":    {"ru": "↘ Спиной-вправо",      "en": "↘ Back-right"},
    "back-left":     {"ru": "↙ Спиной-влево",       "en": "↙ Back-left"},
    "unknown":       {"ru": "? Неизвестно",          "en": "? Unknown"},
}


def _direction_label(direction: str, lang: str) -> str:
    return _DIRECTION_LABELS.get(
        direction, _DIRECTION_LABELS["unknown"]
    ).get(lang, direction)


# ── Кэш кадров ────────────────────────────────────────────────────────────────

class FrameCache:
    def __init__(self, cache_dir: str, limit: int = 200) -> None:
        self._dir   = Path(cache_dir)
        self._limit = limit
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, path: str, frame: int, w: int, h: int) -> Path:
        vh = hashlib.md5(path.encode()).hexdigest()
        return self._dir / f"{vh}_{frame}_{w}x{h}.jpg"

    def get(self, path: str, frame: int,
            w: int, h: int) -> np.ndarray | None:
        p = self._key(path, frame, w, h)
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img
        return None

    def put(self, path: str, frame: int, w: int, h: int,
            img: np.ndarray) -> None:
        p = self._key(path, frame, w, h)
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self._evict()

    def _evict(self) -> None:
        files = sorted(self._dir.glob("*.jpg"),
                       key=lambda f: f.stat().st_mtime)
        for old in files[: max(0, len(files) - self._limit)]:
            try:
                old.unlink()
            except OSError:
                pass


# ── Классификатор матчей ──────────────────────────────────────────────────────

class MatchClassifier:
    """Классифицирует список матчей."""

    def __init__(self, lang: str = "ru") -> None:
        self.lang = lang

    def classify_all(self, matches: list[dict]) -> list[str]:
        """
        Классифицирует все матчи.
        Возвращает список локализованных меток категорий
        (для отображения в UI).
        Также записывает в каждый матч:
          m["_cat_key"]  — внутренний ключ категории
          m["category"]  — локализованная метка
        """
        seen:    set[str]  = set()
        ordered: list[str] = []

        for m in matches:
            cat_key, label = self._classify_one(m)
            if label not in seen:
                seen.add(label)
                ordered.append(label)

        return ordered

    def relabel_all(self, matches: list[dict]) -> list[str]:
        """
        Перелокализовать уже классифицированные матчи
        при смене языка. Использует сохранённый _cat_key.
        """
        seen:    set[str]  = set()
        ordered: list[str] = []

        for m in matches:
            cat_key = m.get("_cat_key", "")
            lang    = self.lang

            if cat_key and _HAS_CLASSIFIER:
                label = _CAT_LABELS.get(
                    cat_key, {}).get(lang, cat_key)
            else:
                # Направление — перелокализуем
                direction = m.get("direction", "unknown")
                label     = _direction_label(direction, lang)

            # Обновляем отображаемые поля
            dir_lbl  = _direction_label(
                m.get("direction", "unknown"), lang)
            movement = _CAT_LABELS.get(
                cat_key, {}).get(lang, cat_key) if cat_key else dir_lbl

            m["category"]  = label
            m["movement"]  = movement
            m["dir_label"] = dir_lbl

            if label not in seen:
                seen.add(label)
                ordered.append(label)

        return ordered

    def _classify_one(self, m: dict) -> tuple[str, str]:
        """
        Возвращает (cat_key, localized_label).
        Сохраняет cat_key в m["_cat_key"].
        """
        direction = m.get("direction", "unknown")
        lang      = self.lang

        if _HAS_CLASSIFIER and _motion_clf is not None:
            try:
                kp_raw = m.get("kp1") or m.get("kp2")

                if kp_raw is not None:
                    kp_arr = np.array(kp_raw, dtype=float)

                    if kp_arr.ndim == 1:
                        n = len(kp_arr)
                        if n == 51:
                            kp_arr = kp_arr.reshape(17, 3)
                        elif n == 34:
                            kp_arr = np.hstack([
                                kp_arr.reshape(17, 2),
                                np.ones((17, 1))
                            ])
                        else:
                            raise ValueError(
                                f"Неожиданная форма kp: {n}")

                    elif kp_arr.ndim == 2:
                        if kp_arr.shape[1] == 2:
                            kp_arr = np.hstack([
                                kp_arr,
                                np.ones((kp_arr.shape[0], 1))
                            ])
                        elif kp_arr.shape[1] != 3:
                            raise ValueError(
                                f"Неожиданное число столбцов: "
                                f"{kp_arr.shape[1]}")

                    else:
                        raise ValueError(
                            f"Неожиданная размерность: "
                            f"{kp_arr.ndim}")

                    if kp_arr.shape[0] < 17:
                        raise ValueError(
                            f"Мало точек: {kp_arr.shape[0]}")

                    kp_arr = kp_arr[:17]
                    res    = _motion_clf.classify(kp_arr, lang)

                    cat_key  = res["category"]
                    movement = res["movement"]
                    dir_lbl  = _direction_label(direction, lang)
                    label    = _CAT_LABELS.get(
                        cat_key, {}).get(lang, cat_key)

                    m["_cat_key"]  = cat_key
                    m["category"]  = label
                    m["movement"]  = movement
                    m["dir_label"] = dir_lbl

                    return cat_key, label

                # kp_raw is None
                lbl = _direction_label(direction, lang)
                m["_cat_key"]  = ""
                m["category"]  = lbl
                m["movement"]  = lbl
                m["dir_label"] = lbl
                return "", lbl

            except Exception as ex:
                print(f"[Classify] {ex}")
                lbl = _direction_label(direction, lang)
                m["_cat_key"]  = ""
                m["category"]  = lbl
                m["dir_label"] = lbl
                return "", lbl

        # Нет классификатора
        lbl = _direction_label(direction, lang)
        m["_cat_key"]  = ""
        m["category"]  = lbl
        m["movement"]  = lbl
        m["dir_label"] = lbl
        return "", lbl

# ── Диалог выбора модели ──────────────────────────────────────────────────────

class ModelSelectorDialog:
    ROW_H = 44

    def __init__(self, parent: tk.Tk, colors: dict,
                 lang: str, current_model: str,
                 on_apply) -> None:
        self._parent   = parent
        self._c        = colors
        self._lang     = lang
        self._current  = current_model
        self._on_apply = on_apply
        self._models   = _build_model_list()
        self._local    = set(list_local_models())
        self._selected = tk.IntVar(value=0)

        try:
            self._selected.set(
                self._models.index(current_model))
        except ValueError:
            pass

        self._dlg = self._build()

    _STRINGS: dict[str, dict[str, str]] = {
        "title":     {"ru": "Выбор модели",    "en": "Select Model"},
        "legend":    {
            "ru": "✓ — загружена локально  |  ↓ — требует скачивания",
            "en": "✓ — installed locally  |  ↓ — needs download",
        },
        "installed": {"ru": "Установлено",     "en": "Installed"},
        "available": {"ru": "Доступно",        "en": "Available"},
        "active":    {"ru": "● активна",       "en": "● active"},
        "download":  {"ru": "↓ скачать",       "en": "↓ download"},
        "apply":     {"ru": "Применить",       "en": "Apply"},
        "cancel":    {"ru": "Отмена",          "en": "Cancel"},
    }

    def _s(self, key: str) -> str:
        return self._STRINGS.get(key, {}).get(self._lang, key)

    def _build(self) -> tk.Toplevel:
        c   = self._c
        dlg = tk.Toplevel(self._parent)
        dlg.title(self._s("title"))
        dlg.geometry("520x560")
        dlg.configure(bg=c["bg"])
        dlg.transient(self._parent)
        dlg.grab_set()
        dlg.resizable(False, False)

        dlg.update_idletasks()
        x = (self._parent.winfo_x()
             + (self._parent.winfo_width() - 520) // 2)
        y = (self._parent.winfo_y()
             + (self._parent.winfo_height() - 560) // 2)
        dlg.geometry(f"520x560+{x}+{y}")

        tk.Label(dlg, text=self._s("title"),
                 font=("Inter", 14, "bold"),
                 bg=c["bg"], fg=c["text"]).pack(pady=(20, 4))

        tk.Label(dlg, text=self._s("legend"),
                 font=("Inter", 9),
                 bg=c["bg"],
                 fg=c["text_secondary"]).pack(pady=(0, 10))

        local_c = len(self._local & set(self._models))
        counter = (f"{self._s('installed')}: {local_c}  |  "
                   f"{self._s('available')}: {len(self._models)}")
        tk.Label(dlg, text=counter,
                 font=("Inter", 8),
                 bg=c["bg"],
                 fg=c.get("text_secondary", "#6b7280")).pack(
            pady=(0, 6))

        outer = tk.Frame(dlg, bg=c["card"],
                         highlightbackground=c["border"],
                         highlightthickness=1)
        outer.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

        self._cv = tk.Canvas(outer, bg=c["card"],
                             highlightthickness=0)
        sb = tk.Scrollbar(outer, orient=tk.VERTICAL,
                          command=self._cv.yview)
        self._cv.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._local_count = sum(
            1 for m in self._models if m in self._local)

        self._cv.bind("<Configure>", lambda _: self._draw())
        self._cv.bind("<Button-1>",  self._on_click)
        self._cv.bind("<MouseWheel>", self._on_scroll)
        self._cv.bind("<Button-4>",
                      lambda _: (self._cv.yview_scroll(-1, "units"),
                                 self._draw()))
        self._cv.bind("<Button-5>",
                      lambda _: (self._cv.yview_scroll(1, "units"),
                                 self._draw()))

        dlg.bind("<Up>",     lambda _: self._move(-1))
        dlg.bind("<Down>",   lambda _: self._move(+1))
        dlg.bind("<Return>", lambda _: self._apply())
        dlg.focus_set()

        btn_row = tk.Frame(dlg, bg=c["bg"])
        btn_row.pack(fill=tk.X, padx=20, pady=(0, 16))

        GlowButton(btn_row,
                   text=self._s("apply"),
                   command=self._apply,
                   bg_color=c["accent"],
                   hover_color=c["accent_hover"],
                   height=36,
                   font=("Inter", 10, "bold")).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        GlowButton(btn_row,
                   text=self._s("cancel"),
                   command=dlg.destroy,
                   bg_color=c["highlight"],
                   hover_color=c["border"],
                   height=36,
                   font=("Inter", 10)).pack(
            side=tk.LEFT, fill=tk.X, expand=True)

        dlg.after(80, lambda: self._ensure_visible(
            self._selected.get()))
        return dlg

    def _draw(self) -> None:
        c      = self._c
        cv     = self._cv
        models = self._models
        local  = self._local

        cv.delete("all")
        cw = cv.winfo_width() or 460
        n  = len(models)
        if n == 0:
            return

        cv.configure(scrollregion=(0, 0, cw, n * self.ROW_H))
        vt       = cv.yview()[0]
        total_h  = n * self.ROW_H
        y_off    = vt * total_h
        ch_      = cv.winfo_height() or 350
        si       = max(0, int(y_off // self.ROW_H) - 1)
        ei       = min(n, si + int(ch_ // self.ROW_H) + 3)

        for i in range(si, ei):
            name   = models[i]
            y      = i * self.ROW_H - y_off
            is_loc = name in local
            is_sel = i == self._selected.get()
            is_cur = name == self._current

            bg_row = self._row_color(is_sel, is_loc, i)
            cv.create_rectangle(0, y, cw, y + self.ROW_H - 1,
                                fill=bg_row, outline="")

            if (i == self._local_count
                    and 0 < self._local_count < n):
                cv.create_line(0, y, cw, y,
                               fill=c.get("accent", "#3b82f6"),
                               width=1, dash=(4, 4))

            icon = "✓" if is_loc else "↓"
            icon_col = (c.get("success", "#22c55e") if is_loc
                        else c.get("text_secondary", "#6b7280"))
            cv.create_text(20, y + self.ROW_H // 2,
                           text=icon, fill=icon_col,
                           font=("Inter", 12, "bold"),
                           anchor="center")

            name_col = (c.get("text", "#fff") if is_sel
                        else (c.get("text", "#e2e8f0") if is_loc
                              else c.get("text_secondary", "#6b7280")))
            weight = "bold" if (is_sel or is_cur) else "normal"
            cv.create_text(42, y + self.ROW_H // 2,
                           text=name.replace(".pt", ""),
                           fill=name_col,
                           font=("Inter", 10, weight),
                           anchor="w")

            if is_cur:
                cv.create_text(cw - 12, y + self.ROW_H // 2,
                               text=self._s("active"),
                               fill=c.get("accent", "#3b82f6"),
                               font=("Inter", 9), anchor="e")
            elif not is_loc:
                cv.create_text(cw - 12, y + self.ROW_H // 2,
                               text=self._s("download"),
                               fill=c.get("text_secondary", "#6b7280"),
                               font=("Inter", 9), anchor="e")

            cv.create_line(0, y + self.ROW_H - 1,
                           cw, y + self.ROW_H - 1,
                           fill=c.get("border", "#2a2f38"))

    def _row_color(self, is_sel: bool,
                   is_loc: bool, i: int) -> str:
        c = self._c
        if is_sel:
            return c.get("active_row", "#1e3a5f")
        if is_loc:
            return c.get("card", "#14171c")
        return (c.get("highlight", "#1e293b") if i % 2 == 0
                else c.get("card", "#14171c"))

    def _on_click(self, e: tk.Event) -> None:
        vt      = self._cv.yview()[0]
        total_h = len(self._models) * self.ROW_H
        abs_y   = e.y + vt * total_h
        i       = int(abs_y // self.ROW_H)
        if 0 <= i < len(self._models):
            self._selected.set(i)
            self._draw()

    def _on_scroll(self, e: tk.Event) -> None:
        self._cv.yview_scroll(
            -1 if e.delta > 0 else 1, "units")
        self._draw()

    def _move(self, delta: int) -> None:
        cur = self._selected.get()
        nxt = max(0, min(len(self._models) - 1, cur + delta))
        self._selected.set(nxt)
        self._ensure_visible(nxt)

    def _ensure_visible(self, i: int) -> None:
        total_h = len(self._models) * self.ROW_H
        if total_h == 0:
            return
        ch_  = self._cv.winfo_height() or 350
        top  = i * self.ROW_H
        bot  = top + self.ROW_H
        vt   = self._cv.yview()[0] * total_h
        vb   = self._cv.yview()[1] * total_h
        if top < vt:
            self._cv.yview_moveto(top / max(total_h, 1))
        elif bot > vb:
            self._cv.yview_moveto((bot - ch_) / max(total_h, 1))
        self._draw()

    def _apply(self) -> None:
        i = self._selected.get()
        if 0 <= i < len(self._models):
            chosen = self._models[i]
            self._dlg.destroy()
            if chosen != self._current:
                self._on_apply(chosen)


# ══════════════════════════════════════════════════════════════════════════════
class ParallelFinderApp:
    """Главное окно Parallel Finder."""

    def __init__(self) -> None:
        self.root      = self._create_root()
        self._cfg      = ConfigManager.load()
        self._lang     = ConfigManager.get_language(self._cfg)
        self.colors    = THEMES["dark"]
        self.state     = AppState()

        self._configure_root()
        self.root.withdraw()
        self.root.update_idletasks()
        self.root.deiconify()
        self._init_model_name()
        self._init_backend()
        self._init_subsystems()

        self._overlay      = ModelLoadingOverlay(self.root, self.colors)
        self._overlay = ModelLoadingOverlay(
            self.root, self.colors, lang=self._lang)
        self._fullscreen   = False
        self._tray_icon    = None
        self._child_procs: list[subprocess.Popen] = []


        self._build_ui()
        self._init_controllers()
        self._setup_drag_drop()

        self._nav_ctrl.setup_hotkeys(
            self.root,
            on_start         = self._analysis_ctrl.start,
            on_fullscreen    = self._toggle_fullscreen,
            on_exit_fullscreen = self._exit_fullscreen)

        self._create_tray()
        self.root.protocol("WM_DELETE_WINDOW", self._hide_to_tray)

        try:
            cfg_q = self._analysis_ctrl.auto_tune()
            self._settings_panel.set_quality(cfg_q["quality"])
        except Exception:
            pass

        self.root.after(0, self._start_model_load)

    # ── Инициализация ─────────────────────────────────────────────────────

    def _create_root(self) -> tk.Tk:
        try:
            if _DND_AVAILABLE:
                return TkinterDnD.Tk()
        except Exception:
            pass
        return tk.Tk()

    def _configure_root(self) -> None:
        self.root.attributes("-alpha", 0.0)
        ui = self._cfg.get("ui", {})
        self.root.title(APP_DISPLAY_NAME)
        ui = self._cfg.get("ui", {})
        self.root.title(APP_DISPLAY_NAME)
        self.root.geometry(
            f"{ui.get('window_width', 1600)}x"
            f"{ui.get('window_height', 960)}")
        self.root.minsize(1400, 800)
        self.root.configure(bg=self.colors["bg"])

        icon = Path(__file__).parent.parent / "icons" / "icon.ico"
        if icon.exists():
            try:
                self.root.iconbitmap(str(icon))
            except Exception:
                pass
        self.root.after(50, lambda: self.root.attributes("-alpha", 1.0))

    def _init_model_name(self) -> None:
        saved        = ConfigManager.get_model(self._cfg)
        local_models = list_local_models()
        if saved in local_models:
            self.state.current_model_name = saved
        elif local_models:
            self.state.current_model_name = local_models[0]
        else:
            self.state.current_model_name = DEFAULT_MODEL_NAME

    def _init_backend(self) -> None:
        self.yolo    = YoloEngine()
        self.backend = None
        if _HAS_BACKEND:
            try:
                device       = "cuda" if torch.cuda.is_available() else None
                self.backend = AnalysisBackend()
                self.backend.yolo = self.yolo
            except Exception as e:
                print(f"[Backend] {e}")

    def _init_subsystems(self) -> None:
        self.project = None
        if _HAS_PROJECT:
            try:
                from core.project import ProjectManager
                self.project = ProjectManager()
            except Exception:
                pass

        self._frame_cache = FrameCache(
            str(Path(__file__).parent.parent / "cache" / "previews"),
            limit=200)
        self._classifier = MatchClassifier(self._lang)

    def _start_model_load(self) -> None:
        try:
            name = self.state.current_model_name
            # Показываем оверлей сразу до отрисовки основного UI
            self._overlay.show(
                title      = "Подготавливаем модель распознавания поз",
                subtitle   = "Подготавливаем файлы модели",
                model_name = name)
            # Принудительно отрисовываем оверлей до загрузки модели
            self.root.update_idletasks()
            self._model_ctrl.load_model(name)
        except Exception as e:
            print(f"[ModelLoad] {e}")

    # ── Сборка UI ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        c = self.colors
        self._build_topbar(c)

        main = tk.Frame(self.root, bg=c["bg"])
        main.pack(fill=tk.BOTH, expand=True, padx=28, pady=(6, 28))

        self._stats_bar = StatsBar(main, c, lang=self._lang)
        self._stats_bar.pack(fill=tk.X)

        content = tk.Frame(main, bg=c["bg"])
        content.pack(fill=tk.BOTH, expand=True, pady=12)

        left = tk.Frame(content, bg=c["bg"], width=400)
        left.pack(side=tk.LEFT, fill=tk.Y,
                  padx=(0, 10), expand=False)
        left.pack_propagate(False)

        center = tk.Frame(content, bg=c["bg"])
        center.pack(side=tk.LEFT, fill=tk.BOTH,
                    expand=True, padx=5)

        right = tk.Frame(content, bg=c["bg"], width=400)
        right.pack(side=tk.RIGHT, fill=tk.Y,
                   padx=(10, 0), expand=False)
        right.pack_propagate(False)

        self._build_left_panels(left)
        self._build_center_panel(center)
        self._build_right_panel(right)

    def _build_topbar(self, c: dict) -> None:
        topbar = tk.Frame(self.root, bg=c["bg"], height=44)
        topbar.pack(fill=tk.X, padx=28, pady=(12, 0))
        topbar.pack_propagate(False)

        tk.Label(topbar,
                 text="PARALLEL  ALPHA",
                 font=("Inter", 11, "bold"),
                 bg=c["bg"],
                 fg=c["text_secondary"]).pack(
            side=tk.LEFT, pady=8)

        self._lang_btn = GlowButton(
            topbar,
            text=self._lang.upper(),
            command=self._toggle_language,
            bg_color=c["highlight"],
            hover_color=c["border"],
            width=48, height=28,
            font=("Inter", 9, "bold"))
        self._lang_btn.pack(side=tk.RIGHT, padx=(6, 0), pady=8)

        self._model_lbl = tk.Label(
            topbar,
            text=self._model_short_label,
            font=("Inter", 9),
            bg=c["bg"],
            fg=c["accent"],
            cursor="hand2")
        self._model_lbl.pack(side=tk.RIGHT, padx=(0, 4), pady=8)
        self._model_lbl.bind("<Button-1>",
                             lambda _: self._open_model_menu())

        self._models_btn = GlowButton(
            topbar,
            text=_S("models_btn", self._lang),
            command=self._open_model_menu,
            bg_color=c["highlight"],
            hover_color=c["border"],
            width=80, height=28,
            font=("Inter", 9, "bold"))
        self._models_btn.pack(side=tk.RIGHT, padx=(6, 2), pady=8)

    def _build_left_panels(self, parent: tk.Frame) -> None:
        self._source_panel = SourcePanel(
            parent, self.state, self.colors,
            callbacks={"on_queue_changed": self._on_queue_changed},
            video_extensions=VIDEO_EXTENSIONS,
            is_video_file_fn=is_video_file,
            lang=self._lang)
        self._source_panel.pack(fill=tk.X)

        self._settings_panel = SettingsPanel(
            parent, self.state, self.colors,
            callbacks={
                "on_start": lambda: self._analysis_ctrl.start(),
                "on_stop":  lambda: self._analysis_ctrl.stop(),
                "on_queue_cleared": self._on_queue_cleared,
            },
            lang=self._lang)
        self._settings_panel.pack(fill=tk.BOTH, expand=True)

    def _build_center_panel(self, parent: tk.Frame) -> None:
        self._preview_panel = PreviewPanel(
            parent, self.state, self.colors,
            callbacks={
                "on_resize":         self._on_preview_resize,
                "on_timeline_click": self._on_timeline_click,
            })
        self._preview_panel.pack(fill=tk.BOTH, expand=True)

    def _build_right_panel(self, parent: tk.Frame) -> None:
        self._results_panel = ResultsPanel(
            parent, self.state, self.colors,
            callbacks={
                "on_prev":        lambda: self._nav_ctrl.prev_match(),
                "on_next":        lambda: self._nav_ctrl.next_match(),
                "on_select":      self._on_result_select,
                "on_export_json": lambda: self._export_ctrl.export_json(),
                "on_export_txt":  lambda: self._export_ctrl.export_txt(),
                "on_export_edl":  lambda: self._export_ctrl.export_edl(),
            },
            lang=self._lang)
        self._results_panel.pack(fill=tk.BOTH, expand=True)

    # ── Контроллеры ───────────────────────────────────────────────────────

    def _init_controllers(self) -> None:
        vars_ = self._settings_panel.get_vars()

        self._analysis_ctrl = AnalysisController(
            root             = self.root,
            state            = self.state,
            yolo             = self.yolo,
            backend          = self.backend if _HAS_BACKEND else None,
            on_progress      = self._on_progress,
            on_complete      = self._on_analysis_complete,
            on_batch_status  = self._on_batch_status)
        self._analysis_ctrl.bind_vars(
            threshold        = vars_["threshold"],
            scene_interval   = vars_["scene_interval"],
            match_gap        = vars_["match_gap"],
            quality          = vars_["quality"],
            use_scale_inv    = vars_["use_scale_invariance"],
            use_mirror_inv   = vars_["use_mirror_invariance"],
            use_body_weights = vars_["use_body_weights"])

        self._model_ctrl = ModelController(
            root               = self.root,
            state              = self.state,
            yolo               = self.yolo,
            overlay            = self._overlay,
            all_models         = YOLO_AVAILABLE_MODELS,
            available_models   = YOLO_AVAILABLE_MODELS,
            is_model_local_fn  = is_model_local,
            get_model_path_fn  = get_model_path,
            on_model_ready     = self._on_model_ready)
        self._model_ctrl.set_colors(self.colors)

        self._export_ctrl = ExportController(
            state              = self.state,
            app_display_name   = APP_DISPLAY_NAME,
            app_short_version  = APP_SHORT_VERSION,
            app_build_version  = APP_BUILD_VERSION,
            app_author         = APP_AUTHOR)

        self._nav_ctrl = NavigationController(
            root             = self.root,
            state            = self.state,
            on_show_preview  = self._show_preview)

    # ── Свойства ──────────────────────────────────────────────────────────

    @property
    def _model_short_label(self) -> str:
        name = self.state.current_model_name
        icon = "✓" if is_model_local(name) else "↓"
        return f"{icon} {name.replace('.pt', '')}"

    # ── Model selector ────────────────────────────────────────────────────

    def _open_model_menu(self) -> None:
        ModelSelectorDialog(
            parent        = self.root,
            colors        = self.colors,
            lang          = self._lang,
            current_model = self.state.current_model_name,
            on_apply      = self._on_model_selected,
        )

    def _on_model_selected(self, model_name: str) -> None:
        self._models_btn.set_disabled(True)
        self._model_ctrl.load_model(model_name)

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _on_model_ready(self, model_name: str) -> None:
        self.state.current_model_name = model_name
        ConfigManager.set_model(model_name)
        try:
            color = (self.colors["success"]
                     if is_model_local(model_name)
                     else self.colors["accent"])
            self._model_lbl.config(
                text=self._model_short_label, fg=color)
            self._models_btn.set_disabled(False)
        except Exception:
            pass

    def _on_queue_changed(self) -> None:
        self.state.reset()
        self._results_panel.hide_results()
        try:
            self._preview_panel._cached_durations = []
            self._preview_panel.timeline_canvas.delete("all")
        except Exception:
            pass
        for key, val in [("patterns", "0"), ("accuracy", "0%"),
                         ("frames", "0"), ("duration", "00:00:00"),
                         ("time_left", "--:--:--")]:
            self._stats_bar.set(key, val)
        self._results_panel.update_counter(0, 0)
        self.root.title(APP_DISPLAY_NAME)

    def _on_queue_cleared(self) -> None:
        """Сброс после очистки очереди из панели настроек."""
        try:
            self._source_panel.refresh_ui()
        except Exception:
            pass
        self._on_queue_changed()

    def _on_progress(self, pct: float, status: str = "") -> None:
        try:
            self._preview_panel.set_progress(pct, status)

            # Длительность — из кэша один раз
            if self.state.video_queue:
                total_dur = sum(
                    self._get_duration(p)
                    for p in self.state.video_queue)
                self._stats_bar.set(
                    "duration", _fmt_hms(total_dur))

                # Кадров — плавно, пропорционально прогрессу
                if pct < 100:
                    total_frames = sum(
                        self._get_frame_count(p)
                        for p in self.state.video_queue)
                    current_frames = int(total_frames * pct / 100)
                    self._stats_bar.set(
                        "frames", _fmt_num(current_frames))

                # Осталось — оценка по прогрессу
                if 0 < pct < 100:
                    ratio     = pct / 100
                    remaining = (total_dur
                                 * (1 - ratio)
                                 / max(ratio, 0.01))
                    self._stats_bar.set(
                        "time_left", _fmt_hms(remaining))
                elif pct >= 100:
                    self._stats_bar.set("time_left", "00:00:00")

        except Exception as e:
            print(f"[Progress] {e}")

    def _on_batch_status(self, path: str,
                         idx: int, total: int) -> None:
        try:
            self._preview_panel.set_status(
                f"{idx + 1}/{total}: {os.path.basename(path)}")
            self._stats_bar.set(
                "batch_progress", f"{idx + 1}/{total}")
        except Exception:
            pass

    def _on_analysis_complete(self, matches: list) -> None:
        try:
            self.state.matches          = matches
            self.state.analysis_running = False

            self._settings_panel.set_analysis_state(False)
            self._models_btn.set_disabled(False)
            self._preview_panel.stop_animation()

            if matches:
                self._process_matches(matches)
            else:
                self._preview_panel.set_status(
                    _S("no_repeats", self._lang))
                self._results_panel.hide_results()

            self._preview_panel.set_progress(100)
            self._stats_bar.set("time_left", "00:00:00")

            # Финальное обновление кадров
            if self.state.video_queue:
                total_frames = sum(
                    self._get_frame_count(p)
                    for p in self.state.video_queue)
                self._stats_bar.set(
                    "frames", _fmt_num(total_frames))
                total_dur = sum(
                    self._get_duration(p)
                    for p in self.state.video_queue)
                self._stats_bar.set(
                    "duration", _fmt_hms(total_dur))

        except Exception as e:
            print(f"[AnalysisComplete] {e}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def _process_matches(self, matches: list) -> None:
        self._classifier.lang = self._lang
        self.state.found_categories = self._classifier.classify_all(
            matches)

        self._results_panel.show_results(
            matches, self.state.found_categories)

        sims = [m.get("sim", 0.0) for m in matches]
        avg  = float(np.mean(sims)) * 100

        self._stats_bar.set("patterns", str(len(matches)))
        self._stats_bar.set("accuracy", f"{avg:.1f}%")
        self._results_panel.update_counter(0, len(matches))
        self._show_preview(0)

        # ── Переводимая строка статуса ────────────────────────────────
        try:
            self._preview_panel.set_done_status(len(matches))
        except Exception:
            self._preview_panel.set_status(
                f"{_S('done_found', self._lang)} "
                f"{len(matches)} "
                f"{_S('repeats', self._lang)}")

        self._redraw_heatmap()

    def _on_result_select(self, match_idx: int) -> None:
        try:
            self._show_preview(match_idx)
        except Exception as e:
            print(f"[ResultSelect] {e}")

    def _on_preview_resize(self) -> None:
        if self.state.matches:
            self.root.after_idle(
                lambda: self._show_preview(
                    self.state.current_match))
        if self.state.video_queue:
            self.root.after_idle(self._redraw_heatmap)

    def _on_timeline_click(self, event: tk.Event) -> None:
        try:
            if not self.state.matches:
                return
            cw = self._preview_panel.timeline_canvas.winfo_width()
            if cw <= 0:
                return

            durations = [self._get_duration(p)
                         for p in self.state.video_queue]
            total_dur = sum(durations)
            if total_dur <= 0:
                return

            view_start = self.state.timeline_pan * total_dur
            view_span  = total_dur / self.state.timeline_zoom
            clicked_t  = (view_start
                          + (event.x / cw) * view_span)
            starts     = np.cumsum([0.0] + durations[:-1])

            best_i = min(
                range(len(self.state.matches)),
                key=lambda i: min(
                    abs(clicked_t
                        - (starts[self.state.matches[i]["v1_idx"]]
                           + self.state.matches[i]["t1"])),
                    abs(clicked_t
                        - (starts[self.state.matches[i]["v2_idx"]]
                           + self.state.matches[i]["t2"])),
                ))
            self._show_preview(best_i)
        except Exception as e:
            print(f"[TimelineClick] {e}")

    # ── Превью ────────────────────────────────────────────────────────────

    def _show_preview(self, index: int) -> None:
        try:
            if not self.state.matches:
                return
            index = max(0, min(index, len(self.state.matches) - 1))
            self.state.current_match = index
            m = self.state.matches[index]

            pw, ph = self._preview_panel.get_preview_size()
            pw, ph = max(10, pw - 4), max(10, ph - 4)

            img1 = self._load_frame(m["v1_idx"], m["f1"], pw, ph)
            img2 = self._load_frame(m["v2_idx"], m["f2"], pw, ph)

            self._preview_panel.update_preview(
                m, img1, img2, index, len(self.state.matches))
            self._results_panel.update_counter(
                index, len(self.state.matches))
            self._results_panel.vlist.select_by_match_idx(index)
        except Exception as e:
            print(f"[ShowPreview] {e}")

    def _load_frame(self, video_idx: int, frame_num: int,
                    max_w: int = 520,
                    max_h: int = 320) -> ImageTk.PhotoImage | None:
        if (video_idx < 0
                or video_idx >= len(self.state.video_queue)):
            return None
        path = self.state.video_queue[video_idx]

        frame = self._frame_cache.get(
            path, frame_num, max_w, max_h)

        if frame is None:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, raw = cap.read()
            cap.release()
            if not ret or raw is None:
                return None
            h, w  = raw.shape[:2]
            scale = min(max_w / max(w, 1),
                        max_h / max(h, 1), 1.0)
            nw    = max(1, int(w * scale))
            nh    = max(1, int(h * scale))
            frame = cv2.resize(raw, (nw, nh),
                               interpolation=cv2.INTER_AREA)
            self._frame_cache.put(
                path, frame_num, max_w, max_h, frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def _get_duration(self, path: str) -> float:
        if path in self.state.video_durations_cache:
            return self.state.video_durations_cache[path]
        try:
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frm = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            dur = frm / fps if fps > 0 else 0.0
            cap.release()
        except Exception:
            dur = 0.0
        self.state.video_durations_cache[path] = dur
        return dur

    def _get_frame_count(self, path: str) -> int:
        try:
            cap = cv2.VideoCapture(path)
            frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return frm
        except Exception:
            return 0

    def _redraw_heatmap(self) -> None:
        try:
            if not self.state.video_queue:
                return
            durations = [self._get_duration(p)
                         for p in self.state.video_queue]
            self._preview_panel.draw_heatmap(durations)
        except Exception as e:
            print(f"[Heatmap] {e}")

    # ── Язык ──────────────────────────────────────────────────────────────

    def _toggle_language(self) -> None:
        langs      = list(SUPPORTED_LANGUAGES)
        self._lang = langs[
            (langs.index(self._lang) + 1) % len(langs)]

        # Топбар
        self._lang_btn.set_text(self._lang.upper())
        self._models_btn.set_text(_S("models_btn", self._lang))

        # Оверлей
        try:
            self._overlay.set_lang(self._lang)
        except Exception:
            pass

        # Все панели
        for panel in (self._stats_bar,
                      self._preview_panel,
                      self._results_panel,
                      self._settings_panel):
            try:
                panel.set_lang(self._lang)
            except Exception:
                pass

        try:
            self._source_panel.set_lang(self._lang)
        except Exception:
            pass

        ConfigManager.set_language(self._lang)

        # Перелокализуем матчи и UI без повторной классификации
        if self.state.matches:
            self._classifier.lang = self._lang

            # relabel использует сохранённые _cat_key
            self.state.found_categories = (
                self._classifier.relabel_all(self.state.matches))

            # Обновляем список и превью
            self._results_panel.show_results(
                self.state.matches,
                self.state.found_categories)

            # Обновляем превью текущего матча
            try:
                self._show_preview(self.state.current_match)
            except Exception:
                pass

            # Обновляем строку статуса
            try:
                self._preview_panel.set_done_status(
                    len(self.state.matches))
            except Exception:
                pass

    # ── Fullscreen ────────────────────────────────────────────────────────

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        self.root.attributes("-fullscreen", self._fullscreen)

    def _exit_fullscreen(self) -> None:
        if self._fullscreen:
            self._fullscreen = False
            self.root.attributes("-fullscreen", False)

    # ── Drag & Drop ───────────────────────────────────────────────────────

    def _setup_drag_drop(self) -> None:
        if not _DND_AVAILABLE:
            return
        try:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<Drop>>", self._on_dnd_drop)
        except Exception as e:
            print(f"[DnD] {e}")

    def _on_dnd_drop(self, event: tk.Event) -> None:
        import re
        raw   = event.data
        paths = re.findall(r"\{([^}]+)\}", raw) if "{" in raw else []
        paths += (re.sub(r"\{[^}]+\}", "", raw).split()
                  if "{" in raw else raw.split())
        valid = [p for p in paths
                 if os.path.isfile(p) and is_video_file(p)]
        if valid:
            self._source_panel.add_to_queue(valid)

    # ── Трей ──────────────────────────────────────────────────────────────

    def _create_tray(self) -> None:
        if not TRAY_AVAILABLE:
            return
        try:
            tray_path = (Path(__file__).parent.parent
                         / "icons" / "tray_icon.png")
            image = (PILImage.open(tray_path)
                     if tray_path.exists()
                     else PILImage.new("RGB", (64, 64), "#3b82f6"))
            menu = pystray.Menu(
                pystray.MenuItem("Показать", self._show_window),
                pystray.MenuItem("Скрыть",   self._hide_window),
                pystray.MenuItem("Выход",    self._quit),
            )
            self._tray_icon = pystray.Icon(
                "parallel_finder", image,
                APP_DISPLAY_NAME, menu)
            self._tray_icon.run_detached()
        except Exception as e:
            print(f"[Tray] {e}")

    def _show_window(self) -> None:
        self.root.after(0, lambda: (
            self.root.deiconify(),
            self.root.lift(),
            self.root.focus_force()))

    def _hide_window(self) -> None:
        self.root.withdraw()

    def _hide_to_tray(self) -> None:
        if TRAY_AVAILABLE and self._tray_icon:
            self._hide_window()
        else:
            self._quit()

    def _quit(self) -> None:
        self.state.analysis_running = False

        if self.backend:
            try:
                self.backend.stop_analysis()
            except Exception:
                pass

        for proc in list(self._child_procs):
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._child_procs.clear()

        if platform.system() == "Windows":
            try:
                for proc in psutil.process_iter(["pid", "name"]):
                    if "powershell" in (
                            proc.info.get("name") or "").lower():
                        try:
                            psutil.Process(
                                proc.info["pid"]).terminate()
                        except Exception:
                            pass
            except Exception:
                pass

        if self._tray_icon:
            try:
                self._tray_icon.stop()
            except Exception:
                pass
            self._tray_icon = None

        try:
            self.root.quit()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self) -> None:
        self.root.mainloop()


# ── Точка входа ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        ParallelFinderApp().run()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        try:
            messagebox.showerror("Критическая ошибка", str(exc))
        except Exception:
            pass