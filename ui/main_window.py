#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from threading import Thread
import time
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk
import hashlib
import gc
import warnings
import platform
import psutil
import bisect

from core.engine import YoloEngine
from core.project import ProjectManager

# Попытка импорта для трея
try:
    import pystray
    from PIL import Image as PILImage
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False
    print("Для трея установи: pip install pystray")

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

THEMES = {
    'dark': {
        'bg':             '#0a0c10',
        'card':           '#14171c',
        'text':           '#ffffff',
        'text_secondary': '#8a8f99',
        'accent':         '#3b82f6',
        'accent_hover':   '#60a5fa',
        'success':        '#10b981',
        'error':          '#ef4444',
        'border':         '#1f2937',
        'highlight':      '#1e293b',
        'glow':           '#3b82f6',
    }
}

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


# ══════════════════════════════════════════════════════════════════
# Вспомогательные виджеты
# ══════════════════════════════════════════════════════════════════

class AnimatedProgressbar(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, height=8,
                                bg=THEMES['dark']['bg'], highlightthickness=0)
        self.canvas.pack(fill=tk.X, expand=True)
        self.value = 0
        self._anim_running = False

    def set(self, value: float):
        self.value = min(100.0, max(0.0, value))
        self._redraw()

    def _redraw(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 400
        fill_w = (self.value / 100.0) * w
        c = THEMES['dark']

        # Фон
        self.canvas.create_rectangle(0, 0, w, 8, fill=c['border'], outline='')

        if self.value > 0:
            # Основная заливка
            self.canvas.create_rectangle(0, 0, fill_w, 8,
                                         fill=c['accent'], outline='')
            # Световой блик на конце
            if 5 < self.value < 95:
                glow_x1 = max(0, fill_w - 14)
                self.canvas.create_rectangle(
                    glow_x1, 1, fill_w, 7,
                    fill=c['glow'], outline='',
                )

    def start_animation(self):
        if not self._anim_running:
            self._anim_running = True
            self._animate()

    def stop_animation(self):
        self._anim_running = False

    def _animate(self):
        if not self._anim_running:
            return
        self._redraw()
        self.after(32, self._animate)   # ~30 fps достаточно


class GlowButton(tk.Canvas):
    """Кастомная кнопка с закруглёнными углами и hover-эффектом."""

    def __init__(self, parent, text, command, bg_color, hover_color,
                 width=120, height=32, state='normal'):
        super().__init__(parent, highlightthickness=0,
                         width=width, height=height,
                         bg=THEMES['dark']['bg'])
        self.command     = command
        self.bg_color    = bg_color
        self.hover_color = hover_color
        self.text        = text
        self._state      = state
        self._is_hover   = False

        self.bind("<Enter>",     self._on_enter)
        self.bind("<Leave>",     self._on_leave)
        self.bind("<Button-1>",  self._on_click)
        self.bind("<Configure>", lambda e: self._draw(self._is_hover))

        self._draw(False)

    # ── Рисование ───────────────────────────────────────────────

    def _draw(self, is_hover: bool = False):
        self._is_hover = is_hover
        self.delete("all")

        w = self.winfo_width()  or int(self['width'])
        h = self.winfo_height() or int(self['height'])

        color = self.hover_color if (is_hover and self._state != 'disabled') \
                else self.bg_color

        if self._state == 'disabled':
            color = THEMES['dark']['border']

        self._round_rect(0, 0, w, h, radius=8, fill=color)
        fg = THEMES['dark']['text_secondary'] if self._state == 'disabled' \
             else 'white'
        self.create_text(w // 2, h // 2, text=self.text,
                         fill=fg, font=('Inter', 10, 'bold'), justify='center')

    def _round_rect(self, x1, y1, x2, y2, radius, **kwargs):
        pts = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]
        return self.create_polygon(pts, smooth=True, **kwargs)

    # ── Обработчики ─────────────────────────────────────────────

    def _on_enter(self, _e):
        if self._state != 'disabled':
            self._draw(True)

    def _on_leave(self, _e):
        self._draw(False)

    def _on_click(self, _e):
        if self._state != 'disabled' and self.command:
            self.command()

    # ── Конфиг (совместимость с tk.Widget.config) ───────────────

    def config(self, **kwargs):
        if 'state' in kwargs:
            self._state = kwargs.pop('state')
            self._draw(self._is_hover)
        if kwargs:
            super().config(**kwargs)

    configure = config


# ══════════════════════════════════════════════════════════════════
# Главное приложение
# ══════════════════════════════════════════════════════════════════

class ParallelFinderApp:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Parallel Finder")
        self.root.geometry("1600x900")
        self.root.minsize(1400, 800)
        self.colors = THEMES['dark']
        self.root.configure(bg=self.colors['bg'])

        # Иконка окна
        try:
            icon_path = os.path.join(os.path.dirname(__file__),
                                     "..", "icons", "icon.ico")
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except Exception:
            pass

        # ── Tkinter-переменные настроек ──────────────────────────────────
        self.threshold             = tk.IntVar(value=70)
        self.scene_interval        = tk.IntVar(value=3)
        self.match_gap             = tk.IntVar(value=4)
        self.motion_length         = tk.DoubleVar(value=2.0)
        self.frame_skip            = tk.IntVar(value=2)
        self.quality               = tk.StringVar(value='Средне')
        self.use_scale_invariance  = tk.BooleanVar(value=True)
        self.use_mirror_invariance = tk.BooleanVar(value=False)
        self.use_body_weights      = tk.BooleanVar(value=False)

        # ── YOLO ─────────────────────────────────────────────────────────
        self.yolo = YoloEngine()
        self.yolo.load()

        # ── Состояние приложения ─────────────────────────────────────────
        self.video_path               = None
        self.video_queue:        list = []
        self.batch_mode:         bool = False
        self.analysis_running:   bool = False
        self.poses_tensor              = None
        self.poses_meta:         list = []
        self.matches:            list = []
        self.current_match:      int  = 0
        self.current_batch_index:int  = 0
        self.total_batch_videos: int  = 0

        # ── Параметры (будут переопределены авто-настройкой) ─────────────
        self.BATCH_SIZE            = 32
        self.CHUNK_SIZE            = 3000
        self.CHUNK_OVERLAP         = 300
        self.MIN_MATCH_GAP         = 5.0
        self.max_matches_per_chunk = 500_000
        self.max_total_matches     = 10_000_000
        self._max_unique_results   = 1000
        self._junk_ratio           = 0.20

        # ── Кеш превью ───────────────────────────────────────────────────
        self.preview_cache_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "cache", "previews",
        )
        os.makedirs(self.preview_cache_dir, exist_ok=True)

        # ── Вспомогательные структуры ────────────────────────────────────
        self.value_labels:          dict = {}
        self.metric_values:         dict = {}
        self.video_durations_cache: dict = {}

        self.setup_ui()
        self.setup_hotkeys()

        # Трей
        self._create_tray_icon()
        self.root.protocol("WM_DELETE_WINDOW", self._hide_to_tray)

        # Автонастройка под железо (в конце, чтобы UI уже был готов)
        self._auto_tune_hardware()

    # ══════════════════════════════════════════════════════════════
    # ТРЕЙ
    # ══════════════════════════════════════════════════════════════

    def _create_tray_icon(self):
        if not TRAY_AVAILABLE:
            return
        try:
            tray_path = os.path.join(os.path.dirname(__file__),
                                     "..", "icons", "tray_icon.png")
            if os.path.exists(tray_path):
                image = PILImage.open(tray_path)
            else:
                image = PILImage.new('RGB', (64, 64), color='#3b82f6')

            menu = pystray.Menu(
                pystray.MenuItem("Показать", self._show_window),
                pystray.MenuItem("Скрыть",   self._hide_window),
                pystray.MenuItem("Выход",    self._quit_app),
            )
            self.tray_icon = pystray.Icon(
                "parallel_finder", image, "Parallel Finder", menu
            )
            self.tray_icon.run_detached()
        except Exception as e:
            print(f"Ошибка создания трея: {e}")

    def _show_window(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _hide_window(self):
        self.root.withdraw()

    def _hide_to_tray(self):
        self._hide_window()

    def _quit_app(self):
        if hasattr(self, 'tray_icon'):
            try:
                self.tray_icon.stop()
            except Exception:
                pass
        self.analysis_running = False
        self.root.quit()
        self.root.destroy()

    # ══════════════════════════════════════════════════════════════
    # АВТОНАСТРОЙКА
    # ══════════════════════════════════════════════════════════════

    def _auto_tune_hardware(self):
        """
        Автоматическая настройка параметров под железо пользователя.

        Логика выбора пресета:
        - CPU-only: по объёму RAM (< 4 / 4-8 / 8-16 / 16-32 / >32 GB).
        - GPU:      по объёму VRAM с детальным охватом всех актуальных карт
                    включая RTX 40xx и RTX 50xx серии.

        Каждый пресет задаёт:
          batch         – кадров в одном YOLO-запросе
          chunk         – размер окна при матричном сравнении поз
          overlap       – перекрытие чанков (уменьшает пропуски на границах)
          quality       – режим FPS-семплинга ('Быстро'/'Средне'/'Макс')
          matches_chunk – лимит пар за один чанк (защита от OOM)
          total_matches – глобальный лимит пар до дедупликации
          max_unique    – максимум финальных уникальных совпадений
          min_gap       – минимальный интервал между найденными позами (с)
          fp16          – использование float16 (только GPU)
          junk_ratio    – доля "мусорных" (sim < 0.85) совпадений для дедупл.
          desc          – человекочитаемое описание пресета
        """
        print("\n" + "=" * 70)
        print("АВТОНАСТРОЙКА ПОД ЖЕЛЕЗО")
        print("=" * 70)

        cpu_cores = psutil.cpu_count(logical=True)  or 1
        cpu_phys  = psutil.cpu_count(logical=False) or 1
        cpu_name  = platform.processor() or "Unknown"
        ram_gb    = psutil.virtual_memory().total / 1e9

        print(f"CPU : {cpu_name}")
        print(f"      ядра: {cpu_phys} физических / {cpu_cores} логических")
        print(f"RAM : {ram_gb:.1f} GB")

        gpu_available = torch.cuda.is_available()
        vram_gb  = 0.0
        gpu_name = "Не обнаружен"

        if gpu_available:
            try:
                gpu_name  = torch.cuda.get_device_name(0)
                vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1e9
                cuda_ver  = torch.version.cuda or "?"
                cc_major  = torch.cuda.get_device_properties(0).major
                cc_minor  = torch.cuda.get_device_properties(0).minor
                print(f"GPU : {gpu_name}")
                print(f"      VRAM: {vram_gb:.1f} GB  |  CUDA: {cuda_ver}  "
                      f"|  CC: {cc_major}.{cc_minor}")
            except Exception as e:
                print(f"Ошибка определения GPU: {e}")
                gpu_available = False

        # ── Таблица пресетов ─────────────────────────────────────────────────
        # Примеры карт для ориентира:
        #   < 2 GB  : GT 1030, iGPU, MX150
        #   2-3 GB  : GTX 1050, GTX 750 Ti, MX450
        #   3-5 GB  : GTX 1050 Ti (4G), GTX 1650 (4G), RX 570 (4G)
        #   5-7 GB  : RTX 2060 (6G), GTX 1660 (6G), RX 5600 XT (6G)
        #   7-9 GB  : RTX 2070 (8G), RTX 3070 (8G), RX 6700 XT (8G)
        #   9-11 GB : RTX 3080 (10G), RX 6800 (10G), A2000 (10G)
        #  11-14 GB : RTX 2080 Ti (11G), RTX 3080 Ti (12G), RTX 3060 (12G),
        #             RTX 4070 (12G), A2000 (12G)
        #  14-18 GB : RTX 4070 Ti (16G), RTX 4080 (16G), A4000 (16G),
        #             RTX 5070 (15G), RTX 5070 Ti (16G)
        #  18-26 GB : RTX 3090 (24G), RTX 4090 (24G), RTX 5080 (16G→24G),
        #             A5000 (24G), RTX 5090 (32G)
        #  >= 26 GB : RTX 5090 (32G), A6000 (48G), серверные GPU

        CONFIGS: dict = {

            # ─── CPU-only ────────────────────────────────────────────────────
            'cpu_very_low': {   # < 4 GB RAM — нетбуки, слабые ноуты
                'batch': 2,   'chunk': 400,   'overlap': 40,
                'quality': 'Быстро',
                'matches_chunk': 30_000,    'total_matches': 300_000,
                'max_unique': 150, 'min_gap': 3.0,
                'fp16': False, 'junk_ratio': 0.10,
                'desc': 'CPU (< 4 GB RAM) — аварийный режим',
            },
            'cpu_low': {        # 4–8 GB RAM
                'batch': 4,   'chunk': 700,   'overlap': 70,
                'quality': 'Быстро',
                'matches_chunk': 80_000,    'total_matches': 800_000,
                'max_unique': 250, 'min_gap': 3.0,
                'fp16': False, 'junk_ratio': 0.12,
                'desc': 'CPU (4–8 GB RAM) — минимальный режим',
            },
            'cpu_mid': {        # 8–16 GB RAM
                'batch': 6,   'chunk': 1000,  'overlap': 100,
                'quality': 'Быстро',
                'matches_chunk': 150_000,   'total_matches': 2_000_000,
                'max_unique': 400, 'min_gap': 3.0,
                'fp16': False, 'junk_ratio': 0.15,
                'desc': 'CPU (8–16 GB RAM) — стабильный режим',
            },
            'cpu_high': {       # 16–32 GB RAM
                'batch': 10,  'chunk': 1800,  'overlap': 180,
                'quality': 'Средне',
                'matches_chunk': 400_000,   'total_matches': 6_000_000,
                'max_unique': 600, 'min_gap': 3.0,
                'fp16': False, 'junk_ratio': 0.18,
                'desc': 'CPU (16–32 GB RAM) — комфортный режим',
            },
            'cpu_workstation': {  # > 32 GB RAM (серверы / рабочие станции)
                'batch': 16,  'chunk': 2500,  'overlap': 250,
                'quality': 'Средне',
                'matches_chunk': 800_000,   'total_matches': 12_000_000,
                'max_unique': 700, 'min_gap': 3.0,
                'fp16': False, 'junk_ratio': 0.20,
                'desc': 'CPU (> 32 GB RAM) — рабочая станция',
            },

            # ─── GPU слабый ──────────────────────────────────────────────────
            'gpu_igpu': {       # iGPU / GT 1030 / MX150 / < 2 GB VRAM
                'batch': 4,   'chunk': 600,   'overlap': 60,
                'quality': 'Быстро',
                'matches_chunk': 60_000,    'total_matches': 800_000,
                'max_unique': 250, 'min_gap': 3.0,
                'fp16': True,  'junk_ratio': 0.10,
                'desc': 'iGPU / GT 1030 (< 2 GB VRAM)',
            },
            'gpu_2gb': {        # GTX 1050 / GTX 750 Ti / MX450 / 2–3 GB
                'batch': 10,  'chunk': 1000,  'overlap': 100,
                'quality': 'Быстро',
                'matches_chunk': 120_000,   'total_matches': 1_500_000,
                'max_unique': 350, 'min_gap': 3.0,
                'fp16': True,  'junk_ratio': 0.12,
                'desc': 'GTX 1050 / GTX 750 Ti / MX450 (2–3 GB VRAM)',
            },
            'gpu_4gb': {        # GTX 1050 Ti / GTX 1650 / RX 570 / 3–5 GB
                'batch': 18,  'chunk': 1500,  'overlap': 150,
                'quality': 'Быстро',
                'matches_chunk': 250_000,   'total_matches': 4_000_000,
                'max_unique': 450, 'min_gap': 3.0,
                'fp16': True,  'junk_ratio': 0.15,
                'desc': 'GTX 1050 Ti / GTX 1650 / RX 570 (3–5 GB VRAM)',
            },

            # ─── GPU средний ─────────────────────────────────────────────────
            'gpu_6gb': {        # RTX 2060 / GTX 1660 / RX 5600 XT / 5–7 GB
                'batch': 28,  'chunk': 2200,  'overlap': 220,
                'quality': 'Средне',
                'matches_chunk': 500_000,   'total_matches': 8_000_000,
                'max_unique': 600, 'min_gap': 4.0,
                'fp16': True,  'junk_ratio': 0.18,
                'desc': 'RTX 2060 / GTX 1660 / RX 5600 XT (5–7 GB VRAM)',
            },
            'gpu_8gb': {        # RTX 2070/3070 / RX 6700 XT / 7–9 GB
                'batch': 42,  'chunk': 3000,  'overlap': 300,
                'quality': 'Средне',
                'matches_chunk': 900_000,   'total_matches': 12_000_000,
                'max_unique': 750, 'min_gap': 4.0,
                'fp16': True,  'junk_ratio': 0.20,
                'desc': 'RTX 2070/3070 / RX 6700 XT (7–9 GB VRAM)',
            },

            # ─── GPU хороший ─────────────────────────────────────────────────
            'gpu_10gb': {       # RTX 3080 10G / RX 6800 / A2000 10G / 9–11 GB
                'batch': 52,  'chunk': 4000,  'overlap': 400,
                'quality': 'Средне',
                'matches_chunk': 1_300_000, 'total_matches': 18_000_000,
                'max_unique': 850, 'min_gap': 4.0,
                'fp16': True,  'junk_ratio': 0.20,
                'desc': 'RTX 3080 10 GB / RX 6800 / A2000 (9–11 GB VRAM)',
            },
            'gpu_12gb': {       # RTX 3060 12G / RTX 3080 Ti / RTX 4070 12G / 11–14 GB
                'batch': 64,  'chunk': 5000,  'overlap': 500,
                'quality': 'Макс',
                'matches_chunk': 1_800_000, 'total_matches': 25_000_000,
                'max_unique': 1000, 'min_gap': 5.0,
                'fp16': True,  'junk_ratio': 0.22,
                'desc': 'RTX 3060/4070 12 GB / RTX 3080 Ti (11–14 GB VRAM)',
            },

            # ─── GPU флагман (включая RTX 40xx и RTX 50xx) ───────────────────
            'gpu_16gb': {       # RTX 4070 Ti / RTX 4080 / RTX 5070/Ti / A4000 / 14–20 GB
                'batch': 88,  'chunk': 7000,  'overlap': 700,
                'quality': 'Макс',
                'matches_chunk': 2_800_000, 'total_matches': 45_000_000,
                'max_unique': 1000, 'min_gap': 5.0,
                'fp16': True,  'junk_ratio': 0.25,
                'desc': 'RTX 4070 Ti / RTX 4080 / RTX 5070 Ti / A4000 (14–20 GB VRAM)',
            },
            'gpu_24gb': {       # RTX 3090 / RTX 4090 / RTX 5080 / A5000 / 20–28 GB
                'batch': 120, 'chunk': 10000, 'overlap': 1000,
                'quality': 'Макс',
                'matches_chunk': 4_500_000, 'total_matches': 80_000_000,
                'max_unique': 1000, 'min_gap': 5.0,
                'fp16': True,  'junk_ratio': 0.28,
                'desc': 'RTX 3090/4090/5080 / A5000 (20–28 GB VRAM)',
            },
            'gpu_32gb': {       # RTX 5090 / A6000 / L40 / >= 28 GB
                'batch': 160, 'chunk': 14000, 'overlap': 1400,
                'quality': 'Макс',
                'matches_chunk': 7_000_000, 'total_matches': 150_000_000,
                'max_unique': 1000, 'min_gap': 5.0,
                'fp16': True,  'junk_ratio': 0.30,
                'desc': 'RTX 5090 / A6000 / L40 (>= 28 GB VRAM) — абсолютный максимум',
            },
        }

        # ── Выбор пресета ────────────────────────────────────────────────────
        if not gpu_available:
            if ram_gb < 4:
                cfg_key = 'cpu_very_low'
            elif ram_gb < 8:
                cfg_key = 'cpu_low'
            elif ram_gb < 16:
                cfg_key = 'cpu_mid'
            elif ram_gb < 32:
                cfg_key = 'cpu_high'
            else:
                cfg_key = 'cpu_workstation'
        else:
            # GPU-ветка: выбор по VRAM
            if vram_gb < 2:
                cfg_key = 'gpu_igpu'
            elif vram_gb < 3:
                cfg_key = 'gpu_2gb'
            elif vram_gb < 5:
                cfg_key = 'gpu_4gb'
            elif vram_gb < 7:
                cfg_key = 'gpu_6gb'
            elif vram_gb < 9:
                cfg_key = 'gpu_8gb'
            elif vram_gb < 11:
                cfg_key = 'gpu_10gb'
            elif vram_gb < 14:
                cfg_key = 'gpu_12gb'
            elif vram_gb < 20:
                cfg_key = 'gpu_16gb'
            elif vram_gb < 28:
                cfg_key = 'gpu_24gb'
            else:
                cfg_key = 'gpu_32gb'

        cfg = CONFIGS[cfg_key]

        # ── Применение пресета ───────────────────────────────────────────────
        self.BATCH_SIZE            = cfg['batch']
        self.CHUNK_SIZE            = cfg['chunk']
        self.CHUNK_OVERLAP         = cfg['overlap']
        self.MIN_MATCH_GAP         = cfg['min_gap']
        self.max_matches_per_chunk = cfg['matches_chunk']
        self.max_total_matches     = cfg['total_matches']
        self._max_unique_results   = cfg['max_unique']
        self._junk_ratio           = cfg['junk_ratio']
        self.quality.set(cfg['quality'])

        # FP16 доступен только при CUDA
        self.yolo.use_fp16 = (gpu_available and cfg['fp16'])

        print(f"\nВыбран пресет  : [{cfg_key}] {cfg['desc']}")
        print(f"  BATCH_SIZE            = {self.BATCH_SIZE}")
        print(f"  CHUNK_SIZE            = {self.CHUNK_SIZE}")
        print(f"  CHUNK_OVERLAP         = {self.CHUNK_OVERLAP}")
        print(f"  MIN_MATCH_GAP         = {self.MIN_MATCH_GAP} с")
        print(f"  max_matches_per_chunk = {self.max_matches_per_chunk:,}")
        print(f"  max_total_matches     = {self.max_total_matches:,}")
        print(f"  max_unique_results    = {self._max_unique_results}")
        print(f"  junk_ratio            = {self._junk_ratio:.0%}")
        print(f"  Качество (FPS-режим)  = {self.quality.get()}")
        print(f"  FP16                  = {self.yolo.use_fp16}")
        print("=" * 70 + "\n")

    # ══════════════════════════════════════════════════════════════
    # UI — сборка
    # ══════════════════════════════════════════════════════════════

    def on_closing(self):
        self._quit_app()

    def setup_ui(self):
        main = tk.Frame(self.root, bg=self.colors['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=32, pady=32)

        self.setup_stats_panel(main)

        content = tk.Frame(main, bg=self.colors['bg'])
        content.pack(fill=tk.BOTH, expand=True, pady=24)

        left = tk.Frame(content, bg=self.colors['bg'], width=400)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12), expand=False)
        left.pack_propagate(False)

        center = tk.Frame(content, bg=self.colors['bg'])
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        right = tk.Frame(content, bg=self.colors['bg'], width=400)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0), expand=False)
        right.pack_propagate(False)

        self.setup_video_panel(left)
        self.setup_settings_panel(left)
        self.setup_preview_panel(center)
        self.setup_results_panel(right)

    def setup_stats_panel(self, parent):
        stats = tk.Frame(parent, bg=self.colors['bg'])
        stats.pack(fill=tk.X)

        self.metric_values = {
            'frames':         tk.StringVar(value="0"),
            'patterns':       tk.StringVar(value="0"),
            'duration':       tk.StringVar(value="0:00"),
            'accuracy':       tk.StringVar(value="0%"),
            'time_left':      tk.StringVar(value="--:--"),
            'batch_progress': tk.StringVar(value="0/0"),
        }

        metrics = [
            ('Кадров',       'frames'),
            ('Повторов',     'patterns'),
            ('Длительность', 'duration'),
            ('Схожесть',     'accuracy'),
            ('Осталось',     'time_left'),
            ('Прогресс',     'batch_progress'),
        ]

        for i, (label, var) in enumerate(metrics):
            card = tk.Frame(stats, bg=self.colors['card'], padx=15, pady=12)
            card.pack(side=tk.LEFT, padx=(0 if i == 0 else 8, 0),
                      expand=True, fill=tk.BOTH)

            tk.Label(card, textvariable=self.metric_values[var],
                     font=('Inter', 20, 'bold'),
                     bg=self.colors['card'], fg=self.colors['text']).pack(anchor='w')
            tk.Label(card, text=label, font=('Inter', 10),
                     bg=self.colors['card'],
                     fg=self.colors['text_secondary']).pack(anchor='w')

    def setup_video_panel(self, parent):
        card = tk.Frame(parent, bg=self.colors['card'])
        card.pack(fill=tk.X, pady=(0, 16))

        inner = tk.Frame(card, bg=self.colors['card'], padx=20, pady=20)
        inner.pack(fill=tk.X)

        tk.Label(inner, text="Источник", font=('Inter', 14, 'bold'),
                 bg=self.colors['card'], fg=self.colors['text']).pack(
            anchor='w', pady=(0, 12))

        self.video_label = tk.Label(inner, text="Файл не выбран",
                                    font=('Inter', 11),
                                    bg=self.colors['card'],
                                    fg=self.colors['text_secondary'],
                                    wraplength=320)
        self.video_label.pack(anchor='w', pady=(0, 4))

        self.video_info = tk.Label(inner, text="", font=('Inter', 10),
                                   bg=self.colors['card'],
                                   fg=self.colors['text_secondary'])
        self.video_info.pack(anchor='w', pady=(0, 12))

        btn_frame = tk.Frame(inner, bg=self.colors['card'])
        btn_frame.pack(fill=tk.X, pady=(0, 5))

        GlowButton(btn_frame, "Выбрать видео", self.select_video,
                   self.colors['accent'], self.colors['accent_hover'],
                   width=120, height=34).pack(side=tk.LEFT, padx=(0, 8))

        GlowButton(btn_frame, "Выбрать папку", self.select_folder,
                   self.colors['success'], '#2ecc71',
                   width=120, height=34).pack(side=tk.LEFT)

        self.batch_info = tk.Label(inner, text="", font=('Inter', 9),
                                   bg=self.colors['card'],
                                   fg=self.colors['text_secondary'])
        self.batch_info.pack(anchor='w', pady=(8, 0))

    def setup_settings_panel(self, parent):
        container = tk.Frame(parent, bg=self.colors['card'])
        container.pack(fill=tk.BOTH, expand=True)

        canvas    = tk.Canvas(container, bg=self.colors['card'],
                              highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(container, orient=tk.VERTICAL,
                                 command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=self.colors['card'])

        scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        win_id = canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(
                win_id, width=e.width - scrollbar.winfo_width()
            ),
        )

        card = scrollable

        tk.Label(card, text="Настройки анализа", font=('Inter', 14, 'bold'),
                 bg=self.colors['card'], fg=self.colors['text']).pack(
            anchor='w', padx=20, pady=(20, 8))

        tk.Frame(card, bg=self.colors['border'], height=1).pack(
            fill=tk.X, padx=20, pady=(10, 16))

        # ── Точность ──────────────────────────────────────────────────────
        g1 = tk.Frame(card, bg=self.colors['card'])
        g1.pack(fill=tk.X, padx=20, pady=(0, 20))

        tk.Label(g1, text="Точность", font=('Inter', 12, 'bold'),
                 bg=self.colors['card'], fg=self.colors['text']).pack(
            anchor='w', pady=(0, 10))

        self.add_setting_row(g1, "Порог схожести",
                             self.threshold, 50, 98, 70, "%")

        q_frame = tk.Frame(g1, bg=self.colors['card'])
        q_frame.pack(fill=tk.X, pady=(0, 12))
        tk.Label(q_frame, text="Качество", font=('Inter', 10),
                 bg=self.colors['card'],
                 fg=self.colors['text_secondary']).pack(anchor='w')
        self.quality_combo = ttk.Combobox(
            q_frame, textvariable=self.quality,
            values=['Быстро', 'Средне', 'Макс'],
            font=('Inter', 10), state='readonly',
        )
        self.quality_combo.pack(fill=tk.X, pady=(4, 0))

        # ── Временные параметры ───────────────────────────────────────────
        g2 = tk.Frame(card, bg=self.colors['card'])
        g2.pack(fill=tk.X, padx=20, pady=(0, 20))

        tk.Label(g2, text="Временные параметры", font=('Inter', 12, 'bold'),
                 bg=self.colors['card'], fg=self.colors['text']).pack(
            anchor='w', pady=(0, 10))

        self.add_setting_row(g2, "Интервал сцен",
                             self.scene_interval, 1, 30, 3, "сек")
        self.add_setting_row(g2, "Мин. интервал между повторами",
                             self.match_gap, 1, 30, 4, "сек")

        # ── Расширенные опции ─────────────────────────────────────────────
        g3 = tk.Frame(card, bg=self.colors['card'])
        g3.pack(fill=tk.X, padx=20, pady=(0, 20))

        tk.Label(g3, text="Расширенные опции", font=('Inter', 12, 'bold'),
                 bg=self.colors['card'], fg=self.colors['text']).pack(
            anchor='w', pady=(0, 10))

        for text, var in [
            ("Нормализация по размеру",         self.use_scale_invariance),
            ("Учитывать зеркальное отражение",  self.use_mirror_invariance),
            ("Веса частей тела",                self.use_body_weights),
        ]:
            tk.Checkbutton(
                g3, text=text, variable=var,
                bg=self.colors['card'], fg=self.colors['text_secondary'],
                selectcolor=self.colors['bg'],
                activebackground=self.colors['card'],
            ).pack(anchor='w')

        tk.Frame(card, bg=self.colors['border'], height=1).pack(
            fill=tk.X, padx=20, pady=(10, 16))

        btn_frame = tk.Frame(card, bg=self.colors['card'])
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        self.start_btn = GlowButton(
            btn_frame, "Старт", self.start_analysis,
            self.colors['success'], '#2ecc71', width=400, height=42,
        )
        self.start_btn.pack(fill=tk.X, pady=(0, 8))

        self.stop_btn = GlowButton(
            btn_frame, "Стоп", self.stop_analysis,
            self.colors['error'], '#ff6b6b', width=400, height=38,
        )
        self.stop_btn.pack(fill=tk.X)

    def add_setting_row(self, parent, label, var, from_, to, default, unit,
                        step=1):
        frame = tk.Frame(parent, bg=self.colors['card'])
        frame.pack(fill=tk.X, pady=(0, 12))

        top = tk.Frame(frame, bg=self.colors['card'])
        top.pack(fill=tk.X, pady=(0, 4))

        tk.Label(top, text=label, font=('Inter', 10),
                 bg=self.colors['card'],
                 fg=self.colors['text_secondary']).pack(side=tk.LEFT)

        val_label = tk.Label(top, text=f"{default}{unit}",
                             font=('Inter', 14, 'bold'),
                             bg=self.colors['card'],
                             fg=self.colors['accent'])
        val_label.pack(side=tk.RIGHT)
        self.value_labels[label] = val_label

        scale = tk.Scale(
            frame, from_=from_, to=to, resolution=step,
            orient=tk.HORIZONTAL, variable=var,
            bg=self.colors['card'], fg=self.colors['text'],
            highlightbackground=self.colors['card'],
            troughcolor=self.colors['bg'],
            sliderrelief='flat', sliderlength=18, width=6,
            command=lambda v, l=label, u=unit: self._update_value(l, v, u),
        )
        var.set(default)
        scale.pack(fill=tk.X)

    def _update_value(self, label: str, value, unit: str):
        if label in self.value_labels:
            self.value_labels[label].config(text=f"{int(float(value))}{unit}")

    def update_value(self, label, value, unit):
        self._update_value(label, value, unit)

    def setup_preview_panel(self, parent):
        # Прогресс
        progress_card = tk.Frame(parent, bg=self.colors['card'])
        progress_card.pack(fill=tk.X, pady=(0, 12))

        p_inner = tk.Frame(progress_card, bg=self.colors['card'],
                           padx=16, pady=16)
        p_inner.pack(fill=tk.X)

        header = tk.Frame(p_inner, bg=self.colors['card'])
        header.pack(fill=tk.X, pady=(0, 8))

        tk.Label(header, text="Прогресс", font=('Inter', 13, 'bold'),
                 bg=self.colors['card'],
                 fg=self.colors['text']).pack(side=tk.LEFT)

        self.progress_label = tk.Label(header, text="0%",
                                       font=('Inter', 14, 'bold'),
                                       bg=self.colors['card'],
                                       fg=self.colors['accent'])
        self.progress_label.pack(side=tk.RIGHT)

        self.progress = AnimatedProgressbar(p_inner)
        self.progress.pack(fill=tk.X, pady=(8, 4))

        self.status_label = tk.Label(p_inner, text="Ожидание",
                                     font=('Inter', 10),
                                     bg=self.colors['card'],
                                     fg=self.colors['text_secondary'])
        self.status_label.pack(anchor='w')

        # Превью
        preview_card = tk.Frame(parent, bg=self.colors['card'])
        preview_card.pack(fill=tk.BOTH, expand=True)

        pv_inner = tk.Frame(preview_card, bg=self.colors['card'],
                            padx=16, pady=16)
        pv_inner.pack(fill=tk.BOTH, expand=True)

        title_frame = tk.Frame(pv_inner, bg=self.colors['card'])
        title_frame.pack(fill=tk.X, pady=(0, 12))

        tk.Label(title_frame, text="Сравнение", font=('Inter', 13, 'bold'),
                 bg=self.colors['card'],
                 fg=self.colors['text']).pack(side=tk.LEFT)

        self.match_info = tk.Label(title_frame, text="",
                                   font=('Inter', 10),
                                   bg=self.colors['card'],
                                   fg=self.colors['text_secondary'])
        self.match_info.pack(side=tk.RIGHT)

        video_row = tk.Frame(pv_inner, bg=self.colors['bg'])
        video_row.pack(fill=tk.BOTH, expand=True)

        self.preview1, self.time1_label, self.action1_label = \
            self._create_video_widget(video_row)
        self.preview2, self.time2_label, self.action2_label = \
            self._create_video_widget(video_row)

        self.setup_timeline_panel(pv_inner)

    def _create_video_widget(self, parent):
        video_frame = tk.Frame(parent, bg=self.colors['bg'])
        side = tk.LEFT if not parent.winfo_children() else tk.RIGHT
        video_frame.pack(side=side, padx=8, expand=True, fill=tk.BOTH)

        container = tk.Frame(video_frame, bg=self.colors['bg'])
        container.pack(fill=tk.BOTH, expand=True)

        preview = tk.Label(container, bg=self.colors['bg'])
        preview.pack(fill=tk.BOTH, expand=True)

        time_label = tk.Label(container, text="--:--",
                              font=('Inter', 14, 'bold'),
                              bg='#1e1e2e', fg='white', padx=6, pady=2)
        time_label.place(relx=0.02, rely=0.02)

        action_label = tk.Label(container, text="",
                                font=('Inter', 10),
                                bg='#1e1e2e', fg=self.colors['accent'],
                                padx=6, pady=2)
        action_label.place(relx=0.02, rely=0.12)

        return preview, time_label, action_label

    def setup_timeline_panel(self, parent):
        timeline_frame = tk.Frame(parent, bg=self.colors['card'], pady=5)
        timeline_frame.pack(fill=tk.X, pady=(12, 0))

        tk.Label(timeline_frame, text="Таймлайн", font=('Inter', 10),
                 bg=self.colors['card'],
                 fg=self.colors['text_secondary']).pack(side=tk.LEFT,
                                                        padx=(0, 10))

        self.timeline_canvas = tk.Canvas(timeline_frame, height=40,
                                         bg=self.colors['bg'],
                                         highlightthickness=0)
        self.timeline_canvas.pack(fill=tk.X, expand=True)
        self.timeline_canvas.bind("<Button-1>", self.on_timeline_click)

    def setup_results_panel(self, parent):
        card = tk.Frame(parent, bg=self.colors['card'])
        card.pack(fill=tk.BOTH, expand=True)

        inner = tk.Frame(card, bg=self.colors['card'], padx=16, pady=16)
        inner.pack(fill=tk.BOTH, expand=True)

        tk.Label(inner, text="Результаты", font=('Inter', 13, 'bold'),
                 bg=self.colors['card'],
                 fg=self.colors['text']).pack(anchor='w', pady=(0, 12))

        # Навигация
        nav_frame = tk.Frame(inner, bg=self.colors['card'])
        nav_frame.pack(fill=tk.X, pady=(0, 10))

        self.prev_btn = GlowButton(
            nav_frame, "Предыдущий", self.prev_match,
            self.colors['highlight'], self.colors['border'],
            width=100, height=30,
        )
        self.prev_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))

        self.match_counter = tk.Label(nav_frame, text="0/0",
                                      font=('Inter', 10),
                                      bg=self.colors['card'],
                                      fg=self.colors['text'])
        self.match_counter.pack(side=tk.LEFT, padx=8)

        self.next_btn = GlowButton(
            nav_frame, "Следующий", self.next_match,
            self.colors['highlight'], self.colors['border'],
            width=100, height=30,
        )
        self.next_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        # Список результатов
        list_frame = tk.Frame(inner, bg=self.colors['bg'])
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_list = tk.Listbox(
            list_frame, yscrollcommand=scrollbar.set,
            bg=self.colors['bg'], fg=self.colors['text'],
            selectbackground=self.colors['accent'],
            font=('Inter', 10), bd=0,
            highlightthickness=1,
            highlightbackground=self.colors['border'],
        )
        self.results_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_list.yview)
        self.results_list.bind('<<ListboxSelect>>', self.on_select)

        # Экспорт
        export_frame = tk.Frame(inner, bg=self.colors['card'])
        export_frame.pack(fill=tk.X, pady=(10, 0))

        for label, cmd in [
            ("JSON", self.export_json),
            ("TXT",  self.export_txt),
            ("EDL",  self.export_edl),
        ]:
            GlowButton(export_frame, label, cmd,
                       self.colors['highlight'], self.colors['border'],
                       width=60, height=30).pack(side=tk.LEFT, padx=(0, 4))

    def setup_hotkeys(self):
        self.root.bind('<space>', lambda e: self.start_analysis())
        self.root.bind('<Up>',    lambda e: self.prev_match())
        self.root.bind('<Down>',  lambda e: self.next_match())

    # ══════════════════════════════════════════════════════════════
    # ВЫБОР ВИДЕО
    # ══════════════════════════════════════════════════════════════

    def select_video(self):
        try:
            path = filedialog.askopenfilename(
                filetypes=[("Видео",
                            "*.mp4 *.avi *.mkv *.mov *.ts *.webm "
                            "*.flv *.m4v *.wmv *.3gp")]
            )
            if path:
                self.video_queue = [path]
                self.batch_mode  = False
                self.video_label.config(text=os.path.basename(path))
                self.batch_info.config(text="")
                self.update_video_info(path)
                self.video_path = path
                self._reset_state()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть видео: {e}")

    def select_folder(self):
        try:
            folder = filedialog.askdirectory(title="Выберите папку с видео")
            if folder:
                exts = ('.mp4', '.avi', '.mkv', '.mov', '.ts',
                        '.webm', '.flv', '.m4v', '.wmv', '.3gp')
                self.video_queue = sorted([
                    os.path.join(folder, f)
                    for f in os.listdir(folder)
                    if f.lower().endswith(exts)
                ])
                if self.video_queue:
                    self.batch_mode = True
                    self.video_label.config(
                        text=f"Папка: {os.path.basename(folder)}")
                    self.batch_info.config(
                        text=f"Видео: {len(self.video_queue)}")
                    self.update_video_info(self.video_queue[0])
                    self.video_path = self.video_queue[0]
                    self._reset_state()
                else:
                    messagebox.showwarning(
                        "Пусто", "В выбранной папке нет поддерживаемых видеофайлов.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть папку: {e}")

    def _reset_state(self):
        self.matches        = []
        self.poses_tensor   = None
        self.poses_meta     = []
        self.current_match  = 0
        self.results_list.delete(0, tk.END)
        self.timeline_canvas.delete("all")
        self.metric_values['patterns'].set("0")
        self.metric_values['accuracy'].set("0%")
        self.root.title("Parallel Finder")

    def update_video_info(self, path: str):
        try:
            cap    = cv2.VideoCapture(path)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
            dur    = frames / fps
            cap.release()
            self.metric_values['frames'].set(self._fmt_num(frames))
            self.metric_values['duration'].set(self._fmt_time(dur))
            self.video_info.config(
                text=f"{self._fmt_num(frames)} кадров • {fps:.0f} fps")
        except Exception:
            pass

    def _get_video_duration(self, path: str) -> float:
        """Длительность видео в секундах (с кешированием)."""
        if path in self.video_durations_cache:
            return self.video_durations_cache[path]
        dur = 0.0
        try:
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frm = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            dur = frm / fps if fps > 0 else 0.0
            cap.release()
        except Exception:
            pass
        self.video_durations_cache[path] = dur
        return dur

    def _fmt_num(self, n: int) -> str:
        return f"{n / 1000:.1f}K" if n >= 1000 else str(n)

    def _fmt_time(self, secs: float) -> str:
        s = max(0.0, secs)
        return f"{int(s // 60):02d}:{int(s % 60):02d}"

    # Публичные алиасы для обратной совместимости
    def _format_number(self, n):   return self._fmt_num(n)
    def _format_time(self, s):     return self._fmt_time(s)

    # ══════════════════════════════════════════════════════════════
    # АНАЛИЗ
    # ══════════════════════════════════════════════════════════════

    def start_analysis(self):
        if not self.video_queue:
            messagebox.showwarning("Нет видео",
                                   "Выберите видео или папку для анализа")
            return
        if self.analysis_running:
            return  # Защита от двойного запуска

        self.analysis_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self._reset_state()
        self.total_batch_videos = len(self.video_queue)
        self.progress.start_animation()
        Thread(target=self._run_analysis_pipeline, daemon=True).start()

    def stop_analysis(self):
        self.analysis_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="Остановлено")
        self.progress.stop_animation()

    def _run_analysis_pipeline(self):
        try:
            all_poses_meta: list = []
            all_poses_vecs: list = []

            for idx, path in enumerate(self.video_queue):
                if not self.analysis_running:
                    break

                self.current_batch_index = idx
                self.root.after(
                    0,
                    lambda p=path, i=idx: self._update_batch_status(p, i)
                )

                poses_meta, poses_vecs = self._extract_poses_from_video(path)

                for meta in poses_meta:
                    meta['video_idx'] = idx

                all_poses_meta.extend(poses_meta)
                all_poses_vecs.extend(poses_vecs)

            if not self.analysis_running or not all_poses_vecs:
                self.root.after(0, self._on_analysis_complete)
                return

            self.poses_meta = all_poses_meta

            # Всегда используем float32 для тензора поз —
            # FP16 применяется только внутри YOLO-инференса
            self.poses_tensor = torch.tensor(
                np.array(all_poses_vecs, dtype=np.float32),
                dtype=torch.float32,
                device=self.yolo.device,
            )

            n_poses = len(self.poses_tensor)
            self.root.after(0, lambda: self.status_label.config(
                text=f"Поиск совпадений среди {n_poses} поз…"
            ))

            self.matches = self._find_matches_tensor_chunked()
            self.root.after(0, self._on_analysis_complete)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
            self.root.after(0, self._on_analysis_complete)

    def _update_batch_status(self, path: str, idx: int):
        self.status_label.config(
            text=f"{idx + 1}/{self.total_batch_videos}: "
                 f"{os.path.basename(path)}"
        )
        self.metric_values['batch_progress'].set(
            f"{idx + 1}/{self.total_batch_videos}"
        )

    # ── Извлечение поз из видео ───────────────────────────────────

    def _extract_poses_from_video(self, path: str):
        """
        Извлечение поз из видео.

        ИСПРАВЛЕНИЕ фриза: batch_size вычисляется ОДИН РАЗ перед циклом
        по первому кадру, а не заново на каждой итерации.
        Flush происходит ровно каждые batch_size кадров.
        """
        cap          = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0

        quality  = self.quality.get()
        base_fps = {'Быстро': 8, 'Средне': 15, 'Макс': 30}.get(quality, 15)

        # Читаем размер первого кадра для вычисления skip и batch_size
        h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280

        # Размер после возможного ресайза
        target = 640
        scale  = min(target / w_vid, target / h_vid, 1.0)
        h_eff  = int(h_vid * scale)
        w_eff  = int(w_vid * scale)

        # Адаптивный skip на основе fps видео и желаемого fps анализа
        pixel_load = (h_vid * w_vid * fps) / 1e6
        res_factor = min(2.0, max(1.0, pixel_load / 50.0))
        skip       = max(1, min(int((fps / base_fps) * res_factor), 15))

        # batch_size вычисляем ОДИН РАЗ — нет фризов каждые N кадров
        effective_batch = self.yolo.get_dynamic_batch_size(
            [(h_eff, w_eff)] * 4,   # передаём несколько одинаковых для усреднения
            self.BATCH_SIZE,
        )
        print(f"[Extract] {os.path.basename(path)} | "
              f"skip={skip} | batch={effective_batch} | "
              f"res={w_eff}x{h_eff}")

        start_time        = time.time()
        frame_idx         = 0
        poses_meta: list  = []
        poses_vecs: list  = []
        batch_frames      = []
        batch_meta_pre    = []
        video_hash        = hashlib.md5(path.encode()).hexdigest()
        use_bw            = self.use_body_weights.get()

        while cap.isOpened() and self.analysis_running:
            ret = cap.grab()
            if not ret:
                break

            if frame_idx % skip == 0:
                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    # Ресайз до 640 только если нужно
                    fh, fw = frame.shape[:2]
                    s = min(target / fw, target / fh, 1.0)
                    if s < 1.0:
                        frame = cv2.resize(
                            frame,
                            (int(fw * s), int(fh * s)),
                            interpolation=cv2.INTER_AREA,
                        )

                    batch_frames.append(frame)
                    batch_meta_pre.append({
                        'frame': frame_idx,
                        'time':  frame_idx / fps,
                    })

                    # Flush только когда накопили ровно effective_batch кадров
                    if len(batch_frames) >= effective_batch:
                        self._flush_batch(
                            batch_frames, batch_meta_pre,
                            poses_meta, poses_vecs,
                            video_hash, use_bw,
                        )
                        batch_frames.clear()
                        batch_meta_pre.clear()

            frame_idx += 1

            # Обновление прогресса каждые 60 кадров
            if frame_idx % 60 == 0 and total_frames > 0:
                pct     = (frame_idx / total_frames) * 100.0
                elapsed = time.time() - start_time
                speed   = frame_idx / elapsed if elapsed > 0 else 1.0
                total_work = total_frames * len(self.video_queue)
                done_work  = (frame_idx
                              + self.current_batch_index * total_frames)
                remaining  = max(
                    0.0, (total_work - done_work) / speed
                ) if speed > 0 else 0.0
                self.root.after(
                    0,
                    lambda p=pct, r=remaining,
                           cf=frame_idx, tf=total_frames:
                    self._update_progress(p, r, cf, tf),
                )

        # Остаток батча (< effective_batch кадров)
        if batch_frames and self.analysis_running:
            self._flush_batch(
                batch_frames, batch_meta_pre,
                poses_meta, poses_vecs,
                video_hash, use_bw,
            )

        cap.release()
        print(f"[Extract] Готово: {len(poses_meta)} поз из "
              f"{frame_idx} кадров")
        return poses_meta, poses_vecs

    def _flush_batch(self, batch_frames, batch_meta_pre,
                     poses_meta, poses_vecs, video_hash, use_bw):
        """Обработка накопленного батча кадров YOLO и препроцессинг поз."""
        poses_data = self.yolo.detect_batch(batch_frames)
        for pose_data, meta in zip(poses_data, batch_meta_pre):
            if pose_data is None:
                continue
            kps = pose_data['keypoints']
            if kps.shape[0] < 17:
                continue

            vec = self._preprocess_pose(pose_data, use_bw)
            poses_meta.append({
                't':   meta['time'],
                'f':   meta['frame'],
                'dir': pose_data.get('direction', 'forward'),
                'vec': vec.reshape(17, 2),
            })
            poses_vecs.append(vec.flatten())

    def _preprocess_pose(self, pose_data: dict,
                         use_body_weights: bool = False) -> np.ndarray:
        """
        Нормализация позы.
        Если use_body_weights=True — применяет весовую схему по частям тела.
        """
        kps      = pose_data['keypoints'][:17, :2].astype(np.float32)
        center   = np.mean(kps, axis=0)
        centered = kps - center
        scale    = np.max(np.abs(centered)) + 1e-5
        normed   = centered / scale

        if use_body_weights:
            from core.matcher import MotionMatcher
            weights = MotionMatcher.BODY_WEIGHTS[:, None]
            normed  = normed * weights
            s2 = np.max(np.abs(normed)) + 1e-5
            normed = normed / s2

        return normed  # (17, 2)

    # ── Матричный поиск совпадений ────────────────────────────────

    def _find_matches_tensor_chunked(self) -> list:
        """
        Поиск похожих поз с чанкованием и дедупликацией.

        Изменения:
        - Шаг чанка = chunk_size - overlap (правильное скользящее окно).
        - Дедупликация учитывает video_idx.
        - junk_ratio берётся из авто-настроенного self._junk_ratio.
        - Тензор всегда float32 (стабильнее для косинусного сходства).
        """
        if self.poses_tensor is None or len(self.poses_tensor) < 10:
            return []

        device  = self.yolo.device
        thresh  = self.threshold.get() / 100.0
        min_gap = self.scene_interval.get()

        n = len(self.poses_tensor)
        print(f"\nПоиск среди {n} поз | порог={thresh} | интервал={min_gap}с")

        # Нормируем один раз, float32
        V = self.poses_tensor.to(dtype=torch.float32, device=device)
        V = V.view(n, -1)
        V = F.normalize(V, p=2, dim=1)

        # Зеркальная версия (опционально)
        V_mirror = None
        if self.use_mirror_invariance.get():
            from core.matcher import MotionMatcher
            V_mirror = MotionMatcher._mirror_vector(V)
            V_mirror = F.normalize(V_mirror, p=2, dim=1)

        times = torch.tensor(
            [m['t'] for m in self.poses_meta],
            dtype=torch.float32, device=device,
        )

        raw_matches:   list = []
        chunk_size     = self.CHUNK_SIZE
        overlap        = self.CHUNK_OVERLAP
        max_per_chunk  = self.max_matches_per_chunk
        max_total      = self.max_total_matches

        effective_step = max(1, chunk_size - overlap)
        n_chunks       = (n + effective_step - 1) // effective_step

        for chunk_idx in range(n_chunks):
            start = chunk_idx * effective_step
            end   = min(start + chunk_size, n)
            if start >= n:
                break

            if not self.analysis_running:
                break
            if len(raw_matches) >= max_total:
                print(f"Достигнут глобальный лимит ({max_total:,}), стоп.")
                break

            V_chunk = V[start:end]
            T_chunk = times[start:end]

            sim = torch.mm(V_chunk, V.t())

            if V_mirror is not None:
                sim_m = torch.mm(V_chunk, V_mirror.t())
                sim   = torch.maximum(sim, sim_m)
                del sim_m

            time_diff = torch.abs(T_chunk.unsqueeze(1) - times.unsqueeze(0))

            col_idx = torch.arange(n, device=device).unsqueeze(0)
            row_idx = torch.arange(start, end, device=device).unsqueeze(1)
            upper   = col_idx > row_idx

            valid = (sim >= thresh) & (time_diff >= min_gap) & upper

            indices_t = torch.nonzero(valid, as_tuple=False)
            scores_t  = sim[valid]

            # Переносим на CPU за один раз
            indices = indices_t.cpu().numpy()
            scores  = scores_t.cpu().numpy()

            del sim, time_diff, upper, valid, indices_t, scores_t
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

            limit = min(len(indices), max_per_chunk)
            for k in range(limit):
                if len(raw_matches) >= max_total:
                    break
                i_local = int(indices[k, 0])
                j       = int(indices[k, 1])
                i       = start + i_local

                raw_matches.append({
                    'm1_idx':    i,
                    'm2_idx':    j,
                    't1':        self.poses_meta[i]['t'],
                    't2':        self.poses_meta[j]['t'],
                    'f1':        self.poses_meta[i]['f'],
                    'f2':        self.poses_meta[j]['f'],
                    'v1_idx':    self.poses_meta[i].get('video_idx', 0),
                    'v2_idx':    self.poses_meta[j].get('video_idx', 0),
                    'sim':       float(scores[k]),
                    'direction': self.poses_meta[i].get('dir', 'forward'),
                })

            print(f"Чанк {chunk_idx + 1}/{n_chunks} [{start}:{end}]: "
                  f"+{limit} пар (всего {len(raw_matches):,})")

        print(f"Сырых совпадений: {len(raw_matches):,}")
        if not raw_matches:
            return []

        return self._deduplicate_matches(raw_matches)

    def _deduplicate_matches(self, matches: list) -> list:
        """
        Жадная дедупликация с бинарным поиском (O(n log n) вместо O(n²)).

        - Сортируем по убыванию sim.
        - Делим на хорошие (>= 0.85) и мусор.
        - Из мусора берём junk_ratio.
        - Жадный фильтр с учётом video_idx.
        """
        matches.sort(key=lambda x: x['sim'], reverse=True)

        junk_ratio = getattr(self, '_junk_ratio', 0.20)
        good       = [m for m in matches if m['sim'] >= 0.85]
        junk       = [m for m in matches if m['sim'] <  0.85]
        junk_take  = int(len(junk) * junk_ratio)
        selected   = good + junk[:junk_take]

        print(f"Хороших (>=0.85): {len(good):,} | "
              f"мусора (<0.85): {len(junk):,} (взято {junk_take:,})")
        print(f"На дедупликацию: {len(selected):,}")

        min_gap    = self.match_gap.get()
        max_unique = getattr(self, '_max_unique_results', 1000)

        # Словарь {video_idx: sorted list of used times}
        # Бинарный поиск: O(log n) вместо O(n) перебора
        used_times: dict = {}

        def _is_close(vid: int, t: float) -> bool:
            arr = used_times.get(vid)
            if not arr:
                return False
            idx = bisect.bisect_left(arr, t)
            if idx < len(arr) and abs(arr[idx] - t) < min_gap:
                return True
            if idx > 0 and abs(arr[idx - 1] - t) < min_gap:
                return True
            return False

        def _mark_used(vid: int, t: float):
            if vid not in used_times:
                used_times[vid] = []
            bisect.insort(used_times[vid], t)

        unique: list = []

        for m in selected:
            t1, t2 = m['t1'], m['t2']
            v1, v2 = m['v1_idx'], m['v2_idx']

            if _is_close(v1, t1) or _is_close(v2, t2):
                continue

            unique.append(m)
            _mark_used(v1, t1)
            _mark_used(v2, t2)

            if len(unique) >= max_unique:
                break

        print(f"Уникальных после дедупликации: {len(unique)}")
        return unique

    # ══════════════════════════════════════════════════════════════
    # ЗАВЕРШЕНИЕ И ОТОБРАЖЕНИЕ
    # ══════════════════════════════════════════════════════════════

    def _on_analysis_complete(self):
        self.analysis_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.progress.stop_animation()

        if self.matches:
            self.status_label.config(
                text=f"Готово. Найдено {len(self.matches)} повторов")
            self._display_results()
            self._draw_heatmap()
        else:
            self.status_label.config(
                text="Повторений не найдено. Попробуйте снизить порог")
            self.results_list.delete(0, tk.END)
            self.results_list.insert(tk.END, "Повторений не найдено")

        self.progress.set(100)
        self.progress_label.config(text="100%")
        self.metric_values['time_left'].set("00:00")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _update_progress(self, percent: float, remaining: float,
                          current_frames: int, total_frames: int):
        self.progress.set(percent)
        self.progress_label.config(text=f"{percent:.0f}%")
        self.metric_values['frames'].set(
            f"{self._fmt_num(current_frames)}/{self._fmt_num(total_frames)}")
        self.metric_values['time_left'].set(self._fmt_time(remaining))

    def _display_results(self):
        self.results_list.delete(0, tk.END)
        if not self.matches:
            return

        arrow_map = {"left": "←", "right": "→", "forward": "↑",
                     "unknown": "?"}

        for m in self.matches:
            t1    = self._fmt_time(m['t1'])
            t2    = self._fmt_time(m['t2'])
            d     = m.get('direction', 'forward')
            arrow = arrow_map.get(d, "")

            if self.batch_mode:
                v1 = os.path.basename(
                    self.video_queue[m['v1_idx']])[:14]
                v2 = os.path.basename(
                    self.video_queue[m['v2_idx']])[:14]
                display = (f"{arrow} {v1} ({t1}) ↔ "
                           f"{v2} ({t2})  {m['sim']:.0%}")
            else:
                display = f"{arrow} {t1} → {t2}  {m['sim']:.0%}"

            self.results_list.insert(tk.END, display)

        self.metric_values['patterns'].set(str(len(self.matches)))
        avg_sim = np.mean([m['sim'] for m in self.matches]) * 100
        self.metric_values['accuracy'].set(f"{avg_sim:.0f}%")

        self._show_preview(0)

    def _draw_heatmap(self):
        self.timeline_canvas.delete("all")
        if not self.matches or not self.video_queue:
            return

        self.timeline_canvas.update_idletasks()
        canvas_w = self.timeline_canvas.winfo_width() or 800

        video_durations = [self._get_video_duration(p)
                           for p in self.video_queue]
        total_duration  = sum(video_durations)
        if total_duration <= 0:
            return

        # Разделители файлов
        cumulative = 0.0
        for i, dur in enumerate(video_durations):
            if i > 0:
                x = (cumulative / total_duration) * canvas_w
                self.timeline_canvas.create_line(
                    x, 0, x, 40, fill=self.colors['border'], width=1)
            cumulative += dur

        start_times = np.cumsum([0.0] + video_durations[:-1])

        for m in self.matches:
            sim = m['sim']
            x1  = ((start_times[m['v1_idx']] + m['t1']) / total_duration) * canvas_w
            x2  = ((start_times[m['v2_idx']] + m['t2']) / total_duration) * canvas_w

            # Цвет: от синего (слабое) к красному (сильное)
            intensity = max(0.0, (sim - 0.7) / 0.3)
            r = int(90  + 165 * intensity)
            g = int(80  +  95 * intensity)
            b = int(255 - 100 * intensity)
            color = f"#{r:02x}{g:02x}{b:02x}"

            self.timeline_canvas.create_line(
                x1, 0, x1, 40, fill=color, width=2)
            self.timeline_canvas.create_line(
                x2, 0, x2, 40, fill=color, width=2)

    def _show_preview(self, index: int):
        if not self.matches:
            return
        if not (0 <= index < len(self.matches)):
            return

        self.current_match = index
        m = self.matches[index]

        self.time1_label.config(text=self._fmt_time(m['t1']))
        self.time2_label.config(text=self._fmt_time(m['t2']))

        d     = m.get('direction', 'forward')
        arrow = {"left": "←", "right": "→", "forward": "↑",
                 "unknown": "?"}.get(d, "")
        self.match_info.config(text=f"{arrow} {d} • {m['sim']:.0%}")
        self.action1_label.config(text=d)
        self.action2_label.config(text=d)
        self.match_counter.config(
            text=f"{index + 1}/{len(self.matches)}")

        img1 = self._load_frame_with_cache(m['v1_idx'], m['f1'])
        img2 = self._load_frame_with_cache(m['v2_idx'], m['f2'])

        if img1:
            self.preview1.config(image=img1)
            self.preview1.image = img1
        if img2:
            self.preview2.config(image=img2)
            self.preview2.image = img2

        self.results_list.selection_clear(0, tk.END)
        self.results_list.selection_set(index)
        self.results_list.see(index)

    def _load_frame_with_cache(self, video_idx: int,
                                frame_num: int) -> 'ImageTk.PhotoImage | None':
        """Загрузить кадр из видео с дисковым кешом."""
        if video_idx < 0 or video_idx >= len(self.video_queue):
            return None

        video_path = self.video_queue[video_idx]
        video_hash = hashlib.md5(video_path.encode()).hexdigest()
        cache_path = os.path.join(
            self.preview_cache_dir, f"{video_hash}_{frame_num}.jpg")

        frame = None
        if os.path.exists(cache_path):
            frame = cv2.imread(cache_path)

        if frame is None:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, raw = cap.read()
                cap.release()
                if ret and raw is not None:
                    max_w, max_h = 520, 320
                    h, w = raw.shape[:2]
                    scale = min(max_w / w, max_h / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(raw, (new_w, new_h),
                                       interpolation=cv2.INTER_AREA)
                    cv2.imwrite(cache_path, frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 82])

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return ImageTk.PhotoImage(Image.fromarray(rgb))
        return None

    # ══════════════════════════════════════════════════════════════
    # НАВИГАЦИЯ И СОБЫТИЯ
    # ══════════════════════════════════════════════════════════════

    def on_select(self, event):
        sel = self.results_list.curselection()
        if sel:
            self._show_preview(sel[0])

    def on_timeline_click(self, event):
        if not self.matches or not self.video_queue:
            return

        canvas_w = self.timeline_canvas.winfo_width()
        if canvas_w <= 0:
            return

        video_durations = [self._get_video_duration(p)
                           for p in self.video_queue]
        total_duration  = sum(video_durations)
        if total_duration <= 0:
            return

        clicked_time = (event.x / canvas_w) * total_duration
        start_times  = np.cumsum([0.0] + video_durations[:-1])

        closest_idx = -1
        min_dist    = float('inf')

        for i, m in enumerate(self.matches):
            m_t1 = start_times[m['v1_idx']] + m['t1']
            m_t2 = start_times[m['v2_idx']] + m['t2']
            dist = min(abs(clicked_time - m_t1), abs(clicked_time - m_t2))
            if dist < min_dist:
                min_dist    = dist
                closest_idx = i

        if closest_idx != -1:
            self._show_preview(closest_idx)

    def prev_match(self):
        if self.current_match > 0:
            self._show_preview(self.current_match - 1)

    def next_match(self):
        if self.current_match < len(self.matches) - 1:
            self._show_preview(self.current_match + 1)

    # ══════════════════════════════════════════════════════════════
    # ЭКСПОРТ
    # ══════════════════════════════════════════════════════════════

    def export_json(self):
        if not self.matches:
            messagebox.showwarning("Нет данных", "Нет данных для экспорта")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return

        clean = []
        for m in self.matches:
            row = m.copy()
            row.pop('vec', None)   # убираем тяжёлые данные
            clean.append(row)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'matches': clean}, f, indent=2, ensure_ascii=False)
        messagebox.showinfo("Экспорт", f"JSON сохранён:\n{path}")

    def export_txt(self):
        if not self.matches:
            messagebox.showwarning("Нет данных", "Нет данных для экспорта")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text File", "*.txt")],
        )
        if not path:
            return

        with open(path, 'w', encoding='utf-8') as f:
            f.write("PARALLEL FINDER — РЕЗУЛЬТАТЫ\n" + "=" * 44 + "\n")
            for i, m in enumerate(self.matches):
                v1 = os.path.basename(self.video_queue[m['v1_idx']])
                v2 = os.path.basename(self.video_queue[m['v2_idx']])
                f.write(
                    f"{i + 1:03d}. {m['sim']:.1%} | "
                    f"{v1} @ {self._fmt_time(m['t1'])} ↔ "
                    f"{v2} @ {self._fmt_time(m['t2'])}\n"
                )
        messagebox.showinfo("Экспорт", f"TXT сохранён:\n{path}")

    def export_edl(self):
        if not self.matches:
            messagebox.showwarning("Нет данных", "Нет данных для экспорта в EDL")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".edl",
            filetypes=[("Edit Decision List", "*.edl")],
        )
        if not path:
            return

        try:
            cap = cv2.VideoCapture(self.video_queue[0])
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
        except Exception:
            fps = 30.0

        def timecode(seconds: float) -> str:
            s   = max(0.0, seconds)
            h   = int(s / 3600)
            m   = int((s % 3600) / 60)
            sec = int(s % 60)
            fr  = int((s - int(s)) * fps)
            return f"{h:02d}:{m:02d}:{sec:02d}:{fr:02d}"

        with open(path, 'w', encoding='utf-8') as f:
            f.write("TITLE: Parallel Finder Export\nFCM: NON-DROP FRAME\n\n")
            timeline = 3600.0
            for i, m in enumerate(self.matches):
                clip1 = os.path.basename(self.video_queue[m['v1_idx']])
                clip2 = os.path.basename(self.video_queue[m['v2_idx']])

                f.write(
                    f"{i * 2 + 1:03d}  AX       V     C        "
                    f"{timecode(m['t1'])} {timecode(m['t1'] + 2)} "
                    f"{timecode(timeline)} {timecode(timeline + 2)}\n"
                )
                f.write(f"* FROM CLIP NAME: {clip1}\n")

                f.write(
                    f"{i * 2 + 2:03d}  AX       V     C        "
                    f"{timecode(m['t2'])} {timecode(m['t2'] + 2)} "
                    f"{timecode(timeline + 2)} {timecode(timeline + 4)}\n"
                )
                f.write(f"* FROM CLIP NAME: {clip2}\n\n")
                timeline += 5.0

        messagebox.showinfo("Экспорт", f"EDL сохранён:\n{path}")

    # ══════════════════════════════════════════════════════════════
    # ЗАПУСК
    # ══════════════════════════════════════════════════════════════

    def run(self):
        self.root.mainloop()


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        app = ParallelFinderApp()
        app.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        messagebox.showerror("Критическая ошибка", str(e))
