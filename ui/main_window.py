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

from core.engine import YoloEngine
from core.project import ProjectManager

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

THEMES = {
    'dark': {
        'bg': '#0a0c10',
        'card': '#14171c',
        'text': '#ffffff',
        'text_secondary': '#8a8f99',
        'accent': '#3b82f6',
        'accent_hover': '#60a5fa',
        'success': '#10b981',
        'error': '#ef4444',
        'border': '#1f2937',
        'highlight': '#1e293b',
        'glow': '#3b82f6'
    }
}

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


class AnimatedProgressbar(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(style='Animated.TFrame')
        self.canvas = tk.Canvas(self, height=8, bg=THEMES['dark']['bg'], highlightthickness=0)
        self.canvas.pack(fill=tk.X, expand=True)
        self.value = 0
        self.anim_frame = 0
        self._anim_running = False
        
    def set(self, value):
        self.value = min(100, max(0, value))
        self.redraw()
        
    def redraw(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        if w <= 0:
            w = 400
        
        fill_width = (self.value / 100.0) * w
        
        self.canvas.create_rectangle(0, 0, w, 8, fill=THEMES['dark']['border'], outline='')
        
        if self.value > 0:
            for i in range(3):
                self.canvas.create_rectangle(
                    fill_width - 15 + i*5, 0, fill_width + 5 + i*5, 8,
                    fill=THEMES['dark']['accent'], outline='',
                    stipple='gray25'
                )
            
            self.canvas.create_rectangle(0, 0, fill_width, 8, fill=THEMES['dark']['accent'], outline='')
            
            if self.value > 5 and self.value < 95:
                self.canvas.create_rectangle(
                    fill_width - 10, 2, fill_width, 6,
                    fill=THEMES['dark']['glow'], outline=''
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
        self.anim_frame += 1
        if self.anim_frame % 8 == 0:
            self.redraw()
        self.after(16, self._animate)


class GlowButton(tk.Canvas):
    def __init__(self, parent, text, command, bg_color, hover_color, width=120, height=32, state='normal'):
        super().__init__(parent, highlightthickness=0, width=width, height=height)
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text = text
        self.state = state
        
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        self.bind("<Configure>", self._on_configure)
        
        self._is_hover = False
        self._draw(False)
        
    def _on_configure(self, event):
        self._draw(self._is_hover)
    
    def _draw(self, is_hover=False):
        self._is_hover = is_hover
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        
        if w <= 1:
            w = int(self['width'])
        if h <= 1:
            h = int(self['height'])
        
        color = self.hover_color if is_hover else self.bg_color
        
        self.create_round_rect(0, 0, w, h, 8, fill=color, outline='')
        self.create_text(w//2, h//2, text=self.text, fill='white', font=('Inter', 10, 'bold'), justify='center')
        
    def create_round_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = []
        points.append((x1 + radius, y1))
        points.append((x2 - radius, y1))
        points.append((x2, y1))
        points.append((x2, y1 + radius))
        points.append((x2, y2 - radius))
        points.append((x2, y2))
        points.append((x2 - radius, y2))
        points.append((x1 + radius, y2))
        points.append((x1, y2))
        points.append((x1, y2 - radius))
        points.append((x1, y1 + radius))
        points.append((x1, y1))
        
        flat_points = []
        for p in points:
            flat_points.extend(p)
        return self.create_polygon(flat_points, smooth=True, **kwargs)
    
    def on_enter(self, e):
        if self.state != 'disabled':
            self._draw(True)
    
    def on_leave(self, e):
        if self.state != 'disabled':
            self._draw(False)
    
    def on_click(self, e):
        if self.state != 'disabled' and self.command:
            self.command()
    
    def config(self, **kwargs):
        if 'state' in kwargs:
            self.state = kwargs['state']
            if self.state == 'disabled':
                self._draw(False)
        super().config(**kwargs)


class ParallelFinderApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Parallel Finder")
        self.root.geometry("1600x900")
        self.root.minsize(1400, 800)
        self.colors = THEMES['dark']
        self.root.configure(bg=self.colors['bg'])
        
        self.threshold = tk.IntVar(value=70)
        self.scene_interval = tk.IntVar(value=3)
        self.motion_length = tk.DoubleVar(value=2.0)
        self.frame_skip = tk.IntVar(value=2)
        self.quality = tk.StringVar(value='Средне')
        self.use_scale_invariance = tk.BooleanVar(value=True)
        self.use_mirror_invariance = tk.BooleanVar(value=False)
        self.use_body_weights = tk.BooleanVar(value=False)
        
        self.yolo = YoloEngine()
        self.yolo.load()
        
        self.video_path, self.video_queue = None, []
        self.batch_mode, self.analysis_running = False, False
        self.poses_tensor, self.poses_meta = None, []
        self.matches, self.current_match = [], 0
        self.current_batch_index, self.total_batch_videos = 0, 0
        
        # Параметры по умолчанию (будут изменены автонастройкой)
        self.BATCH_SIZE = 64
        self.CHUNK_SIZE = 5000
        self.CHUNK_OVERLAP = 500
        self.MIN_MATCH_GAP = 5.0
        self.max_matches_per_chunk = 2000000
        self.max_total_matches = 30000000

        self.preview_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "previews")
        os.makedirs(self.preview_cache_dir, exist_ok=True)

        self.value_labels = {}
        self.metric_values = {}

        self.setup_ui()
        self.setup_hotkeys()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Автонастройка под железо
        self._auto_tune_hardware()

    def _auto_tune_hardware(self):
        """Автоматическая настройка параметров под железо пользователя"""
        print("\n" + "="*50)
        print("🔧 АВТОНАСТРОЙКА ПОД ЖЕЛЕЗО")
        print("="*50)
        
        # Определяем CPU
        cpu_cores = psutil.cpu_count(logical=True)
        cpu_name = platform.processor()
        ram_gb = psutil.virtual_memory().total / 1e9
        print(f"💻 CPU: {cpu_name or 'Unknown'} ({cpu_cores} ядер)")
        print(f"💾 RAM: {ram_gb:.1f} GB")
        
        # Определяем GPU
        gpu_available = torch.cuda.is_available()
        vram_gb = 0
        gpu_name = "Не обнаружен"
        
        if gpu_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"🖥️ GPU: {gpu_name}")
                print(f"💾 VRAM: {vram_gb:.1f} GB")
            except Exception as e:
                print(f"⚠️ Ошибка определения GPU: {e}")
                gpu_available = False
        
        # ========== РЕКОМЕНДУЕМЫЕ НАСТРОЙКИ ==========
        # Словарь конфигураций для разных типов железа
        configs = {
            # Слабые GPU (GTX 1050, 950, 2-4GB)
            'weak_gpu': {
                'batch': 8,
                'chunk': 1500,
                'quality': 'Быстро',
                'matches_per_chunk': 300000,
                'total_matches': 5000000,
                'desc': 'Эконом (для слабых GPU, 2-4GB VRAM)'
            },
            # Средние GPU (GTX 1060, 1650, 4-6GB)
            'medium_gpu': {
                'batch': 24,
                'chunk': 2500,
                'quality': 'Средне',
                'matches_per_chunk': 1000000,
                'total_matches': 15000000,
                'desc': 'Средний (GTX 1060/1660, 4-6GB VRAM)'
            },
            # Хорошие GPU (RTX 2060, 3060, 8GB)
            'good_gpu': {
                'batch': 48,
                'chunk': 4000,
                'quality': 'Средне',
                'matches_per_chunk': 1500000,
                'total_matches': 20000000,
                'desc': 'Высокий (RTX 2060/3060, 8GB VRAM)'
            },
            # Мощные GPU (RTX 3070+, 12GB+)
            'powerful_gpu': {
                'batch': 64,
                'chunk': 5000,
                'quality': 'Макс',
                'matches_per_chunk': 2000000,
                'total_matches': 30000000,
                'desc': 'Максимум (RTX 3070+, 12GB+ VRAM)'
            },
            # CPU режим (без GPU)
            'cpu_mode': {
                'batch': 8,
                'chunk': 1500,
                'quality': 'Быстро',
                'matches_per_chunk': 300000,
                'total_matches': 5000000,
                'desc': 'CPU режим (медленно, но стабильно)'
            }
        }
        
        # Выбираем конфигурацию
        selected_config = None
        
        if not gpu_available:
            selected_config = configs['cpu_mode']
            print(f"\n⚡ Выбран режим: {selected_config['desc']}")
        elif vram_gb < 4:
            selected_config = configs['weak_gpu']
            print(f"\n⚡ Выбран режим: {selected_config['desc']}")
        elif vram_gb < 8:
            selected_config = configs['medium_gpu']
            print(f"\n⚡ Выбран режим: {selected_config['desc']}")
        elif vram_gb < 12:
            selected_config = configs['good_gpu']
            print(f"\n⚡ Выбран режим: {selected_config['desc']}")
        else:
            selected_config = configs['powerful_gpu']
            print(f"\n⚡ Выбран режим: {selected_config['desc']}")
        
        # Применяем настройки
        self.BATCH_SIZE = selected_config['batch']
        self.CHUNK_SIZE = selected_config['chunk']
        self.quality.set(selected_config['quality'])
        self.max_matches_per_chunk = selected_config['matches_per_chunk']
        self.max_total_matches = selected_config['total_matches']
        
        print(f"   BATCH_SIZE: {self.BATCH_SIZE}")
        print(f"   CHUNK_SIZE: {self.CHUNK_SIZE}")
        print(f"   Качество: {self.quality.get()}")
        print(f"   Лимит на чанк: {self.max_matches_per_chunk:,}")
        print(f"   Общий лимит: {self.max_total_matches:,}")
        print("="*50 + "\n")

    def on_closing(self):
        self.root.destroy()

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
            'frames': tk.StringVar(value="0"),
            'patterns': tk.StringVar(value="0"),
            'duration': tk.StringVar(value="0:00"),
            'accuracy': tk.StringVar(value="0%"),
            'time_left': tk.StringVar(value="--:--"),
            'batch_progress': tk.StringVar(value="0/0")
        }
        
        metrics = [
            ('Кадров', 'frames'),
            ('Повторов', 'patterns'),
            ('Длительность', 'duration'),
            ('Схожесть', 'accuracy'),
            ('Осталось', 'time_left'),
            ('Прогресс', 'batch_progress')
        ]
        
        for i, (label, var) in enumerate(metrics):
            card = tk.Frame(stats, bg=self.colors['card'], padx=15, pady=12)
            card.pack(side=tk.LEFT, padx=(0 if i == 0 else 8, 0), expand=True, fill=tk.BOTH)
            
            top = tk.Frame(card, bg=self.colors['card'])
            top.pack(fill=tk.X)
            
            tk.Label(top, textvariable=self.metric_values[var],
                     font=('Inter', 20, 'bold'), bg=self.colors['card'], fg=self.colors['text']).pack(side=tk.LEFT)
            tk.Label(card, text=label, font=('Inter', 10),
                     bg=self.colors['card'], fg=self.colors['text_secondary']).pack(anchor='w')

    def setup_video_panel(self, parent):
        card = tk.Frame(parent, bg=self.colors['card'])
        card.pack(fill=tk.X, pady=(0, 16))
        
        inner = tk.Frame(card, bg=self.colors['card'], padx=20, pady=20)
        inner.pack(fill=tk.X)
        
        tk.Label(inner, text="Источник", font=('Inter', 14, 'bold'),
                 bg=self.colors['card'], fg=self.colors['text']).pack(anchor='w', pady=(0, 12))
        
        self.video_label = tk.Label(inner, text="Файл не выбран",
                                     font=('Inter', 11), bg=self.colors['card'],
                                     fg=self.colors['text_secondary'], wraplength=320)
        self.video_label.pack(anchor='w', pady=(0, 4))
        
        self.video_info = tk.Label(inner, text="", font=('Inter', 10),
                                    bg=self.colors['card'], fg=self.colors['text_secondary'])
        self.video_info.pack(anchor='w', pady=(0, 12))
        
        btn_frame = tk.Frame(inner, bg=self.colors['card'])
        btn_frame.pack(fill=tk.X, pady=(0, 5))
        
        btn1 = GlowButton(btn_frame, "Выбрать видео", self.select_video,
                          self.colors['accent'], self.colors['accent_hover'], width=120, height=34)
        btn1.pack(side=tk.LEFT, padx=(0, 8))
        
        btn2 = GlowButton(btn_frame, "Выбрать папку", self.select_folder,
                          self.colors['success'], '#2ecc71', width=120, height=34)
        btn2.pack(side=tk.LEFT)
        
        self.batch_info = tk.Label(inner, text="", font=('Inter', 9),
                                    bg=self.colors['card'], fg=self.colors['text_secondary'])
        self.batch_info.pack(anchor='w', pady=(8, 0))

    def setup_settings_panel(self, parent):
        container = tk.Frame(parent, bg=self.colors['card'])
        container.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(container, bg=self.colors['card'], highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=self.colors['card'])
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        card = scrollable
        
        tk.Label(card, text="Настройки анализа", font=('Inter', 14, 'bold'),
                 bg=self.colors['card'], fg=self.colors['text']).pack(anchor='w', padx=20, pady=(20, 8))
        
        tk.Frame(card, bg=self.colors['border'], height=1).pack(fill=tk.X, padx=20, pady=(10, 16))
        
        g1 = tk.Frame(card, bg=self.colors['card'])
        g1.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        tk.Label(g1, text="Точность", font=('Inter', 12, 'bold'),
                 bg=self.colors['card'], fg=self.colors['text']).pack(anchor='w', pady=(0, 10))
        self.add_setting_row(g1, "Порог схожести", self.threshold, 50, 98, 70, "%")
        
        q_frame = tk.Frame(g1, bg=self.colors['card'])
        q_frame.pack(fill=tk.X, pady=(0, 12))
        tk.Label(q_frame, text="Качество", font=('Inter', 10),
                 bg=self.colors['card'], fg=self.colors['text_secondary']).pack(anchor='w')
        self.quality_combo = ttk.Combobox(q_frame, textvariable=self.quality,
                                           values=['Быстро', 'Средне', 'Макс'],
                                           font=('Inter', 10), state='readonly')
        self.quality_combo.pack(fill=tk.X, pady=(4, 0))
        
        g2 = tk.Frame(card, bg=self.colors['card'])
        g2.pack(fill=tk.X, padx=20, pady=(0, 20))
        tk.Label(g2, text="Временные параметры", font=('Inter', 12, 'bold'),
                 bg=self.colors['card'], fg=self.colors['text']).pack(anchor='w', pady=(0, 10))
        self.add_setting_row(g2, "Интервал сцен", self.scene_interval, 1, 30, 3, "сек")
        
        g3 = tk.Frame(card, bg=self.colors['card'])
        g3.pack(fill=tk.X, padx=20, pady=(0, 20))
        tk.Label(g3, text="Расширенные опции", font=('Inter', 12, 'bold'),
                 bg=self.colors['card'], fg=self.colors['text']).pack(anchor='w', pady=(0, 10))
        tk.Checkbutton(g3, text="Нормализация по размеру", variable=self.use_scale_invariance,
                       bg=self.colors['card'], fg=self.colors['text_secondary'],
                       selectcolor=self.colors['bg'], activebackground=self.colors['card']).pack(anchor='w')
        tk.Checkbutton(g3, text="Учитывать зеркальное отражение", variable=self.use_mirror_invariance,
                       bg=self.colors['card'], fg=self.colors['text_secondary'],
                       selectcolor=self.colors['bg'], activebackground=self.colors['card']).pack(anchor='w')
        tk.Checkbutton(g3, text="Веса частей тела", variable=self.use_body_weights,
                       bg=self.colors['card'], fg=self.colors['text_secondary'],
                       selectcolor=self.colors['bg'], activebackground=self.colors['card']).pack(anchor='w')
        
        tk.Frame(card, bg=self.colors['border'], height=1).pack(fill=tk.X, padx=20, pady=(10, 16))
        
        btn_frame = tk.Frame(card, bg=self.colors['card'])
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        self.start_btn = GlowButton(btn_frame, "Старт", self.start_analysis,
                                      self.colors['success'], '#2ecc71', width=400, height=42)
        self.start_btn.pack(fill=tk.X, pady=(0, 8))
        
        self.stop_btn = GlowButton(btn_frame, "Стоп", self.stop_analysis,
                                     self.colors['error'], '#ff6b6b', width=400, height=38)
        self.stop_btn.pack(fill=tk.X)

    def add_setting_row(self, parent, label, var, from_, to, default, unit, step=1):
        frame = tk.Frame(parent, bg=self.colors['card'])
        frame.pack(fill=tk.X, pady=(0, 12))
        
        top = tk.Frame(frame, bg=self.colors['card'])
        top.pack(fill=tk.X, pady=(0, 4))
        
        tk.Label(top, text=label, font=('Inter', 10),
                 bg=self.colors['card'], fg=self.colors['text_secondary']).pack(side=tk.LEFT)
        
        value_label = tk.Label(top, text=f"{default}{unit}", font=('Inter', 14, 'bold'),
                                bg=self.colors['card'], fg=self.colors['accent'])
        value_label.pack(side=tk.RIGHT)
        self.value_labels[label] = value_label
        
        scale = tk.Scale(frame, from_=from_, to=to, resolution=step, orient=tk.HORIZONTAL,
                         variable=var, bg=self.colors['card'], fg=self.colors['text'],
                         highlightbackground=self.colors['card'], troughcolor=self.colors['bg'],
                         sliderrelief='flat', sliderlength=18, width=6,
                         command=lambda v, l=label, u=unit: self.update_value(l, v, u))
        var.set(default)
        scale.pack(fill=tk.X)

    def update_value(self, label, value, unit):
        if label in self.value_labels:
            self.value_labels[label].config(text=f"{int(float(value))}{unit}")

    def setup_preview_panel(self, parent):
        progress_card = tk.Frame(parent, bg=self.colors['card'])
        progress_card.pack(fill=tk.X, pady=(0, 12))
        
        progress_inner = tk.Frame(progress_card, bg=self.colors['card'], padx=16, pady=16)
        progress_inner.pack(fill=tk.X)
        
        header = tk.Frame(progress_inner, bg=self.colors['card'])
        header.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(header, text="Прогресс", font=('Inter', 13, 'bold'),
                 bg=self.colors['card'], fg=self.colors['text']).pack(side=tk.LEFT)
        
        self.progress_label = tk.Label(header, text="0%", font=('Inter', 14, 'bold'),
                                        bg=self.colors['card'], fg=self.colors['accent'])
        self.progress_label.pack(side=tk.RIGHT)
        
        self.progress = AnimatedProgressbar(progress_inner)
        self.progress.pack(fill=tk.X, pady=(8, 4))
        
        self.status_label = tk.Label(progress_inner, text="Ожидание", font=('Inter', 10),
                                      bg=self.colors['card'], fg=self.colors['text_secondary'])
        self.status_label.pack(anchor='w')
        
        preview_card = tk.Frame(parent, bg=self.colors['card'])
        preview_card.pack(fill=tk.BOTH, expand=True)
        
        preview_inner = tk.Frame(preview_card, bg=self.colors['card'], padx=16, pady=16)
        preview_inner.pack(fill=tk.BOTH, expand=True)
        
        title_frame = tk.Frame(preview_inner, bg=self.colors['card'])
        title_frame.pack(fill=tk.X, pady=(0, 12))
        
        tk.Label(title_frame, text="Сравнение", font=('Inter', 13, 'bold'),
                 bg=self.colors['card'], fg=self.colors['text']).pack(side=tk.LEFT)
        
        self.match_info = tk.Label(title_frame, text="", font=('Inter', 10),
                                    bg=self.colors['card'], fg=self.colors['text_secondary'])
        self.match_info.pack(side=tk.RIGHT)
        
        video_row = tk.Frame(preview_inner, bg=self.colors['bg'])
        video_row.pack(fill=tk.BOTH, expand=True)
        
        self.preview1, self.time1_label, self.action1_label = self._create_video_widget(video_row)
        self.preview2, self.time2_label, self.action2_label = self._create_video_widget(video_row)
        
        self.setup_timeline_panel(preview_inner)

    def _create_video_widget(self, parent):
        video_frame = tk.Frame(parent, bg=self.colors['bg'])
        side = tk.LEFT if not parent.winfo_children() else tk.RIGHT
        video_frame.pack(side=side, padx=8, expand=True, fill=tk.BOTH)
        
        container = tk.Frame(video_frame, bg=self.colors['bg'])
        container.pack(fill=tk.BOTH, expand=True)
        
        preview = tk.Label(container, bg=self.colors['bg'])
        preview.pack(fill=tk.BOTH, expand=True)
        
        time_label = tk.Label(container, text="--:--", font=('Inter', 14, 'bold'),
                               bg='#1e1e2e', fg='white', padx=6, pady=2)
        time_label.place(relx=0.02, rely=0.02)
        
        action_label = tk.Label(container, text="", font=('Inter', 10),
                                 bg='#1e1e2e', fg=self.colors['accent'], padx=6, pady=2)
        action_label.place(relx=0.02, rely=0.12)
        
        return preview, time_label, action_label

    def setup_timeline_panel(self, parent):
        timeline_frame = tk.Frame(parent, bg=self.colors['card'], pady=5)
        timeline_frame.pack(fill=tk.X, pady=(12, 0))
        
        tk.Label(timeline_frame, text="Таймлайн", font=('Inter', 10),
                 bg=self.colors['card'], fg=self.colors['text_secondary']).pack(side=tk.LEFT, padx=(0, 10))
        
        self.timeline_canvas = tk.Canvas(timeline_frame, height=40, bg=self.colors['bg'], highlightthickness=0)
        self.timeline_canvas.pack(fill=tk.X, expand=True)
        self.timeline_canvas.bind("<Button-1>", self.on_timeline_click)

    def setup_results_panel(self, parent):
        card = tk.Frame(parent, bg=self.colors['card'])
        card.pack(fill=tk.BOTH, expand=True)
        
        inner = tk.Frame(card, bg=self.colors['card'], padx=16, pady=16)
        inner.pack(fill=tk.BOTH, expand=True)
        
        header = tk.Frame(inner, bg=self.colors['card'])
        header.pack(fill=tk.X, pady=(0, 12))
        
        tk.Label(header, text="Результаты", font=('Inter', 13, 'bold'),
                 bg=self.colors['card'], fg=self.colors['text']).pack(side=tk.LEFT)
        
        nav_frame = tk.Frame(inner, bg=self.colors['card'])
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.prev_btn = GlowButton(nav_frame, "Предыдущий", self.prev_match,
                                     self.colors['highlight'], self.colors['border'], width=100, height=30)
        self.prev_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        
        self.match_counter = tk.Label(nav_frame, text="0/0", font=('Inter', 10),
                                       bg=self.colors['card'], fg=self.colors['text'])
        self.match_counter.pack(side=tk.LEFT, padx=8)
        
        self.next_btn = GlowButton(nav_frame, "Следующий", self.next_match,
                                     self.colors['highlight'], self.colors['border'], width=100, height=30)
        self.next_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))
        
        list_frame = tk.Frame(inner, bg=self.colors['bg'])
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_list = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                        bg=self.colors['bg'], fg=self.colors['text'],
                                        selectbackground=self.colors['accent'],
                                        font=('Inter', 10), bd=0,
                                        highlightthickness=1, highlightbackground=self.colors['border'])
        self.results_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_list.yview)
        self.results_list.bind('<<ListboxSelect>>', self.on_select)
        
        export_frame = tk.Frame(inner, bg=self.colors['card'])
        export_frame.pack(fill=tk.X, pady=(10, 0))
        
        btn_json = GlowButton(export_frame, "JSON", self.export_json, self.colors['highlight'], self.colors['border'], width=60, height=30)
        btn_json.pack(side=tk.LEFT, padx=(0, 4))
        btn_txt = GlowButton(export_frame, "TXT", self.export_txt, self.colors['highlight'], self.colors['border'], width=60, height=30)
        btn_txt.pack(side=tk.LEFT, padx=(0, 4))
        btn_edl = GlowButton(export_frame, "EDL", self.export_edl, self.colors['highlight'], self.colors['border'], width=60, height=30)
        btn_edl.pack(side=tk.LEFT)

    def setup_hotkeys(self):
        self.root.bind('<space>', lambda e: self.start_analysis())
        self.root.bind('<Up>', lambda e: self.prev_match())
        self.root.bind('<Down>', lambda e: self.next_match())

    def select_video(self):
        try:
            path = filedialog.askopenfilename(filetypes=[("Видео", "*.mp4 *.avi *.mkv *.mov *.ts *.webm *.flv *.m4v *.wmv *.3gp")])
            if path:
                self.video_queue = [path]
                self.batch_mode = False
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
                self.video_queue = [os.path.join(folder, f) for f in os.listdir(folder)
                                    if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.ts', '.webm', '.flv', '.m4v', '.wmv', '.3gp'))]
                if self.video_queue:
                    self.batch_mode = True
                    self.video_label.config(text=f"Папка: {os.path.basename(folder)}")
                    self.batch_info.config(text=f"Видео: {len(self.video_queue)}")
                    self.update_video_info(self.video_queue[0])
                    self.video_path = self.video_queue[0]
                    self._reset_state()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть папку: {e}")

    def _reset_state(self):
        self.matches = []
        self.poses_tensor, self.poses_meta = None, []
        self.results_list.delete(0, tk.END)
        self.timeline_canvas.delete("all")
        self.metric_values['patterns'].set("0")
        self.metric_values['accuracy'].set("0%")
        self.root.title("Parallel Finder")

    def update_video_info(self, path):
        try:
            cap = cv2.VideoCapture(path)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            duration = frames / fps
            cap.release()
            self.metric_values['frames'].set(self._format_number(frames))
            self.metric_values['duration'].set(self._format_time(duration))
            self.video_info.config(text=f"{self._format_number(frames)} кадров • {fps:.0f} fps")
        except:
            pass

    def _format_number(self, num):
        return f"{num/1000:.1f}K" if num >= 1000 else str(num)

    def _format_time(self, seconds):
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    def start_analysis(self):
        if not self.video_queue:
            messagebox.showwarning("Нет видео", "Выберите видео или папку для анализа")
            return
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
            all_poses_meta = []
            all_poses_vecs = []
            
            for idx, path in enumerate(self.video_queue):
                if not self.analysis_running:
                    break
                
                self.current_batch_index = idx
                self.root.after(0, lambda p=path, i=idx: self._update_batch_status(p, i))
                
                poses_meta, poses_vecs = self._extract_poses_from_video(path)
                
                for meta in poses_meta:
                    meta['video_idx'] = idx
                
                all_poses_meta.extend(poses_meta)
                all_poses_vecs.extend(poses_vecs)

            if not self.analysis_running or not all_poses_vecs:
                self.root.after(0, self._on_analysis_complete)
                return

            self.poses_meta = all_poses_meta
            self.poses_tensor = torch.tensor(np.array(all_poses_vecs), dtype=torch.float16 if self.yolo.use_fp16 else torch.float32, device=self.yolo.device)
            
            self.root.after(0, lambda: self.status_label.config(text=f"Поиск совпадений среди {len(self.poses_tensor)} поз..."))
            self.matches = self._find_matches_tensor_chunked()

            self.root.after(0, self._on_analysis_complete)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
            self.root.after(0, self._on_analysis_complete)

    def _update_batch_status(self, path, idx):
        self.status_label.config(text=f"{idx+1}/{self.total_batch_videos}: {os.path.basename(path)}")
        self.metric_values['batch_progress'].set(f"{idx+1}/{self.total_batch_videos}")

    def _extract_poses_from_video(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        quality = self.quality.get()
        base_fps = {'Быстро': 10, 'Средне': 15, 'Макс': 30}[quality]
        
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        pixel_load = (h * w * fps) / 1e6
        res_factor = min(2.0, pixel_load / 50)
        adaptive_skip = max(1, int((fps / base_fps) * res_factor))
        skip = min(adaptive_skip, 10)

        start_time = time.time()
        frame_idx = 0
        
        poses_meta = []
        poses_vecs = []
        
        batch_frames, batch_meta_pre = [], []
        video_hash = hashlib.md5(path.encode()).hexdigest()

        while cap.isOpened() and self.analysis_running:
            ret = cap.grab()
            if not ret:
                break

            if frame_idx % skip == 0:
                ret, frame = cap.retrieve()
                if ret:
                    h, w = frame.shape[:2]
                    target_size = 640
                    scale = min(target_size / w, target_size / h, 1.0)
                    if scale < 1.0:
                        new_w, new_h = int(w * scale), int(h * scale)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    batch_frames.append(frame)
                    batch_meta_pre.append({'frame': frame_idx, 'time': frame_idx / fps})
            
            dynamic_batch = self._get_dynamic_batch_size([f.shape[:2] for f in batch_frames])
            if len(batch_frames) >= dynamic_batch:
                poses_data = self.yolo.detect_batch(batch_frames)
                for i, (pose_data, meta) in enumerate(zip(poses_data, batch_meta_pre)):
                    if pose_data:
                        processed_vec = self._preprocess_pose(pose_data)
                        poses_meta.append({
                            't': meta['time'],
                            'f': meta['frame'],
                            'dir': pose_data.get('direction', 'forward'),
                            'vec': processed_vec.reshape(17, 2)
                        })
                        poses_vecs.append(processed_vec.flatten())

                        cache_path = os.path.join(self.preview_cache_dir, f"{video_hash}_{meta['frame']}.jpg")
                        if not os.path.exists(cache_path):
                            resized = cv2.resize(batch_frames[i], (320, 180), interpolation=cv2.INTER_AREA)
                            cv2.imwrite(cache_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 80])

                batch_frames.clear()
                batch_meta_pre.clear()
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                elapsed = time.time() - start_time
                speed = frame_idx / elapsed if elapsed > 0 else 0
                total_work = total_frames * len(self.video_queue)
                done_work = frame_idx + self.current_batch_index * total_frames
                remaining = (total_work - done_work) / speed if speed > 0 else 0
                self.root.after(0, lambda p=progress, r=remaining, cf=frame_idx, tf=total_frames: self._update_progress(p, r, cf, tf))

        if batch_frames and self.analysis_running:
            poses_data = self.yolo.detect_batch(batch_frames)
            for i, (pose_data, meta) in enumerate(zip(poses_data, batch_meta_pre)):
                if pose_data:
                    processed_vec = self._preprocess_pose(pose_data)
                    poses_meta.append({
                        't': meta['time'],
                        'f': meta['frame'],
                        'dir': pose_data.get('direction', 'forward'),
                        'vec': processed_vec.reshape(17, 2)
                    })
                    poses_vecs.append(processed_vec.flatten())

        cap.release()
        return poses_meta, poses_vecs

    def _get_dynamic_batch_size(self, frame_sizes):
        if not frame_sizes:
            return self.BATCH_SIZE
        
        avg_area = np.mean([h * w for h, w in frame_sizes]) / 1e6
        
        vram_factor = 1.0
        if torch.cuda.is_available():
            try:
                vram_used = torch.cuda.memory_reserved() / 1e9
                vram_factor = max(0.3, (12 - vram_used) / 8.0)
            except:
                pass
        
        if avg_area > 1.5:
            batch = int(self.BATCH_SIZE * 0.4 * vram_factor)
        elif avg_area > 0.8:
            batch = int(self.BATCH_SIZE * 0.7 * vram_factor)
        elif avg_area < 0.3:
            batch = int(self.BATCH_SIZE * 1.8 * vram_factor)
        else:
            batch = self.BATCH_SIZE
        
        return max(4, min(batch, 192))

    def _preprocess_pose(self, pose_data):
        kps = pose_data['keypoints'][:, :2]
        center = np.mean(kps, axis=0)
        centered = kps - center
        scale = np.max(np.abs(centered)) + 1e-5
        normalized = centered / scale
        return normalized

    def _find_matches_tensor_chunked(self):
        if self.poses_tensor is None or len(self.poses_tensor) < 10:
            return []
        
        device = self.yolo.device
        thresh = self.threshold.get() / 100.0
        min_gap = self.scene_interval.get()
        
        print(f"Поиск среди {len(self.poses_tensor)} поз, порог={thresh}, интервал={min_gap}")
        
        V = self.poses_tensor.to(dtype=torch.float32)
        V = V.view(len(V), -1)
        V = torch.nn.functional.normalize(V, p=2, dim=1)
        
        times = [m['t'] for m in self.poses_meta]
        T = torch.tensor(times, device=device)
        
        matches = []
        chunk_size = self.CHUNK_SIZE
        max_matches_per_chunk = self.max_matches_per_chunk
        max_total_matches = self.max_total_matches
        
        for start in range(0, len(V), chunk_size):
            end = min(start + chunk_size, len(V))
            V_chunk = V[start:end]
            T_chunk = T[start:end]
            
            sim_chunk = torch.mm(V_chunk, V.t())
            
            time_diff_chunk = torch.abs(T_chunk.unsqueeze(1) - T.unsqueeze(0))
            
            valid_mask = (sim_chunk > thresh) & (time_diff_chunk >= min_gap)
            
            if start > 0:
                diag_mask = torch.ones_like(valid_mask, dtype=torch.bool)
                diag_mask[:, :start] = False
                valid_mask = valid_mask & diag_mask
            
            indices = torch.nonzero(valid_mask).cpu().numpy()
            scores = sim_chunk[valid_mask].cpu().numpy()
            
            chunk_matches = 0
            for k in range(len(indices)):
                if len(matches) >= max_total_matches:
                    print(f"Достигнут лимит совпадений ({max_total_matches}), остановка")
                    break
                    
                i_local, j = indices[k]
                i = start + i_local
                
                # Добавляем только если i < j (убираем дубликаты A↔B и B↔A)
                if i < j:
                    matches.append({
                        'm1_idx': i, 'm2_idx': j,
                        't1': self.poses_meta[i]['t'],
                        't2': self.poses_meta[j]['t'],
                        'f1': self.poses_meta[i]['f'],
                        'f2': self.poses_meta[j]['f'],
                        'v1_idx': self.poses_meta[i].get('video_idx', 0),
                        'v2_idx': self.poses_meta[j].get('video_idx', 0),
                        'sim': float(scores[k]),
                        'direction': self.poses_meta[i].get('dir', 'forward')
                    })
                    chunk_matches += 1
                
                if chunk_matches >= max_matches_per_chunk:
                    print(f"Лимит чанка {max_matches_per_chunk}, переход к следующему")
                    break
            
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            gc.collect()
            
            print(f"Чанк {start//chunk_size + 1}: найдено {chunk_matches} пар (всего {len(matches)})")
            
            if len(matches) >= max_total_matches:
                print(f"Достигнут глобальный лимит ({max_total_matches}), завершаем поиск")
                break
        
        print(f"Найдено сырых совпадений: {len(matches)}")
        
        if not matches:
            return []
        
        # ========== УМНЫЙ ФИЛЬТР МУСОРА ==========
        matches.sort(key=lambda x: x['sim'], reverse=True)
        
        good_matches = [m for m in matches if m['sim'] >= 0.85]
        junk_matches = [m for m in matches if m['sim'] < 0.85]
        
        junk_limit = int(len(junk_matches) * 0.2)
        selected = good_matches + junk_matches[:junk_limit]
        
        print(f"✅ Хороших (≥0.85): {len(good_matches)}")
        print(f"🗑️ Мусора (<0.85): {len(junk_matches)} (взято {junk_limit})")
        print(f"📊 Всего на дедупликацию: {len(selected)}")
        
        # Дедупликация
        used_intervals = []
        unique = []
        min_unique_gap = self.scene_interval.get()  # интервал между уникальными повторами
        
        for m in selected:
            is_overlap = False
            t1 = m['t1']
            t2 = m['t2']
            
            for used in used_intervals:
                # Если хотя бы один из моментов слишком близок к уже найденному
                if (abs(t1 - used[0]) < min_unique_gap or abs(t2 - used[1]) < min_unique_gap):
                    is_overlap = True
                    break
            
            if not is_overlap:
                unique.append(m)
                used_intervals.append((t1, t2))
            
            if len(unique) >= 1000:
                break
        
        print(f"🎯 Уникальных после дедупликации: {len(unique)}")
        return unique[:1000]

    def _on_analysis_complete(self):
        self.analysis_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.progress.stop_animation()
        
        if self.matches:
            self.status_label.config(text=f"Готово. Найдено {len(self.matches)} повторов")
            self._display_results()
            self._draw_heatmap()
        else:
            self.status_label.config(text="Повторений не найдено. Попробуйте снизить порог")
            self.results_list.delete(0, tk.END)
            self.results_list.insert(tk.END, "Повторений не найдено")

        self.progress.set(100)
        self.progress_label.config(text="100%")
        self.metric_values['time_left'].set("00:00")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _update_progress(self, percent, remaining, current_frames, total_frames):
        self.progress.set(percent)
        self.progress_label.config(text=f"{percent:.0f}%")
        frame_str = f"{self._format_number(current_frames)}/{self._format_number(total_frames)}"
        self.metric_values['frames'].set(frame_str)
        self.metric_values['time_left'].set(self._format_time(remaining))

    def _display_results(self):
        self.results_list.delete(0, tk.END)
        if not self.matches:
            return
        
        for m in self.matches:
            t1, t2 = self._format_time(m['t1']), self._format_time(m['t2'])
            d = m.get('direction', 'f')
            arrow = {"left": "←", "right": "→", "forward": "↑"}.get(d, "")
            
            if self.batch_mode:
                v1_name = os.path.basename(self.video_queue[m['v1_idx']])
                v2_name = os.path.basename(self.video_queue[m['v2_idx']])
                display = f"{arrow} {v1_name[:12]} ({t1}) ↔ {v2_name[:12]} ({t2}) | {m['sim']:.0%}"
            else:
                display = f"{arrow} {t1} → {t2}  {m['sim']:.0%}"
            self.results_list.insert(tk.END, display)
        
        self.metric_values['patterns'].set(len(self.matches))
        if self.matches:
            avg_sim = np.mean([m['sim'] for m in self.matches]) * 100
            self.metric_values['accuracy'].set(f"{avg_sim:.0f}%")
        
        if self.matches:
            self._show_preview(0)

    def _draw_heatmap(self):
        self.timeline_canvas.delete("all")
        if not self.matches or not self.video_queue:
            return
        
        self.timeline_canvas.update_idletasks()
        canvas_w = self.timeline_canvas.winfo_width()
        if canvas_w <= 0:
            canvas_w = 800
        
        total_duration = 0
        video_durations = []
        for path in self.video_queue:
            try:
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frames / fps if fps > 0 else 0
                total_duration += duration
                video_durations.append(duration)
                cap.release()
            except:
                video_durations.append(0)

        if total_duration == 0:
            return

        current_x = 0
        for i, dur in enumerate(video_durations):
            video_w = (dur / total_duration) * canvas_w
            if i > 0:
                self.timeline_canvas.create_line(current_x, 0, current_x, 40, fill=self.colors['border'])
            current_x += video_w

        start_times = np.cumsum([0] + video_durations[:-1])
        for m in self.matches:
            sim = m['sim']
            x1 = ((start_times[m['v1_idx']] + m['t1']) / total_duration) * canvas_w
            x2 = ((start_times[m['v2_idx']] + m['t2']) / total_duration) * canvas_w
            
            intensity = max(0, (sim - 0.7) / 0.3)
            red = int(90 + 165 * intensity)
            green = int(80 + 95 * intensity)
            blue = int(255 - 100 * intensity)
            color = f"#{red:02x}{green:02x}{blue:02x}"

            self.timeline_canvas.create_line(x1, 0, x1, 40, fill=color, width=2)
            self.timeline_canvas.create_line(x2, 0, x2, 40, fill=color, width=2)

    def _show_preview(self, index):
        if not self.matches or not (0 <= index < len(self.matches)):
            return
        
        self.current_match = index
        m = self.matches[index]
        
        self.time1_label.config(text=self._format_time(m['t1']))
        self.time2_label.config(text=self._format_time(m['t2']))
        
        d = m.get('direction', 'f')
        arrow = {"left": "←", "right": "→", "forward": "↑"}.get(d, "")
        self.match_info.config(text=f"{arrow} {d} • {m['sim']:.0%}")
        self.action1_label.config(text=d)
        self.action2_label.config(text=d)
        self.match_counter.config(text=f"{index+1}/{len(self.matches)}")

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

    def _load_frame_with_cache(self, video_idx, frame_num):
        video_path = self.video_queue[video_idx]
        video_hash = hashlib.md5(video_path.encode()).hexdigest()
        cache_path = os.path.join(self.preview_cache_dir, f"{video_hash}_{frame_num}.jpg")

        if os.path.exists(cache_path):
            frame = cv2.imread(cache_path)
        else:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    resized = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(cache_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame = resized

        if frame is not None:
            return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return None

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
        
        total_duration = 0
        video_durations = []
        for path in self.video_queue:
            try:
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frames / fps if fps > 0 else 0
                total_duration += duration
                video_durations.append(duration)
                cap.release()
            except:
                video_durations.append(0)
        
        if total_duration == 0:
            return
        
        clicked_time = (event.x / canvas_w) * total_duration
        
        start_times = np.cumsum([0] + video_durations[:-1])
        closest_match_idx = -1
        min_dist = float('inf')
        
        for i, m in enumerate(self.matches):
            m_time1 = start_times[m['v1_idx']] + m['t1']
            m_time2 = start_times[m['v2_idx']] + m['t2']
            dist = min(abs(clicked_time - m_time1), abs(clicked_time - m_time2))
            if dist < min_dist:
                min_dist = dist
                closest_match_idx = i
        
        if closest_match_idx != -1:
            self._show_preview(closest_match_idx)

    def prev_match(self):
        if self.current_match > 0:
            self._show_preview(self.current_match - 1)

    def next_match(self):
        if self.current_match < len(self.matches) - 1:
            self._show_preview(self.current_match + 1)

    def export_json(self):
        if not self.matches:
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            clean = []
            for m in self.matches:
                clean_m = m.copy()
                clean_m.pop('vec', None)
                clean.append(clean_m)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'matches': clean}, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Экспорт", "JSON файл сохранен")

    def export_txt(self):
        if not self.matches:
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text File", "*.txt")])
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("PARALLEL FINDER - РЕЗУЛЬТАТЫ\n" + "="*40 + "\n")
                for i, m in enumerate(self.matches):
                    v1_name = os.path.basename(self.video_queue[m['v1_idx']])
                    v2_name = os.path.basename(self.video_queue[m['v2_idx']])
                    f.write(f"{i+1:03d}. {m['sim']:.1%} | {v1_name} @ {self._format_time(m['t1'])} ↔ {v2_name} @ {self._format_time(m['t2'])}\n")
            messagebox.showinfo("Экспорт", "TXT файл сохранен")

    def export_edl(self):
        if not self.matches:
            messagebox.showwarning("Нет данных", "Нет данных для экспорта в EDL")
            return

        path = filedialog.asksaveasfilename(defaultextension=".edl", filetypes=[("Edit Decision List", "*.edl")])
        if not path:
            return

        try:
            cap = cv2.VideoCapture(self.video_queue[0])
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            cap.release()
        except:
            fps = 30

        def timecode(seconds):
            h = int(seconds / 3600)
            m = int((seconds % 3600) / 60)
            s = int(seconds % 60)
            f = int((seconds - int(seconds)) * fps)
            return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"

        with open(path, 'w', encoding='utf-8') as f:
            f.write("TITLE: Parallel Finder Export\nFCM: NON-DROP FRAME\n\n")
            timeline = 3600
            for i, m in enumerate(self.matches):
                clip = os.path.basename(self.video_queue[m['v1_idx']])
                f.write(f"{i*2+1:03d}  AX       V     C        {timecode(m['t1'])} {timecode(m['t1']+2)} {timecode(timeline)} {timecode(timeline+2)}\n")
                f.write(f"* FROM CLIP NAME: {clip}\n")
                
                clip2 = os.path.basename(self.video_queue[m['v2_idx']])
                f.write(f"{i*2+2:03d}  AX       V     C        {timecode(m['t2'])} {timecode(m['t2']+2)} {timecode(timeline+2)} {timecode(timeline+4)}\n")
                f.write(f"* FROM CLIP NAME: {clip2}\n\n")
                timeline += 5
        
        messagebox.showinfo("Экспорт", f"EDL файл сохранен")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    try:
        app = ParallelFinderApp()
        app.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        messagebox.showerror("Ошибка", str(e))