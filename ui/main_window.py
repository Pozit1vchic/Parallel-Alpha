#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional
import hashlib

import cv2
import numpy as np

from PySide6.QtCore import Qt, Signal, QTimer, QThread
from PySide6.QtGui import QAction, QCloseEvent, QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

# Импорты из нового core
from core.analysis_backend import AnalysisBackend
from core.engine import YoloEngine
from core.matcher import MatchResult
from core.project import ProjectManager
from utils.constants import CACHE_DIR, MODELS_DIR, PREVIEW_SIZE
from utils.helpers import (
    format_time,
    compact_number,
    numpy_to_qpixmap,
    direction_to_emoji,
    to_timecode,
)


class MetricCard(QFrame):
    """Карточка метрики как в V12.2"""
    
    def __init__(self, title: str, value: str = "0", icon: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setMinimumHeight(84)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)
        
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        
        self.icon_label = QLabel(icon)
        self.icon_label.setStyleSheet("color: #8a8f99; font-size: 16px;")
        
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        
        header.addWidget(self.icon_label)
        header.addWidget(self.value_label)
        header.addStretch(1)
        
        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricTitle")
        
        layout.addLayout(header)
        layout.addWidget(self.title_label)
    
    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class VideoPreview(QFrame):
    """Превью как в V12.2"""
    
    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("videoPreview")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        
        self.title_label = QLabel(title)
        self.title_label.setObjectName("previewTitle")
        
        self.image_label = QLabel("Нет кадра")
        self.image_label.setObjectName("previewImage")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(250)
        
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        
        self.time_label = QLabel("00:00")
        self.time_label.setObjectName("previewTime")
        self.time_label.setAlignment(Qt.AlignCenter)
        
        self.action_label = QLabel("")
        self.action_label.setObjectName("previewAction")
        self.action_label.setAlignment(Qt.AlignCenter)
        
        footer.addWidget(self.time_label)
        footer.addStretch(1)
        footer.addWidget(self.action_label)
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, 1)
        layout.addLayout(footer)
        
        self._current_pixmap: Optional[QPixmap] = None
    
    def set_frame(self, frame: Any, timestamp: float, action: str = "") -> None:
        """Установить кадр и обновить метки"""
        self.time_label.setText(format_time(timestamp))
        self.action_label.setText(action)
        
        pixmap = self._to_pixmap(frame)
        self._current_pixmap = pixmap
        
        if pixmap is None or pixmap.isNull():
            self.image_label.setText("Кадр недоступен")
            self.image_label.setPixmap(QPixmap())
        else:
            self.image_label.setText("")
            self._refresh_scaled_pixmap()
    
    def clear_frame(self) -> None:
        """Очистить превью"""
        self._current_pixmap = None
        self.image_label.clear()
        self.image_label.setText("Нет кадра")
        self.time_label.setText("00:00")
        self.action_label.setText("")
    
    def _to_pixmap(self, frame: Any) -> Optional[QPixmap]:
        """Конвертировать кадр в QPixmap"""
        if frame is None:
            return None
        if isinstance(frame, QPixmap):
            return frame
        if isinstance(frame, np.ndarray):
            return numpy_to_qpixmap(frame)
        if isinstance(frame, (str, Path)):
            path = Path(frame)
            if path.exists():
                return QPixmap(str(path))
        return None
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_scaled_pixmap()
    
    def _refresh_scaled_pixmap(self):
        if self._current_pixmap is None:
            return
        target = self.image_label.size()
        if target.width() < 2 or target.height() < 2:
            return
        scaled = self._current_pixmap.scaled(
            target,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)


class TimelineHeatmap(QWidget):
    """Кликабельный таймлайн как в V12.2"""
    
    time_clicked = Signal(float)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(56)
        self._segments: list[tuple[float, float, float]] = []
        self._duration: float = 1.0
        self._colors = {
            'bg': '#0a0c10',
            'card': '#14171c',
            'accent': '#3b82f6',
            'border': '#2a2f38',
        }
    
    def set_data(self, segments: list[tuple[float, float, float]], duration: float) -> None:
        """Установить данные для отображения"""
        self._segments = list(segments)
        self._duration = max(1e-6, float(duration))
        self.update()
    
    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QColor
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Фон
        painter.fillRect(self.rect(), QColor(self._colors['bg']))
        painter.setPen(QColor(self._colors['border']))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)
        
        # Область для полосок
        track_rect = self.rect().adjusted(8, 10, -8, -10)
        painter.fillRect(track_rect, QColor(self._colors['card']))
        
        # Полоски схожести
        for start, end, score in self._segments:
            left = track_rect.left() + int((start / self._duration) * track_rect.width())
            right = track_rect.left() + int((end / self._duration) * track_rect.width())
            width = max(3, right - left)
            alpha = max(45, min(230, int(float(score) * 255)))
            color = QColor(self._colors['accent'])
            color.setAlpha(alpha)
            painter.fillRect(left, track_rect.top(), width, track_rect.height(), color)
    
    def mousePressEvent(self, event):
        ratio = event.position().x() / max(1.0, float(self.width()))
        self.time_clicked.emit(float(max(0.0, min(1.0, ratio)) * self._duration))
        super().mousePressEvent(event)


class ParallelFinderMainWindow(QMainWindow):
    """Главное окно в стиле V12.2 с новым core"""
    
    # Сигналы для связи с бэкендом
    analysisRequested = Signal(dict)
    stopRequested = Signal()
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        # ===== СОСТОЯНИЕ =====
        self.video_paths: list[str] = []
        self.batch_mode: bool = False
        self.analysis_running: bool = False
        self.matches: list[MatchResult] = []
        self.current_match: int = 0
        self.reference_path: str = ""
        self.preview_cache: dict[str, dict[int, np.ndarray]] = {}
        
        # ===== БЭКЕНД =====
        self.backend = AnalysisBackend()
        self.project_manager = ProjectManager()
        
        # ===== UI =====
        self.setWindowTitle("Parallel Finder V12.2")
        self.setMinimumSize(1200, 700)
        self.resize(1600, 900)
        
        # Цвета как в V12.2
        self.colors = {
            'bg': '#0a0c10',
            'card': '#14171c',
            'text': '#ffffff',
            'text_secondary': '#8a8f99',
            'accent': '#3b82f6',
            'accent2': '#f59e0b',
            'success': '#10b981',
            'error': '#ef4444',
            'border': '#1f2937',
            'highlight': '#1e293b'
        }
        
        # Метрики
        self.metric_values: dict[str, QLabel] = {}
        
        # Настройка UI
        self._setup_ui()
        self._setup_hotkeys()
        self._connect_signals()
        self._apply_stylesheet()
        
        self._flash_status("Parallel Finder готов")
    
    # ==========================================
    # 🎨 UI КАК В V12.2
    # ==========================================
    
    def _setup_ui(self) -> None:
        """Сборка интерфейса в 3 колонки как в V12.2"""
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(20)
        
        # Метрики сверху
        self._setup_metrics(root_layout)
        
        # Основной контент
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)
        
        # Левая панель
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(14)
        
        # Центр
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(14)
        
        # Правая панель
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(14)
        
        self._setup_video_panel(left_layout)
        self._setup_settings_panel(left_layout)
        self._setup_progress_panel(center_layout)
        self._setup_preview_panel(center_layout)
        self._setup_results_panel(right_layout)
        
        content_layout.addWidget(left, 2)
        content_layout.addWidget(center, 3)
        content_layout.addWidget(right, 2)
        root_layout.addWidget(content, 1)
        
        # Статус бар
        status = QStatusBar()
        self.setStatusBar(status)
        self.status_label = QLabel("Ожидание...")
        status.addPermanentWidget(self.status_label, 1)
    
    def _setup_metrics(self, parent_layout: QVBoxLayout) -> None:
        """Панель метрик сверху"""
        metrics_widget = QWidget()
        layout = QHBoxLayout(metrics_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        metrics = [
            ('frames', 'Кадров', '0', '🎬'),
            ('patterns', 'Повторов', '0', '🔄'),
            ('duration', 'Длительность', '00:00', '⏱️'),
            ('accuracy', 'Схожесть', '0%', '🎯'),
            ('time_left', 'Осталось', '--:--', '⏳'),
            ('batch', 'Batch', '0/0', '📂'),
        ]
        
        for key, title, value, icon in metrics:
            card = MetricCard(title=title, value=value, icon=icon)
            layout.addWidget(card, 1)
            self.metric_values[key] = card.value_label
        
        parent_layout.addWidget(metrics_widget)
    
    def _setup_video_panel(self, parent_layout: QVBoxLayout) -> None:
        """Панель выбора видео"""
        card = QFrame()
        card.setObjectName("panel")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        
        title = QLabel("📹 Видео")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        
        self.video_label = QLabel("Файл не выбран")
        self.video_label.setWordWrap(True)
        layout.addWidget(self.video_label)
        
        self.video_info = QLabel("")
        layout.addWidget(self.video_info)
        
        buttons = QHBoxLayout()
        self.add_video_btn = QPushButton("📁 Одно видео")
        self.add_folder_btn = QPushButton("📂 Папка")
        self.clear_queue_btn = QPushButton("🗑 Очистить")
        
        self.add_video_btn.clicked.connect(self.select_video)
        self.add_folder_btn.clicked.connect(self.select_folder)
        self.clear_queue_btn.clicked.connect(self.clear_queue)
        
        buttons.addWidget(self.add_video_btn)
        buttons.addWidget(self.add_folder_btn)
        buttons.addWidget(self.clear_queue_btn)
        layout.addLayout(buttons)
        
        self.batch_info = QLabel("")
        layout.addWidget(self.batch_info)
        
        self.video_list = QListWidget()
        self.video_list.setMinimumHeight(150)
        self.video_list.itemClicked.connect(self._on_video_selected)
        layout.addWidget(self.video_list, 1)
        
        parent_layout.addWidget(card)
    
    def _setup_settings_panel(self, parent_layout: QVBoxLayout) -> None:
        """Панель настроек (скроллируемая)"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        card = QFrame()
        card.setObjectName("panel")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        
        title = QLabel("⚙️ Настройки")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        
        # Порог схожести
        self._add_slider_row(layout, "Порог схожести (%)", "threshold", 50, 98, 75)
        
        # Длина повтора
        self._add_slider_row(layout, "Длина повтора (сек)", "repeat_length", 1, 30, 3)
        
        # Мин. разница
        self._add_slider_row(layout, "Мин. разница (сек)", "min_gap", 1, 30, 5)
        
        # Качество
        quality_row = QHBoxLayout()
        quality_label = QLabel("Качество")
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Быстро", "Средне", "Макс"])
        self.quality_combo.setCurrentText("Средне")
        quality_row.addWidget(quality_label)
        quality_row.addWidget(self.quality_combo, 1)
        layout.addLayout(quality_row)
        
        # Кнопки
        buttons_layout = QVBoxLayout()
        self.start_btn = QPushButton("🚀 СТАРТ")
        self.stop_btn = QPushButton("⏹ СТОП")
        self.reset_btn = QPushButton("↺ Сброс")
        
        self.start_btn.clicked.connect(self.start_analysis)
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.reset_btn.clicked.connect(self.reset_settings)
        self.stop_btn.setEnabled(False)
        
        buttons_layout.addWidget(self.start_btn)
        buttons_layout.addWidget(self.stop_btn)
        buttons_layout.addWidget(self.reset_btn)
        layout.addLayout(buttons_layout)
        
        scroll.setWidget(card)
        parent_layout.addWidget(scroll, 1)
    
    def _add_slider_row(self, parent_layout: QVBoxLayout, label: str, attr_name: str, 
                        min_val: int, max_val: int, default: int) -> None:
        """Добавить строку со слайдером"""
        row = QWidget()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 4, 0, 4)
        
        # Заголовок и значение
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        
        label_widget = QLabel(label)
        value_label = QLabel(str(default))
        value_label.setObjectName("sliderValue")
        
        header.addWidget(label_widget)
        header.addStretch(1)
        header.addWidget(value_label)
        
        # Слайдер
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        
        # Сохраняем ссылки
        setattr(self, f"{attr_name}_slider", slider)
        setattr(self, f"{attr_name}_label", value_label)
        
        # Обновление значения
        slider.valueChanged.connect(lambda v, l=value_label: l.setText(str(v)))
        
        row_layout.addLayout(header)
        row_layout.addWidget(slider)
        parent_layout.addWidget(row)
    
    def _setup_progress_panel(self, parent_layout: QVBoxLayout) -> None:
        """Панель прогресса"""
        card = QFrame()
        card.setObjectName("panel")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)
        
        header = QHBoxLayout()
        title = QLabel("📊 Прогресс")
        title.setObjectName("panelTitle")
        
        self.progress_label = QLabel("0%")
        self.progress_label.setObjectName("progressValue")
        
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        
        self.progress_status = QLabel("Ожидание...")
        
        layout.addLayout(header)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_status)
        
        parent_layout.addWidget(card)
    
    def _setup_preview_panel(self, parent_layout: QVBoxLayout) -> None:
        """Панель превью"""
        card = QFrame()
        card.setObjectName("panel")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        
        header = QHBoxLayout()
        title = QLabel("🎬 Сравнение")
        title.setObjectName("panelTitle")
        
        self.match_info = QLabel("")
        
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.match_info)
        
        previews = QHBoxLayout()
        previews.setSpacing(10)
        
        self.preview1 = VideoPreview("Сегмент A")
        self.preview2 = VideoPreview("Сегмент B")
        
        previews.addWidget(self.preview1, 1)
        previews.addWidget(self.preview2, 1)
        
        # Ползунок схожести
        sim_layout = QHBoxLayout()
        sim_label = QLabel("Схожесть")
        self.similarity_bar = QProgressBar()
        self.similarity_bar.setRange(0, 100)
        self.similarity_bar.setValue(0)
        self.similarity_bar.setTextVisible(False)
        sim_layout.addWidget(sim_label)
        sim_layout.addWidget(self.similarity_bar, 1)
        
        # Навигация
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("←")
        self.next_btn = QPushButton("→")
        self.match_counter = QLabel("0/0")
        self.match_counter.setAlignment(Qt.AlignCenter)
        
        self.prev_btn.clicked.connect(self.prev_match)
        self.next_btn.clicked.connect(self.next_match)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.match_counter, 1)
        nav_layout.addWidget(self.next_btn)
        
        # Таймлайн
        self.timeline = TimelineHeatmap()
        self.timeline.time_clicked.connect(self._on_timeline_clicked)
        
        layout.addLayout(header)
        layout.addLayout(previews, 1)
        layout.addLayout(sim_layout)
        layout.addLayout(nav_layout)
        layout.addWidget(self.timeline)
        
        parent_layout.addWidget(card, 1)
    
    def _setup_results_panel(self, parent_layout: QVBoxLayout) -> None:
        """Панель результатов"""
        card = QFrame()
        card.setObjectName("panel")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        
        header = QHBoxLayout()
        title = QLabel("📋 Результаты")
        title.setObjectName("panelTitle")
        
        export_frame = QHBoxLayout()
        self.json_btn = QPushButton("JSON")
        self.txt_btn = QPushButton("TXT")
        self.edl_btn = QPushButton("EDL")
        
        self.json_btn.clicked.connect(self.export_json)
        self.txt_btn.clicked.connect(self.export_txt)
        self.edl_btn.clicked.connect(self.export_edl)
        
        export_frame.addWidget(self.json_btn)
        export_frame.addWidget(self.txt_btn)
        export_frame.addWidget(self.edl_btn)
        
        header.addWidget(title)
        header.addStretch(1)
        header.addLayout(export_frame)
        
        self.results_list = QListWidget()
        self.results_list.itemClicked.connect(self.on_select)
        
        layout.addLayout(header)
        layout.addWidget(self.results_list, 1)
        
        parent_layout.addWidget(card, 1)
    
    def _apply_stylesheet(self) -> None:
        """Применить стили как в V12.2"""
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {self.colors['bg']};
                color: {self.colors['text']};
                font-family: Inter, Segoe UI, sans-serif;
                font-size: 13px;
            }}
            QFrame#panel, QFrame#metricCard, QFrame#videoPreview {{
                background-color: {self.colors['card']};
                border: 1px solid {self.colors['border']};
                border-radius: 8px;
            }}
            QLabel#metricTitle {{
                color: {self.colors['text_secondary']};
                font-size: 9px;
            }}
            QLabel#metricValue {{
                color: {self.colors['text']};
                font-size: 18px;
                font-weight: bold;
            }}
            QLabel#panelTitle {{
                color: {self.colors['text']};
                font-size: 14px;
                font-weight: bold;
            }}
            QLabel#previewTitle {{
                color: {self.colors['text']};
                font-size: 11px;
                font-weight: bold;
            }}
            QLabel#previewTime, QLabel#previewAction {{
                background-color: {self.colors['bg']};
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }}
            QLabel#sliderValue {{
                color: {self.colors['accent']};
                font-weight: bold;
            }}
            QLabel#progressValue {{
                color: {self.colors['accent']};
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton {{
                background-color: {self.colors['card']};
                color: {self.colors['text']};
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {self.colors['highlight']};
            }}
            QPushButton:disabled {{
                color: {self.colors['text_secondary']};
                border-color: {self.colors['border']};
            }}
            QProgressBar {{
                background-color: {self.colors['bg']};
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                min-height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {self.colors['accent']};
                border-radius: 4px;
            }}
            QListWidget, QComboBox, QScrollArea {{
                background-color: {self.colors['bg']};
                color: {self.colors['text']};
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
            }}
            QListWidget::item:selected {{
                background-color: {self.colors['accent']};
            }}
            QSlider::groove:horizontal {{
                background: {self.colors['bg']};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {self.colors['accent']};
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
        """)
        
        # Специальные стили для кнопок
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors['success']};
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 12px;
            }}
        """)
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors['error']};
                color: white;
                font-weight: bold;
                padding: 8px;
            }}
        """)
    
    def _setup_hotkeys(self) -> None:
        """Горячие клавиши как в V12.2"""
        shortcuts = [
            ("Space", self.start_analysis),
            ("Up", self.prev_match),
            ("Down", self.next_match),
            ("Ctrl+S", lambda: self.project_manager.save_project()),
        ]
        for key, callback in shortcuts:
            action = QAction(self)
            action.setShortcut(key)
            action.triggered.connect(callback)
            self.addAction(action)
    
    def _connect_signals(self) -> None:
        """Подключение сигналов бэкенда"""
        self.analysisRequested.connect(self._on_analysis_requested)
        self.stopRequested.connect(self._on_stop_requested)
        
        self.backend.analysisStarted.connect(
            lambda: self.set_analysis_running(True, "Анализ запущен")
        )
        self.backend.analysisFinished.connect(
            lambda: self.set_analysis_running(False, "Анализ завершен")
        )
        self.backend.analysisStopped.connect(
            lambda: self.set_analysis_running(False, "Анализ остановлен")
        )
        self.backend.progressChanged.connect(self._on_progress)
        self.backend.resultsReady.connect(self._on_results_ready)
        self.backend.previewFramesReady.connect(self._on_preview_frames_ready)
        self.backend.errorOccurred.connect(self._on_error)
    
    # ==========================================
    # 🧠 ЛОГИКА ВЫБОРА ВИДЕО
    # ==========================================
    
    def select_video(self) -> None:
        """Выбрать одно видео"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите видео", "",
            "Видео (*.mp4 *.avi *.mkv *.mov *.ts *.m4v *.3gp *.wmv *.flv *.webm *.mpeg *.mpg *.m2ts *.vob)"
        )
        if not path:
            return
        
        self.video_paths = [path]
        self.batch_mode = False
        self.video_label.setText(Path(path).name)
        self.batch_info.setText("")
        self._update_video_info(path)
        self._refresh_video_list()
    
    def select_folder(self) -> None:
        """Выбрать папку с видео"""
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с видео")
        if not folder:
            return
        
        self.video_paths = []
        for ext in ['.mp4', '.avi', '.mkv', '.mov', '.ts']:
            self.video_paths.extend([str(p) for p in Path(folder).glob(f"*{ext}")])
            self.video_paths.extend([str(p) for p in Path(folder).glob(f"*{ext.upper()}")])
        
        if self.video_paths:
            self.batch_mode = True
            self.video_label.setText(f"📁 ПАПКА: {Path(folder).name}")
            self.batch_info.setText(f"🎬 Найдено видео: {len(self.video_paths)}")
            self._update_video_info(self.video_paths[0])
            self._refresh_video_list()
    
    def clear_queue(self) -> None:
        """Очистить очередь видео"""
        self.video_paths = []
        self.batch_mode = False
        self.matches = []
        self.preview_cache.clear()
        
        self.video_label.setText("Файл не выбран")
        self.video_info.clear()
        self.batch_info.clear()
        self.results_list.clear()
        self.preview1.clear_frame()
        self.preview2.clear_frame()
        self.timeline.set_data([], 1.0)
        
        self._set_metric("frames", "0")
        self._set_metric("patterns", "0")
        self._set_metric("duration", "00:00")
        self._set_metric("accuracy", "0%")
        self._set_metric("batch", "0/0")
        
        self._refresh_video_list()
    
    def _update_video_info(self, path: str) -> None:
        """Обновить информацию о видео"""
        cap = cv2.VideoCapture(path)
        try:
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            duration = frames / max(fps, 1e-6)
            
            self._set_metric("frames", compact_number(frames))
            self._set_metric("duration", format_time(duration))
            self.video_info.setText(
                f"{compact_number(frames)} кадров • {fps:.0f} fps"
            )
        finally:
            cap.release()
    
    def _refresh_video_list(self) -> None:
        """Обновить список видео"""
        self.video_list.clear()
        for path in self.video_paths:
            self.video_list.addItem(QListWidgetItem(Path(path).name))
        
        total = len(self.video_paths)
        self._set_metric("batch", f"{total}/{total}" if total else "0/0")
    
    def _on_video_selected(self, item: QListWidgetItem) -> None:
        """Обработка выбора видео из списка"""
        index = self.video_list.row(item)
        if 0 <= index < len(self.video_paths):
            self._update_video_info(self.video_paths[index])
    
    # ==========================================
    # 🎛️ НАСТРОЙКИ
    # ==========================================
    
    def collect_settings(self) -> dict[str, Any]:
        """Собрать настройки из UI"""
        quality_map = {"Быстро": "fast", "Средне": "balanced", "Макс": "max"}
        
        return {
            "similarity_threshold": float(self.threshold_slider.value()),
            "repeat_length": float(self.repeat_length_slider.value()),
            "min_gap": float(self.min_gap_slider.value()),
            "quality": quality_map.get(self.quality_combo.currentText(), "balanced"),
            "model_variant": "m",  # можно добавить выбор модели
        }
    
    def reset_settings(self) -> None:
        """Сбросить настройки"""
        self.threshold_slider.setValue(75)
        self.repeat_length_slider.setValue(3)
        self.min_gap_slider.setValue(5)
        self.quality_combo.setCurrentText("Средне")
    
    # ==========================================
    # 🚀 АНАЛИЗ
    # ==========================================
    
    def start_analysis(self) -> None:
        """Запустить анализ"""
        if not self.video_paths:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите видео!")
            return
        
        self.matches = []
        self.results_list.clear()
        self.preview1.clear_frame()
        self.preview2.clear_frame()
        self.timeline.set_data([], 1.0)
        self.current_match = 0
        
        self._set_metric("patterns", "0")
        self._set_metric("accuracy", "0%")
        self._set_metric("time_left", "--:--")
        self.similarity_bar.setValue(0)
        
        # Формируем payload для бэкенда
        payload = {
            "videos": self.video_paths,
            "reference": self.reference_path,
            "analysis_settings": self.collect_settings(),
            "model_path": str(MODELS_DIR / "yolov8m-pose.pt"),
        }
        
        self.analysisRequested.emit(payload)
        self.set_analysis_running(True, "Запуск анализа...")
    
    def stop_analysis(self) -> None:
        """Остановить анализ"""
        self.stopRequested.emit()
    
    def set_analysis_running(self, running: bool, status_text: str) -> None:
        """Обновить состояние анализа"""
        self.analysis_running = running
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.status_label.setText(status_text)
        self.progress_status.setText(status_text)
    
    def _on_analysis_requested(self, payload: dict) -> None:
        """Обработка сигнала запроса анализа"""
        try:
            self.backend.start_analysis(payload)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            self.set_analysis_running(False, "Ошибка запуска")
    
    def _on_stop_requested(self) -> None:
        """Обработка сигнала остановки"""
        self.backend.stop_analysis()
    
    def _on_progress(self, percent: float, eta: float, status: str) -> None:
        """Обновление прогресса"""
        self.progress_bar.setValue(int(percent))
        self.progress_label.setText(f"{int(percent)}%")
        self.progress_status.setText(status)
        self.status_label.setText(status)
        
        if eta > 0:
            self._set_metric("time_left", format_time(eta))
        else:
            self._set_metric("time_left", "--:--")
    
    def _on_results_ready(self, matches: list[MatchResult]) -> None:
        """Получение результатов"""
        self.matches = list(matches)
        self.display_results()
        self._flash_status(f"Найдено совпадений: {len(self.matches)}")
    
    def _on_preview_frames_ready(self, payload: dict) -> None:
        """Получение кадров для превью"""
        video_path = payload.get("video_path", "")
        frames = payload.get("frames", {})
        
        if not video_path or not frames:
            return
        
        # Сохраняем в локальный кэш
        if video_path not in self.preview_cache:
            self.preview_cache[video_path] = {}
        
        for idx, frame in frames.items():
            try:
                idx_int = int(idx)
                self.preview_cache[video_path][idx_int] = np.asarray(frame)
            except Exception:
                continue
    
    def _on_error(self, error: str) -> None:
        """Обработка ошибки"""
        QMessageBox.critical(self, "Ошибка", error)
        self.set_analysis_running(False, "Ошибка")
    
    # ==========================================
    # 📊 ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
    # ==========================================
    
    def display_results(self) -> None:
        """Отобразить результаты в списке"""
        self.results_list.clear()
        
        if not self.matches:
            self.results_list.addItem("❌ Совпадения не найдены")
            self.match_counter.setText("0/0")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return
        
        for idx, match in enumerate(self.matches):
            sim = match.get("similarity", 0) * 100
            t1 = format_time(match.get("t1", 0))
            t2 = format_time(match.get("t2", 0))
            direction = match.get("direction", -1)
            context = match.get("context", 0)
            
            emoji = direction_to_emoji(direction)
            context_map = {0: "стоя", 1: "ходьба", 2: "бег", 3: "присед",
                          4: "удар", 5: "удар ногой", 6: "прыжок", 7: "падение"}
            ctx_str = context_map.get(context, "")
            
            display = f"{idx+1:03d}. {emoji} {t1} → {t2}  |  {sim:.1f}%  |  {ctx_str}"
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, idx)
            self.results_list.addItem(item)
        
        # Обновляем метрики
        avg_sim = sum(m.get("similarity", 0) for m in self.matches) / len(self.matches) * 100
        self._set_metric("patterns", str(len(self.matches)))
        self._set_metric("accuracy", f"{avg_sim:.0f}%")
        
        # Обновляем таймлайн
        self._update_timeline()
        
        # Показываем первый результат
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.show_preview(0)
    
    def show_preview(self, index: int) -> None:
        """Показать превью для выбранного совпадения"""
        if not (0 <= index < len(self.matches)):
            return
        
        self.current_match = index
        match = self.matches[index]
        
        v1_idx = match.get("v1_idx", 0)
        v2_idx = match.get("v2_idx", 0)
        frame_i = match.get("frame_i", 0)
        frame_j = match.get("frame_j", 0)
        t1 = match.get("t1", frame_i / 30.0)
        t2 = match.get("t2", frame_j / 30.0)
        direction = match.get("direction", -1)
        context = match.get("context", 0)
        similarity = match.get("similarity", 0) * 100
        
        context_map = {0: "стоя", 1: "ходьба", 2: "бег", 3: "присед",
                      4: "удар", 5: "удар ногой", 6: "прыжок", 7: "падение"}
        ctx_str = context_map.get(context, "")
        
        # Получаем кадры из кэша
        path_a = self._get_video_path(v1_idx)
        path_b = self._get_video_path(v2_idx)
        
        frame1 = self._get_cached_frame(path_a, frame_i)
        frame2 = self._get_cached_frame(path_b, frame_j)
        
        self.preview1.set_frame(frame1, t1, ctx_str)
        self.preview2.set_frame(frame2, t2, ctx_str)
        
        self.similarity_bar.setValue(int(similarity))
        self.match_info.setText(f"{ctx_str} • {similarity:.0f}%")
        self.match_counter.setText(f"{index+1}/{len(self.matches)}")
        
        # Подсветка в списке
        self.results_list.setCurrentRow(index)
    
    def on_select(self, item: QListWidgetItem) -> None:
        """Обработка выбора в списке"""
        index = item.data(Qt.UserRole)
        if index is not None:
            self.show_preview(index)
    
    def prev_match(self) -> None:
        """Предыдущее совпадение"""
        if self.current_match > 0:
            self.show_preview(self.current_match - 1)
    
    def next_match(self) -> None:
        """Следующее совпадение"""
        if self.current_match < len(self.matches) - 1:
            self.show_preview(self.current_match + 1)
    
    def _on_timeline_clicked(self, clicked_time: float) -> None:
        """Обработка клика по таймлайну"""
        if not self.matches:
            return
        
        # Ищем ближайшее совпадение
        best_idx = min(
            range(len(self.matches)),
            key=lambda i: abs(self.matches[i].get("t1", 0) - clicked_time)
        )
        self.show_preview(best_idx)
    
    def _update_timeline(self) -> None:
        """Обновить таймлайн"""
        if not self.matches:
            self.timeline.set_data([], 1.0)
            return
        
        segments = []
        max_duration = 1.0
        
        for match in self.matches:
            start = match.get("t1", 0)
            duration = match.get("duration", 0)
            score = match.get("similarity", 0)
            end = start + duration
            segments.append((start, end, score))
            max_duration = max(max_duration, end)
        
        self.timeline.set_data(segments, max_duration)
    
    def _get_video_path(self, index: int) -> str:
        """Получить путь к видео по индексу"""
        if 0 <= index < len(self.video_paths):
            return self.video_paths[index]
        return self.video_paths[0] if self.video_paths else ""
    
    def _get_cached_frame(self, video_path: str, frame_idx: int) -> Optional[np.ndarray]:
        """Получить кадр из кэша"""
        if not video_path:
            return None
        
        # Проверяем в памяти
        video_cache = self.preview_cache.get(video_path, {})
        if frame_idx in video_cache:
            return video_cache[frame_idx]
        
        # Проверяем в ProjectManager
        preview_cache = self.project_manager.get_preview_cache()
        cached = preview_cache.get(video_path, frame_idx)
        
        if cached is not None:
            # Сохраняем в память
            if video_path not in self.preview_cache:
                self.preview_cache[video_path] = {}
            self.preview_cache[video_path][frame_idx] = np.asarray(cached)
            return cached
        
        # Если нет в кэше, читаем из видео
        frame = self._read_video_frame(video_path, frame_idx)
        if frame is not None:
            preview_cache.put(video_path, frame_idx, frame)
            if video_path not in self.preview_cache:
                self.preview_cache[video_path] = {}
            self.preview_cache[video_path][frame_idx] = frame
        
        return frame
    
    def _read_video_frame(self, video_path: str, frame_idx: int) -> Optional[np.ndarray]:
        """Прочитать кадр из видео"""
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            return None
        finally:
            cap.release()
    
    # ==========================================
    # 💾 ЭКСПОРТ
    # ==========================================
    
    def export_json(self) -> None:
        """Экспорт в JSON"""
        if not self.matches:
            QMessageBox.warning(self, "Экспорт", "Нет данных для экспорта")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт JSON", "matches.json", "JSON (*.json)")
        if not path:
            return
        
        try:
            self.project_manager.export_matches(self.matches, "json", path)
            QMessageBox.information(self, "Экспорт", "JSON сохранён")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def export_txt(self) -> None:
        """Экспорт в TXT"""
        if not self.matches:
            QMessageBox.warning(self, "Экспорт", "Нет данных для экспорта")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт TXT", "matches.txt", "Text (*.txt)")
        if not path:
            return
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("PARALLEL FINDER V12.2 - РЕЗУЛЬТАТЫ\n")
                f.write("=" * 60 + "\n\n")
                
                for idx, match in enumerate(self.matches, 1):
                    t1 = format_time(match.get("t1", 0))
                    t2 = format_time(match.get("t2", 0))
                    sim = match.get("similarity", 0) * 100
                    direction = match.get("direction", -1)
                    context = match.get("context", 0)
                    
                    context_map = {0: "стоя", 1: "ходьба", 2: "бег", 3: "присед",
                                  4: "удар", 5: "удар ногой", 6: "прыжок", 7: "падение"}
                    
                    f.write(f"{idx:03d}. {t1} → {t2}  |  {sim:.1f}%  |  {context_map.get(context, '')}\n")
            
            QMessageBox.information(self, "Экспорт", "TXT сохранён")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    def export_edl(self) -> None:
        """Экспорт в EDL"""
        if not self.matches:
            QMessageBox.warning(self, "Экспорт", "Нет данных для экспорта")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт EDL", "matches.edl", "EDL (*.edl)")
        if not path:
            return
        
        try:
            lines = ["TITLE: PARALLEL FINDER MATCHES", "FCM: NON-DROP FRAME", ""]
            
            for idx, match in enumerate(self.matches, 1):
                t1 = match.get("t1", 0)
                t2 = match.get("t2", 0)
                duration = match.get("duration", 1.0)
                
                start_a = to_timecode(t1)
                end_a = to_timecode(t1 + duration)
                start_b = to_timecode(t2)
                end_b = to_timecode(t2 + duration)
                
                lines.append(f"{idx:03d}  AX V C  {start_a} {end_a} {start_b} {end_b}")
                lines.append(f"* FROM CLIP NAME: match_{idx:03d}")
                lines.append("")
            
            Path(path).write_text("\n".join(lines), encoding="utf-8")
            QMessageBox.information(self, "Экспорт", "EDL сохранён")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
    
    # ==========================================
    # 🛠️ ВСПОМОГАТЕЛЬНЫЕ
    # ==========================================
    
    def _set_metric(self, key: str, value: str) -> None:
        """Установить значение метрики"""
        if key in self.metric_values:
            self.metric_values[key].setText(value)
    
    def _flash_status(self, message: str, timeout_ms: int = 3000) -> None:
        """Показать сообщение в статус-баре"""
        if self.statusBar():
            self.statusBar().showMessage(message, timeout_ms)
    
    def closeEvent(self, event: QCloseEvent) -> None:
        """Обработка закрытия окна"""
        self.backend.stop_analysis()
        super().closeEvent(event)


# Для обратной совместимости
MainWindow = ParallelFinderMainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ParallelFinderMainWindow()
    window.show()
    sys.exit(app.exec())