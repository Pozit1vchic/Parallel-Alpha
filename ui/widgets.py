from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QEasingCurve, Property, QPropertyAnimation, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPaintEvent, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .styles import THEMES


THEME = THEMES["dark"]

SETTINGS_DIALOG_I18N: dict[str, dict[str, str]] = {
    "en": {
        "title": "Application Settings",
        "tab.general": "General",
        "tab.models": "Models",
        "tab.cache": "Cache",
        "tab.advanced": "Advanced",
        "label.language": "Language",
        "label.theme": "Theme",
        "label.default_output": "Default Output",
        "label.model": "Model",
        "label.download": "Download",
        "label.tensorrt": "TensorRT",
        "label.cache": "Cache",
        "label.preview_cache": "Preview Cache (MB)",
        "label.behavior": "Behavior",
        "label.action": "Action",
        "label.startup": "Startup",
        "label.default_preset": "Default Preset",
        "label.updates": "Updates",
        "label.gpu": "GPU Device",
        "label.logging": "Logging",
        "label.advanced": "Mode",
        "btn.output": "Select Output Directory",
        "btn.download_model": "Download Model",
        "btn.compile_tensorrt": "Build TensorRT",
        "btn.clear_model_cache": "Clear Model Cache",
        "btn.clear_cache": "Clear Cache",
        "check.auto_clean": "Auto-clean on exit",
        "check.auto_load": "Load last project on startup",
        "check.check_updates": "Check for updates",
        "check.dev_mode": "Developer Mode",
        "dialog.select_output": "Select Default Output Directory",
    },
    "ru": {
        "title": "Настройки приложения",
        "tab.general": "Общие",
        "tab.models": "Модели",
        "tab.cache": "Кэш",
        "tab.advanced": "Дополнительно",
        "label.language": "Язык",
        "label.theme": "Тема",
        "label.default_output": "Папка вывода",
        "label.model": "Модель",
        "label.download": "Загрузка",
        "label.tensorrt": "TensorRT",
        "label.cache": "Кэш",
        "label.preview_cache": "Кэш превью (МБ)",
        "label.behavior": "Поведение",
        "label.action": "Действие",
        "label.startup": "Запуск",
        "label.default_preset": "Пресет по умолчанию",
        "label.updates": "Обновления",
        "label.gpu": "GPU устройство",
        "label.logging": "Логирование",
        "label.advanced": "Режим",
        "btn.output": "Выбрать папку вывода",
        "btn.download_model": "Скачать модель",
        "btn.compile_tensorrt": "Собрать TensorRT",
        "btn.clear_model_cache": "Очистить кэш моделей",
        "btn.clear_cache": "Очистить кэш",
        "check.auto_clean": "Очищать при выходе",
        "check.auto_load": "Загружать последний проект",
        "check.check_updates": "Проверять обновления",
        "check.dev_mode": "Режим разработчика",
        "dialog.select_output": "Выберите папку вывода по умолчанию",
    },
}


@dataclass(slots=True)
class AppSettingsData:
    """Application-level preferences stored by the settings dialog."""

    language: str = "Русский"
    theme: str = "Dark"
    output_dir: str = ""
    preview_cache_mb: int = 512
    auto_clean_cache: bool = False
    auto_load_project: bool = False
    default_preset: str = "Balanced"
    check_updates: bool = True
    gpu_device: str = "Auto"
    log_level: str = "INFO"
    developer_mode: bool = False
    selected_model_id: str = "pose_balanced"


class SettingsRow(QWidget):
    """Compact setting row with label, control and tracked value text."""

    def __init__(self, label: str, control: QWidget, suffix: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.control = control
        self.suffix = suffix

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(12)

        self.label_widget = QLabel(label)
        self.label_widget.setMinimumWidth(160)
        self.label_widget.setStyleSheet(f"color: {THEME['text_secondary']}; font-size: 11px;")

        self.value_widget = QLabel("-")
        self.value_widget.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.value_widget.setMinimumWidth(72)
        self.value_widget.setObjectName("settingValue")
        self.value_widget.setStyleSheet(f"color: {THEME['accent']}; font-weight: 600;")

        layout.addWidget(self.label_widget)
        layout.addWidget(control, 1)
        layout.addWidget(self.value_widget)

        self._bind_value_tracking()

    def _bind_value_tracking(self) -> None:
        """Bind widget signals to value label updates."""
        if isinstance(self.control, QSlider):
            self.control.valueChanged.connect(self._set_numeric_value)
            self._set_numeric_value(self.control.value())
        elif isinstance(self.control, QSpinBox):
            self.control.valueChanged.connect(self._set_numeric_value)
            self._set_numeric_value(self.control.value())
        elif isinstance(self.control, QDoubleSpinBox):
            self.control.valueChanged.connect(self._set_float_value)
            self._set_float_value(self.control.value())
        elif isinstance(self.control, QComboBox):
            self.control.currentTextChanged.connect(self._set_text_value)
            self._set_text_value(self.control.currentText())
        elif isinstance(self.control, QCheckBox):
            self.control.stateChanged.connect(lambda _: self._set_text_value("On" if self.control.isChecked() else "Off"))
            self._set_text_value("On" if self.control.isChecked() else "Off")
        elif isinstance(self.control, QLineEdit):
            self.control.textChanged.connect(self._set_text_value)
            self._set_text_value(self.control.text())

    def _set_numeric_value(self, value: int) -> None:
        self.value_widget.setText(f"{value}{self.suffix}")

    def _set_float_value(self, value: float) -> None:
        self.value_widget.setText(f"{value:.1f}{self.suffix}")

    def _set_text_value(self, text: str) -> None:
        self.value_widget.setText(text or "-")


class MetricCard(QFrame):
    """Metric card with icon, current value and title in V11.8 style."""

    def __init__(self, title: str, value: str = "0", icon: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setMinimumHeight(84)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)

        self.icon_label = QLabel(icon)
        self.icon_label.setStyleSheet(f"color: {THEME['text_secondary']}; font-size: 15px;")

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
        """Update the metric value text."""
        self.value_label.setText(value)


class VideoPreview(QFrame):
    """Preview tile that accepts numpy arrays, QImage, QPixmap or image paths."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("videoPreview")
        self._pixmap: QPixmap | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("previewTitle")

        self.image_label = QLabel("No frame")
        self.image_label.setObjectName("previewImage")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(250)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)

        self.time_label = QLabel("00:00.00")
        self.time_label.setObjectName("previewMetaLeft")

        self.action_label = QLabel("")
        self.action_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.action_label.setObjectName("previewMetaRight")

        footer.addWidget(self.time_label)
        footer.addStretch(1)
        footer.addWidget(self.action_label)

        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, 1)
        layout.addLayout(footer)

    def set_frame(self, frame: Any, timestamp: float, action: str = "") -> None:
        """Render a frame and update timestamp/action labels."""
        self.time_label.setText(self._format_timestamp(timestamp))
        self.action_label.setText(action)
        pixmap = self._to_pixmap(frame)
        self._pixmap = pixmap
        if pixmap is None:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Frame unavailable")
            return
        self.image_label.setText("")
        self._refresh_scaled_pixmap()

    def clear_frame(self) -> None:
        """Reset the tile to its empty state."""
        self._pixmap = None
        self.image_label.clear()
        self.image_label.setText("No frame")
        self.time_label.setText("00:00.00")
        self.action_label.setText("")

    def _to_pixmap(self, frame: Any) -> QPixmap | None:
        """Convert supported frame inputs into a pixmap."""
        if frame is None:
            return None
        if isinstance(frame, QPixmap):
            return frame
        if isinstance(frame, QImage):
            return QPixmap.fromImage(frame)
        if isinstance(frame, (str, Path)):
            path = Path(frame)
            if path.exists():
                pixmap = QPixmap(str(path))
                return pixmap if not pixmap.isNull() else None
            return None
        if isinstance(frame, np.ndarray):
            array = np.asarray(frame)
            if array.ndim == 2:
                gray = np.ascontiguousarray(array.astype(np.uint8))
                image = QImage(gray.data, gray.shape[1], gray.shape[0], gray.strides[0], QImage.Format.Format_Grayscale8).copy()
                return QPixmap.fromImage(image)
            if array.ndim == 3 and array.shape[2] in {3, 4}:
                clipped = np.clip(array, 0, 255).astype(np.uint8) if array.dtype != np.uint8 else np.ascontiguousarray(array)
                clipped = np.ascontiguousarray(clipped)
                if clipped.shape[2] == 3:
                    image = QImage(clipped.data, clipped.shape[1], clipped.shape[0], clipped.strides[0], QImage.Format.Format_RGB888).copy()
                else:
                    image = QImage(clipped.data, clipped.shape[1], clipped.shape[0], clipped.strides[0], QImage.Format.Format_RGBA8888).copy()
                return QPixmap.fromImage(image)
        return None

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_scaled_pixmap()

    def _refresh_scaled_pixmap(self) -> None:
        if self._pixmap is None:
            return
        target = self.image_label.size()
        if target.width() < 2 or target.height() < 2:
            return
        scaled = self._pixmap.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        total = max(0.0, float(seconds))
        minutes = int(total // 60)
        sec = total % 60
        return f"{minutes:02d}:{sec:05.2f}"


class _AnimatedArea(QScrollArea):
    """Internal animated area used by CollapsiblePanel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._content = QWidget()
        self._content.setLayout(QVBoxLayout())
        self._content.layout().setContentsMargins(0, 0, 0, 0)
        self._content.layout().setSpacing(8)
        self.setWidget(self._content)

    def content_layout(self) -> QLayout:
        return self._content.layout()


class CollapsiblePanel(QFrame):
    """Animated collapsible panel for advanced settings."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("collapsiblePanel")
        self._expanded = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.toggle_button = QPushButton(f"▸ {title}")
        self.toggle_button.setCheckable(True)
        self.toggle_button.clicked.connect(self.toggle)

        self.content_area = _AnimatedArea()
        self.content_area.setMaximumHeight(0)

        self.animation = QPropertyAnimation(self, b"contentHeight")
        self.animation.setDuration(220)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_area)

    def content_layout(self) -> QLayout:
        """Return the layout where extra widgets should be added."""
        return self.content_area.content_layout()

    def toggle(self) -> None:
        """Expand or collapse the panel with animation."""
        self._expanded = not self._expanded
        self.toggle_button.setChecked(self._expanded)
        self.toggle_button.setText(("▾ " if self._expanded else "▸ ") + self.toggle_button.text()[2:])
        target = self._target_height() if self._expanded else 0
        self.animation.stop()
        self.animation.setStartValue(self.content_area.maximumHeight())
        self.animation.setEndValue(target)
        self.animation.start()

    def _target_height(self) -> int:
        self.content_area.widget().adjustSize()
        return self.content_area.widget().sizeHint().height() + 8

    def getContentHeight(self) -> int:
        return self.content_area.maximumHeight()

    def setContentHeight(self, value: int) -> None:
        self.content_area.setMaximumHeight(max(0, int(value)))

    contentHeight = Property(int, getContentHeight, setContentHeight)


class TimelineHeatmap(QWidget):
    """Clickable match heatmap rendered in the V11.8 palette."""

    time_clicked = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(56)
        self._segments: list[tuple[float, float, float]] = []
        self._duration: float = 1.0

    def set_data(self, segments: list[tuple[float, float, float]], duration: float) -> None:
        """Update heatmap segments as (start, end, score)."""
        self._segments = list(segments)
        self._duration = max(1e-6, float(duration))
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(THEME["bg"]))
        painter.setPen(QColor(THEME["border"]))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 10, 10)

        track_rect = self.rect().adjusted(8, 10, -8, -10)
        painter.fillRect(track_rect, QColor(THEME["card"]))
        for start, end, score in self._segments:
            left = track_rect.left() + int((start / self._duration) * track_rect.width())
            right = track_rect.left() + int((end / self._duration) * track_rect.width())
            width = max(3, right - left)
            alpha = max(45, min(230, int(float(score) * 255)))
            color = QColor(THEME["accent"])
            color.setAlpha(alpha)
            painter.fillRect(left, track_rect.top(), width, track_rect.height(), color)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        ratio = event.position().x() / max(1.0, float(self.width()))
        self.time_clicked.emit(float(max(0.0, min(1.0, ratio)) * self._duration))
        super().mousePressEvent(event)


class SettingsDialog(QDialog):
    """Tabbed application settings dialog with basic i18n."""

    clear_cache_requested = Signal()
    clear_models_requested = Signal()

    def __init__(self, settings: AppSettingsData | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings or AppSettingsData()
        self._language_code = "ru" if self._settings.language.lower().startswith(("р", "ru")) else "en"
        self.resize(720, 560)
        self._build_ui()
        self._apply_data(self._settings)
        self.retranslate_ui(self._language_code)

    def _t(self, key: str) -> str:
        return SETTINGS_DIALOG_I18N.get(self._language_code, SETTINGS_DIALOG_I18N["en"]).get(key, key)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.general_tab = QWidget()
        self.general_form = QFormLayout(self.general_tab)
        self.language = QComboBox()
        self.language.addItem("Русский", "ru")
        self.language.addItem("English", "en")
        self.theme = QComboBox()
        self.theme.addItems(["Dark", "Light"])
        self.output_dir = QLabel("")
        self.output_btn = QPushButton()
        self.output_btn.clicked.connect(self._pick_output_dir)
        output_wrap = QWidget()
        output_layout = QHBoxLayout(output_wrap)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_dir, 1)
        output_layout.addWidget(self.output_btn)
        self.general_form.addRow("Language", self.language)
        self.general_form.addRow("Theme", self.theme)
        self.general_form.addRow("Output", output_wrap)
        self.tabs.addTab(self.general_tab, "General")

        self.models_tab = QWidget()
        self.models_form = QFormLayout(self.models_tab)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"])
        self.download_btn = QPushButton()
        self.trt_btn = QPushButton()
        self.clear_models_btn = QPushButton()
        self.clear_models_btn.clicked.connect(self.clear_models_requested.emit)
        self.models_form.addRow("Model", self.model_combo)
        self.models_form.addRow("Download", self.download_btn)
        self.models_form.addRow("TensorRT", self.trt_btn)
        self.models_form.addRow("Cache", self.clear_models_btn)
        self.tabs.addTab(self.models_tab, "Models")

        self.cache_tab = QWidget()
        self.cache_form = QFormLayout(self.cache_tab)
        self.preview_cache = QSlider(Qt.Orientation.Horizontal)
        self.preview_cache.setRange(128, 8192)
        self.auto_clean = QCheckBox()
        self.clear_cache_btn = QPushButton()
        self.clear_cache_btn.clicked.connect(self.clear_cache_requested.emit)
        self.cache_form.addRow("Preview Cache", self.preview_cache)
        self.cache_form.addRow("Behavior", self.auto_clean)
        self.cache_form.addRow("Action", self.clear_cache_btn)
        self.tabs.addTab(self.cache_tab, "Cache")

        self.advanced_tab = QWidget()
        self.advanced_form = QFormLayout(self.advanced_tab)
        self.auto_load = QCheckBox()
        self.default_preset = QComboBox()
        self.default_preset.addItems(["Fast", "Balanced", "Max"])
        self.check_updates = QCheckBox()
        self.gpu_device = QComboBox()
        self.gpu_device.addItems(["Auto", "cuda", "cpu"])
        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.dev_mode = QCheckBox()
        self.advanced_form.addRow("Startup", self.auto_load)
        self.advanced_form.addRow("Default preset", self.default_preset)
        self.advanced_form.addRow("Updates", self.check_updates)
        self.advanced_form.addRow("GPU Device", self.gpu_device)
        self.advanced_form.addRow("Logging", self.log_level)
        self.advanced_form.addRow("Developer Mode", self.dev_mode)
        self.tabs.addTab(self.advanced_tab, "Advanced")

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _pick_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, self._t("dialog.select_output"))
        if directory:
            self.output_dir.setText(directory)

    def retranslate_ui(self, language_code: str) -> None:
        """Re-apply texts according to the selected language."""
        self._language_code = "ru" if language_code == "ru" else "en"
        self.setWindowTitle(self._t("title"))

        self.tabs.setTabText(0, self._t("tab.general"))
        self.tabs.setTabText(1, self._t("tab.models"))
        self.tabs.setTabText(2, self._t("tab.cache"))
        self.tabs.setTabText(3, self._t("tab.advanced"))

        self.output_btn.setText(self._t("btn.output"))
        self.download_btn.setText(self._t("btn.download_model"))
        self.trt_btn.setText(self._t("btn.compile_tensorrt"))
        self.clear_models_btn.setText(self._t("btn.clear_model_cache"))
        self.clear_cache_btn.setText(self._t("btn.clear_cache"))
        self.auto_clean.setText(self._t("check.auto_clean"))
        self.auto_load.setText(self._t("check.auto_load"))
        self.check_updates.setText(self._t("check.check_updates"))
        self.dev_mode.setText(self._t("check.dev_mode"))

        self._set_form_label(self.general_form, self.language, self._t("label.language"))
        self._set_form_label(self.general_form, self.theme, self._t("label.theme"))
        self._set_form_label(self.general_form, self.output_btn.parentWidget(), self._t("label.default_output"))

        self._set_form_label(self.models_form, self.model_combo, self._t("label.model"))
        self._set_form_label(self.models_form, self.download_btn, self._t("label.download"))
        self._set_form_label(self.models_form, self.trt_btn, self._t("label.tensorrt"))
        self._set_form_label(self.models_form, self.clear_models_btn, self._t("label.cache"))

        self._set_form_label(self.cache_form, self.preview_cache, self._t("label.preview_cache"))
        self._set_form_label(self.cache_form, self.auto_clean, self._t("label.behavior"))
        self._set_form_label(self.cache_form, self.clear_cache_btn, self._t("label.action"))

        self._set_form_label(self.advanced_form, self.auto_load, self._t("label.startup"))
        self._set_form_label(self.advanced_form, self.default_preset, self._t("label.default_preset"))
        self._set_form_label(self.advanced_form, self.check_updates, self._t("label.updates"))
        self._set_form_label(self.advanced_form, self.gpu_device, self._t("label.gpu"))
        self._set_form_label(self.advanced_form, self.log_level, self._t("label.logging"))
        self._set_form_label(self.advanced_form, self.dev_mode, self._t("label.advanced"))

    @staticmethod
    def _set_form_label(form: QFormLayout, field: QWidget, text: str) -> None:
        label = form.labelForField(field)
        if label is not None:
            label.setText(text)

    def _apply_data(self, data: AppSettingsData) -> None:
        lang_code = "ru" if data.language.lower().startswith(("р", "ru")) else "en"
        lang_index = self.language.findData(lang_code)
        if lang_index >= 0:
            self.language.setCurrentIndex(lang_index)
        self.theme.setCurrentText(data.theme)
        self.output_dir.setText(data.output_dir)
        self.preview_cache.setValue(int(data.preview_cache_mb))
        self.auto_clean.setChecked(bool(data.auto_clean_cache))
        self.auto_load.setChecked(bool(data.auto_load_project))
        self.default_preset.setCurrentText(data.default_preset)
        self.check_updates.setChecked(bool(data.check_updates))
        self.gpu_device.setCurrentText(data.gpu_device)
        self.log_level.setCurrentText(data.log_level)
        self.dev_mode.setChecked(bool(data.developer_mode))

    def data(self) -> AppSettingsData:
        """Collect the current dialog state into a dataclass."""
        lang_code = str(self.language.currentData() or "ru")
        return AppSettingsData(
            language="Русский" if lang_code == "ru" else "English",
            theme=self.theme.currentText(),
            output_dir=self.output_dir.text(),
            preview_cache_mb=int(self.preview_cache.value()),
            auto_clean_cache=self.auto_clean.isChecked(),
            auto_load_project=self.auto_load.isChecked(),
            default_preset=self.default_preset.currentText(),
            check_updates=self.check_updates.isChecked(),
            gpu_device=self.gpu_device.currentText(),
            log_level=self.log_level.currentText(),
            developer_mode=self.dev_mode.isChecked(),
            selected_model_id=self._settings.selected_model_id,
        )


__all__ = [
    "SettingsRow",
    "MetricCard",
    "VideoPreview",
    "TimelineHeatmap",
    "CollapsiblePanel",
    "SettingsDialog",
    "AppSettingsData",
]
