from __future__ import annotations

from typing import Any


THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#0a0c10",
        "card": "#14171c",
        "text": "#ffffff",
        "text_secondary": "#8a8f99",
        "accent": "#3b82f6",
        "accent2": "#f59e0b",
        "success": "#22c55e",
        "warning": "#facc15",
        "error": "#ef4444",
        "border": "#2a2f38",
        "hover": "#1e232b",
    }
}


class AppStyle:
    """Centralized QSS provider for the Parallel Finder UI."""

    @staticmethod
    def _colors(theme: str = "dark") -> dict[str, str]:
        return THEMES.get(theme, THEMES["dark"])

    @staticmethod
    def main_window(theme: str = "dark") -> str:
        colors = AppStyle._colors(theme)
        return f"""
            QMainWindow, QWidget {{
                background: {colors['bg']};
                color: {colors['text']};
                font-family: Inter, Segoe UI, Arial;
                font-size: 13px;
            }}
            {AppStyle.panel(theme)}
            {AppStyle.metric_card(theme)}
            {AppStyle.button(theme)}
            {AppStyle.progress_bar(theme)}
            {AppStyle.list_widget(theme)}
            {AppStyle.slider(theme)}
            QLabel#panelTitle {{
                color: {colors['text']};
                font-size: 14px;
                font-weight: 700;
            }}
            QLabel#previewTitle {{
                color: {colors['text']};
                font-weight: 700;
            }}
            QLabel#previewImage {{
                background: {colors['bg']};
                border: 1px solid {colors['border']};
                border-radius: 12px;
            }}
            QLabel#previewMetaLeft, QLabel#previewMetaRight {{
                background: {colors['bg']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                padding: 6px 10px;
                border-radius: 8px;
            }}
            QLabel#previewMetaRight {{
                color: {colors['accent']};
                font-weight: 600;
            }}
            QLabel#subtle {{
                color: {colors['text_secondary']};
            }}
            QCheckBox {{
                color: {colors['text']};
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
            }}
            QCheckBox::indicator:unchecked {{
                border: 1px solid {colors['border']};
                background: {colors['bg']};
                border-radius: 4px;
            }}
            QCheckBox::indicator:checked {{
                border: 1px solid {colors['accent']};
                background: {colors['accent']};
                border-radius: 4px;
            }}
            QStatusBar {{
                background: {colors['card']};
                border-top: 1px solid {colors['border']};
            }}
            QToolTip {{
                background: {colors['card']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                padding: 4px 8px;
            }}
        """

    @staticmethod
    def panel(theme: str = "dark") -> str:
        colors = AppStyle._colors(theme)
        return f"""
            QFrame#panel, QFrame#videoPreview, QFrame#collapsiblePanel {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 {colors['card']},
                    stop:1 #12161c
                );
                border: 1px solid {colors['border']};
                border-radius: 14px;
                box-shadow: 0 10px 24px rgba(0, 0, 0, 0.28);
            }}
            QScrollArea {{
                border: none;
                background: transparent;
            }}
        """

    @staticmethod
    def metric_card(theme: str = "dark") -> str:
        colors = AppStyle._colors(theme)
        return f"""
            QFrame#metricCard {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #11151b,
                    stop:1 #0f1319
                );
                border: 1px solid {colors['border']};
                border-radius: 14px;
                box-shadow: 0 8px 22px rgba(0, 0, 0, 0.24);
            }}
            QLabel#metricTitle {{
                color: {colors['text_secondary']};
                font-size: 10px;
            }}
            QLabel#metricValue {{
                color: {colors['text']};
                font-size: 19px;
                font-weight: 700;
            }}
        """

    @staticmethod
    def slider(theme: str = "dark") -> str:
        colors = AppStyle._colors(theme)
        return f"""
            QSlider::groove:horizontal {{
                background: {colors['bg']};
                border: 1px solid {colors['border']};
                border-radius: 5px;
                height: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {colors['accent']};
                border-radius: 5px;
            }}
            QSlider::handle:horizontal {{
                background: {colors['accent']};
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
                border: 1px solid #6ea5ff;
            }}
            QSlider::handle:horizontal:hover {{
                background: #5a98fa;
            }}
        """

    @staticmethod
    def button(theme: str = "dark", variant: str = "default") -> str:
        colors = AppStyle._colors(theme)
        variants: dict[str, dict[str, Any]] = {
            "default": {
                "bg": colors["card"],
                "hover": colors["hover"],
                "text": colors["text"],
                "border": colors["border"],
                "padding": "9px 12px",
            },
            "success": {
                "bg": colors["success"],
                "hover": "#1fb463",
                "text": "white",
                "border": "transparent",
                "padding": "12px 14px",
            },
            "danger": {
                "bg": colors["error"],
                "hover": "#d73a3a",
                "text": "white",
                "border": "transparent",
                "padding": "11px 13px",
            },
            "accent": {
                "bg": colors["accent"],
                "hover": "#2563eb",
                "text": "white",
                "border": "transparent",
                "padding": "9px 12px",
            },
            "warning": {
                "bg": colors["accent2"],
                "hover": "#e18c08",
                "text": "white",
                "border": "transparent",
                "padding": "9px 12px",
            },
        }
        current = variants.get(variant, variants["default"])
        if variant == "default":
            return f"""
                QPushButton {{
                    background: {current['bg']};
                    color: {current['text']};
                    border: 1px solid {current['border']};
                    border-radius: 10px;
                    padding: {current['padding']};
                    transition: all 150ms ease;
                }}
                QPushButton:hover {{
                    background: {current['hover']};
                }}
                QPushButton:pressed {{
                    background: #151b24;
                }}
                QPushButton:disabled {{
                    color: {colors['text_secondary']};
                    background: #12161d;
                }}
            """

        return (
            "QPushButton {"
            f"background:{current['bg']};"
            f"color:{current['text']};"
            f"border:1px solid {current['border']};"
            "border-radius:12px;"
            f"padding:{current['padding']};"
            "font-weight:700;"
            "}"
            "QPushButton:hover {"
            f"background:{current['hover']};"
            "}"
        )

    @staticmethod
    def progress_bar(theme: str = "dark") -> str:
        colors = AppStyle._colors(theme)
        return f"""
            QProgressBar {{
                background: {colors['bg']};
                border: 1px solid {colors['border']};
                border-radius: 7px;
                min-height: 12px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2563eb,
                    stop:1 {colors['accent']}
                );
                border-radius: 7px;
            }}
        """

    @staticmethod
    def list_widget(theme: str = "dark") -> str:
        colors = AppStyle._colors(theme)
        return f"""
            QListWidget, QComboBox, QLineEdit {{
                background: #0c1016;
                color: {colors['text']};
                border: 1px solid {colors['border']};
                border-radius: 10px;
                padding: 5px;
            }}
            QListWidget::item {{
                padding: 6px;
                border-radius: 8px;
            }}
            QListWidget::item:selected {{
                background: {colors['accent']};
                color: white;
            }}
            QListWidget::item:hover {{
                background: {colors['hover']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 22px;
            }}
        """
