# ui/panels/stats_bar.py
from __future__ import annotations

import tkinter as tk


_LABELS: dict[str, dict[str, str]] = {
    "frames":         {"ru": "Кадров",       "en": "Frames"},
    "patterns":       {"ru": "Повторов",      "en": "Repeats"},
    "duration":       {"ru": "Длительность",  "en": "Duration"},
    "accuracy":       {"ru": "Схожесть",      "en": "Similarity"},
    "time_left":      {"ru": "Осталось",      "en": "Remaining"},
    "batch_progress": {"ru": "Прогресс",      "en": "Progress"},
}

_ITEMS = [
    "frames", "patterns", "duration",
    "accuracy", "time_left", "batch_progress",
]

_DEFAULTS = {
    "frames":         "0",
    "patterns":       "0",
    "duration":       "00:00:00",
    "accuracy":       "0%",
    "time_left":      "--:--:--",
    "batch_progress": "0/0",
}


class StatsBar(tk.Frame):
    """Верхняя строка с метриками."""

    def __init__(self, parent: tk.Widget,
                 colors: dict,
                 lang: str = "ru") -> None:
        super().__init__(parent, bg=colors["bg"])
        self.colors = colors
        self._lang  = lang
        self.metric_values:  dict[str, tk.StringVar] = {}
        self._label_widgets: dict[str, tk.Label]     = {}
        self._build()

    def _build(self) -> None:
        for i, key in enumerate(_ITEMS):
            var = tk.StringVar(value=_DEFAULTS.get(key, "0"))
            self.metric_values[key] = var

            card = tk.Frame(self, bg=self.colors["card"],
                            width=160, height=72)
            card.pack_propagate(False)
            card.pack(side=tk.LEFT,
                      padx=(0 if i == 0 else 8, 0),
                      expand=True, fill=tk.BOTH)

            inner = tk.Frame(card, bg=self.colors["card"],
                             padx=12, pady=8)
            inner.pack(fill=tk.BOTH, expand=True)

            tk.Label(inner,
                     textvariable=var,
                     font=("Inter", 18, "bold"),
                     bg=self.colors["card"],
                     fg=self.colors["text"],
                     width=10, anchor="w").pack(anchor="w")

            lbl = tk.Label(
                inner,
                text=_LABELS[key].get(self._lang, key),
                font=("Inter", 9),
                bg=self.colors["card"],
                fg=self.colors["text_secondary"])
            lbl.pack(anchor="w")
            self._label_widgets[key] = lbl

    def set_lang(self, lang: str) -> None:
        """Обновить подписи без пересборки."""
        if self._lang == lang:
            return
        self._lang = lang
        for key, lbl in self._label_widgets.items():
            try:
                lbl.config(text=_LABELS[key].get(lang, key))
            except Exception:
                pass

    def set(self, key: str, value: str) -> None:
        if key in self.metric_values:
            self.metric_values[key].set(value)