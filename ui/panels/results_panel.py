# ui/panels/results_panel.py
from __future__ import annotations

import tkinter as tk
from typing import Callable

from ui.app_state import AppState
from ui.widgets.glow_button import GlowButton
from ui.widgets.virtual_list import VirtualResultsList

_UI: dict[str, dict[str, str]] = {
    "results":    {"ru": "Результаты",         "en": "Results"},
    "prev":       {"ru": "◀ Пред.",             "en": "◀ Prev"},
    "next":       {"ru": "След. ▶",             "en": "Next ▶"},
    "no_results": {"ru": "Результатов пока нет.\nЗапустите анализ.",
                   "en": "No results yet.\nRun analysis."},
    "categories": {"ru": "Категории движений", "en": "Motion categories"},
    "all":        {"ru": "Все",                "en": "All"},
    "done_found": {"ru": "Готово. Найдено",    "en": "Done. Found"},
    "repeats":    {"ru": "повторов",           "en": "repeats"},
}


class ResultsPanel(tk.Frame):
    """Правая панель — список совпадений, категории, экспорт."""

    def __init__(self, parent: tk.Widget,
                 state:     AppState,
                 colors:    dict,
                 callbacks: dict,
                 lang:      str = "ru") -> None:
        super().__init__(parent, bg=colors["card"])
        self.state     = state
        self.colors    = colors
        self.callbacks = callbacks
        self._lang     = lang
        self._build()

    # ── Язык ──────────────────────────────────────────────────────────────

    def set_lang(self, lang: str) -> None:
        if self._lang == lang:
            return
        self._lang = lang
        self._update_lang_labels()
        self.vlist.set_lang(lang)

    def _update_lang_labels(self) -> None:
        lang = self._lang
        try:
            self._title_lbl.config(text=_UI["results"][lang])
        except Exception:
            pass
        try:
            self.prev_btn.set_text(_UI["prev"][lang])
        except Exception:
            pass
        try:
            self.next_btn.set_text(_UI["next"][lang])
        except Exception:
            pass
        try:
            self._no_results_lbl.config(
                text=_UI["no_results"][lang])
        except Exception:
            pass
        try:
            self._cats_title_lbl.config(
                text=_UI["categories"][lang])
        except Exception:
            pass

    # ── Построение ────────────────────────────────────────────────────────

    def _build(self) -> None:
        c    = self.colors
        lang = self._lang

        inner = tk.Frame(self, bg=c["card"], padx=14, pady=12)
        inner.pack(fill=tk.BOTH, expand=True)

        # Заголовок
        self._title_lbl = tk.Label(
            inner,
            text=_UI["results"][lang],
            font=("Inter", 13, "bold"),
            bg=c["card"], fg=c["text"])
        self._title_lbl.pack(anchor="w", pady=(0, 8))

        # Навигация
        nav = tk.Frame(inner, bg=c["card"])
        nav.pack(fill=tk.X, pady=(0, 6))

        self.prev_btn = GlowButton(
            nav,
            text=_UI["prev"][lang],
            command=self.callbacks.get("on_prev"),
            bg_color=c["highlight"],
            hover_color=c["border"],
            width=80, height=28)
        self.prev_btn.pack(side=tk.LEFT)

        self.match_counter = tk.Label(
            nav, text="0 / 0",
            font=("Inter", 10),
            bg=c["card"], fg=c["text"],
            width=10)
        self.match_counter.pack(side=tk.LEFT, expand=True)

        self.next_btn = GlowButton(
            nav,
            text=_UI["next"][lang],
            command=self.callbacks.get("on_next"),
            bg_color=c["highlight"],
            hover_color=c["border"],
            width=80, height=28)
        self.next_btn.pack(side=tk.RIGHT)

        # Экспорт
        exp_row = tk.Frame(inner, bg=c["card"])
        exp_row.pack(fill=tk.X, pady=(0, 8))
        for lbl, key in [("JSON", "on_export_json"),
                          ("TXT",  "on_export_txt"),
                          ("EDL",  "on_export_edl")]:
            GlowButton(
                exp_row,
                text=lbl,
                command=self.callbacks.get(key),
                bg_color=c["highlight"],
                hover_color=c["border"],
                width=56, height=26,
                font=("Inter", 9, "bold")).pack(
                side=tk.LEFT, padx=(0, 4))

        # Заглушка «нет результатов»
        self._no_results = tk.Frame(inner, bg=c["card"])
        self._no_results.pack(fill=tk.BOTH, expand=True)
        self._no_results_lbl = tk.Label(
            self._no_results,
            text=_UI["no_results"][lang],
            font=("Inter", 11),
            bg=c["card"], fg=c["text_secondary"],
            justify="center")
        self._no_results_lbl.pack(expand=True)

        # Категории
        self._cats_frame = tk.Frame(inner, bg=c["card"])
        self._cats_title_lbl: tk.Label | None = None

        # Список
        self._list_outer = tk.Frame(inner, bg=c["card"])
        self.vlist = VirtualResultsList(
            self._list_outer, c,
            on_select=self.callbacks.get(
                "on_select", lambda _: None),
            lang=lang)

    # ── Публичный API ─────────────────────────────────────────────────────

    def show_results(self,
                     matches:    list,
                     categories: list[str]) -> None:
        self._no_results.pack_forget()
        self._rebuild_categories(categories)
        self._list_outer.pack(fill=tk.BOTH, expand=True)
        self.vlist.set_matches(matches)

    def hide_results(self) -> None:
        self._cats_frame.pack_forget()
        self._list_outer.pack_forget()
        self._no_results.pack(fill=tk.BOTH, expand=True)

    def update_counter(self, index: int, total: int) -> None:
        self.match_counter.config(
            text=(f"{index + 1} / {total}"
                  if total > 0 else "0 / 0"))

    # ── Категории ─────────────────────────────────────────────────────────

    def _rebuild_categories(self,
                             categories: list[str]) -> None:
        for w in self._cats_frame.winfo_children():
            w.destroy()
        self._cats_title_lbl = None

        if not categories:
            self._cats_frame.pack_forget()
            return

        c    = self.colors
        lang = self._lang

        self._cats_frame.pack(fill=tk.X, pady=(0, 6))

        self._cats_title_lbl = tk.Label(
            self._cats_frame,
            text=_UI["categories"][lang],
            font=("Inter", 10, "bold"),
            bg=c["card"], fg=c["text"])
        self._cats_title_lbl.pack(anchor="w", pady=(0, 4))

        total  = len(self.state.matches)
        btns:  dict[str, GlowButton] = {}

        # Подсчёт по категориям
        counts: dict[str, int] = {}
        for m in self.state.matches:
            cc = m.get("category", "")
            if cc:
                counts[cc] = counts.get(cc, 0) + 1

        def _make_filter(cat: str):
            def _cb():
                self.state.active_filter_cat = cat
                self.vlist.set_filter(cat=cat)
                for cc, b in btns.items():
                    is_active = (self.state.active_filter_cat == cc)
                    b.set_bg(c["accent"] if is_active
                             else c["highlight"])
            return _cb

        # Кнопка «Все»
        all_lbl = f"{_UI['all'][lang]}  ({total})"
        all_btn = GlowButton(
            self._cats_frame,
            text=all_lbl,
            command=_make_filter(""),
            bg_color=(c["accent"]
                      if self.state.active_filter_cat == ""
                      else c["highlight"]),
            hover_color=c["accent_hover"],
            width=110, height=24,
            font=("Inter", 9, "bold"))
        all_btn.pack(anchor="w", pady=(0, 2))
        btns[""] = all_btn

        # Кнопки категорий
        for cat in categories:
            cnt = counts.get(cat, 0)
            if cnt == 0:
                continue
            btn = GlowButton(
                self._cats_frame,
                text=f"{cat}  ({cnt})",
                command=_make_filter(cat),
                bg_color=(c["accent"]
                          if self.state.active_filter_cat == cat
                          else c["highlight"]),
                hover_color=c["accent_hover"],
                width=200, height=24,
                font=("Inter", 9, "bold"))
            btn.pack(anchor="w", pady=1)
            btns[cat] = btn