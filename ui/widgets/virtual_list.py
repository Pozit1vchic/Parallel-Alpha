#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ui/widgets/virtual_list.py"""
from __future__ import annotations

from typing import Callable
import tkinter as tk

from ui.widgets.smooth_scroll import SmoothScrollMixin


def _fmt_hms(secs: float) -> str:
    s  = max(0.0, float(secs))
    h  = int(s // 3600)
    m  = int((s % 3600) // 60)
    ss = int(s % 60)
    return f"{h:02d}:{m:02d}:{ss:02d}"


_DIRECTION_LABELS: dict[str, dict[str, str]] = {
    "forward":      {"ru": "Лицом к камере",    "en": "Facing camera",  "arrow": "↑"},
    "left":         {"ru": "Смотрит влево",      "en": "Facing left",    "arrow": "←"},
    "right":        {"ru": "Смотрит вправо",     "en": "Facing right",   "arrow": "→"},
    "back":         {"ru": "Спиной к камере",    "en": "Facing away",    "arrow": "↓"},
    "forward-right":{"ru": "Пол-оборота вправо", "en": "Half-turn right","arrow": "↗"},
    "forward-left": {"ru": "Пол-оборота влево",  "en": "Half-turn left", "arrow": "↖"},
    "back-right":   {"ru": "Спиной-вправо",      "en": "Back-right",     "arrow": "↘"},
    "back-left":    {"ru": "Спиной-влево",       "en": "Back-left",      "arrow": "↙"},
    "unknown":      {"ru": "Неизвестно",         "en": "Unknown",        "arrow": "?"},
}


def _dir_label(direction: str, lang: str = "ru") -> str:
    entry = _DIRECTION_LABELS.get(direction, _DIRECTION_LABELS["unknown"])
    return entry.get(lang, entry["ru"])


def _dir_arrow(direction: str) -> str:
    return _DIRECTION_LABELS.get(
        direction, _DIRECTION_LABELS["unknown"])["arrow"]


_NO_RESULTS = {"ru": "Нет результатов", "en": "No results"}
_CTX_EXCLUDE = {"ru": "Исключить из списка", "en": "Exclude from list"}
_CTX_RESTORE = {"ru": "Восстановить все",    "en": "Restore all"}


class VirtualResultsList(SmoothScrollMixin):
    """Виртуализированный список результатов."""

    ROW_H = 62

    def __init__(self, parent: tk.Widget,
                 colors: dict[str, str],
                 on_select: Callable[[int], None],
                 lang: str = "ru") -> None:
        self.colors    = colors
        self.on_select = on_select
        self.lang      = lang

        self._all_matches: list[dict] = []
        self._filtered:    list[int]  = []
        self._hidden_set:  set[int]   = set()
        self._selected_fi: int        = -1

        self._sort_key:   str   = "sim"
        self._sort_desc:  bool  = True
        self._filter_cat: str   = ""
        self._min_sim:    float = 0.0

        outer = tk.Frame(parent, bg=colors["bg"])
        outer.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(outer, bg=colors["bg"],
                                 highlightthickness=0)
        self._sb = tk.Scrollbar(outer, orient=tk.VERTICAL,
                                command=self._on_scroll_cmd)
        self._canvas.configure(yscrollcommand=self._sb.set)
        self._sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._canvas.bind("<Configure>",       self._on_configure)
        self._canvas.bind("<Button-1>",        self._on_click)
        self._canvas.bind("<Double-Button-1>", self._on_dbl_click)
        self._canvas.bind("<Button-3>",        self._on_right_click)
        self._bind_smooth_scroll(self._canvas, self._scroll_units)

        self._ctx = tk.Menu(self._canvas, tearoff=0,
                            bg=colors["card"], fg=colors["text"])
        self._rebuild_ctx()
        self._ctx_fi: int = -1

    # ── Публичный API ─────────────────────────────────────────────────────

    def set_matches(self, matches: list[dict]) -> None:
        self._all_matches = matches
        self._hidden_set.clear()
        self._selected_fi = -1
        self._apply_filter()
        self._update_scrollregion()
        self._render()

    def select_by_match_idx(self, mi: int) -> None:
        for fi, idx in enumerate(self._filtered):
            if idx == mi:
                self._selected_fi = fi
                self._ensure_visible(fi)
                self._render()
                return

    def set_filter(self, cat: str = "",
                   min_sim: float = 0.0) -> None:
        self._filter_cat = cat
        self._min_sim    = min_sim
        self._apply_filter()
        self._update_scrollregion()
        self._render()

    def set_sort(self, key: str, desc: bool = True) -> None:
        self._sort_key  = key
        self._sort_desc = desc
        self._apply_filter()
        self._render()

    def set_lang(self, lang: str) -> None:
        self.lang = lang
        self._rebuild_ctx()
        self._render()

    def get_visible_count(self) -> int:
        return len(self._filtered)

    def restore_all_matches(self) -> None:
        self._restore_all()

    # ── Внутреннее ────────────────────────────────────────────────────────

    def _rebuild_ctx(self) -> None:
        self._ctx.delete(0, "end")
        self._ctx.add_command(
            label=_CTX_EXCLUDE.get(self.lang, "Exclude"),
            command=self._exclude)
        self._ctx.add_separator()
        self._ctx.add_command(
            label=_CTX_RESTORE.get(self.lang, "Restore"),
            command=self._restore_all)

    def _apply_filter(self) -> None:
        result: list[int] = []
        for i, m in enumerate(self._all_matches):
            if i in self._hidden_set:
                continue
            if m.get("sim", 0.0) < self._min_sim:
                continue
            if self._filter_cat:
                if m.get("category", "") != self._filter_cat:
                    continue
            result.append(i)

        result.sort(
            key=lambda i: self._all_matches[i].get(self._sort_key, 0.0),
            reverse=self._sort_desc)
        self._filtered    = result
        self._selected_fi = max(0, min(
            self._selected_fi, len(self._filtered) - 1))

    def _on_configure(self, _e: tk.Event) -> None:
        self._update_scrollregion()
        self._render()

    def _update_scrollregion(self) -> None:
        if not self._filtered:
            self._canvas.config(scrollregion=(0, 0, 0, 0))
            self._canvas.yview_moveto(0)
            return
        total_h = len(self._filtered) * self.ROW_H
        w       = self._canvas.winfo_width() or 380
        ch      = self._canvas.winfo_height() or 400

        # Скролл только если контент больше видимой области
        if total_h <= ch:
            self._canvas.config(
                scrollregion=(0, 0, w, total_h))
            self._canvas.yview_moveto(0)
            # Скрываем скроллбар
            try:
                self._sb.pack_forget()
            except Exception:
                pass
        else:
            self._canvas.config(
                scrollregion=(0, 0, w, total_h))
            # Показываем скроллбар
            try:
                if not self._sb.winfo_ismapped():
                    self._sb.pack(
                        side=tk.RIGHT, fill=tk.Y)
            except Exception:
                pass

    def _scroll_units(self, n: int,
                       unit: str = "units") -> None:
        if not self._filtered:
            return
        total_h = len(self._filtered) * self.ROW_H
        ch      = self._canvas.winfo_height() or 400
        if total_h <= ch:
            return          # нечего скроллить
        self._canvas.yview_scroll(n, unit)
        self._render()

    def _on_scroll_cmd(self, *args) -> None:
        if not self._filtered:
            return
        self._canvas.yview(*args)
        self._render()

    def _ensure_visible(self, fi: int) -> None:
        if not self._filtered:
            return
        total_h  = len(self._filtered) * self.ROW_H
        ch       = self._canvas.winfo_height() or 400
        item_top = fi * self.ROW_H
        item_bot = item_top + self.ROW_H
        view_top = self._canvas.yview()[0] * total_h
        view_bot = self._canvas.yview()[1] * total_h
        if item_top < view_top:
            self._canvas.yview_moveto(item_top / max(total_h, 1))
        elif item_bot > view_bot:
            self._canvas.yview_moveto(
                (item_bot - ch) / max(total_h, 1))

    def _render(self) -> None:
        self._canvas.delete("all")
        c = self.colors

        if not self._filtered:
            cw = self._canvas.winfo_width() or 380
            ch = self._canvas.winfo_height() or 200
            self._canvas.create_text(
                cw // 2, ch // 2,
                text=_NO_RESULTS.get(self.lang, "No results"),
                fill=c["text_secondary"],
                font=("Inter", 11))
            return

        self._update_scrollregion()
        cw = self._canvas.winfo_width() or 380
        ch = self._canvas.winfo_height() or 400

        view_top = self._canvas.yview()[0]
        total_h  = len(self._filtered) * self.ROW_H
        y_offset = view_top * total_h

        start_fi = max(0, int(y_offset // self.ROW_H) - 1)
        end_fi   = min(len(self._filtered),
                       start_fi + int(ch // self.ROW_H) + 3)

        for fi in range(start_fi, end_fi):
            mi  = self._filtered[fi]
            m   = self._all_matches[mi]
            y   = fi * self.ROW_H - y_offset
            sel = fi == self._selected_fi

            # Фон
            if sel:
                bg = c.get("active_row", "#1e3a5f")
            elif fi % 2 == 0:
                bg = c.get("highlight", "#1e293b")
            else:
                bg = c.get("bg", "#0d1117")

            self._canvas.create_rectangle(
                0, y, cw, y + self.ROW_H - 1,
                fill=bg, outline="")

            sim = m.get("sim", 0.0)
            sim_pct = sim * 100

            # Цветная полоска слева (3px)
            bar_col = (c.get("success", "#22c55e") if sim >= 0.85
                       else c.get("accent", "#3b82f6") if sim >= 0.70
                       else c.get("error", "#ef4444"))
            self._canvas.create_rectangle(
                0, y + 2, 3, y + self.ROW_H - 3,
                fill=bar_col, outline="")

            # Мини прогресс-бар схожести
            bar_w = int((cw - 16) * sim)
            self._canvas.create_rectangle(
                8, y + self.ROW_H - 8,
                8 + (cw - 16), y + self.ROW_H - 5,
                fill=c.get("highlight", "#1e293b"), outline="")
            if bar_w > 0:
                self._canvas.create_rectangle(
                    8, y + self.ROW_H - 8,
                    8 + bar_w, y + self.ROW_H - 5,
                    fill=bar_col, outline="")

            # Процент схожести (крупный)
            self._canvas.create_text(
                12, y + 14,
                text=f"{sim_pct:.1f}%",
                anchor="nw",
                fill=bar_col,
                font=("Inter", 12, "bold"))

            # Направление
            direction = m.get("direction", "forward")
            arrow     = _dir_arrow(direction)
            dir_text  = _dir_label(direction, self.lang)
            self._canvas.create_text(
                12, y + 34,
                text=f"{arrow}  {dir_text}",
                anchor="nw",
                fill=c["text_secondary"],
                font=("Inter", 9))

            # Времена — правая сторона
            t1 = _fmt_hms(m.get("t1", 0.0))
            t2 = _fmt_hms(m.get("t2", 0.0))
            self._canvas.create_text(
                cw - 8, y + 14,
                text=t1,
                anchor="ne",
                fill=c["text"],
                font=("Inter", 10, "bold"))
            self._canvas.create_text(
                cw - 8, y + 34,
                text=f"→  {t2}",
                anchor="ne",
                fill=c["text_secondary"],
                font=("Inter", 9))

            # Разделитель
            self._canvas.create_line(
                0, y + self.ROW_H - 1,
                cw, y + self.ROW_H - 1,
                fill=c["border"])

    def _fi_at_y(self, y_canvas: int) -> int:
        if not self._filtered:
            return -1
        view_top = self._canvas.yview()[0]
        total_h  = len(self._filtered) * self.ROW_H
        abs_y    = y_canvas + view_top * total_h
        fi       = int(abs_y // self.ROW_H)
        return fi if 0 <= fi < len(self._filtered) else -1

    def _on_click(self, event: tk.Event) -> None:
        fi = self._fi_at_y(event.y)
        if fi < 0:
            return
        self._selected_fi = fi
        self._render()
        mi = self._filtered[fi]
        self.on_select(mi)

    def _on_dbl_click(self, event: tk.Event) -> None:
        self._on_click(event)

    def _on_right_click(self, event: tk.Event) -> None:
        fi = self._fi_at_y(event.y)
        if fi < 0:
            return
        self._ctx_fi = fi
        self._ctx.post(event.x_root, event.y_root)

    def _exclude(self) -> None:
        if 0 <= self._ctx_fi < len(self._filtered):
            mi = self._filtered[self._ctx_fi]
            self._hidden_set.add(mi)
            self._apply_filter()
            self._update_scrollregion()
            self._render()

    def _restore_all(self) -> None:
        self._hidden_set.clear()
        self._apply_filter()
        self._update_scrollregion()
        self._render()