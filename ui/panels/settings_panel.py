#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ui/panels/settings_panel.py"""
from __future__ import annotations

import json
import os
import tkinter as tk

from ui.app_state import AppState
from ui.widgets.glow_button import GlowButton

_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config.json")


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_config(data: dict) -> None:
    try:
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ── Локализация ───────────────────────────────────────────────────────────────

_L: dict[str, dict[str, str]] = {
    "title":          {"ru": "Настройки анализа",     "en": "Analysis Settings"},
    "accuracy":       {"ru": "Точность",               "en": "Accuracy"},
    "threshold":      {"ru": "Порог схожести",         "en": "Similarity threshold"},
    "threshold_tip":  {
        "ru": "Минимальная схожесть поз (%). Выше = строже.",
        "en": "Minimum pose similarity (%). Higher = stricter.",
    },
    "quality":        {"ru": "Качество анализа",       "en": "Analysis quality"},
    "quality_tip":    {
        "ru": "Быстро — меньше кадров анализируется. Максимум — каждый кадр.",
        "en": "Fast — fewer frames analysed. Maximum — every frame.",
    },
    "fast":           {"ru": "Быстро",                 "en": "Fast"},
    "medium":         {"ru": "Средне",                 "en": "Medium"},
    "maximum":        {"ru": "Максимум",               "en": "Maximum"},
    "timing":         {"ru": "Временные параметры",    "en": "Timing"},
    "scene_interval": {"ru": "Интервал сцен",          "en": "Scene interval"},
    "scene_tip":      {
        "ru": "Минимальный разрыв между сценами (секунды).",
        "en": "Minimum gap between scenes (seconds).",
    },
    "match_gap":      {"ru": "Мин. интервал повторов", "en": "Min. repeat gap"},
    "match_tip":      {
        "ru": "Игнорировать повторы ближе указанного времени (сек).",
        "en": "Ignore repeats closer than this time (sec).",
    },
    "options":        {"ru": "Опции",                  "en": "Options"},
    "scale":          {"ru": "Нормализация размера",   "en": "Scale normalization"},
    "scale_tip":      {
        "ru": "Сравнивать позы независимо от расстояния до камеры.",
        "en": "Compare poses regardless of distance to camera.",
    },
    "mirror":         {"ru": "Зеркальные позы",        "en": "Mirror poses"},
    "mirror_tip":     {
        "ru": "Считать зеркально отражённые движения похожими.",
        "en": "Treat mirrored movements as similar.",
    },
    "weights":        {"ru": "Веса частей тела",       "en": "Body part weights"},
    "weights_tip":    {
        "ru": "Плечи и бёдра важнее других суставов при сравнении.",
        "en": "Shoulders and hips are weighted more when comparing.",
    },
    "start":          {"ru": "▶  Старт",               "en": "▶  Start"},
    "stop":           {"ru": "■  Стоп",                "en": "■  Stop"},
    "clear_queue":    {"ru": "Очистить",               "en": "Clear"},
    "sec":            {"ru": "с",                      "en": "s"},
    "pct":            {"ru": "%",                      "en": "%"},
}


def _t(key: str, lang: str) -> str:
    return _L.get(key, {}).get(lang, key)


# ── Нейтральные ключи качества ────────────────────────────────────────────────
# Хранятся в переменной всегда как нейтральный ключ: "fast"/"medium"/"maximum"
# При отображении конвертируются в текущий язык

_QUALITY_KEYS   = ("fast", "medium", "maximum")
_QUALITY_DEFAULT = "medium"


def _quality_key_to_label(key: str, lang: str) -> str:
    """Нейтральный ключ → локализованная метка."""
    if key in _QUALITY_KEYS:
        return _t(key, lang)
    # Попытка обратного маппинга для старых значений
    for k in _QUALITY_KEYS:
        if _t(k, "ru") == key or _t(k, "en") == key:
            return _t(k, lang)
    return _t("medium", lang)


def _quality_label_to_key(label: str) -> str:
    """Локализованная метка → нейтральный ключ."""
    for k in _QUALITY_KEYS:
        if label in (_t(k, "ru"), _t(k, "en"), k):
            return k
    return _QUALITY_DEFAULT


# ── Tooltip ───────────────────────────────────────────────────────────────────

class _Tooltip:
    def __init__(self, widget: tk.Widget,
                 text: str, colors: dict) -> None:
        self._w    = widget
        self._text = text
        self._c    = colors
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def update_text(self, text: str) -> None:
        self._text = text

    def _show(self, _=None) -> None:
        try:
            x = self._w.winfo_rootx() + 20
            y = self._w.winfo_rooty() + 20
            self._tip = tk.Toplevel(self._w)
            self._tip.wm_overrideredirect(True)
            self._tip.wm_geometry(f"+{x}+{y}")
            self._tip.attributes("-topmost", True)
            f = tk.Frame(
                self._tip,
                bg=self._c.get("card", "#14171c"),
                highlightbackground=self._c.get("border", "#2a2f38"),
                highlightthickness=1)
            f.pack()
            tk.Label(
                f, text=self._text,
                font=("Inter", 9),
                bg=self._c.get("card", "#14171c"),
                fg=self._c.get("text", "#ffffff"),
                wraplength=240, justify="left",
                padx=10, pady=6).pack()
        except Exception:
            pass

    def _hide(self, _=None) -> None:
        if self._tip:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


# ── CheckBox ──────────────────────────────────────────────────────────────────

class _CheckBox(tk.Canvas):
    """Кастомный чекбокс без артефактов."""
    S = 18

    def __init__(self, parent, variable: tk.BooleanVar,
                 colors: dict, **kwargs) -> None:
        for k in ("text", "font", "fg", "anchor", "relief", "bd",
                  "padx", "pady", "activebackground",
                  "activeforeground", "selectcolor"):
            kwargs.pop(k, None)

        self._c   = colors
        self._var = variable
        self._hov = False

        super().__init__(
            parent,
            width=self.S, height=self.S,
            bg=colors.get("card", "#14171c"),
            highlightthickness=0, bd=0,
            cursor="hand2", **kwargs)

        self._draw()
        self._var.trace_add("write", self._on_var)
        self.bind("<Destroy>", self._on_destroy)
        self.bind("<Enter>",   lambda _e: self._set_hov(True))
        self.bind("<Leave>",   lambda _e: self._set_hov(False))
        self.bind("<Button-1>", lambda _e: self._toggle())

    def _on_destroy(self, _=None) -> None:
        try:
            for name, _, cb in self._var.trace_info():
                if cb == str(self._on_var):
                    self._var.trace_remove(name, cb)
        except Exception:
            pass

    def _on_var(self, *_) -> None:
        if self.winfo_exists():
            self._draw()

    def _set_hov(self, v: bool) -> None:
        self._hov = v
        self._draw()

    def _toggle(self) -> None:
        try:
            self._var.set(not self._var.get())
        except Exception:
            pass

    def _draw(self) -> None:
        if not self.winfo_exists():
            return
        self.delete("all")
        c       = self._c
        checked = self._var.get()
        S, r    = self.S, 4

        if checked:
            fill_c   = c.get("accent",    "#3b82f6")
            border_c = c.get("accent",    "#3b82f6")
        elif self._hov:
            fill_c   = c.get("highlight", "#1e293b")
            border_c = c.get("accent",    "#3b82f6")
        else:
            fill_c   = c.get("bg",        "#0a0c10")
            border_c = c.get("border",    "#2a2f38")

        self._rrect(0, 0, S, S, r, fill_c, border_c)

        if checked:
            lx, ly  = 3,        S // 2 + 1
            mx, my  = S // 2-1, S - 4
            rx_, ry = S - 3,    3
            self.create_line(lx, ly, mx, my,
                             fill="#ffffff", width=2,
                             capstyle=tk.ROUND, joinstyle=tk.ROUND)
            self.create_line(mx, my, rx_, ry,
                             fill="#ffffff", width=2,
                             capstyle=tk.ROUND, joinstyle=tk.ROUND)

    def _rrect(self, x1, y1, x2, y2, r,
               fill: str, border: str) -> None:
        r = max(0, min(r, (x2-x1)//2, (y2-y1)//2))
        f = {"fill": fill, "outline": ""}
        self.create_rectangle(x1+r, y1,   x2-r, y2,   **f)
        self.create_rectangle(x1,   y1+r, x2,   y2-r, **f)
        corners = [
            (x1,       y1,       90),
            (x2-r*2,   y1,        0),
            (x1,       y2-r*2,  180),
            (x2-r*2,   y2-r*2,  270),
        ]
        for ax, ay, st in corners:
            self.create_arc(ax, ay, ax+r*2, ay+r*2,
                            start=st, extent=90, fill=fill, outline="")
        self.create_line(x1+r, y1,   x2-r, y1,   fill=border, width=1)
        self.create_line(x1+r, y2,   x2-r, y2,   fill=border, width=1)
        self.create_line(x1,   y1+r, x1,   y2-r, fill=border, width=1)
        self.create_line(x2,   y1+r, x2,   y2-r, fill=border, width=1)
        for ax, ay, st in corners:
            self.create_arc(ax, ay, ax+r*2, ay+r*2,
                            start=st, extent=90,
                            outline=border, style="arc", width=1)


# ── SegmentedQuality ──────────────────────────────────────────────────────────

class _SegmentedQuality(tk.Canvas):
    """
    Сегментированный выбор качества.
    Работает с НЕЙТРАЛЬНЫМИ ключами ("fast"/"medium"/"maximum") внутри,
    отображает локализованные метки снаружи.
    """
    H = 32

    def __init__(self, parent,
                 quality_key_var: tk.StringVar,
                 lang: str,
                 colors: dict) -> None:
        self._var      = quality_key_var   # хранит нейтральный ключ
        self._lang     = lang
        self._c        = colors
        self._hov_i    = -1
        self._w_total  = 0
        self._last_key = quality_key_var.get()
        self._alive    = True

        try:
            pbg = parent.cget("bg")
        except Exception:
            pbg = colors.get("card", "#14171c")

        super().__init__(parent, height=self.H, bg=pbg,
                         highlightthickness=0, bd=0, cursor="hand2")

        self.bind("<Configure>", self._on_configure)
        self.bind("<Motion>",    self._on_motion)
        self.bind("<Leave>",     self._on_leave)
        self.bind("<Button-1>",  self._on_click)
        self.bind("<Destroy>",   self._on_destroy)
        self._poll()

    def _on_destroy(self, _=None) -> None:
        self._alive = False

    def _poll(self) -> None:
        if not self._alive:
            return
        try:
            if not self.winfo_exists():
                self._alive = False
                return
            v = self._var.get()
            if v != self._last_key:
                self._last_key = v
                self._draw()
            self.after(100, self._poll)
        except Exception:
            self._alive = False

    def _labels(self) -> list[str]:
        return [_t(k, self._lang) for k in _QUALITY_KEYS]

    def _on_configure(self, e) -> None:
        self._w_total = e.width
        self._draw()

    def _on_motion(self, e) -> None:
        i = self._idx_at(e.x)
        if i != self._hov_i:
            self._hov_i = i
            self._draw()

    def _on_leave(self, _=None) -> None:
        self._hov_i = -1
        self._draw()

    def _on_click(self, e) -> None:
        i = self._idx_at(e.x)
        if 0 <= i < len(_QUALITY_KEYS):
            # Сохраняем нейтральный ключ
            self._var.set(_QUALITY_KEYS[i])

    def _idx_at(self, x: int) -> int:
        N = len(_QUALITY_KEYS)
        if N == 0 or self._w_total <= 0:
            return -1
        return max(0, min(N-1, int(x / (self._w_total / N))))

    def _draw(self) -> None:
        if not self._alive:
            return
        try:
            if not self.winfo_exists():
                return
        except Exception:
            return

        self.delete("all")
        c      = self._c
        labels = self._labels()
        N      = len(labels)
        W      = self._w_total or self.winfo_width() or 300
        H      = self.H
        if N == 0 or W == 0:
            return

        seg_w  = W / N
        r      = 8
        cur_key = self._var.get()

        self._rr(0, 0, W, H, r,
                 c.get("bg", "#0a0c10"),
                 c.get("border", "#2a2f38"))

        for i, key in enumerate(_QUALITY_KEYS):
            x1       = int(i * seg_w)
            x2       = int((i+1) * seg_w)
            selected = (key == cur_key)
            hovered  = (i == self._hov_i and not selected)

            if selected:
                fill = c.get("accent", "#3b82f6")
            elif hovered:
                fill = c.get("highlight", "#1e293b")
            else:
                fill = None

            if fill:
                rl = r if i == 0     else 0
                rr = r if i == N - 1 else 0
                self._seg(x1, 0, x2, H, rl, rr, fill)

        for i in range(1, N):
            xi = int(i * seg_w)
            self.create_line(xi, 4, xi, H-4,
                             fill=c.get("border", "#2a2f38"), width=1)

        for i, (key, label) in enumerate(zip(_QUALITY_KEYS, labels)):
            x1 = int(i * seg_w)
            x2 = int((i+1) * seg_w)
            selected = (key == cur_key)
            fg   = "#ffffff" if selected else c.get("text_secondary", "#6b7280")
            font = ("Inter", 9, "bold") if selected else ("Inter", 9)
            self.create_text((x1+x2)//2, H//2,
                             text=label, fill=fg,
                             font=font, anchor="center")

    def _rr(self, x1, y1, x2, y2, r, fill, outline=None) -> None:
        r = max(0, min(r, (x2-x1)//2, (y2-y1)//2))
        kw = {"fill": fill, "outline": ""}
        self.create_rectangle(x1+r, y1,   x2-r, y2,   **kw)
        self.create_rectangle(x1,   y1+r, x2,   y2-r, **kw)
        for st, ax, ay in [(90,x1,y1), (0,x2-r*2,y1),
                            (180,x1,y2-r*2), (270,x2-r*2,y2-r*2)]:
            self.create_arc(ax, ay, ax+r*2, ay+r*2,
                            start=st, extent=90, **kw)
        if outline:
            self.create_line(x1+r, y1,   x2-r, y1,   fill=outline)
            self.create_line(x1+r, y2,   x2-r, y2,   fill=outline)
            self.create_line(x1,   y1+r, x1,   y2-r, fill=outline)
            self.create_line(x2,   y1+r, x2,   y2-r, fill=outline)
            for st, ax, ay in [(90,x1,y1), (0,x2-r*2,y1),
                                (180,x1,y2-r*2), (270,x2-r*2,y2-r*2)]:
                self.create_arc(ax, ay, ax+r*2, ay+r*2,
                                start=st, extent=90,
                                outline=outline, style="arc")

    def _seg(self, x1, y1, x2, y2, rl, rr, fill) -> None:
        if rl == 0 and rr == 0:
            self.create_rectangle(x1, y1, x2, y2,
                                  fill=fill, outline="")
            return
        self.create_rectangle(x1+rl, y1, x2-rr, y2,
                              fill=fill, outline="")
        if rl:
            self.create_rectangle(x1, y1+rl, x1+rl, y2-rl,
                                  fill=fill, outline="")
            self.create_arc(x1, y1, x1+rl*2, y1+rl*2,
                            start=90, extent=90, fill=fill, outline="")
            self.create_arc(x1, y2-rl*2, x1+rl*2, y2,
                            start=180, extent=90, fill=fill, outline="")
        if rr:
            self.create_rectangle(x2-rr, y1+rr, x2, y2-rr,
                                  fill=fill, outline="")
            self.create_arc(x2-rr*2, y1, x2, y1+rr*2,
                            start=0, extent=90, fill=fill, outline="")
            self.create_arc(x2-rr*2, y2-rr*2, x2, y2,
                            start=270, extent=90, fill=fill, outline="")


# ── SettingsPanel ─────────────────────────────────────────────────────────────

class SettingsPanel(tk.Frame):
    """Панель настроек анализа со скроллом."""

    def __init__(self, parent, state: AppState,
                 colors: dict, callbacks: dict,
                 lang: str = "ru") -> None:
        super().__init__(parent, bg=colors["card"])
        self.state     = state
        self.colors    = colors
        self.callbacks = callbacks
        self._lang     = lang

        cfg = _load_config()
        ana = cfg.get("analysis", {})

        self.threshold             = tk.IntVar(
            value=int(ana.get("threshold", 75)))
        self.scene_interval        = tk.IntVar(
            value=int(ana.get("scene_interval",
                              ana.get("min_gap", 4))))
        self.match_gap             = tk.IntVar(
            value=int(ana.get("match_gap",
                              ana.get("min_gap", 5))))

        # quality хранит НЕЙТРАЛЬНЫЙ КЛЮЧ: "fast" / "medium" / "maximum"
        saved_q = ana.get("quality", "medium")
        self.quality = tk.StringVar(
            value=_quality_label_to_key(saved_q))

        self.use_scale_invariance  = tk.BooleanVar(
            value=bool(ana.get("scale_norm", True)))
        self.use_mirror_invariance = tk.BooleanVar(
            value=bool(ana.get("use_mirror", False)))
        self.use_body_weights      = tk.BooleanVar(
            value=bool(ana.get("use_body_weights", True)))
        self._batch_size  = tk.IntVar(
            value=int(ana.get("batch_size", 32)))
        self._chunk_size  = tk.IntVar(
            value=int(ana.get("chunk_size", 3000)))
        self._max_results = tk.IntVar(
            value=int(ana.get("max_unique_results", 1000)))

        self._val_labels: dict[str, tk.Label] = {}
        self._canvas: tk.Canvas | None = None

        self._build()

    # ── Публичный API ─────────────────────────────────────────────────────

    def get_vars(self) -> dict:
        return {
            "threshold":             self.threshold,
            "scene_interval":        self.scene_interval,
            "match_gap":             self.match_gap,
            "quality":               self.quality,
            "use_scale_invariance":  self.use_scale_invariance,
            "use_mirror_invariance": self.use_mirror_invariance,
            "use_body_weights":      self.use_body_weights,
        }

    def set_lang(self, lang: str) -> None:
        if self._lang == lang:
            return
        self._lang = lang
        self._rebuild()

    def set_quality(self, value: str) -> None:
        """Принимает нейтральный ключ или локализованную метку."""
        self.quality.set(_quality_label_to_key(value))


    def set_analysis_state(self, running: bool) -> None:
        try:
            if hasattr(self, "start_btn"):
                self.start_btn.set_disabled(running)
            if hasattr(self, "stop_btn"):
                self.stop_btn.set_disabled(not running)
        except Exception:
            pass

    # ── Внутренние методы ─────────────────────────────────────────────────

    def _rebuild(self) -> None:
        self._detach_scroll()
        for w in self.winfo_children():
            w.destroy()
        self._val_labels.clear()
        self._canvas = None
        self._build()

    # ── Построение UI ─────────────────────────────────────────────────────

    def _build(self) -> None:
        c   = self.colors
        lng = self._lang

        outer = tk.Frame(self, bg=c["card"])
        outer.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(outer, bg=c["card"],
                           highlightthickness=0, bd=0)
        self._canvas = canvas

        sb = tk.Scrollbar(outer, orient=tk.VERTICAL,
                          command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=c["card"])

        scrollable.bind(
            "<Configure>",
            lambda _e: canvas.configure(
                scrollregion=canvas.bbox("all")))

        win_id = canvas.create_window(
            (0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(
                win_id, width=e.width - sb.winfo_width()))

        self._attach_scroll(canvas)

        pad = {"padx": 14}

        # Заголовок
        tk.Label(scrollable,
                 text=_t("title", lng),
                 font=("Inter", 13, "bold"),
                 bg=c["card"], fg=c["text"]).pack(
            anchor="w", pady=(14, 6), **pad)
        self._divider(scrollable)

        # ── Группа: Точность ──────────────────────────────────────────────
        g1 = self._group(scrollable, _t("accuracy", lng))

        self._slider_row(
            g1,
            label   = _t("threshold", lng),
            tooltip = _t("threshold_tip", lng),
            var     = self.threshold,
            from_   = 40, to=99, unit="%")

        # Качество — сегментированный контрол
        qrow = tk.Frame(g1, bg=c["card"])
        qrow.pack(fill=tk.X, pady=(0, 4))

        qhdr = tk.Frame(qrow, bg=c["card"])
        qhdr.pack(fill=tk.X, pady=(0, 6))
        tk.Label(qhdr,
                 text=_t("quality", lng),
                 font=("Inter", 10),
                 bg=c["card"],
                 fg=c["text_secondary"]).pack(side=tk.LEFT)
        self._info(qhdr, _t("quality_tip", lng))

        # _SegmentedQuality работает с нейтральными ключами
        seg = _SegmentedQuality(qrow, self.quality, lng, c)
        seg.pack(fill=tk.X)

        # ── Группа: Тайминг ───────────────────────────────────────────────
        g2  = self._group(scrollable, _t("timing", lng))
        su  = _t("sec", lng)
        self._slider_row(
            g2,
            label   = _t("scene_interval", lng),
            tooltip = _t("scene_tip", lng),
            var     = self.scene_interval,
            from_   = 1, to=30, unit=su)
        self._slider_row(
            g2,
            label   = _t("match_gap", lng),
            tooltip = _t("match_tip", lng),
            var     = self.match_gap,
            from_   = 1, to=60, unit=su)

        # ── Группа: Опции ─────────────────────────────────────────────────
        g3 = self._group(scrollable, _t("options", lng))
        for key, tip_key, var in [
            ("scale",   "scale_tip",   self.use_scale_invariance),
            ("mirror",  "mirror_tip",  self.use_mirror_invariance),
            ("weights", "weights_tip", self.use_body_weights),
        ]:
            row = tk.Frame(g3, bg=c["card"])
            row.pack(fill=tk.X, pady=5)
            cb = _CheckBox(row, var, colors=c)
            cb.pack(side=tk.LEFT)
            lbl = tk.Label(row,
                           text=_t(key, lng),
                           font=("Inter", 10),
                           bg=c["card"], fg=c["text"],
                           cursor="hand2")
            lbl.pack(side=tk.LEFT, padx=(8, 4))
            lbl.bind("<Button-1>",
                     lambda _e, v=var: v.set(not v.get()))
            self._info(row, _t(tip_key, lng))

        self._divider(scrollable, pady=(10, 14))

        # ── Кнопки ────────────────────────────────────────────────────────
        bf = tk.Frame(scrollable, bg=c["card"])
        bf.pack(fill=tk.X, pady=(0, 16), **pad)

        self.start_btn = GlowButton(
            bf,
            text        = _t("start", lng),
            command     = self._on_start,
            bg_color    = c["accent"],
            hover_color = c.get("accent_hover", "#1d4ed8"),
            height      = 44,
            font        = ("Inter", 11, "bold"))
        self.start_btn.pack(fill=tk.X, pady=(0, 8))

        stop_row = tk.Frame(bf, bg=c["card"])
        stop_row.pack(fill=tk.X)

        self.stop_btn = GlowButton(
            stop_row,
            text        = _t("stop", lng),
            command     = self.callbacks.get("on_stop"),
            bg_color    = c["highlight"],
            hover_color = c["border"],
            height      = 36,
            font        = ("Inter", 10, "bold"))
        self.stop_btn.pack(
            side=tk.LEFT, fill=tk.X,
            expand=True, padx=(0, 4))
        self.stop_btn.set_disabled(True)

        self._clear_queue_btn = GlowButton(
            stop_row,
            text        = _t("clear_queue", lng),
            command     = self._on_clear_queue,
            bg_color    = c["highlight"],
            hover_color = c["border"],
            width       = 90,
            height      = 36,
            font        = ("Inter", 9, "bold"))
        self._clear_queue_btn.pack(side=tk.LEFT)

    # ── Скролл ────────────────────────────────────────────────────────────

    def _scroll_fn(self, event) -> None:
        cv = self._canvas
        if cv is None:
            return
        try:
            if not cv.winfo_exists():
                return
        except Exception:
            return
        if event.delta:
            cv.yview_scroll(-1 * (event.delta // 120), "units")
        elif event.num == 4:
            cv.yview_scroll(-1, "units")
        elif event.num == 5:
            cv.yview_scroll(1, "units")

    def _attach_scroll(self, canvas: tk.Canvas) -> None:
        def _enter(_e):
            canvas.bind_all("<MouseWheel>", self._scroll_fn)
            canvas.bind_all("<Button-4>",   self._scroll_fn)
            canvas.bind_all("<Button-5>",   self._scroll_fn)

        def _leave(_e):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        self.bind("<Enter>",  _enter, add="+")
        self.bind("<Leave>",  _leave, add="+")
        canvas.bind("<Enter>", _enter, add="+")
        canvas.bind("<Leave>", _leave, add="+")

    def _detach_scroll(self) -> None:
        try:
            if self._canvas and self._canvas.winfo_exists():
                self._canvas.unbind_all("<MouseWheel>")
                self._canvas.unbind_all("<Button-4>")
                self._canvas.unbind_all("<Button-5>")
        except Exception:
            pass

    # ── Хелперы ───────────────────────────────────────────────────────────

    def _info(self, parent: tk.Widget, tip: str) -> None:
        c   = self.colors
        lbl = tk.Label(parent,
                       text="ⓘ", font=("Inter", 10),
                       bg=c["card"],
                       fg=c.get("text_secondary", "#6b7280"),
                       cursor="hand2")
        lbl.pack(side=tk.LEFT, padx=(2, 0))
        _Tooltip(lbl, tip, c)

    def _group(self, parent: tk.Widget, title: str) -> tk.Frame:
        c    = self.colors
        wrap = tk.Frame(parent, bg=c["card"],
                        highlightbackground=c["border"],
                        highlightthickness=1)
        wrap.pack(fill=tk.X, padx=10, pady=(0, 10))
        inner = tk.Frame(wrap, bg=c["card"])
        inner.pack(fill=tk.X, padx=12, pady=10)
        tk.Label(inner,
                 text=title,
                 font=("Inter", 10, "bold"),
                 bg=c["card"],
                 fg=c["accent"]).pack(anchor="w", pady=(0, 8))
        return inner

    def _divider(self, parent: tk.Widget,
                 pady: tuple = (0, 10)) -> None:
        tk.Frame(parent,
                 bg=self.colors["border"],
                 height=1).pack(fill=tk.X, padx=12, pady=pady)

    def _slider_row(self, parent, label: str, tooltip: str,
                    var: tk.IntVar, from_: int, to: int,
                    unit: str) -> None:
        c     = self.colors
        frame = tk.Frame(parent, bg=c["card"])
        frame.pack(fill=tk.X, pady=(0, 10))

        top = tk.Frame(frame, bg=c["card"])
        top.pack(fill=tk.X, pady=(0, 4))

        tk.Label(top,
                 text=label,
                 font=("Inter", 10),
                 bg=c["card"],
                 fg=c["text_secondary"]).pack(side=tk.LEFT)
        self._info(top, tooltip)

        val_lbl = tk.Label(top,
                           text=f"{var.get()}{unit}",
                           font=("Inter", 12, "bold"),
                           bg=c["card"],
                           fg=c["accent"],
                           width=7, anchor="e")
        val_lbl.pack(side=tk.RIGHT)
        self._val_labels[label] = val_lbl

        tk.Scale(
            frame,
            from_=from_, to=to, resolution=1,
            orient=tk.HORIZONTAL, variable=var,
            bg=c["card"], fg=c["text"],
            highlightbackground=c["card"],
            troughcolor="#2a3a5a",
            activebackground=c["accent"],
            sliderrelief="flat",
            sliderlength=18, width=8,
            showvalue=False,
            command=lambda v, u=unit, l=val_lbl:
                l.config(text=f"{int(float(v))}{u}"),
        ).pack(fill=tk.X)

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _on_start(self) -> None:
        try:
            self._save_settings()
            # Сразу блокируем кнопку чтобы не было двойного нажатия
            if hasattr(self, "start_btn"):
                self.start_btn.set_disabled(True)
            if hasattr(self, "stop_btn"):
                self.stop_btn.set_disabled(False)
            cb = self.callbacks.get("on_start")
            if cb:
                cb()
        except Exception as e:
            print(f"[Settings._on_start] {e}")
            # При ошибке — разблокируем
            if hasattr(self, "start_btn"):
                self.start_btn.set_disabled(False)
            if hasattr(self, "stop_btn"):
                self.stop_btn.set_disabled(True)

    def _on_clear_queue(self) -> None:
        """Очистить очередь видео и сбросить состояние."""
        if self.state.analysis_running:
            return
        try:
            self.state.video_queue.clear()
            self.state.batch_mode    = False
            self.state.matches       = []
            self.state.current_match = 0
            cb = self.callbacks.get("on_queue_cleared")
            if cb:
                cb()
        except Exception as e:
            print(f"[Settings._on_clear_queue] {e}")

    def _save_settings(self) -> None:
        """
        Сохраняет quality как нейтральный ключ.
        analysis_backend читает quality и сравнивает с QUALITY_FPS —
        нужно передавать локализованное значение для текущего языка.
        """
        try:
            cfg = _load_config()
            # Для backend передаём русское значение (QUALITY_FPS использует русские ключи)
            q_key      = self.quality.get()           # "fast"/"medium"/"maximum"
            q_for_backend = _t(q_key, "ru")           # "Быстро"/"Средне"/"Максимум"

            cfg.setdefault("analysis", {}).update({
                "threshold":          self.threshold.get(),
                "scene_interval":     self.scene_interval.get(),
                "min_gap":            self.match_gap.get(),
                "match_gap":          self.match_gap.get(),
                "quality":            q_key,           # сохраняем нейтральный ключ
                "scale_norm":         self.use_scale_invariance.get(),
                "use_mirror":         self.use_mirror_invariance.get(),
                "use_body_weights":   self.use_body_weights.get(),
                "batch_size":         self._batch_size.get(),
                "chunk_size":         self._chunk_size.get(),
                "max_unique_results": self._max_results.get(),
            })
            _save_config(cfg)
        except Exception as e:
            print(f"[Settings] save error: {e}")

    def get_quality_for_backend(self) -> str:
        """
        Возвращает значение quality в формате который понимает backend.
        QUALITY_FPS в analysis_backend использует русские ключи.
        """
        return _t(self.quality.get(), "ru")