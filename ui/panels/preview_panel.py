# ui/panels/preview_panel.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable

import numpy as np
import tkinter as tk

from ui.app_state import AppState
from ui.widgets.progress_bar import AnimatedProgressbar
from ui.widgets.glow_button import GlowButton
from ui.widgets.smooth_scroll import SmoothScrollMixin


def _fmt_hms(secs: float) -> str:
    s  = max(0.0, float(secs))
    h  = int(s // 3600)
    m  = int((s % 3600) // 60)
    ss = int(s % 60)
    return f"{h:02d}:{m:02d}:{ss:02d}"


_DIR_LABELS: dict[str, dict[str, str]] = {
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

_UI: dict[str, dict[str, str]] = {
    "progress":    {"ru": "Прогресс",             "en": "Progress"},
    "waiting":     {"ru": "Ожидание",             "en": "Waiting"},
    "loading_photo": {"ru": "Загрузка фото-референса...",
                       "en": "Loading photo reference..."},
    "comparison":  {"ru": "Сравнение",            "en": "Comparison"},
    "timeline":    {"ru": "Таймлайн",             "en": "Timeline"},
    "done_found":  {"ru": "Готово. Найдено {} повторов",
                    "en": "Done. Found {} repeats"},
}


def _dir_label(direction: str, lang: str) -> str:
    return _DIR_LABELS.get(
        direction, _DIR_LABELS["unknown"]
    ).get(lang, direction)


class PreviewPanel(tk.Frame, SmoothScrollMixin):
    """Центральная панель — прогресс + сравнение кадров + таймлайн."""

    def __init__(self, parent, state: AppState,
                 colors: dict, callbacks: dict,
                 lang: str = "ru") -> None:
        super().__init__(parent, bg=colors["bg"])
        self.state     = state
        self.colors    = colors
        self.callbacks = callbacks
        self._lang     = lang

        self._preview_zoom: float = 1.0
        self._img1_orig = None
        self._img2_orig = None
        self._cached_durations: list[float] = []

        self._build()

    # ── Язык ──────────────────────────────────────────────────────────────

    def set_lang(self, lang: str) -> None:
        if self._lang == lang:
            return
        self._lang = lang
        self._update_lang_labels()

    def _update_lang_labels(self) -> None:
        lang = self._lang
        try:
            self._progress_title_lbl.config(
                text=_UI["progress"][lang])
        except Exception:
            pass
        try:
            cur = self.status_label.cget("text")
            # Обновляем только если это дефолтный текст
            waiting_texts = (
                _UI["waiting"]["ru"],
                _UI["waiting"]["en"],
            )
            if cur in waiting_texts:
                self.status_label.config(
                    text=_UI["waiting"][lang])
        except Exception:
            pass
        try:
            self._comparison_title_lbl.config(
                text=_UI["comparison"][lang])
        except Exception:
            pass
        try:
            self._timeline_title_lbl.config(
                text=_UI["timeline"][lang])
        except Exception:
            pass

    # ── Построение ────────────────────────────────────────────────────────

    def _build(self) -> None:
        self._build_progress()
        self._build_comparison()

    def _build_progress(self) -> None:
        c         = self.colors
        lang      = self._lang
        prog_card = tk.Frame(self, bg=c["card"])
        prog_card.pack(fill=tk.X, pady=(0, 10))
        p_in = tk.Frame(prog_card, bg=c["card"], padx=14, pady=12)
        p_in.pack(fill=tk.X)

        hdr = tk.Frame(p_in, bg=c["card"])
        hdr.pack(fill=tk.X, pady=(0, 6))

        self._progress_title_lbl = tk.Label(
            hdr,
            text=_UI["progress"][lang],
            font=("Inter", 12, "bold"),
            bg=c["card"], fg=c["text"])
        self._progress_title_lbl.pack(side=tk.LEFT)

        self.progress_label = tk.Label(
            hdr, text="0%",
            font=("Inter", 13, "bold"),
            bg=c["card"], fg=c["accent"],
            width=6, anchor="e")
        self.progress_label.pack(side=tk.RIGHT)

        self.progress = AnimatedProgressbar(p_in)
        self.progress.pack(fill=tk.X, pady=(6, 3))

        self.status_label = tk.Label(
            p_in,
            text=_UI["waiting"][lang],
            font=("Inter", 10),
            bg=c["card"], fg=c["text_secondary"])
        self.status_label.pack(anchor="w")

    def _build_comparison(self) -> None:
        c    = self.colors
        lang = self._lang

        pv_card = tk.Frame(self, bg=c["card"])
        pv_card.pack(fill=tk.BOTH, expand=True)
        pv_in = tk.Frame(pv_card, bg=c["card"], padx=14, pady=12)
        pv_in.pack(fill=tk.BOTH, expand=True)

        title_row = tk.Frame(pv_in, bg=c["card"])
        title_row.pack(fill=tk.X, pady=(0, 8))

        self._comparison_title_lbl = tk.Label(
            title_row,
            text=_UI["comparison"][lang],
            font=("Inter", 12, "bold"),
            bg=c["card"], fg=c["text"])
        self._comparison_title_lbl.pack(side=tk.LEFT)

        self.match_info = tk.Label(
            title_row, text="",
            font=("Inter", 10),
            bg=c["card"], fg=c["text_secondary"])
        self.match_info.pack(side=tk.RIGHT)

        video_row = tk.Frame(pv_in, bg=c["bg"])
        video_row.pack(fill=tk.BOTH, expand=True)
        video_row.bind("<Configure>", self._on_resize)
        self._video_row = video_row

        (self.preview1,
         self.time1_label,
         self.action1_label) = self._make_video_widget(video_row)
        (self.preview2,
         self.time2_label,
         self.action2_label) = self._make_video_widget(video_row)

        for w in (self.preview1, self.preview2,
                  self.time1_label, self.time2_label):
            w.bind("<MouseWheel>",  self._on_preview_scroll)
            w.bind("<Button-4>",
                   lambda e: self._on_preview_scroll(e, delta=120))
            w.bind("<Button-5>",
                   lambda e: self._on_preview_scroll(e, delta=-120))

        self._build_timeline(pv_in)

    def _make_video_widget(self, parent):
        c    = self.colors
        side = (tk.LEFT
                if not parent.winfo_children()
                else tk.RIGHT)
        vf   = tk.Frame(parent, bg=c["bg"])
        vf.pack(side=side, padx=6, expand=True, fill=tk.BOTH)

        cont = tk.Frame(vf, bg=c["bg"],
                        highlightbackground=c["border"],
                        highlightthickness=1)
        cont.pack(fill=tk.BOTH, expand=True)

        prev = tk.Label(cont, bg=c["bg"])
        prev.pack(fill=tk.BOTH, expand=True)

        time_lbl = tk.Label(
            cont, text="--:--:--",
            font=("Inter", 14, "bold"),
            bg="#0d1117", fg="white",
            padx=8, pady=3)
        time_lbl.place(relx=0.02, rely=0.02)

        act_lbl = tk.Label(
            cont, text="",
            font=("Inter", 9),
            bg="#0d1117", fg=c["accent"],
            padx=6, pady=2)
        act_lbl.place(relx=0.02, rely=0.16)

        return prev, time_lbl, act_lbl

    def _build_timeline(self, parent: tk.Widget) -> None:
        c    = self.colors
        lang = self._lang
        tl   = tk.Frame(parent, bg=c["card"], pady=4)
        tl.pack(fill=tk.X, pady=(10, 0))

        ctrl = tk.Frame(tl, bg=c["card"])
        ctrl.pack(fill=tk.X, pady=(0, 4))

        self._timeline_title_lbl = tk.Label(
            ctrl,
            text=_UI["timeline"][lang],
            font=("Inter", 10),
            bg=c["card"], fg=c["text_secondary"])
        self._timeline_title_lbl.pack(side=tk.LEFT)

        for sym, cmd in [
            ("↺", self._zoom_reset),
            ("−", self._zoom_out),
            ("+", self._zoom_in),
        ]:
            GlowButton(
                ctrl, sym, cmd,
                c["highlight"], c["border"],
                width=28, height=24,
                font=("Inter", 12, "bold")).pack(
                side=tk.RIGHT, padx=2)

        tl_wrap = tk.Frame(tl, bg=c["bg"],
                           highlightbackground=c["accent"],
                           highlightthickness=1)
        tl_wrap.pack(fill=tk.X, expand=True)

        self.timeline_canvas = tk.Canvas(
            tl_wrap, height=54, bg=c["bg"],
            highlightthickness=0)
        self.timeline_canvas.pack(fill=tk.X, expand=True)
        self.timeline_canvas.bind(
            "<Button-1>",  self._on_tl_click)
        self.timeline_canvas.bind(
            "<B1-Motion>", self._on_tl_drag)
        self._bind_smooth_scroll(
            self.timeline_canvas, self._on_tl_scroll)
        self._tl_drag_x: int = 0

    # ── Прогресс ──────────────────────────────────────────────────────────

    def set_progress(self, pct: float,
                     status: str = "") -> None:
        try:
            self.progress.set(pct)
            self.progress_label.config(text=f"{pct:.0f}%")
            if status:
                self.status_label.config(text=status)
        except Exception:
            pass

    def set_status(self, text: str) -> None:
        try:
            self.status_label.config(text=text)
        except Exception:
            pass

    def set_done_status(self, count: int) -> None:
        """Финальная строка «Готово. Найдено N повторов»."""
        lang = self._lang
        text = _UI["done_found"][lang].format(count)
        self.set_status(text)

    def set_photo_loading_status(self) -> None:
        """Статус пока грузится фото-референс."""
        lang = self._lang
        self.set_status(_UI["loading_photo"][lang])

    def start_animation(self) -> None:
        try:
            self.progress.start_animation()
        except Exception:
            pass

    def stop_animation(self) -> None:
        try:
            self.progress.stop_animation()
        except Exception:
            pass

    # ── Превью ────────────────────────────────────────────────────────────

    def update_preview(self, match: dict,
                       img1, img2,
                       index: int,
                       total: int) -> None:
        try:
            self.time1_label.config(
                text=_fmt_hms(match["t1"]))
            self.time2_label.config(
                text=_fmt_hms(match["t2"]))

            direction = match.get("direction", "forward")
            dir_text  = _dir_label(direction, self._lang)
            movement  = match.get("movement", "")
            display   = (f"{movement}  •  {dir_text}"
                         if movement else dir_text)

            sim = match.get("sim", 0.0)
            self.match_info.config(
                text=f"{dir_text}  •  {sim * 100:.1f}%")
            self.action1_label.config(text=display)
            self.action2_label.config(text=display)

            self._img1_orig = img1
            self._img2_orig = img2
            self._apply_zoom()

            if self._cached_durations:
                self.after(10, lambda: self.draw_heatmap(
                    self._cached_durations))
        except Exception as e:
            print(f"[PreviewPanel.update_preview] {e}")

    def _apply_zoom(self) -> None:
        if self._img1_orig:
            self.preview1.config(image=self._img1_orig)
            self.preview1.image = self._img1_orig
        if self._img2_orig:
            self.preview2.config(image=self._img2_orig)
            self.preview2.image = self._img2_orig

    def _on_preview_scroll(self, event,
                            delta: int | None = None) -> None:
        d = delta if delta is not None else event.delta
        if d > 0:
            self._preview_zoom = min(
                3.0, self._preview_zoom * 1.15)
        else:
            self._preview_zoom = max(
                0.3, self._preview_zoom / 1.15)
        if "on_resize" in self.callbacks:
            self.callbacks["on_resize"]()

    def get_preview_size(self) -> tuple[int, int]:
        w = max(100,
                int(self._video_row.winfo_width() // 2
                    * self._preview_zoom) - 20)
        h = max(60,
                int(self._video_row.winfo_height()
                    * self._preview_zoom) - 30)
        return w, h

    def _on_resize(self, _event: tk.Event) -> None:
        if "on_resize" in self.callbacks:
            self.callbacks["on_resize"]()
        if self._cached_durations:
            self.after(20, lambda: self.draw_heatmap(
                self._cached_durations))

    # ── Таймлайн ──────────────────────────────────────────────────────────

    def _zoom_in(self) -> None:
        self.state.timeline_zoom = min(
            20.0, self.state.timeline_zoom * 1.5)
        self.draw_heatmap(self._cached_durations)

    def _zoom_out(self) -> None:
        self.state.timeline_zoom = max(
            1.0, self.state.timeline_zoom / 1.5)
        self.draw_heatmap(self._cached_durations)

    def _zoom_reset(self) -> None:
        self.state.timeline_zoom = 1.0
        self.state.timeline_pan  = 0.0
        self.draw_heatmap(self._cached_durations)

    def _on_tl_scroll(self, n: int,
                       unit: str = "units") -> None:
        z = self.state.timeline_zoom
        self.state.timeline_pan = max(
            0.0,
            min(1.0 - 1.0 / z,
                self.state.timeline_pan + n * 0.02))
        self.draw_heatmap(self._cached_durations)

    def _on_tl_drag(self, event: tk.Event) -> None:
        dx   = event.x - self._tl_drag_x
        w    = self.timeline_canvas.winfo_width() or 1
        frac = dx / w / self.state.timeline_zoom
        self.state.timeline_pan = max(
            0.0,
            min(1.0 - 1.0 / self.state.timeline_zoom,
                self.state.timeline_pan - frac))
        self._tl_drag_x = event.x
        self.draw_heatmap(self._cached_durations)

    def _on_tl_click(self, event: tk.Event) -> None:
        self._tl_drag_x = event.x
        if "on_timeline_click" in self.callbacks:
            self.callbacks["on_timeline_click"](event)

    def draw_heatmap(self,
                     durations: list[float] | None = None
                     ) -> None:
        if durations is not None:
            self._cached_durations = list(durations)
        else:
            durations = self._cached_durations

        self.timeline_canvas.delete("all")
        if not durations:
            return

        self.timeline_canvas.update_idletasks()
        cw        = self.timeline_canvas.winfo_width() or 800
        ch        = 54
        total_dur = sum(durations)
        if total_dur <= 0:
            return

        zoom       = self.state.timeline_zoom
        pan        = self.state.timeline_pan
        view_start = pan * total_dur
        view_span  = total_dur / zoom

        def _t2x(t: float) -> float:
            return ((t - view_start) / view_span) * cw

        def _fmt(s: float) -> str:
            s  = max(0.0, s)
            h  = int(s // 3600)
            m  = int((s % 3600) // 60)
            ss = int(s % 60)
            return (f"{h}:{m:02d}:{ss:02d}" if h > 0
                    else f"{m}:{ss:02d}")

        c = self.colors
        self.timeline_canvas.create_rectangle(
            0, 0, cw, ch,
            fill=c["bg"], outline="")

        nice_steps = [1, 2, 5, 10, 15, 30, 60,
                      120, 300, 600, 1800, 3600]
        px_per_sec = cw / max(view_span, 1e-6)
        step       = nice_steps[-1]
        for s in nice_steps:
            if s * px_per_sec >= 80:
                step = s
                break

        t = int(view_start / step) * step
        while t <= view_start + view_span:
            x = _t2x(t)
            if 0 <= x <= cw:
                self.timeline_canvas.create_line(
                    x, 0, x, ch,
                    fill=c.get("border", "#2a2f38"),
                    width=1, dash=(2, 4))
                self.timeline_canvas.create_text(
                    x + 3, 4, text=_fmt(t),
                    anchor="nw",
                    fill=c.get("text_secondary", "#6b7280"),
                    font=("Inter", 7))
            t += step

        cumulative = 0.0
        for i, dur in enumerate(durations):
            if i > 0:
                x = _t2x(cumulative)
                if 0 <= x <= cw:
                    self.timeline_canvas.create_line(
                        x, 0, x, ch,
                        fill=c.get("accent", "#3b82f6"),
                        width=1, dash=(4, 2))
                    self.timeline_canvas.create_text(
                        x + 4, ch - 10,
                        text=f"V{i+1}", anchor="sw",
                        fill=c.get("accent", "#3b82f6"),
                        font=("Inter", 7))
            cumulative += dur

        if not self.state.matches:
            return

        starts = np.cumsum([0.0] + list(durations[:-1]))
        if self.state.current_match >= len(self.state.matches):
            return
        current_m = self.state.matches[
            self.state.current_match]

        for m in self.state.matches:
            for vi, ti in [(m["v1_idx"], m["t1"]),
                           (m["v2_idx"], m["t2"])]:
                if vi >= len(starts):
                    continue
                abs_t = float(starts[vi]) + ti
                if (abs_t < view_start
                        or abs_t > view_start + view_span):
                    continue
                x   = _t2x(abs_t)
                sim = m.get("sim", 0.0)
                col = (c.get("success", "#22c55e")
                       if sim >= 0.85
                       else c.get("accent", "#3b82f6")
                       if sim >= 0.70
                       else c.get("error", "#ef4444"))
                self.timeline_canvas.create_line(
                    x, 14, x, ch - 2,
                    fill=col, width=2)
                self.timeline_canvas.create_polygon(
                    x-4, 14, x+4, 14, x, 20,
                    fill=col, outline="")

        for vi, ti in [
            (current_m["v1_idx"], current_m["t1"]),
            (current_m["v2_idx"], current_m["t2"]),
        ]:
            if vi >= len(starts):
                continue
            abs_t = float(starts[vi]) + ti
            if (abs_t < view_start
                    or abs_t > view_start + view_span):
                continue
            xc = _t2x(abs_t)
            self.timeline_canvas.create_line(
                xc, 0, xc, ch,
                fill="white", width=2)
            self.timeline_canvas.create_polygon(
                xc-5, 0, xc+5, 0, xc, 8,
                fill="white", outline="")
            self.timeline_canvas.create_text(
                xc, ch - 2, text=_fmt(ti),
                anchor="s", fill="white",
                font=("Inter", 7, "bold"))