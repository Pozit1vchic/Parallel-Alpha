#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ui/widgets/custom_slider.py — Красивый ползунок с глоу
"""
from __future__ import annotations
import tkinter as tk


class CustomSlider(tk.Canvas):
    """
    Кастомный ползунок:
    - полностью вписывается в свою рамку
    - глоу при drag/hover
    - скруглённый трек и заполнение
    - красивый thumb
    """

    def __init__(
        self,
        parent,
        from_: float = 0,
        to: float = 100,
        value: float = 50,
        command=None,
        track_color:  str = "#1e293b",
        fill_color:   str = "#3B82F6",
        thumb_color:  str = "#60A5FA",
        height: int = 24,
        track_height: int = 4,
        thumb_r: int = 7,
        **kwargs,
    ):
        bg = kwargs.pop("bg", "#0f0f1a")
        super().__init__(
            parent,
            height=height,
            bg=bg,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
            **kwargs,
        )
        self._from        = from_
        self._to          = to
        self._value       = float(value)
        self._command     = command
        self._track_color = track_color
        self._fill_color  = fill_color
        self._thumb_color = thumb_color
        self._height      = height
        self._track_h     = track_height
        self._thumb_r     = thumb_r
        self._bg          = bg

        self._dragging    = False
        self._hover_thumb = False
        self._glow_alpha  = 0.0
        self._animating   = False

        self.bind("<Configure>",      self._draw)
        self.bind("<Button-1>",       self._on_press)
        self.bind("<B1-Motion>",      self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>",          self._on_enter)
        self.bind("<Leave>",          self._on_leave)
        self.bind("<MouseWheel>",     self._on_wheel)

    # ── Public API ────────────────────────────────────────────────────────

    def get(self) -> float:
        return self._value

    def set(self, value: float) -> None:
        self._value = max(self._from, min(self._to, float(value)))
        self._draw()

    def configure(self, **kwargs):
        if "state" in kwargs:
            state = kwargs.pop("state")
            self.config(
                cursor="arrow" if state == "disabled" else "hand2"
            )
        super().configure(**kwargs)

    # ── Events ────────────────────────────────────────────────────────────

    def _on_enter(self, _=None):
        self._hover_thumb = True
        if not self._animating:
            self._animating = True
            self._animate_glow()

    def _on_leave(self, _=None):
        self._hover_thumb = False
        self._dragging = False

    def _on_press(self, event):
        self._dragging = True
        self._update_from_x(event.x)

    def _on_drag(self, event):
        if self._dragging:
            self._update_from_x(event.x)

    def _on_release(self, _=None):
        self._dragging = False

    def _on_wheel(self, event):
        step = (self._to - self._from) / 100
        if event.delta > 0:
            self._value = min(self._to, self._value + step * 3)
        else:
            self._value = max(self._from, self._value - step * 3)
        self._draw()
        if self._command:
            self._command(self._value)

    def _update_from_x(self, x: int) -> None:
        w      = self.winfo_width()
        pad    = self._thumb_r + 2
        usable = w - pad * 2
        if usable <= 0:
            return
        ratio       = max(0.0, min(1.0, (x - pad) / usable))
        self._value = self._from + ratio * (self._to - self._from)
        self._draw()
        if self._command:
            self._command(self._value)

    # ── Animation ─────────────────────────────────────────────────────────

    def _animate_glow(self) -> None:
        target = 1.0 if (self._hover_thumb or self._dragging) else 0.0
        diff   = target - self._glow_alpha
        if abs(diff) < 0.02:
            self._glow_alpha = target
            self._animating  = False
            self._draw()
            return
        self._glow_alpha += diff * 0.18
        self._draw()
        try:
            self.after(16, self._animate_glow)
        except tk.TclError:
            pass

    # ── Drawing ───────────────────────────────────────────────────────────

    def _draw(self, event=None) -> None:
        try:
            self.delete("all")
            w  = self.winfo_width()
            h  = self.winfo_height()
            if w < 4 or h < 4:
                return

            pad    = self._thumb_r + 2
            cy     = h // 2
            track_y1 = cy - self._track_h // 2
            track_y2 = cy + self._track_h // 2
            r_track  = self._track_h // 2

            ratio  = (self._value - self._from) / max(
                0.001, self._to - self._from
            )
            thumb_x = pad + ratio * (w - pad * 2)

            # ── Трек (фон) ────────────────────────────────────────────────
            self._rounded_rect(
                pad, track_y1, w - pad, track_y2,
                r_track, self._track_color,
            )

            # ── Заполненная часть ─────────────────────────────────────────
            if thumb_x > pad + 1:
                self._rounded_rect(
                    pad, track_y1, thumb_x, track_y2,
                    r_track, self._fill_color,
                )

            # ── Thumb glоу ────────────────────────────────────────────────
            if self._glow_alpha > 0.01:
                for i in range(3, 0, -1):
                    gr   = self._thumb_r + i * 3
                    ga   = self._glow_alpha * 0.06 * i
                    gcol = self._blend(
                        self._fill_color, self._bg, ga
                    )
                    self.create_oval(
                        thumb_x - gr, cy - gr,
                        thumb_x + gr, cy + gr,
                        fill=gcol, outline="",
                    )

            # ── Thumb (внешний круг) ──────────────────────────────────────
            r = self._thumb_r
            self.create_oval(
                thumb_x - r, cy - r,
                thumb_x + r, cy + r,
                fill=self._thumb_color,
                outline=self._fill_color,
                width=2,
            )

            # ── Thumb (внутренний блик) ───────────────────────────────────
            inner_r = max(2, r - 3)
            self.create_oval(
                thumb_x - inner_r, cy - inner_r,
                thumb_x + inner_r // 2, cy,
                fill=self._lighten(self._thumb_color, 0.3),
                outline="",
            )

        except tk.TclError:
            pass

    def _rounded_rect(self, x1, y1, x2, y2, r, color) -> None:
        r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
        pts = [
            x1 + r, y1,  x2 - r, y1,
            x2, y1,      x2, y1 + r,
            x2, y2 - r,  x2, y2,
            x2 - r, y2,  x1 + r, y2,
            x1, y2,      x1, y2 - r,
            x1, y1 + r,  x1, y1,
        ]
        self.create_polygon(pts, fill=color, outline="", smooth=True)

    @staticmethod
    def _hex_to_rgb(c):
        c = c.lstrip("#")
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

    @staticmethod
    def _rgb_to_hex(r, g, b):
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

    def _blend(self, c1, c2, a):
        r1, g1, b1 = self._hex_to_rgb(c1)
        r2, g2, b2 = self._hex_to_rgb(c2)
        return self._rgb_to_hex(
            r1 * a + r2 * (1 - a),
            g1 * a + g2 * (1 - a),
            b1 * a + b2 * (1 - a),
        )

    def _lighten(self, c, f):
        r, g, b = self._hex_to_rgb(c)
        return self._rgb_to_hex(
            min(255, r + (255 - r) * f),
            min(255, g + (255 - g) * f),
            min(255, b + (255 - b) * f),
        )