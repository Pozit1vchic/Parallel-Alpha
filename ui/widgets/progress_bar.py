#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ui/widgets/progress_bar.py
Алиасы для совместимости — сама логика в preview_panel._ProgressBar
"""
from __future__ import annotations
import tkinter as tk


class SmoothProgressBar(tk.Canvas):
    """Плавный прогресс-бар — используется вне PreviewPanel."""

    def __init__(
        self, parent,
        height: int = 8,
        radius: int = 4,
        color: str  = "#3b82f6",
        bg: str     = "#14171c",
        **kwargs,
    ):
        super().__init__(
            parent,
            height=height + 10,
            bg=bg,
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self._h     = height
        self._r     = radius
        self._color = color
        self._bg    = bg
        self._cur   = 0.0
        self._tgt   = 0.0
        self._anim  = False
        self.bind("<Configure>", self._draw)

    def set(self, value: float) -> None:
        self._tgt = max(0.0, min(1.0, value / 100.0))
        if not self._anim:
            self._anim = True
            self._step()

    def reset(self) -> None:
        self._tgt = self._cur = 0.0
        self._anim = False
        self._draw()

    def start_animation(self) -> None:
        pass  # stub

    def stop_animation(self) -> None:
        pass  # stub

    def _step(self) -> None:
        diff = self._tgt - self._cur
        if abs(diff) < 0.003:
            self._cur = self._tgt
            self._anim = False
            self._draw()
            return
        self._cur += diff * 0.10
        self._draw()
        try:
            self.after(16, self._step)
        except tk.TclError:
            pass

    def _draw(self, _=None) -> None:
        try:
            self.delete("all")
            w = self.winfo_width()
            h = self.winfo_height()
            if w < 4 or h < 4:
                return
            bar_h = self._h
            cy    = h // 2
            ty1   = cy - bar_h // 2
            ty2   = cy + bar_h // 2
            r     = min(self._r, bar_h // 2)
            self._rr(0, ty1, w, ty2, r, "#1e293b")
            fw = max(bar_h, int(w * self._cur))
            if self._cur > 0.005:
                for i in range(3, 0, -1):
                    ex = i * 2
                    ga = 0.04 * i
                    gc = self._blend(self._color, self._bg, ga)
                    self._rr(-ex, ty1-ex, fw+ex, ty2+ex, r+ex, gc)
                self._rr(0, ty1, fw, ty2, r, self._color)
        except tk.TclError:
            pass

    def _rr(self, x1, y1, x2, y2, r, color):
        r = max(0, min(r, (x2-x1)/2, (y2-y1)/2))
        pts = [
            x1+r,y1, x2-r,y1, x2,y1, x2,y1+r,
            x2,y2-r, x2,y2, x2-r,y2, x1+r,y2,
            x1,y2, x1,y2-r, x1,y1+r, x1,y1,
        ]
        self.create_polygon(pts, fill=color, outline="", smooth=True)

    @staticmethod
    def _h2r(c):
        c = c.lstrip("#")
        return int(c[0:2],16), int(c[2:4],16), int(c[4:6],16)

    @staticmethod
    def _r2h(r,g,b):
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

    def _blend(self, c1, c2, a):
        r1,g1,b1 = self._h2r(c1)
        r2,g2,b2 = self._h2r(c2)
        return self._r2h(
            r1*a+r2*(1-a), g1*a+g2*(1-a), b1*a+b2*(1-a)
        )


# Алиас для совместимости со старым кодом
AnimatedProgressbar = SmoothProgressBar