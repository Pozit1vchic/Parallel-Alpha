#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ui/widgets/glow_button.py"""
from __future__ import annotations
import tkinter as tk


class GlowButton(tk.Canvas):
    """Закруглённая кнопка без бликов."""

    def __init__(
        self, parent,
        text:        str       = "",
        command                 = None,
        bg_color:    str | None = None,
        hover_color: str | None = None,
        color:       str        = "#1e293b",
        bg:          str        = "#0a0c10",
        width:       int        = 120,
        height:      int        = 32,
        font                    = None,
        radius:      int        = 12,
        fg:          str        = "#e2e8f0",
        **kwargs,
    ):
        for k in ("bg_override", "padx", "pady",
                  "relief", "bd", "anchor"):
            kwargs.pop(k, None)

        self._col    = bg_color or color
        self._hover  = hover_color or self._lighter(self._col, 0.18)
        self._press  = self._darker(self._col, 0.18)
        self._fg     = fg
        self._font   = font or ("Segoe UI", 9, "bold")
        self._text   = text
        self._cmd    = command
        self._radius = radius
        self._dis    = False
        self._w      = max(width, 40)
        self._h      = max(height, 22)
        self._cur_bg = self._col

        try:
            pbg = parent.cget("bg")
        except Exception:
            pbg = "#0a0c10"

        super().__init__(
            parent,
            width=self._w, height=self._h,
            bg=pbg,
            highlightthickness=0, bd=0,
            cursor="hand2",
            **kwargs)

        self._draw(self._col)

        self.bind("<Enter>",           self._on_enter)
        self.bind("<Leave>",           self._on_leave)
        self.bind("<Button-1>",        self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Configure>",       lambda _e: self._draw(self._cur_bg))

    # ── Draw ──────────────────────────────────────────────────────────────

    def _draw(self, fill: str) -> None:
        self.delete("all")
        w = self.winfo_width()  or self._w
        h = self.winfo_height() or self._h
        r = min(self._radius, h // 2, w // 2)

        # Основная кнопка — без тени и без блика
        self._rr(0, 0, w, h, r, fill)

        # Текст
        fg = "#444466" if self._dis else self._fg
        self.create_text(
            w // 2, h // 2,
            text=self._text,
            fill=fg,
            font=self._font,
            anchor="center")

    def _rr(self, x1, y1, x2, y2, r, color) -> None:
        r = max(0, min(r, (x2-x1)//2, (y2-y1)//2))
        self.create_arc(x1, y1, x1+r*2, y1+r*2,
                        start=90, extent=90,
                        fill=color, outline="")
        self.create_arc(x2-r*2, y1, x2, y1+r*2,
                        start=0, extent=90,
                        fill=color, outline="")
        self.create_arc(x1, y2-r*2, x1+r*2, y2,
                        start=180, extent=90,
                        fill=color, outline="")
        self.create_arc(x2-r*2, y2-r*2, x2, y2,
                        start=270, extent=90,
                        fill=color, outline="")
        self.create_rectangle(x1+r, y1, x2-r, y2,
                               fill=color, outline="")
        self.create_rectangle(x1, y1+r, x2, y2-r,
                               fill=color, outline="")

    # ── Events ────────────────────────────────────────────────────────────

    def _on_enter(self, _=None):
        if not self._dis:
            self._cur_bg = self._hover
            self._draw(self._hover)

    def _on_leave(self, _=None):
        if not self._dis:
            self._cur_bg = self._col
            self._draw(self._col)

    def _on_press(self, _=None):
        if not self._dis:
            self._cur_bg = self._press
            self._draw(self._press)

    def _on_release(self, _=None):
        if self._dis:
            return
        self._cur_bg = self._hover
        self._draw(self._hover)
        if self._cmd:
            try:
                self._cmd()
            except Exception as e:
                print(f"[GlowButton] {e}")

    # ── Public ────────────────────────────────────────────────────────────

    def set_bg(self, color: str) -> None:
        """Сменить фоновый цвет кнопки."""
        self._col    = color
        self._hover  = self._lighter(color, 0.18)
        self._press  = self._darker(color, 0.18)
        self._cur_bg = color
        self._draw(color)

    def set_disabled(self, d: bool) -> None:
        self._dis = d
        self.config(cursor="arrow" if d else "hand2")
        col = self._darker(self._col, 0.45) if d else self._col
        self._cur_bg = col
        self._draw(col)

    def set_text(self, text: str) -> None:
        self._text = text
        self._draw(self._cur_bg)

    def configure_text(self, text: str) -> None:
        self.set_text(text)

    def config(self, **kw):
        redraw = False
        if "state" in kw:
            self.set_disabled(kw.pop("state") in ("disabled", tk.DISABLED))
        if "text" in kw:
            self._text = kw.pop("text")
            redraw = True
        if "bg_color" in kw:
            self._col   = kw.pop("bg_color")
            self._hover = self._lighter(self._col, 0.18)
            self._press = self._darker(self._col, 0.18)
            redraw = True
        if "hover_color" in kw:
            self._hover = kw.pop("hover_color")
        if "fg" in kw:
            self._fg = kw.pop("fg")
            redraw = True
        if kw:
            try:
                super().config(**kw)
            except Exception:
                pass
        if redraw:
            self._draw(self._cur_bg)

    configure = config

    # ── Colors ────────────────────────────────────────────────────────────

    @staticmethod
    def _h2r(c: str):
        c = c.lstrip("#")
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

    @staticmethod
    def _r2h(r, g, b) -> str:
        return (f"#{max(0,min(255,int(r))):02x}"
                f"{max(0,min(255,int(g))):02x}"
                f"{max(0,min(255,int(b))):02x}")

    def _lighter(self, c: str, f: float) -> str:
        r, g, b = self._h2r(c)
        return self._r2h(r+(255-r)*f, g+(255-g)*f, b+(255-b)*f)

    def _darker(self, c: str, f: float) -> str:
        r, g, b = self._h2r(c)
        return self._r2h(r*(1-f), g*(1-f), b*(1-f))