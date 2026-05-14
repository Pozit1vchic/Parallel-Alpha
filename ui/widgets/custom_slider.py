from __future__ import annotations

import tkinter as tk


class CustomSlider(tk.Canvas):

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
        height: int = 28,
        track_height: int = 5,
        thumb_r: int = 9,
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
        self._from        = float(from_)
        self._to          = float(to)
        self._value       = float(max(from_, min(to, value)))
        self._command     = command
        self._track_color = track_color
        self._fill_color  = fill_color
        self._thumb_color = thumb_color
        self._height      = height
        self._track_h     = track_height
        self._thumb_r     = thumb_r
        self._bg          = bg

        self._dragging    = False
        self._hover       = False
        self._glow_alpha  = 0.0
        self._animating   = False
        self._disabled    = False

        self._precomputed: dict = {}
        self._last_w: int = 0
        self._last_h: int = 0

        self.bind("<Configure>",       self._on_configure)
        self.bind("<Button-1>",        self._on_press)
        self.bind("<B1-Motion>",       self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>",           self._on_enter)
        self.bind("<Leave>",           self._on_leave)
        self.bind("<MouseWheel>",      self._on_wheel)
        self.bind("<Button-4>",        lambda _: self._step(-1))
        self.bind("<Button-5>",        lambda _: self._step(1))

    def get(self) -> float:
        return self._value

    def set(self, value: float) -> None:
        clamped = max(self._from, min(self._to, float(value)))
        if clamped == self._value:
            return
        self._value = clamped
        self._draw()

    def configure(self, **kwargs) -> None:
        if "state" in kwargs:
            state = kwargs.pop("state")
            self._disabled = state == "disabled"
            self.config(cursor="arrow" if self._disabled else "hand2")
        if "command" in kwargs:
            self._command = kwargs.pop("command")
        if kwargs:
            super().configure(**kwargs)

    def _on_configure(self, event: tk.Event) -> None:
        w, h = event.width, event.height
        if w != self._last_w or h != self._last_h:
            self._last_w = w
            self._last_h = h
            self._precompute(w, h)
            self._draw()

    def _precompute(self, w: int, h: int) -> None:
        pad      = self._thumb_r + 4
        cy       = h // 2
        usable   = max(1, w - pad * 2)
        track_y1 = cy - self._track_h // 2
        track_y2 = cy + self._track_h // 2
        r_track  = self._track_h // 2

        self._precomputed = {
            "pad":      pad,
            "cy":       cy,
            "usable":   usable,
            "track_y1": track_y1,
            "track_y2": track_y2,
            "r_track":  r_track,
            "w":        w,
            "h":        h,
        }

    def _on_enter(self, _=None) -> None:
        if self._disabled:
            return
        self._hover = True
        self._start_glow()

    def _on_leave(self, _=None) -> None:
        self._hover    = False
        self._dragging = False
        self._start_glow()

    def _on_press(self, event: tk.Event) -> None:
        if self._disabled:
            return
        self._dragging = True
        self._update_from_x(event.x)

    def _on_drag(self, event: tk.Event) -> None:
        if self._dragging and not self._disabled:
            self._update_from_x(event.x)

    def _on_release(self, _=None) -> None:
        self._dragging = False

    def _on_wheel(self, event: tk.Event) -> None:
        if self._disabled:
            return
        self._step(-1 if event.delta > 0 else 1)

    def _step(self, direction: int) -> None:
        step = (self._to - self._from) / 100
        self._value = max(
            self._from,
            min(self._to, self._value + direction * step * 3)
        )
        self._draw()
        self._fire()

    def _update_from_x(self, x: int) -> None:
        p = self._precomputed
        if not p:
            return
        pad    = p["pad"]
        usable = p["usable"]
        ratio  = max(0.0, min(1.0, (x - pad) / usable))
        self._value = self._from + ratio * (self._to - self._from)
        self._draw()
        self._fire()

    def _fire(self) -> None:
        if self._command:
            try:
                self._command(self._value)
            except Exception:
                pass

    def _start_glow(self) -> None:
        if not self._animating:
            self._animating = True
            self._tick_glow()

    def _tick_glow(self) -> None:
        target = 1.0 if (self._hover or self._dragging) else 0.0
        diff   = target - self._glow_alpha

        if abs(diff) < 0.015:
            self._glow_alpha = target
            self._animating  = False
            self._draw()
            return

        self._glow_alpha += diff * 0.20
        self._draw()

        try:
            self.after(14, self._tick_glow)
        except tk.TclError:
            pass

    def _draw(self, _=None) -> None:
        try:
            p = self._precomputed
            if not p:
                w = self.winfo_width()
                h = self.winfo_height()
                if w < 4 or h < 4:
                    return
                self._precompute(w, h)
                p = self._precomputed

            w        = p["w"]
            pad      = p["pad"]
            cy       = p["cy"]
            usable   = p["usable"]
            track_y1 = p["track_y1"]
            track_y2 = p["track_y2"]
            r_track  = p["r_track"]

            ratio   = (self._value - self._from) / max(0.001, self._to - self._from)
            thumb_x = pad + ratio * usable

            alpha_mul = 0.55 if self._disabled else 1.0

            self.delete("all")

            self._rounded_rect(
                pad, track_y1, w - pad, track_y2,
                r_track, self._track_color,
            )

            if thumb_x > pad + 1:
                fill_col = self._desaturate(self._fill_color, alpha_mul)
                self._rounded_rect(
                    pad, track_y1, thumb_x, track_y2,
                    r_track, fill_col,
                )

                segment_h = (track_y2 - track_y1) * 0.4
                shine_y1  = track_y1 + 1
                shine_y2  = track_y1 + max(1, int(segment_h))
                shine_x2  = min(thumb_x - 2, w - pad)
                if shine_x2 > pad + 2:
                    shine_col = self._lighten(fill_col, 0.25)
                    self._rounded_rect(
                        pad + 2, shine_y1, shine_x2, shine_y2,
                        r_track // 2, shine_col,
                    )

            if self._glow_alpha > 0.01 and not self._disabled:
                glow_col = self._fill_color
                for i in range(4, 0, -1):
                    gr  = self._thumb_r + i * 4
                    ga  = self._glow_alpha * 0.055 * i
                    col = self._blend(glow_col, self._bg, ga)
                    self.create_oval(
                        thumb_x - gr, cy - gr,
                        thumb_x + gr, cy + gr,
                        fill=col, outline="",
                    )

            r = self._thumb_r

            shadow_col = self._darken(self._thumb_color, 0.35)
            self.create_oval(
                thumb_x - r + 1, cy - r + 2,
                thumb_x + r + 1, cy + r + 2,
                fill=shadow_col, outline="",
            )

            thumb_col = (
                self._desaturate(self._thumb_color, 0.55)
                if self._disabled
                else self._thumb_color
            )
            border_col = (
                self._desaturate(self._fill_color, 0.55)
                if self._disabled
                else self._fill_color
            )

            self.create_oval(
                thumb_x - r, cy - r,
                thumb_x + r, cy + r,
                fill=thumb_col,
                outline=border_col,
                width=2,
            )

            if not self._disabled:
                shine_r = max(2, r - 2)
                shine_x = thumb_x - shine_r * 0.5
                shine_y = cy - shine_r * 0.6
                self.create_oval(
                    shine_x - shine_r * 0.5,
                    shine_y - shine_r * 0.4,
                    shine_x + shine_r * 0.5,
                    shine_y + shine_r * 0.4,
                    fill=self._lighten(thumb_col, 0.45),
                    outline="",
                )

        except tk.TclError:
            pass

    def _rounded_rect(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        r: float, color: str,
    ) -> None:
        r = max(0.0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
        if r < 0.5:
            self.create_rectangle(x1, y1, x2, y2,
                                  fill=color, outline="")
            return
        self.create_polygon(
            x1 + r, y1,   x2 - r, y1,
            x2,     y1,   x2,     y1 + r,
            x2,     y2 - r, x2,   y2,
            x2 - r, y2,   x1 + r, y2,
            x1,     y2,   x1,     y2 - r,
            x1,     y1 + r, x1,   y1,
            fill=color, outline="", smooth=True,
        )

    @staticmethod
    def _hex_to_rgb(c: str) -> tuple[int, int, int]:
        c = c.lstrip("#")
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

    @staticmethod
    def _rgb_to_hex(r: float, g: float, b: float) -> str:
        return f"#{int(max(0, min(255, r))):02x}{int(max(0, min(255, g))):02x}{int(max(0, min(255, b))):02x}"

    def _blend(self, c1: str, c2: str, a: float) -> str:
        r1, g1, b1 = self._hex_to_rgb(c1)
        r2, g2, b2 = self._hex_to_rgb(c2)
        return self._rgb_to_hex(
            r1 * a + r2 * (1 - a),
            g1 * a + g2 * (1 - a),
            b1 * a + b2 * (1 - a),
        )

    def _lighten(self, c: str, f: float) -> str:
        r, g, b = self._hex_to_rgb(c)
        return self._rgb_to_hex(
            r + (255 - r) * f,
            g + (255 - g) * f,
            b + (255 - b) * f,
        )

    def _darken(self, c: str, f: float) -> str:
        r, g, b = self._hex_to_rgb(c)
        return self._rgb_to_hex(
            r * (1 - f),
            g * (1 - f),
            b * (1 - f),
        )

    def _desaturate(self, c: str, factor: float) -> str:
        r, g, b = self._hex_to_rgb(c)
        grey = 0.299 * r + 0.587 * g + 0.114 * b
        return self._rgb_to_hex(
            r + (grey - r) * (1 - factor),
            g + (grey - g) * (1 - factor),
            b + (grey - b) * (1 - factor),
        )