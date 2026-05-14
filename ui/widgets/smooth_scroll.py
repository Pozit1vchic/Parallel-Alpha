from __future__ import annotations

import platform
import time
from typing import Callable

import tkinter as tk


class SmoothScrollMixin:
    """
    Плавный скролл с инерцией для tkinter Canvas виджетов.
    
    Используется в:
    - VirtualResultsList (основной список результатов)
    - PreviewPanel (таймлайн)
    - ModelSelectorDialog (список моделей)
    """

    _SCROLL_DECAY:    float = 0.85
    _SCROLL_MIN_VEL:  float = 0.3
    _SCROLL_FPS:      int   = 60
    _SCROLL_INTERVAL: int   = 1000 // 60
    _SCROLL_STEP:     float = 2.5

    _PLATFORM = platform.system()

    def _bind_smooth_scroll(
        self,
        widget: tk.Widget,
        scroll_cmd: Callable[[int, str], None],
    ) -> None:
        state = _ScrollState(
            widget       = widget,
            scroll_cmd   = scroll_cmd,
            decay        = self._SCROLL_DECAY,
            min_vel      = self._SCROLL_MIN_VEL,
            interval     = self._SCROLL_INTERVAL,
            step         = self._SCROLL_STEP,
        )

        if self._PLATFORM == "Windows":
            widget.bind("<MouseWheel>",
                        lambda e: state.push(
                            -1 if e.delta > 0 else 1))
        elif self._PLATFORM == "Darwin":
            widget.bind("<MouseWheel>",
                        lambda e: state.push(
                            -1 if e.delta > 0 else 1))
        else:
            widget.bind("<Button-4>", lambda _e: state.push(-1))
            widget.bind("<Button-5>", lambda _e: state.push(1))

        widget.bind("<Destroy>", lambda _e: state.stop())

    def _unbind_smooth_scroll(self, widget: tk.Widget) -> None:
        for seq in ("<MouseWheel>", "<Button-4>",
                    "<Button-5>", "<Destroy>"):
            try:
                widget.unbind(seq)
            except Exception:
                pass


class _ScrollState:
    """Состояние анимации инерционного скролла для одного виджета."""

    __slots__ = (
        "widget", "scroll_cmd", "decay", "min_vel",
        "interval", "step", "velocity", "_anim_id",
        "_alive",
    )

    def __init__(
        self,
        widget:     tk.Widget,
        scroll_cmd: Callable[[int, str], None],
        decay:      float,
        min_vel:    float,
        interval:   int,
        step:       float,
    ) -> None:
        self.widget     = widget
        self.scroll_cmd = scroll_cmd
        self.decay      = decay
        self.min_vel    = min_vel
        self.interval   = interval
        self.step       = step
        self.velocity   = 0.0
        self._anim_id: str | None = None
        self._alive     = True

    def push(self, direction: int) -> None:
        self.velocity += direction * self.step
        self._clamp_velocity()
        self._ensure_animating()

    def stop(self) -> None:
        self._alive = False
        if self._anim_id is not None:
            try:
                self.widget.after_cancel(self._anim_id)
            except Exception:
                pass
            self._anim_id = None

    def _clamp_velocity(self) -> None:
        max_vel = self.step * 8
        if self.velocity > max_vel:
            self.velocity = max_vel
        elif self.velocity < -max_vel:
            self.velocity = -max_vel

    def _ensure_animating(self) -> None:
        if self._anim_id is None and self._alive:
            self._anim_id = self.widget.after(
                self.interval, self._tick)

    def _tick(self) -> None:
        self._anim_id = None

        if not self._alive:
            return

        try:
            if not self.widget.winfo_exists():
                self._alive = False
                return
        except Exception:
            self._alive = False
            return

        if abs(self.velocity) < self.min_vel:
            self.velocity = 0.0
            return

        n = int(round(self.velocity))
        if n != 0:
            try:
                self.scroll_cmd(n, "units")
            except Exception:
                self._alive = False
                return

        self.velocity *= self.decay

        if abs(self.velocity) >= self.min_vel:
            self._anim_id = self.widget.after(
                self.interval, self._tick)
        else:
            self.velocity = 0.0