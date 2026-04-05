from __future__ import annotations

import platform
from typing import Callable
import tkinter as tk


class SmoothScrollMixin:
    """Миксин для плавного скролла колесом мыши."""

    def _bind_smooth_scroll(self, widget: tk.Widget,
                             scroll_cmd: Callable) -> None:
        def _on_wheel(event: tk.Event) -> None:
            if platform.system() == "Windows":
                delta = -int(event.delta / 120)
            elif platform.system() == "Darwin":
                delta = -event.delta
            else:
                delta = -1 if event.num == 4 else 1
            for _ in range(abs(delta)):
                scroll_cmd(int(delta / abs(delta)), "units")

        widget.bind("<MouseWheel>", _on_wheel)
        widget.bind("<Button-4>",   _on_wheel)
        widget.bind("<Button-5>",   _on_wheel)