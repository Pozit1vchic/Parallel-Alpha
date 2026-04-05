from __future__ import annotations

import tkinter as tk
from typing import Callable

from ui.app_state import AppState


class NavigationController:
    def __init__(self, root: tk.Tk,
                 state: AppState,
                 on_show_preview: Callable[[int], None]) -> None:
        self.root            = root
        self.state           = state
        self.on_show_preview = on_show_preview

    def prev_match(self) -> None:
        if self.state.current_match > 0:
            self.on_show_preview(self.state.current_match - 1)

    def next_match(self) -> None:
        if self.state.current_match < len(self.state.matches) - 1:
            self.on_show_preview(self.state.current_match + 1)

    def go_to(self, index: int) -> None:
        if not self.state.matches:
            return
        index = max(0, min(index, len(self.state.matches) - 1))
        self.on_show_preview(index)

    def setup_hotkeys(self, root: tk.Tk,
                      on_start: Callable,
                      on_fullscreen: Callable,
                      on_exit_fullscreen: Callable) -> None:
        root.bind("<space>",  lambda _e: on_start())
        root.bind("<Up>",     lambda _e: self.prev_match())
        root.bind("<Down>",   lambda _e: self.next_match())
        root.bind("<w>",      lambda _e: self.prev_match())
        root.bind("<s>",      lambda _e: self.next_match())
        root.bind("<W>",      lambda _e: self.prev_match())
        root.bind("<S>",      lambda _e: self.next_match())
        root.bind("<F11>",    lambda _e: on_fullscreen())
        root.bind("<Escape>", lambda _e: on_exit_fullscreen())