#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ui/dialogs/model_selector.py — Диалог выбора модели YOLO
"""
from __future__ import annotations
import tkinter as tk
from tkinter import messagebox
from typing import Callable
from utils.locales import t


class ModelSelectorDialog(tk.Toplevel):
    """
    Диалог выбора модели.
    - Тёмная тема, кастомный список
    - ✓ — локальная, ↓ — будет скачана
    - Без иконок в названиях
    - Кнопки Apply/Cancel красивые
    """

    def __init__(
        self,
        parent,
        models: list[str],
        current: str,
        local_models: set[str],
        on_select: Callable[[str], None],
    ):
        super().__init__(parent)
        self._models       = models
        self._current      = current
        self._local        = local_models
        self._on_select    = on_select
        self._selected     = current

        # Окно
        self.title(t("model_selector_title"))
        self.configure(bg="#0f0f1a")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        w, h = 360, 480
        px   = parent.winfo_rootx() + parent.winfo_width() // 2 - w // 2
        py   = parent.winfo_rooty() + parent.winfo_height() // 2 - h // 2
        self.geometry(f"{w}x{h}+{px}+{py}")

        self._build()
        self.bind("<Escape>", lambda _: self.destroy())
        self.bind("<Return>", lambda _: self._apply())

    def _build(self) -> None:
        # Заголовок
        tk.Label(
            self,
            text=t("choose_yolo_model"),
            bg="#0f0f1a", fg="#e2e8f0",
            font=("Segoe UI", 13, "bold"),
        ).pack(pady=(20, 4), padx=20, anchor="w")

        # Легенда
        legend = tk.Frame(self, bg="#0f0f1a")
        legend.pack(fill="x", padx=20, pady=(0, 12))
        tk.Label(
            legend,
            text="✓ — " + t("local_model"),
            bg="#0f0f1a", fg="#22c55e",
            font=("Segoe UI", 8),
        ).pack(side="left", padx=(0, 12))
        tk.Label(
            legend,
            text="↓ — " + t("will_download"),
            bg="#0f0f1a", fg="#64748b",
            font=("Segoe UI", 8),
        ).pack(side="left")

        # Разделитель
        tk.Frame(self, bg="#1e293b", height=1).pack(
            fill="x", padx=0, pady=(0, 4)
        )

        # Список
        list_frame = tk.Frame(self, bg="#0f0f1a")
        list_frame.pack(fill="both", expand=True, padx=0)

        scrollbar = tk.Scrollbar(list_frame, bg="#1e293b", troughcolor="#0f0f1a")
        scrollbar.pack(side="right", fill="y")

        self._listbox = tk.Listbox(
            list_frame,
            bg="#0f0f1a",
            fg="#94a3b8",
            selectbackground="#1d4ed8",
            selectforeground="#ffffff",
            font=("Consolas", 10),
            relief="flat",
            bd=0,
            highlightthickness=0,
            activestyle="none",
            yscrollcommand=scrollbar.set,
        )
        self._listbox.pack(side="left", fill="both", expand=True, padx=(20, 0))
        scrollbar.config(command=self._listbox.yview)

        # Заполнить список
        self._fill_list()

        self._listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        self._listbox.bind("<Double-Button-1>", lambda _: self._apply())

        # Разделитель
        tk.Frame(self, bg="#1e293b", height=1).pack(fill="x", pady=(4, 0))

        # Кнопки
        btn_frame = tk.Frame(self, bg="#0f0f1a")
        btn_frame.pack(fill="x", padx=20, pady=12)

        tk.Button(
            btn_frame,
            text=t("apply"),
            command=self._apply,
            bg="#1d4ed8", fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief="flat", bd=0,
            padx=24, pady=8,
            cursor="hand2",
            activebackground="#2563eb",
            activeforeground="#ffffff",
        ).pack(side="left", padx=(0, 8))

        tk.Button(
            btn_frame,
            text=t("cancel"),
            command=self.destroy,
            bg="#1e293b", fg="#94a3b8",
            font=("Segoe UI", 10),
            relief="flat", bd=0,
            padx=24, pady=8,
            cursor="hand2",
            activebackground="#334155",
            activeforeground="#e2e8f0",
        ).pack(side="left")

    def _fill_list(self) -> None:
        self._listbox.delete(0, "end")
        select_idx = 0
        for i, model in enumerate(self._models):
            is_local   = model in self._local
            is_current = model == self._current
            prefix     = "✓" if is_local else "↓"
            suffix     = f"  ◀ {t('active')}" if is_current else ""
            self._listbox.insert("end", f"  {prefix}  {model}{suffix}")

            # Цвет
            if is_current:
                self._listbox.itemconfig(i, fg="#60A5FA")
            elif is_local:
                self._listbox.itemconfig(i, fg="#e2e8f0")
            else:
                self._listbox.itemconfig(i, fg="#475569")

            if is_current:
                select_idx = i

        # Выбрать текущую
        self._listbox.selection_set(select_idx)
        self._listbox.see(select_idx)

    def _on_listbox_select(self, _=None) -> None:
        sel = self._listbox.curselection()
        if sel:
            self._selected = self._models[sel[0]]

    def _apply(self) -> None:
        if self._selected:
            self._on_select(self._selected)
        self.destroy()