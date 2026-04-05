#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ui/controllers/model_controller.py"""
from __future__ import annotations

import tkinter as tk
from threading import Thread
from tkinter import messagebox
from typing import Callable

from ui.app_state import AppState
from ui.overlays.model_loading import ModelLoadingOverlay


class ModelController:
    def __init__(self,
                 root: tk.Tk,
                 state: AppState,
                 yolo,
                 overlay: ModelLoadingOverlay,
                 all_models: list[str],
                 available_models: list[str],
                 is_model_local_fn: Callable,
                 get_model_path_fn: Callable,
                 on_model_ready: Callable[[str], None]) -> None:
        self.root              = root
        self.state             = state
        self.yolo              = yolo
        self.overlay           = overlay
        self.all_models        = all_models
        self.available_models  = available_models
        self.is_model_local    = is_model_local_fn
        self.get_model_path    = get_model_path_fn
        self.on_model_ready    = on_model_ready
        self._colors: dict     = {}

    # ── Public ────────────────────────────────────────────────────────────

    def load_model(self, model_name: str) -> None:
        """
        Загружает модель асинхронно.
        Если файл есть локально — грузит напрямую по пути
        (работает с любой .pt, включая кастомные вроде yolo26x-pose.pt).
        Если нет — пытается скачать через YOLO API.
        """
        if self.state.model_loading:
            return

        self.state.model_loading      = True
        self.state.current_model_name = model_name

        # Показываем overlay
        try:
            self.overlay.show(
                title="Загрузка модели",
                subtitle=f"Подготовка {model_name}…",
                model_name=model_name)
        except Exception:
            pass

        Thread(target=self._load_async,
               args=(model_name,),
               daemon=True).start()

    def open_model_menu(self) -> None:
        if self.state.model_loading:
            messagebox.showinfo(
                "Загрузка",
                "Дождитесь окончания загрузки модели.")
            return
        if self.state.analysis_running:
            messagebox.showwarning(
                "Анализ",
                "Нельзя сменить модель во время анализа.")
            return

        # Импортируем диалог только если он есть
        try:
            from ui.dialogs.model_selector import ModelSelectorDialog
            ModelSelectorDialog(
                parent=self.root,
                colors=self._colors,
                all_models=self.all_models,
                available_models=self.available_models,
                current_model=self.state.current_model_name,
                is_model_local_fn=self.is_model_local,
                get_model_path_fn=self.get_model_path,
                on_apply=self._switch_model,
            )
        except Exception as e:
            print(f"[ModelCtrl] open_model_menu: {e}")

    def set_colors(self, colors: dict) -> None:
        self._colors = colors

    # ── Private ───────────────────────────────────────────────────────────

    def _switch_model(self, name: str) -> None:
        if self.state.analysis_running:
            messagebox.showwarning(
                "Анализ",
                "Остановите анализ перед сменой модели.")
            return
        if self.state.model_loading:
            return
        self.load_model(name)

    def _load_async(self, model_name: str) -> None:
        """Фоновый поток загрузки модели."""

        def _status(msg: str) -> None:
            try:
                self.root.after(
                    0, lambda m=msg: self.overlay.update(
                        getattr(self.overlay, "current_value", 0), m))
            except Exception:
                pass

        def _progress(pct: float) -> None:
            try:
                self.root.after(
                    0, lambda p=pct: self.overlay.update(p))
            except Exception:
                pass

        def _source(is_local: bool) -> None:
            src = "локально" if is_local else "скачивание"
            _status(f"Источник: {src}")

        try:
            _progress(5.0)
            _status(f"Подготовка {model_name}…")

            # Путь к файлу — работает с ЛЮБЫМ именем модели
            model_path = self.get_model_path(model_name)
            is_local   = self.is_model_local(model_name)

            _source(is_local)

            if is_local:
                _status(f"Загрузка из файла: {model_path.name}")
            else:
                _status(f"Скачивание {model_name}…")

            if hasattr(self.yolo, "load"):
                self.yolo.load(
                    model_path=str(model_path),
                    on_status=_status,
                    on_progress=_progress,
                    on_source=_source,
                    force=False,
                )
            else:
                # Заглушка для тестов
                import time
                for p in range(10, 101, 10):
                    time.sleep(0.05)
                    _progress(float(p))
                _status("Готово")

        except Exception as e:
            print(f"[ModelCtrl] Ошибка загрузки {model_name}: {e}")
            _progress(100.0)
            _status(f"Ошибка: {e}")

        finally:
            self.state.model_loading = False
            try:
                self.root.after(0, lambda: self.overlay.hide(500))
                self.root.after(200, lambda: self.on_model_ready(model_name))
            except Exception:
                pass