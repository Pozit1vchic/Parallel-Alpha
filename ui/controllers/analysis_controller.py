#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ui/controllers/analysis_controller.py
"""
from __future__ import annotations

import gc
import os
from threading import Thread
from typing import Callable

import numpy as np
import torch

from ui.app_state import AppState
from core.matcher.pose_processor import (
    preprocess_pose,
    is_pose_valid,
    compute_pose_features,
    COCO_N_KPS,
)

try:
    from core.analysis_backend import (
        AnalysisBackend,
        AnalysisProgress,
        AnalysisResult,
        SearchMode,
    )
    _HAS_BACKEND = True
except ImportError:
    _HAS_BACKEND = False


# ── Конвертация quality ───────────────────────────────────────────────────────

_QUALITY_KEY_TO_RU: dict[str, str] = {
    "fast":    "Быстро",
    "medium":  "Средне",
    "maximum": "Максимум",
}
_QUALITY_KEY_TO_FPS: dict[str, int] = {
    "fast":    8,
    "medium":  15,
    "maximum": 30,
}
_ANY_TO_KEY: dict[str, str] = {
    "быстро":   "fast",
    "средне":   "medium",
    "максимум": "maximum",
    "макс":     "maximum",
    "fast":     "fast",
    "medium":   "medium",
    "maximum":  "maximum",
}


def _normalize_quality(raw: str) -> str:
    return _ANY_TO_KEY.get(raw.lower().strip(), "medium")


def _quality_to_fps(raw: str) -> int:
    return _QUALITY_KEY_TO_FPS.get(_normalize_quality(raw), 15)


def _quality_to_ru(raw: str) -> str:
    return _QUALITY_KEY_TO_RU.get(_normalize_quality(raw), "Средне")


# ─────────────────────────────────────────────────────────────────────────────

class AnalysisController:
    """
    Контроллер анализа. Связывает UI с AnalysisBackend / YoloEngine.
    """

    def __init__(
        self,
        root,
        state:           AppState,
        yolo,
        backend:         "AnalysisBackend | None",
        on_progress:     Callable[[float, str], None],
        on_complete:     Callable[[list], None],
        on_batch_status: Callable[[str, int, int], None],
    ) -> None:
        self.root            = root
        self.state           = state
        self.yolo            = yolo
        self.backend         = backend
        self.on_progress     = on_progress
        self.on_complete     = on_complete
        self.on_batch_status = on_batch_status

        self._var_threshold:        object = None
        self._var_scene_interval:   object = None
        self._var_match_gap:        object = None
        self._var_quality:          object = None
        self._var_use_scale_inv:    object = None
        self._var_use_mirror_inv:   object = None
        self._var_use_body_weights: object = None

        self._photo_matcher = None

    def bind_vars(
        self,
        threshold,
        scene_interval,
        match_gap,
        quality,
        use_scale_inv,
        use_mirror_inv,
        use_body_weights,
    ) -> None:
        self._var_threshold         = threshold
        self._var_scene_interval    = scene_interval
        self._var_match_gap         = match_gap
        self._var_quality           = quality
        self._var_use_scale_inv     = use_scale_inv
        self._var_use_mirror_inv    = use_mirror_inv
        self._var_use_body_weights  = use_body_weights

    def _get(self, var, default):
        if var is None:
            return default
        try:
            return var.get()
        except Exception:
            return default

    # ── Настройки из UI ───────────────────────────────────────────────────

    def _build_settings(self) -> dict:
        try:
            threshold = float(self._get(self._var_threshold, 75))
        except (TypeError, ValueError):
            threshold = 75.0

        try:
            scene_interval = float(
                self._get(self._var_scene_interval, 3))
        except (TypeError, ValueError):
            scene_interval = 3.0

        try:
            match_gap = float(self._get(self._var_match_gap, 5))
        except (TypeError, ValueError):
            match_gap = 5.0

        quality_raw = self._get(self._var_quality, "medium")
        quality_ru  = _quality_to_ru(quality_raw)

        use_mirror       = bool(
            self._get(self._var_use_mirror_inv,  False))
        use_body_weights = bool(
            self._get(self._var_use_body_weights, True))

        return {
            "threshold":        threshold,
            "scene_interval":   scene_interval,
            "match_gap":        match_gap,
            "quality":          quality_ru,
            "quality_key":      quality_raw,
            "use_mirror":       use_mirror,
            "use_body_weights": use_body_weights,
        }

    # ── Фото-референс ─────────────────────────────────────────────────────

    def _load_photo_matcher(self) -> None:
        """
        Загружает PhotoMatcher в фоновом треде.
        Вызывать только из треда — detect_batch блокирует выполнение.
        """
        self._photo_matcher = None

        use_photo = getattr(self.state, "use_photo_search", False)
        if not use_photo:
            return
        if not self.state.ref_photos:
            return

        self.root.after(
            0, lambda: self.on_progress(
                0.0, "⏳ Анализ фото-референса..."))

        try:
            from core.photo_matcher import PhotoMatcher
            pm = PhotoMatcher()
            ok = pm.load_references(
                self.state.ref_photos, self.yolo)
            if ok:
                self._photo_matcher = pm
                print(
                    f"[Analysis] фото-референс загружен: "
                    f"{len(self.state.ref_photos)} фото")
                self.root.after(
                    0, lambda: self.on_progress(
                        0.0,
                        "✓ Фото загружено. Начинаем анализ..."))
            else:
                print("[Analysis] фото-референс: "
                      "поза не найдена")
                self.root.after(
                    0, lambda: self.on_progress(
                        0.0, "⚠ Поза не найдена в фото"))
        except Exception as e:
            print(f"[Analysis] ошибка загрузки фото: {e}")

    # ── Запуск / остановка ────────────────────────────────────────────────

    def start(self) -> None:
        if self.state.analysis_running:
            print("[AnalysisCtrl] Уже запущен.")
            return

        if not self.state.video_queue:
            from tkinter import messagebox
            messagebox.showwarning(
                "Нет видео", "Добавьте хотя бы одно видео.")
            return

        if self.state.model_loading:
            from tkinter import messagebox
            messagebox.showwarning(
                "Загрузка",
                "Дождитесь окончания загрузки модели.")
            return

        if not self.yolo.is_loaded:
            from tkinter import messagebox
            messagebox.showwarning(
                "Модель",
                "Модель не загружена. Дождитесь загрузки.")
            return

        self.state.analysis_running = True
        self._photo_matcher = None

        settings = self._build_settings()

        if _HAS_BACKEND and self.backend is not None:
            self._start_with_backend(settings)
        else:
            self._start_legacy(settings)

    def stop(self) -> None:
        self.state.analysis_running = False
        if _HAS_BACKEND and self.backend is not None:
            try:
                self.backend.stop_analysis()
            except Exception as e:
                print(f"[AnalysisCtrl] stop error: {e}")

    # ── Backend-режим ─────────────────────────────────────────────────────

    def _start_with_backend(self, settings: dict) -> None:
        def _run_with_photo() -> None:
            # Фото грузим в треде — не блокируем UI
            self._load_photo_matcher()

            def _on_progress(progress: "AnalysisProgress") -> None:
                pct    = float(progress.percent)
                status = str(progress.status)
                self.root.after(
                    0, lambda: self.on_progress(pct, status))

            def _on_result(result: "AnalysisResult") -> None:
                matches = result.matches or []

                # Фильтр ДО матчинга уже применён в _run_analysis
                # Здесь применяем лёгкий пост-фильтр если нужно
                if self._photo_matcher is not None:
                    try:
                        before  = len(matches)
                        matches = self._photo_matcher.filter_matches(
                            matches, threshold=0.7)
                        print(
                            f"[Analysis] пост-фильтр: "
                            f"{before} → {len(matches)}")
                    except Exception as e:
                        print(f"[Analysis] пост-фильтр ошибка: {e}")

                self.root.after(0, lambda: self._finish(matches))

            # Передаём photo_matcher в backend через атрибут
            if self._photo_matcher is not None:
                self.backend._photo_matcher = self._photo_matcher
            else:
                self.backend._photo_matcher = None

            self.backend.start_analysis(
                video_paths       = list(self.state.video_queue),
                settings          = settings,
                progress_callback = _on_progress,
                result_callback   = _on_result,
                on_error          = self._on_backend_error,
            )

        Thread(
            target=_run_with_photo,
            daemon=True,
            name="AnalysisBackendWithPhoto",
        ).start()

    def _on_backend_error(self, msg: str, exc=None) -> None:
        print(f"[AnalysisCtrl] Backend error: {msg}")
        self.root.after(
            0, lambda: self.on_progress(0.0, f"Ошибка: {msg}"))
        self.root.after(0, lambda: self._finish([]))

    # ── Legacy-режим (без backend) ────────────────────────────────────────

    def _start_legacy(self, settings: dict) -> None:
        Thread(
            target=self._run_legacy,
            args=(settings,),
            daemon=True,
            name="AnalysisLegacy",
        ).start()

    def _run_legacy(self, settings: dict) -> None:
        try:
            from core.matcher import MotionMatcher, build_poses_tensor

            # Фото грузим в треде — не блокируем UI
            self._load_photo_matcher()

            video_paths      = list(self.state.video_queue)
            threshold        = settings["threshold"] / 100.0
            min_gap          = settings["scene_interval"]
            quality_key      = settings.get("quality_key", "medium")
            base_fps         = _quality_to_fps(quality_key)
            use_mirror       = settings["use_mirror"]
            use_body_weights = settings["use_body_weights"]

            all_frames_data: list[dict] = []
            n_videos = len(video_paths)

            for v_idx, v_path in enumerate(video_paths):
                if not self.state.analysis_running:
                    break

                self.root.after(
                    0,
                    lambda i=v_idx, p=v_path:
                        self.on_batch_status(p, i, n_videos))

                frames_data = self._extract_poses_legacy(
                    path             = v_path,
                    video_idx        = v_idx,
                    base_fps         = base_fps,
                    use_body_weights = use_body_weights,
                    n_videos         = n_videos,
                    v_idx            = v_idx,
                )
                all_frames_data.extend(frames_data)

            if not self.state.analysis_running:
                self.root.after(0, lambda: self._finish([]))
                return

            # ── Фильтр по фото ДО матчинга ───────────────────────────────
            if self._photo_matcher is not None:
                before          = len(all_frames_data)
                all_frames_data = (
                    self._photo_matcher
                    .filter_poses_by_reference(
                        all_frames_data,
                        threshold=0.7))
                print(
                    f"[Analysis] кадры после фото-фильтра: "
                    f"{before} → {len(all_frames_data)}")

                if len(all_frames_data) < 2:
                    print("[Analysis] недостаточно кадров "
                          "после фото-фильтра")
                    self.root.after(0, lambda: self._finish([]))
                    return

            self.root.after(
                0, lambda: self.on_progress(
                    75.0, "Сборка тензора…"))

            poses_tensor, poses_meta = build_poses_tensor(
                all_frames_data,
                use_body_weights=use_body_weights)

            if poses_tensor is None:
                self.root.after(0, lambda: self._finish([]))
                return

            self.root.after(
                0, lambda: self.on_progress(
                    80.0, "Поиск совпадений…"))

            device  = "cuda" if torch.cuda.is_available() else "cpu"
            matcher = MotionMatcher(device=device)
            matches = matcher.find_matches(
                poses_tensor = poses_tensor,
                poses_meta   = poses_meta,
                threshold    = threshold,
                min_gap      = min_gap,
                use_mirror   = use_mirror,
            )

            self.root.after(0, lambda: self._finish(matches))

        except Exception:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self._finish([]))

    def _extract_poses_legacy(
        self,
        path:             str,
        video_idx:        int,
        base_fps:         int,
        use_body_weights: bool,
        n_videos:         int,
        v_idx:            int,
    ) -> list[dict]:
        import cv2

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip         = max(1, int(fps / base_fps))

        frames_data:  list[dict]       = []
        batch_frames: list[np.ndarray] = []
        batch_meta:   list[dict]       = []
        frame_idx = 0
        BATCH     = 16

        while cap.isOpened() and self.state.analysis_running:
            grabbed = cap.grab()
            if not grabbed:
                break

            if frame_idx % skip == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    h, w  = frame.shape[:2]
                    scale = min(640 / w, 640 / h, 1.0)
                    if scale < 1.0:
                        frame = cv2.resize(
                            frame,
                            (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                    batch_frames.append(frame)
                    batch_meta.append({
                        "time":  frame_idx / fps,
                        "frame": frame_idx,
                    })

            if len(batch_frames) >= BATCH:
                self._flush(
                    batch_frames, batch_meta,
                    frames_data, video_idx,
                    use_body_weights,
                )
                batch_frames = []
                batch_meta   = []

            frame_idx += 1

            if frame_idx % 60 == 0 and total_frames > 0:
                local_pct = frame_idx / total_frames
                pct       = (v_idx / n_videos
                             + local_pct / n_videos) * 70.0
                self.root.after(
                    0,
                    lambda p=pct, f=os.path.basename(path):
                        self.on_progress(
                            p, f"Извлечение: {f}"))

        if batch_frames and self.state.analysis_running:
            self._flush(
                batch_frames, batch_meta,
                frames_data, video_idx,
                use_body_weights,
            )

        cap.release()
        return frames_data

    def _flush(
        self,
        batch_frames:     list[np.ndarray],
        batch_meta:       list[dict],
        frames_data:      list[dict],
        video_idx:        int,
        use_body_weights: bool,
    ) -> None:
        if not self.yolo.is_loaded:
            return

        try:
            poses_data = self.yolo.detect_batch(batch_frames)
        except Exception as exc:
            msg = str(exc)
            if ("не загружена" not in msg
                    and "not loaded" not in msg):
                print(f"[AnalysisCtrl] detect_batch error: {exc}")
            return

        for pose_data, meta in zip(poses_data, batch_meta):
            if pose_data is None:
                continue

            kps = pose_data.get("keypoints")
            if kps is None or kps.shape[0] < COCO_N_KPS:
                continue

            if not is_pose_valid(pose_data):
                continue

            scale    = pose_data.get("scale")
            anchor_y = pose_data.get("anchor_y")

            if scale is None or anchor_y is None:
                kps_xy          = kps[:COCO_N_KPS, :2].astype(
                    np.float32)
                scale, anchor_y = compute_pose_features(kps_xy)
                orig_h          = pose_data.get("orig_h", 720)
                anchor_y        = anchor_y / (orig_h + 1e-5)

            kp_raw  = pose_data.get("kp") or pose_data.get(
                "keypoints")
            kp_list = (kp_raw.tolist()
                       if isinstance(kp_raw, np.ndarray)
                       else kp_raw)

            frames_data.append({
                "t":         meta["time"],
                "f":         meta["frame"],
                "video_idx": video_idx,
                "dir":       pose_data.get("direction", "forward"),
                "scale":     float(scale),
                "anchor_y":  float(anchor_y),
                "kp":        kp_list,
                "poses":     [pose_data],
            })

    # ── Завершение ────────────────────────────────────────────────────────

    def _finish(self, matches: list) -> None:
        # Сбрасываем photo_matcher
        self._photo_matcher = None

        self.state.analysis_running = False
        self.on_complete(matches)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Auto-tune ─────────────────────────────────────────────────────────

    def auto_tune(self) -> dict:
        """
        Автоматически выбрать качество по VRAM/RAM.
        Возвращает dict с нейтральным ключом quality.
        """
        try:
            if torch.cuda.is_available():
                vram_gb = (torch.cuda.get_device_properties(0)
                           .total_memory / 1e9)
                if vram_gb >= 10:
                    quality = "maximum"
                elif vram_gb >= 6:
                    quality = "medium"
                else:
                    quality = "fast"
            else:
                import psutil
                ram_gb  = psutil.virtual_memory().total / 1e9
                quality = "medium" if ram_gb >= 16 else "fast"
        except Exception:
            quality = "medium"

        return {"quality": quality}