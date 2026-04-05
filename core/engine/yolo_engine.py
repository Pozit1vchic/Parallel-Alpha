#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YoloEngine — инференс YOLO-pose.
"""
from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch

from core.engine.model_manager import (
    ModelManager,
    MODELS_DIR,
    DEFAULT_MODEL_NAME,
    AVAILABLE_MODELS,
    _safe_cb,
)

DEFAULT_CONF     = 0.25
KP_VIS_THRESHOLD = 0.30


class YoloEngine:
    """
    Обёртка над YOLO11/YOLO26-pose.

    Публичный API:
      load(model_path, *, on_status, on_progress, on_source, force)
      reload(model_path, ...)
      detect_batch(frames_batch) → list[dict | None]
      classify_direction(kp17)   → str
      get_dynamic_batch_size(frame_sizes, default_batch) → int
      model_name  → str
      is_loaded   → bool
    """

    AVAILABLE_MODELS = AVAILABLE_MODELS
    DEFAULT_CONF     = DEFAULT_CONF
    KP_VIS_THRESHOLD = KP_VIS_THRESHOLD

    def __init__(self, device: str | None = None) -> None:
        if device is not None:
            self.device = (
                device
                if (torch.cuda.is_available() or device == "cpu")
                else "cpu"
            )
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.use_fp16 = (self.device == "cuda")

        self._manager     = ModelManager()
        self.model        = None
        self._model_name: str = ""
        self._model_path: str = ""

        self._is_loading = threading.Event()

        self._cached_batch_size: int   = 32
        self._cached_avg_area:   float = -1.0
        self._cached_default:    int   = 32

    # ── Загрузка ──────────────────────────────────────────────────────────

    def load(
        self,
        model_path:  str = DEFAULT_MODEL_NAME,
        *,
        on_status:   Optional[Callable[[str],   None]] = None,
        on_progress: Optional[Callable[[float], None]] = None,
        on_source:   Optional[Callable[[bool],  None]] = None,
        force:       bool = False,
    ) -> None:
        if self._is_loading.is_set():
            print("[YoloEngine] ⚠ Загрузка уже выполняется.")
            return

        name = Path(model_path).name

        if (not force
                and self.model is not None
                and self._model_name == name):
            _safe_cb(on_status,   f"Модель {name} уже загружена.")
            _safe_cb(on_progress, 100.0)
            return

        self._is_loading.set()
        try:
            self._load_impl(
                name=name,
                on_status=on_status,
                on_progress=on_progress,
                on_source=on_source,
            )
        finally:
            self._is_loading.clear()

    def reload(
        self,
        model_path:  str,
        *,
        on_status:   Optional[Callable[[str],   None]] = None,
        on_progress: Optional[Callable[[float], None]] = None,
        on_source:   Optional[Callable[[bool],  None]] = None,
    ) -> None:
        self.load(
            model_path,
            on_status=on_status,
            on_progress=on_progress,
            on_source=on_source,
            force=True,
        )

    def _load_impl(
        self,
        name:        str,
        on_status:   Optional[Callable[[str],   None]],
        on_progress: Optional[Callable[[float], None]],
        on_source:   Optional[Callable[[bool],  None]],
    ) -> None:
        from ultralytics import YOLO

        try:
            local_path = self._manager.prepare(
                name,
                on_status=on_status,
                on_progress=lambda p: _safe_cb(on_progress, p * 0.6),
                on_source=on_source,
            )
        except RuntimeError as exc:
            print(f"[YoloEngine] ✗ Не удалось подготовить: {exc}")
            _safe_cb(on_status, f"Ошибка: {exc}")
            return

        yolo_arg = local_path if local_path.is_file() else name

        _safe_cb(on_status,   f"Загружается {name}…")
        _safe_cb(on_progress, 65.0)

        if self.model is not None:
            self._release()

        try:
            self.model = YOLO(str(yolo_arg))
            self.model.to(self.device)
        except Exception as exc:
            print(f"[YoloEngine] ✗ Ошибка загрузки: {exc}")
            _safe_cb(on_status, f"Ошибка: {exc}")
            self.model = None
            return

        _safe_cb(on_progress, 80.0)

        if self.device == "cpu":
            self.use_fp16 = False
        else:
            self.use_fp16 = self._test_fp16()
            if not self.use_fp16:
                print("[YoloEngine] ⚠ FP16 недоступен, используем FP32.")

        self._model_name = name
        self._model_path = str(yolo_arg)

        print(
            f"[YoloEngine] ✓ {name} | {self.device.upper()} "
            f"| FP16={self.use_fp16}"
        )

        _safe_cb(on_progress, 85.0)
        _safe_cb(on_status,   f"Прогрев {name}…")

        if self.device == "cuda":
            self._warmup()

        _safe_cb(on_progress, 100.0)
        _safe_cb(on_status,   f"{name} готова.")

    def _release(self) -> None:
        try:
            del self.model
        except Exception:
            pass
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _test_fp16(self) -> bool:
        try:
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self.model.predict(
                dummy, imgsz=64, verbose=False,
                half=True, conf=self.DEFAULT_CONF,
                stream=False, device=self.device,
            )
            return True
        except Exception:
            return False

    def _warmup(self, runs: int = 3) -> None:
        try:
            imgsz = int(
                getattr(self.model, "args", {}).get("imgsz", 640)
            )
        except Exception:
            imgsz = 640

        imgsz = max(320, min(imgsz, 1280))
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

        for i in range(runs):
            try:
                self.model.predict(
                    dummy, imgsz=imgsz, verbose=False,
                    half=self.use_fp16, conf=self.DEFAULT_CONF,
                    stream=False, device=self.device,
                )
            except Exception as exc:
                print(f"[YoloEngine] ⚠ Warmup прерван на шаге {i+1}: {exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break

        print(f"[YoloEngine] Warmup: {runs} прохода — готово")

    # ── Свойства ──────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def model_path(self) -> str:
        return self._model_path

    @model_path.setter
    def model_path(self, v: str) -> None:
        self._model_path = v

    def get_model_info(self) -> dict:
        return {
            "name":      self._model_name,
            "path":      self._model_path,
            "device":    self.device,
            "fp16":      self.use_fp16,
            "is_loaded": self.model is not None,
            "is_local":  (self._manager.is_local(self._model_name)
                          if self._model_name else False),
        }

    def list_local_models(self) -> list[str]:
        return self._manager.list_local()

    # ── Инференс ──────────────────────────────────────────────────────────

    def detect_batch(self, frames_batch: list) -> list[dict | None]:
        """
        Детектировать позы в батче BGR-кадров.

        Returns list[dict | None]:
          dict ключи: keypoints(17,3), kp(list), confidence, bbox,
                      direction, orig_w, orig_h, scale, anchor_y
        """
        if not frames_batch:
            return []
        if self.model is None:
            raise RuntimeError(
                "Модель не загружена. Вызовите load() перед detect_batch()."
            )

        try:
            results = self.model.predict(
                frames_batch,
                imgsz=640,
                verbose=False,
                half=self.use_fp16,
                conf=self.DEFAULT_CONF,
                stream=False,
                device=self.device,
            )
            return [self._parse_result(r) for r in results]

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(
                    f"[YoloEngine] OOM ({len(frames_batch)} кадров), "
                    f"поштучная обработка."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return [self._detect_single(f) for f in frames_batch]
            print(f"[YoloEngine] detect_batch RuntimeError: {exc}")
            return [None] * len(frames_batch)

        except Exception as exc:
            print(f"[YoloEngine] detect_batch error: {exc}")
            return [None] * len(frames_batch)

    def _detect_single(self, frame: np.ndarray) -> dict | None:
        try:
            results = self.model.predict(
                frame, imgsz=640, verbose=False,
                half=False, conf=self.DEFAULT_CONF,
                stream=False, device=self.device,
            )
            if results:
                return self._parse_result(results[0])
        except Exception as exc:
            print(f"[YoloEngine] _detect_single error: {exc}")
        return None

    def _parse_result(self, res) -> dict | None:
        """
        Разобрать результат YOLO → dict.

        Поля:
          keypoints  : np.ndarray (17, 3) — x, y, conf
          kp         : list[[x,y,conf],...] — для JSON/matcher/классификатора
          confidence : float
          bbox       : [x1,y1,x2,y2]
          direction  : str
          orig_w/h   : int
          scale      : float  — для matcher re-scoring
          anchor_y   : float  — для matcher re-scoring
        """
        if res.keypoints is None or len(res.keypoints.data) == 0:
            return None

        kp_all         = res.keypoints.data
        orig_h, orig_w = res.orig_shape

        best: dict | None = None
        best_conf         = -1.0

        for person_idx in range(len(kp_all)):
            kp = kp_all[person_idx].cpu().numpy()   # (N, 3)
            if kp.shape[0] < 17:
                continue

            kp17    = kp[:17].copy()
            visible = kp17[kp17[:, 2] >= self.KP_VIS_THRESHOLD]
            if len(visible) < 5:
                continue

            conf = float(np.mean(kp17[:, 2]))
            if conf <= best_conf:
                continue

            if len(visible) > 0:
                bbox = [
                    float(np.min(visible[:, 0])),
                    float(np.min(visible[:, 1])),
                    float(np.max(visible[:, 0])),
                    float(np.max(visible[:, 1])),
                ]
            else:
                bbox = [0.0, 0.0, float(orig_w), float(orig_h)]

            # Масштаб и позиция для matcher re-scoring
            anchor_kps = kp17[[5, 6, 11, 12], :2]
            anchor_xy  = anchor_kps.mean(axis=0)
            centered   = kp17[:, :2] - anchor_xy
            scale      = float(np.max(np.abs(centered)) + 1e-5)
            anchor_y   = float(anchor_xy[1] / (orig_h + 1e-5))

            best_conf = conf
            best = {
                "keypoints":  kp17,
                # ── kp как list — для matcher и классификатора ────────────
                # Формат: [[x, y, conf], ...] × 17
                "kp":         kp17.tolist(),
                "confidence": conf,
                "bbox":       bbox,
                "direction":  self._classify_direction(kp17),
                "orig_w":     int(orig_w),
                "orig_h":     int(orig_h),
                "scale":      scale,
                "anchor_y":   anchor_y,
            }

        return best

    # ── Классификация направления ─────────────────────────────────────────

    def _classify_direction(self, kp17: np.ndarray) -> str:
        """
        Определить направление тела по COCO-17.
        Многоуровневый алгоритм: плечи → уши → нос → смещение.
        """
        NOSE                   = 0
        L_EAR,  R_EAR          = 3, 4
        L_SHOULDER, R_SHOULDER = 5, 6
        VIS                    = self.KP_VIS_THRESHOLD

        try:
            conf_ls   = float(kp17[L_SHOULDER, 2])
            conf_rs   = float(kp17[R_SHOULDER, 2])
            conf_nose = float(kp17[NOSE,       2])
            conf_le   = float(kp17[L_EAR,      2])
            conf_re   = float(kp17[R_EAR,      2])

            # Оба плеча не видны
            if conf_ls < VIS and conf_rs < VIS:
                if conf_le >= VIS and conf_re < VIS:
                    return "right"
                if conf_re >= VIS and conf_le < VIS:
                    return "left"
                return "unknown"

            # Только одно плечо
            if conf_ls >= VIS and conf_rs < VIS - 0.05:
                return "right"
            if conf_rs >= VIS and conf_ls < VIS - 0.05:
                return "left"

            # Оба плеча — смотрим нос/уши
            shoulder_cx = (kp17[L_SHOULDER, 0] + kp17[R_SHOULDER, 0]) / 2.0
            shoulder_w  = (
                abs(kp17[L_SHOULDER, 0] - kp17[R_SHOULDER, 0]) + 1e-5
            )

            if conf_nose >= VIS:
                offset = (kp17[NOSE, 0] - shoulder_cx) / shoulder_w
            elif conf_le >= VIS and conf_re >= VIS:
                ear_cx = (kp17[L_EAR, 0] + kp17[R_EAR, 0]) / 2.0
                offset = (ear_cx - shoulder_cx) / shoulder_w
            elif conf_le >= VIS:
                offset = (kp17[L_EAR, 0] - shoulder_cx) / shoulder_w
            elif conf_re >= VIS:
                offset = (kp17[R_EAR, 0] - shoulder_cx) / shoulder_w
            else:
                offset = (conf_rs - conf_ls) * 0.15

            thr = max(0.06, min(0.15, 0.3 / (shoulder_w / 50.0 + 1e-5)))
            if abs(offset) < thr:
                return "forward"
            return "right" if offset > 0 else "left"

        except Exception:
            return "unknown"

    def classify_direction(self, kp17: np.ndarray) -> str:
        """Публичный псевдоним."""
        return self._classify_direction(kp17)

    # ── Адаптивный батч ───────────────────────────────────────────────────

    def get_dynamic_batch_size(
        self,
        frame_sizes:   list,
        default_batch: int = 32,
    ) -> int:
        """
        Батч-размер по разрешению кадров и VRAM.

        Пороги (Мпкс):
          > 7.0  → 4K
          > 3.5  → 1440p
          > 1.5  → 1080p
          > 0.7  → 720p
          > 0.3  → 540p
          ≤ 0.3  → 360p и меньше
        """
        if not frame_sizes:
            return default_batch

        avg_area = float(
            np.mean([h * w for h, w in frame_sizes])
        ) / 1e6

        if (abs(avg_area - self._cached_avg_area) < 0.05
                and default_batch == self._cached_default):
            return self._cached_batch_size

        vram_factor = 1.0
        if torch.cuda.is_available():
            try:
                props      = torch.cuda.get_device_properties(0)
                total_vram = props.total_memory / 1e9
                used_vram  = torch.cuda.memory_reserved() / 1e9
                free_ratio = max(
                    0.0, 1.0 - used_vram / (total_vram + 1e-6))
                vram_factor = max(0.3, free_ratio)
            except Exception:
                pass

        match True:
            case _ if avg_area > 7.0:
                coeff = 0.20
            case _ if avg_area > 3.5:
                coeff = 0.35
            case _ if avg_area > 1.5:
                coeff = 0.55
            case _ if avg_area > 0.7:
                coeff = 0.80
            case _ if avg_area > 0.3:
                coeff = 1.10
            case _:
                coeff = 1.60

        result = max(1, min(int(default_batch * coeff * vram_factor), 256))

        self._cached_batch_size = result
        self._cached_avg_area   = avg_area
        self._cached_default    = default_batch

        return result