#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import gc
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from core.engine.model_manager import (
    ModelManager,
    DEFAULT_MODEL_NAME,
    AVAILABLE_MODELS,
    _safe_cb,
)

try:
    from utils.constants import (
        YOLO_CONF               as _CONF,
        YOLO_IMGSZ              as _IMGSZ,
        KEYPOINT_CONF_THRESHOLD as _KP_VIS,
    )
except ImportError:
    _CONF   = 0.25
    _IMGSZ  = 640
    _KP_VIS = 0.30

# ---------------------------------------------------------------------------
# Constants (ALL PRESERVED + additions)
# ---------------------------------------------------------------------------

DEFAULT_CONF     : float = _CONF
IMGSZ            : int   = _IMGSZ
KP_VIS_THRESHOLD : float = _KP_VIS

BATCH_SIZE_GPU   : int   = 96
BATCH_SIZE_CPU   : int   = 16

_NOSE       = 0
_L_EAR      = 3
_R_EAR      = 4
_L_SHOULDER = 5
_R_SHOULDER = 6

_ADAPTIVE_MIN = 0.06
_ADAPTIVE_MAX = 0.15

_MIN_CUDA_CC_FOR_FP16  = (7, 0)
_WARMUP_MAX_RETRIES    = 3
_WARMUP_BATCH_DIVIDER  = 2
_BATCH_OOM_MIN         = 4
_PARSE_MIN_VISIBLE_KP  = 5


# ---------------------------------------------------------------------------
# GPU capability check
# ---------------------------------------------------------------------------

def _supports_fp16() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        cc = torch.cuda.get_device_capability()
        return cc >= _MIN_CUDA_CC_FOR_FP16
    except Exception:
        return False


# ---------------------------------------------------------------------------
# FULLY VECTORISED batch direction classifier
# ---------------------------------------------------------------------------

# Маппинг для эффективного перевода числовых меток в строки
_DIR_LABELS: tuple[str, ...] = ("unknown", "forward", "left", "right")
# 0=unknown, 1=forward, 2=left, 3=right


def _classify_directions_batch(
    kp_batch: torch.Tensor,
    vis_thr:  float,
) -> list[str]:
    """
    Полностью векторизованная классификация направления для N поз.
    kp_batch: (N, 17, 3) — на GPU/CPU.

    В исходной версии тут был Python-цикл с .item() вызовами, что
    приводило к синхронизации GPU stream на каждой позе. Теперь весь
    расчёт идёт батчем, и только финальный mapping в строки делается
    одним numpy-индексированием.
    """
    N = kp_batch.shape[0]
    if N == 0:
        return []

    # Работаем в float32 на исходном устройстве — без копий
    kp = kp_batch
    if kp.dtype != torch.float32:
        kp = kp.float()

    c_nose = kp[:, _NOSE,       2]
    c_le   = kp[:, _L_EAR,      2]
    c_re   = kp[:, _R_EAR,      2]
    c_ls   = kp[:, _L_SHOULDER, 2]
    c_rs   = kp[:, _R_SHOULDER, 2]

    lsx    = kp[:, _L_SHOULDER, 0]
    rsx    = kp[:, _R_SHOULDER, 0]
    nose_x = kp[:, _NOSE,       0]
    le_x   = kp[:, _L_EAR,      0]
    re_x   = kp[:, _R_EAR,      0]

    shoulder_cx = (lsx + rsx) * 0.5
    shoulder_w  = (lsx - rsx).abs() + 1e-5

    ls_ok = c_ls >= vis_thr
    rs_ok = c_rs >= vis_thr
    le_ok = c_le >= vis_thr
    re_ok = c_re >= vis_thr
    nose_ok = c_nose >= vis_thr
    margin = vis_thr - 0.05

    # Адаптивный порог
    adaptive_thr = torch.clamp(
        0.3 / (shoulder_w / 50.0 + 1e-5),
        min=_ADAPTIVE_MIN, max=_ADAPTIVE_MAX,
    )

    # ---- head_x (по приоритету: nose → both ears → l_ear → r_ear) ----
    # Если плечи невидимы и точек головы тоже нет, head_offset считается отдельно.
    head_x = torch.where(nose_ok, nose_x,
             torch.where(le_ok & re_ok, (le_x + re_x) * 0.5,
             torch.where(le_ok, le_x,
             torch.where(re_ok, re_x, shoulder_cx))))

    head_offset_norm = (head_x - shoulder_cx) / shoulder_w

    # Случай "нет головы вообще" — fallback по разнице конфидансов плеч
    no_head = ~(nose_ok | le_ok | re_ok)
    head_offset_fallback = (c_rs - c_ls) * 0.15  # абсолютный, не нормированный
    # Но мы хотим работать в той же шкале — порог тоже подменяем для no_head:
    # Используем |head_offset_fallback| и сравним с adaptive_thr.

    # ---- финальные метки (int8) ----
    # 0=unknown, 1=forward, 2=left, 3=right
    labels = torch.zeros(N, dtype=torch.int8, device=kp.device)

    both_shoulders_invisible = (~ls_ok) & (~rs_ok)

    # Сценарий 1: плечи невидимы → решаем только по ушам
    s1_right = both_shoulders_invisible & le_ok & (~re_ok)
    s1_left  = both_shoulders_invisible & re_ok & (~le_ok)
    labels = torch.where(s1_right, torch.tensor(3, dtype=torch.int8, device=kp.device), labels)
    labels = torch.where(s1_left,  torch.tensor(2, dtype=torch.int8, device=kp.device), labels)

    # Сценарий 2: одно плечо видимо, другое слабее margin
    s2_right = (~both_shoulders_invisible) & ls_ok & (~(c_rs >= margin))
    s2_left  = (~both_shoulders_invisible) & rs_ok & (~(c_ls >= margin)) & (~s2_right)
    labels = torch.where(s2_right, torch.tensor(3, dtype=torch.int8, device=kp.device), labels)
    labels = torch.where(s2_left,  torch.tensor(2, dtype=torch.int8, device=kp.device), labels)

    # Сценарий 3: оба плеча видимы (или хотя бы одно достаточно видно)
    handled = both_shoulders_invisible | s2_right | s2_left

    # Под-сценарий 3a: голова видна — по нормированному offset
    s3_has_head = (~handled) & (~no_head)
    abs_off = head_offset_norm.abs()
    s3_forward = s3_has_head & (abs_off < adaptive_thr)
    s3_right   = s3_has_head & (~s3_forward) & (head_offset_norm > 0)
    s3_left    = s3_has_head & (~s3_forward) & (head_offset_norm <= 0)
    labels = torch.where(s3_forward, torch.tensor(1, dtype=torch.int8, device=kp.device), labels)
    labels = torch.where(s3_right,   torch.tensor(3, dtype=torch.int8, device=kp.device), labels)
    labels = torch.where(s3_left,    torch.tensor(2, dtype=torch.int8, device=kp.device), labels)

    # Под-сценарий 3b: головы нет — fallback по конфидансам плеч
    s3_no_head = (~handled) & no_head
    abs_fb = head_offset_fallback.abs()
    s3b_forward = s3_no_head & (abs_fb < adaptive_thr)
    s3b_right   = s3_no_head & (~s3b_forward) & (head_offset_fallback > 0)
    s3b_left    = s3_no_head & (~s3b_forward) & (head_offset_fallback <= 0)
    labels = torch.where(s3b_forward, torch.tensor(1, dtype=torch.int8, device=kp.device), labels)
    labels = torch.where(s3b_right,   torch.tensor(3, dtype=torch.int8, device=kp.device), labels)
    labels = torch.where(s3b_left,    torch.tensor(2, dtype=torch.int8, device=kp.device), labels)

    # Один синхронный трансфер N int8 значений → numpy → mapping
    labels_np = labels.cpu().numpy()
    return [_DIR_LABELS[i] for i in labels_np]


# ---------------------------------------------------------------------------
# YoloEngine
# ---------------------------------------------------------------------------

class YoloEngine:
    AVAILABLE_MODELS = AVAILABLE_MODELS
    DEFAULT_CONF     = _CONF
    KP_VIS_THRESHOLD = _KP_VIS

    def __init__(self, device: str | None = None) -> None:
        if device is None:
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cpu":
            self.device = "cpu"
        else:
            self.device = device if torch.cuda.is_available() else "cpu"

        self.use_fp16: bool = (self.device != "cpu") and _supports_fp16()
        self._device_str: str = self.device

        self._manager    = ModelManager()
        self.model       = None
        self._model_name = ""
        self._model_path = ""
        self._load_lock  = threading.Lock()

        self._current_batch_size: int = (
            BATCH_SIZE_GPU if self.device != "cpu" else BATCH_SIZE_CPU
        )

        # Кэш предкомпилированного torch device
        self._torch_device = torch.device(self.device)

        if self.device != "cpu":
            torch.backends.cudnn.benchmark        = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32       = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Load / reload
    # ------------------------------------------------------------------

    def load(
        self,
        model_path: str = DEFAULT_MODEL_NAME,
        *,
        on_status  : Optional[Callable[[str],   None]] = None,
        on_progress: Optional[Callable[[float], None]] = None,
        on_source  : Optional[Callable[[bool],  None]] = None,
        force      : bool = False,
    ) -> None:
        if not self._load_lock.acquire(blocking=False):
            return
        try:
            name = Path(model_path).name
            if not force and self.model is not None and self._model_name == name:
                _safe_cb(on_status, f"Модель {name} уже загружена.")
                _safe_cb(on_progress, 100.0)
                return
            self._load_impl(name, on_status, on_progress, on_source)
        finally:
            self._load_lock.release()

    def reload(
        self,
        model_path: str,
        *,
        on_status  : Optional[Callable[[str],   None]] = None,
        on_progress: Optional[Callable[[float], None]] = None,
        on_source  : Optional[Callable[[bool],  None]] = None,
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
        name       : str,
        on_status  : Optional[Callable[[str],   None]],
        on_progress: Optional[Callable[[float], None]],
        on_source  : Optional[Callable[[bool],  None]],
    ) -> None:
        try:
            local_path = self._manager.prepare(
                name,
                on_status=on_status,
                on_progress=lambda p: _safe_cb(on_progress, p * 0.60),
                on_source=on_source,
            )
        except RuntimeError as exc:
            _safe_cb(on_status, f"Ошибка: {exc}")
            return

        yolo_arg = str(local_path) if local_path.is_file() else name

        _safe_cb(on_status, f"Загружается {name}…")
        _safe_cb(on_progress, 65.0)

        if self.model is not None:
            self._release()

        try:
            self.model = YOLO(yolo_arg, task="pose")
            self.model.to(self.device)
        except Exception as exc:
            _safe_cb(on_status, f"Ошибка загрузки: {exc}")
            self.model = None
            return

        _safe_cb(on_progress, 80.0)

        if self.device != "cpu":
            if _supports_fp16():
                try:
                    self.model.model.half()
                    self.use_fp16 = True
                except Exception:
                    self.use_fp16 = False
            else:
                self.use_fp16 = False
        else:
            self.use_fp16 = False

        self._model_name = name
        self._model_path = str(local_path) if local_path.is_file() else yolo_arg

        _safe_cb(on_progress, 85.0)
        _safe_cb(on_status, f"Прогрев {name}…")

        if self.device != "cpu":
            self._warmup()

        _safe_cb(on_progress, 100.0)
        _safe_cb(on_status, f"{name} готова.")

    def _release(self) -> None:
        tmp, self.model = self.model, None
        del tmp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup(self, imgsz: int = IMGSZ, runs: int = 3) -> None:
        bs = self._current_batch_size
        dummy_single = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

        for attempt in range(_WARMUP_MAX_RETRIES):
            dummy_batch = [dummy_single] * bs
            try:
                with torch.inference_mode():
                    for _ in range(runs):
                        self.model.predict(
                            dummy_batch,
                            imgsz=imgsz,
                            verbose=False,
                            half=self.use_fp16,
                            conf=_CONF,
                            stream=False,
                            device=self.device,
                        )
                self._current_batch_size = bs
                return
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    return
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                bs = max(_BATCH_OOM_MIN, bs // _WARMUP_BATCH_DIVIDER)

        try:
            with torch.inference_mode():
                self.model.predict(
                    dummy_single,
                    imgsz=imgsz,
                    verbose=False,
                    half=self.use_fp16,
                    conf=_CONF,
                    stream=False,
                    device=self.device,
                )
            self._current_batch_size = 1
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Batch size
    # ------------------------------------------------------------------

    def get_batch_size(self) -> int:
        return self._current_batch_size

    # ------------------------------------------------------------------
    # Detect
    # ------------------------------------------------------------------

    def detect_batch(self, frames_batch: list) -> list[dict | None]:
        if not frames_batch:
            return []
        if self.model is None:
            raise RuntimeError("Модель не загружена. Вызовите load().")
        return self._run_batch(frames_batch)

    def _run_batch(self, frames: list) -> list[dict | None]:
        if not frames:
            return []

        bs = len(frames)

        while bs >= 1:
            try:
                with torch.inference_mode():
                    results = self.model.predict(
                        frames[:bs],
                        imgsz=_IMGSZ,
                        verbose=False,
                        half=self.use_fp16,
                        conf=_CONF,
                        stream=False,
                        device=self.device,
                    )
                parsed = self._parse_batch_results(results, bs)

                if bs < len(frames):
                    tail = self._run_batch(frames[bs:])
                    parsed.extend(tail)

                if bs < self._current_batch_size:
                    self._current_batch_size = bs

                return parsed

            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    return [None] * len(frames)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                new_bs = bs // 2
                if new_bs < 1:
                    break
                bs = new_bs

        return [self._detect_single(f) for f in frames]

    # ------------------------------------------------------------------
    # Parse results — оптимизировано:
    #   • один stack на все валидные позы
    #   • один cpu().numpy() трансфер
    #   • направления считаются ТЕМ ЖЕ stack-ом → нет лишнего копирования
    #   • bbox считается векторно для всех поз сразу
    # ------------------------------------------------------------------

    def _parse_batch_results(self, results, n: int) -> list[dict | None]:
        if not results:
            return [None] * n

        gpu_tensors:   list[torch.Tensor | None] = [None] * n
        orig_shapes:   list[tuple[int, int]]     = [(0, 0)] * n
        valid_indices: list[int]                 = []

        for i, res in enumerate(results):
            if res.orig_shape:
                orig_shapes[i] = (res.orig_shape[0], res.orig_shape[1])

            if (
                res.keypoints is not None
                and res.keypoints.data is not None
                and len(res.keypoints.data) > 0
            ):
                kp_tensor = res.keypoints.data[0]
                if kp_tensor.shape[0] >= 17:
                    gpu_tensors[i] = kp_tensor[:17]
                    valid_indices.append(i)

        output: list[dict | None] = [None] * n
        if not valid_indices:
            return output

        # ── Один stack для всех валидных поз ──────────────────────────
        try:
            stacked_gpu = torch.stack([gpu_tensors[i] for i in valid_indices])  # (M, 17, 3)
        except RuntimeError:
            # fallback — возможна разная форма
            for idx in valid_indices:
                kp = gpu_tensors[idx].cpu().numpy()
                d = self._classify_direction_gpu(gpu_tensors[idx])
                conf = float(kp[:, 2].mean())
                h, w = orig_shapes[idx]
                output[idx] = self._parse_single_result(kp, h, w, conf, d)
            return output

        # ── Направления — векторно ────────────────────────────────────
        try:
            dirs = _classify_directions_batch(stacked_gpu, KP_VIS_THRESHOLD)
        except Exception:
            dirs = [self._classify_direction_gpu(stacked_gpu[j]) for j in range(len(valid_indices))]

        # ── Один трансфер на CPU ──────────────────────────────────────
        cpu_batch = stacked_gpu.cpu().numpy()  # (M, 17, 3)

        # Векторный расчёт средних confidence
        confs = cpu_batch[:, :, 2].mean(axis=1)  # (M,)

        # ── Векторно строим словари ───────────────────────────────────
        # vis-маска для всех поз сразу
        vis_mask_all = cpu_batch[:, :, 2] >= KP_VIS_THRESHOLD       # (M, 17)
        n_visible_all = vis_mask_all.sum(axis=1)                    # (M,)

        for j, idx in enumerate(valid_indices):
            kp = cpu_batch[j]
            n_vis = int(n_visible_all[j])
            if n_vis < _PARSE_MIN_VISIBLE_KP:
                continue

            vis_mask = vis_mask_all[j]
            visible_xy = kp[vis_mask, :2]
            min_xy = visible_xy.min(axis=0)
            max_xy = visible_xy.max(axis=0)

            orig_h, orig_w = orig_shapes[idx]
            output[idx] = {
                "keypoints" : kp,
                "confidence": float(confs[j]),
                "bbox"      : [
                    float(min_xy[0]), float(min_xy[1]),
                    float(max_xy[0]), float(max_xy[1]),
                ],
                "direction" : dirs[j],
                "orig_w"    : int(orig_w),
                "orig_h"    : int(orig_h),
                "scale"     : 1.0,
                "anchor_y"  : 0.0,
            }

        return output

    def _detect_single(self, frame) -> dict | None:
        if not isinstance(frame, np.ndarray):
            return None
        try:
            with torch.inference_mode():
                results = self.model.predict(
                    frame,
                    imgsz=_IMGSZ,
                    verbose=False,
                    half=self.use_fp16,
                    conf=_CONF,
                    stream=False,
                    device=self.device,
                )
            if results and results[0].keypoints is not None:
                res = results[0]
                if len(res.keypoints.data) > 0:
                    kp_gpu    = res.keypoints.data[0][:17]
                    direction = self._classify_direction_gpu(kp_gpu)
                    kp        = kp_gpu.cpu().numpy()
                    h, w      = res.orig_shape
                    conf      = float(kp[:, 2].mean())
                    return self._parse_single_result(kp, h, w, conf, direction)
        except Exception:
            pass
        return None

    def _parse_single_result(
        self,
        kp       : np.ndarray,
        orig_h   : int,
        orig_w   : int,
        conf     : float,
        direction: str,
    ) -> dict | None:
        if kp is None or kp.shape[0] < 17:
            return None

        vis_mask = kp[:, 2] >= KP_VIS_THRESHOLD
        if vis_mask.sum() < _PARSE_MIN_VISIBLE_KP:
            return None

        visible = kp[vis_mask]
        min_xy  = visible[:, :2].min(axis=0)
        max_xy  = visible[:, :2].max(axis=0)
        bbox    = [
            float(min_xy[0]), float(min_xy[1]),
            float(max_xy[0]), float(max_xy[1]),
        ]

        return {
            "keypoints" : kp,
            "confidence": conf,
            "bbox"      : bbox,
            "direction" : direction,
            "orig_w"    : int(orig_w),
            "orig_h"    : int(orig_h),
            "scale"     : 1.0,
            "anchor_y"  : 0.0,
        }

    # ------------------------------------------------------------------
    # Direction (single, GPU) — fallback
    # ------------------------------------------------------------------

    def _classify_direction_gpu(self, kp17_gpu: torch.Tensor) -> str:
        # Используем батчевую функцию с N=1 — DRY и стабильно
        if kp17_gpu.dim() == 2:
            kp17_gpu = kp17_gpu.unsqueeze(0)
        return _classify_directions_batch(kp17_gpu, KP_VIS_THRESHOLD)[0]

    def classify_direction(self, kp17: np.ndarray) -> str:
        kp_gpu = torch.from_numpy(kp17).float().to(self.device)
        return self._classify_direction_gpu(kp_gpu)

    # ------------------------------------------------------------------
    # Info / utils
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict:
        return {
            "name"       : self._model_name,
            "path"       : self._model_path,
            "device"     : self.device,
            "device_idx" : 0,
            "fp16"       : self.use_fp16,
            "is_loaded"  : self.model is not None,
            "is_local"   : (
                self._manager.is_local(self._model_name)
                if self._model_name else False
            ),
            "imgsz"      : _IMGSZ,
            "conf"       : _CONF,
            "kp_vis"     : _KP_VIS,
            "avg_fps"    : 0.0,
            "frame_cache": 0,
            "dir_cache"  : 0,
            "batch_size" : self._current_batch_size,
            "fp16_supported": _supports_fp16(),
        }

    def list_local_models(self) -> list[str]:
        return self._manager.list_local()

    def warmup_video(self, video_path: str, frames_count: int = 100) -> float:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return -1.0

        frames: list = []
        for _ in range(frames_count):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            return -1.0

        t0 = time.perf_counter()
        bs = self.get_batch_size()

        for i in range(0, len(frames), bs):
            batch = frames[i: i + bs]
            if not batch:
                break
            try:
                self.detect_batch(batch)
            except Exception:
                pass

        return time.perf_counter() - t0

    def cleanup(self) -> None:
        self._release()
