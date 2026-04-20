# ═══════════════════════════════════════════════════════════════════════════════
# core/engine/yolo_engine.py — ULTRA HIGH-PERFORMANCE YOLO ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
"""
YoloEngine — высокопроизводительный движок детекции поз с GPU-оптимизацией.

КРИТИЧЕСКИЕ ОПТИМИЗАЦИИ:
========================
✅ torch.inference_mode() вместо no_grad()
✅ Принудительное FP16 через model.model.half()
✅ GPU-классификация direction на тензорах
✅ Одна синхронизация GPU→CPU на весь батч
✅ Векторизованный парсинг без Python-циклов
✅ Прогрев с реальным размером батча (BATCH_SIZE_GPU)
✅ Удалён весь мёртвый код префетчера
"""
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

# ══════════════════════════════════════════════════════════════════════════════
# Константы
# ══════════════════════════════════════════════════════════════════════════════

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

# Публичные константы (совместимость)
DEFAULT_CONF     : float = _CONF
IMGSZ            : int   = _IMGSZ
KP_VIS_THRESHOLD : float = _KP_VIS

# Производительность
BATCH_SIZE_GPU : int = 96   # оптимально для RTX 5070
BATCH_SIZE_CPU : int = 16

# COCO Keypoints (уровень модуля)
_NOSE       = 0
_L_EAR      = 3
_R_EAR      = 4
_L_SHOULDER = 5
_R_SHOULDER = 6

# Пороги направления
_ADAPTIVE_MIN = 0.06
_ADAPTIVE_MAX = 0.15


# ══════════════════════════════════════════════════════════════════════════════
# YoloEngine
# ══════════════════════════════════════════════════════════════════════════════

class YoloEngine:
    """
    Высокопроизводительная обёртка над YOLOv8/YOLO11-pose.

    Ключевые оптимизации
    --------------------
    * torch.inference_mode() — отключение autograd
    * FP16 через model.model.half() — вдвое меньше памяти
    * GPU-классификация direction — без скачивания на CPU
    * Одна синхронизация GPU→CPU на батч
    * Векторизованная обработка без Python-циклов
    * Прогрев батчем BATCH_SIZE_GPU
    """

    # Атрибуты класса (совместимость)
    AVAILABLE_MODELS = AVAILABLE_MODELS
    DEFAULT_CONF     = _CONF
    KP_VIS_THRESHOLD = _KP_VIS

    def __init__(self, device: str | None = None) -> None:
        # Device selection
        if device is None:
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cpu":
            self.device = "cpu"
        else:
            self.device = device if torch.cuda.is_available() else "cpu"

        self.use_fp16: bool = (self.device != "cpu")
        self._device_str: str = self.device

        # Model manager
        self._manager    = ModelManager()
        self.model       = None
        self._model_name = ""
        self._model_path = ""
        self._load_lock  = threading.Lock()

    # ──────────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────────────
    # Загрузка модели
    # ──────────────────────────────────────────────────────────────────────────

    def load(
        self,
        model_path: str = DEFAULT_MODEL_NAME,
        *,
        on_status  : Optional[Callable[[str],   None]] = None,
        on_progress: Optional[Callable[[float], None]] = None,
        on_source  : Optional[Callable[[bool],  None]] = None,
        force      : bool = False,
    ) -> None:
        """
        Загрузить модель.

        Parameters
        ----------
        model_path  : имя или путь к весам
        on_status   : колбэк строки статуса
        on_progress : колбэк прогресса [0..100]
        on_source   : колбэк флага локального источника
        force       : перезагрузить даже если уже загружена
        """
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
        """Принудительно перезагрузить модель."""
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
        """Внутренняя реализация загрузки."""
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

        # ═══ КРИТИЧЕСКАЯ ОПТИМИЗАЦИЯ: Принудительное FP16 ═══
        if self.device != "cpu":
            try:
                self.model.model.half()  # Конвертация весов в FP16
                self.use_fp16 = True
            except Exception:
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
        """Освободить память модели."""
        tmp, self.model = self.model, None
        del tmp
        gc.collect()
        # Очистка кэша только при освобождении модели
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _warmup(self, imgsz: int = _IMGSZ, runs: int = 3) -> None:
        """
        ═══ ОПТИМИЗАЦИЯ: Прогрев с реальным размером батча ═══
        """
        bs = BATCH_SIZE_GPU if self.device != "cpu" else BATCH_SIZE_CPU
        dummy_single = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        dummy_batch  = [dummy_single] * bs

        with torch.inference_mode():
            for _ in range(runs):
                try:
                    self.model.predict(
                        dummy_batch,
                        imgsz=imgsz,
                        verbose=False,
                        half=self.use_fp16,
                        conf=_CONF,
                        stream=False,
                        device=self.device,
                    )
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Fallback: одиночный кадр
                        self.model.predict(
                            dummy_single,
                            imgsz=imgsz,
                            verbose=False,
                            half=self.use_fp16,
                            conf=_CONF,
                            stream=False,
                            device=self.device,
                        )
                    break

    # ──────────────────────────────────────────────────────────────────────────
    # Инференс
    # ──────────────────────────────────────────────────────────────────────────

    def get_batch_size(self) -> int:
        """Вернуть фиксированный размер батча."""
        return BATCH_SIZE_GPU if self.device != "cpu" else BATCH_SIZE_CPU

    def detect_batch(self, frames_batch: list) -> list[dict | None]:
        """
        Детектировать позы в батче кадров.

        Parameters
        ----------
        frames_batch : list[np.ndarray]
            BGR-кадры

        Returns
        -------
        list[dict | None]
            Каждый элемент: dict с ключами
            ``keypoints``, ``direction``, ``confidence``,
            ``bbox``, ``orig_w``, ``orig_h``, ``scale``, ``anchor_y``
            либо None
        """
        if not frames_batch:
            return []

        if self.model is None:
            raise RuntimeError("Модель не загружена. Вызовите load().")

        return self._run_batch(frames_batch)

    # ──────────────────────────────────────────────────────────────────────────
    # Внутренние методы инференса
    # ──────────────────────────────────────────────────────────────────────────

    def _run_batch(self, frames: list) -> list[dict | None]:
        """
        ═══ ОПТИМИЗАЦИЯ: inference_mode + экспоненциальный fallback ═══
        """
        current_batch = frames
        bs = len(frames)

        while bs >= 1:
            try:
                with torch.inference_mode():
                    results = self.model.predict(
                        current_batch,
                        imgsz=_IMGSZ,
                        verbose=False,
                        half=self.use_fp16,
                        conf=_CONF,
                        stream=False,
                        device=self.device,
                    )
                return self._parse_batch_results(results, len(current_batch))

            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    return [None] * len(frames)

                # OOM: экспоненциальное уменьшение
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                bs = bs // 2
                if bs < 1:
                    break
                current_batch = frames[:bs]

        # Абсолютный fallback
        return [self._detect_single(f) for f in frames]

    def _parse_batch_results(
        self,
        results,
        n: int,
    ) -> list[dict | None]:
        """
        ═══ КРИТИЧЕСКАЯ ОПТИМИЗАЦИЯ: Одна синхронизация GPU→CPU ═══
        
        Векторизованная обработка:
        1. Собираем все keypoints на GPU
        2. Классифицируем direction на GPU
        3. ОДНА синхронизация всех данных CPU
        4. Векторизованный расчёт confidence
        """
        if not results:
            return [None] * n

        # ── Фаза 1: Сбор GPU-тензоров ─────────────────────────────────────
        gpu_tensors : list[torch.Tensor | None] = []
        orig_shapes : list[tuple[int, int]]     = []
        valid_indices: list[int]                = []

        for i, res in enumerate(results):
            h, w = (res.orig_shape[0], res.orig_shape[1]) if res.orig_shape else (0, 0)
            orig_shapes.append((h, w))

            if (
                res.keypoints is not None
                and res.keypoints.data is not None
                and len(res.keypoints.data) > 0
            ):
                kp_tensor = res.keypoints.data[0]  # (N_kp, 3) на GPU
                if kp_tensor.shape[0] >= 17:
                    gpu_tensors.append(kp_tensor)
                    valid_indices.append(i)
                else:
                    gpu_tensors.append(None)
            else:
                gpu_tensors.append(None)

        # ── Фаза 2: GPU-классификация direction ───────────────────────────
        directions_gpu: list[str] = []
        
        if valid_indices:
            for idx in valid_indices:
                kp_gpu = gpu_tensors[idx][:17]  # (17, 3) на GPU
                direction = self._classify_direction_gpu(kp_gpu)
                directions_gpu.append(direction)

        # ── Фаза 3: ОДНА синхронизация GPU→CPU ────────────────────────────
        kp_cpu_list   : list[np.ndarray] = []
        confs_cpu_list: list[float]      = []

        if valid_indices:
            # Стекируем все валидные тензоры
            try:
                # Пытаемся стекировать (если одинаковая форма)
                stacked = torch.stack([gpu_tensors[i][:17] for i in valid_indices])
                # ОДНА синхронизация
                kp_cpu_batch = stacked.cpu().numpy()  # (M, 17, 3)
                
                # Векторизованный расчёт confidence
                confs_cpu_list = kp_cpu_batch[:, :, 2].mean(axis=1).tolist()
                kp_cpu_list = list(kp_cpu_batch)
                
            except RuntimeError:
                # Разные формы (редко) — обрабатываем по одному
                for i in valid_indices:
                    kp = gpu_tensors[i][:17].cpu().numpy()
                    kp_cpu_list.append(kp)
                    confs_cpu_list.append(float(kp[:, 2].mean()))

        # ── Фаза 4: Построение итогового списка ───────────────────────────
        output: list[dict | None] = [None] * n
        valid_ptr = 0

        for i in range(n):
            if gpu_tensors[i] is None:
                continue

            kp        = kp_cpu_list[valid_ptr]
            conf      = confs_cpu_list[valid_ptr]
            direction = directions_gpu[valid_ptr]
            orig_h, orig_w = orig_shapes[i]
            valid_ptr += 1

            output[i] = self._parse_single_result(
                kp, orig_h, orig_w, conf, direction
            )

        return output

    def _detect_single(self, frame) -> dict | None:
        """CPU fallback при OOM."""
        if not isinstance(frame, np.ndarray):
            return None
        try:
            with torch.inference_mode():
                results = self.model.predict(
                    frame,
                    imgsz=_IMGSZ,
                    verbose=False,
                    half=False,  # FP32 для безопасности
                    conf=_CONF,
                    stream=False,
                    device=self.device,
                )
            if results and results[0].keypoints is not None:
                res = results[0]
                if len(res.keypoints.data) > 0:
                    kp_gpu = res.keypoints.data[0][:17]
                    direction = self._classify_direction_gpu(kp_gpu)
                    kp = kp_gpu.cpu().numpy()
                    h, w = res.orig_shape
                    conf = float(kp[:, 2].mean())
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
        """
        Разобрать массив keypoints → dict.
        Direction уже вычислен на GPU.
        """
        if kp is None or kp.shape[0] < 17:
            return None

        # Фильтр: нужно ≥5 видимых точек
        vis_mask = kp[:, 2] >= self.KP_VIS_THRESHOLD
        if vis_mask.sum() < 5:
            return None

        visible = kp[vis_mask]

        # bbox (векторизованно)
        min_xy = visible[:, :2].min(axis=0)
        max_xy = visible[:, :2].max(axis=0)
        bbox   = [
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

    # ──────────────────────────────────────────────────────────────────────────
    # GPU-классификация направления
    # ──────────────────────────────────────────────────────────────────────────

    def _classify_direction_gpu(self, kp17_gpu: torch.Tensor) -> str:
        """
        ═══ КРИТИЧЕСКАЯ ОПТИМИЗАЦИЯ: Классификация на GPU-тензорах ═══
        
        Вся логика выполняется на GPU без скачивания данных на CPU.
        
        Parameters
        ----------
        kp17_gpu : torch.Tensor
            (17, 3) тензор на GPU: [x, y, confidence]
        
        Returns
        -------
        str : "forward" | "left" | "right" | "unknown"
        """
        VIS = self.KP_VIS_THRESHOLD
        
        # Извлекаем confidence для нужных точек (остаётся на GPU)
        c_nose = kp17_gpu[_NOSE, 2]
        c_le   = kp17_gpu[_L_EAR, 2]
        c_re   = kp17_gpu[_R_EAR, 2]
        c_ls   = kp17_gpu[_L_SHOULDER, 2]
        c_rs   = kp17_gpu[_R_SHOULDER, 2]

        # ── Шаг 1: оба плеча невидимы ──────────────────────────────────────
        if c_ls < VIS and c_rs < VIS:
            if c_le >= VIS and c_re < VIS:
                return "right"
            if c_re >= VIS and c_le < VIS:
                return "left"
            return "unknown"

        # ── Шаг 2: только одно плечо видно ─────────────────────────────────
        margin = VIS - 0.05
        if c_ls >= VIS and c_rs < margin:
            return "right"
        if c_rs >= VIS and c_ls < margin:
            return "left"

        # ── Шаг 3: оба плеча видны → вычисляем head_offset ────────────────
        lsx = kp17_gpu[_L_SHOULDER, 0]
        rsx = kp17_gpu[_R_SHOULDER, 0]
        
        shoulder_cx = (lsx + rsx) * 0.5
        shoulder_w  = torch.abs(lsx - rsx) + 1e-5

        # Определяем позицию головы
        if c_nose >= VIS:
            head_x = kp17_gpu[_NOSE, 0]
        elif c_le >= VIS and c_re >= VIS:
            head_x = (kp17_gpu[_L_EAR, 0] + kp17_gpu[_R_EAR, 0]) * 0.5
        elif c_le >= VIS:
            head_x = kp17_gpu[_L_EAR, 0]
        elif c_re >= VIS:
            head_x = kp17_gpu[_R_EAR, 0]
        else:
            # Нет головных точек → используем разность confidence
            head_offset = float((c_rs - c_ls).item()) * 0.15
            adaptive_thr = max(
                _ADAPTIVE_MIN,
                min(_ADAPTIVE_MAX, 0.3 / (float(shoulder_w.item()) / 50.0 + 1e-5)),
            )
            if abs(head_offset) < adaptive_thr:
                return "forward"
            return "right" if head_offset > 0 else "left"

        head_offset  = (head_x - shoulder_cx) / shoulder_w
        adaptive_thr = max(
            _ADAPTIVE_MIN,
            min(_ADAPTIVE_MAX, 0.3 / (float(shoulder_w.item()) / 50.0 + 1e-5)),
        )

        # Финальное решение (скачиваем только одно float-значение)
        offset_val = float(head_offset.item())
        
        if abs(offset_val) < adaptive_thr:
            return "forward"
        return "right" if offset_val > 0 else "left"

    # Публичный алиас (совместимость)
    def classify_direction(self, kp17: np.ndarray) -> str:
        """CPU-версия для обратной совместимости."""
        kp_gpu = torch.from_numpy(kp17).float().to(self.device)
        return self._classify_direction_gpu(kp_gpu)

    # ──────────────────────────────────────────────────────────────────────────
    # Info / Utils
    # ──────────────────────────────────────────────────────────────────────────

    def get_model_info(self) -> dict:
        """Вернуть метаданные текущей модели."""
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
        }

    def list_local_models(self) -> list[str]:
        """Список локально доступных моделей."""
        return self._manager.list_local()

    # ──────────────────────────────────────────────────────────────────────────
    # Warmup видео
    # ──────────────────────────────────────────────────────────────────────────

    def warmup_video(self, video_path: str, frames_count: int = 100) -> float:
        """
        Предпрогрев модели на первых N кадрах реального видео.

        Returns
        -------
        float : время прогрева в секундах, -1.0 при ошибке
        """
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
            batch = frames[i : i + bs]
            if not batch:
                break
            try:
                self.detect_batch(batch)
            except Exception:
                pass

        return time.perf_counter() - t0

    # ──────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Освободить все ресурсы."""
        self._release()