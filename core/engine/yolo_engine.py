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

# ── Константы ──────────────────────────────────────────────────────────────────
try:
    from utils.constants import (
        YOLO_CONF               as _CONF,
        YOLO_IMGSZ              as _IMGSZ,
        KEYPOINT_CONF_THRESHOLD as _KP_VIS,
    )
except ImportError:
    _CONF           = 0.25
    _IMGSZ          = 848
    _KP_VIS         = 0.30

# ── ЖЁРСТКИЕ константы для производительности ───────────────────────────────────
BATCH_SIZE_GPU = 96   # Фиксированный размер батча для GPU
BATCH_SIZE_CPU = 16   # Фиксированный размер батча для CPU


class YoloEngine:
    """
    Обёртка над YOLOv8/YOLO11/YOLO26-pose с поддержкой FP16 / FP32,
    асинхронного предзагрузчика и фиксированного батч-размера.
    
    Основные улучшения:
    - Асинхронный префетч кадров (отдельный поток)
    - FP16 по умолчанию для GPU (без проверок)
    - Убраны кэши (_frame_cache, _dir_cache, _fps_history)
    - Жёсткий батч-размер: 64 для GPU, 16 для CPU
    - Прямой stream=True от YOLO
    - Векторизованный парсинг batch результатов
    - Метод warmup_video() для предпрогрева на 100 кадрах
    """

    AVAILABLE_MODELS = AVAILABLE_MODELS
    DEFAULT_CONF     = _CONF
    KP_VIS_THRESHOLD = _KP_VIS

    def __init__(self, device: str | None = None) -> None:
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cpu":
            self.device = "cpu"
        else:
            self.device = device if torch.cuda.is_available() else "cpu"

        # FP16 по умолчанию для GPU (без проверок на каждом вызове)
        self.use_fp16: bool = (self.device != "cpu")
        self._device_str: str = self.device

        self._manager = ModelManager()
        self.model = None
        self._model_name = ""
        self._model_path = ""

        self._load_lock = threading.Lock()

        # Асинхронный префетчер
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_queue: list = []
        self._prefetch_lock = threading.Lock()
        self._prefetch_stop = threading.Event()

    # ─────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────────
    # Асинхронный префетчер
    # ─────────────────────────────────────────────────────────────

    def _prefetch_worker(self, video_path: str, batch_size: int) -> None:
        """Поток для предзагрузки кадров из видео."""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            while not self._prefetch_stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                if len(frames) >= batch_size:
                    with self._prefetch_lock:
                        self._prefetch_queue.append(frames)
                    frames = []
            if frames:
                with self._prefetch_lock:
                    self._prefetch_queue.append(frames)
            cap.release()
        except Exception:
            pass

    def start_prefetch(self, video_path: str, batch_size: int = BATCH_SIZE_GPU) -> None:
        """Запустить асинхронный предзагрузчик кадров."""
        try:
            import cv2
        except ImportError:
            return
        self._prefetch_stop.clear()
        self._prefetch_queue.clear()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(video_path, batch_size),
            daemon=True
        )
        self._prefetch_thread.start()

    def stop_prefetch(self) -> None:
        """Остановить асинхронный предзагрузчик."""
        self._prefetch_stop.set()
        if self._prefetch_thread:
            self._prefetch_thread.join(timeout=1.0)
            self._prefetch_thread = None

    def get_prefetched_batch(self) -> list | None:
        """Получить подготовленный батч кадров."""
        with self._prefetch_lock:
            if self._prefetch_queue:
                return self._prefetch_queue.pop(0)
        return None

    # ─────────────────────────────────────────────────────────────
    # Загрузка модели
    # ─────────────────────────────────────────────────────────────

    def load(
        self,
        model_path: str = DEFAULT_MODEL_NAME,
        *,
        on_status: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[float], None]] = None,
        on_source: Optional[Callable[[bool], None]] = None,
        force: bool = False,
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
        on_status: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[float], None]] = None,
        on_source: Optional[Callable[[bool], None]] = None,
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
        name: str,
        on_status: Optional[Callable[[str], None]],
        on_progress: Optional[Callable[[float], None]],
        on_source: Optional[Callable[[bool], None]],
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
            self.model = YOLO(yolo_arg, task='pose')
            self.model.to(self.device)
            if self.device == 'cuda':
                self.use_fp16 = True
                print(f"[YoloEngine] FP16: {self.use_fp16} (принудительно)")
        except Exception as exc:
            _safe_cb(on_status, f"Ошибка загрузки: {exc}")
            self.model = None
            return

        _safe_cb(on_progress, 80.0)

        # FP16 по умолчанию для GPU (без проверок)
        self.use_fp16 = (self.device != "cpu")

        self._model_name = name
        self._model_path = str(local_path) if local_path.is_file() else yolo_arg

        _safe_cb(on_progress, 85.0)
        _safe_cb(on_status, f"Прогрев {name}…")

        if self.device != "cpu":
            self._warmup()

        _safe_cb(on_progress, 100.0)
        _safe_cb(on_status, f"{name} готова.")

    def _release(self) -> None:
        tmp = self.model
        self.model = None
        del tmp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _warmup(self, imgsz: int = _IMGSZ, runs: int = 3) -> None:
        """Прогрев модели на GPU."""
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        for _ in range(runs):
            self.model.predict(
                dummy,
                imgsz=imgsz,
                verbose=False,
                half=self.use_fp16,
                conf=_CONF,
                stream=False,
                device=self.device,
            )

    # ─────────────────────────────────────────────────────────────
    # Инференс
    # ─────────────────────────────────────────────────────────────

    def get_batch_size(self) -> int:
        """Получить фиксированный размер батча (64 для GPU, 16 для CPU)."""
        return BATCH_SIZE_GPU if self.device != "cpu" else BATCH_SIZE_CPU

    def detect_batch(self, frames_batch: list) -> list[dict | None]:
        import time
        t_yolo_start = time.perf_counter()
        """
        Детектировать позы в батче кадров.
        """
        if not frames_batch:
            return []

        if self.model is None:
            raise RuntimeError("Модель не загружена. Вызовите load().")

        try:
            # Batch inference без stream — весь батч за один вызов
            results = self.model.predict(
                frames_batch,
                imgsz=_IMGSZ,
                verbose=False,
                half=self.use_fp16,
                conf=_CONF,
                stream=False,  # Batch inference
                device=self.device,
            )

            # Векторизованный парсинг
            batch_poses = self._parse_batch_results(results)
            t_yolo_end = time.perf_counter()
            dt = t_yolo_end - t_yolo_start
            if dt > 1.0:
                print(f"[YOLO] Батч {len(frames_batch)} кадров: {dt:.2f}s")

            return batch_poses

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                # OOM только при реальной ошибке
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return [self._detect_single(f) for f in frames_batch]
            return [None] * len(frames_batch)

        except Exception:
            return [None] * len(frames_batch)

    def _parse_batch_results(self, results) -> list[dict | None]:
        """
        Разобрать результаты YOLO → список dict или None.
        Векторизованная обработка через numpy — один batch GPU→CPU.
        """
        if not results:
            return []
        
        # Собрать все keypoints в один numpy массив за один раз
        # res.keypoints.data — это torch.Tensor (N, 17, 3) или []
        all_kp = []
        all_orig_shapes = []
        all_conf = []
        
        for res in results:
            if res.keypoints is None or len(res.keypoints.data) == 0:
                all_kp.append(None)
                all_orig_shapes.append((0, 0))
                all_conf.append(0.0)
            else:
                # Один вызов .cpu().numpy() на весь батч
                kp = res.keypoints.data[0].cpu().numpy()
                all_kp.append(kp)
                all_orig_shapes.append(res.orig_shape)
                all_conf.append(float(np.mean(kp[:, 2])))
        
        # Сформировать результаты
        return [
            self._parse_single_result(kp, orig_h, orig_w, conf)
            for kp, (orig_h, orig_w), conf in zip(all_kp, all_orig_shapes, all_conf)
        ]

    def _detect_single(self, frame) -> dict | None:
        """Обработка одного кадра (fallback при OOM)."""
        if not isinstance(frame, np.ndarray):
            return None
        try:
            results = self.model.predict(
                frame,
                imgsz=_IMGSZ,
                verbose=False,
                half=False,
                conf=_CONF,
                stream=False,
                device=self.device,
            )
            if results:
                return self._parse_single_result(results[0])
        except Exception:
            pass
        return None

    def _parse_single_result(self, kp: np.ndarray, orig_h: int, orig_w: int, conf: float) -> dict | None:
        """
        Разобрать один результат YOLO → структурированный dict или None.
        Векторизованная версия для batch обработки.
        
        Parameters
        ----------
        kp : np.ndarray
            Keypoints array (N, 3)
        orig_h : int
            Original height
        orig_w : int
            Original width
        conf : float
            Confidence score
        """
        if kp is None or kp.shape[0] < 17:
            return None

        kp17 = kp[:17]  # гарантированно 17 точек COCO

        # Фильтр: нужно хотя бы 5 видимых точек
        visible = kp17[kp17[:, 2] >= self.KP_VIS_THRESHOLD]
        if len(visible) < 5:
            return None

        # bbox по видимым точкам
        if len(visible) > 0:
            min_x = float(np.min(visible[:, 0]))
            min_y = float(np.min(visible[:, 1]))
            max_x = float(np.max(visible[:, 0]))
            max_y = float(np.max(visible[:, 1]))
            bbox = [min_x, min_y, max_x, max_y]
        else:
            bbox = [0.0, 0.0, float(orig_w), float(orig_h)]

        direction = self._classify_direction(kp17)

        return {
            "keypoints": kp17,  # numpy array (17, 3)
            "confidence": conf,
            "bbox": bbox,
            "direction": direction,
            "orig_w": int(orig_w),
            "orig_h": int(orig_h),
            "scale": 1.0,
            "anchor_y": 0.0,
        }

    # ─────────────────────────────────────────────────────────────
    # Классификация направления
    # ─────────────────────────────────────────────────────────────

    def _classify_direction(self, kp17: np.ndarray) -> str:
        """
        Определить направление тела по ключевым точкам COCO.
        """
        NOSE = 0
        L_EAR = 3
        R_EAR = 4
        L_SHOULDER = 5
        R_SHOULDER = 6

        VIS = self.KP_VIS_THRESHOLD

        try:
            conf_ls = kp17[L_SHOULDER, 2]
            conf_rs = kp17[R_SHOULDER, 2]
            conf_nose = kp17[NOSE, 2]
            conf_le = kp17[L_EAR, 2]
            conf_re = kp17[R_EAR, 2]

            # Шаг 1: оба плеча не видны
            if conf_ls < VIS and conf_rs < VIS:
                if conf_le >= VIS and conf_re < VIS:
                    return "right"
                if conf_re >= VIS and conf_le < VIS:
                    return "left"
                return "unknown"

            # Шаг 2: только одно плечо видно
            if conf_ls >= VIS and conf_rs < VIS - 0.05:
                return "right"
            if conf_rs >= VIS and conf_ls < VIS - 0.05:
                return "left"

            # Шаг 3: оба плеча видны
            shoulder_cx = (kp17[L_SHOULDER, 0] + kp17[R_SHOULDER, 0]) / 2.0
            shoulder_w = abs(kp17[L_SHOULDER, 0] - kp17[R_SHOULDER, 0]) + 1e-5

            if conf_nose >= VIS:
                head_offset = (kp17[NOSE, 0] - shoulder_cx) / shoulder_w
            else:
                if conf_le >= VIS and conf_re >= VIS:
                    ear_cx = (kp17[L_EAR, 0] + kp17[R_EAR, 0]) / 2.0
                    head_offset = (ear_cx - shoulder_cx) / shoulder_w
                elif conf_le >= VIS:
                    head_offset = (kp17[L_EAR, 0] - shoulder_cx) / shoulder_w
                elif conf_re >= VIS:
                    head_offset = (kp17[R_EAR, 0] - shoulder_cx) / shoulder_w
                else:
                    head_offset = (conf_rs - conf_ls) * 0.15

            # Адаптивный порог
            adaptive_threshold = max(0.06, min(0.15, 0.3 / (shoulder_w / 50.0 + 1e-5)))
            if abs(head_offset) < adaptive_threshold:
                return "forward"
            return "right" if head_offset > 0 else "left"

        except Exception:
            return "unknown"

    # публичный алиас
    def classify_direction(self, kp17: np.ndarray) -> str:
        return self._classify_direction(kp17)

    # ─────────────────────────────────────────────────────────────
    # Info
    # ─────────────────────────────────────────────────────────────

    def get_model_info(self) -> dict:
        return {
            "name": self._model_name,
            "path": self._model_path,
            "device": self.device,
            "device_idx": 0,
            "fp16": self.use_fp16,
            "is_loaded": self.model is not None,
            "is_local": (
                self._manager.is_local(self._model_name)
                if self._model_name else False
            ),
            "imgsz": _IMGSZ,
            "conf": _CONF,
            "kp_vis": _KP_VIS,
            "avg_fps": 0.0,
            "frame_cache": 0,
            "dir_cache": 0,
        }

    def list_local_models(self) -> list[str]:
        return self._manager.list_local()

    # ─────────────────────────────────────────────────────────────
    # Warmup видео
    # ─────────────────────────────────────────────────────────────

    def warmup_video(self, video_path: str, frames_count: int = 100) -> float:
        """
        Предпрогрев модели на первых N кадрах видео.
        
        Parameters
        ----------
        video_path : str
            Путь к видеофайлу
        frames_count : int
            Количество кадров для прогрева (по умолчанию 100)
            
        Returns
        -------
        float : время прогрева в секундах
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return -1.0
            
        frames = []
        for _ in range(frames_count):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if not frames:
            return -1.0
            
        start_time = time.time()
        
        # Прогрев на батче
        batch_size = self.get_batch_size()
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            if not batch:
                break
            try:
                self.detect_batch(batch)
            except Exception:
                pass
        
        elapsed = time.time() - start_time
        # OOM очистка только при реальной ошибке, не при каждом прогреве
        return elapsed

    # ─────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Очистка ресурсов."""
        self.stop_prefetch()
        self._release()