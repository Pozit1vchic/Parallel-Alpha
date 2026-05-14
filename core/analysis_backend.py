from __future__ import annotations

import gc
import hashlib
import logging
import os
import queue
import time
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable, Optional, Protocol

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from core.engine import YoloEngine
from core.matcher import (
    MotionMatcher,
    build_poses_tensor,
    is_pose_valid,
    preprocess_pose,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Status
# ---------------------------------------------------------------------------

class AnalysisStatus(Enum):
    IDLE             = auto()
    LOADING_MODEL    = auto()
    ANALYZING_VIDEO  = auto()
    EXTRACTING_POSES = auto()
    MATCHING         = auto()
    CLASSIFYING      = auto()
    DONE             = auto()
    STOPPED          = auto()
    ERROR            = auto()


STATUS_LABELS: dict[AnalysisStatus, str] = {
    AnalysisStatus.IDLE:             "Ожидание",
    AnalysisStatus.LOADING_MODEL:    "Загрузка модели…",
    AnalysisStatus.ANALYZING_VIDEO:  "Анализ видео…",
    AnalysisStatus.EXTRACTING_POSES: "Извлечение поз…",
    AnalysisStatus.MATCHING:         "Поиск совпадений…",
    AnalysisStatus.CLASSIFYING:      "Классификация движений…",
    AnalysisStatus.DONE:             "Готово",
    AnalysisStatus.STOPPED:          "Остановлено",
    AnalysisStatus.ERROR:            "Ошибка",
}


class SearchMode(Enum):
    MOTION_MATCH  = auto()
    PERSON_SEARCH = auto()


class PhotoMatcherProtocol(Protocol):
    def filter_poses_by_reference(
        self, frames_data: list[dict], threshold: float,
    ) -> list[dict]: ...

    def filter_matches(
        self, matches: list[dict], threshold: float,
    ) -> list[dict]: ...


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AnalysisProgress:
    percent:       float          = 0.0
    status:        str            = "Ожидание"
    status_code:   AnalysisStatus = AnalysisStatus.IDLE
    video_idx:     int            = 0
    video_count:   int            = 0
    current_frame: int            = 0
    total_frames:  int            = 0
    current_video: str            = ""
    eta_seconds:   float | None   = None


@dataclass
class VideoMeta:
    path:         str
    video_idx:    int
    fps:          float
    total_frames: int
    width:        int
    height:       int

    @property
    def duration(self) -> float:
        return self.total_frames / max(self.fps, 1.0)

    @property
    def basename(self) -> str:
        return os.path.basename(self.path)


@dataclass
class MotionGroup:
    label:     str
    direction: str
    sim_range: tuple[float, float]
    matches:   list[dict] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.matches)


@dataclass
class AnalysisResult:
    matches:       list[dict]          = field(default_factory=list)
    poses_meta:    list[dict]          = field(default_factory=list)
    poses_tensor:  torch.Tensor | None = None
    video_paths:   list[str]           = field(default_factory=list)
    video_metas:   list[VideoMeta]     = field(default_factory=list)
    motion_groups: list[MotionGroup]   = field(default_factory=list)
    stats:         dict[str, Any]      = field(default_factory=dict)
    mode:          SearchMode          = SearchMode.MOTION_MATCH
    stopped:       bool                = False
    error:         str | None          = None


# ---------------------------------------------------------------------------
# Constants (ALL PRESERVED)
# ---------------------------------------------------------------------------

QUALITY_FPS: dict[str, int] = {
    "Быстро": 8,
    "Средне": 15,
    "Макс":   30,
}

QUALITY_MAPPING: dict[str, str] = {
    "fast":    "Быстро",
    "medium":  "Средне",
    "maximum": "Макс",
    "быстро":  "Быстро",
    "средне":  "Средне",
    "максимум": "Макс",
    "макс":    "Макс",
}

DEFAULT_QUALITY      = "Средне"
PREVIEW_SIZE         = (320, 180)
PREVIEW_JPEG_Q       = 80
MAX_FRAME_SIDE       = 640
ETA_WINDOW           = 60
PREVIEW_CACHE_LIMIT  = 500
DEFAULT_BATCH_SIZE   = 128
DEFAULT_CHUNK_SIZE   = 5000

TARGET_WIDTH_HD      = 1280
TARGET_HEIGHT_HD     = 720
TARGET_WIDTH_FHD     = 1920
TARGET_HEIGHT_FHD    = 1080

YOLO_INPUT_SIZE      = 720

MAX_VIDEO_SIZE_GB    = 50
MAX_TOTAL_POSES      = 2_000_000

READER_QUEUE_SIZE    = 2048
PROGRESS_UPDATE_PCT  = 3

_VALID_MODES         = {SearchMode.MOTION_MATCH, SearchMode.PERSON_SEARCH}
_VALID_QUALITIES     = set(QUALITY_FPS.keys()) | set(QUALITY_MAPPING.keys())
_MAX_VIDEO_FILES     = 500
_MIN_THRESHOLD       = 0.0
_MAX_THRESHOLD       = 1.0
_MIN_FPS             = 1.0
_MAX_FPS             = 240.0
_MIN_SCENE_INTERVAL  = 0.0
_MAX_SCENE_INTERVAL  = 3600.0
_MAX_FRAME_DIM       = 16000
_SUPPORTED_EXTS      = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv",
    ".flv", ".webm", ".m4v", ".ts", ".mts",
    ".mpg", ".mpeg", ".3gp", ".mxf",
}

_4K_WIDTH            = 3840
_4K_HEIGHT           = 2160
_GPU_MEMORY_RESERVE  = 0.15
_CUDA_EMPTY_EVERY    = 8
_INCREMENTAL_TENSOR_CHUNK = 50_000
_STOP_CHECK_INTERVAL = 32
_READER_PUT_TIMEOUT  = 1.0
_MAX_4K_FPS_AUTO     = 8

# Pipeline tuning
_VIDEO_PREFETCH_QUEUE = 2          # сколько видео декодируется параллельно
_GC_EVERY_VIDEOS      = 4          # gc.collect раз в N видео


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _sanitize_video_paths(paths: Any) -> list[str]:
    if paths is None:
        return []
    if isinstance(paths, str):
        paths = [paths]
    if not isinstance(paths, (list, tuple)):
        try:
            paths = list(paths)
        except Exception:
            return []

    clean: list[str] = []
    seen: set[str] = set()

    for p in paths[:_MAX_VIDEO_FILES]:
        if p is None:
            continue
        try:
            p = str(p).strip()
        except Exception:
            continue
        if not p:
            continue

        ext = Path(p).suffix.lower()
        if ext not in _SUPPORTED_EXTS:
            logger.warning(f"Неподдерживаемый формат: {p} (ext={ext})")
            continue

        try:
            size_gb = os.path.getsize(p) / (1024 ** 3)
            if size_gb > MAX_VIDEO_SIZE_GB:
                logger.warning(f"Файл слишком большой ({size_gb:.1f} GB): {p}")
                continue
        except OSError:
            logger.warning(f"Файл не найден или недоступен: {p}")
            continue

        real = os.path.realpath(p)
        if real in seen:
            logger.debug(f"Дубликат видео пропущен: {p}")
            continue
        seen.add(real)
        clean.append(p)

    return clean


def _sanitize_settings(settings: Any) -> dict[str, Any]:
    if not isinstance(settings, dict):
        logger.warning("settings не является dict, используются значения по умолчанию")
        settings = {}

    result: dict[str, Any] = {}

    raw_quality = settings.get("quality", "medium")
    if not isinstance(raw_quality, str):
        raw_quality = "medium"
    raw_quality = raw_quality.strip().lower()
    if raw_quality not in _VALID_QUALITIES:
        logger.warning(f"Неизвестное качество '{raw_quality}', используется 'medium'")
        raw_quality = "medium"
    result["quality"] = raw_quality

    raw_threshold = settings.get("threshold", 70)
    try:
        raw_threshold = float(raw_threshold)
    except (TypeError, ValueError):
        raw_threshold = 70.0
    if not (0.0 <= raw_threshold <= 100.0):
        logger.warning(f"threshold вне диапазона: {raw_threshold}, clamp → [0, 100]")
        raw_threshold = _clamp(raw_threshold, 0.0, 100.0)
    result["threshold"] = raw_threshold

    raw_interval = settings.get("scene_interval", 3)
    try:
        raw_interval = float(raw_interval)
    except (TypeError, ValueError):
        raw_interval = 3.0
    if not (_MIN_SCENE_INTERVAL <= raw_interval <= _MAX_SCENE_INTERVAL):
        logger.warning(f"scene_interval вне диапазона: {raw_interval}, clamp")
        raw_interval = _clamp(raw_interval, _MIN_SCENE_INTERVAL, _MAX_SCENE_INTERVAL)
    result["scene_interval"] = raw_interval

    raw_mirror = settings.get("use_mirror", False)
    result["use_mirror"] = bool(raw_mirror)

    return result


def _sanitize_frames(frames: Any) -> list[np.ndarray]:
    if frames is None:
        return []
    if isinstance(frames, np.ndarray):
        frames = [frames]
    if not isinstance(frames, (list, tuple)):
        try:
            frames = list(frames)
        except Exception:
            return []

    clean: list[np.ndarray] = []
    for f in frames:
        if not isinstance(f, np.ndarray):
            continue
        if f.ndim != 3 or f.shape[2] not in (1, 3, 4):
            continue
        h, w = f.shape[:2]
        if h < 8 or w < 8:
            continue
        if h > _MAX_FRAME_DIM or w > _MAX_FRAME_DIM:
            continue
        if f.dtype != np.uint8:
            try:
                f = f.astype(np.uint8)
            except Exception:
                continue
        clean.append(f)
    return clean


def _get_resize_target(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return TARGET_WIDTH_HD, TARGET_HEIGHT_HD

    total_pixels = width * height

    if width >= _4K_WIDTH or height >= _4K_HEIGHT:
        scale = min(TARGET_WIDTH_HD / width, TARGET_HEIGHT_HD / height)
        new_w = int(width * scale) & ~1
        new_h = int(height * scale) & ~1
        return max(new_w, 2), max(new_h, 2)

    if total_pixels <= TARGET_WIDTH_HD * TARGET_HEIGHT_HD:
        return width, height

    if total_pixels <= TARGET_WIDTH_FHD * TARGET_HEIGHT_FHD:
        scale = min(TARGET_WIDTH_FHD / width, TARGET_HEIGHT_FHD / height)
        return int(width * scale), int(height * scale)

    scale = min(TARGET_WIDTH_HD / width, TARGET_HEIGHT_HD / height)
    new_w = int(width * scale) & ~1
    new_h = int(height * scale) & ~1
    return max(new_w, 2), max(new_h, 2)


def _fast_resize(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == target_w and h == target_h:
        return frame
    if w > target_w * 2 or h > target_h * 2:
        mid_w = min(w // 2, target_w * 2)
        mid_h = min(h // 2, target_h * 2)
        frame = cv2.resize(frame, (mid_w, mid_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _evict_preview_cache(cache_dir: Path, limit: int) -> None:
    """Сэмплированная очистка: проверяет только если файлов > limit*1.2."""
    try:
        if not cache_dir.exists():
            return
        # Дешёвая оценка количества (без stat()) — итерируем с ранним выходом
        n_estimate = 0
        for _ in cache_dir.iterdir():
            n_estimate += 1
            if n_estimate > int(limit * 1.2):
                break
        if n_estimate <= limit:
            return
        files = sorted(cache_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
        for old in files[: max(0, len(files) - limit)]:
            try:
                old.unlink()
            except OSError:
                pass
    except Exception as e:
        logger.warning(f"Ошибка очистки превью-кэша: {e}")


def _compute_file_hash(path: str) -> str:
    hasher = hashlib.md5()
    try:
        stat = os.stat(path)
        signature = f"{path}:{stat.st_size}:{stat.st_mtime}"
        hasher.update(signature.encode("utf-8"))
    except Exception:
        hasher.update(path.encode("utf-8"))
    return hasher.hexdigest()[:12]


def _is_4k(width: int, height: int) -> bool:
    return width >= _4K_WIDTH or height >= _4K_HEIGHT


def _get_free_gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        free, total = torch.cuda.mem_get_info()
        return free / (1024 ** 3)
    except Exception:
        return 0.0


@contextmanager
def _video_capture(path: str):
    cap = cv2.VideoCapture(path)
    try:
        yield cap
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Frame batch — accumulator (легче, чем _PoseAccumulator)
# ---------------------------------------------------------------------------

class _PoseAccumulator:
    """Совместимость со старым API; используется в тестах."""
    __slots__ = ("_meta", "_kp_chunks", "_count")

    def __init__(self) -> None:
        self._meta: list[dict] = []
        self._kp_chunks: list[np.ndarray] = []
        self._count: int = 0

    def add(self, meta: dict, kp: np.ndarray) -> None:
        self._meta.append(meta)
        self._kp_chunks.append(kp)
        self._count += 1

    def to_frames_data(self) -> list[dict]:
        result: list[dict] = []
        for m, kp in zip(self._meta, self._kp_chunks):
            entry = dict(m)
            if entry.get("poses") is None:
                entry["poses"] = [{"keypoints": kp}]
            result.append(entry)
        return result

    @property
    def count(self) -> int:
        return self._count

    def clear(self) -> None:
        self._meta.clear()
        self._kp_chunks.clear()
        self._count = 0


# ---------------------------------------------------------------------------
# Video frame producer (общий для prefetch-pipeline)
# ---------------------------------------------------------------------------

class _VideoFrameProducer:
    """
    Декодирует видео в отдельном потоке и кладёт (f_idx, frame) в очередь.
    Sentinel = None.

    В отличие от исходной версии:
    * для skip-кадров используется только cap.grab() (не retrieve+decode) —
      экономит 30–50% времени декодирования.
    * resize выполняется тут же, чтобы main thread получал готовый uint8.
    """

    def __init__(
        self,
        path: str,
        skip: int,
        target_w: int, target_h: int,
        needs_resize: bool,
        out_queue: queue.Queue,
        stop_event: Event,
    ) -> None:
        self.path = path
        self.skip = max(1, skip)
        self.target_w = target_w
        self.target_h = target_h
        self.needs_resize = needs_resize
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.thread: Thread | None = None

    def start(self, name: str = "VideoReader") -> None:
        self.thread = Thread(target=self._run, daemon=True, name=name)
        self.thread.start()

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            self._send_sentinel()
            return

        f_idx = 0
        skip = self.skip
        try:
            while not self.stop_event.is_set():
                if f_idx % skip == 0:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame is None or frame.size == 0:
                        f_idx += 1
                        continue
                    if self.needs_resize:
                        frame = _fast_resize(frame, self.target_w, self.target_h)
                    try:
                        self.out_queue.put(
                            (f_idx, frame), block=True, timeout=_READER_PUT_TIMEOUT,
                        )
                    except queue.Full:
                        # Пропускаем кадр чтобы не блокировать декодер
                        pass
                else:
                    if not cap.grab():
                        break
                f_idx += 1
        except Exception as exc:
            logger.error(f"Ошибка _VideoFrameProducer({self.path}): {exc}")
        finally:
            cap.release()
            self._send_sentinel()

    def _send_sentinel(self) -> None:
        for _ in range(5):
            try:
                self.out_queue.put(None, block=True, timeout=1.0)
                return
            except queue.Full:
                time.sleep(0.1)

    def join(self, timeout: float = 3.0) -> None:
        if self.thread:
            self.thread.join(timeout=timeout)


# ---------------------------------------------------------------------------
# Main backend
# ---------------------------------------------------------------------------

class AnalysisBackend:
    SIM_RANGES = {
        "high": (0.90, 1.00),
        "mid":  (0.80, 0.90),
        "low":  (0.0,  0.80),
    }
    DIR_LABELS = {
        "forward": "Лицом к камере",
        "left":    "Влево",
        "right":   "Вправо",
        "unknown": "Неизвестно",
    }
    BAND_LABELS = {
        "high": "Высокое сходство",
        "mid":  "Среднее",
        "low":  "Низкое",
    }

    def __init__(
        self,
        device: str | None = None,
        yolo:   YoloEngine | None = None,
    ) -> None:
        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if yolo is not None and not isinstance(yolo, YoloEngine):
            raise TypeError(f"yolo должен быть YoloEngine, получено: {type(yolo)}")

        self.yolo    = yolo if yolo is not None else YoloEngine(device=_device)
        self.matcher = MotionMatcher(device=_device)

        self._start_lock:    Lock  = Lock()
        self._running_event: Event = Event()
        self._stop_event:    Event = Event()
        self._thread: Thread | None = None

        self.BATCH_SIZE = self.yolo.get_batch_size()
        self.CHUNK_SIZE = DEFAULT_CHUNK_SIZE

        self._progress_cb: Callable | None = None
        self._result_cb:   Callable | None = None
        self._error_cb:    Callable | None = None

        self.preview_cache_dir = Path("cache/previews")
        self.preview_cache_dir.mkdir(parents=True, exist_ok=True)

        self._query_poses:  list[np.ndarray] = []
        self._query_images: list[np.ndarray] = []
        self._search_mode:  SearchMode       = SearchMode.MOTION_MATCH
        self._photo_matcher: PhotoMatcherProtocol | None = None

        # Текущий producer (для остановки)
        self._current_producer: _VideoFrameProducer | None = None
        # Сохраняем атрибуты для совместимости со старым API
        self._reader_queue: queue.Queue | None = None
        self._reader_thread: Thread | None = None

        self._eta_timestamps: deque[tuple[float, int]] = deque(maxlen=ETA_WINDOW)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def analysis_running(self) -> bool:
        return self._running_event.is_set()

    @analysis_running.setter
    def analysis_running(self, value: bool) -> None:
        if value:
            self._running_event.set()
        else:
            self._running_event.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_analysis(
        self,
        video_paths:       Any,
        settings:          Any,
        progress_callback: Callable | None = None,
        result_callback:   Callable | None = None,
        *,
        on_error: Callable | None = None,
        mode:     SearchMode      = SearchMode.MOTION_MATCH,
    ) -> None:
        video_paths = _sanitize_video_paths(video_paths)
        settings    = _sanitize_settings(settings)

        if not video_paths:
            logger.warning("Нет валидных видео для анализа.")
            if on_error:
                try:
                    on_error("Нет валидных видео для анализа.", None)
                except Exception:
                    pass
            return

        if mode not in _VALID_MODES:
            logger.warning(f"Неизвестный режим {mode}, используется MOTION_MATCH")
            mode = SearchMode.MOTION_MATCH

        with self._start_lock:
            if self.analysis_running:
                logger.warning("Анализ уже запущен.")
                return

            self._progress_cb = progress_callback
            self._result_cb   = result_callback
            self._error_cb    = on_error
            self._search_mode = mode

            self.analysis_running = True
            self._stop_event.clear()
            self._eta_timestamps.clear()

        target = (
            self._run_person_search
            if mode == SearchMode.PERSON_SEARCH
            else self._run_analysis
        )

        self._thread = Thread(
            target=target, args=(video_paths, settings),
            daemon=True, name="AnalysisBackend",
        )
        self._thread.start()

    def stop_analysis(self, timeout: float = 5.0) -> None:
        self.analysis_running = False
        self._stop_event.set()

        if self._current_producer:
            self._current_producer.join(timeout=1.0)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Поток анализа не завершился за отведённое время")

    def set_query_poses(self, query_images: Any) -> int:
        query_images = _sanitize_frames(query_images if query_images is not None else [])

        self._query_images = query_images
        self._query_poses  = []

        if not query_images:
            return 0

        try:
            detections = self.yolo.detect_batch(query_images)
            for det in detections:
                if det and is_pose_valid(det):
                    vec = preprocess_pose(det, use_body_weights=True)
                    self._query_poses.append(vec)
        except Exception as exc:
            logger.error(f"Ошибка извлечения query-поз: {exc}")

        logger.info(f"Query-поз: {len(self._query_poses)}/{len(query_images)}")
        return len(self._query_poses)

    def clear_query(self) -> None:
        self._query_poses  = []
        self._query_images = []

    def start_person_search(
        self,
        query_images:      Any,
        video_paths:       Any,
        settings:          Any,
        progress_callback: Callable | None = None,
        result_callback:   Callable | None = None,
        *,
        on_error: Callable | None = None,
    ) -> int:
        n = self.set_query_poses(query_images)
        if n == 0:
            logger.warning("Нет поз в query-изображениях.")
            return 0
        self.start_analysis(
            video_paths, settings,
            progress_callback=progress_callback,
            result_callback=result_callback,
            on_error=on_error,
            mode=SearchMode.PERSON_SEARCH,
        )
        return n

    def set_photo_matcher(self, matcher: Any) -> None:
        if matcher is not None and not hasattr(matcher, "filter_poses_by_reference"):
            logger.warning("photo_matcher не имеет нужного интерфейса, сброшен.")
            matcher = None
        self._photo_matcher = matcher

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _emit_progress(self, progress: AnalysisProgress) -> None:
        if not self._progress_cb:
            return
        if not isinstance(progress, AnalysisProgress):
            return
        try:
            self._progress_cb(progress)
        except Exception as exc:
            logger.error(f"Ошибка в progress_callback: {exc}")

    def _emit_result(self, result: AnalysisResult) -> None:
        if not self._result_cb:
            return
        try:
            self._result_cb(result)
        except Exception as exc:
            logger.error(f"Ошибка в result_callback: {exc}")

    def _emit_error(self, msg: str, exc: Exception | None = None) -> None:
        if self._error_cb:
            try:
                self._error_cb(str(msg)[:1000], exc)
            except Exception:
                pass
        logger.error(msg)
        if exc:
            logger.error(traceback.format_exc())

    # ------------------------------------------------------------------
    # ETA
    # ------------------------------------------------------------------

    def _update_eta(self, done_frames: int) -> float | None:
        now = time.monotonic()
        self._eta_timestamps.append((now, done_frames))
        if len(self._eta_timestamps) < 2:
            return None
        t0, f0 = self._eta_timestamps[0]
        t1, f1 = self._eta_timestamps[-1]
        dt = t1 - t0
        df = f1 - f0
        if dt < 0.1 or df <= 0:
            return None
        return None

    # ------------------------------------------------------------------
    # Core extraction with PIPELINE (prefetch следующего видео)
    # ------------------------------------------------------------------

    def _extract_all_poses(
        self,
        video_paths: list[str],
        settings:    dict[str, Any],
        base_start:  float,
        base_end:    float,
        n_videos:    int,
    ) -> tuple[list[dict], list[VideoMeta]]:
        quality_raw = settings.get("quality", "medium")
        quality = QUALITY_MAPPING.get(quality_raw, quality_raw)
        if quality not in QUALITY_FPS:
            quality = DEFAULT_QUALITY

        all_frames_data: list[dict]      = []
        video_metas:     list[VideoMeta] = []

        # Pre-open следующего видео параллельно с обработкой текущего:
        # держим в pending словарь {video_idx: (queue, producer, vmeta, ctx)}
        # Для простоты — pipeline через "prefetched_meta" одного видео вперёд.
        # Полный N+1 pipeline:
        #   * подготавливаем очередь+producer для следующего видео ДО того,
        #     как закончим текущее → как только текущее закончилось,
        #     GPU сразу получает данные из новой очереди.

        pending_meta: list[Optional[tuple]] = [None] * n_videos

        def _prepare_video_ctx(v_idx: int) -> Optional[tuple]:
            """Открывает видео, читает meta, запускает producer thread."""
            if v_idx < 0 or v_idx >= n_videos:
                return None
            v_path = video_paths[v_idx]

            cap = cv2.VideoCapture(v_path)
            if not cap.isOpened():
                logger.warning(f"Не удалось открыть: {v_path}")
                return None

            fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            fps          = _clamp(fps, _MIN_FPS, _MAX_FPS)
            total_frames = max(total_frames, 1)
            width        = int(_clamp(width,  1, _MAX_FRAME_DIM))
            height       = int(_clamp(height, 1, _MAX_FRAME_DIM))

            vmeta = VideoMeta(
                path=v_path, video_idx=v_idx,
                fps=fps, total_frames=total_frames,
                width=width, height=height,
            )

            target_fps = QUALITY_FPS[quality]
            is_4k = _is_4k(width, height)
            if is_4k:
                target_fps = min(target_fps, _MAX_4K_FPS_AUTO)
                logger.info(
                    f"[{v_idx}] 4K видео ({width}x{height}), "
                    f"FPS ограничен до {target_fps}"
                )

            skip = max(1, round(fps / target_fps))
            target_w, target_h = _get_resize_target(width, height)
            needs_resize = (width != target_w or height != target_h)

            effective_batch = self.BATCH_SIZE
            if is_4k:
                effective_batch = max(1, self.BATCH_SIZE // 2)

            logger.info(
                f"Video [{v_idx}] {os.path.basename(v_path)}: "
                f"{width}x{height} → {target_w}x{target_h}, "
                f"fps={fps:.1f}, skip={skip}, quality={quality}, "
                f"4K={is_4k}, batch={effective_batch}"
            )

            q = queue.Queue(maxsize=READER_QUEUE_SIZE)
            producer = _VideoFrameProducer(
                v_path, skip, target_w, target_h,
                needs_resize, q, self._stop_event,
            )
            producer.start(name=f"VideoReader-{v_idx}")

            return (q, producer, vmeta, {
                "skip": skip, "fps": fps, "total_frames": total_frames,
                "effective_batch": effective_batch, "path": v_path,
                "v_idx": v_idx,
            })

        # Стартуем prefetch первых N видео
        for i in range(min(_VIDEO_PREFETCH_QUEUE, n_videos)):
            pending_meta[i] = _prepare_video_ctx(i)

        # Главный цикл: обрабатываем готовые видео
        for v_idx in range(n_videos):
            if self._stop_event.is_set():
                break

            # Освобождаем CUDA-кэш периодически
            if v_idx > 0 and v_idx % _CUDA_EMPTY_EVERY == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            ctx = pending_meta[v_idx]
            if ctx is None:
                # Если по какой-то причине не подготовили — открываем сейчас
                ctx = _prepare_video_ctx(v_idx)
                pending_meta[v_idx] = ctx
            if ctx is None:
                continue

            q, producer, vmeta, info = ctx
            self._current_producer = producer
            self._reader_queue = q  # для совместимости
            self._reader_thread = producer.thread

            # Запускаем prefetch следующего видео заранее (pipeline)
            next_idx = v_idx + _VIDEO_PREFETCH_QUEUE
            if next_idx < n_videos and pending_meta[next_idx] is None:
                pending_meta[next_idx] = _prepare_video_ctx(next_idx)

            status_prefix = (
                "Поиск в" if self._search_mode == SearchMode.PERSON_SEARCH
                else "Анализ"
            )
            self._emit_progress(AnalysisProgress(
                percent     = base_start + (v_idx / max(n_videos, 1)) * (base_end - base_start) * 0.1,
                status      = f"{status_prefix}: {os.path.basename(info['path'])}",
                status_code = AnalysisStatus.ANALYZING_VIDEO,
                video_idx   = v_idx, video_count = n_videos,
                current_video = info["path"],
            ))

            try:
                frames_data = self._consume_video_queue(
                    q, producer, info,
                )
            except Exception as exc:
                logger.error(f"Ошибка обработки {info['path']}: {exc}\n{traceback.format_exc()}")
                continue

            # Фото-фильтр (если задан)
            if self._photo_matcher is not None and frames_data:
                try:
                    before = len(frames_data)
                    frames_data = self._photo_matcher.filter_poses_by_reference(
                        frames_data, threshold=0.50,
                    )
                    logger.info(
                        f"Фото-фильтр [{v_idx}]: {before} → {len(frames_data)}"
                    )
                except Exception as e:
                    logger.error(f"Ошибка фото-фильтрации: {e}")

            # Глобальный лимит
            new_total = len(all_frames_data) + len(frames_data)
            if new_total > MAX_TOTAL_POSES:
                remaining = MAX_TOTAL_POSES - len(all_frames_data)
                frames_data = frames_data[:remaining]
                logger.warning(
                    f"Достигнут глобальный лимит поз ({MAX_TOTAL_POSES})."
                )
                all_frames_data.extend(frames_data)
                video_metas.append(vmeta)
                # Останавливаем оставшиеся prefetch-producer'ы
                self._stop_event.set()
                break

            all_frames_data.extend(frames_data)
            video_metas.append(vmeta)

            logger.info(
                f"[{v_idx}] {os.path.basename(info['path'])}: "
                f"{len(frames_data)} поз обработано"
            )

            # Периодический gc — но не на каждом видео
            if (v_idx + 1) % _GC_EVERY_VIDEOS == 0:
                gc.collect()

        # Если был ранний выход — гасим оставшиеся producer'ы
        for ctx in pending_meta:
            if ctx is None:
                continue
            _, producer, _, _ = ctx
            producer.join(timeout=0.5)

        self._current_producer = None
        return all_frames_data, video_metas

    def _consume_video_queue(
        self,
        q: queue.Queue,
        producer: _VideoFrameProducer,
        info: dict,
    ) -> list[dict]:
        """Читает кадры из очереди, формирует батчи, отдаёт detect_batch."""
        v_idx        = info["v_idx"]
        fps          = info["fps"]
        total_frames = info["total_frames"]
        skip         = info["skip"]
        eff_batch    = info["effective_batch"]
        path         = info["path"]
        stop_ev      = self._stop_event

        frames_data: list[dict] = []
        batch_frames:    list[np.ndarray] = []
        batch_frame_ids: list[int]        = []
        last_progress_pct = -1
        frames_processed  = 0
        denom = max(total_frames // skip, 1)

        while True:
            if stop_ev.is_set():
                break
            try:
                item = q.get(block=True, timeout=0.5)
            except queue.Empty:
                if not producer.thread or not producer.thread.is_alive():
                    break
                continue

            if item is None:
                break

            f_idx, frame = item
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                continue

            batch_frames.append(frame)
            batch_frame_ids.append(f_idx)
            frames_processed += 1

            if len(batch_frames) >= eff_batch:
                if stop_ev.is_set():
                    break
                self._flush_batch(batch_frames, batch_frame_ids,
                                  v_idx, fps, frames_data)
                batch_frames.clear()
                batch_frame_ids.clear()

                pct = int((frames_processed / denom) * 100)
                pct = min(pct, 99)
                if pct >= last_progress_pct + PROGRESS_UPDATE_PCT:
                    self._emit_progress(AnalysisProgress(
                        percent       = pct,
                        status        = f"Извлечение: {os.path.basename(path)}",
                        status_code   = AnalysisStatus.EXTRACTING_POSES,
                        video_idx     = v_idx, video_count = 1,
                        current_frame = f_idx, total_frames = total_frames,
                        current_video = path,
                    ))
                    last_progress_pct = pct

        if batch_frames and not stop_ev.is_set():
            self._flush_batch(batch_frames, batch_frame_ids,
                              v_idx, fps, frames_data)

        producer.join(timeout=3.0)
        return frames_data

    # ------------------------------------------------------------------
    # Совместимый wrapper над старым API
    # ------------------------------------------------------------------

    def _process_video_file(
        self, path: str, video_idx: int, quality: str,
    ) -> tuple[list[dict], VideoMeta | None]:
        """Старый API; теперь делегирует в новые методы."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            logger.warning(f"Не удалось открыть: {path}")
            return [], None

        fps          = _clamp(cap.get(cv2.CAP_PROP_FPS) or 30.0, _MIN_FPS, _MAX_FPS)
        total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        width        = int(_clamp(cap.get(cv2.CAP_PROP_FRAME_WIDTH),  1, _MAX_FRAME_DIM))
        height       = int(_clamp(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 1, _MAX_FRAME_DIM))
        cap.release()

        vmeta = VideoMeta(
            path=path, video_idx=video_idx,
            fps=fps, total_frames=total_frames,
            width=width, height=height,
        )
        target_fps = QUALITY_FPS[quality]
        is_4k = _is_4k(width, height)
        if is_4k:
            target_fps = min(target_fps, _MAX_4K_FPS_AUTO)
        skip = max(1, round(fps / target_fps))
        target_w, target_h = _get_resize_target(width, height)
        needs_resize = (width != target_w or height != target_h)
        eff_batch = max(1, self.BATCH_SIZE // 2) if is_4k else self.BATCH_SIZE

        q = queue.Queue(maxsize=READER_QUEUE_SIZE)
        producer = _VideoFrameProducer(
            path, skip, target_w, target_h, needs_resize, q, self._stop_event,
        )
        producer.start(name=f"VideoReader-{video_idx}")
        self._current_producer = producer
        self._reader_queue = q
        self._reader_thread = producer.thread

        info = {
            "skip": skip, "fps": fps, "total_frames": total_frames,
            "effective_batch": eff_batch, "path": path, "v_idx": video_idx,
        }
        try:
            frames_data = self._consume_video_queue(q, producer, info)
        finally:
            self._current_producer = None

        if self._photo_matcher is not None and frames_data:
            try:
                before = len(frames_data)
                frames_data = self._photo_matcher.filter_poses_by_reference(
                    frames_data, threshold=0.50,
                )
                logger.info(f"Фото-фильтр [{video_idx}]: {before} → {len(frames_data)}")
            except Exception as e:
                logger.error(f"Ошибка фото-фильтрации: {e}")

        return frames_data, vmeta

    # ------------------------------------------------------------------
    # _flush_batch — оптимизирован: меньше Python overhead
    # ------------------------------------------------------------------

    def _flush_batch(
        self,
        batch_frames:    list[np.ndarray],
        batch_frame_ids: list[int],
        video_idx:       int,
        fps:             float,
        frames_data:     list[dict],
    ) -> None:
        if not batch_frames:
            return

        try:
            detections = self.yolo.detect_batch(batch_frames)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.warning("OOM в detect_batch, рекурсивное разбиение")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                mid = max(len(batch_frames) // 2, 1)
                self._flush_batch(
                    batch_frames[:mid], batch_frame_ids[:mid],
                    video_idx, fps, frames_data,
                )
                if len(batch_frames) > mid:
                    self._flush_batch(
                        batch_frames[mid:], batch_frame_ids[mid:],
                        video_idx, fps, frames_data,
                    )
                return
            logger.error(f"Ошибка detect_batch: {exc}")
            return
        except Exception as exc:
            logger.error(f"Неожиданная ошибка detect_batch: {exc}")
            return

        inv_fps = 1.0 / max(fps, 1.0)
        current_count = len(frames_data)
        # extend через append в локальную ссылку (быстрее)
        append = frames_data.append

        for det, fid in zip(detections, batch_frame_ids):
            if current_count >= MAX_TOTAL_POSES:
                break
            if det is None or not is_pose_valid(det):
                continue

            append({
                "t":         fid * inv_fps,
                "f":         fid,
                "video_idx": video_idx,
                "dir":       det.get("direction", "forward"),
                "scale":     det.get("scale",     1.0),
                "anchor_y":  det.get("anchor_y",  0.5),
                "keypoints": det.get("keypoints"),
                "poses":     [det],
            })
            current_count += 1

    # ------------------------------------------------------------------
    # Run: motion match
    # ------------------------------------------------------------------

    def _run_analysis(
        self,
        video_paths: list[str],
        settings:    dict[str, Any],
    ) -> None:
        t_start = time.monotonic()
        result  = AnalysisResult(
            video_paths=list(video_paths),
            mode=SearchMode.MOTION_MATCH,
        )

        try:
            _evict_preview_cache(self.preview_cache_dir, PREVIEW_CACHE_LIMIT)

            threshold  = settings.get("threshold", 70) / 100.0
            min_gap    = float(settings.get("scene_interval", 3))
            use_mirror = bool(settings.get("use_mirror", False))

            threshold = _clamp(threshold, _MIN_THRESHOLD, _MAX_THRESHOLD)
            min_gap   = _clamp(min_gap,  _MIN_SCENE_INTERVAL, _MAX_SCENE_INTERVAL)

            n_videos = len(video_paths)

            all_frames_data, video_metas = self._extract_all_poses(
                video_paths, settings,
                base_start=0.0, base_end=70.0, n_videos=n_videos,
            )

            result.video_metas = video_metas

            if self._stop_event.is_set():
                result.stopped = True
                self._finalize(result, t_start)
                return

            if not all_frames_data:
                result.stats["total_poses"] = 0
                self._finalize(result, t_start)
                return

            self._emit_progress(AnalysisProgress(
                percent     = 72.0,
                status      = "Сборка тензора поз…",
                status_code = AnalysisStatus.EXTRACTING_POSES,
                video_count = n_videos,
            ))

            poses_tensor, poses_meta = build_poses_tensor(
                all_frames_data, use_body_weights=True,
            )

            del all_frames_data
            # Без gc.collect здесь — оставляем CPython делать это лениво

            if poses_tensor is None or len(poses_meta) == 0:
                result.stats["total_poses"] = 0
                self._finalize(result, t_start)
                return

            result.poses_meta           = poses_meta
            result.poses_tensor         = poses_tensor
            result.stats["total_poses"] = len(poses_meta)

            self._emit_progress(AnalysisProgress(
                percent     = 75.0,
                status      = "Поиск совпадений…",
                status_code = AnalysisStatus.MATCHING,
                video_count = n_videos,
            ))

            matches = self.matcher.find_matches(
                poses_tensor = poses_tensor,
                poses_meta   = poses_meta,
                threshold    = threshold,
                min_gap      = min_gap,
                use_mirror   = use_mirror,
            )

            if self._photo_matcher is not None:
                try:
                    before = len(matches)
                    matches = self._photo_matcher.filter_matches(matches, threshold=0.7)
                    logger.info(f"Фото пост-фильтр: {before} → {len(matches)}")
                except Exception as e:
                    logger.error(f"Ошибка фото-фильтрации матчей: {e}")

            result.matches                = matches
            result.stats["total_matches"] = len(matches)

            self._emit_progress(AnalysisProgress(
                percent     = 92.0,
                status      = "Классификация движений…",
                status_code = AnalysisStatus.CLASSIFYING,
                video_count = n_videos,
            ))

            result.motion_groups          = self._build_motion_groups(matches)
            result.stats["motion_groups"] = len(result.motion_groups)

        except Exception as exc:
            result.error = str(exc)[:500]
            self._emit_error("Ошибка в ходе анализа", exc)

        finally:
            self._finalize(result, t_start)

    # ------------------------------------------------------------------
    # Run: person search — с CUDA streams и pinned memory
    # ------------------------------------------------------------------

    def _run_person_search(
        self,
        video_paths: list[str],
        settings:    dict[str, Any],
    ) -> None:
        t_start = time.monotonic()
        result  = AnalysisResult(
            video_paths=list(video_paths),
            mode=SearchMode.PERSON_SEARCH,
        )

        if not self._query_poses:
            result.error = "Нет поз запроса — вызовите set_query_poses()."
            self._emit_error(result.error)
            self._finalize(result, t_start)
            return

        try:
            _evict_preview_cache(self.preview_cache_dir, PREVIEW_CACHE_LIMIT)

            n_videos = len(video_paths)

            all_frames_data, video_metas = self._extract_all_poses(
                video_paths, settings,
                base_start=0.0, base_end=70.0, n_videos=n_videos,
            )

            result.video_metas = video_metas

            if self._stop_event.is_set():
                result.stopped = True
                self._finalize(result, t_start)
                return

            self._emit_progress(AnalysisProgress(
                percent     = 75.0,
                status      = "Сравнение с запросом…",
                status_code = AnalysisStatus.MATCHING,
                video_count = n_videos,
            ))

            poses_tensor, poses_meta = build_poses_tensor(
                all_frames_data, use_body_weights=True,
            )

            del all_frames_data

            if poses_tensor is None:
                result.stats["total_poses"] = 0
                self._finalize(result, t_start)
                return

            result.poses_meta           = poses_meta
            result.poses_tensor         = poses_tensor
            result.stats["total_poses"] = len(poses_meta)

            sims_np = self._person_search_similarities(poses_tensor)

            threshold = _clamp(settings.get("threshold", 60) / 100.0, 0.0, 1.0)

            # Векторная фильтрация по threshold
            mask = sims_np >= threshold
            valid_idx = np.where(mask)[0]

            if valid_idx.size == 0:
                result.matches = []
                result.stats["total_matches"] = 0
            else:
                sims_valid = sims_np[valid_idx]
                # Сортируем по убыванию sim, берём top max_unique
                cap = self.matcher.max_unique
                if valid_idx.size > cap:
                    top = np.argpartition(-sims_valid, cap)[:cap]
                    top = top[np.argsort(-sims_valid[top])]
                else:
                    top = np.argsort(-sims_valid)

                candidates = []
                for j in top:
                    i = int(valid_idx[j])
                    meta = poses_meta[i]
                    sim_val = float(sims_valid[j])
                    candidates.append({
                        "m1_idx":    i,
                        "m2_idx":    -1,
                        "sim":       sim_val,
                        "sim_raw":   sim_val,
                        "t1":        meta["t"],
                        "t2":        -1.0,
                        "f1":        meta["f"],
                        "f2":        -1,
                        "v1_idx":    meta["video_idx"],
                        "v2_idx":    -1,
                        "direction": meta.get("dir", "unknown"),
                        "kp1":       meta.get("kp"),
                    })
                result.matches                = candidates
                result.stats["total_matches"] = len(candidates)

            self._emit_progress(AnalysisProgress(
                percent     = 92.0,
                status      = "Классификация…",
                status_code = AnalysisStatus.CLASSIFYING,
                video_count = n_videos,
            ))
            result.motion_groups = self._build_motion_groups(result.matches)

        except Exception as exc:
            result.error = str(exc)[:500]
            self._emit_error("Ошибка при поиске человека", exc)

        finally:
            self._finalize(result, t_start)

    def _person_search_similarities(
        self,
        poses_tensor: torch.Tensor,
    ) -> np.ndarray:
        """
        Вычисляет max-cosine между каждой позой и набором query-поз.

        Использует CUDA streams для overlap H2D/compute/D2H, если GPU доступен.
        """
        device = self.matcher.device
        N = len(poses_tensor)

        # Запросы готовим один раз
        q_arr = np.stack(self._query_poses, axis=0).astype(np.float32, copy=False)
        q_t = torch.from_numpy(q_arr)
        if device != "cpu":
            q_t = q_t.to(device, non_blocking=True)
        q_t = F.normalize(q_t.float(), p=2, dim=1)

        chunk = _INCREMENTAL_TENSOR_CHUNK
        sims_out = np.empty(N, dtype=np.float32)

        # CPU тензор источника (избегаем огромного to(GPU) одним куском)
        # poses_tensor может быть на CPU — в этом случае в for будем брать срезы.
        if poses_tensor.device.type == "cpu":
            src_cpu = poses_tensor.float()
        else:
            src_cpu = poses_tensor.float().cpu()

        if device == "cpu":
            # Простой CPU путь — батчами через mm
            src = F.normalize(src_cpu, p=2, dim=1)
            q_cpu = q_t.cpu()
            for i in range(0, N, chunk):
                v_chunk = src[i: i + chunk]
                sim_chunk = torch.mm(v_chunk, q_cpu.t()).max(dim=1).values
                sims_out[i: i + chunk] = sim_chunk.numpy()
            return sims_out

        # ── GPU путь с streams + pinned memory ────────────────────────
        # Pinned memory ускоряет H2D-копирование в 2–3 раза.
        try:
            src_cpu = src_cpu.pin_memory()
        except Exception:
            pass  # не критично

        # Два stream'а — пока compute идёт на одном чанке, другой грузит
        s_compute = torch.cuda.Stream()
        s_copy    = torch.cuda.Stream()

        # Готовим список (start, end) chunk-ов
        ranges = [(i, min(i + chunk, N)) for i in range(0, N, chunk)]

        # Pre-allocate GPU buffer для одного чанка (переиспользуем)
        # Размер D — известен после первого чанка
        D = poses_tensor.shape[1]
        gpu_bufs = [
            torch.empty((chunk, D), dtype=torch.float32, device=device),
            torch.empty((chunk, D), dtype=torch.float32, device=device),
        ]

        # Pinned host out-buffer для D2H
        host_outs = [
            torch.empty(chunk, dtype=torch.float32).pin_memory(),
            torch.empty(chunk, dtype=torch.float32).pin_memory(),
        ]

        events_compute = [torch.cuda.Event() for _ in range(2)]
        events_copy = [torch.cuda.Event() for _ in range(2)]

        # Запускаем pipeline:
        #   на parity i%2: copy_in[parity] || compute[1-parity] на предыдущем
        for idx_r, (lo, hi) in enumerate(ranges):
            sz = hi - lo
            par = idx_r & 1
            buf = gpu_bufs[par]
            host_out = host_outs[par]

            # ── copy stream: H2D ──
            with torch.cuda.stream(s_copy):
                # Дожидаемся, пока буфер этого parity освободился предыдущим compute
                if idx_r >= 2:
                    events_compute[par].wait(s_copy)
                buf[:sz].copy_(src_cpu[lo:hi], non_blocking=True)
                events_copy[par].record(s_copy)

            # ── compute stream: normalize + mm + max ──
            with torch.cuda.stream(s_compute):
                events_copy[par].wait(s_compute)
                v_chunk = F.normalize(buf[:sz], p=2, dim=1)
                sim = torch.mm(v_chunk, q_t.t())
                sim_max = sim.max(dim=1).values
                # D2H
                host_out[:sz].copy_(sim_max, non_blocking=True)
                events_compute[par].record(s_compute)

            # Если предыдущий parity ещё не записан в numpy — синхронизируем
            if idx_r >= 1:
                prev_par = 1 - par
                # Дожидаемся завершения compute предыдущего parity (D2H завершён)
                events_compute[prev_par].synchronize()
                # Записываем в выходной numpy
                prev_lo, prev_hi = ranges[idx_r - 1]
                prev_sz = prev_hi - prev_lo
                sims_out[prev_lo:prev_hi] = host_outs[prev_par][:prev_sz].numpy()

        # Финальная синхронизация последнего chunk
        last_par = (len(ranges) - 1) & 1
        events_compute[last_par].synchronize()
        last_lo, last_hi = ranges[-1]
        last_sz = last_hi - last_lo
        sims_out[last_lo:last_hi] = host_outs[last_par][:last_sz].numpy()

        # Освобождаем ресурсы
        del gpu_bufs, host_outs, q_t
        torch.cuda.empty_cache()

        return sims_out

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def _finalize(self, result: AnalysisResult, t_start: float) -> None:
        elapsed = time.monotonic() - t_start
        result.stats["elapsed_seconds"] = round(elapsed, 2)

        status_code = (
            AnalysisStatus.STOPPED if result.stopped
            else AnalysisStatus.ERROR  if result.error
            else AnalysisStatus.DONE
        )

        self._emit_progress(AnalysisProgress(
            percent     = 100.0,
            status      = STATUS_LABELS[status_code],
            status_code = status_code,
        ))
        self._emit_result(result)
        self.analysis_running = False

        if result.stats.get("total_poses", 0) > 10_000:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Motion groups
    # ------------------------------------------------------------------

    def _build_motion_groups(self, matches: list[dict]) -> list[MotionGroup]:
        if not matches:
            return []

        def _dir(m: dict) -> str:
            d = m.get("direction", "unknown") or "unknown"
            return d if d in ("forward", "left", "right") else "unknown"

        def _band(s: float) -> str:
            if s >= 0.90:
                return "high"
            if s >= 0.80:
                return "mid"
            return "low"

        buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for m in matches:
            sim = m.get("sim", 0.0)
            if not isinstance(sim, (int, float)):
                sim = 0.0
            buckets[(_dir(m), _band(float(sim)))].append(m)

        groups: list[MotionGroup] = []
        for (direction, band), ms in buckets.items():
            groups.append(MotionGroup(
                label     = (
                    f"{self.DIR_LABELS.get(direction, direction)} — "
                    f"{self.BAND_LABELS.get(band, band)}"
                ),
                direction = direction,
                sim_range = self.SIM_RANGES[band],
                matches   = sorted(
                    ms, key=lambda x: x.get("sim", 0.0), reverse=True,
                ),
            ))

        groups.sort(key=lambda g: g.count, reverse=True)
        return groups
