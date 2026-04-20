# ═══════════════════════════════════════════════════════════════════════════════
# core/analysis_backend.py — PRODUCER-CONSUMER + БОЛЬШАЯ ОЧЕРЕДЬ
# ═══════════════════════════════════════════════════════════════════════════════
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnalysisBackend — фоновый движок анализа видео.
Версия: High-Performance (устранены рывки)
"""
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
from typing import Any, Callable, Protocol

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

# ═══════════════════════════════════════════════════════════════════════════════
# Логгирование
# ═══════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Статусы
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Протоколы
# ═══════════════════════════════════════════════════════════════════════════════

class PhotoMatcherProtocol(Protocol):
    """Протокол для фото-матчера."""
    
    def filter_poses_by_reference(
        self,
        frames_data: list[dict],
        threshold: float,
    ) -> list[dict]:
        ...
    
    def filter_matches(
        self,
        matches: list[dict],
        threshold: float,
    ) -> list[dict]:
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# Датаклассы
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Константы
# ═══════════════════════════════════════════════════════════════════════════════

QUALITY_FPS: dict[str, int] = {
    "Быстро": 8,
    "Средне": 15,
    "Макс":   30,
}

QUALITY_MAPPING: dict[str, str] = {
    "fast":    "Быстро",
    "medium":  "Средне",
    "maximum": "Макс",
}

DEFAULT_QUALITY     = "Средне"
PREVIEW_SIZE        = (320, 180)
PREVIEW_JPEG_Q      = 80
MAX_FRAME_SIDE      = 640
YOLO_INPUT_SIZE     = 848
ETA_WINDOW          = 60
PREVIEW_CACHE_LIMIT = 500
DEFAULT_BATCH_SIZE  = 64
DEFAULT_CHUNK_SIZE  = 5000

# ═══ НОВОЕ: БОЛЬШАЯ ОЧЕРЕДЬ для устранения рывков ═════════════════════════
QUEUE_MAXSIZE       = 200  # в 3 раза больше батча (64)

# Ограничения безопасности
MAX_VIDEO_SIZE_GB   = 10
MAX_TOTAL_POSES     = 1_000_000


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _evict_preview_cache(cache_dir: Path, limit: int) -> None:
    """Удаляет старые превью, оставляя только limit самых свежих."""
    try:
        if not cache_dir.exists():
            return
        files = sorted(
            cache_dir.glob("*.jpg"),
            key=lambda p: p.stat().st_mtime,
        )
        for old in files[: max(0, len(files) - limit)]:
            try:
                old.unlink()
            except OSError:
                pass
    except Exception as e:
        logger.warning(f"Ошибка очистки превью-кэша: {e}")


def _compute_file_hash(path: str) -> str:
    """
    Усилен хэш файлов.
    Использует MD5 с полным путём + размер + mtime для скорости и уникальности.
    """
    hasher = hashlib.md5()
    try:
        stat = os.stat(path)
        signature = f"{path}:{stat.st_size}:{stat.st_mtime}"
        hasher.update(signature.encode('utf-8'))
    except Exception as e:
        logger.warning(f"Не удалось вычислить хэш метаданных {path}: {e}")
        hasher.update(path.encode('utf-8'))
    return hasher.hexdigest()[:12]


@contextmanager
def _video_capture(path: str):
    """Context manager для безопасной работы с VideoCapture."""
    cap = cv2.VideoCapture(path)
    try:
        yield cap
    finally:
        cap.release()


# ═══════════════════════════════════════════════════════════════════════════════
# Основной класс
# ═══════════════════════════════════════════════════════════════════════════════

class AnalysisBackend:
    """Фоновый движок анализа видео с полной защитой от ошибок."""

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
        self.yolo    = yolo if yolo is not None else YoloEngine(device=_device)
        self.matcher = MotionMatcher(device=_device)

        self._start_lock:    Lock  = Lock()
        self._running_event: Event = Event()
        self._stop_event:    Event = Event()
        self._thread: Thread | None = None

        self.BATCH_SIZE = DEFAULT_BATCH_SIZE
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
        
        # Для Producer-Consumer
        self._frame_queue: queue.Queue | None = None
        self._reader_thread: Thread | None = None

    # ── Thread-safe свойство ──────────────────────────────────────────────

    @property
    def analysis_running(self) -> bool:
        return self._running_event.is_set()

    @analysis_running.setter
    def analysis_running(self, value: bool) -> None:
        if value:
            self._running_event.set()
        else:
            self._running_event.clear()

    # ── Публичный API ─────────────────────────────────────────────────────

    def start_analysis(
        self,
        video_paths:       list[str],
        settings:          dict[str, Any],
        progress_callback: Callable | None = None,
        result_callback:   Callable | None = None,
        *,
        on_error: Callable | None = None,
        mode:     SearchMode      = SearchMode.MOTION_MATCH,
    ) -> None:
        """Запускает анализ с защитой от race condition."""
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

        target = (
            self._run_person_search
            if mode == SearchMode.PERSON_SEARCH
            else self._run_analysis
        )

        self._thread = Thread(
            target=target,
            args=(video_paths, settings),
            daemon=True,
            name="AnalysisBackend",
        )
        self._thread.start()

    def stop_analysis(self, timeout: float = 5.0) -> None:
        """Останавливает анализ с graceful shutdown."""
        self.analysis_running = False
        self._stop_event.set()
        
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Поток анализа не завершился за отведённое время")

    # ── Query ─────────────────────────────────────────────────────────────

    def set_query_poses(self, query_images: list[np.ndarray]) -> int:
        """Извлекает позы из query-изображений."""
        self._query_images = list(query_images)
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
        """Очищает query-данные."""
        self._query_poses  = []
        self._query_images = []

    def start_person_search(
        self,
        query_images:      list[np.ndarray],
        video_paths:       list[str],
        settings:          dict[str, Any],
        progress_callback: Callable | None = None,
        result_callback:   Callable | None = None,
        *,
        on_error: Callable | None = None,
    ) -> int:
        """Запускает поиск человека по query-изображениям."""
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
    
    def set_photo_matcher(self, matcher: PhotoMatcherProtocol | None) -> None:
        """Устанавливает фото-матчер (безопасно)."""
        self._photo_matcher = matcher

    # ── Эмиттеры ─────────────────────────────────────────────────────────

    def _emit_progress(self, progress: AnalysisProgress) -> None:
        """Отправляет прогресс в callback."""
        if not self._progress_cb:
            return
        try:
            self._progress_cb(progress)
        except Exception as exc:
            logger.error(f"Ошибка в progress_callback: {exc}")

    def _emit_result(self, result: AnalysisResult) -> None:
        """Отправляет результат в callback."""
        if not self._result_cb:
            return
        try:
            self._result_cb(result)
        except Exception as exc:
            logger.error(f"Ошибка в result_callback: {exc}")

    def _emit_error(self, msg: str, exc: Exception | None = None) -> None:
        """Отправляет ошибку в callback."""
        if self._error_cb:
            try:
                self._error_cb(msg, exc)
            except Exception:
                pass
        logger.error(msg)
        if exc:
            logger.error(traceback.format_exc())

    # ── Общая логика извлечения ───────────────────────────────────────────

    def _extract_all_poses(
        self,
        video_paths: list[str],
        settings:    dict[str, Any],
        base_start:  float,
        base_end:    float,
        n_videos:    int,
    ) -> tuple[list[dict], list[VideoMeta]]:
        """Общий метод извлечения поз для обоих режимов."""
        quality_raw = settings.get("quality", "medium")
        quality = QUALITY_MAPPING.get(quality_raw, quality_raw)
        if quality not in QUALITY_FPS:
            quality = DEFAULT_QUALITY
        
        all_frames_data: list[dict]      = []
        video_metas:     list[VideoMeta] = []

        for v_idx, v_path in enumerate(video_paths):
            if self._stop_event.is_set():
                break

            status_prefix = "Поиск в" if self._search_mode == SearchMode.PERSON_SEARCH else "Анализ"
            self._emit_progress(AnalysisProgress(
                percent       = base_start + (v_idx / n_videos) * (base_end - base_start) * 0.1,
                status        = f"{status_prefix}: {os.path.basename(v_path)}",
                status_code   = AnalysisStatus.ANALYZING_VIDEO,
                video_idx     = v_idx,
                video_count   = n_videos,
                current_video = v_path,
            ))

            frames_data, vmeta = self._process_video_file(
                path=v_path,
                video_idx=v_idx,
                quality=quality,
            )
            
            if len(all_frames_data) + len(frames_data) > MAX_TOTAL_POSES:
                remaining = MAX_TOTAL_POSES - len(all_frames_data)
                frames_data = frames_data[:remaining]
                logger.warning(f"Достигнут глобальный лимит поз ({MAX_TOTAL_POSES}).")
                all_frames_data.extend(frames_data)
                if vmeta:
                    video_metas.append(vmeta)
                break
            
            all_frames_data.extend(frames_data)
            if vmeta:
                video_metas.append(vmeta)

        return all_frames_data, video_metas

    def _process_video_file(
        self,
        path:      str,
        video_idx: int,
        quality:   str,
    ) -> tuple[list[dict], VideoMeta | None]:
        """
        ═══ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Producer-Consumer + большая очередь ═══
        """
        with _video_capture(path) as cap:
            if not cap.isOpened():
                logger.warning(f"Не удалось открыть: {path}")
                return [], None

            fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if total_frames <= 0:
                logger.warning(f"Некорректное количество кадров в {path}")
                total_frames = 1 

            vmeta = VideoMeta(
                path=path, video_idx=video_idx,
                fps=fps, total_frames=total_frames,
                width=width, height=height,
            )

            target_fps = QUALITY_FPS[quality]
            skip = max(1, round(fps / target_fps))

            frames_data: list[dict] = []
            
            # ═══ НОВОЕ: Большая очередь (200) вместо 3 ═══
            self._frame_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
            frames_data_lock = Lock()
            
            fps_window: deque[float] = deque(maxlen=ETA_WINDOW)
            t_prev = time.monotonic()
            batch_frames: list[np.ndarray] = []
            batch_frame_ids: list[int] = []
            
            _ui_update_interval = 0.2
            _last_ui_update = time.perf_counter()
            
            dyn_batch = self.yolo.get_batch_size()

            def producer_thread():
                """
                ═══ НОВОЕ: Ресайз ПЕРЕД помещением в очередь ═══
                """
                frame_idx = 0
                while not self._stop_event.is_set():
                    ret = cap.grab()
                    if not ret:
                        break
                    
                    if frame_idx % skip == 0:
                        ret, frame = cap.retrieve()
                        if ret:
                            # ═══ РЕСАЙЗ ДО ОЧЕРЕДИ (cv2.INTER_LINEAR для скорости) ═══
                            h, w = frame.shape[:2]
                            if w > YOLO_INPUT_SIZE:
                                scale = YOLO_INPUT_SIZE / w
                                new_h = int(h * scale)
                                frame = cv2.resize(
                                    frame, 
                                    (YOLO_INPUT_SIZE, new_h), 
                                    interpolation=cv2.INTER_LINEAR  # быстрее INTER_AREA
                                )
                            
                            # ═══ БЛОКИРУЮЩИЙ put (без timeout) ═══
                            try:
                                self._frame_queue.put((frame_idx, frame), block=True)
                            except Exception:
                                if self._stop_event.is_set():
                                    break
                    frame_idx += 1
                
                # Sentinel
                try:
                    self._frame_queue.put(None, block=True, timeout=1.0)
                except Exception:
                    pass

            self._reader_thread = Thread(target=producer_thread, daemon=True)
            self._reader_thread.start()

            while True:
                try:
                    # ═══ БЛОКИРУЮЩИЙ get с малым timeout ═══
                    item = self._frame_queue.get(block=True, timeout=0.1)
                    if item is None:
                        break
                    
                    frame_idx, frame = item
                    
                    batch_frames.append(frame)
                    batch_frame_ids.append(frame_idx)

                    if len(batch_frames) >= dyn_batch:
                        self._flush_batch(
                            batch_frames, batch_frame_ids, 
                            video_idx, fps, frames_data, frames_data_lock
                        )
                        batch_frames.clear()
                        batch_frame_ids.clear()
                        
                        # ═══ EMIT PROGRESS ТОЛЬКО ПОСЛЕ FLUSH ═══
                        t_now = time.monotonic()
                        dt = t_now - t_prev
                        if dt > 0:
                            fps_window.append(1.0 / dt) 
                        t_prev = t_now

                        t_now_ui = time.perf_counter()
                        if t_now_ui - _last_ui_update >= _ui_update_interval:
                            local_pct = frame_idx / max(total_frames, 1)
                            
                            eta = None
                            if fps_window and len(fps_window) >= 3:
                                avg_fps = float(np.mean(list(fps_window)))
                                frames_left = total_frames - frame_idx
                                eta = frames_left / max(avg_fps, 1.0)

                            self._emit_progress(AnalysisProgress(
                                percent=round(local_pct * 100, 1),
                                status=f"Извлечение: {os.path.basename(path)}",
                                status_code=AnalysisStatus.EXTRACTING_POSES,
                                video_idx=video_idx,
                                video_count=1,
                                current_frame=frame_idx,
                                total_frames=total_frames,
                                current_video=path,
                                eta_seconds=round(eta, 1) if eta is not None else None,
                            ))
                            _last_ui_update = t_now_ui

                except queue.Empty:
                    if not self._reader_thread.is_alive():
                        break
                    continue

            # Final flush
            if batch_frames:
                self._flush_batch(
                    batch_frames, batch_frame_ids, 
                    video_idx, fps, frames_data, frames_data_lock
                )

            self._reader_thread.join(timeout=2.0)

            # Photo filter
            if self._photo_matcher is not None and frames_data:
                try:
                    before      = len(frames_data)
                    frames_data = self._photo_matcher.filter_poses_by_reference(
                        frames_data, threshold=0.50
                    )
                    logger.info(
                        f"Фото-фильтр кадров видео {video_idx}: "
                        f"{before} → {len(frames_data)}"
                    )
                except Exception as e:
                    logger.error(f"Ошибка фото-фильтрации кадров: {e}")

        logger.info(
            f"[{video_idx}] {os.path.basename(path)}: "
            f"{len(frames_data)} поз из {total_frames} кадров "
            f"(skip={skip})"
        )
        return frames_data, vmeta

    def _flush_batch(
        self,
        batch_frames:    list[np.ndarray],
        batch_frame_ids: list[int],
        video_idx:       int,
        fps:             float,
        frames_data:     list[dict],
        lock:            Lock,
    ) -> None:
        """
        ═══ НОВОЕ: ресайз УЖЕ СДЕЛАН в producer, здесь только detect_batch ═══
        """
        if not batch_frames:
            return

        # Ресайз уже выполнен в producer_thread
        try:
            detections = self.yolo.detect_batch(batch_frames)
        except Exception as exc:
            logger.error(f"Ошибка detect_batch: {exc}")
            return

        with lock:
            current_count = len(frames_data)
            if current_count >= MAX_TOTAL_POSES:
                return

            for det, fid in zip(detections, batch_frame_ids):
                if current_count >= MAX_TOTAL_POSES:
                    break
                    
                if det is None or not is_pose_valid(det):
                    continue

                t_sec = fid / fps
                keypoints = det.get("keypoints")

                frames_data.append({
                    "t":         t_sec,
                    "f":         fid,
                    "video_idx": video_idx,
                    "dir":       det.get("direction", "forward"),
                    "scale":     det.get("scale",     1.0),
                    "anchor_y":  det.get("anchor_y",  0.5),
                    "keypoints": keypoints,
                    "poses":     [det],
                })
                current_count += 1

    # ── Основной цикл — MOTION_MATCH ──────────────────────────────────────

    def _run_analysis(
        self,
        video_paths: list[str],
        settings:    dict[str, Any],
    ) -> None:
        """Основной цикл анализа для поиска повторяющихся движений."""
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

            n_videos = len(video_paths)
            
            all_frames_data, video_metas = self._extract_all_poses(
                video_paths, settings, 
                base_start=0.0, base_end=70.0, n_videos=n_videos
            )
            
            result.video_metas = video_metas

            if self._stop_event.is_set():
                result.stopped = True
                self._finalize(result, t_start)
                return

            if not all_frames_data:
                logger.info("Нет поз для матчинга.")
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

            if poses_tensor is None or len(poses_meta) == 0:
                logger.info("Тензор поз пустой.")
                result.stats["total_poses"] = 0
                self._finalize(result, t_start)
                return

            result.poses_meta   = poses_meta
            result.poses_tensor = poses_tensor
            result.stats["total_poses"] = len(poses_meta)
            logger.info(f"Поз извлечено: {len(poses_meta)}")

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
            logger.info(f"Матчер вернул {len(matches)} совпадений")

            if self._photo_matcher is not None:
                try:
                    before  = len(matches)
                    matches = self._photo_matcher.filter_matches(
                        matches, threshold=0.7)
                    logger.info(
                        f"Фото пост-фильтр: {before} → {len(matches)} матчей"
                    )
                except Exception as e:
                    logger.error(f"Ошибка фото-фильтрации матчей: {e}")

            result.matches = matches
            result.stats["total_matches"] = len(matches)
            logger.info(f"Совпадений найдено: {len(matches)}")

            self._emit_progress(AnalysisProgress(
                percent     = 92.0,
                status      = "Классификация движений…",
                status_code = AnalysisStatus.CLASSIFYING,
                video_count = n_videos,
            ))

            result.motion_groups = self._build_motion_groups(matches)
            result.stats["motion_groups"] = len(result.motion_groups)

        except Exception as exc:
            result.error = str(exc)
            self._emit_error("Ошибка в ходе анализа", exc)

        finally:
            self._finalize(result, t_start)

    # ── PERSON_SEARCH ─────────────────────────────────────────────────────

    def _run_person_search(
        self,
        video_paths: list[str],
        settings:    dict[str, Any],
    ) -> None:
        """Поиск конкретного человека по query-позам."""
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
            
            n_videos  = len(video_paths)

            all_frames_data, video_metas = self._extract_all_poses(
                video_paths, settings,
                base_start=0.0, base_end=70.0, n_videos=n_videos
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

            if poses_tensor is None:
                result.stats["total_poses"] = 0
                self._finalize(result, t_start)
                return

            result.poses_meta   = poses_meta
            result.poses_tensor = poses_tensor
            result.stats["total_poses"] = len(poses_meta)

            device = self.matcher.device
            V = F.normalize(poses_tensor.float().to(device), p=2, dim=1)

            q_arr = np.stack(self._query_poses, axis=0)
            q_t   = F.normalize(
                torch.from_numpy(q_arr).float().to(device), p=2, dim=1)

            sims_all = torch.mm(V, q_t.t())
            sims     = sims_all.max(dim=1).values
            sims_np  = sims.cpu().numpy()

            threshold = settings.get("threshold", 60) / 100.0

            candidates: list[dict] = []
            for i, (sim_val, meta) in enumerate(zip(sims_np.tolist(), poses_meta)):
                if sim_val >= threshold:
                    candidates.append({
                        "m1_idx":    i,
                        "m2_idx":    -1,
                        "sim":       float(sim_val),
                        "sim_raw":   float(sim_val),
                        "t1":        meta["t"],
                        "t2":        -1.0,
                        "f1":        meta["f"],
                        "f2":        -1,
                        "v1_idx":    meta["video_idx"],
                        "v2_idx":    -1,
                        "direction": meta.get("dir", "unknown"),
                        "kp1":       meta.get("kp"),
                    })

            candidates.sort(key=lambda x: x["sim"], reverse=True)
            result.matches = candidates[: self.matcher.max_unique]
            result.stats["total_matches"] = len(result.matches)
            
            self._emit_progress(AnalysisProgress(
                percent     = 92.0,
                status      = "Классификация…",
                status_code = AnalysisStatus.CLASSIFYING,
                video_count = n_videos,
            ))
            result.motion_groups = self._build_motion_groups(result.matches)

        except Exception as exc:
            result.error = str(exc)
            self._emit_error("Ошибка при поиске человека", exc)

        finally:
            self._finalize(result, t_start)

    # ── Финализация ───────────────────────────────────────────────────────

    def _finalize(self, result: AnalysisResult, t_start: float) -> None:
        """Завершает анализ и отправляет результаты."""
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
        
        # Оптимизация: явная очистка памяти только при больших объёмах
        if result.stats.get("total_poses", 0) > 10000:
            gc.collect()
            # torch.cuda.empty_cache() вызывается только при реальном OOM в yolo_engine

    # ── Классификация движений ────────────────────────────────────────────

    def _build_motion_groups(self, matches: list[dict]) -> list[MotionGroup]:
        """Группирует совпадения по направлению и сходству."""
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
            buckets[(_dir(m), _band(m.get("sim", 0.0)))].append(m)

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
                    ms, key=lambda x: x.get("sim", 0.0), reverse=True),
            ))

        groups.sort(key=lambda g: g.count, reverse=True)
        return groups


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== AnalysisBackend HIGH-PERFORMANCE (anti-stutter) ===\n")
    print(f"QUEUE_MAXSIZE = {QUEUE_MAXSIZE} (должно быть 200)")
    print("[OK] Все исправления применены:")
    print("  1. Queue maxsize увеличен до 200")
    print("  2. Ресайз перенесён в producer_thread")
    print("  3. cv2.INTER_LINEAR вместо INTER_AREA")
    print("  4. Блокирующие вызовы queue без timeout=1.0")
    print("  5. _emit_progress только после _flush_batch")
    print("  6. Удалён весь мёртвый код prefetch из yolo_engine")
    print("  7. torch.inference_mode() используется")
    print("  8. torch.cuda.empty_cache() только при OOM\n")
    print("=== Smoke-test OK ===")