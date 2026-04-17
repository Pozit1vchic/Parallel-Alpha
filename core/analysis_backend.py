#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnalysisBackend — фоновый движок анализа видео.
Исправлены все критические баги, добавлена оптимизация и робастность.
"""
from __future__ import annotations

import gc
import hashlib
import logging
import os
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

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AnalysisBackend")


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

# Исправлено: унифицированы ключи quality mapping
QUALITY_FPS: dict[str, int] = {
    "Быстро": 8,
    "Средне": 15,
    "Макс":   30,
}

QUALITY_MAPPING: dict[str, str] = {
    "fast":    "Быстро",
    "medium":  "Средне",
    "maximum": "Макс",      # Исправлено: было "Максимум"
}

DEFAULT_QUALITY     = "Средне"
PREVIEW_SIZE        = (320, 180)
PREVIEW_JPEG_Q      = 80
MAX_FRAME_SIDE      = 640
MAX_ADAPTIVE_SKIP   = 12
ETA_WINDOW          = 60
PREVIEW_CACHE_LIMIT = 500

# Ограничения безопасности
MAX_VIDEO_SIZE_GB   = 10
MAX_TOTAL_POSES     = 1_000_000
MAX_ANALYSIS_TIME_H = 24


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _resize_frame(frame: np.ndarray, max_side: int) -> np.ndarray:
    """Уменьшает кадр без копирования, если не требуется изменение размера."""
    h, w  = frame.shape[:2]
    scale = min(max_side / w, max_side / h, 1.0)
    if scale >= 0.99:  # Threshold для избежания микро-ресайзов
        return frame
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _evict_preview_cache(cache_dir: Path, limit: int) -> None:
    """Удаляет старые превью, оставляя только limit самых свежих."""
    try:
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


def _save_preview(
    frame:      np.ndarray,
    cache_dir:  Path,
    video_hash: str,
    frame_id:   int,
) -> None:
    """Сохраняет превью кадра с оптимизацией — сразу в нужном размере."""
    try:
        path = cache_dir / f"{video_hash}_{frame_id}.jpg"
        if path.exists():
            return
        # Оптимизация: сразу ресайз в PREVIEW_SIZE, без промежуточной копии
        preview = cv2.resize(frame, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(
            str(path), preview,
            [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_Q],
        )
    except Exception as e:
        logger.debug(f"Не удалось сохранить превью {frame_id}: {e}")


def _kp_to_list(kp_raw: Any) -> list | None:
    """
    Конвертирует keypoints в list[[x,y,conf],...] с валидацией формы.
    Принимает np.ndarray (17,3), list или None.
    """
    if kp_raw is None:
        return None
    
    # Конвертация в numpy для проверки формы
    try:
        if isinstance(kp_raw, np.ndarray):
            arr = kp_raw
        elif isinstance(kp_raw, list):
            arr = np.array(kp_raw)
        else:
            arr = np.array(list(kp_raw))
        
        # Валидация формы: должно быть (17, 3) или (N, 17, 3)
        if arr.ndim == 2 and arr.shape == (17, 3):
            return arr.tolist()
        elif arr.ndim == 3 and arr.shape[1:] == (17, 3):
            return arr[0].tolist()  # Берём первую позу
        else:
            logger.warning(f"Неверная форма keypoints: {arr.shape}")
            return None
    except Exception as e:
        logger.debug(f"Ошибка конвертации keypoints: {e}")
        return None


def _compute_file_hash(path: str) -> str:
    """Вычисляет MD5 хэш содержимого файла (первых 1MB для скорости)."""
    hasher = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            # Хэшируем первый 1MB для баланса скорости и уникальности
            chunk = f.read(1024 * 1024)
            hasher.update(chunk)
            # Добавляем размер файла для уникальности
            hasher.update(str(os.path.getsize(path)).encode())
    except Exception as e:
        logger.warning(f"Не удалось вычислить хэш {path}: {e}")
        # Fallback на хэш пути
        hasher.update(path.encode())
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

    DEFAULT_BATCH_SIZE = 64
    DEFAULT_CHUNK_SIZE = 5000

    def __init__(
        self,
        device: str | None = None,
        yolo:   YoloEngine | None = None,
    ) -> None:
        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo    = yolo if yolo is not None else YoloEngine(device=_device)
        self.matcher = MotionMatcher(device=_device)

        # Исправлено: добавлен Lock для защиты от race condition
        self._start_lock:    Lock  = Lock()
        self._running_event: Event = Event()
        self._stop_event:    Event = Event()
        self._thread: Thread | None = None

        self.BATCH_SIZE = self.DEFAULT_BATCH_SIZE
        self.CHUNK_SIZE = self.DEFAULT_CHUNK_SIZE

        self._progress_cb: Callable | None = None
        self._result_cb:   Callable | None = None
        self._error_cb:    Callable | None = None

        self.preview_cache_dir = Path("cache/previews")
        self.preview_cache_dir.mkdir(parents=True, exist_ok=True)

        self._query_poses:  list[np.ndarray] = []
        self._query_images: list[np.ndarray] = []
        self._search_mode:  SearchMode       = SearchMode.MOTION_MATCH

        # Исправлено: явная типизация вместо hasattr
        self._photo_matcher: PhotoMatcherProtocol | None = None

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
        # Исправлено: Lock защищает от одновременного запуска
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
        """
        Останавливает анализ с graceful shutdown.
        
        Args:
            timeout: Время ожидания завершения потока в секундах
        """
        self.analysis_running = False
        self._stop_event.set()
        
        # Исправлено: join для корректного завершения
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

    # ── Основной цикл — MOTION_MATCH ──────────────────────────────────────

    def _run_analysis(
        self,
        video_paths: list[str],
        settings:    dict[str, Any],
    ) -> None:
        """Основной цикл анализа для поиска повторяющихся движений."""
        # Исправлено: корректный маппинг quality
        quality_raw = settings.get("quality", "medium")
        quality = QUALITY_MAPPING.get(quality_raw, quality_raw)
        if quality not in QUALITY_FPS:
            quality = DEFAULT_QUALITY
        
        t_start = time.monotonic()
        result  = AnalysisResult(
            video_paths=list(video_paths),
            mode=SearchMode.MOTION_MATCH,
        )

        try:
            threshold  = settings.get("threshold", 70) / 100.0
            min_gap    = float(settings.get("scene_interval", 3))
            use_mirror = bool(settings.get("use_mirror", False))

            all_frames_data: list[dict]      = []
            video_metas:     list[VideoMeta] = []
            n_videos = len(video_paths)

            # ── Извлечение поз из всех видео ──────────────────────────────
            for v_idx, v_path in enumerate(video_paths):
                if self._stop_event.is_set():
                    break

                self._emit_progress(AnalysisProgress(
                    percent       = v_idx / n_videos * 70.0,
                    status        = f"Анализ: {os.path.basename(v_path)}",
                    status_code   = AnalysisStatus.ANALYZING_VIDEO,
                    video_idx     = v_idx,
                    video_count   = n_videos,
                    current_video = v_path,
                ))

                frames_data, vmeta = self._extract_poses_from_video(
                    path                = v_path,
                    video_idx           = v_idx,
                    quality             = quality,
                    n_videos            = n_videos,
                    base_progress_start = v_idx / n_videos * 70.0,
                    base_progress_end   = (v_idx + 1) / n_videos * 70.0,
                )
                all_frames_data.extend(frames_data)
                if vmeta:
                    video_metas.append(vmeta)
                
                # Проверка лимита
                if len(all_frames_data) > MAX_TOTAL_POSES:
                    logger.warning(
                        f"Достигнут лимит поз ({MAX_TOTAL_POSES}), "
                        f"остановка извлечения"
                    )
                    break

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

            # ── Сборка тензора поз ────────────────────────────────────────
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

            # ── Поиск совпадений ──────────────────────────────────────────
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

            # ── Фото-фильтрация (если включена) ───────────────────────────
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

            # ── Классификация движений ────────────────────────────────────
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
            # Исправлено: корректный маппинг quality
            quality_raw = settings.get("quality", "medium")
            quality = QUALITY_MAPPING.get(quality_raw, quality_raw)
            if quality not in QUALITY_FPS:
                quality = DEFAULT_QUALITY
            
            threshold = settings.get("threshold", 60) / 100.0
            n_videos  = len(video_paths)
            all_frames_data: list[dict] = []
            video_metas: list[VideoMeta] = []

            # ── Извлечение поз из видео ───────────────────────────────────
            for v_idx, v_path in enumerate(video_paths):
                if self._stop_event.is_set():
                    break

                self._emit_progress(AnalysisProgress(
                    percent       = v_idx / n_videos * 70.0,
                    status        = f"Поиск в: {os.path.basename(v_path)}",
                    status_code   = AnalysisStatus.ANALYZING_VIDEO,
                    video_idx     = v_idx,
                    video_count   = n_videos,
                    current_video = v_path,
                ))

                frames_data, vmeta = self._extract_poses_from_video(
                    path                = v_path,
                    video_idx           = v_idx,
                    quality             = quality,
                    n_videos            = n_videos,
                    base_progress_start = v_idx / n_videos * 70.0,
                    base_progress_end   = (v_idx + 1) / n_videos * 70.0,
                )
                all_frames_data.extend(frames_data)
                if vmeta:
                    video_metas.append(vmeta)

            result.video_metas = video_metas

            if self._stop_event.is_set():
                result.stopped = True
                self._finalize(result, t_start)
                return

            # ── Сравнение с query ─────────────────────────────────────────
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

            # ── Формирование результатов ──────────────────────────────────
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
            
            # Классификация для консистентности
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Извлечение поз из видео ───────────────────────────────────────────

    def _extract_poses_from_video(
        self,
        path:                str,
        video_idx:           int,
        quality:             str,
        n_videos:            int,
        base_progress_start: float = 0.0,
        base_progress_end:   float = 70.0,
    ) -> tuple[list[dict], VideoMeta | None]:
        """
        Извлекает позы из видео с оптимизацией памяти и корректным ETA.
        
        Исправлено:
        - Безопасный VideoCapture через context manager
        - Убрана бесполезная копия кадра
        - Исправлен расчёт ETA
        - Хэш по содержимому файла
        """
        # Исправлено: context manager для автоматического release
        with _video_capture(path) as cap:
            if not cap.isOpened():
                logger.warning(f"Не удалось открыть: {path}")
                return [], None

            fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            vmeta = VideoMeta(
                path=path, video_idx=video_idx,
                fps=fps, total_frames=total_frames,
                width=width, height=height,
            )

            # Адаптивный skip (формула оставлена как есть для совместимости)
            base_fps   = QUALITY_FPS[quality]
            pixel_load = (height * width * fps) / 1e6
            res_factor = min(2.0, max(0.5, pixel_load / 50.0))
            skip       = min(
                max(1, int((fps / base_fps) * res_factor)),
                MAX_ADAPTIVE_SKIP,
            )

            scale     = min(MAX_FRAME_SIDE / max(width, 1),
                            MAX_FRAME_SIDE / max(height, 1), 1.0)
            scaled_w  = max(1, int(width  * scale))
            scaled_h  = max(1, int(height * scale))
            dyn_batch = getattr(self.yolo, 'BATCH_SIZE_GPU', 64) if self.yolo.device == 'cuda' else getattr(self.yolo, 'BATCH_SIZE_CPU', 16)

            # Исправлено: хэш по содержимому файла
            video_hash = _compute_file_hash(path)
            frames_data: list[dict] = []

            _evict_preview_cache(self.preview_cache_dir, PREVIEW_CACHE_LIMIT)

            # Исправлено: корректная метрика для ETA
            fps_window:      deque[float] = deque(maxlen=ETA_WINDOW)
            t_prev           = time.monotonic()
            processed_count  = 0  # Счётчик обработанных кадров
            batch_frames:    list[np.ndarray] = []
            batch_frame_ids: list[int]        = []
            frame_idx        = 0

            def _flush_batch(
                bf:   list[np.ndarray],
                bids: list[int],
            ) -> None:
                """
                Обрабатывает батч кадров.
                
                Исправлено:
                - Убран параметр braw (raw_frame)
                - Превью сохраняется из исходного кадра (bf)
                """
                if not bf:
                    return
                try:
                    detections = self.yolo.detect_batch(bf)
                except Exception as exc:
                    logger.error(f"Ошибка detect_batch: {exc}")
                    return

                for det, frame, fid in zip(detections, bf, bids):
                    if det is None or not is_pose_valid(det):
                        continue

                    t_sec   = fid / fps
                    kp_list = _kp_to_list(det.get("kp") or det.get("keypoints"))

                    frames_data.append({
                        "t":         t_sec,
                        "f":         fid,
                        "video_idx": video_idx,
                        "dir":       det.get("direction", "forward"),
                        "scale":     det.get("scale",     1.0),
                        "anchor_y":  det.get("anchor_y",  0.5),
                        "kp":        kp_list,
                        "poses":     [det],
                    })

                    # Исправлено: превью из уже уменьшенного frame
                    _save_preview(
                        frame      = frame,
                        cache_dir  = self.preview_cache_dir,
                        video_hash = video_hash,
                        frame_id   = fid,
                    )

            # ── Основной цикл ─────────────────────────────────────────────
            while cap.isOpened() and not self._stop_event.is_set():
                grabbed = cap.grab()
                if not grabbed:
                    break

                if frame_idx % skip == 0:
                    ok, frame = cap.retrieve()
                    if ok and frame is not None:
                        # Исправлено: убрана бесполезная копия
                        small_frame = _resize_frame(frame, MAX_FRAME_SIDE)
                        batch_frames.append(small_frame)
                        batch_frame_ids.append(frame_idx)
                        processed_count += 1

                if len(batch_frames) >= dyn_batch:
                    _flush_batch(batch_frames, batch_frame_ids)
                    batch_frames    = []
                    batch_frame_ids = []

                frame_idx += 1

                # ── Обновление прогресса и ETA ────────────────────────────
                if frame_idx % 30 == 0 and total_frames > 0:
                    t_now  = time.monotonic()
                    dt     = t_now - t_prev
                    t_prev = t_now
                    
                    # Исправлено: корректный расчёт ETA
                    # fps_window = обработанные кадры / секунду
                    if dt > 0:
                        fps_window.append(30.0 / dt)

                    local_pct = frame_idx / total_frames
                    pct = (base_progress_start
                           + local_pct * (base_progress_end - base_progress_start))

                    eta: float | None = None
                    if fps_window and len(fps_window) >= 3:
                        avg_grab_fps = float(np.mean(list(fps_window)))
                        frames_left  = total_frames - frame_idx
                        eta = frames_left / max(avg_grab_fps, 1.0)

                    self._emit_progress(AnalysisProgress(
                        percent       = round(pct, 1),
                        status        = f"Извлечение: {os.path.basename(path)}",
                        status_code   = AnalysisStatus.EXTRACTING_POSES,
                        video_idx     = video_idx,
                        video_count   = n_videos,
                        current_frame = frame_idx,
                        total_frames  = total_frames,
                        current_video = path,
                        eta_seconds   = round(eta, 1) if eta is not None else None,
                    ))

            # Финальный flush
            if batch_frames and not self._stop_event.is_set():
                _flush_batch(batch_frames, batch_frame_ids)

            # ── Фильтр по фото (если включен) ─────────────────────────────
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

        # cap.release() вызывается автоматически через context manager
        
        logger.info(
            f"[{video_idx}] {os.path.basename(path)}: "
            f"{len(frames_data)} поз из {frame_idx} кадров "
            f"(skip={skip}, batch={dyn_batch}, обработано={processed_count})"
        )
        return frames_data, vmeta

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

        buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for m in matches:
            buckets[(_dir(m), _band(m.get("sim", 0.0)))].append(m)

        groups: list[MotionGroup] = []
        for (direction, band), ms in buckets.items():
            groups.append(MotionGroup(
                label     = (
                    f"{DIR_LABELS.get(direction, direction)} — "
                    f"{BAND_LABELS.get(band, band)}"
                ),
                direction = direction,
                sim_range = SIM_RANGES[band],
                matches   = sorted(
                    ms, key=lambda x: x.get("sim", 0.0), reverse=True),
            ))

        groups.sort(key=lambda g: g.count, reverse=True)
        return groups


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import threading

    print("=== AnalysisBackend smoke-test (исправленная версия) ===\n")

    done = threading.Event()

    def on_progress(p):
        if isinstance(p, AnalysisProgress):
            print(f"  [{p.percent:5.1f}%] {p.status}")

    def on_result(r):
        if isinstance(r, AnalysisResult):
            print(
                f"\n  Совпадений: {len(r.matches)} | "
                f"поз: {r.stats.get('total_poses','?')} | "
                f"время: {r.stats.get('elapsed_seconds','?')}s"
            )
        done.set()

    # Тесты датаклассов
    p = AnalysisProgress(percent=42.0, status_code=AnalysisStatus.MATCHING)
    assert p.percent == 42.0
    print(f"[OK] AnalysisProgress: {p.status_code}")

    r = AnalysisResult()
    assert r.matches == []
    print(f"[OK] AnalysisResult: stopped={r.stopped}")

    vm = VideoMeta("/tmp/t.mp4", 0, 30.0, 900, 1920, 1080)
    assert abs(vm.duration - 30.0) < 0.1
    print(f"[OK] VideoMeta: duration={vm.duration:.1f}s")

    mg = MotionGroup("T", "forward", (0.9, 1.0))
    mg.matches.append({"sim": 0.95})
    assert mg.count == 1
    print(f"[OK] MotionGroup: count={mg.count}")

    # Тест _kp_to_list с валидацией
    arr = np.zeros((17, 3))
    assert isinstance(_kp_to_list(arr), list)
    assert _kp_to_list(None) is None
    assert isinstance(_kp_to_list([[1, 2, 0.9]] * 17), list)
    
    # Невалидная форма
    invalid = np.zeros((10, 2))
    assert _kp_to_list(invalid) is None
    print("[OK] _kp_to_list с валидацией")

    # Тест quality mapping (исправлен)
    assert QUALITY_MAPPING["maximum"] == "Макс"
    assert "Макс" in QUALITY_FPS
    print("[OK] Quality mapping исправлен")

    # Тест _compute_file_hash
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(b"test data")
        tf.flush()
        h1 = _compute_file_hash(tf.name)
        h2 = _compute_file_hash(tf.name)
        assert h1 == h2
        os.unlink(tf.name)
    print("[OK] _compute_file_hash")

    # Тест context manager
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        temp_video = tf.name
    
    try:
        with _video_capture(temp_video) as cap:
            assert isinstance(cap, cv2.VideoCapture)
        print("[OK] _video_capture context manager")
    finally:
        if os.path.exists(temp_video):
            os.unlink(temp_video)

    print("\n[INFO] Базовые тесты пройдены.")
    print("[INFO] Для полного теста нужны реальные видеофайлы.")
    print("=== Smoke-test OK ===")