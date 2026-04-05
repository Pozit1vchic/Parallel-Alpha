#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnalysisBackend — фоновый движок анализа видео.
"""
from __future__ import annotations

import gc
import hashlib
import os
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from threading import Event, Thread
from typing import Any, Callable

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

DEFAULT_QUALITY     = "Средне"
PREVIEW_SIZE        = (320, 180)
PREVIEW_JPEG_Q      = 80
MAX_FRAME_SIDE      = 640
MAX_ADAPTIVE_SKIP   = 12
ETA_WINDOW          = 60
PREVIEW_CACHE_LIMIT = 500


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_callback_arity(fn: Callable | None) -> bool:
    if fn is None:
        return True
    import inspect
    try:
        return len(inspect.signature(fn).parameters) == 1
    except (ValueError, TypeError):
        return True


def _resize_frame(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w  = frame.shape[:2]
    scale = min(max_side / w, max_side / h, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(
        frame,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _evict_preview_cache(cache_dir: Path, limit: int) -> None:
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
    except Exception:
        pass


def _save_preview(
    frame:      np.ndarray,
    cache_dir:  Path,
    video_hash: str,
    frame_id:   int,
) -> None:
    try:
        path = cache_dir / f"{video_hash}_{frame_id}.jpg"
        if path.exists():
            return
        preview = cv2.resize(frame, PREVIEW_SIZE,
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(
            str(path), preview,
            [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_Q],
        )
    except Exception:
        pass


def _kp_to_list(kp_raw: Any) -> list | None:
    """
    Конвертирует keypoints в list[[x,y,conf],...].
    Принимает np.ndarray (17,3), list или None.
    """
    if kp_raw is None:
        return None
    if isinstance(kp_raw, np.ndarray):
        return kp_raw.tolist()
    if isinstance(kp_raw, list):
        return kp_raw
    try:
        return list(kp_raw)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Основной класс
# ═══════════════════════════════════════════════════════════════════════════════

class AnalysisBackend:
    """Фоновый движок анализа видео."""

    DEFAULT_BATCH_SIZE = 64
    DEFAULT_CHUNK_SIZE = 5000

    def __init__(
        self,
        device: str | None = None,
        yolo:   YoloEngine | None = None,
    ) -> None:
        _device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.yolo    = (yolo if yolo is not None
                        else YoloEngine(device=_device))
        self.matcher = MotionMatcher(device=_device)

        self._running_event: Event = Event()
        self._stop_event:    Event = Event()
        self._thread: Thread | None = None

        self.BATCH_SIZE = self.DEFAULT_BATCH_SIZE
        self.CHUNK_SIZE = self.DEFAULT_CHUNK_SIZE

        self._progress_cb:         Callable | None = None
        self._result_cb:           Callable | None = None
        self._error_cb:            Callable | None = None
        self._progress_new_format: bool = True
        self._result_new_format:   bool = True

        self.preview_cache_dir = Path("cache/previews")
        self.preview_cache_dir.mkdir(parents=True, exist_ok=True)

        self._query_poses:  list[np.ndarray] = []
        self._query_images: list[np.ndarray] = []
        self._search_mode:  SearchMode       = SearchMode.MOTION_MATCH

        # Фото-матчер — устанавливается из AnalysisController
        self._photo_matcher = None

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
        if self.analysis_running:
            print("[Backend] Анализ уже запущен.")
            return

        self._progress_cb         = progress_callback
        self._result_cb           = result_callback
        self._error_cb            = on_error
        self._progress_new_format = _detect_callback_arity(progress_callback)
        self._result_new_format   = _detect_callback_arity(result_callback)
        self._search_mode         = mode

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

    def stop_analysis(self) -> None:
        self.analysis_running = False
        self._stop_event.set()

    # ── Query ─────────────────────────────────────────────────────────────

    def set_query_poses(self, query_images: list[np.ndarray]) -> int:
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
            print(f"[Backend] set_query_poses error: {exc}")
        print(f"[Backend] Query-поз: {len(self._query_poses)}/{len(query_images)}")
        return len(self._query_poses)

    def clear_query(self) -> None:
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
        n = self.set_query_poses(query_images)
        if n == 0:
            print("[Backend] Нет поз в query-изображениях.")
            return 0
        self.start_analysis(
            video_paths, settings,
            progress_callback=progress_callback,
            result_callback=result_callback,
            on_error=on_error,
            mode=SearchMode.PERSON_SEARCH,
        )
        return n

    # ── Эмиттеры ─────────────────────────────────────────────────────────

    def _emit_progress(self, progress: AnalysisProgress) -> None:
        if not self._progress_cb:
            return
        try:
            if self._progress_new_format:
                self._progress_cb(progress)
            else:
                self._progress_cb(
                    progress.percent,
                    progress.status,
                    progress.current_frame,
                    progress.total_frames,
                    progress.current_video,
                )
        except Exception as exc:
            print(f"[Backend] progress_callback error: {exc}")

    def _emit_result(self, result: AnalysisResult) -> None:
        if not self._result_cb:
            return
        try:
            if self._result_new_format:
                self._result_cb(result)
            else:
                self._result_cb(
                    result.matches,
                    result.poses_meta,
                    result.video_paths,
                )
        except Exception as exc:
            print(f"[Backend] result_callback error: {exc}")

    def _emit_error(self, msg: str, exc: Exception | None = None) -> None:
        if self._error_cb:
            try:
                self._error_cb(msg, exc)
            except Exception:
                pass
        print(f"[Backend] ERROR: {msg}")
        if exc:
            traceback.print_exc()

    # ── Основной цикл — MOTION_MATCH ──────────────────────────────────────

    def _run_analysis(
        self,
        video_paths: list[str],
        settings:    dict[str, Any],
    ) -> None:
        quality_raw = settings.get("quality", "medium")
        _Q_TO_RU = {
            "fast":    "Быстро",
            "medium":  "Средне",
            "maximum": "Максимум",
        }
        quality = _Q_TO_RU.get(quality_raw, quality_raw)
        t_start = time.monotonic()
        result  = AnalysisResult(
            video_paths=list(video_paths),
            mode=SearchMode.MOTION_MATCH,
        )

        try:
            threshold  = settings.get("threshold", 70) / 100.0
            min_gap    = float(settings.get("scene_interval", 3))
            use_mirror = bool(settings.get("use_mirror", False))

            all_frames_data: list[dict]    = []
            video_metas:     list[VideoMeta] = []
            n_videos = len(video_paths)

            for v_idx, v_path in enumerate(video_paths):
                if self._stop_event.is_set():
                    break

                self._emit_progress(AnalysisProgress(
                    percent       = v_idx / n_videos * 70.0,
                    status        = (f"Анализ: "
                                    f"{os.path.basename(v_path)}"),
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

            if not all_frames_data:
                print("[Backend] Нет поз для матчинга.")
                result.stats["total_poses"] = 0
                self._finalize(result, t_start)
                return

            self._emit_progress(AnalysisProgress(
                percent     = 72.0,
                status      = "Сборка тензора поз…",
                status_code = AnalysisStatus.EXTRACTING_POSES,
                video_count = n_videos,
            ))

            # ── Диагностика frames_data ───────────────────────────────────
            if all_frames_data:
                fd0 = all_frames_data[0]
                print(
                    f"[Backend] frames_data[0] keys: "
                    f"{list(fd0.keys())} | "
                    f"kp present: {'kp' in fd0} | "
                    f"kp value type: {type(fd0.get('kp'))}")
                if fd0.get("kp") is not None:
                    arr = np.array(fd0["kp"])
                    print(
                        f"[Backend] frames_data[0].kp shape: "
                        f"{arr.shape}")

            poses_tensor, poses_meta = build_poses_tensor(
                all_frames_data, use_body_weights=True,
            )

            if poses_tensor is None or len(poses_meta) == 0:
                print("[Backend] Тензор пустой.")
                result.stats["total_poses"] = 0
                self._finalize(result, t_start)
                return

            result.poses_meta   = poses_meta
            result.poses_tensor = poses_tensor
            result.stats["total_poses"] = len(poses_meta)
            print(f"[Backend] Поз: {len(poses_meta)}")

            # ── Диагностика poses_meta ────────────────────────────────────
            if poses_meta:
                m0 = poses_meta[0]
                print(
                    f"[Backend] meta[0] keys: {list(m0.keys())} | "
                    f"kp present: {'kp' in m0}")
                if m0.get("kp") is not None:
                    arr = np.array(m0["kp"])
                    print(
                        f"[Backend] meta[0].kp shape: {arr.shape}")

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

            # ── Пост-фильтр по фото ───────────────────────────────────────
            if (hasattr(self, "_photo_matcher")
                    and self._photo_matcher is not None):
                try:
                    before  = len(matches)
                    matches = self._photo_matcher.filter_matches(
                        matches, threshold=0.7)
                    print(
                        f"[Backend] фото пост-фильтр: "
                        f"{before} → {len(matches)}")
                except Exception as e:
                    print(f"[Backend] фото фильтр ошибка: {e}")

            result.matches = matches
            result.stats["total_matches"] = len(matches)
            print(f"[Backend] Совпадений: {len(matches)}")

            # ── Диагностика матчей ────────────────────────────────────────
            if matches:
                m0 = matches[0]
                print(
                    f"[Backend] match[0] keys: {list(m0.keys())} | "
                    f"kp1: {'kp1' in m0} | kp2: {'kp2' in m0}")

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
            quality   = settings.get("quality", DEFAULT_QUALITY)
            threshold = settings.get("threshold", 60) / 100.0
            n_videos  = len(video_paths)
            all_frames_data: list[dict] = []

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

                frames_data, _ = self._extract_poses_from_video(
                    path                = v_path,
                    video_idx           = v_idx,
                    quality             = quality,
                    n_videos            = n_videos,
                    base_progress_start = v_idx / n_videos * 70.0,
                    base_progress_end   = (v_idx + 1) / n_videos * 70.0,
                )
                all_frames_data.extend(frames_data)

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

            device = self.matcher.device
            V = F.normalize(poses_tensor.float().to(device), p=2, dim=1)

            q_arr = np.stack(self._query_poses, axis=0)
            q_t   = F.normalize(
                torch.from_numpy(q_arr).float().to(device), p=2, dim=1)

            sims_all = torch.mm(V, q_t.t())
            sims     = sims_all.max(dim=1).values
            sims_np  = sims.cpu().numpy()

            candidates: list[dict] = []
            for i, (sim_val, meta) in enumerate(
                    zip(sims_np.tolist(), poses_meta)):
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
            result.stats["total_poses"]   = len(poses_meta)
            result.stats["total_matches"] = len(result.matches)

        except Exception as exc:
            result.error = str(exc)
            self._emit_error("Ошибка при поиске человека", exc)

        finally:
            self._finalize(result, t_start)

    # ── Финализация ───────────────────────────────────────────────────────

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
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[Backend] Не удалось открыть: {path}")
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

        base_fps   = QUALITY_FPS.get(
            quality, QUALITY_FPS[DEFAULT_QUALITY])
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
        dyn_batch = self.yolo.get_dynamic_batch_size(
            [(scaled_h, scaled_w)], self.BATCH_SIZE)

        video_hash   = hashlib.md5(path.encode()).hexdigest()[:12]
        frames_data: list[dict] = []

        _evict_preview_cache(
            self.preview_cache_dir, PREVIEW_CACHE_LIMIT)

        fps_window:      deque[float] = deque(maxlen=ETA_WINDOW)
        t_prev           = time.monotonic()
        batch_frames:    list[np.ndarray] = []
        batch_raw:       list[np.ndarray] = []
        batch_frame_ids: list[int]        = []
        frame_idx        = 0

        def _flush_batch(
            bf:   list[np.ndarray],
            braw: list[np.ndarray],
            bids: list[int],
        ) -> None:
            if not bf:
                return
            try:
                detections = self.yolo.detect_batch(bf)
            except Exception as exc:
                print(f"[Backend] detect_batch error: {exc}")
                return

            for det, raw_frame, fid in zip(detections, braw, bids):
                if det is None or not is_pose_valid(det):
                    continue

                t_sec   = fid / fps
                kp_list = _kp_to_list(
                    det.get("kp") or det.get("keypoints"))

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

                _save_preview(
                    frame      = raw_frame,
                    cache_dir  = self.preview_cache_dir,
                    video_hash = video_hash,
                    frame_id   = fid,
                )

        # ── Основной цикл ─────────────────────────────────────────────────
        while cap.isOpened() and not self._stop_event.is_set():
            grabbed = cap.grab()
            if not grabbed:
                break

            if frame_idx % skip == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    raw_frame   = frame.copy()
                    small_frame = _resize_frame(
                        frame, MAX_FRAME_SIDE)
                    batch_frames.append(small_frame)
                    batch_raw.append(raw_frame)
                    batch_frame_ids.append(frame_idx)

            if len(batch_frames) >= dyn_batch:
                _flush_batch(
                    batch_frames, batch_raw, batch_frame_ids)
                batch_frames    = []
                batch_raw       = []
                batch_frame_ids = []

            frame_idx += 1

            if frame_idx % 30 == 0 and total_frames > 0:
                t_now  = time.monotonic()
                dt     = t_now - t_prev
                t_prev = t_now
                fps_window.append(30.0 / max(dt, 1e-6))

                local_pct = frame_idx / total_frames
                pct = (base_progress_start
                       + local_pct
                       * (base_progress_end - base_progress_start))

                eta: float | None = None
                if fps_window:
                    avg_fps     = float(np.mean(list(fps_window)))
                    frames_left = ((total_frames - frame_idx)
                                   / max(skip, 1))
                    eta         = (frames_left
                                   / max(avg_fps * skip, 1.0))

                self._emit_progress(AnalysisProgress(
                    percent       = round(pct, 1),
                    status        = (f"Извлечение: "
                                    f"{os.path.basename(path)}"),
                    status_code   = AnalysisStatus.EXTRACTING_POSES,
                    video_idx     = video_idx,
                    video_count   = n_videos,
                    current_frame = frame_idx,
                    total_frames  = total_frames,
                    current_video = path,
                    eta_seconds   = (round(eta, 1)
                                    if eta is not None else None),
                ))

        if batch_frames and not self._stop_event.is_set():
            _flush_batch(
                batch_frames, batch_raw, batch_frame_ids)

        # ── Фильтр кадров по фото ─────────────────────────────────────────
        if (hasattr(self, "_photo_matcher")
                and self._photo_matcher is not None
                and frames_data):
            try:
                before      = len(frames_data)
                frames_data = (
                    self._photo_matcher
                    .filter_poses_by_reference(
                        frames_data,
                        threshold=0.50))
                print(
                    f"[Backend] фото-фильтр кадров "
                    f"видео {video_idx}: "
                    f"{before} → {len(frames_data)}")
            except Exception as e:
                print(f"[Backend] фото-фильтр ошибка: {e}")

        cap.release()
        print(
            f"[Backend] [{video_idx}] "
            f"{os.path.basename(path)}: "
            f"{len(frames_data)} поз, {frame_idx} кадров "
            f"(skip={skip}, batch={dyn_batch})")
        return frames_data, vmeta

    # ── Классификация движений ────────────────────────────────────────────

    def _build_motion_groups(
        self, matches: list[dict]
    ) -> list[MotionGroup]:
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

    print("=== AnalysisBackend smoke-test ===\n")

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

    assert _detect_callback_arity(lambda x: x) is True
    assert _detect_callback_arity(lambda a, b, c, d, e: None) is False
    print("[OK] _detect_callback_arity")

    arr = np.zeros((17, 3))
    assert isinstance(_kp_to_list(arr), list)
    assert _kp_to_list(None) is None
    assert isinstance(_kp_to_list([[1, 2, 0.9]] * 17), list)
    print("[OK] _kp_to_list")

    print("\n[INFO] Для полного теста нужны реальные видеофайлы.")
    print("=== Smoke-test OK ===")