from __future__ import annotations

import copy
import gzip
import hashlib
import logging
import pickle
import threading
import time
import traceback
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Iterable
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neural_verifier import SimpleReranker

import cv2
import numpy as np

try:
    from PySide6.QtCore import QObject, QThread, Signal
except Exception:  # pragma: no cover - compatibility fallback when Qt is unavailable
    class _SignalInstance:
        def __init__(self) -> None:
            self._callbacks: list[Any] = []

        def connect(self, callback: Any) -> None:
            self._callbacks.append(callback)

        def emit(self, *args: Any, **kwargs: Any) -> None:
            for callback in list(self._callbacks):
                callback(*args, **kwargs)

    class Signal:  # type: ignore[override]
        def __init__(self, *args: Any) -> None:
            self._storage_name = ""

        def __set_name__(self, owner: type[object], name: str) -> None:
            self._storage_name = f"__signal_{name}"

        def __get__(self, instance: object | None, owner: type[object]) -> Any:
            if instance is None:
                return self
            if self._storage_name not in instance.__dict__:
                instance.__dict__[self._storage_name] = _SignalInstance()
            return instance.__dict__[self._storage_name]

    class QObject:
        def __init__(self, parent: object | None = None) -> None:
            super().__init__()

    class QThread(threading.Thread):
        def __init__(self, parent: object | None = None) -> None:
            super().__init__(daemon=True)
            self._is_running = False

        def start(self) -> None:
            self._is_running = True
            super().start()

        def isRunning(self) -> bool:
            return self.is_alive()

        def wait(self, msecs: int = 0) -> bool:
            timeout = None if msecs <= 0 else msecs / 1000.0
            self.join(timeout)
            return not self.is_alive()

from core.engine import EngineConfig, MultiPoseData, YoloEngine
from core.matcher import MatchResult, MatcherConfig, MotionMatcher
from core.project import ProjectManager

LOGGER = logging.getLogger(__name__)
DEFAULT_CACHE_DIR = Path("cache") / "analysis"


class AnalysisSignals(QObject):
    """Worker signal bundle used by the Qt backend and tests."""

    progress = Signal(float, float, str)
    video_started = Signal(str)
    video_progress = Signal(str, float)
    video_finished = Signal(str, int)
    video_error = Signal(str, str)
    preview_ready = Signal(str, dict)
    results_ready = Signal(list)
    finished = Signal()


class AnalysisWorker(QThread):
    """Background worker that runs pose extraction, caching and matching."""

    def __init__(
        self,
        video_paths: list[str],
        engine_config: EngineConfig,
        matcher_config: MatcherConfig,
        model_path: str,
        cache_dir: Path,
        reference_path: str = "",
        tracked_person_id: int | str | None = None,
        tensorrt_path: str | None = None,
        device: str | None = None,
        use_tensorrt: bool = True,
    ) -> None:
        super().__init__()
        self.signals = AnalysisSignals()
        self.video_paths = [str(Path(path)) for path in video_paths]
        self.engine_config = copy.deepcopy(engine_config)
        self.matcher_config = copy.deepcopy(matcher_config)
        self.model_path = model_path
        from utils.constants import CACHE_DIR
        self.cache_dir = CACHE_DIR / "analysis"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.reference_path = str(reference_path) if reference_path else ""
        self.tracked_person_id = tracked_person_id
        self.tensorrt_path = tensorrt_path
        self.device = device
        self.use_tensorrt = use_tensorrt
        self.project_manager = ProjectManager(self.cache_dir.parent)
        self._stop_event = threading.Event()
        self._overall_start = 0.0
        self._runtime_dtw_batch_size: int | None = None
        self._fps_cache: dict[str, float] = {}

    def stop(self) -> None:
        """Request a graceful worker shutdown."""
        self._stop_event.set()

    def run(self) -> None:
        """Execute video processing and motion matching in a worker thread."""
        self._overall_start = time.perf_counter()
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            poses_by_video: dict[str, list[MultiPoseData]] = {}

            self._emit_progress(0.0, "Loading models...")
            engine = YoloEngine(
                model_path=self.model_path,
                tensorrt_path=self.tensorrt_path,
                device=self.device or "cuda",
                config=self.engine_config,
            )
            from neural_verifier import SimpleReranker
            reranker = SimpleReranker()
            matcher = MotionMatcher(
                config=self._build_matcher_config(),
                device=self.device,
                neural_reranker=reranker.rerank
                )


            ordered_paths = self._resolve_video_order()
            total_units = max(1, len(ordered_paths) + max(1, self._estimate_pair_count(ordered_paths)))
            completed_units = 0.0

            for video_path in ordered_paths:
                if self._stop_event.is_set():
                    break
                self.signals.video_started.emit(video_path)
                self._emit_progress((completed_units / total_units) * 100.0, f"Preparing {Path(video_path).name}")

                cached = self._load_from_cache(video_path)
                if cached is not None:
                    poses_by_video[video_path] = cached
                    completed_units += 1.0
                    self.signals.video_progress.emit(video_path, 100.0)
                    self.signals.video_finished.emit(video_path, len(cached))
                    self._emit_progress((completed_units / total_units) * 100.0, f"Loaded cache: {Path(video_path).name}")
                    continue

                def on_video_progress(progress: float) -> None:
                    local = float(np.clip(progress, 0.0, 1.0))
                    self.signals.video_progress.emit(video_path, local * 100.0)
                    overall = ((completed_units + local) / total_units) * 100.0
                    self._emit_progress(overall, f"Analyzing {Path(video_path).name}")

                poses = engine.process_video(video_path=video_path, progress_callback=on_video_progress)
                poses_by_video[video_path] = poses
                self._save_to_cache(video_path, poses)
                self._cache_previews_from_engine(engine, video_path, poses)
                completed_units += 1.0
                self.signals.video_finished.emit(video_path, len(poses))
                self._emit_progress((completed_units / total_units) * 100.0, f"Finished {Path(video_path).name}")

            if self._stop_event.is_set():
                return

            if not poses_by_video:
                self.signals.results_ready.emit([])
                return

            match_results: list[MatchResult] = []
            pair_total = max(1, self._estimate_pair_count(list(poses_by_video)))
            pair_done = 0
            for result_chunk in self._match_batches(matcher, poses_by_video):
                if self._stop_event.is_set():
                    break
                pair_done += 1
                match_results.extend(result_chunk)
                overall = ((completed_units + (pair_done / pair_total)) / total_units) * 100.0
                self._emit_progress(overall, "Verifying motion matches...")

            match_results.sort(key=lambda item: float(item.get("similarity", 0.0)), reverse=True)
            self.signals.results_ready.emit(match_results)
            self._emit_progress(100.0, "Analysis complete")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Analysis worker failed: %s", exc)
            self.signals.video_error.emit("", traceback.format_exc())
        finally:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            self.signals.finished.emit()

    def _build_matcher_config(self) -> MatcherConfig:
        """Clone matcher config and inject runtime tracking preferences."""
        config = copy.deepcopy(self.matcher_config)
        field_names = {field.name for field in fields(config)}
        if "tracked_person_id" in field_names and self.tracked_person_id is not None:
            config = replace(config, tracked_person_id=self.tracked_person_id)
        return config

    def _resolve_video_order(self) -> list[str]:
        """Build stable processing order, keeping reference first when provided."""
        unique: list[str] = []
        seen: set[str] = set()
        for value in ([self.reference_path] if self.reference_path else []) + self.video_paths:
            if value and value not in seen:
                unique.append(value)
                seen.add(value)
        return unique

    def _estimate_pair_count(self, ordered_paths: list[str]) -> int:
        """Estimate how many matcher passes are needed for progress reporting."""
        if len(ordered_paths) <= 1:
            return 1
        if self.reference_path and self.reference_path in ordered_paths:
            return max(1, len(ordered_paths) - 1)
        return max(1, len(ordered_paths) - 1)

    def _process_batch(self, video_paths: list[str]) -> list[MultiPoseData]:
        """Sequentially process a small batch of videos and flatten the outputs."""
        poses_flat: list[MultiPoseData] = []
        engine = YoloEngine(
            model_path=self.model_path,
            tensorrt_path=self.tensorrt_path,
            device=self.device or "cuda",
            config=self.engine_config,
        )
        for video_path in video_paths:
            if self._stop_event.is_set():
                break
            cached = self._load_from_cache(video_path)
            if cached is None:
                cached = engine.process_video(video_path)
                self._save_to_cache(video_path, cached)
            poses_flat.extend(cached)
        return poses_flat

    def _get_cache_key(self, video_path: str) -> str:
        """Build a stable cache key from path, file size and modification time."""
        resolved = Path(video_path).expanduser().resolve()
        stat = resolved.stat()
        payload = f"{resolved.as_posix()}::{stat.st_size}::{stat.st_mtime_ns}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    def _load_from_cache(self, video_path: str) -> list[MultiPoseData] | None:
        """Load cached pose extraction output if present and valid."""
        cache_file = self.cache_dir / f"{self._get_cache_key(video_path)}.pkl.gz"
        if not cache_file.exists():
            return None
        try:
            with gzip.open(cache_file, "rb") as fh:
                payload = pickle.load(fh)
            if not isinstance(payload, list):
                return None
            return payload
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Pose cache read failed for %s: %s", video_path, exc)
            try:
                cache_file.unlink(missing_ok=True)
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            return None

    def _save_to_cache(self, video_path: str, poses: list[MultiPoseData]) -> None:
        """Persist extracted poses for future runs."""
        cache_file = self.cache_dir / f"{self._get_cache_key(video_path)}.pkl.gz"
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(cache_file, "wb") as fh:
                pickle.dump(poses, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Pose cache write failed for %s: %s", video_path, exc)

    def _cache_previews_from_engine(self, engine: YoloEngine, video_path: str, poses: list[MultiPoseData]) -> None:
        """Copy a small preview sample from the engine into the project preview cache."""
        preview_cache = self.project_manager.get_preview_cache()
        previews: dict[int, np.ndarray] = {}
        sample_count = min(12, len(poses))
        sample_indices = list(range(sample_count))
        for frame_idx in sample_indices:
            try:
                preview = engine.get_preview(frame_idx)
            except Exception:  # pragma: no cover - best effort access
                preview = None
            if preview is None:
                continue
            preview_cache.put(video_path, frame_idx, preview)
            previews[frame_idx] = preview
        if previews:
            self.signals.preview_ready.emit(video_path, previews)

    def _match_batches(
        self,
        matcher: MotionMatcher,
        poses_by_video: dict[str, list[MultiPoseData]],
    ) -> Iterable[list[MatchResult]]:
        """Yield match results by video pair to keep memory usage predictable."""
        ordered_paths = list(poses_by_video.keys())
        if not ordered_paths:
            return

        if len(ordered_paths) == 1:
            only_path = ordered_paths[0]
            yield self._normalize_match_list(
                matcher.match(poses_by_video[only_path]),
                video_idx_a=0,
                video_idx_b=0,
                fps_a=self._video_fps(only_path),
                fps_b=self._video_fps(only_path),
            )
            return

        if self.reference_path and self.reference_path in poses_by_video:
            ref_path = self.reference_path
            ref_index = ordered_paths.index(ref_path)
            for target_path in ordered_paths:
                if target_path == ref_path:
                    continue
                target_index = ordered_paths.index(target_path)
                yield self._match_video_pair(
                    matcher,
                    poses_by_video[ref_path],
                    poses_by_video[target_path],
                    ref_index,
                    target_index,
                    self._video_fps(ref_path),
                    self._video_fps(target_path),
                )
            return

        base_path = ordered_paths[0]
        base_index = 0
        for target_index, target_path in enumerate(ordered_paths[1:], start=1):
            yield self._match_video_pair(
                matcher,
                poses_by_video[base_path],
                poses_by_video[target_path],
                base_index,
                target_index,
                self._video_fps(base_path),
                self._video_fps(target_path),
            )

    def _match_video_pair(
        self,
        matcher: MotionMatcher,
        poses_a: list[MultiPoseData],
        poses_b: list[MultiPoseData],
        video_idx_a: int,
        video_idx_b: int,
        fps_a: float,
        fps_b: float,
    ) -> list[MatchResult]:
        """Run pairwise matching by concatenating sequences with a neutral padding gap."""
        if not poses_a or not poses_b:
            return []
        pad_len = max(self._max_window_size(), 8)
        combined = list(poses_a) + self._padding_frames(start=len(poses_a), count=pad_len) + list(poses_b)
        raw_matches = matcher.match(combined)
        split = len(poses_a)
        return self._filter_cross_video_matches(
            raw_matches,
            split=split,
            pad_len=pad_len,
            video_idx_a=video_idx_a,
            video_idx_b=video_idx_b,
            fps_a=fps_a,
            fps_b=fps_b,
        )

    def _filter_cross_video_matches(
        self,
        matches: list[dict[str, Any]],
        *,
        split: int,
        pad_len: int,
        video_idx_a: int,
        video_idx_b: int,
        fps_a: float,
        fps_b: float,
    ) -> list[MatchResult]:
        """Keep only cross-video matches and remap frame/time coordinates."""
        out: list[MatchResult] = []
        pad_start = split
        pad_end = split + pad_len
        for match in matches:
            frame_i = int(match.get("frame_i", 0))
            frame_j = int(match.get("frame_j", 0))
            duration_frames = max(1, int(round(float(match.get("duration", 1.0)))))
            end_i = frame_i + duration_frames
            end_j = frame_j + duration_frames

            if self._overlaps_padding(frame_i, end_i, pad_start, pad_end) or self._overlaps_padding(frame_j, end_j, pad_start, pad_end):
                continue

            i_in_a = frame_i < split
            j_in_a = frame_j < split
            i_in_b = frame_i >= pad_end
            j_in_b = frame_j >= pad_end
            if not ((i_in_a and j_in_b) or (j_in_a and i_in_b)):
                continue

            if i_in_a:
                local_i = frame_i
                local_j = frame_j - pad_end
            else:
                local_i = frame_j
                local_j = frame_i - pad_end

            duration_sec = min(duration_frames / max(fps_a, 1e-6), duration_frames / max(fps_b, 1e-6))
            out.append(
                MatchResult(
                    t1=local_i / max(fps_a, 1e-6),
                    t2=local_j / max(fps_b, 1e-6),
                    duration=duration_sec,
                    similarity=float(match.get("similarity", 0.0)),
                    motion_type=int(match.get("motion_type", 0)),
                    motion_label=str(match.get("motion_label", "unknown")),
                    v1_idx=int(video_idx_a),
                    v2_idx=int(video_idx_b),
                    frame_i=int(local_i),
                    frame_j=int(local_j),
                    energy=float(match.get("energy", 0.0)),
                    source=str(match.get("source", "unknown")),
                )
            )
        out.sort(key=lambda item: float(item["similarity"]), reverse=True)
        return out

    def _normalize_match_list(
        self,
        matches: list[dict[str, Any]],
        *,
        video_idx_a: int,
        video_idx_b: int,
        fps_a: float,
        fps_b: float,
    ) -> list[MatchResult]:
        """Normalize matcher output into canonical MatchResult dictionaries."""
        normalized: list[MatchResult] = []
        for match in matches:
            frame_i = int(match.get("frame_i", 0))
            frame_j = int(match.get("frame_j", 0))
            duration = float(match.get("duration", 0.0))
            normalized.append(
                MatchResult(
                    t1=float(match.get("t1", frame_i / max(fps_a, 1e-6))),
                    t2=float(match.get("t2", frame_j / max(fps_b, 1e-6))),
                    duration=duration if duration <= 10.0 else duration / max(fps_a, fps_b, 1e-6),
                    similarity=float(match.get("similarity", 0.0)),
                    direction=int(match.get("direction", -1)),
                    context=int(match.get("context", 0)),
                    v1_idx=int(match.get("v1_idx", video_idx_a)),
                    v2_idx=int(match.get("v2_idx", video_idx_b)),
                    frame_i=frame_i,
                    frame_j=frame_j,
                    energy=float(match.get("energy", 0.0)),
                    cluster_id=int(match.get("cluster_id", -1)),
                )
            )
        normalized.sort(key=lambda item: float(item["similarity"]), reverse=True)
        return normalized

    def _padding_frames(self, start: int, count: int) -> list[MultiPoseData]:
        """Generate empty frames to separate concatenated videos for the matcher."""
        return [MultiPoseData(frame_idx=start + idx, people=[], track_ids=[]) for idx in range(max(0, count))]

    def _max_window_size(self) -> int:
        window_sizes = getattr(self.matcher_config, "window_sizes", (48,))
        try:
            return max(int(item) for item in window_sizes)
        except Exception:  # pragma: no cover - defensive fallback
            return 48

    def _video_fps(self, video_path: str) -> float:
        """Read and cache FPS with a safe default when metadata is missing."""
        if video_path in self._fps_cache:
            return self._fps_cache[video_path]
        fps = 30.0
        capture = cv2.VideoCapture(video_path)
        try:
            if capture.isOpened():
                candidate = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                if candidate > 1e-3:
                    fps = candidate
        except Exception:  # pragma: no cover - metadata lookup best effort
            pass
        finally:
            capture.release()
        self._fps_cache[video_path] = fps
        return fps

    def _emit_progress(self, percent: float, status: str) -> None:
        """Emit throttled overall progress with ETA estimation."""
        clipped = float(np.clip(percent, 0.0, 100.0))
        elapsed = max(0.0, time.perf_counter() - self._overall_start)
        fraction = clipped / 100.0
        eta = (elapsed / fraction - elapsed) if fraction > 1e-6 else 0.0
        self.signals.progress.emit(clipped, max(0.0, eta), status)

    @staticmethod
    def _overlaps_padding(start: int, end: int, pad_start: int, pad_end: int) -> bool:
        return max(start, pad_start) < min(end, pad_end)


class AnalysisBackend(QObject):
    """Qt-friendly backend bridge between the UI and the analysis pipeline."""

    analysisStarted = Signal()
    analysisFinished = Signal()
    analysisStopped = Signal()
    statusChanged = Signal(str)
    progressChanged = Signal(float, float, str)
    sessionStatusChanged = Signal(dict)
    resultsReady = Signal(list)
    resultsUpdated = Signal(list)
    previewFramesReady = Signal(dict)
    timelineChanged = Signal(dict)
    modelChanged = Signal(dict)
    errorOccurred = Signal(str)

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        super().__init__()
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            from utils.constants import CACHE_DIR
            self.cache_dir = CACHE_DIR / "analysis"

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.worker: AnalysisWorker | None = None
        self.current_results: list[MatchResult] = []
        self.current_model: dict[str, Any] = {
            "name": "YOLO Pose",
            "status": "Ready",
            "profile": "Balanced",
            "description": "Pose analysis backend compatible with MultiPoseData and MatchResult.",
            }

    def start_analysis(self, payload: dict[str, Any]) -> None:
        """Start a new analysis session from a UI payload."""
        if self.worker is not None and self.worker.isRunning():
            self.errorOccurred.emit("Analysis is already running")
            return

        videos = [str(item) for item in payload.get("videos", []) if str(item)]
        reference_path = str(payload.get("reference", payload.get("reference_path", "")))
        settings = payload.get("analysis_settings", {}) or {}
        model_path = str(payload.get("model_path", settings.get("model_path", "models/yolov8m-pose.pt")))
        tensorrt_path = payload.get("tensorrt_path", settings.get("tensorrt_path"))
        device = payload.get("device", settings.get("device"))
        tracked_person_id = payload.get("tracked_person_id", settings.get("tracked_person_id"))

        if not videos and not reference_path:
            self.errorOccurred.emit("No videos selected")
            return

        engine_config = self._build_engine_config(settings)
        matcher_config = self._build_matcher_config(settings)
        if hasattr(matcher_config, "tracked_person_id") and tracked_person_id is not None:
            try:
                matcher_config = replace(matcher_config, tracked_person_id=tracked_person_id)
            except Exception:
                setattr(matcher_config, "tracked_person_id", tracked_person_id)

        self.worker = AnalysisWorker(
            video_paths=videos,
            engine_config=engine_config,
            matcher_config=matcher_config,
            model_path=model_path,
            cache_dir=self.cache_dir,
            reference_path=reference_path,
            tracked_person_id=tracked_person_id,
            tensorrt_path=str(tensorrt_path) if tensorrt_path else None,
            device=str(device) if device else None,
            use_tensorrt=bool(settings.get("use_tensorrt", True)),
        )

        self._connect_worker(self.worker)
        self.analysisStarted.emit()
        self.statusChanged.emit("Analysis started")
        self.sessionStatusChanged.emit({"state": "running", "videos": videos, "reference_path": reference_path})
        self.modelChanged.emit(dict(self.current_model, model_path=model_path))
        self.worker.start()

    def stop_analysis(self) -> None:
        """Request worker shutdown without blocking the UI thread."""
        if self.worker is None:
            return
        if self.worker.isRunning():
            self.worker.stop()
            self.statusChanged.emit("Stopping analysis...")
            self.analysisStopped.emit()
            self.sessionStatusChanged.emit({"state": "stopping"})

    def _connect_worker(self, worker: AnalysisWorker) -> None:
        worker.signals.progress.connect(self._on_progress)
        worker.signals.video_started.connect(self._on_video_started)
        worker.signals.video_progress.connect(self._on_video_progress)
        worker.signals.video_finished.connect(self._on_video_finished)
        worker.signals.video_error.connect(self._on_video_error)
        worker.signals.preview_ready.connect(self._on_preview_ready)
        worker.signals.results_ready.connect(self._on_results_ready)
        worker.signals.finished.connect(self._on_finished)

    def _on_progress(self, percent: float, eta: float, status: str) -> None:
        self.progressChanged.emit(percent, eta, status)
        self.statusChanged.emit(status)
        self.sessionStatusChanged.emit({"state": "running", "progress": percent, "eta": eta, "status": status})

    def _on_video_started(self, video_path: str) -> None:
        self.timelineChanged.emit({"event": "video_started", "video_path": video_path})

    def _on_video_progress(self, video_path: str, percent: float) -> None:
        self.timelineChanged.emit({"event": "video_progress", "video_path": video_path, "progress": percent})

    def _on_video_finished(self, video_path: str, poses_count: int) -> None:
        self.timelineChanged.emit({"event": "video_finished", "video_path": video_path, "poses_count": poses_count})

    def _on_preview_ready(self, video_path: str, previews: dict[int, np.ndarray]) -> None:
        self.previewFramesReady.emit({"video_path": video_path, "frames": previews})

    def _on_video_error(self, video_path: str, error: str) -> None:
        prefix = f"Error processing {video_path}: " if video_path else "Analysis error: "
        self.errorOccurred.emit(prefix + error)
        self.sessionStatusChanged.emit({"state": "error", "video_path": video_path, "error": error})

    def _on_results_ready(self, results: list[MatchResult]) -> None:
        self.current_results = results
        self.resultsReady.emit(results)
        self.resultsUpdated.emit(results)
        self.sessionStatusChanged.emit({"state": "results", "count": len(results)})

    def _on_finished(self) -> None:
        self.analysisFinished.emit()
        self.statusChanged.emit("Analysis finished")
        self.sessionStatusChanged.emit({"state": "finished", "results": len(self.current_results)})
        self.worker = None

    def _build_engine_config(self, settings: dict[str, Any]) -> EngineConfig:
        """Build EngineConfig from UI settings with graceful fallbacks."""
        field_names = {field.name for field in fields(EngineConfig)}
        values = {name: settings[name] for name in field_names if name in settings}
        if "model_variant" not in values and "model_variant" in field_names:
            values["model_variant"] = settings.get("model_variant", "m")
        return EngineConfig(**values)

    def _build_matcher_config(self, settings: dict[str, Any]) -> MatcherConfig:
        """Build MatcherConfig from UI settings with graceful fallbacks."""
        field_names = {field.name for field in fields(MatcherConfig)}
        values = {name: settings[name] for name in field_names if name in settings}
        threshold = settings.get("similarity_threshold")
        if threshold is not None and "similarity_threshold" in field_names:
            values["similarity_threshold"] = float(threshold) / 100.0 if float(threshold) > 1.0 else float(threshold)
        return MatcherConfig(**values)


__all__ = ["AnalysisBackend", "AnalysisSignals", "AnalysisWorker"]
