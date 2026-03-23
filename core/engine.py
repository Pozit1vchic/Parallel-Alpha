from __future__ import annotations

import os
import re
from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import IntEnum
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Callable, Literal, Protocol, TypedDict

import cv2
import numpy as np
import numpy.typing as npt

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional at import time
    torch = None  # type: ignore[assignment]

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - ultralytics is optional at import time
    YOLO = None  # type: ignore[assignment]


# ============================================================================
# Core constants and types
# ============================================================================


class Coord(IntEnum):
    """Column indices for keypoint arrays shaped as (N, 3)."""

    X = 0
    Y = 1
    CONF = 2


class Keypoint(IntEnum):
    """COCO keypoint indices used by YOLO pose models."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


NUM_KEYPOINTS = len(Keypoint)
KEYPOINT_DATA_DIM = 3
POSE_XY_DIM = 2
EPSILON = 1e-6
DEFAULT_CPU_BATCH = 4
DEFAULT_DISABLED_TRACK_ID = -1

Variant = Literal["n", "s", "m", "l", "x"]
Direction = Literal[
    "forward",
    "forward-right",
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
    "forward-left",
    "unknown",
]

ArrayF32 = npt.NDArray[np.float32]
ArrayU8 = npt.NDArray[np.uint8]
FrameSize = tuple[int, int]
DecodeItem = tuple[int, ArrayU8, ArrayU8] | None


LEFT_SHOULDERS = (Keypoint.LEFT_SHOULDER, Keypoint.RIGHT_SHOULDER)
LEFT_HIPS = (Keypoint.LEFT_HIP, Keypoint.RIGHT_HIP)
TORSO_KEYPOINTS = (
    Keypoint.LEFT_SHOULDER,
    Keypoint.RIGHT_SHOULDER,
    Keypoint.LEFT_HIP,
    Keypoint.RIGHT_HIP,
)
DIRECTION_KEYPOINTS = (
    Keypoint.LEFT_SHOULDER,
    Keypoint.RIGHT_SHOULDER,
    Keypoint.LEFT_HIP,
    Keypoint.RIGHT_HIP,
    Keypoint.LEFT_KNEE,
    Keypoint.RIGHT_KNEE,
    Keypoint.LEFT_ANKLE,
    Keypoint.RIGHT_ANKLE,
)
DIRECTION_LABELS: tuple[Direction, ...] = (
    "forward",
    "forward-right",
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
    "forward-left",
)

MIRRORED_KEYPOINTS: dict[Keypoint, Keypoint] = {
    Keypoint.LEFT_EYE: Keypoint.RIGHT_EYE,
    Keypoint.RIGHT_EYE: Keypoint.LEFT_EYE,
    Keypoint.LEFT_EAR: Keypoint.RIGHT_EAR,
    Keypoint.RIGHT_EAR: Keypoint.LEFT_EAR,
    Keypoint.LEFT_SHOULDER: Keypoint.RIGHT_SHOULDER,
    Keypoint.RIGHT_SHOULDER: Keypoint.LEFT_SHOULDER,
    Keypoint.LEFT_ELBOW: Keypoint.RIGHT_ELBOW,
    Keypoint.RIGHT_ELBOW: Keypoint.LEFT_ELBOW,
    Keypoint.LEFT_WRIST: Keypoint.RIGHT_WRIST,
    Keypoint.RIGHT_WRIST: Keypoint.LEFT_WRIST,
    Keypoint.LEFT_HIP: Keypoint.RIGHT_HIP,
    Keypoint.RIGHT_HIP: Keypoint.LEFT_HIP,
    Keypoint.LEFT_KNEE: Keypoint.RIGHT_KNEE,
    Keypoint.RIGHT_KNEE: Keypoint.LEFT_KNEE,
    Keypoint.LEFT_ANKLE: Keypoint.RIGHT_ANKLE,
    Keypoint.RIGHT_ANKLE: Keypoint.LEFT_ANKLE,
}

INTERPOLATION_NEIGHBORS: dict[Keypoint, tuple[Keypoint, ...]] = {
    Keypoint.NOSE: (
        Keypoint.LEFT_EYE,
        Keypoint.RIGHT_EYE,
        Keypoint.LEFT_SHOULDER,
        Keypoint.RIGHT_SHOULDER,
    ),
    Keypoint.LEFT_EYE: (Keypoint.NOSE, Keypoint.LEFT_EAR),
    Keypoint.RIGHT_EYE: (Keypoint.NOSE, Keypoint.RIGHT_EAR),
    Keypoint.LEFT_EAR: (Keypoint.LEFT_EYE, Keypoint.LEFT_SHOULDER),
    Keypoint.RIGHT_EAR: (Keypoint.RIGHT_EYE, Keypoint.RIGHT_SHOULDER),
    Keypoint.LEFT_SHOULDER: (Keypoint.RIGHT_SHOULDER, Keypoint.LEFT_ELBOW, Keypoint.LEFT_HIP),
    Keypoint.RIGHT_SHOULDER: (Keypoint.LEFT_SHOULDER, Keypoint.RIGHT_ELBOW, Keypoint.RIGHT_HIP),
    Keypoint.LEFT_ELBOW: (Keypoint.LEFT_SHOULDER, Keypoint.LEFT_WRIST),
    Keypoint.RIGHT_ELBOW: (Keypoint.RIGHT_SHOULDER, Keypoint.RIGHT_WRIST),
    Keypoint.LEFT_WRIST: (Keypoint.LEFT_ELBOW,),
    Keypoint.RIGHT_WRIST: (Keypoint.RIGHT_ELBOW,),
    Keypoint.LEFT_HIP: (Keypoint.RIGHT_HIP, Keypoint.LEFT_SHOULDER, Keypoint.LEFT_KNEE),
    Keypoint.RIGHT_HIP: (Keypoint.LEFT_HIP, Keypoint.RIGHT_SHOULDER, Keypoint.RIGHT_KNEE),
    Keypoint.LEFT_KNEE: (Keypoint.LEFT_HIP, Keypoint.LEFT_ANKLE),
    Keypoint.RIGHT_KNEE: (Keypoint.RIGHT_HIP, Keypoint.RIGHT_ANKLE),
    Keypoint.LEFT_ANKLE: (Keypoint.LEFT_KNEE,),
    Keypoint.RIGHT_ANKLE: (Keypoint.RIGHT_KNEE,),
}


# ============================================================================
# Public data contracts
# ============================================================================


class PoseData(TypedDict):
    """Normalized information for one detected person in one frame."""

    frame_idx: int
    keypoints: ArrayF32
    normalized: ArrayF32
    confidence: float
    direction: Direction
    velocity: ArrayF32
    track_id: int | None


class MultiPoseData(TypedDict):
    """Pose results for all people detected in a frame."""

    frame_idx: int
    people: list[PoseData]
    track_ids: list[int]


FrameResult = tuple[int, MultiPoseData, ArrayU8]


# ============================================================================
# Configuration and internal state
# ============================================================================


@dataclass(slots=True)
class EngineConfig:
    """Runtime configuration for the YOLO pose engine."""

    # Model selection
    model_variant: Variant = "m"
    auto_select_model_variant: bool = True
    use_tensorrt: bool = True

    # Batching
    batch_size: int = 32
    tensorrt_batch: int = 32
    min_batch_size: int = 4
    max_batch_size: int = 96
    base_vram_gb_for_batch32: float = 8.0
    reference_frame_pixels: int = 1280 * 720
    batch_target_latency_ms: float = 35.0
    batch_growth_factor: float = 1.15
    batch_shrink_factor: float = 0.85

    # Pose quality
    keypoint_conf_threshold: float = 0.3
    min_visible_keypoints: int = 6
    interpolate_missing: bool = True
    direction_velocity_threshold: float = 0.015
    normalization_epsilon: float = EPSILON
    torso_scale_multiplier: float = 2.0

    # Performance
    use_fp16: bool = True
    use_cuda_streams: bool = True
    cuda_stream_count: int = 2
    preview_size: tuple[int, int] = (320, 180)
    preview_cache_size: int = 256
    decode_queue_size: int = 128
    result_queue_size: int = 128
    queue_timeout_sec: float = 0.1
    gpu_cleanup_interval: int = 8
    model_load_wait_timeout_sec: float = 300.0

    # Caching
    normalization_cache_size: int = 4096
    normalization_cache_quantization: float = 2.0

    # Tracking
    enable_tracking: bool = False
    max_track_lost_frames: int = 30
    track_match_threshold: float = 1.25
    track_center_weight: float = 0.65
    track_pose_weight: float = 0.35


@dataclass(slots=True)
class NormalizedPoseResult:
    """Internal normalized-pose payload reused across features."""

    normalized: ArrayF32
    center: ArrayF32
    scale: float
    filled_xy: ArrayF32


@dataclass(slots=True)
class PoseCandidate:
    """Intermediate representation for one detected person."""

    keypoints: ArrayF32
    keypoint_confidence: ArrayF32
    normalized: ArrayF32
    filled_xy: ArrayF32
    center: ArrayF32
    scale: float
    confidence: float


@dataclass(slots=True)
class TrackState:
    """Persistent state for frame-to-frame person association."""

    track_id: int
    normalized: ArrayF32
    keypoint_confidence: ArrayF32
    center: ArrayF32
    scale: float
    confidence: float
    last_frame_idx: int
    lost_frames: int = 0


# ============================================================================
# Exceptions and model protocol
# ============================================================================


class YoloEngineError(Exception):
    """Base exception for engine failures."""


class TensorRTError(YoloEngineError):
    """Raised when TensorRT execution fails."""


class DecodeError(YoloEngineError):
    """Raised when video decoding fails."""


class InferenceError(YoloEngineError):
    """Raised when model inference fails."""


class PoseModel(Protocol):
    """Common prediction interface shared by PyTorch and TensorRT YOLO models."""

    def predict(
        self,
        source: list[ArrayU8],
        verbose: bool = False,
        device: str | None = None,
        **kwargs: object,
    ) -> list[object]:
        ...


# ============================================================================
# Main engine
# ============================================================================


class YoloEngine:
    """High-throughput multi-person pose engine with TensorRT fallback support."""

    def __init__(
        self,
        model_path: str,
        tensorrt_path: str | None = None,
        device: str = "cuda",
        config: EngineConfig | None = None,
    ) -> None:
        """Initialize the engine and start asynchronous model loading."""

        self.model_path = model_path
        self.tensorrt_path = tensorrt_path
        self.device = device
        self.config = config or EngineConfig()

        self._pytorch_model: PoseModel | None = None
        self._tensorrt_model: PoseModel | None = None
        self._selected_variant: Variant = self.config.model_variant

        self._stop_event = Event()
        self._models_ready_event = Event()
        self._model_error: YoloEngineError | None = None
        self._model_loader_thread: Thread | None = None

        self._preview_cache: OrderedDict[int, ArrayU8] = OrderedDict()
        self._preview_lock = Lock()

        self._normalization_cache: OrderedDict[bytes, NormalizedPoseResult] = OrderedDict()
        self._normalization_lock = Lock()

        self._tracks: dict[int, TrackState] = {}
        self._next_track_id = 0

        self._perf_lock = Lock()
        self._backend_latency_ms: dict[str, deque[float]] = {
            "tensorrt": deque(maxlen=64),
            "pytorch": deque(maxlen=64),
        }
        self._processed_batches = 0

        self._cuda_streams: tuple[object, ...] = ()
        self._stream_cursor = 0

        self._load_models()

    def process_video(
        self,
        video_path: str,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[MultiPoseData]:
        """Decode a video, run pose inference, and return per-frame multi-person poses."""

        self._wait_for_models()
        self._reset_runtime_state(clear_previews=True)
        self._stop_event.clear()

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise DecodeError(f"Cannot open video: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        decode_queue: Queue[DecodeItem] = Queue(maxsize=self.config.decode_queue_size)
        result_queue: Queue[FrameResult | None] = Queue(maxsize=self.config.result_queue_size)
        error_queue: Queue[YoloEngineError] = Queue()
        frame_map: dict[int, MultiPoseData] = {}

        decoder = Thread(
            target=self._decode_worker,
            args=(capture, decode_queue, error_queue),
            daemon=True,
            name="yolo-engine-decoder",
        )
        gpu_worker = Thread(
            target=self._gpu_worker,
            args=(decode_queue, result_queue, error_queue),
            daemon=True,
            name="yolo-engine-gpu",
        )
        writer = Thread(
            target=self._writer_worker,
            args=(result_queue, frame_map, total_frames, progress_callback, error_queue),
            daemon=True,
            name="yolo-engine-writer",
        )

        decoder.start()
        gpu_worker.start()
        writer.start()

        decoder.join()
        gpu_worker.join()
        writer.join()
        capture.release()
        self._maybe_cleanup_gpu(force=True)

        if not error_queue.empty():
            raise error_queue.get()

        return [frame_map[idx] for idx in sorted(frame_map)]

    def get_preview(self, frame_idx: int) -> ArrayU8 | None:
        """Return a cached preview image for a processed frame, if available."""

        with self._preview_lock:
            preview = self._preview_cache.get(frame_idx)
            if preview is None:
                return None
            self._preview_cache.move_to_end(frame_idx)
            return preview.copy()

    def _load_models(self) -> None:
        """Start asynchronous model loading without blocking the UI thread."""

        if self._model_loader_thread is not None and self._model_loader_thread.is_alive():
            return

        self._models_ready_event.clear()
        self._model_error = None
        self._model_loader_thread = Thread(
            target=self._load_models_worker,
            daemon=True,
            name="yolo-engine-model-loader",
        )
        self._model_loader_thread.start()

    def _load_models_worker(self) -> None:
        """Load the PyTorch and optional TensorRT backends in a background thread."""

        try:
            if YOLO is None:
                raise YoloEngineError("ultralytics is required for YoloEngine")

            self._selected_variant = self._select_model_variant()
            resolved_model_path = self._resolve_model_path(self.model_path, self._selected_variant)
            resolved_tensorrt_path = self._resolve_model_path(self.tensorrt_path, self._selected_variant)

            try:
                self._pytorch_model = YOLO(resolved_model_path)
            except Exception as exc:  # noqa: BLE001
                raise YoloEngineError(f"Failed to load PyTorch YOLO model: {resolved_model_path}") from exc

            if self.config.use_tensorrt and resolved_tensorrt_path is not None:
                try:
                    self._tensorrt_model = YOLO(resolved_tensorrt_path)
                except Exception:
                    self._tensorrt_model = None

            self._initialize_cuda_streams()
        except YoloEngineError as exc:
            self._model_error = exc
        except Exception as exc:  # noqa: BLE001
            self._model_error = YoloEngineError(f"Unexpected model loading failure: {exc}")
        finally:
            self._models_ready_event.set()

    def _wait_for_models(self) -> None:
        """Wait until background model loading completes or fails."""

        ready = self._models_ready_event.wait(timeout=self.config.model_load_wait_timeout_sec)
        if not ready:
            raise YoloEngineError("Timed out while waiting for models to load")
        if self._model_error is not None:
            raise self._model_error
        if self._pytorch_model is None:
            raise YoloEngineError("PyTorch model is not available")

    def _initialize_cuda_streams(self) -> None:
        """Create CUDA streams used to overlap GPU work when supported."""

        if not self._should_use_cuda_streams():
            self._cuda_streams = ()
            return

        stream_count = max(1, self.config.cuda_stream_count)
        self._cuda_streams = tuple(torch.cuda.Stream() for _ in range(stream_count))  # type: ignore[union-attr]
        self._stream_cursor = 0

    def _should_use_cuda_streams(self) -> bool:
        """Return True when CUDA streams can be used safely."""

        return bool(
            self.config.use_cuda_streams
            and torch is not None
            and self.device != "cpu"
            and torch.cuda.is_available()  # type: ignore[union-attr]
        )

    def _select_model_variant(self) -> Variant:
        """Choose the most appropriate YOLO pose variant for the current hardware."""

        if not self.config.auto_select_model_variant:
            return self.config.model_variant

        if torch is None or self.device == "cpu" or not torch.cuda.is_available():  # type: ignore[union-attr]
            return "n"

        free_bytes, _ = torch.cuda.mem_get_info()  # type: ignore[union-attr]
        free_gb = free_bytes / 1e9

        if free_gb >= 18.0:
            return "x"
        if free_gb >= 12.0:
            return "l"
        if free_gb >= 8.0:
            return "m"
        if free_gb >= 5.0:
            return "s"
        return "n"

    def _resolve_model_path(self, path: str | None, variant: Variant) -> str | None:
        """Resolve a model path by applying the selected variant when possible."""

        if path is None:
            return None

        if "{variant}" in path:
            candidate = path.format(variant=variant)
            return candidate if os.path.exists(candidate) else path

        if os.path.isdir(path):
            extensions = (".engine", ".pt")
            candidates: list[str] = []
            for extension in extensions:
                candidates.extend(
                    [
                        os.path.join(path, f"yolo11{variant}-pose{extension}"),
                        os.path.join(path, f"yolov8{variant}-pose{extension}"),
                        os.path.join(path, f"yolo{variant}-pose{extension}"),
                    ]
                )
            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate
            return path

        basename = os.path.basename(path)
        dirname = os.path.dirname(path)
        variant_pattern = re.compile(r"(?P<prefix>yolo(?:v8|11)?)(?P<variant>[nslmx])(?P<suffix>-pose\.(?:pt|engine))$")
        match = variant_pattern.search(basename)
        if match is None:
            return path

        candidate_name = f"{match.group('prefix')}{variant}{match.group('suffix')}"
        candidate = os.path.join(dirname, candidate_name)
        return candidate if os.path.exists(candidate) else path

    def _decode_worker(
        self,
        capture: cv2.VideoCapture,
        output_queue: Queue[DecodeItem],
        error_queue: Queue[YoloEngineError],
    ) -> None:
        """Read frames, convert them to RGB, and push them to the decode queue."""

        frame_idx = 0
        try:
            while not self._stop_event.is_set():
                ok, frame_bgr = capture.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                preview = self._prepare_preview(frame_rgb)
                self._put_queue(output_queue, (frame_idx, frame_rgb, preview))
                frame_idx += 1
        except cv2.error as exc:
            self._stop_event.set()
            error_queue.put(DecodeError(f"Decoder thread failed: {exc}"))
        except YoloEngineError as exc:
            self._stop_event.set()
            error_queue.put(exc)
        finally:
            try:
                self._put_queue(output_queue, None)
            except YoloEngineError:
                pass

    def _gpu_worker(
        self,
        decode_queue: Queue[DecodeItem],
        result_queue: Queue[FrameResult | None],
        error_queue: Queue[YoloEngineError],
    ) -> None:
        """Build adaptive batches, run GPU inference, and emit per-frame pose results."""

        batch_items: list[tuple[int, ArrayU8, ArrayU8]] = []

        try:
            while not self._stop_event.is_set():
                item = self._get_queue(decode_queue)
                if item is None:
                    if batch_items:
                        self._process_batch(batch_items, result_queue)
                    break

                batch_items.append(item)
                current_sizes = [(frame.shape[0], frame.shape[1]) for _, frame, _ in batch_items]
                target_batch_size = self._adaptive_batch_size(current_sizes)

                if len(batch_items) >= target_batch_size:
                    self._process_batch(batch_items, result_queue)
                    batch_items = []
        except YoloEngineError as exc:
            self._stop_event.set()
            error_queue.put(exc)
        except Exception as exc:  # noqa: BLE001
            self._stop_event.set()
            error_queue.put(InferenceError(f"GPU worker failed: {exc}"))
        finally:
            try:
                self._put_queue(result_queue, None)
            except YoloEngineError:
                pass

    def _writer_worker(
        self,
        result_queue: Queue[FrameResult | None],
        frame_map: dict[int, MultiPoseData],
        total_frames: int,
        progress_callback: Callable[[float], None] | None,
        error_queue: Queue[YoloEngineError],
    ) -> None:
        """Persist frame results, cache previews, and report progress back to the caller."""

        processed = 0
        try:
            while not self._stop_event.is_set():
                item = result_queue.get()
                if item is None:
                    break

                frame_idx, frame_result, preview = item
                frame_map[frame_idx] = frame_result
                self._cache_preview(frame_idx, preview)
                processed += 1

                if progress_callback is not None and total_frames > 0:
                    progress_callback(min(processed / total_frames, 1.0))
        except Exception as exc:  # noqa: BLE001
            self._stop_event.set()
            error_queue.put(YoloEngineError(f"Writer thread failed: {exc}"))

    def _put_queue(self, queue_obj: Queue[object], item: object) -> None:
        """Put an item into a queue while honoring engine stop requests."""

        while not self._stop_event.is_set():
            try:
                queue_obj.put(item, timeout=self.config.queue_timeout_sec)
                return
            except Full:
                continue
        raise YoloEngineError("Queue put interrupted because the engine was stopped")

    def _get_queue(self, queue_obj: Queue[DecodeItem]) -> DecodeItem:
        """Get an item from a queue while honoring engine stop requests."""

        while not self._stop_event.is_set():
            try:
                return queue_obj.get(timeout=self.config.queue_timeout_sec)
            except Empty:
                continue
        raise YoloEngineError("Queue get interrupted because the engine was stopped")

    def _process_batch(
        self,
        batch_items: list[tuple[int, ArrayU8, ArrayU8]],
        result_queue: Queue[FrameResult | None],
    ) -> None:
        """Run inference for one prepared batch and emit frame-aligned results."""

        if not batch_items:
            return

        indices = [item[0] for item in batch_items]
        frames = [item[1] for item in batch_items]
        previews = [item[2] for item in batch_items]
        results = self._run_inference(frames)

        if len(results) != len(indices):
            raise InferenceError("Inference result count does not match input batch size")

        for frame_idx, preview, result in zip(indices, previews, results):
            frame_result = self._extract_poses(result, frame_idx)
            self._put_queue(result_queue, (frame_idx, frame_result, preview))

        self._processed_batches += 1
        self._maybe_cleanup_gpu(force=False)

    def _adaptive_batch_size(self, frame_sizes: list[FrameSize]) -> int:
        """Estimate a safe and efficient batch size from frame sizes and runtime pressure."""

        base_batch = float(self.config.batch_size)
        frame_sizes = frame_sizes or [(720, 1280)]
        average_pixels = float(np.mean([height * width for height, width in frame_sizes]))
        average_pixels = max(average_pixels, 1.0)

        size_factor = (self.config.reference_frame_pixels / average_pixels) ** 0.5
        size_factor = float(np.clip(size_factor, 0.5, 2.0))

        vram_factor = 1.0
        if torch is not None and self.device != "cpu" and torch.cuda.is_available():  # type: ignore[union-attr]
            free_bytes, _ = torch.cuda.mem_get_info()  # type: ignore[union-attr]
            free_gb = free_bytes / 1e9
            vram_factor = max(free_gb / max(self.config.base_vram_gb_for_batch32, EPSILON), 0.25)
            vram_factor = float(np.clip(vram_factor, 0.25, 2.5))
        else:
            return max(self.config.min_batch_size, min(self.config.max_batch_size, DEFAULT_CPU_BATCH))

        latency_factor = 1.0
        recent_ms = self._recent_backend_latency_ms()
        if recent_ms is not None:
            if recent_ms > self.config.batch_target_latency_ms:
                latency_factor = self.config.batch_shrink_factor
            elif recent_ms < self.config.batch_target_latency_ms * 0.6:
                latency_factor = self.config.batch_growth_factor

        suggested = int(round(base_batch * (vram_factor**0.5) * size_factor * latency_factor))
        return max(self.config.min_batch_size, min(self.config.max_batch_size, suggested))

    def _run_inference(self, frames: list[ArrayU8]) -> list[object]:
        """Run inference with TensorRT first and automatically fall back to PyTorch."""

        if not frames:
            return []

        if self._tensorrt_model is not None:
            try:
                return self._run_tensorrt_inference(frames)
            except TensorRTError:
                self._tensorrt_model = None

        if self._pytorch_model is None:
            raise InferenceError("No inference backend is available")

        try:
            return self._predict_in_chunks(
                model=self._pytorch_model,
                frames=frames,
                backend="pytorch",
                max_batch_size=max(1, min(len(frames), self.config.max_batch_size)),
                pad_to_batch=None,
            )
        except Exception as exc:  # noqa: BLE001
            raise InferenceError("PyTorch inference failed") from exc

    def _run_tensorrt_inference(self, frames: list[ArrayU8]) -> list[object]:
        """Run TensorRT inference with batch padding, profiling, and graceful degradation."""

        if self._tensorrt_model is None:
            raise TensorRTError("TensorRT model is unavailable")

        if not frames:
            return []

        try:
            return self._predict_in_chunks(
                model=self._tensorrt_model,
                frames=frames,
                backend="tensorrt",
                max_batch_size=max(1, self.config.tensorrt_batch),
                pad_to_batch=max(1, self.config.tensorrt_batch),
            )
        except Exception as exc:  # noqa: BLE001
            raise TensorRTError("TensorRT inference failed") from exc

    def _predict_in_chunks(
        self,
        model: PoseModel,
        frames: list[ArrayU8],
        backend: Literal["tensorrt", "pytorch"],
        max_batch_size: int,
        pad_to_batch: int | None,
    ) -> list[object]:
        """Predict over a list of frames while reusing the same batching path for all backends."""

        aggregated_results: list[object] = []
        chunk_size = max(1, max_batch_size)

        for start in range(0, len(frames), chunk_size):
            chunk = frames[start : start + chunk_size]
            original_len = len(chunk)
            predict_frames = chunk
            if pad_to_batch is not None and original_len < pad_to_batch:
                predict_frames = chunk + [chunk[-1]] * (pad_to_batch - original_len)

            batch_start = perf_counter()
            batch_results = self._predict_model(model, predict_frames, backend)
            elapsed_ms = (perf_counter() - batch_start) * 1000.0
            self._record_backend_latency(backend, elapsed_ms / max(original_len, 1))
            aggregated_results.extend(list(batch_results[:original_len]))

        return aggregated_results

    def _predict_model(
        self,
        model: PoseModel,
        frames: list[ArrayU8],
        backend: Literal["tensorrt", "pytorch"],
    ) -> list[object]:
        """Execute one backend prediction call, optionally under a CUDA stream."""

        predict_kwargs: dict[str, object] = {
            "source": frames,
            "verbose": False,
            "device": self.device,
        }
        if backend == "pytorch":
            predict_kwargs["half"] = self.config.use_fp16 and self.device != "cpu"

        stream = self._next_cuda_stream()
        if stream is None:
            return list(model.predict(**predict_kwargs))

        with torch.cuda.stream(stream):  # type: ignore[union-attr]
            results = list(model.predict(**predict_kwargs))
        stream.synchronize()
        return results

    def _next_cuda_stream(self) -> object | None:
        """Return the next CUDA stream in a round-robin fashion."""

        if not self._cuda_streams:
            return None

        stream = self._cuda_streams[self._stream_cursor % len(self._cuda_streams)]
        self._stream_cursor = (self._stream_cursor + 1) % len(self._cuda_streams)
        return stream

    def _record_backend_latency(self, backend: Literal["tensorrt", "pytorch"], latency_ms: float) -> None:
        """Record rolling backend latency measurements for adaptive batching."""

        with self._perf_lock:
            self._backend_latency_ms[backend].append(float(latency_ms))

    def _recent_backend_latency_ms(self) -> float | None:
        """Return the best available recent latency estimate across active backends."""

        with self._perf_lock:
            all_values = list(self._backend_latency_ms["tensorrt"]) + list(self._backend_latency_ms["pytorch"])
        if not all_values:
            return None
        return float(np.mean(all_values))

    def _extract_poses(self, result: object, frame_idx: int) -> MultiPoseData:
        """Extract all people from a model result, compute velocity, and assign track IDs."""

        candidates = self._extract_people_candidates(result)
        if not candidates:
            self._advance_tracks_without_matches()
            return {
                "frame_idx": frame_idx,
                "people": [],
                "track_ids": [],
            }

        internal_ids, output_ids, velocities = self._assign_tracks(candidates, frame_idx)
        people: list[PoseData] = []

        for candidate, internal_id, velocity in zip(candidates, internal_ids, velocities):
            direction = self.classify_direction(candidate.keypoints, velocity)
            people.append(
                {
                    "frame_idx": frame_idx,
                    "keypoints": candidate.keypoints.copy(),
                    "normalized": candidate.normalized.copy(),
                    "confidence": float(candidate.confidence),
                    "direction": direction,
                    "velocity": velocity.copy(),
                    "track_id": internal_id if self.config.enable_tracking else None,
                }
            )

        return {
            "frame_idx": frame_idx,
            "people": people,
            "track_ids": output_ids,
        }

    def _extract_people_candidates(self, result: object) -> list[PoseCandidate]:
        """Convert raw model output into filtered pose candidates for all detected people."""

        keypoints_array = self._parse_keypoints(result)
        if keypoints_array is None or keypoints_array.size == 0:
            return []

        candidates: list[PoseCandidate] = []
        for keypoints in keypoints_array.astype(np.float32):
            keypoint_confidence = np.clip(keypoints[:, Coord.CONF], 0.0, 1.0).astype(np.float32)
            visible_count = int(np.count_nonzero(keypoint_confidence >= self.config.keypoint_conf_threshold))
            if visible_count < self.config.min_visible_keypoints:
                continue

            confidence = float(np.mean(keypoint_confidence[keypoint_confidence >= self.config.keypoint_conf_threshold]))
            if confidence < self.config.keypoint_conf_threshold:
                continue

            normalized_details = self._normalize_pose_details(keypoints)
            candidates.append(
                PoseCandidate(
                    keypoints=keypoints.astype(np.float32),
                    keypoint_confidence=keypoint_confidence,
                    normalized=normalized_details.normalized,
                    filled_xy=normalized_details.filled_xy,
                    center=normalized_details.center,
                    scale=normalized_details.scale,
                    confidence=confidence,
                )
            )

        candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
        return candidates

    def _parse_keypoints(self, result: object) -> ArrayF32 | None:
        """Safely extract a (people, 17, 3) float32 keypoint tensor from a YOLO result."""

        if not hasattr(result, "keypoints"):
            return None

        keypoints_obj = getattr(result, "keypoints")
        if not hasattr(keypoints_obj, "data"):
            return None

        data = getattr(keypoints_obj, "data")
        if torch is not None and isinstance(data, torch.Tensor):
            array = data.detach().float().cpu().numpy()
        elif isinstance(data, np.ndarray):
            array = data.astype(np.float32)
        else:
            return None

        if array.ndim != 3 or array.shape[1] != NUM_KEYPOINTS or array.shape[2] < KEYPOINT_DATA_DIM:
            return None

        return array[:, :, :KEYPOINT_DATA_DIM].astype(np.float32)

    def normalize_pose(self, keypoints: ArrayF32) -> ArrayF32:
        """Normalize a pose around a torso-aware center while preserving body proportions."""

        return self._normalize_pose_details(keypoints).normalized.copy()

    def _normalize_pose_details(self, keypoints: ArrayF32) -> NormalizedPoseResult:
        """Return normalized coordinates together with the center, scale, and interpolated XY."""

        cache_key = self._make_normalization_cache_key(keypoints)
        cached = self._get_normalization_cache(cache_key)
        if cached is not None:
            return cached

        keypoints_f32 = keypoints.astype(np.float32, copy=False)
        xy = keypoints_f32[:, :POSE_XY_DIM]
        conf = np.clip(keypoints_f32[:, Coord.CONF], 0.0, 1.0).astype(np.float32)

        filled_xy = self._interpolate_missing_keypoints(xy, conf) if self.config.interpolate_missing else xy.copy()
        center = self._estimate_body_center(filled_xy, conf)
        scale = self._estimate_body_scale(filled_xy, conf, center)
        normalized = ((filled_xy - center) / max(scale, self.config.normalization_epsilon)).astype(np.float32)

        result = NormalizedPoseResult(
            normalized=normalized,
            center=center.astype(np.float32),
            scale=float(scale),
            filled_xy=filled_xy.astype(np.float32),
        )
        self._set_normalization_cache(cache_key, result)
        return self._clone_normalized_pose_result(result)

    def _make_normalization_cache_key(self, keypoints: ArrayF32) -> bytes:
        """Create a compact hashable cache key for pose normalization."""

        quantized = np.round(keypoints.astype(np.float32) * self.config.normalization_cache_quantization)
        return quantized.astype(np.int16, copy=False).tobytes()

    def _get_normalization_cache(self, cache_key: bytes) -> NormalizedPoseResult | None:
        """Return a cached normalization result if it exists."""

        with self._normalization_lock:
            cached = self._normalization_cache.get(cache_key)
            if cached is None:
                return None
            self._normalization_cache.move_to_end(cache_key)
            return self._clone_normalized_pose_result(cached)

    def _set_normalization_cache(self, cache_key: bytes, result: NormalizedPoseResult) -> None:
        """Store a normalization result inside the bounded LRU cache."""

        with self._normalization_lock:
            self._normalization_cache[cache_key] = self._clone_normalized_pose_result(result)
            self._normalization_cache.move_to_end(cache_key)
            while len(self._normalization_cache) > self.config.normalization_cache_size:
                self._normalization_cache.popitem(last=False)

    def _clone_normalized_pose_result(self, result: NormalizedPoseResult) -> NormalizedPoseResult:
        """Clone cached normalization arrays to avoid accidental shared mutation."""

        return NormalizedPoseResult(
            normalized=result.normalized.copy(),
            center=result.center.copy(),
            scale=float(result.scale),
            filled_xy=result.filled_xy.copy(),
        )

    def _interpolate_missing_keypoints(self, xy: ArrayF32, conf: ArrayF32) -> ArrayF32:
        """Fill missing keypoints using torso geometry, mirrored joints, and skeleton neighbors."""

        filled = xy.astype(np.float32, copy=True)
        visible = conf >= self.config.keypoint_conf_threshold
        center = self._estimate_body_center(filled, conf)

        for keypoint in Keypoint:
            if visible[keypoint]:
                continue

            mirrored = MIRRORED_KEYPOINTS.get(keypoint)
            if mirrored is not None and visible[mirrored]:
                filled[keypoint, Coord.X] = (2.0 * center[Coord.X]) - filled[mirrored, Coord.X]
                filled[keypoint, Coord.Y] = filled[mirrored, Coord.Y]
                continue

            neighbors = [filled[neighbor] for neighbor in INTERPOLATION_NEIGHBORS.get(keypoint, ()) if visible[neighbor]]
            if len(neighbors) >= 2:
                filled[keypoint] = np.mean(np.stack(neighbors, axis=0), axis=0).astype(np.float32)
                continue
            if len(neighbors) == 1:
                filled[keypoint] = ((neighbors[0] * 0.75) + (center * 0.25)).astype(np.float32)
                continue

            filled[keypoint] = center.astype(np.float32)

        return filled.astype(np.float32)

    def _estimate_body_center(self, xy: ArrayF32, conf: ArrayF32) -> ArrayF32:
        """Estimate a robust body center using shoulders and pelvis with sensible fallbacks."""

        threshold = self.config.keypoint_conf_threshold
        visible = conf >= threshold

        shoulder_points = [xy[keypoint] for keypoint in LEFT_SHOULDERS if visible[keypoint]]
        hip_points = [xy[keypoint] for keypoint in LEFT_HIPS if visible[keypoint]]

        shoulder_center = np.mean(np.stack(shoulder_points, axis=0), axis=0) if shoulder_points else None
        pelvis_center = np.mean(np.stack(hip_points, axis=0), axis=0) if hip_points else None

        if shoulder_center is not None and pelvis_center is not None:
            return ((shoulder_center + pelvis_center) * 0.5).astype(np.float32)
        if pelvis_center is not None:
            return pelvis_center.astype(np.float32)
        if shoulder_center is not None:
            return shoulder_center.astype(np.float32)

        visible_points = xy[visible]
        if visible_points.size > 0:
            return np.mean(visible_points, axis=0).astype(np.float32)
        return np.zeros((POSE_XY_DIM,), dtype=np.float32)

    def _estimate_body_scale(self, xy: ArrayF32, conf: ArrayF32, center: ArrayF32) -> float:
        """Estimate a stable normalization scale while preserving relative body proportions."""

        threshold = self.config.keypoint_conf_threshold
        visible = conf >= threshold
        scales: list[float] = []

        if visible[Keypoint.LEFT_SHOULDER] and visible[Keypoint.RIGHT_SHOULDER]:
            scales.append(float(np.linalg.norm(xy[Keypoint.LEFT_SHOULDER] - xy[Keypoint.RIGHT_SHOULDER])))
        if visible[Keypoint.LEFT_HIP] and visible[Keypoint.RIGHT_HIP]:
            scales.append(float(np.linalg.norm(xy[Keypoint.LEFT_HIP] - xy[Keypoint.RIGHT_HIP])))

        shoulder_points = [xy[keypoint] for keypoint in LEFT_SHOULDERS if visible[keypoint]]
        hip_points = [xy[keypoint] for keypoint in LEFT_HIPS if visible[keypoint]]
        if shoulder_points and hip_points:
            shoulder_center = np.mean(np.stack(shoulder_points, axis=0), axis=0)
            hip_center = np.mean(np.stack(hip_points, axis=0), axis=0)
            torso_height = float(np.linalg.norm(shoulder_center - hip_center))
            scales.append(torso_height * self.config.torso_scale_multiplier)

        visible_points = xy[visible]
        if visible_points.size > 0:
            bbox_extent = float(np.max(np.ptp(visible_points, axis=0)))
            if bbox_extent > 0.0:
                scales.append(bbox_extent)

        radial_extent = float(np.max(np.linalg.norm(xy - center, axis=1))) if xy.size > 0 else 0.0
        if radial_extent > 0.0:
            scales.append(radial_extent)

        valid_scales = [scale for scale in scales if scale > self.config.normalization_epsilon]
        if not valid_scales:
            return 1.0

        return max(float(np.median(np.asarray(valid_scales, dtype=np.float32))), self.config.normalization_epsilon)

    def _assign_tracks(
        self,
        candidates: list[PoseCandidate],
        frame_idx: int,
    ) -> tuple[list[int], list[int], list[ArrayF32]]:
        """Associate current people with prior tracks and compute confidence-aware velocities."""

        internal_ids = [DEFAULT_DISABLED_TRACK_ID] * len(candidates)
        output_ids = [DEFAULT_DISABLED_TRACK_ID] * len(candidates)
        velocities = [np.zeros((NUM_KEYPOINTS, POSE_XY_DIM), dtype=np.float32) for _ in candidates]

        active_track_ids = list(self._tracks.keys())
        candidate_track_costs: list[tuple[float, int, int]] = []
        for candidate_idx, candidate in enumerate(candidates):
            for track_id in active_track_ids:
                track = self._tracks[track_id]
                distance = self._pose_match_distance(candidate, track)
                if distance <= self.config.track_match_threshold:
                    candidate_track_costs.append((distance, candidate_idx, track_id))

        candidate_track_costs.sort(key=lambda item: item[0])
        used_candidates: set[int] = set()
        used_tracks: set[int] = set()

        for _, candidate_idx, track_id in candidate_track_costs:
            if candidate_idx in used_candidates or track_id in used_tracks:
                continue

            track = self._tracks[track_id]
            frame_gap = max(1, frame_idx - track.last_frame_idx)
            velocities[candidate_idx] = self._compute_velocity(
                current=candidates[candidate_idx].normalized,
                current_confidence=candidates[candidate_idx].keypoint_confidence,
                previous=track.normalized,
                previous_confidence=track.keypoint_confidence,
                frame_gap=frame_gap,
            )
            internal_ids[candidate_idx] = track_id
            used_candidates.add(candidate_idx)
            used_tracks.add(track_id)

        for track_id, track in list(self._tracks.items()):
            if track_id not in used_tracks:
                track.lost_frames += 1

        for candidate_idx, candidate in enumerate(candidates):
            track_id = internal_ids[candidate_idx]
            if track_id == DEFAULT_DISABLED_TRACK_ID:
                track_id = self._next_track_id
                self._next_track_id += 1
                internal_ids[candidate_idx] = track_id

            self._tracks[track_id] = self._create_track_state(candidate, track_id, frame_idx)
            output_ids[candidate_idx] = track_id if self.config.enable_tracking else DEFAULT_DISABLED_TRACK_ID

        self._purge_stale_tracks()
        return internal_ids, output_ids, velocities

    def _pose_match_distance(self, candidate: PoseCandidate, track: TrackState) -> float:
        """Compute a scale-aware matching distance between a candidate and an existing track."""

        effective_scale = max((candidate.scale + track.scale) * 0.5, self.config.normalization_epsilon)
        center_distance = float(np.linalg.norm(candidate.center - track.center) / effective_scale)

        weights = np.sqrt(np.clip(candidate.keypoint_confidence, 0.0, 1.0) * np.clip(track.keypoint_confidence, 0.0, 1.0))
        weights = weights.astype(np.float32)
        shared = weights >= self.config.keypoint_conf_threshold

        if np.any(shared):
            delta = candidate.normalized[shared] - track.normalized[shared]
            per_point_distance = np.linalg.norm(delta, axis=1)
            pose_distance = float(np.average(per_point_distance, weights=weights[shared]))
        else:
            pose_distance = center_distance

        return (self.config.track_center_weight * center_distance) + (self.config.track_pose_weight * pose_distance)

    def _create_track_state(self, candidate: PoseCandidate, track_id: int, frame_idx: int) -> TrackState:
        """Create a new persistent track state from the current candidate."""

        return TrackState(
            track_id=track_id,
            normalized=candidate.normalized.copy(),
            keypoint_confidence=candidate.keypoint_confidence.copy(),
            center=candidate.center.copy(),
            scale=float(candidate.scale),
            confidence=float(candidate.confidence),
            last_frame_idx=frame_idx,
            lost_frames=0,
        )

    def _advance_tracks_without_matches(self) -> None:
        """Age existing tracks when a frame contains no valid people."""

        for track in self._tracks.values():
            track.lost_frames += 1
        self._purge_stale_tracks()

    def _purge_stale_tracks(self) -> None:
        """Remove tracks that have been lost for too long."""

        loss_limit = self.config.max_track_lost_frames if self.config.enable_tracking else 1
        stale_track_ids = [track_id for track_id, track in self._tracks.items() if track.lost_frames > loss_limit]
        for track_id in stale_track_ids:
            self._tracks.pop(track_id, None)

    def _compute_velocity(
        self,
        current: ArrayF32,
        current_confidence: ArrayF32,
        previous: ArrayF32,
        previous_confidence: ArrayF32,
        frame_gap: int,
    ) -> ArrayF32:
        """Compute per-keypoint velocity weighted by current and previous confidence."""

        if previous.shape != current.shape:
            return np.zeros_like(current, dtype=np.float32)

        delta = (current - previous) / max(frame_gap, 1)
        confidence_weight = np.sqrt(np.clip(current_confidence, 0.0, 1.0) * np.clip(previous_confidence, 0.0, 1.0))
        confidence_weight = np.where(confidence_weight >= self.config.keypoint_conf_threshold, confidence_weight, 0.0)
        return (delta * confidence_weight[:, None]).astype(np.float32)

    def classify_direction(self, keypoints: ArrayF32, velocity: ArrayF32) -> Direction:
        """Classify movement direction using eight compass sectors and confidence-aware motion."""

        motion_vector = self._weighted_motion_vector(keypoints, velocity)
        motion_norm = float(np.linalg.norm(motion_vector))
        if motion_norm < self.config.direction_velocity_threshold:
            return "unknown"

        angle = float(np.arctan2(motion_vector[Coord.X], -motion_vector[Coord.Y]))
        full_turn = float(2.0 * np.pi)
        sector_angle = full_turn / len(DIRECTION_LABELS)
        normalized_angle = (angle + full_turn) % full_turn
        sector_index = int(np.floor((normalized_angle + (sector_angle * 0.5)) / sector_angle)) % len(DIRECTION_LABELS)
        return DIRECTION_LABELS[sector_index]

    def _weighted_motion_vector(self, keypoints: ArrayF32, velocity: ArrayF32) -> ArrayF32:
        """Aggregate keypoint motion into a single 2D direction vector."""

        confidence = np.clip(keypoints[:, Coord.CONF], 0.0, 1.0).astype(np.float32)
        priority_indices = np.asarray([int(keypoint) for keypoint in DIRECTION_KEYPOINTS], dtype=np.int32)
        priority_weights = confidence[priority_indices]

        if float(np.sum(priority_weights)) > self.config.normalization_epsilon:
            vector = np.average(velocity[priority_indices], axis=0, weights=priority_weights)
            return vector.astype(np.float32)

        if float(np.sum(confidence)) > self.config.normalization_epsilon:
            vector = np.average(velocity, axis=0, weights=confidence)
            return vector.astype(np.float32)

        return np.zeros((POSE_XY_DIM,), dtype=np.float32)

    def _prepare_preview(self, frame: ArrayU8) -> ArrayU8:
        """Resize a frame to preview resolution for fast UI rendering."""

        preview_width, preview_height = self.config.preview_size
        return cv2.resize(frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA).astype(np.uint8)

    def _cache_preview(self, frame_idx: int, preview: ArrayU8) -> None:
        """Store a preview inside a bounded LRU cache."""

        with self._preview_lock:
            self._preview_cache[frame_idx] = preview.copy()
            self._preview_cache.move_to_end(frame_idx)
            while len(self._preview_cache) > self.config.preview_cache_size:
                self._preview_cache.popitem(last=False)

    def _clear_preview_cache(self) -> None:
        """Clear all cached previews."""

        with self._preview_lock:
            self._preview_cache.clear()

    def _reset_runtime_state(self, clear_previews: bool) -> None:
        """Reset transient runtime state before starting a new video session."""

        self._tracks.clear()
        self._next_track_id = 0
        self._processed_batches = 0
        if clear_previews:
            self._clear_preview_cache()

    def _maybe_cleanup_gpu(self, force: bool) -> None:
        """Release cached GPU memory periodically to reduce fragmentation."""

        if torch is None or self.device == "cpu" or not torch.cuda.is_available():  # type: ignore[union-attr]
            return

        if force or (self._processed_batches % max(self.config.gpu_cleanup_interval, 1) == 0):
            torch.cuda.empty_cache()  # type: ignore[union-attr]
