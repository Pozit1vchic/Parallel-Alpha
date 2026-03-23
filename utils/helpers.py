from __future__ import annotations

"""Shared helper utilities for Parallel Finder.

The functions in this module are designed to be:
- safe to import in headless environments;
- compatible with the new core engine/matcher/UI stack;
- vectorized where NumPy-based processing matters.
"""

import hashlib
import math
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from .constants import DEFAULT_FPS, KEYPOINT_CONF_THRESHOLD, NUM_KEYPOINTS

try:
    from PySide6.QtGui import QImage, QPixmap
except Exception:  # pragma: no cover - optional during non-UI runtime
    QImage = None  # type: ignore[assignment]
    QPixmap = None  # type: ignore[assignment]


ArrayF32 = npt.NDArray[np.float32]
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

_DIRECTION_NAMES: Final[tuple[Direction, ...]] = (
    "forward",
    "forward-right",
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
    "forward-left",
)

_DIRECTION_EMOJIS: Final[tuple[str, ...]] = (
    "⬆️",
    "↗️",
    "➡️",
    "↘️",
    "⬇️",
    "↙️",
    "⬅️",
    "↖️",
    "❓",
)


def get_file_hash(filepath: str | Path, chunk_size: int = 65_536) -> str:
    """Compute an MD5 hash for a file.

    Args:
        filepath: File path to hash.
        chunk_size: Chunk size used while reading the file.

    Returns:
        File hash as a hex string, or an empty string if the file cannot be
        read safely.
    """

    path = Path(filepath)
    if not path.exists() or not path.is_file():
        return ""

    hasher = hashlib.md5()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(max(1, int(chunk_size)))
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except (OSError, PermissionError):
        return ""


@lru_cache(maxsize=4_096)
def format_time(seconds: float | int | None, always_show_hours: bool = False) -> str:
    """Format seconds as ``MM:SS`` or ``HH:MM:SS``.

    Args:
        seconds: Duration in seconds.
        always_show_hours: Force ``HH:MM:SS`` formatting.

    Returns:
        A human-readable time string. Invalid values degrade gracefully.
    """

    if seconds is None:
        return "--:--:--" if always_show_hours else "--:--"

    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "--:--:--" if always_show_hours else "--:--"

    if not math.isfinite(value) or value < 0:
        return "--:--:--" if always_show_hours else "--:--"

    total_seconds = int(round(max(0.0, value)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0 or always_show_hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@lru_cache(maxsize=4_096)
def to_timecode(seconds: float | int, fps: float = DEFAULT_FPS, drop_frame: bool = False) -> str:
    """Convert seconds to SMPTE-like timecode.

    Args:
        seconds: Time in seconds.
        fps: Frames per second.
        drop_frame: Whether to emit drop-frame style timecode for broadcast
            rates such as 29.97 or 59.94.

    Returns:
        Timecode formatted as ``HH:MM:SS:FF`` or ``HH:MM:SS;FF``.
    """

    try:
        seconds_value = float(seconds)
        fps_value = float(fps)
    except (TypeError, ValueError):
        return "00:00:00:00"

    if not math.isfinite(seconds_value) or seconds_value < 0:
        return "00:00:00:00"
    if not math.isfinite(fps_value) or fps_value <= 0:
        fps_value = DEFAULT_FPS

    nominal_fps = max(1, int(round(fps_value)))
    total_frames = int(round(seconds_value * fps_value))

    separator = ":"
    frame_number = total_frames
    if drop_frame:
        if abs(fps_value - 29.97) < 0.02:
            nominal_fps = 30
            drop_frames = 2
        elif abs(fps_value - 59.94) < 0.02:
            nominal_fps = 60
            drop_frames = 4
        else:
            drop_frames = 0

        if drop_frames > 0:
            separator = ";"
            frames_per_minute = nominal_fps * 60
            total_minutes = total_frames // frames_per_minute
            dropped = drop_frames * (total_minutes - total_minutes // 10)
            frame_number = total_frames + dropped

    frames_per_hour = nominal_fps * 3600
    frames_per_minute = nominal_fps * 60

    hours = frame_number // frames_per_hour
    minutes = (frame_number // frames_per_minute) % 60
    secs = (frame_number // nominal_fps) % 60
    frames = frame_number % nominal_fps
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{frames:02d}"


def compact_number(num: int | float) -> str:
    """Format a number in compact notation such as ``1.2K`` or ``3.4M``."""

    try:
        number = float(num)
    except (TypeError, ValueError):
        return "0"

    if not math.isfinite(number):
        return "0"

    abs_number = abs(number)
    if abs_number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.1f}B"
    if abs_number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if abs_number >= 1_000:
        return f"{number / 1_000:.1f}K"
    if float(number).is_integer():
        return str(int(number))
    return f"{number:.1f}"


def _coerce_keypoints_array(keypoints: npt.ArrayLike) -> tuple[ArrayF32, npt.NDArray[np.float32]]:
    """Convert arbitrary keypoint input to ``(xy, conf)`` arrays.

    Returns:
        Tuple of ``xy`` with shape ``(N, 2)`` and confidence with shape ``(N,)``.
    """

    array = np.asarray(keypoints, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] < 2:
        zeros_xy = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        zeros_conf = np.zeros((NUM_KEYPOINTS,), dtype=np.float32)
        return zeros_xy, zeros_conf

    xy = array[:, :2].astype(np.float32, copy=False)
    if array.shape[1] >= 3:
        conf = array[:, 2].astype(np.float32, copy=False)
    else:
        conf = np.ones((array.shape[0],), dtype=np.float32)
    return xy, conf


def normalize_pose(
    keypoints: npt.ArrayLike,
    center: npt.ArrayLike | None = None,
    scale: float | None = None,
    use_torso: bool = True,
    conf_threshold: float = KEYPOINT_CONF_THRESHOLD,
) -> ArrayF32:
    """Normalize pose coordinates around a robust body center.

    The function supports ``(N, 2)`` and ``(N, 3)`` keypoint arrays. If
    confidence is provided, visible torso joints are preferred for determining
    center and scale. Otherwise, it gracefully falls back to visible points or
    the global mean.

    Args:
        keypoints: Keypoint array of shape ``(N, 2)`` or ``(N, 3)``.
        center: Optional precomputed center point.
        scale: Optional precomputed scale factor.
        use_torso: Prefer torso-based centering/scaling when possible.
        conf_threshold: Visibility threshold for confidence-aware operations.

    Returns:
        Normalized coordinates with shape ``(N, 2)`` and ``float32`` dtype.
    """

    xy, conf = _coerce_keypoints_array(keypoints)
    point_count = xy.shape[0]

    valid = np.isfinite(xy).all(axis=1)
    visible = valid & np.isfinite(conf) & (conf >= float(conf_threshold))

    def _mean_of(indices: tuple[int, ...]) -> ArrayF32 | None:
        selected = [idx for idx in indices if idx < point_count and visible[idx]]
        if not selected:
            return None
        return xy[np.asarray(selected, dtype=np.int32)].mean(axis=0, dtype=np.float32)

    if center is not None:
        center_arr = np.asarray(center, dtype=np.float32).reshape(-1)
        center_point = center_arr[:2] if center_arr.size >= 2 else None
    else:
        center_point = None

    if center_point is None:
        shoulders = _mean_of((5, 6))
        hips = _mean_of((11, 12))
        if use_torso and shoulders is not None and hips is not None:
            center_point = (shoulders + hips) * 0.5
        elif hips is not None:
            center_point = hips
        elif shoulders is not None:
            center_point = shoulders
        elif np.any(visible):
            center_point = xy[visible].mean(axis=0, dtype=np.float32)
        elif np.any(valid):
            center_point = xy[valid].mean(axis=0, dtype=np.float32)
        else:
            center_point = np.zeros((2,), dtype=np.float32)

    if scale is not None and math.isfinite(float(scale)) and float(scale) > 0:
        scale_factor = float(scale)
    else:
        candidates: list[float] = []

        def _pair_distance(i: int, j: int) -> None:
            if i < point_count and j < point_count and visible[i] and visible[j]:
                distance = float(np.linalg.norm(xy[i] - xy[j]))
                if distance > 1e-6:
                    candidates.append(distance)

        _pair_distance(5, 6)   # shoulder width
        _pair_distance(11, 12)  # hip width
        _pair_distance(5, 11)  # torso left
        _pair_distance(6, 12)  # torso right

        if use_torso and len(candidates) >= 2:
            scale_factor = float(np.mean(candidates, dtype=np.float32))
        elif candidates:
            scale_factor = float(np.median(np.asarray(candidates, dtype=np.float32)))
        else:
            centered = xy[valid] - center_point if np.any(valid) else xy - center_point
            distances = np.linalg.norm(centered, axis=1)
            positive = distances[np.isfinite(distances) & (distances > 0)]
            if positive.size > 0:
                scale_factor = float(np.percentile(positive, 90))
            else:
                scale_factor = 1.0

    scale_factor = max(float(scale_factor), 1e-6)
    normalized = (xy - center_point.astype(np.float32)) / np.float32(scale_factor)
    return normalized.astype(np.float32, copy=False)


def numpy_to_qpixmap(array: npt.ArrayLike | None) -> QPixmap | None:
    """Convert a NumPy image array to ``QPixmap``.

    Supported layouts:
    - ``(H, W)`` grayscale
    - ``(H, W, 3)`` RGB
    - ``(H, W, 4)`` RGBA

    Returns ``None`` when PySide6 is unavailable or conversion fails.
    """

    if QImage is None or QPixmap is None or array is None:
        return None

    try:
        image_array = np.asarray(array)
    except Exception:
        return None

    if image_array.size == 0:
        return None

    try:
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

        image_array = np.ascontiguousarray(image_array)

        if image_array.ndim == 2:
            height, width = image_array.shape
            qimage = QImage(
                image_array.data,
                width,
                height,
                image_array.strides[0],
                QImage.Format.Format_Grayscale8,
            ).copy()
            return QPixmap.fromImage(qimage)

        if image_array.ndim == 3 and image_array.shape[2] == 3:
            height, width, channels = image_array.shape
            qimage = QImage(
                image_array.data,
                width,
                height,
                image_array.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()
            return QPixmap.fromImage(qimage)

        if image_array.ndim == 3 and image_array.shape[2] == 4:
            height, width, channels = image_array.shape
            qimage = QImage(
                image_array.data,
                width,
                height,
                image_array.strides[0],
                QImage.Format.Format_RGBA8888,
            ).copy()
            return QPixmap.fromImage(qimage)
    except Exception:
        return None

    return None


@lru_cache(maxsize=32)
def direction_to_string(direction: int) -> Direction:
    """Convert an 8-way direction index to a canonical string label."""

    if 0 <= int(direction) < len(_DIRECTION_NAMES):
        return _DIRECTION_NAMES[int(direction)]
    return "unknown"


@lru_cache(maxsize=32)
def direction_to_emoji(direction: int) -> str:
    """Return an emoji arrow for an 8-way direction index."""

    if 0 <= int(direction) < 8:
        return _DIRECTION_EMOJIS[int(direction)]
    return _DIRECTION_EMOJIS[-1]


__all__ = [
    "ArrayF32",
    "Direction",
    "get_file_hash",
    "format_time",
    "to_timecode",
    "compact_number",
    "normalize_pose",
    "numpy_to_qpixmap",
    "direction_to_string",
    "direction_to_emoji",
]
