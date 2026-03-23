from __future__ import annotations

"""Shared application constants and path helpers for Parallel Finder.

This module centralizes:
- application metadata and versioning;
- cross-platform project, cache, log, and model directories;
- pose/model/UI constants shared across core and UI layers.

All public constants are typed with ``Final`` to make intent explicit.
"""

import os
import sys
from pathlib import Path
from typing import Final


APP_NAME: Final[str] = "Parallel Finder"
APP_VERSION: Final[str] = "2.0.0"


def _resolve_app_root() -> Path:
    """Resolve the writable application root.

    When the application is frozen (PyInstaller / cx_Freeze style), the root is
    the directory containing the executable. Otherwise, it is the project root
    one level above the ``utils`` package.
    """

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


APP_ROOT: Final[Path] = _resolve_app_root()
BASE_DIR: Final[Path] = APP_ROOT  # Backward-compatible alias.


def get_user_data_dir() -> Path:
    """Return a per-user application data directory for the current OS."""

    if sys.platform == "win32":
        base = Path(
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("APPDATA")
            or (Path.home() / "AppData" / "Local")
        )
        return base / "ParallelFinder"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ParallelFinder"

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg_config_home) if xdg_config_home else Path.home() / ".config"
    return base / "parallel-finder"


USER_DATA_DIR: Final[Path] = get_user_data_dir()

# Application directories bundled with or placed next to the app.
MODELS_DIR: Final[Path] = APP_ROOT / "models"
CONFIG_DIR: Final[Path] = APP_ROOT / "config"

# User-specific directories for runtime data.
CACHE_DIR: Final[Path] = USER_DATA_DIR / "cache"
PREVIEW_CACHE_DIR: Final[Path] = CACHE_DIR / "previews"
POSE_CACHE_DIR: Final[Path] = CACHE_DIR / "poses"
LOG_DIR: Final[Path] = USER_DATA_DIR / "logs"
PROJECTS_DIR: Final[Path] = USER_DATA_DIR / "projects"
EXPORT_DIR: Final[Path] = USER_DATA_DIR / "exports"


def _ensure_directory(path: Path) -> None:
    """Create a directory if possible, ignoring permission-related failures.

    The application can still operate with graceful degradation even if some
    directories cannot be created immediately.
    """

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


for _directory in (
    MODELS_DIR,
    CONFIG_DIR,
    USER_DATA_DIR,
    CACHE_DIR,
    PREVIEW_CACHE_DIR,
    POSE_CACHE_DIR,
    LOG_DIR,
    PROJECTS_DIR,
    EXPORT_DIR,
):
    _ensure_directory(_directory)


# Supported video formats.
VIDEO_EXTENSIONS: Final[set[str]] = {
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".mpeg",
    ".mpg",
    ".m4v",
    ".3gp",
    ".3g2",
    ".ogv",
    ".ogg",
    ".drc",
    ".gif",
    ".gifv",
    ".mng",
    ".qt",
    ".yuv",
    ".rm",
    ".rmvb",
    ".asf",
    ".amv",
    ".m4p",
    ".m2v",
    ".mxf",
    ".ts",
    ".m2ts",
    ".mts",
    ".divx",
    ".vob",
    ".evo",
    ".mod",
    ".tod",
}

# YOLO defaults.
YOLO_CONF: Final[float] = 0.25
YOLO_IMGSZ: Final[int] = 640

# Pose defaults.
NUM_KEYPOINTS: Final[int] = 17
POSE_KEYPOINTS: Final[int] = NUM_KEYPOINTS  # Backward-compatible alias.
KEYPOINT_CONF_THRESHOLD: Final[float] = 0.3

# UI defaults.
PREVIEW_SIZE: Final[tuple[int, int]] = (320, 180)
BATCH_SIZE_DEFAULT: Final[int] = 32
AUTOSAVE_INTERVAL: Final[int] = 300_000

# Matching / analysis defaults.
DUPLICATE_WINDOW: Final[float] = 2.5
HEAD_TURN_THRESHOLD: Final[float] = 15.0
DEFAULT_FPS: Final[float] = 30.0

# COCO 17-keypoint mirror indices.
FLIP_INDICES: Final[tuple[int, ...]] = (
    0,
    2,
    1,
    4,
    3,
    6,
    5,
    8,
    7,
    10,
    9,
    12,
    11,
    14,
    13,
    16,
    15,
)

# Skeleton edges used by preview / debug visualization.
SKELETON_CONNECTIONS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

# Shared V11.8 theme palette.
THEMES: Final[dict[str, dict[str, str]]] = {
    "dark": {
        "bg": "#0a0c10",
        "card": "#14171c",
        "text": "#ffffff",
        "text_secondary": "#8a8f99",
        "accent": "#3b82f6",
        "accent2": "#f59e0b",
        "success": "#22c55e",
        "warning": "#facc15",
        "error": "#ef4444",
        "border": "#2a2f38",
        "hover": "#1e232b",
    }
}


__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "APP_ROOT",
    "BASE_DIR",
    "USER_DATA_DIR",
    "MODELS_DIR",
    "CONFIG_DIR",
    "CACHE_DIR",
    "PREVIEW_CACHE_DIR",
    "POSE_CACHE_DIR",
    "LOG_DIR",
    "PROJECTS_DIR",
    "EXPORT_DIR",
    "VIDEO_EXTENSIONS",
    "YOLO_CONF",
    "YOLO_IMGSZ",
    "NUM_KEYPOINTS",
    "POSE_KEYPOINTS",
    "KEYPOINT_CONF_THRESHOLD",
    "PREVIEW_SIZE",
    "BATCH_SIZE_DEFAULT",
    "AUTOSAVE_INTERVAL",
    "DUPLICATE_WINDOW",
    "HEAD_TURN_THRESHOLD",
    "DEFAULT_FPS",
    "FLIP_INDICES",
    "SKELETON_CONNECTIONS",
    "THEMES",
    "get_user_data_dir",
]
