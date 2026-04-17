from __future__ import annotations

"""Shared application constants and path helpers for Parallel Finder Alpha v13.

This module centralizes:
- application metadata and versioning;
- cross-platform project, cache, log, and model directories;
- YOLO model registry and path helpers;
- pose / model / UI constants shared across core and UI layers.

All public constants are typed with ``Final`` to make intent explicit.
Import from here — never hard-code paths or magic strings elsewhere.
"""

import os
import sys
from pathlib import Path
from typing import Final

# ═══════════════════════════════════════════════════════════════════════════════
# § 1. APPLICATION METADATA
# ═══════════════════════════════════════════════════════════════════════════════

# Full display name shown in UI window title, about screen, logs, builds.
APP_DISPLAY_NAME: Final[str] = "Parallel Alpha 13.4.5"

# Short human-readable version label (for UI badges, splash, about).
APP_SHORT_VERSION: Final[str] = "13.4.5"

# Numeric build version (for packaging, auto-update checks, crash reports).
APP_BUILD_VERSION: Final[str] = "v13.4.5"

# Author / studio identifier.
APP_AUTHOR: Final[str] = "Pozit1vchic"

# ── Backward-compatible aliases ───────────────────────────────────────────────
# Old code imports APP_NAME / APP_VERSION — keep them working.
APP_NAME: Final[str] = APP_DISPLAY_NAME   # was "Parallel Finder"
APP_VERSION: Final[str] = APP_BUILD_VERSION  # was "2.0.0"


# ═══════════════════════════════════════════════════════════════════════════════
# § 2. PROJECT ROOT & FROZEN-EXE SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_app_root() -> Path:
    """Resolve the writable application root directory.

    - **Frozen** (PyInstaller / cx_Freeze): directory that contains the .exe.
    - **Development**: project root — one level above the ``utils/`` package.

    Returns
    -------
    Path
        Absolute, resolved path to the application root.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # utils/constants.py → utils/ → project root
    return Path(__file__).resolve().parent.parent


APP_ROOT: Final[Path] = _resolve_app_root()

# Backward-compatible alias (old code uses BASE_DIR).
BASE_DIR: Final[Path] = APP_ROOT


# ═══════════════════════════════════════════════════════════════════════════════
# § 3. PER-USER DATA DIRECTORY
# ═══════════════════════════════════════════════════════════════════════════════

def get_user_data_dir() -> Path:
    """Return a per-user application data directory for the current OS.

    Platform mapping
    ----------------
    Windows : ``%LOCALAPPDATA%\\ParallelFinder``
    macOS   : ``~/Library/Application Support/ParallelFinder``
    Linux   : ``$XDG_CONFIG_HOME/parallel-finder``  (defaults to ``~/.config``)

    Returns
    -------
    Path
        Absolute path; directory is **not** created here — use
        ``_ensure_directory`` after this call.
    """
    match sys.platform:
        case "win32":
            base = Path(
                os.environ.get("LOCALAPPDATA")
                or os.environ.get("APPDATA")
                or (Path.home() / "AppData" / "Local")
            )
            return base / "ParallelFinder"

        case "darwin":
            return Path.home() / "Library" / "Application Support" / "ParallelFinder"

        case _:
            xdg = os.environ.get("XDG_CONFIG_HOME")
            base = Path(xdg) if xdg else Path.home() / ".config"
            return base / "parallel-finder"


USER_DATA_DIR: Final[Path] = get_user_data_dir()


# ═══════════════════════════════════════════════════════════════════════════════
# § 4. DIRECTORY LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Bundled / project-local directories ──────────────────────────────────────

# PRIMARY model storage. YoloEngine must look HERE first and only here.
MODELS_DIR: Final[Path] = APP_ROOT / "models"

# Application configuration files (settings.json, hotkeys, etc.).
CONFIG_DIR: Final[Path] = APP_ROOT / "config"

# ── User-specific runtime directories ────────────────────────────────────────

CACHE_DIR: Final[Path]         = USER_DATA_DIR / "cache"
PREVIEW_CACHE_DIR: Final[Path] = CACHE_DIR / "previews"
POSE_CACHE_DIR: Final[Path]    = CACHE_DIR / "poses"
TEMP_DIR: Final[Path]          = CACHE_DIR / "tmp"       # session-scoped scratch

LOG_DIR: Final[Path]           = USER_DATA_DIR / "logs"
PROJECTS_DIR: Final[Path]      = USER_DATA_DIR / "projects"
EXPORT_DIR: Final[Path]        = USER_DATA_DIR / "exports"


def _ensure_directory(path: Path) -> None:
    """Create *path* (and all parents) silently, ignoring permission errors.

    The application degrades gracefully when a specific directory cannot be
    created (e.g. read-only filesystem in a sandboxed environment).
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


# Create all directories at import time so the rest of the app can assume
# they exist without additional checks.
for _dir in (
    MODELS_DIR,
    CONFIG_DIR,
    USER_DATA_DIR,
    CACHE_DIR,
    PREVIEW_CACHE_DIR,
    POSE_CACHE_DIR,
    TEMP_DIR,
    LOG_DIR,
    PROJECTS_DIR,
    EXPORT_DIR,
):
    _ensure_directory(_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# § 5. YOLO MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL_NAME: Final[str] = "yolo11x-pose.pt"

# Базовый список официально поддерживаемых моделей
_BASE_MODELS: list[str] = [
    # YOLO11 pose
    "yolo11n-pose.pt",
    "yolo11s-pose.pt",
    "yolo11m-pose.pt",
    "yolo11l-pose.pt",
    "yolo11x-pose.pt",
    # YOLO26 pose (n, s, m, l, x) — 26-class
    "yolo26n-pose.pt",
    "yolo26s-pose.pt",
    "yolo26m-pose.pt",
    "yolo26l-pose.pt",
    "yolo26x-pose.pt",
]


def list_local_models() -> list[str]:
    """Все .pt файлы в MODELS_DIR (включая кастомные)."""
    if not MODELS_DIR.is_dir():
        return []
    return sorted(p.name for p in MODELS_DIR.glob("*.pt") if p.is_file())


def _build_model_list() -> list[str]:
    """
    Объединяет базовый список с локальными моделями.
    Локальные кастомные модели (yolo26x-pose.pt и др.) — в начале списка.
    """
    local  = set(list_local_models())
    result = list(_BASE_MODELS)

    # Добавляем локальные модели которых нет в базовом списке — вверх
    extras = sorted(lm for lm in local if lm not in result)
    for ex in reversed(extras):
        result.insert(0, ex)

    return result


# Динамический список — включает ВСЕ локальные модели
YOLO_AVAILABLE_MODELS: list[str] = _build_model_list()

# Backward-compatible alias
AVAILABLE_MODELS: Final[tuple[str, ...]] = tuple(YOLO_AVAILABLE_MODELS)


def get_model_path(name: str | None = None) -> Path:
    """
    Возвращает абсолютный путь к файлу модели в MODELS_DIR.
    Принимает ЛЮБОЕ имя — валидация по списку НЕ выполняется.
    Это позволяет работать с кастомными моделями (yolo26x-pose.pt и др.)
    """
    resolved = Path(name or DEFAULT_MODEL_NAME).name
    return MODELS_DIR / resolved


def get_default_model_path() -> Path:
    return MODELS_DIR / DEFAULT_MODEL_NAME


def is_model_local(name: str | None = None) -> bool:
    """
    Проверяет наличие файла модели в MODELS_DIR.
    Работает с любым именем — включая кастомные модели.
    """
    try:
        resolved = Path(name or DEFAULT_MODEL_NAME).name
        path = MODELS_DIR / resolved
        return path.is_file() and path.stat().st_size > 0
    except Exception:
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# § 6. SUPPORTED VIDEO FORMATS
# ═══════════════════════════════════════════════════════════════════════════════

# Lowercase dot-prefixed extensions recognised as video files.
VIDEO_EXTENSIONS: Final[frozenset[str]] = frozenset({
    # Common containers
    ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm",
    # MPEG family
    ".mpeg", ".mpg", ".m4v", ".m2v", ".m4p",
    # Mobile / embedded
    ".3gp", ".3g2",
    # Open / legacy
    ".ogv", ".ogg", ".drc", ".gif", ".gifv", ".mng", ".qt",
    ".yuv", ".rm", ".rmvb", ".asf", ".amv",
    # Broadcast / professional
    ".mxf", ".ts", ".m2ts", ".mts",
    # Optical disc
    ".vob", ".evo",
    # Camcorder
    ".mod", ".tod",
    # Other
    ".divx",
})

# Ready-made glob pattern string for tkinter file dialogs and drag-and-drop
# filtering. Example: "*.mp4 *.avi *.mkv …"
VIDEO_EXTENSIONS_GLOB: Final[str] = " ".join(
    f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS)
)


def is_video_file(path: str | Path) -> bool:
    """Return ``True`` if *path* has a recognised video extension.

    Parameters
    ----------
    path:
        File path (string or :class:`~pathlib.Path`).

    Returns
    -------
    bool
    """
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


# ═══════════════════════════════════════════════════════════════════════════════
# § 7. YOLO INFERENCE DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum bounding-box confidence for detections to be returned.
YOLO_CONF: Final[float] = 0.25

# Input image size fed to YOLO (square, pixels).
YOLO_IMGSZ: Final[int] = 640

# ═══════════════════════════════════════════════════════════════════════════════
# § 8. POSE / KEYPOINT DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════

# Number of COCO keypoints used throughout the project.
NUM_KEYPOINTS: Final[int] = 17

# Backward-compatible alias.
POSE_KEYPOINTS: Final[int] = NUM_KEYPOINTS

# Minimum keypoint visibility score to consider a joint "visible".
KEYPOINT_CONF_THRESHOLD: Final[float] = 0.3

# ═══════════════════════════════════════════════════════════════════════════════
# § 9. ANALYSIS & MATCHING DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════

# Default playback / processing frame-rate when metadata is unavailable.
DEFAULT_FPS: Final[float] = 30.0

# Minimum time gap (seconds) between two matched poses from the same video.
DUPLICATE_WINDOW: Final[float] = 2.5

# Head turn threshold (degrees) for direction classification heuristics.
HEAD_TURN_THRESHOLD: Final[float] = 15.0

# Default similarity threshold for MotionMatcher (0–1, cosine).
DEFAULT_SIM_THRESHOLD: Final[float] = 0.75

# Default minimum scene gap forwarded to MotionMatcher.find_matches().
DEFAULT_MIN_GAP: Final[float] = 3.0

# ═══════════════════════════════════════════════════════════════════════════════
# § 10. UI DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════

# Thumbnail / preview dimensions used in result cards (width, height).
PREVIEW_SIZE: Final[tuple[int, int]] = (320, 180)

# Starting batch size before adaptive tuning by YoloEngine.
BATCH_SIZE_DEFAULT: Final[int] = 32

# Auto-save interval for project state, in milliseconds (5 minutes).
AUTOSAVE_INTERVAL: Final[int] = 300_000

# ═══════════════════════════════════════════════════════════════════════════════
# § 11. COCO-17 SKELETON DATA
# ═══════════════════════════════════════════════════════════════════════════════

# Permutation that maps each keypoint index to its left↔right mirror.
# Used by MotionMatcher._mirror_vector() and mirror_vectors().
FLIP_INDICES: Final[tuple[int, ...]] = (
    0,   # nose       → nose
    2,   # l_eye      → r_eye
    1,   # r_eye      → l_eye
    4,   # l_ear      → r_ear
    3,   # r_ear      → l_ear
    6,   # l_shoulder → r_shoulder
    5,   # r_shoulder → l_shoulder
    8,   # l_elbow    → r_elbow
    7,   # r_elbow    → l_elbow
    10,  # l_wrist    → r_wrist
    9,   # r_wrist    → l_wrist
    12,  # l_hip      → r_hip
    11,  # r_hip      → l_hip
    14,  # l_knee     → r_knee
    13,  # r_knee     → l_knee
    16,  # l_ankle    → r_ankle
    15,  # r_ankle    → l_ankle
)

# Skeleton edges for overlay / debug visualisation (joint index pairs).
SKELETON_CONNECTIONS: Final[tuple[tuple[int, int], ...]] = (
    # Head
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    # Torso
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    # Left arm
    (5, 7), (7, 9),
    # Right arm
    (6, 8), (8, 10),
    # Left leg
    (11, 13), (13, 15),
    # Right leg
    (12, 14), (14, 16),
)

# ═══════════════════════════════════════════════════════════════════════════════
# § 12. THEME PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

THEMES: Final[dict[str, dict[str, str]]] = {
    "dark": {
        "bg":             "#0a0c10",
        "card":           "#14171c",
        "text":           "#ffffff",
        "text_secondary": "#8a8f99",
        "accent":         "#3b82f6",
        "accent_hover":   "#1d4ed8",
        "accent2":        "#f59e0b",
        "success":        "#3b82f6",
        "success_hover":  "#1d4ed8",
        "warning":        "#facc15",
        "error":          "#374151",
        "error_hover":    "#4b5563",
        "border":         "#2a2f38",
        "hover":          "#1e232b",
        "highlight":      "#1e293b",
        "active_row":     "#1e3a5f",
        "marked_bad":     "#2d1a1a",
        "glow":           "#1d4ed8",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# § 13. PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

__all__: list[str] = [
    # ── Metadata ──────────────────────────────────────────────────────────────
    "APP_DISPLAY_NAME",
    "APP_SHORT_VERSION",
    "APP_BUILD_VERSION",
    "APP_AUTHOR",
    # Backward-compatible aliases
    "APP_NAME",
    "APP_VERSION",
    # ── Paths ─────────────────────────────────────────────────────────────────
    "APP_ROOT",
    "BASE_DIR",
    "USER_DATA_DIR",
    "MODELS_DIR",
    "CONFIG_DIR",
    "CACHE_DIR",
    "PREVIEW_CACHE_DIR",
    "POSE_CACHE_DIR",
    "TEMP_DIR",
    "LOG_DIR",
    "PROJECTS_DIR",
    "EXPORT_DIR",
    # Path helpers
    "get_user_data_dir",
    # ── YOLO model registry ───────────────────────────────────────────────────
    "DEFAULT_MODEL_NAME",
    "YOLO_AVAILABLE_MODELS",
    "AVAILABLE_MODELS",          # backward-compatible alias
    "get_model_path",
    "get_default_model_path",
    "is_model_local",
    "list_local_models",
    # ── Video formats ─────────────────────────────────────────────────────────
    "VIDEO_EXTENSIONS",
    "VIDEO_EXTENSIONS_GLOB",
    "is_video_file",
    # ── YOLO inference ────────────────────────────────────────────────────────
    "YOLO_CONF",
    "YOLO_IMGSZ",
    # ── Pose / keypoints ──────────────────────────────────────────────────────
    "NUM_KEYPOINTS",
    "POSE_KEYPOINTS",            # backward-compatible alias
    "KEYPOINT_CONF_THRESHOLD",
    # ── Analysis & matching ───────────────────────────────────────────────────
    "DEFAULT_FPS",
    "DUPLICATE_WINDOW",
    "HEAD_TURN_THRESHOLD",
    "DEFAULT_SIM_THRESHOLD",
    "DEFAULT_MIN_GAP",
    # ── UI ────────────────────────────────────────────────────────────────────
    "PREVIEW_SIZE",
    "BATCH_SIZE_DEFAULT",
    "AUTOSAVE_INTERVAL",
    # ── Skeleton ──────────────────────────────────────────────────────────────
    "FLIP_INDICES",
    "SKELETON_CONNECTIONS",
    # ── Theme ─────────────────────────────────────────────────────────────────
    "THEMES",
]
