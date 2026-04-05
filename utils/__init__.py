"""Utilities for Parallel Finder Alpha v13.

This package exposes shared constants, localization helpers and utility
functions used by the core analysis stack and the UI.
"""

from __future__ import annotations

from .constants import (
    APP_DISPLAY_NAME,
    APP_SHORT_VERSION,
    APP_BUILD_VERSION,
    APP_AUTHOR,
    APP_NAME,
    APP_ROOT,
    APP_VERSION,
    AUTOSAVE_INTERVAL,
    BASE_DIR,
    BATCH_SIZE_DEFAULT,
    CACHE_DIR,
    CONFIG_DIR,
    DEFAULT_FPS,
    DEFAULT_MIN_GAP,
    DEFAULT_MODEL_NAME,
    DEFAULT_SIM_THRESHOLD,
    DUPLICATE_WINDOW,
    EXPORT_DIR,
    FLIP_INDICES,
    HEAD_TURN_THRESHOLD,
    KEYPOINT_CONF_THRESHOLD,
    LOG_DIR,
    MODELS_DIR,
    NUM_KEYPOINTS,
    POSE_CACHE_DIR,
    POSE_KEYPOINTS,
    PREVIEW_CACHE_DIR,
    PREVIEW_SIZE,
    PROJECTS_DIR,
    SKELETON_CONNECTIONS,
    TEMP_DIR,
    THEMES,
    USER_DATA_DIR,
    VIDEO_EXTENSIONS,
    VIDEO_EXTENSIONS_GLOB,
    YOLO_AVAILABLE_MODELS,
    AVAILABLE_MODELS,
    YOLO_CONF,
    YOLO_IMGSZ,
    get_default_model_path,
    get_model_path,
    get_user_data_dir,
    is_model_local,
    is_video_file,
    list_local_models,
)
from .helpers import (
    ArrayF32,
    Direction,
    compact_number,
    direction_to_emoji,
    direction_to_string,
    format_time,
    get_file_hash,
    normalize_pose,
    numpy_to_qpixmap,
    to_timecode,
)
from .locales import (
    TRANSLATIONS,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    FALLBACK_LANGUAGE,
    Translator,
    check_sync,
    get_supported_languages,
    get_translator,
    t,
)

__all__ = [
    # constants: metadata
    "APP_DISPLAY_NAME",
    "APP_SHORT_VERSION",
    "APP_BUILD_VERSION",
    "APP_AUTHOR",
    "APP_NAME",
    "APP_VERSION",
    "APP_ROOT",
    "BASE_DIR",

    # constants: paths
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

    # constants: models
    "DEFAULT_MODEL_NAME",
    "YOLO_AVAILABLE_MODELS",
    "AVAILABLE_MODELS",
    "get_model_path",
    "get_default_model_path",
    "is_model_local",
    "list_local_models",

    # constants: video
    "VIDEO_EXTENSIONS",
    "VIDEO_EXTENSIONS_GLOB",
    "is_video_file",

    # constants: inference / pose / analysis / ui
    "YOLO_CONF",
    "YOLO_IMGSZ",
    "NUM_KEYPOINTS",
    "POSE_KEYPOINTS",
    "KEYPOINT_CONF_THRESHOLD",
    "DEFAULT_FPS",
    "DUPLICATE_WINDOW",
    "HEAD_TURN_THRESHOLD",
    "DEFAULT_SIM_THRESHOLD",
    "DEFAULT_MIN_GAP",
    "PREVIEW_SIZE",
    "BATCH_SIZE_DEFAULT",
    "AUTOSAVE_INTERVAL",
    "FLIP_INDICES",
    "SKELETON_CONNECTIONS",
    "THEMES",
    "get_user_data_dir",

    # helpers
    "ArrayF32",
    "Direction",
    "compact_number",
    "direction_to_emoji",
    "direction_to_string",
    "format_time",
    "get_file_hash",
    "normalize_pose",
    "numpy_to_qpixmap",
    "to_timecode",

    # locales
    "TRANSLATIONS",
    "SUPPORTED_LANGUAGES",
    "DEFAULT_LANGUAGE",
    "FALLBACK_LANGUAGE",
    "Translator",
    "t",
    "get_translator",
    "get_supported_languages",
    "check_sync",
]

__version__ = APP_VERSION