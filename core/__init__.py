"""Core modules for Parallel Finder - pose estimation, persistence and motion matching."""

from __future__ import annotations

from .engine import (
    DecodeError,
    Direction,
    EngineConfig,
    InferenceError,
    Keypoint,
    MultiPoseData,
    PoseData,
    TensorRTError,
    YoloEngine,
    YoloEngineError,
)
from .matcher import MatchResult, MatcherConfig, MotionMatcher
from .project import PreviewCache, ProjectData, ProjectManager, ReferencePerson

try:
    from .matcher import Keypoint as MatcherKeypoint
except Exception:  # pragma: no cover - optional re-export
    MatcherKeypoint = Keypoint

__all__ = [
    "YoloEngine",
    "EngineConfig",
    "PoseData",
    "MultiPoseData",
    "Keypoint",
    "Direction",
    "YoloEngineError",
    "DecodeError",
    "InferenceError",
    "TensorRTError",
    "MotionMatcher",
    "MatcherConfig",
    "MatchResult",
    "MatcherKeypoint",
    "ProjectManager",
    "ReferencePerson",
    "ProjectData",
    "PreviewCache",
]

__version__ = "2.0.0"
