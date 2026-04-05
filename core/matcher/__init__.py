"""
Реэкспорт для обратной совместимости.
Старый код: from core.matcher import MotionMatcher — работает без изменений.
"""
from core.matcher.motion_matcher import MotionMatcher
from core.matcher.pose_processor import (
    preprocess_pose,
    is_pose_valid,
    mirror_vectors,
    build_poses_tensor,
    BODY_WEIGHTS,
    COCO_N_KPS,
    ANCHOR_KPS,
    MIRROR_PAIRS,
)

__all__ = [
    "MotionMatcher",
    "preprocess_pose",
    "is_pose_valid",
    "mirror_vectors",
    "build_poses_tensor",
    "BODY_WEIGHTS",
    "COCO_N_KPS",
    "ANCHOR_KPS",
    "MIRROR_PAIRS",
]