#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/matcher/pose_processor.py
v17 — Appearance-aware pose processing with body proportions.

Критические улучшения v13:
1. Добавлен compute_body_proportions() — эмбеддинг пропорций тела
2. Добавлен compare_body_proportions() — сравнение внешности людей
3. build_poses_tensor() сохраняет body_proportions в метаданные
4. Улучшена валидация: отсев нечеловеческих поз
5. Добавлена проверка симметричности тела (human-likeness)
6. MIN_VISIBLE_KPS = 7 (баланс между recall и precision)
7. Добавлен fallback для лиц без anchor-точек

Цель: разделять разных людей, даже если позы похожи.
"""

from __future__ import annotations
from typing import TypeAlias, Optional

import numpy as np
import torch

PoseDict: TypeAlias = dict

# ── Константы COCO-17 ─────────────────────────────────────────────────────────
COCO_N_KPS = 17

ANCHOR_KPS: list[int] = [5, 6, 11, 12]
ANCHOR_KPS_ARR = np.array(ANCHOR_KPS, dtype=np.intp)

MIRROR_PAIRS: list[tuple[int, int]] = [
    (1, 2), (3, 4),
    (5, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16),
]

_PAIRED_INDICES: frozenset[int] = frozenset(i for pair in MIRROR_PAIRS for i in pair)
_UNPAIRED_INDICES: list[int] = [i for i in range(COCO_N_KPS) if i not in _PAIRED_INDICES]

# ── Индексы частей тела ───────────────────────────────────────────────────────
NOSE = 0
EYE_L, EYE_R = 1, 2
EAR_L, EAR_R = 3, 4
SHOULDER_L, SHOULDER_R = 5, 6
ELBOW_L, ELBOW_R = 7, 8
WRIST_L, WRIST_R = 9, 10
HIP_L, HIP_R = 11, 12
KNEE_L, KNEE_R = 13, 14
ANKLE_L, ANKLE_R = 15, 16

# ── Предвычисленные индексы для mirror_vectors ────────────────────────────────
def _build_mirror_indices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = COCO_N_KPS * 2
    src_x = np.arange(n, dtype=np.int64)
    sign_x = np.ones(n, dtype=np.float32)
    src_y = np.arange(n, dtype=np.int64)

    for l_idx, r_idx in MIRROR_PAIRS:
        lx, ly = l_idx * 2, l_idx * 2 + 1
        rx, ry = r_idx * 2, r_idx * 2 + 1

        src_x[lx] = rx
        src_x[rx] = lx
        sign_x[lx] = -1.0
        sign_x[rx] = -1.0

        src_y[ly] = ry
        src_y[ry] = ly

    for idx in _UNPAIRED_INDICES:
        ix = idx * 2
        sign_x[ix] = -1.0

    return src_x, sign_x, src_y

_MIRROR_SRC_X, _MIRROR_SIGN_X, _MIRROR_SRC_Y = _build_mirror_indices()

_MIRROR_SRC_X_T: torch.Tensor | None = None
_MIRROR_SIGN_X_T: torch.Tensor | None = None
_MIRROR_SRC_Y_T: torch.Tensor | None = None
_MIRROR_DEVICE: str = ""

def _get_mirror_tensors(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _MIRROR_SRC_X_T, _MIRROR_SIGN_X_T, _MIRROR_SRC_Y_T, _MIRROR_DEVICE
    if _MIRROR_SRC_X_T is None or _MIRROR_DEVICE != device:
        _MIRROR_SRC_X_T = torch.from_numpy(_MIRROR_SRC_X).to(device)
        _MIRROR_SIGN_X_T = torch.from_numpy(_MIRROR_SIGN_X).to(device)
        _MIRROR_SRC_Y_T = torch.from_numpy(_MIRROR_SRC_Y).to(device)
        _MIRROR_DEVICE = device
    return _MIRROR_SRC_X_T, _MIRROR_SIGN_X_T, _MIRROR_SRC_Y_T

# ── Веса частей тела ──────────────────────────────────────────────────────────
BODY_WEIGHTS = np.array([
    0.9,
    0.5, 0.5,
    0.4, 0.4,
    1.8, 1.8,
    1.4, 1.4,
    1.1, 1.1,
    1.8, 1.8,
    1.5, 1.5,
    1.1, 1.1,
], dtype=np.float32)

BODY_WEIGHTS_2D: np.ndarray = BODY_WEIGHTS[:, np.newaxis]

# ── Пороги (баланс precision/recall) ──────────────────────────────────────────
MIN_KP_CONFIDENCE = 0.24
MIN_ANCHOR_CONFIDENCE = 0.27
ANCHOR_CONF_KPS = [5, 6, 11, 12, 13, 14]
ANCHOR_CONF_KPS_ARR = np.array(ANCHOR_CONF_KPS, dtype=np.intp)

_MIN_VISIBLE_KPS = 7
_MIN_VISIBLE_ANCHORS = 2

MIN_BBOX_AREA = 500
MAX_BBOX_AREA = 500000
MIN_BBOX_ASPECT = 0.25
MAX_BBOX_ASPECT = 4.5

DIRECTION_MIRROR_MAP = {
    "left": "right",
    "right": "left",
    "forward": "forward",
    "back": "back",
    "unknown": "unknown",
}


# ═══════════════════════════════════════════════════════════════════════════════
# § НОВОЕ: Пропорции тела (Appearance Embedding)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_body_proportions(kps: np.ndarray, conf_threshold: float = MIN_KP_CONFIDENCE) -> dict:
    """
    Вычисляет пропорции тела человека для идентификации.
    
    Признаки:
    - leg_to_torso: длина ног / длина торса
    - shoulder_to_height: ширина плеч / полный рост
    - torso_aspect: ширина плеч / длина торса
    - arm_to_torso: длина рук / длина торса
    
    Args:
        kps: Keypoints (17, 2) или (17, 3)
        conf_threshold: Порог уверенности
        
    Returns:
        dict с пропорциями и флагом valid
    """
    xy = kps[:, :2] if kps.shape[1] >= 2 else kps
    conf = kps[:, 2] if kps.shape[1] >= 3 else np.ones(len(kps))
    
    vis = conf >= conf_threshold
    
    props = {
        "leg_to_torso": 1.0,
        "shoulder_to_height": 0.3,
        "torso_aspect": 0.5,
        "arm_to_torso": 0.8,
        "valid": False,
    }
    
    if not (vis[SHOULDER_L] and vis[SHOULDER_R] and vis[HIP_L] and vis[HIP_R]):
        return props
    
    shoulder_center = (xy[SHOULDER_L] + xy[SHOULDER_R]) / 2
    hip_center = (xy[HIP_L] + xy[HIP_R]) / 2
    torso_length = np.linalg.norm(hip_center - shoulder_center)
    
    if torso_length < 1e-3:
        return props
    
    # Длина ног
    leg_length = 0.0
    n_legs = 0
    for hip_idx, knee_idx, ankle_idx in [(HIP_L, KNEE_L, ANKLE_L), (HIP_R, KNEE_R, ANKLE_R)]:
        if vis[hip_idx] and vis[knee_idx] and vis[ankle_idx]:
            thigh = np.linalg.norm(xy[knee_idx] - xy[hip_idx])
            shin = np.linalg.norm(xy[ankle_idx] - xy[knee_idx])
            leg_length += (thigh + shin)
            n_legs += 1
    
    if n_legs > 0:
        leg_length /= n_legs
        props["leg_to_torso"] = leg_length / torso_length
    
    # Ширина плеч
    shoulder_width = np.linalg.norm(xy[SHOULDER_R] - xy[SHOULDER_L])
    props["torso_aspect"] = shoulder_width / torso_length
    
    # Полный рост
    if vis[NOSE]:
        height = np.linalg.norm(xy[NOSE] - hip_center) + leg_length
        if height > 1e-3:
            props["shoulder_to_height"] = shoulder_width / height
    
    # Длина рук
    arm_length = 0.0
    n_arms = 0
    for shoulder_idx, elbow_idx, wrist_idx in [(SHOULDER_L, ELBOW_L, WRIST_L), (SHOULDER_R, ELBOW_R, WRIST_R)]:
        if vis[shoulder_idx] and vis[elbow_idx] and vis[wrist_idx]:
            upper = np.linalg.norm(xy[elbow_idx] - xy[shoulder_idx])
            lower = np.linalg.norm(xy[wrist_idx] - xy[elbow_idx])
            arm_length += (upper + lower)
            n_arms += 1
    
    if n_arms > 0:
        arm_length /= n_arms
        props["arm_to_torso"] = arm_length / torso_length
    
    props["valid"] = True
    return props


def compare_body_proportions(props1: dict, props2: dict) -> float:
    """
    Сравнивает пропорции тела двух людей.
    
    Returns:
        float: similarity (0..1), где 1 = одинаковые пропорции
    """
    if not props1.get("valid") or not props2.get("valid"):
        return 0.5
    
    keys = ["leg_to_torso", "shoulder_to_height", "torso_aspect", "arm_to_torso"]
    diffs = []
    
    for key in keys:
        v1, v2 = props1.get(key, 1.0), props2.get(key, 1.0)
        if max(v1, v2) < 1e-5:
            continue
        rel_diff = abs(v1 - v2) / max(v1, v2)
        diffs.append(rel_diff)
    
    if not diffs:
        return 0.5
    
    avg_diff = sum(diffs) / len(diffs)
    similarity = max(0.0, 1.0 - avg_diff * 3.0)
    return similarity


def is_human_like(kps: np.ndarray, conf_threshold: float = MIN_KP_CONFIDENCE) -> bool:
    """
    Проверяет, похож ли скелет на человека (не животное, не объект).
    
    Критерии:
    1. Симметричность парных точек
    2. Разумные пропорции тела
    3. Вертикальная ориентация
    """
    xy = kps[:, :2] if kps.shape[1] >= 2 else kps
    conf = kps[:, 2] if kps.shape[1] >= 3 else np.ones(len(kps))
    vis = conf >= conf_threshold
    
    # Проверка симметричности
    symmetry_score = 0.0
    n_pairs = 0
    
    for l_idx, r_idx in MIRROR_PAIRS:
        if vis[l_idx] and vis[r_idx]:
            # Y-координаты должны быть близки
            y_diff = abs(xy[l_idx, 1] - xy[r_idx, 1])
            x_dist = abs(xy[l_idx, 0] - xy[r_idx, 0])
            if x_dist > 1e-3:
                symmetry_score += min(1.0, y_diff / x_dist)
                n_pairs += 1
    
    if n_pairs > 0:
        avg_asymmetry = symmetry_score / n_pairs
        if avg_asymmetry > 0.5:
            return False
    
    # Проверка вертикальности
    if vis[NOSE] and (vis[HIP_L] or vis[HIP_R]):
        hip_y = xy[HIP_L, 1] if vis[HIP_L] else xy[HIP_R, 1]
        nose_y = xy[NOSE, 1]
        if nose_y >= hip_y:
            return False
    
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# § ВАЛИДАЦИЯ (улучшенная)
# ═══════════════════════════════════════════════════════════════════════════════
def is_pose_valid(pose_data: PoseDict) -> bool:
    """
    Проверяет валидность позы.
    
    Критерии:
    1. Bbox геометрия
    2. Минимум видимых keypoints
    3. Минимум anchor-точек
    4. Похожесть на человека
    """
    kps = pose_data.get("keypoints")
    if kps is None:
        return False

    bbox = pose_data.get("bbox")
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        
        if area < MIN_BBOX_AREA or area > MAX_BBOX_AREA:
            return False
        
        aspect = (x2 - x1) / max(y2 - y1, 1)
        if aspect < MIN_BBOX_ASPECT or aspect > MAX_BBOX_ASPECT:
            return False

    conf = kps[:COCO_N_KPS, 2].astype(np.float32, copy=False)
    if len(conf) < COCO_N_KPS:
        return False

    vis_mask = conf >= MIN_KP_CONFIDENCE
    n_visible = int(np.count_nonzero(vis_mask))
    
    if n_visible < _MIN_VISIBLE_KPS:
        return False

    if float(conf[vis_mask].sum()) / n_visible < MIN_KP_CONFIDENCE:
        return False

    n_anchor = int(np.count_nonzero(conf[ANCHOR_CONF_KPS_ARR] >= MIN_ANCHOR_CONFIDENCE))
    if n_anchor < _MIN_VISIBLE_ANCHORS:
        return False

    if not is_human_like(kps, MIN_KP_CONFIDENCE):
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# § НОРМАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════
def preprocess_pose(pose_data: PoseDict, use_body_weights: bool = True) -> np.ndarray:
    kps = pose_data["keypoints"][:COCO_N_KPS]
    xy = kps[:, :2].astype(np.float32, copy=False)
    conf = kps[:, 2].astype(np.float32, copy=False)

    anc_conf = conf[ANCHOR_KPS_ARR]
    vis_anc = anc_conf >= MIN_KP_CONFIDENCE

    if int(np.count_nonzero(vis_anc)) >= _MIN_VISIBLE_ANCHORS:
        anc_xy = xy[ANCHOR_KPS_ARR]
        vis_f = vis_anc.astype(np.float32)
        weight_sum = vis_f.sum()
        anchor_xy = (vis_f @ anc_xy) / weight_sum
    else:
        vis_all = conf >= MIN_KP_CONFIDENCE
        n_vis = int(np.count_nonzero(vis_all))
        if n_vis > 0:
            vis_f = vis_all.astype(np.float32)
            anchor_xy = (vis_f @ xy) / n_vis
        else:
            anchor_xy = xy.mean(axis=0)

    centered = xy - anchor_xy
    if use_body_weights:
        centered = centered * BODY_WEIGHTS_2D

    flat = centered.ravel()
    norm_sq = float(np.dot(flat, flat))
    norm = (norm_sq ** 0.5) + 1e-5
    return (flat / norm).astype(np.float32)


def batch_preprocess_poses(kps_batch: np.ndarray, use_body_weights: bool = True) -> np.ndarray:
    M = kps_batch.shape[0]
    xy = kps_batch[:, :COCO_N_KPS, :2].astype(np.float32)
    conf = kps_batch[:, :COCO_N_KPS, 2].astype(np.float32)

    anc_conf = conf[:, ANCHOR_KPS_ARR]
    anc_vis = (anc_conf >= MIN_KP_CONFIDENCE).astype(np.float32)
    anc_count = anc_vis.sum(axis=1, keepdims=True)
    anc_xy = xy[:, ANCHOR_KPS_ARR, :]
    num = np.einsum("mi,mij->mj", anc_vis, anc_xy)

    no_anchor = (anc_count[:, 0] < _MIN_VISIBLE_ANCHORS)
    if no_anchor.any():
        all_vis = (conf >= MIN_KP_CONFIDENCE).astype(np.float32)
        all_count = all_vis.sum(axis=1, keepdims=True)
        all_count = np.maximum(all_count, 1.0)
        num_all = np.einsum("mi,mij->mj", all_vis, xy)
        anchor_fallback = num_all / all_count
        anc_count[no_anchor] = 1.0
        num[no_anchor] = anchor_fallback[no_anchor]

    anchor_xy = num / np.maximum(anc_count, 1.0)
    centered = xy - anchor_xy[:, np.newaxis, :]

    if use_body_weights:
        centered = centered * BODY_WEIGHTS_2D[np.newaxis, :, :]

    flat = centered.reshape(M, 34)
    norms = np.sqrt(np.einsum("ij,ij->i", flat, flat)) + 1e-5
    return (flat / norms[:, np.newaxis]).astype(np.float32)


def compute_pose_features(kps_xy: np.ndarray) -> tuple[float, float]:
    anc = kps_xy[ANCHOR_KPS_ARR]
    anchor_xy = anc.mean(axis=0)
    centered = kps_xy - anchor_xy
    scale = float(np.max(np.abs(centered))) + 1e-5
    anchor_y = float(anchor_xy[1])
    return scale, anchor_y


# ═══════════════════════════════════════════════════════════════════════════════
# § ЗЕРКАЛЬНОЕ ОТРАЖЕНИЕ
# ═══════════════════════════════════════════════════════════════════════════════
def mirror_vectors(vec: torch.Tensor, conf: Optional[torch.Tensor] = None) -> torch.Tensor:
    device = str(vec.device)
    src_x, sign_x, src_y = _get_mirror_tensors(device)

    mirrored = torch.empty_like(vec)
    mirrored[:] = vec.index_select(1, src_x)
    mirrored.mul_(sign_x.unsqueeze(0))

    y_changed = _MIRROR_SRC_Y != np.arange(COCO_N_KPS * 2)
    if y_changed.any():
        y_idx = torch.from_numpy(np.where(y_changed)[0].astype(np.int64)).to(device)
        y_src = src_y[y_idx]
        mirrored[:, y_idx] = vec.index_select(1, y_src)

    if conf is not None:
        mirrored[conf == 0.0] = 0.0

    return mirrored


def mirror_pose_with_meta(pose: np.ndarray, meta: dict) -> tuple[np.ndarray, dict]:
    if isinstance(pose, np.ndarray):
        if pose.ndim == 1:
            pose_t = torch.from_numpy(pose).unsqueeze(0)
            mirrored_t = mirror_vectors(pose_t)
            mirrored_vec = mirrored_t.squeeze(0).numpy()
        else:
            mirrored_vec = pose.copy()
            for l_idx, r_idx in MIRROR_PAIRS:
                mirrored_vec[l_idx, 0] = pose[r_idx, 0]
                mirrored_vec[r_idx, 0] = pose[l_idx, 0]
            center_x = pose[:, 0].mean()
            mirrored_vec[:, 0] = 2 * center_x - mirrored_vec[:, 0]
    else:
        mirrored_vec = pose

    dir_map = DIRECTION_MIRROR_MAP
    original_dir = meta.get("dir", "unknown")
    meta["dir"] = dir_map.get(original_dir, original_dir)
    meta["mirrored"] = True

    return mirrored_vec, meta


# ═══════════════════════════════════════════════════════════════════════════════
# § СБОРКА ТЕНЗОРА (с пропорциями тела)
# ═══════════════════════════════════════════════════════════════════════════════
def build_poses_tensor(
    frames_data: list[dict],
    use_body_weights: bool = True,
) -> tuple[torch.Tensor | None, list[dict]]:
    selected_kps: list[np.ndarray] = []
    selected_meta: list[dict] = []

    for frame in frames_data:
        t = float(frame.get("t", 0.0))
        f = int(frame.get("f", 0))
        video_idx = int(frame.get("video_idx", 0))
        direction = frame.get("dir", "forward")
        frame_kp = frame.get("kp")

        poses = frame.get("poses")
        if not poses:
            continue

        best_kps: np.ndarray | None = None
        best_conf: float = -1.0
        best_kp_meta = frame_kp

        for pose in poses:
            if not is_pose_valid(pose):
                continue

            kps = pose.get("keypoints")
            if kps is None:
                kps = pose.get("kp")
            if kps is None:
                continue

            conf_arr = kps[:COCO_N_KPS, 2]
            vis_mask = conf_arr >= MIN_KP_CONFIDENCE
            n_vis = int(np.count_nonzero(vis_mask))
            if n_vis == 0:
                continue

            conf = float(conf_arr[vis_mask].sum()) / n_vis

            if conf > best_conf:
                best_conf = conf
                best_kps = kps[:COCO_N_KPS].astype(np.float32, copy=False)
                if frame_kp is None:
                    kp_raw = pose.get("kp") or pose.get("keypoints")
                    if isinstance(kp_raw, np.ndarray):
                        best_kp_meta = kp_raw.tolist()
                    elif isinstance(kp_raw, list):
                        best_kp_meta = kp_raw

        if best_kps is None:
            continue

        selected_kps.append(best_kps)
        selected_meta.append({
            "t": t,
            "f": f,
            "video_idx": video_idx,
            "dir": direction,
            "kp": best_kp_meta,
        })

    if not selected_kps:
        return None, []

    kps_batch = np.stack(selected_kps, axis=0)
    vectors = batch_preprocess_poses(kps_batch, use_body_weights)

    xy_batch = kps_batch[:, :, :2]
    anc_batch = xy_batch[:, ANCHOR_KPS_ARR, :]
    anchor_xy_b = anc_batch.mean(axis=1)
    centered_b = xy_batch - anchor_xy_b[:, np.newaxis, :]
    scales = np.abs(centered_b).max(axis=(1, 2)) + 1e-5
    anchor_ys = anchor_xy_b[:, 1]

    for i, m in enumerate(selected_meta):
        m["scale"] = float(scales[i])
        m["anchor_y"] = float(anchor_ys[i])
        m["body_proportions"] = compute_body_proportions(kps_batch[i])

    tensor = torch.from_numpy(vectors)
    return tensor, selected_meta