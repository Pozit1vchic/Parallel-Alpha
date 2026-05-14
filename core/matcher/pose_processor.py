#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

# Pre-computed batched arrays for mirror pairs (для векторных свапов)
_MIRROR_L_IDX = np.array([p[0] for p in MIRROR_PAIRS], dtype=np.intp)
_MIRROR_R_IDX = np.array([p[1] for p in MIRROR_PAIRS], dtype=np.intp)

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

# Группы для батчевых пропорций
_LEG_TRIPLES = np.array([
    [HIP_L, KNEE_L, ANKLE_L],
    [HIP_R, KNEE_R, ANKLE_R],
], dtype=np.intp)
_ARM_TRIPLES = np.array([
    [SHOULDER_L, ELBOW_L, WRIST_L],
    [SHOULDER_R, ELBOW_R, WRIST_R],
], dtype=np.intp)


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

# Заранее вычисленные индексы Y, которые реально меняются (для mirror_vectors)
_MIRROR_Y_CHANGED_NP = np.where(_MIRROR_SRC_Y != np.arange(COCO_N_KPS * 2))[0].astype(np.int64)
_MIRROR_Y_SRC_NP = _MIRROR_SRC_Y[_MIRROR_Y_CHANGED_NP]

_MIRROR_SRC_X_T: torch.Tensor | None = None
_MIRROR_SIGN_X_T: torch.Tensor | None = None
_MIRROR_SRC_Y_T: torch.Tensor | None = None
_MIRROR_Y_CHANGED_T: torch.Tensor | None = None
_MIRROR_Y_SRC_T: torch.Tensor | None = None
_MIRROR_DEVICE: str = ""


def _get_mirror_tensors(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _MIRROR_SRC_X_T, _MIRROR_SIGN_X_T, _MIRROR_SRC_Y_T
    global _MIRROR_Y_CHANGED_T, _MIRROR_Y_SRC_T, _MIRROR_DEVICE
    if _MIRROR_SRC_X_T is None or _MIRROR_DEVICE != device:
        _MIRROR_SRC_X_T = torch.from_numpy(_MIRROR_SRC_X).to(device)
        _MIRROR_SIGN_X_T = torch.from_numpy(_MIRROR_SIGN_X).to(device)
        _MIRROR_SRC_Y_T = torch.from_numpy(_MIRROR_SRC_Y).to(device)
        _MIRROR_Y_CHANGED_T = torch.from_numpy(_MIRROR_Y_CHANGED_NP).to(device)
        _MIRROR_Y_SRC_T = torch.from_numpy(_MIRROR_Y_SRC_NP).to(device)
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

# ── Пороги ────────────────────────────────────────────────────────────────────
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
# § Batched body proportions (НОВОЕ — для build_poses_tensor)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_body_proportions_batch(
    kps_batch: np.ndarray,
    conf_threshold: float = MIN_KP_CONFIDENCE,
) -> list[dict]:
    """
    Векторное вычисление пропорций тела для батча поз.
    kps_batch: (M, 17, 3) или (M, 17, 2)

    Возвращает список dict длиной M с теми же ключами, что и
    одиночная compute_body_proportions.
    """
    M = kps_batch.shape[0]
    if M == 0:
        return []

    has_conf = kps_batch.shape[2] >= 3
    xy = kps_batch[:, :, :2].astype(np.float32, copy=False)
    if has_conf:
        conf = kps_batch[:, :, 2].astype(np.float32, copy=False)
        vis = conf >= conf_threshold                     # (M, 17) bool
    else:
        vis = np.ones((M, COCO_N_KPS), dtype=bool)

    # Дефолты
    leg_to_torso = np.full(M, 1.0, dtype=np.float32)
    shoulder_to_height = np.full(M, 0.3, dtype=np.float32)
    torso_aspect = np.full(M, 0.5, dtype=np.float32)
    arm_to_torso = np.full(M, 0.8, dtype=np.float32)
    valid = np.zeros(M, dtype=bool)

    # Базовое условие: оба плеча и оба бедра видны
    base_ok = vis[:, SHOULDER_L] & vis[:, SHOULDER_R] & vis[:, HIP_L] & vis[:, HIP_R]

    if not base_ok.any():
        return [
            {"leg_to_torso": float(leg_to_torso[i]),
             "shoulder_to_height": float(shoulder_to_height[i]),
             "torso_aspect": float(torso_aspect[i]),
             "arm_to_torso": float(arm_to_torso[i]),
             "valid": bool(valid[i])}
            for i in range(M)
        ]

    # Центры плеч / бёдер
    shoulder_center = (xy[:, SHOULDER_L] + xy[:, SHOULDER_R]) * 0.5  # (M, 2)
    hip_center = (xy[:, HIP_L] + xy[:, HIP_R]) * 0.5
    torso_length = np.linalg.norm(hip_center - shoulder_center, axis=1)  # (M,)
    torso_ok = base_ok & (torso_length >= 1e-3)

    # Длина ног — для каждой ноги (M, 2) считаем сегменты
    leg_lengths = np.zeros((M, 2), dtype=np.float32)
    leg_visible = np.zeros((M, 2), dtype=bool)
    for j, (h_i, k_i, a_i) in enumerate(_LEG_TRIPLES):
        leg_visible[:, j] = vis[:, h_i] & vis[:, k_i] & vis[:, a_i]
        thigh = np.linalg.norm(xy[:, k_i] - xy[:, h_i], axis=1)
        shin = np.linalg.norm(xy[:, a_i] - xy[:, k_i], axis=1)
        leg_lengths[:, j] = thigh + shin

    n_legs = leg_visible.sum(axis=1)  # (M,)
    legs_sum = (leg_lengths * leg_visible.astype(np.float32)).sum(axis=1)
    leg_avg = np.where(n_legs > 0, legs_sum / np.maximum(n_legs, 1), 0.0)

    safe_torso = np.maximum(torso_length, 1e-9)
    leg_to_torso_v = np.where(torso_ok & (n_legs > 0), leg_avg / safe_torso, 1.0)

    # Ширина плеч
    shoulder_width = np.linalg.norm(xy[:, SHOULDER_R] - xy[:, SHOULDER_L], axis=1)
    torso_aspect_v = np.where(torso_ok, shoulder_width / safe_torso, 0.5)

    # Высота — если виден нос
    nose_ok = torso_ok & vis[:, NOSE]
    nose_to_hip = np.linalg.norm(xy[:, NOSE] - hip_center, axis=1)
    height = nose_to_hip + leg_avg
    shoulder_to_height_v = np.where(
        nose_ok & (height > 1e-3),
        shoulder_width / np.maximum(height, 1e-9),
        0.3,
    )

    # Длина рук
    arm_lengths = np.zeros((M, 2), dtype=np.float32)
    arm_visible = np.zeros((M, 2), dtype=bool)
    for j, (s_i, e_i, w_i) in enumerate(_ARM_TRIPLES):
        arm_visible[:, j] = vis[:, s_i] & vis[:, e_i] & vis[:, w_i]
        upper = np.linalg.norm(xy[:, e_i] - xy[:, s_i], axis=1)
        lower = np.linalg.norm(xy[:, w_i] - xy[:, e_i], axis=1)
        arm_lengths[:, j] = upper + lower

    n_arms = arm_visible.sum(axis=1)
    arms_sum = (arm_lengths * arm_visible.astype(np.float32)).sum(axis=1)
    arm_avg = np.where(n_arms > 0, arms_sum / np.maximum(n_arms, 1), 0.0)
    arm_to_torso_v = np.where(torso_ok & (n_arms > 0), arm_avg / safe_torso, 0.8)

    # Применяем
    leg_to_torso = np.where(torso_ok, leg_to_torso_v, leg_to_torso)
    shoulder_to_height = np.where(torso_ok, shoulder_to_height_v, shoulder_to_height)
    torso_aspect = np.where(torso_ok, torso_aspect_v, torso_aspect)
    arm_to_torso = np.where(torso_ok, arm_to_torso_v, arm_to_torso)
    valid = torso_ok

    return [
        {
            "leg_to_torso": float(leg_to_torso[i]),
            "shoulder_to_height": float(shoulder_to_height[i]),
            "torso_aspect": float(torso_aspect[i]),
            "arm_to_torso": float(arm_to_torso[i]),
            "valid": bool(valid[i]),
        }
        for i in range(M)
    ]


def compute_body_proportions(kps: np.ndarray, conf_threshold: float = MIN_KP_CONFIDENCE) -> dict:
    """Совместимый одиночный wrapper над батчевой версией."""
    if kps.ndim == 2:
        kps = kps[np.newaxis, ...]
    res = compute_body_proportions_batch(kps, conf_threshold)
    return res[0] if res else {
        "leg_to_torso": 1.0, "shoulder_to_height": 0.3,
        "torso_aspect": 0.5, "arm_to_torso": 0.8, "valid": False,
    }


def compare_body_proportions(props1: dict, props2: dict) -> float:
    if not props1.get("valid") or not props2.get("valid"):
        return 0.5
    keys = ("leg_to_torso", "shoulder_to_height", "torso_aspect", "arm_to_torso")
    diffs = []
    for key in keys:
        v1, v2 = props1.get(key, 1.0), props2.get(key, 1.0)
        m = max(v1, v2)
        if m < 1e-5:
            continue
        diffs.append(abs(v1 - v2) / m)
    if not diffs:
        return 0.5
    avg_diff = sum(diffs) / len(diffs)
    return max(0.0, 1.0 - avg_diff * 3.0)


# ═══════════════════════════════════════════════════════════════════════════════
# § Batched is_human_like (НОВОЕ)
# ═══════════════════════════════════════════════════════════════════════════════

def is_human_like_batch(
    kps_batch: np.ndarray,
    conf_threshold: float = MIN_KP_CONFIDENCE,
) -> np.ndarray:
    """
    Векторная проверка "похож на человека" для батча. (M, 17, 3) → (M,) bool.
    """
    M = kps_batch.shape[0]
    if M == 0:
        return np.zeros(0, dtype=bool)

    xy = kps_batch[:, :, :2]
    has_conf = kps_batch.shape[2] >= 3
    if has_conf:
        vis = kps_batch[:, :, 2] >= conf_threshold
    else:
        vis = np.ones((M, COCO_N_KPS), dtype=bool)

    # Симметрия — для всех пар сразу (M, n_pairs)
    n_pairs = len(MIRROR_PAIRS)
    sym_score = np.zeros(M, dtype=np.float32)
    n_pair_ok = np.zeros(M, dtype=np.int32)
    for l_i, r_i in MIRROR_PAIRS:
        pair_ok = vis[:, l_i] & vis[:, r_i]
        y_diff = np.abs(xy[:, l_i, 1] - xy[:, r_i, 1])
        x_dist = np.abs(xy[:, l_i, 0] - xy[:, r_i, 0])
        contributes = pair_ok & (x_dist > 1e-3)
        ratio = np.where(contributes, np.minimum(1.0, y_diff / np.maximum(x_dist, 1e-9)), 0.0)
        sym_score += ratio
        n_pair_ok += contributes.astype(np.int32)

    avg_asym = np.where(n_pair_ok > 0, sym_score / np.maximum(n_pair_ok, 1), 0.0)
    bad_sym = (n_pair_ok > 0) & (avg_asym > 0.5)

    # Вертикальность: nose_y < hip_y (image coords: верх — меньше y)
    has_hip = vis[:, HIP_L] | vis[:, HIP_R]
    hip_y = np.where(vis[:, HIP_L], xy[:, HIP_L, 1], xy[:, HIP_R, 1])
    nose_y = xy[:, NOSE, 1]
    bad_orient = vis[:, NOSE] & has_hip & (nose_y >= hip_y)

    return ~(bad_sym | bad_orient)


def is_human_like(kps: np.ndarray, conf_threshold: float = MIN_KP_CONFIDENCE) -> bool:
    if kps.ndim == 2:
        return bool(is_human_like_batch(kps[np.newaxis, ...], conf_threshold)[0])
    return bool(is_human_like_batch(kps, conf_threshold)[0])


# ═══════════════════════════════════════════════════════════════════════════════
# § Batched validation (НОВОЕ)
# ═══════════════════════════════════════════════════════════════════════════════

def is_pose_valid_batch(
    kps_batch: np.ndarray,
    bboxes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Векторная проверка валидности поз. (M, 17, 3) → (M,) bool.
    bboxes: (M, 4) [x1,y1,x2,y2] или None.
    """
    M = kps_batch.shape[0]
    if M == 0:
        return np.zeros(0, dtype=bool)

    valid = np.ones(M, dtype=bool)

    # bbox-проверка
    if bboxes is not None:
        x1 = bboxes[:, 0]; y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]; y2 = bboxes[:, 3]
        w = x2 - x1
        h = y2 - y1
        area = w * h
        aspect = w / np.maximum(h, 1.0)
        valid &= (area >= MIN_BBOX_AREA) & (area <= MAX_BBOX_AREA)
        valid &= (aspect >= MIN_BBOX_ASPECT) & (aspect <= MAX_BBOX_ASPECT)

    if kps_batch.shape[1] < COCO_N_KPS:
        return np.zeros(M, dtype=bool)

    conf = kps_batch[:, :COCO_N_KPS, 2].astype(np.float32, copy=False)
    vis_mask = conf >= MIN_KP_CONFIDENCE
    n_visible = vis_mask.sum(axis=1)
    valid &= n_visible >= _MIN_VISIBLE_KPS

    # Средний conf по видимым
    sum_conf = (conf * vis_mask).sum(axis=1)
    avg_conf = np.where(n_visible > 0, sum_conf / np.maximum(n_visible, 1), 0.0)
    valid &= avg_conf >= MIN_KP_CONFIDENCE

    # Anchor-точки
    anchor_conf = conf[:, ANCHOR_CONF_KPS_ARR]
    n_anchor = (anchor_conf >= MIN_ANCHOR_CONFIDENCE).sum(axis=1)
    valid &= n_anchor >= _MIN_VISIBLE_ANCHORS

    # Human-like
    if valid.any():
        hl = is_human_like_batch(kps_batch[:, :COCO_N_KPS, :])
        valid &= hl

    return valid


def is_pose_valid(pose_data: PoseDict) -> bool:
    """Совместимый одиночный API."""
    kps = pose_data.get("keypoints")
    if kps is None:
        return False
    bbox = pose_data.get("bbox")
    bboxes = None
    if bbox and len(bbox) == 4:
        bboxes = np.array([bbox], dtype=np.float32)
    if kps.ndim == 2:
        kps_b = kps[np.newaxis, ...]
    else:
        kps_b = kps
    return bool(is_pose_valid_batch(kps_b, bboxes)[0])


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
# § ЗЕРКАЛЬНОЕ ОТРАЖЕНИЕ — кэшированные индексы
# ═══════════════════════════════════════════════════════════════════════════════
def mirror_vectors(vec: torch.Tensor, conf: Optional[torch.Tensor] = None) -> torch.Tensor:
    device = str(vec.device)
    src_x, sign_x, _ = _get_mirror_tensors(device)

    mirrored = vec.index_select(1, src_x) * sign_x.unsqueeze(0)

    # Перестановка y (только реально меняющихся индексов)
    if _MIRROR_Y_CHANGED_T is not None and _MIRROR_Y_CHANGED_T.numel() > 0:
        y_idx = _MIRROR_Y_CHANGED_T
        y_src = _MIRROR_Y_SRC_T
        # Получаем y-исходники из vec и записываем по y_idx
        mirrored.index_copy_(1, y_idx, vec.index_select(1, y_src))

    if conf is not None:
        mirrored = mirrored.clone() if mirrored is vec else mirrored
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
            mirrored_vec[_MIRROR_L_IDX, 0] = pose[_MIRROR_R_IDX, 0]
            mirrored_vec[_MIRROR_R_IDX, 0] = pose[_MIRROR_L_IDX, 0]
            center_x = pose[:, 0].mean()
            mirrored_vec[:, 0] = 2 * center_x - mirrored_vec[:, 0]
    else:
        mirrored_vec = pose

    original_dir = meta.get("dir", "unknown")
    meta["dir"] = DIRECTION_MIRROR_MAP.get(original_dir, original_dir)
    meta["mirrored"] = True

    return mirrored_vec, meta


# ═══════════════════════════════════════════════════════════════════════════════
# § СБОРКА ТЕНЗОРА — переписана: один проход, батчевая валидация
# ═══════════════════════════════════════════════════════════════════════════════
def build_poses_tensor(
    frames_data: list[dict],
    use_body_weights: bool = True,
) -> tuple[torch.Tensor | None, list[dict]]:
    """
    Собирает тензор поз с минимальным Python-overhead-ом.

    Стратегия:
    1) Один проход по frames_data — собираем кандидатов (best-pose на кадр)
       во временный массив. Валидация делается потом батчем.
    2) Батч-валидация is_pose_valid_batch на всех кандидатах.
    3) Батч-нормализация + батч-пропорции тела.
    """
    if not frames_data:
        return None, []

    # Этап 1: для каждого кадра выбираем best-pose (без валидации)
    cand_kps: list[np.ndarray] = []
    cand_meta: list[dict] = []
    cand_bbox: list[Optional[list]] = []

    for frame in frames_data:
        poses = frame.get("poses")
        if not poses:
            continue
        t = float(frame.get("t", 0.0))
        f = int(frame.get("f", 0))
        video_idx = int(frame.get("video_idx", 0))
        direction = frame.get("dir", "forward")
        frame_kp = frame.get("kp")

        best_kps = None
        best_conf = -1.0
        best_kp_meta = frame_kp
        best_bbox = None

        for pose in poses:
            kps = pose.get("keypoints")
            if kps is None:
                kps = pose.get("kp")
            if kps is None:
                continue
            if not isinstance(kps, np.ndarray):
                try:
                    kps = np.asarray(kps, dtype=np.float32)
                except Exception:
                    continue
            if kps.shape[0] < COCO_N_KPS:
                continue

            # Быстрый pre-filter: средний conf по видимым
            conf_arr = kps[:COCO_N_KPS, 2]
            vis_mask = conf_arr >= MIN_KP_CONFIDENCE
            n_vis = int(vis_mask.sum())
            if n_vis == 0:
                continue
            conf = float(conf_arr[vis_mask].sum()) / n_vis

            if conf > best_conf:
                best_conf = conf
                best_kps = kps[:COCO_N_KPS].astype(np.float32, copy=False)
                best_bbox = pose.get("bbox")
                if frame_kp is None:
                    kp_raw = pose.get("kp") or pose.get("keypoints")
                    if isinstance(kp_raw, np.ndarray):
                        best_kp_meta = kp_raw.tolist()
                    elif isinstance(kp_raw, list):
                        best_kp_meta = kp_raw

        if best_kps is None:
            continue

        cand_kps.append(best_kps)
        cand_bbox.append(best_bbox if (best_bbox and len(best_bbox) == 4) else None)
        cand_meta.append({
            "t": t, "f": f, "video_idx": video_idx,
            "dir": direction, "kp": best_kp_meta,
        })

    if not cand_kps:
        return None, []

    # Этап 2: батч-валидация
    kps_batch = np.stack(cand_kps, axis=0)  # (M, 17, 3)

    # Подготовим bboxes (где есть)
    has_bbox = any(b is not None for b in cand_bbox)
    if has_bbox:
        # Для отсутствующих bbox используем sentinel, который пройдёт все проверки
        sentinel = [0.0, 0.0, float(np.sqrt(MIN_BBOX_AREA + 1)),
                    float(np.sqrt(MIN_BBOX_AREA + 1))]
        bb_arr = np.array(
            [b if b is not None else sentinel for b in cand_bbox],
            dtype=np.float32,
        )
    else:
        bb_arr = None

    valid_mask = is_pose_valid_batch(kps_batch, bb_arr)
    if not valid_mask.any():
        return None, []

    kps_batch = kps_batch[valid_mask]
    selected_meta = [cand_meta[i] for i in np.where(valid_mask)[0]]

    # Этап 3: нормализация + scale/anchor + пропорции
    vectors = batch_preprocess_poses(kps_batch, use_body_weights)

    xy_batch = kps_batch[:, :, :2]
    anc_batch = xy_batch[:, ANCHOR_KPS_ARR, :]
    anchor_xy_b = anc_batch.mean(axis=1)
    centered_b = xy_batch - anchor_xy_b[:, np.newaxis, :]
    scales = np.abs(centered_b).max(axis=(1, 2)) + 1e-5
    anchor_ys = anchor_xy_b[:, 1]

    body_props = compute_body_proportions_batch(kps_batch)

    for i, m in enumerate(selected_meta):
        m["scale"] = float(scales[i])
        m["anchor_y"] = float(anchor_ys[i])
        m["body_proportions"] = body_props[i]

    tensor = torch.from_numpy(vectors)
    return tensor, selected_meta
