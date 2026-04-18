#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/matcher/pose_processor.py
v16 — Refactored for maximum accuracy and performance.

Ключевые изменения v16:
- MIN_VISIBLE_KPS снижен с 8 до 6 для частично видимых людей
- Добавлены геометрические проверки bbox (area, aspect ratio)
- mirror_pose_with_meta() теперь инвертирует direction (left↔right)
- Оптимизирована векторизация через np.einsum
- Добавлены docstring для scale и anchor_y
- Улучшен fallback для anchor-точек

Параметры:
- MIN_VISIBLE_KPS = 6 (было 8)
- MIN_KP_CONFIDENCE = 0.22 (было 0.28)
- MIN_ANCHOR_CONFIDENCE = 0.25 (было 0.31)
- Bbox фильтры: area [400, 600000], aspect [0.2, 5.0]
"""

from __future__ import annotations
from typing import TypeAlias, Optional

import numpy as np
import torch

# ── Типы ──────────────────────────────────────────────────────────────────────
PoseDict: TypeAlias = dict

# ── Константы COCO-17 ─────────────────────────────────────────────────────────
COCO_N_KPS = 17

# Опорные точки: плечи (5,6) + бёдра (11,12)
ANCHOR_KPS: list[int] = [5, 6, 11, 12]
ANCHOR_KPS_ARR = np.array(ANCHOR_KPS, dtype=np.intp)

# Парные точки для зеркального отражения
MIRROR_PAIRS: list[tuple[int, int]] = [
    (1, 2), (3, 4),
    (5, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16),
]

_PAIRED_INDICES: frozenset[int] = frozenset(i for pair in MIRROR_PAIRS for i in pair)
_UNPAIRED_INDICES: list[int] = [i for i in range(COCO_N_KPS) if i not in _PAIRED_INDICES]

# ── Предвычисленные индексы для mirror_vectors ────────────────────────────────
def _build_mirror_indices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Строит предвычисленные индексы для быстрого зеркального отражения поз.
    
    Возвращает:
        src_x: индексы для X-координат (с заменой пар)
        sign_x: знаки для X-координат (-1 для инверсии)
        src_y: индексы для Y-координат (с заменой пар)
    """
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
    """Ленивая инициализация тензоров для mirror_vectors на указанном устройстве."""
    global _MIRROR_SRC_X_T, _MIRROR_SIGN_X_T, _MIRROR_SRC_Y_T, _MIRROR_DEVICE
    if _MIRROR_SRC_X_T is None or _MIRROR_DEVICE != device:
        _MIRROR_SRC_X_T = torch.from_numpy(_MIRROR_SRC_X).to(device)
        _MIRROR_SIGN_X_T = torch.from_numpy(_MIRROR_SIGN_X).to(device)
        _MIRROR_SRC_Y_T = torch.from_numpy(_MIRROR_SRC_Y).to(device)
        _MIRROR_DEVICE = device
    return _MIRROR_SRC_X_T, _MIRROR_SIGN_X_T, _MIRROR_SRC_Y_T

# ── Веса частей тела (оптимизированные для точности) ──────────────────────────
BODY_WEIGHTS = np.array([
    0.9,  # 0 нос
    0.5, 0.5,  # 1,2 глаза
    0.4, 0.4,  # 3,4 уши
    1.8, 1.8,  # 5,6 плечи ★ (ключевые)
    1.4, 1.4,  # 7,8 локти
    1.1, 1.1,  # 9,10 кисти
    1.8, 1.8,  # 11,12 бёдра ★ (ключевые)
    1.5, 1.5,  # 13,14 колени
    1.1, 1.1,  # 15,16 стопы
], dtype=np.float32)

BODY_WEIGHTS_2D: np.ndarray = BODY_WEIGHTS[:, np.newaxis]

# ── Пороги (снижены для повышения recall) ─────────────────────────────────────
MIN_KP_CONFIDENCE = 0.22       # Снижено с 0.28 для частично видимых
MIN_ANCHOR_CONFIDENCE = 0.25   # Снижено с 0.31
ANCHOR_CONF_KPS = [5, 6, 11, 12, 13, 14]
ANCHOR_CONF_KPS_ARR = np.array(ANCHOR_CONF_KPS, dtype=np.intp)

_MIN_VISIBLE_KPS = 6           # Снижено с 8
_MIN_VISIBLE_ANCHORS = 2       # Оставлено 2

# ── Bbox фильтры (геометрические проверки) ────────────────────────────────────
MIN_BBOX_AREA = 400            # Минимальная площадь bbox
MAX_BBOX_AREA = 600000         # Максимальная площадь
MIN_BBOX_ASPECT = 0.2          # Минимальный aspect ratio
MAX_BBOX_ASPECT = 5.0          # Максимальный aspect ratio

# ── Маппинг направлений для зеркалирования ────────────────────────────────────
DIRECTION_MIRROR_MAP = {
    "left": "right",
    "right": "left",
    "forward": "forward",
    "back": "back",
    "unknown": "unknown",
}


# ═══════════════════════════════════════════════════════════════════════════════
# § 1. ВАЛИДАЦИЯ (улучшенная с геометрическими проверками)
# ═══════════════════════════════════════════════════════════════════════════════
def is_pose_valid(pose_data: PoseDict) -> bool:
    """
    Проверяет валидность позы по множеству критериев.
    
    Критерии валидации:
    1. Наличие keypoints
    2. Bbox геометрические проверки (area, aspect ratio)
    3. Минимальное количество видимых keypoints (>= 6)
    4. Средняя уверенность видимых keypoints
    5. Минимальное количество видимых anchor-точек (>= 2)
    
    Args:
        pose_data: Словарь с ключами 'keypoints' и опционально 'bbox'
        
    Returns:
        bool: True если поза валидна
    """
    kps = pose_data.get("keypoints")
    if kps is None:
        return False

    # ── Проверка bbox (геометрические фильтры) ─────────────────────────────────
    bbox = pose_data.get("bbox")
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        
        # Фильтр по площади
        if area < MIN_BBOX_AREA or area > MAX_BBOX_AREA:
            return False
        
        # Фильтр по aspect ratio
        aspect = (x2 - x1) / max(y2 - y1, 1)
        if aspect < MIN_BBOX_ASPECT or aspect > MAX_BBOX_ASPECT:
            return False

    # ── Проверка уверенности keypoints ────────────────────────────────────────
    conf = kps[:COCO_N_KPS, 2].astype(np.float32, copy=False)
    if len(conf) < COCO_N_KPS:
        return False

    vis_mask = conf >= MIN_KP_CONFIDENCE
    n_visible = int(np.count_nonzero(vis_mask))
    
    # Минимальное количество видимых keypoints
    if n_visible < _MIN_VISIBLE_KPS:
        return False

    # Средняя уверенность видимых keypoints
    if float(conf[vis_mask].sum()) / n_visible < MIN_KP_CONFIDENCE:
        return False

    # ── Проверка anchor-точек ─────────────────────────────────────────────────
    n_anchor = int(np.count_nonzero(conf[ANCHOR_CONF_KPS_ARR] >= MIN_ANCHOR_CONFIDENCE))
    if n_anchor < _MIN_VISIBLE_ANCHORS:
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# § 2. НОРМАЛИЗАЦИЯ ОДНОЙ ПОЗЫ (оптимизированная)
# ═══════════════════════════════════════════════════════════════════════════════
def preprocess_pose(pose_data: PoseDict, use_body_weights: bool = True) -> np.ndarray:
    """
    Нормализует одну позу: центрирование, взвешивание, L2-нормализация.
    
    Алгоритм:
    1. Вычисляет центр (anchor) по плечам и бёдрам
    2. Центрирует keypoints относительно anchor
    3. Применяет веса частей тела (если use_body_weights=True)
    4. L2-нормализует результат
    
    Args:
        pose_data: Словарь с ключом 'keypoints'
        use_body_weights: Применять ли веса частей тела
        
    Returns:
        np.ndarray: Нормализованный вектор позы (34 элемента)
    """
    kps = pose_data["keypoints"][:COCO_N_KPS]
    xy = kps[:, :2].astype(np.float32, copy=False)
    conf = kps[:, 2].astype(np.float32, copy=False)

    # ── Вычисление anchor-точки ───────────────────────────────────────────────
    anc_conf = conf[ANCHOR_KPS_ARR]
    vis_anc = anc_conf >= MIN_KP_CONFIDENCE

    if int(np.count_nonzero(vis_anc)) >= _MIN_VISIBLE_ANCHORS:
        anc_xy = xy[ANCHOR_KPS_ARR]
        vis_f = vis_anc.astype(np.float32)
        weight_sum = vis_f.sum()
        anchor_xy = (vis_f @ anc_xy) / weight_sum
    else:
        # Fallback: используем все видимые keypoints
        vis_all = conf >= MIN_KP_CONFIDENCE
        n_vis = int(np.count_nonzero(vis_all))
        if n_vis > 0:
            vis_f = vis_all.astype(np.float32)
            anchor_xy = (vis_f @ xy) / n_vis
        else:
            anchor_xy = xy.mean(axis=0)

    # ── Центрирование и взвешивание ───────────────────────────────────────────
    centered = xy - anchor_xy
    if use_body_weights:
        centered = centered * BODY_WEIGHTS_2D

    # ── L2-нормализация ───────────────────────────────────────────────────────
    flat = centered.ravel()
    norm_sq = float(np.dot(flat, flat))
    norm = (norm_sq ** 0.5) + 1e-5
    return (flat / norm).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# § 3. БАТЧ-НОРМАЛИЗАЦИЯ (оптимизированная)
# ═══════════════════════════════════════════════════════════════════════════════
def batch_preprocess_poses(kps_batch: np.ndarray, use_body_weights: bool = True) -> np.ndarray:
    """
    Векторизованная батч-нормализация поз.
    
    Использует np.einsum для эффективного вычисления взвешенных сумм.
    
    Args:
        kps_batch: Массив поз形状 (N, 17, 3)
        use_body_weights: Применять ли веса частей тела
        
    Returns:
        np.ndarray: Нормализованные векторы形状 (N, 34)
    """
    M = kps_batch.shape[0]
    xy = kps_batch[:, :COCO_N_KPS, :2].astype(np.float32)
    conf = kps_batch[:, :COCO_N_KPS, 2].astype(np.float32)

    # ── Вычисление anchor для каждой позы ─────────────────────────────────────
    anc_conf = conf[:, ANCHOR_KPS_ARR]
    anc_vis = (anc_conf >= MIN_KP_CONFIDENCE).astype(np.float32)
    anc_count = anc_vis.sum(axis=1, keepdims=True)
    anc_xy = xy[:, ANCHOR_KPS_ARR, :]
    num = np.einsum("mi,mij->mj", anc_vis, anc_xy)

    # ── Fallback для поз без достаточного количества anchor ───────────────────
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

    # ── L2-нормализация ───────────────────────────────────────────────────────
    flat = centered.reshape(M, 34)
    norms = np.sqrt(np.einsum("ij,ij->i", flat, flat)) + 1e-5
    return (flat / norms[:, np.newaxis]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# § 4. ВЫЧИСЛЕНИЕ ПРИЗНАКОВ (scale, anchor_y)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_pose_features(kps_xy: np.ndarray) -> tuple[float, float]:
    """
    Вычисляет геометрические признаки позы.
    
    Args:
        kps_xy: Keypoints形状 (17, 2)
        
    Returns:
        tuple[float, float]: (scale, anchor_y)
            - scale: Максимальное отклонение от центра (размер позы)
            - anchor_y: Y-координата центра позы (высота в кадре)
    
    Примечание:
        scale используется для штрафа за различие в размерах поз.
        anchor_y используется для штрафа за различие в высоте положения.
    """
    anc = kps_xy[ANCHOR_KPS_ARR]
    anchor_xy = anc.mean(axis=0)
    centered = kps_xy - anchor_xy
    scale = float(np.max(np.abs(centered))) + 1e-5
    anchor_y = float(anchor_xy[1])
    return scale, anchor_y


# ═══════════════════════════════════════════════════════════════════════════════
# § 5. ЗЕРКАЛЬНОЕ ОТРАЖЕНИЕ
# ═══════════════════════════════════════════════════════════════════════════════
def mirror_vectors(vec: torch.Tensor, conf: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Зеркально отражает векторы поз по вертикальной оси.
    
    Args:
        vec: Тензор векторов поз形状 (N, 34)
        conf: Опционально, уверенность keypoints для маскирования
        
    Returns:
        torch.Tensor: Отражённые векторы
    """
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
    """
    Зеркально отражает позу и обновляет метаданные.
    
    Критическое исправление: инвертирует направление (left↔right).
    
    Args:
        pose: Вектор позы или keypoints
        meta: Метаданные с ключом 'dir'
        
    Returns:
        tuple[np.ndarray, dict]: (отражённая поза, обновлённые метаданные)
    """
    # Отражаем вектор
    if isinstance(pose, np.ndarray):
        if pose.ndim == 1:
            # Вектор (34,) — используем torch для отражения
            pose_t = torch.from_numpy(pose).unsqueeze(0)
            mirrored_t = mirror_vectors(pose_t)
            mirrored_vec = mirrored_t.squeeze(0).numpy()
        else:
            # Keypoints (17, 3) — отражаем X-координаты
            mirrored_vec = pose.copy()
            for l_idx, r_idx in MIRROR_PAIRS:
                mirrored_vec[l_idx, 0] = pose[r_idx, 0]
                mirrored_vec[r_idx, 0] = pose[l_idx, 0]
            # Инвертируем X относительно центра
            center_x = pose[:, 0].mean()
            mirrored_vec[:, 0] = 2 * center_x - mirrored_vec[:, 0]
    else:
        mirrored_vec = pose

    # Инвертируем направление
    dir_map = DIRECTION_MIRROR_MAP
    original_dir = meta.get("dir", "unknown")
    meta["dir"] = dir_map.get(original_dir, original_dir)
    meta["mirrored"] = True

    return mirrored_vec, meta


# ═══════════════════════════════════════════════════════════════════════════════
# § 6. СБОРКА ТЕНЗОРА (улучшенная)
# ═══════════════════════════════════════════════════════════════════════════════
def build_poses_tensor(
    frames_data: list[dict],
    use_body_weights: bool = True,
) -> tuple[torch.Tensor | None, list[dict]]:
    """
    Собирает тензор поз из данных кадров.
    
    Для каждой позы вычисляет:
    - Нормализованный вектор (34 элемента)
    - scale: размер позы (для штрафа за различие масштабов)
    - anchor_y: высота центра позы (для штрафа за различие положения)
    - dir: направление движения (left/right/forward/back)
    
    Args:
        frames_data: Список кадров с позами
        use_body_weights: Применять ли веса частей тела
        
    Returns:
        tuple[torch.Tensor | None, list[dict]]: (тензор поз, метаданные)
    """
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

    # ── Векторизованная нормализация ──────────────────────────────────────────
    kps_batch = np.stack(selected_kps, axis=0)
    vectors = batch_preprocess_poses(kps_batch, use_body_weights)

    # ── Вычисление scale и anchor_y для каждой позы ───────────────────────────
    xy_batch = kps_batch[:, :, :2]
    anc_batch = xy_batch[:, ANCHOR_KPS_ARR, :]
    anchor_xy_b = anc_batch.mean(axis=1)
    centered_b = xy_batch - anchor_xy_b[:, np.newaxis, :]
    scales = np.abs(centered_b).max(axis=(1, 2)) + 1e-5
    anchor_ys = anchor_xy_b[:, 1]

    for i, m in enumerate(selected_meta):
        m["scale"] = float(scales[i])
        m["anchor_y"] = float(anchor_ys[i])

    tensor = torch.from_numpy(vectors)
    return tensor, selected_meta
