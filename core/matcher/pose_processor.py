#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pose_processor.py — ЕДИНЫЙ источник правды для нормализации поз.
"""
from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch

# ── Типы ─────────────────────────────────────────────────────────────────────

PoseDict: TypeAlias = dict

# ── Константы COCO-17 ─────────────────────────────────────────────────────────

COCO_N_KPS = 17

# Опорные точки: плечи (5,6) + бёдра (11,12)
ANCHOR_KPS: list[int] = [5, 6, 11, 12]

# Парные точки для зеркального отражения
MIRROR_PAIRS: list[tuple[int, int]] = [
    (1, 2), (3, 4),
    (5, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16),
]

# Веса частей тела
BODY_WEIGHTS = np.array([
    0.8,            # 0  нос
    0.4,  0.4,      # 1,2  глаза
    0.3,  0.3,      # 3,4  уши
    1.5,  1.5,      # 5,6  плечи   ★
    1.2,  1.2,      # 7,8  локти
    1.0,  1.0,      # 9,10 кисти
    1.5,  1.5,      # 11,12 бёдра  ★
    1.3,  1.3,      # 13,14 колени
    1.0,  1.0,      # 15,16 стопы
], dtype=np.float32)

# Пороги уверенности
MIN_KP_CONFIDENCE     = 0.28   # было 0.30
MIN_ANCHOR_CONFIDENCE = 0.31   # было 0.35
ANCHOR_CONF_KPS       = [5, 6, 11, 12, 13, 14]


# ── Валидация ─────────────────────────────────────────────────────────────────

def is_pose_valid(pose_data: PoseDict) -> bool:
    """
    Проверить качество позы.

    Требования:
    - keypoints.shape[0] >= 17
    - Средняя уверенность всех точек >= MIN_KP_CONFIDENCE
    - Средняя уверенность опорных точек >= MIN_ANCHOR_CONFIDENCE
    """
    kps = pose_data.get("keypoints")
    if kps is None or kps.shape[0] < COCO_N_KPS:
        return False

    conf_all    = kps[:COCO_N_KPS, 2].astype(np.float32)
    conf_anchor = kps[ANCHOR_CONF_KPS, 2].astype(np.float32)

    return (
        float(conf_all.mean())    >= MIN_KP_CONFIDENCE
        and float(conf_anchor.mean()) >= MIN_ANCHOR_CONFIDENCE
    )


# ── Нормализация ──────────────────────────────────────────────────────────────

def preprocess_pose(
    pose_data:        PoseDict,
    use_body_weights: bool = True,
) -> np.ndarray:
    """
    Нормализация позы → плоский вектор (34,).

    Алгоритм:
    1. Берём (x, y) первых 17 точек
    2. Центроид по ОПОРНЫМ точкам (плечи + бёдра)
    3. Делим на max(|coords - центроид|) → [-1, 1]
    4. Умножаем на BODY_WEIGHTS (опционально), повторно нормируем
    5. flatten → (34,) float32
    """
    kps      = pose_data["keypoints"][:COCO_N_KPS, :2].astype(np.float32)
    anchor   = kps[ANCHOR_KPS].mean(axis=0)
    centered = kps - anchor
    scale    = np.max(np.abs(centered)) + 1e-5
    normed   = centered / scale

    if use_body_weights:
        normed = normed * BODY_WEIGHTS[:, None]
        s2     = np.max(np.abs(normed)) + 1e-5
        normed = normed / s2

    return normed.flatten().astype(np.float32)  # (34,)


def compute_pose_features(kps_xy: np.ndarray) -> tuple[float, float]:
    """
    Вычислить (scale, anchor_y) для re-scoring в matcher.
    """
    anchor_xy = kps_xy[ANCHOR_KPS].mean(axis=0)
    centered  = kps_xy - anchor_xy
    scale     = float(np.max(np.abs(centered)) + 1e-5)
    anchor_y  = float(anchor_xy[1])
    return scale, anchor_y


# ── Зеркальное отражение ──────────────────────────────────────────────────────

def mirror_vectors(vec: torch.Tensor) -> torch.Tensor:
    """
    Зеркальное отражение батча нормированных векторов поз по оси X.

    Parameters
    ----------
    vec : torch.Tensor (N, 34)

    Returns
    -------
    torch.Tensor (N, 34)
    """
    mirrored = vec.clone()
    mirrored[:, 0::2] = -mirrored[:, 0::2]

    for l_idx, r_idx in MIRROR_PAIRS:
        lx, ly = l_idx * 2,     l_idx * 2 + 1
        rx, ry = r_idx * 2,     r_idx * 2 + 1
        tmp_x = mirrored[:, lx].clone()
        tmp_y = mirrored[:, ly].clone()
        mirrored[:, lx] = mirrored[:, rx]
        mirrored[:, ly] = mirrored[:, ry]
        mirrored[:, rx] = tmp_x
        mirrored[:, ry] = tmp_y

    return mirrored


# ── Сборка тензора ────────────────────────────────────────────────────────────

def build_poses_tensor(
    frames_data:      list[dict],
    use_body_weights: bool = True,
) -> tuple[torch.Tensor | None, list[dict]]:
    """
    Построить тензор поз и метаданные из списка frame-записей.

    Каждая запись frames_data[i]:
      'poses'     : list[dict] — позы на кадре
      't'         : float      — время кадра (сек)
      'f'         : int        — номер кадра
      'video_idx' : int        — индекс видео
      'dir'       : str        — направление
      'kp'        : list|None  — keypoints [[x,y,conf]×17] для классификатора

    Returns
    -------
    tensor : torch.Tensor (M, 34) или None если поз нет
    meta   : list[dict] длиной M
      Ключи: t, f, video_idx, dir, scale, anchor_y, kp
    """
    vectors: list[np.ndarray] = []
    meta:    list[dict]       = []

    for frame in frames_data:
        t         = float(frame.get("t",         0.0))
        f         = int(  frame.get("f",         0))
        video_idx = int(  frame.get("video_idx", 0))
        direction =       frame.get("dir",       "forward")

        # ── kp из frames_data — для классификатора движений ───────────────
        # Приоритет: поле "kp" из frame (уже list),
        # fallback: берём из первой валидной позы
        frame_kp: list | None = frame.get("kp")

        for pose in frame.get("poses", []):
            if not is_pose_valid(pose):
                continue

            vec             = preprocess_pose(pose, use_body_weights)
            kps_xy          = pose["keypoints"][:COCO_N_KPS, :2].astype(np.float32)
            scale, anchor_y = compute_pose_features(kps_xy)

            # Если kp не был передан в frame — берём из pose
            kp_for_meta = frame_kp
            if kp_for_meta is None:
                kp_raw = pose.get("kp") or pose.get("keypoints")
                if isinstance(kp_raw, np.ndarray):
                    kp_for_meta = kp_raw.tolist()
                elif isinstance(kp_raw, list):
                    kp_for_meta = kp_raw

            vectors.append(vec)
            meta.append({
                "t":         t,
                "f":         f,
                "video_idx": video_idx,
                "dir":       direction,
                "scale":     scale,
                "anchor_y":  anchor_y,
                # ── Keypoints для MotionClassifier ────────────────────────
                # Формат: list[[x, y, conf], ...] × 17
                # None если keypoints недоступны
                "kp":        kp_for_meta,
            })

    if not vectors:
        return None, []

    tensor = torch.from_numpy(np.stack(vectors, axis=0))
    return tensor, meta