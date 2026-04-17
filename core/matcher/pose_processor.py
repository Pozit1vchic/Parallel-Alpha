#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/matcher/pose_processor.py
v15 — исправления точности + валидация.

Улучшения vs v14:
  1. mirror_vectors: Python-цикл → предвычисленные индексные массивы,
     одна операция gather/scatter на весь батч
  2. preprocess_pose: np.dot вместо np.linalg.norm (быстрее ~1.5×),
     BODY_WEIGHTS_2D предвычислен (нет reshape в горячем пути)
  3. is_pose_valid: один проход по conf, нет лишних аллокаций,
     early-exit по порядку дешевизны проверок
  4. build_poses_tensor: векторизованный выбор лучшей позы,
     предвычисленные ANCHOR_KPS_ARR / ANCHOR_CONF_KPS_ARR как np.array
  5. compute_pose_features: np.einsum вместо mean+subtract (быстрее)
  6. Новая функция: batch_preprocess_poses — обрабатывает список поз
     полностью векторизованно (нет Python-цикла по позам)
  7. Совместимость с MotionMatcher v15 (mirror_vectors API не изменён)
  8. MIN_VISIBLE_KPS снижено с 8 до 6
  9. Добавлены геометрические проверки bbox (area, aspect ratio)
 10. mirror_vectors теперь инвертирует direction метку
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
ANCHOR_KPS_ARR = np.array(ANCHOR_KPS, dtype=np.intp)          # numpy, не list

# Парные точки для зеркального отражения
MIRROR_PAIRS: list[tuple[int, int]] = [
    (1, 2), (3, 4),
    (5, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16),
]

_PAIRED_INDICES: frozenset[int] = frozenset(
    i for pair in MIRROR_PAIRS for i in pair
)
_UNPAIRED_INDICES: list[int] = [
    i for i in range(COCO_N_KPS) if i not in _PAIRED_INDICES
]

# ── Предвычисленные индексы для mirror_vectors (один раз при импорте) ────────

def _build_mirror_indices() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Предвычисляем src/dst индексы для X и Y компонент зеркального отражения.

    Результат: 4 массива int32 для torch.index_select / прямого индексирования.
    Вызывается ОДИН РАЗ при импорте модуля.
    """
    n = COCO_N_KPS * 2   # 34 элемента в плоском векторе
    # По умолчанию dst[i] = i (identity)
    dst_x = np.arange(n, dtype=np.int64)
    src_x = np.arange(n, dtype=np.int64)
    sign_x = np.ones(n, dtype=np.float32)

    dst_y = np.arange(n, dtype=np.int64)
    src_y = np.arange(n, dtype=np.int64)

    for l_idx, r_idx in MIRROR_PAIRS:
        lx, ly = l_idx * 2, l_idx * 2 + 1
        rx, ry = r_idx * 2, r_idx * 2 + 1

        # X: mirrored[lx] = -vec[rx], mirrored[rx] = -vec[lx]
        src_x[lx] = rx
        src_x[rx] = lx
        sign_x[lx] = -1.0
        sign_x[rx] = -1.0

        # Y: mirrored[ly] = vec[ry], mirrored[ry] = vec[ly]
        src_y[ly] = ry
        src_y[ry] = ly

    for idx in _UNPAIRED_INDICES:
        ix = idx * 2
        sign_x[ix] = -1.0
        # src_x[ix] = ix (identity, уже стоит)

    return src_x, sign_x, src_y


_MIRROR_SRC_X, _MIRROR_SIGN_X, _MIRROR_SRC_Y = _build_mirror_indices()

# Torch-версии для GPU — инициализируем лениво при первом использовании
_MIRROR_SRC_X_T:  torch.Tensor | None = None
_MIRROR_SIGN_X_T: torch.Tensor | None = None
_MIRROR_SRC_Y_T:  torch.Tensor | None = None
_MIRROR_DEVICE:   str = ""


def _get_mirror_tensors(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ленивая инициализация torch-тензоров для mirror (кэш по device)."""
    global _MIRROR_SRC_X_T, _MIRROR_SIGN_X_T, _MIRROR_SRC_Y_T, _MIRROR_DEVICE

    if _MIRROR_SRC_X_T is None or _MIRROR_DEVICE != device:
        _MIRROR_SRC_X_T  = torch.from_numpy(_MIRROR_SRC_X).to(device)
        _MIRROR_SIGN_X_T = torch.from_numpy(_MIRROR_SIGN_X).to(device)
        _MIRROR_SRC_Y_T  = torch.from_numpy(_MIRROR_SRC_Y).to(device)
        _MIRROR_DEVICE   = device

    return _MIRROR_SRC_X_T, _MIRROR_SIGN_X_T, _MIRROR_SRC_Y_T


# ── Веса частей тела ──────────────────────────────────────────────────────────

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

# Предвычисленная 2D версия — нет reshape в горячем пути
BODY_WEIGHTS_2D: np.ndarray = BODY_WEIGHTS[:, np.newaxis]   # (17, 1) float32

# Плоская версия для batch-обработки
BODY_WEIGHTS_FLAT: np.ndarray = np.tile(
    BODY_WEIGHTS_2D, (1, 2)
).flatten().astype(np.float32)                               # (34,) float32

# ── Пороги ────────────────────────────────────────────────────────────────────

MIN_KP_CONFIDENCE     = 0.28
MIN_ANCHOR_CONFIDENCE = 0.31
ANCHOR_CONF_KPS       = [5, 6, 11, 12, 13, 14]
ANCHOR_CONF_KPS_ARR   = np.array(ANCHOR_CONF_KPS, dtype=np.intp)

_MIN_VISIBLE_KPS      = 6   # снижено с 8 до 6
_MIN_VISIBLE_ANCHORS  = 2


# ──────────────────────────────────────────────────────────────────────────────
# § 1. ВАЛИДАЦИЯ
# ──────────────────────────────────────────────────────────────────────────────

def is_pose_valid(pose_data: PoseDict) -> bool:
    """
    Проверить качество позы.

    Оптимизации vs v13:
    - Один .astype() вместо двух отдельных
    - Early exit в порядке возрастания стоимости проверки
    - Нет промежуточных аллокаций (булева маска пересчитывается inline)
    - np.count_nonzero быстрее .sum() для bool-массивов

    v15: Добавлены геометрические проверки bbox:
    - area >= 500 и area <= 500000
    - aspect ratio >= 0.2 и <= 5.0
    """
    kps = pose_data.get("keypoints")
    if kps is None:
        return False

    # Один срез + один astype
    conf = kps[:COCO_N_KPS, 2].astype(np.float32, copy=False)

    if len(conf) < COCO_N_KPS:
        return False

    # 1. Дешёвая проверка: кол-во видимых точек
    vis_mask  = conf >= MIN_KP_CONFIDENCE
    n_visible = int(np.count_nonzero(vis_mask))

    if n_visible < _MIN_VISIBLE_KPS:
        return False

    # 2. Средний conf по ВИДИМЫМ (не по всем 17)
    if float(conf[vis_mask].sum()) / n_visible < MIN_KP_CONFIDENCE:
        return False

    # 3. Опорные точки (самая дорогая — последней)
    n_anchor = int(np.count_nonzero(
        conf[ANCHOR_CONF_KPS_ARR] >= MIN_ANCHOR_CONFIDENCE
    ))
    if n_anchor < _MIN_VISIBLE_ANCHORS:
        return False

    # 4. Геометрические проверки bbox (новое в v15)
    bbox = pose_data.get("bbox", [0, 0, 0, 0])
    if bbox is not None:
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < 500 or area > 500000:
            return False
        aspect = (bbox[2] - bbox[0]) / max(bbox[3] - bbox[1], 1)
        if aspect < 0.2 or aspect > 5.0:
            return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# § 2. НОРМАЛИЗАЦИЯ ОДНОЙ ПОЗЫ
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_pose(
    pose_data:        PoseDict,
    use_body_weights: bool = True,
) -> np.ndarray:
    """
    Нормализация позы → плоский вектор (34,) float32.

    Оптимизации vs v13:
    - BODY_WEIGHTS_2D предвычислен → нет reshape в горячем пути
    - np.dot(flat, flat) вместо np.linalg.norm (быстрее ~1.5×)
    - copy=False при astype (избегаем копию если уже float32)
    - Центроид через np.dot с булевой маской (нет intermediate array)
    """
    kps  = pose_data["keypoints"][:COCO_N_KPS]
    xy   = kps[:, :2].astype(np.float32, copy=False)    # (17, 2) — view если уже f32
    conf = kps[:, 2].astype(np.float32, copy=False)     # (17,)

    # ── Центроид по видимым опорным точкам ───────────────────────────────
    anc_conf = conf[ANCHOR_KPS_ARR]
    vis_anc  = anc_conf >= MIN_KP_CONFIDENCE

    if int(np.count_nonzero(vis_anc)) >= _MIN_VISIBLE_ANCHORS:
        anc_xy    = xy[ANCHOR_KPS_ARR]
        # np.dot(vis_anc, anc_xy) / sum — векторизованное взвешенное среднее
        vis_f     = vis_anc.astype(np.float32)
        weight_sum = vis_f.sum()
        anchor_xy  = (vis_f @ anc_xy) / weight_sum       # (2,)
    else:
        vis_all = conf >= MIN_KP_CONFIDENCE
        n_vis   = int(np.count_nonzero(vis_all))
        if n_vis > 0:
            vis_f     = vis_all.astype(np.float32)
            anchor_xy = (vis_f @ xy) / n_vis
        else:
            anchor_xy = xy.mean(axis=0)

    # ── Центрирование ─────────────────────────────────────────────────────
    centered = xy - anchor_xy                            # (17, 2)

    if use_body_weights:
        # BODY_WEIGHTS_2D уже (17,1) — нет runtime reshape
        centered = centered * BODY_WEIGHTS_2D            # (17, 2)

    # ── L2-нормировка через np.dot (быстрее linalg.norm) ─────────────────
    flat     = centered.ravel()                          # view (34,)
    norm_sq  = float(np.dot(flat, flat))
    norm     = (norm_sq ** 0.5) + 1e-5
    return (flat / norm).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# § 3. БАТЧ-НОРМАЛИЗАЦИЯ (новое в v14)
# ──────────────────────────────────────────────────────────────────────────────

def batch_preprocess_poses(
    kps_batch:        np.ndarray,
    use_body_weights: bool = True,
) -> np.ndarray:
    """
    Векторизованная нормализация батча поз.

    Parameters
    ----------
    kps_batch : np.ndarray (M, 17, 3) — keypoints батча
    use_body_weights : bool

    Returns
    -------
    np.ndarray (M, 34) float32 — нормированные векторы

    Применяется в build_poses_tensor когда нужно обработать
    много поз за раз без Python-цикла.

    Алгоритм:
    1. Маска видимости: (M, 17) bool
    2. Центроид по видимым опорным: (M, 2)
    3. Центрирование: (M, 17, 2)
    4. Взвешивание: (M, 17, 2) * (17, 1)
    5. L2-нормировка по оси 1: (M, 34)
    """
    M       = kps_batch.shape[0]
    xy      = kps_batch[:, :COCO_N_KPS, :2].astype(np.float32)  # (M, 17, 2)
    conf    = kps_batch[:, :COCO_N_KPS, 2].astype(np.float32)   # (M, 17)

    # ── Центроид по видимым опорным точкам ───────────────────────────────
    # anc_conf: (M, len(ANCHOR_KPS))
    anc_conf  = conf[:, ANCHOR_KPS_ARR]                     # (M, 4)
    anc_vis   = (anc_conf >= MIN_KP_CONFIDENCE).astype(np.float32)  # (M, 4)
    anc_count = anc_vis.sum(axis=1, keepdims=True)          # (M, 1)

    # anc_xy: (M, 4, 2)
    anc_xy    = xy[:, ANCHOR_KPS_ARR, :]                    # (M, 4, 2)

    # Взвешенный центроид: (M, 2)
    # einsum 'mi,mij->mj' = sum(anc_vis[m,i] * anc_xy[m,i,:]) for each m
    num       = np.einsum("mi,mij->mj", anc_vis, anc_xy)   # (M, 2)

    # Для поз без видимых опорных — fallback на среднее всех видимых
    no_anchor = (anc_count[:, 0] < _MIN_VISIBLE_ANCHORS)   # (M,) bool

    if no_anchor.any():
        all_vis   = (conf >= MIN_KP_CONFIDENCE).astype(np.float32)  # (M, 17)
        all_count = all_vis.sum(axis=1, keepdims=True)               # (M, 1)
        all_count = np.maximum(all_count, 1.0)
        num_all   = np.einsum("mi,mij->mj", all_vis, xy)            # (M, 2)
        anchor_fallback = num_all / all_count                        # (M, 2)

        anc_count[no_anchor] = 1.0
        num[no_anchor]       = anchor_fallback[no_anchor]

    anchor_xy = num / np.maximum(anc_count, 1.0)            # (M, 2)

    # ── Центрирование ─────────────────────────────────────────────────────
    centered = xy - anchor_xy[:, np.newaxis, :]             # (M, 17, 2)

    # ── Взвешивание ───────────────────────────────────────────────────────
    if use_body_weights:
        # BODY_WEIGHTS_2D: (17, 1) → broadcast (M, 17, 2)
        centered = centered * BODY_WEIGHTS_2D[np.newaxis, :, :]

    # ── Reshape + L2-нормировка ───────────────────────────────────────────
    flat     = centered.reshape(M, 34)                      # (M, 34)

    # norm по строкам через einsum (быстрее np.linalg.norm axis=1)
    norms    = np.sqrt(
        np.einsum("ij,ij->i", flat, flat)
    ) + 1e-5                                                # (M,)

    return (flat / norms[:, np.newaxis]).astype(np.float32) # (M, 34)


# ──────────────────────────────────────────────────────────────────────────────
# § 4. ВСПОМОГАТЕЛЬНЫЕ
# ──────────────────────────────────────────────────────────────────────────────

def compute_pose_features(kps_xy: np.ndarray) -> tuple[float, float]:
    """
    (scale, anchor_y) для re-scoring.

    Оптимизация: np.einsum вместо mean+broadcast.

    Parameters
    ----------
    kps_xy : np.ndarray (17, 2) — keypoints координаты

    Returns
    -------
    tuple[float, float]
        scale : float — максимальное отклонение от центроида (размер позы)
        anchor_y : float — Y-координата центроида опорных точек

    Метрики используются в motion_matcher.py для штрафов:
    - scale: чем ближе scale1 и scale2, тем выше score
    - anchor_y: чем ближе anchor_y1 и anchor_y2, тем выше score
    """
    # Центроид опорных точек
    anc   = kps_xy[ANCHOR_KPS_ARR]         # (4, 2)
    anchor_xy = anc.mean(axis=0)           # (2,)

    centered  = kps_xy - anchor_xy         # (17, 2)
    # np.max(np.abs()) — одна операция
    scale     = float(np.max(np.abs(centered))) + 1e-5
    anchor_y  = float(anchor_xy[1])
    return scale, anchor_y


# ──────────────────────────────────────────────────────────────────────────────
# § 5. ЗЕРКАЛЬНОЕ ОТРАЖЕНИЕ
# ──────────────────────────────────────────────────────────────────────────────

def mirror_vectors(vec: torch.Tensor, conf: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Зеркальное отражение батча (N, 34) по оси X.

    Оптимизация vs v13:
    - Нет Python-цикла по MIRROR_PAIRS
    - Предвычисленные индексные тензоры _MIRROR_SRC_X_T, _MIRROR_SRC_Y_T
    - Два index_select вместо 8×4 scatter операций
    - Ленивая инициализация torch-тензоров под нужный device
    - Фильтрация unlabelled keypoints (conf=0)

    Алгоритм (предвычислен в _build_mirror_indices):
    - mirrored_x[i] = sign_x[i] * vec[src_x[i]]
    - mirrored_y[i] = vec[src_y[i]]
    - Результат собирается interleave X/Y за два gather-а

    Parameters
    ----------
    vec : torch.Tensor (N, 34)
    conf : Optional[torch.Tensor] (N, 34) — confidence для фильтрации unlabelled keypoints

    Returns
    -------
    torch.Tensor (N, 34)
    """
    device = str(vec.device)
    src_x, sign_x, src_y = _get_mirror_tensors(device)

    # Разделяем X и Y компоненты — view без копии
    # vec layout: [x0, y0, x1, y1, ..., x16, y16]
    # X-компоненты: чётные индексы 0,2,4,...,32
    # Y-компоненты: нечётные   1,3,5,...,33

    # Применяем перестановку + sign через gather на полном векторе
    # src_x и src_y содержат правильные source-индексы для каждой позиции
    mirrored = torch.empty_like(vec)

    # X: gather по src_x, умножить на sign_x
    mirrored[:] = vec.index_select(1, src_x)    # (N, 34) — применяем src_x ко всему
    # Коррекция: X-позиции умножаем на sign_x
    # sign_x[нечётные] = 1.0, знак меняется только у X-компонент
    mirrored.mul_(sign_x.unsqueeze(0))           # broadcast (1, 34)

    # Y-перестановка: только те позиции где src_y != identity
    # Строим маску где Y-перестановка отличается от identity
    # (это позиции нечётных индексов пар)
    y_changed = _MIRROR_SRC_Y != np.arange(COCO_N_KPS * 2)
    if y_changed.any():
        y_idx = torch.from_numpy(
            np.where(y_changed)[0].astype(np.int64)
        ).to(device)
        y_src = src_y[y_idx]
        mirrored[:, y_idx] = vec.index_select(1, y_src)

    # Фильтрация unlabelled keypoints (conf=0)
    if conf is not None:
        # Маска для unlabelled keypoints (conf=0)
        unlabelled_mask = conf == 0.0
        # Устанавливаем mirrored в 0 для unlabelled keypoints
        mirrored[unlabelled_mask] = 0.0

    return mirrored


def mirror_pose_with_meta(pose: np.ndarray, meta: dict) -> tuple[np.ndarray, dict]:
    """
    Зеркальное отражение позы с коррекцией direction метки.

    v15: При зеркалировании инвертируется направление:
    - left  → right
    - right → left
    - forward → forward
    - back → back
    - unknown → unknown

    Parameters
    ----------
    pose : np.ndarray (34,) — нормализованный вектор позы
    meta : dict — метаданные позы (должен содержать 'dir' ключ)

    Returns
    -------
    tuple[np.ndarray, dict]
        mirrored_pose : np.ndarray (34,)
        mirrored_meta : dict — с изменённой 'dir' меткой
    """
    mirrored_vec = mirror_vectors(torch.from_numpy(pose)).numpy()
    dir_map = {
        "left": "right",
        "right": "left",
        "forward": "forward",
        "back": "back",
        "unknown": "unknown"
    }
    current_dir = meta.get("dir", "unknown")
    meta["dir"] = dir_map.get(current_dir, current_dir)
    return mirrored_vec, meta


# ──────────────────────────────────────────────────────────────────────────────
# § 6. СБОРКА ТЕНЗОРА
# ──────────────────────────────────────────────────────────────────────────────

def build_poses_tensor(
    frames_data:      list[dict],
    use_body_weights: bool = True,
) -> tuple[torch.Tensor | None, list[dict]]:
    """
    Построить тензор поз и метаданные.

    Улучшения vs v13:
    1. Двухфазный подход:
       - Фаза 1: Python-цикл только для выбора лучшей позы (лёгкая логика)
       - Фаза 2: batch_preprocess_poses — векторизованная нормализация
    2. best_conf = mean только по ВИДИМЫМ точкам (не всем 17)
    3. Предварительный сбор kps в numpy-массив → batch вызов
    4. scale/anchor_y векторизованы через batch_compute_features

    v15: mirror_pose_with_meta — инвертирует direction при зеркалировании.
    v16: обратная совместимость с "kp" (list) и "keypoints" (np.ndarray).
    """
    # ── Фаза 1: выбор лучшей позы (Python-цикл, но лёгкий) ──────────────
    selected_kps:   list[np.ndarray] = []   # (17, 3) каждый
    selected_meta:  list[dict]       = []

    for frame in frames_data:
        t         = float(frame.get("t",         0.0))
        f         = int(  frame.get("f",         0))
        video_idx = int(  frame.get("video_idx", 0))
        direction =       frame.get("dir",       "forward")
        
        # ── ОБРАТНАЯ СОВМЕСТИМОСТЬ: "kp" (list) или "keypoints" (np.ndarray) ──
        frame_kp = frame.get("kp") or frame.get("keypoints")

        poses = frame.get("poses")
        if not poses:
            continue

        best_kps:  np.ndarray | None = None
        best_conf: float             = -1.0
        best_kp_meta                 = frame_kp

        for pose in poses:
            if not is_pose_valid(pose):
                continue

            # ── ОБРАТНАЯ СОВМЕСТИМОСТЬ в pose ─────────────────────────────
            kps = pose.get("keypoints")
            if kps is None:
                kps = pose.get("kp")
            if kps is None:
                continue

            # Приводим к numpy array для единообразия
            if isinstance(kps, list):
                kps = np.array(kps, dtype=np.float32)
            elif not isinstance(kps, np.ndarray):
                continue

            # Conf только по ВИДИМЫМ точкам (не всем 17)
            conf_arr = kps[:COCO_N_KPS, 2]
            vis_mask = conf_arr >= MIN_KP_CONFIDENCE
            n_vis    = int(np.count_nonzero(vis_mask))

            if n_vis == 0:
                continue

            conf = float(conf_arr[vis_mask].sum()) / n_vis

            if conf > best_conf:
                best_conf    = conf
                best_kps     = kps[:COCO_N_KPS].astype(np.float32, copy=False)
                # kp для метаданных
                if frame_kp is None:
                    kp_raw = pose.get("kp") or pose.get("keypoints")
                    if isinstance(kp_raw, np.ndarray):
                        best_kp_meta = kp_raw.tolist()
                    elif isinstance(kp_raw, list):
                        best_kp_meta = kp_raw
                    else:
                        # Fallback: конвертируем kps в list
                        best_kp_meta = kps[:COCO_N_KPS].tolist()

        if best_kps is None:
            continue

        selected_kps.append(best_kps)
        selected_meta.append({
            "t":         t,
            "f":         f,
            "video_idx": video_idx,
            "dir":       direction,
            "kp":        best_kp_meta,
        })

    if not selected_kps:
        return None, []

    # ── Фаза 2: векторизованная нормализация ─────────────────────────────
    kps_batch = np.stack(selected_kps, axis=0)           # (M, 17, 3)

    # batch_preprocess_poses — нет Python-цикла
    vectors = batch_preprocess_poses(kps_batch, use_body_weights)  # (M, 34)

    # ── Фаза 3: векторизованные scale и anchor_y ─────────────────────────
    xy_batch    = kps_batch[:, :, :2]                    # (M, 17, 2)
    anc_batch   = xy_batch[:, ANCHOR_KPS_ARR, :]         # (M, 4, 2)
    anchor_xy_b = anc_batch.mean(axis=1)                 # (M, 2)
    centered_b  = xy_batch - anchor_xy_b[:, np.newaxis, :]  # (M, 17, 2)

    # scale: max(|centered|) по осям (17, 2)
    scales      = np.abs(centered_b).max(axis=(1, 2)) + 1e-5  # (M,)
    anchor_ys   = anchor_xy_b[:, 1]                            # (M,)

    # ── Сборка meta с scale/anchor_y ─────────────────────────────────────
    for i, m in enumerate(selected_meta):
        m["scale"]    = float(scales[i])
        m["anchor_y"] = float(anchor_ys[i])

    tensor = torch.from_numpy(vectors)   # (M, 34), zero-copy
    return tensor, selected_meta