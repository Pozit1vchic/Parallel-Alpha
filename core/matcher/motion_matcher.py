#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import bisect
import gc
import logging
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from usearch.index import Index
    USEARCH_AVAILABLE = True
except ImportError:
    USEARCH_AVAILABLE = False

from core.matcher.pose_processor import (
    mirror_vectors,
    COCO_N_KPS,
    ANCHOR_KPS_ARR,
    compare_body_proportions,
)

log = logging.getLogger(__name__)

# ── Веса метрик ───────────────────────────────────────────────────────────────
WEIGHT_COSINE     = 0.45
WEIGHT_MOTION     = 0.30
WEIGHT_APPEARANCE = 0.25

# ── Параметры USearch ─────────────────────────────────────────────────────────
K_FAISS = 300

# ── Пороги ────────────────────────────────────────────────────────────────────
DEFAULT_SIM_THRESHOLD = 0.68
DEFAULT_GOOD_THRESH   = 0.75

# ── Дедупликация ──────────────────────────────────────────────────────────────
DEFAULT_MIN_MATCH_GAP      = 1.0
SAME_VIDEO_MIN_GAP         = 4.0
CROSS_VIDEO_MIN_GAP        = 1.0
DUPLICATE_THRESHOLD        = 3.0
SCENE_CHANGE_ANCHOR_Y_DIFF = 0.15
DEFAULT_JUNK_RATIO         = 0.10
DEFAULT_MAX_UNIQUE         = 5000

# ── Структура кандидатов ──────────────────────────────────────────────────────
_MATCH_DTYPE = np.dtype([
    ("m1_idx",          np.int32),
    ("m2_idx",          np.int32),
    ("cosine_sim",      np.float32),
    ("sim",             np.float32),
    ("dtw_sim",         np.float32),
    ("motion_sim",      np.float32),
    ("appearance_sim",  np.float32),
    ("t1",              np.float32),
    ("t2",              np.float32),
    ("f1",              np.int32),
    ("f2",              np.int32),
    ("v1_idx",          np.int32),
    ("v2_idx",          np.int32),
    ("final_sim",       np.float32),
    ("scale_penalty",   np.float32),
    ("anchor_penalty",  np.float32),
])

# ── Числовые коды направлений (для векторных сравнений) ───────────────────────
_DIR_CODE = {
    "unknown": 0,
    "forward": 1,
    "back":    2,
    "left":    3,
    "right":   4,
}
_DIR_DEFAULT = 0

# Группы для проверок:
#   * forward/back несовместимо с left/right
#   * left vs right несовместимо
_DIR_FB = (1, 2)   # forward, back
_DIR_LR = (3, 4)   # left, right


def _build_meta_arrays(meta: list[dict]) -> dict[str, np.ndarray]:
    """
    Векторная сборка numpy-массивов из списка словарей метаданных.
    Числовые поля строятся одним проходом, направления кодируются
    в int8 (для последующих векторных операций).

    Body-proportions сохраняются как list (нельзя в массив) — но
    индексируются по позиции; их формат теперь читается через
    готовый список (см. _build_matches_from_candidates).
    """
    n = len(meta)
    fps_aprx = 30.0

    times      = np.empty(n, dtype=np.float32)
    frames     = np.empty(n, dtype=np.int32)
    video_idx  = np.zeros(n, dtype=np.int32)
    direction  = np.empty(n, dtype=object)
    dir_code   = np.empty(n, dtype=np.int8)
    scale      = np.ones(n,  dtype=np.float32)
    anchor_y   = np.zeros(n, dtype=np.float32)

    for i, m in enumerate(meta):
        t = float(m.get("t", 0.0))
        times[i]     = t
        frames[i]    = int(m.get("f", t * fps_aprx))
        video_idx[i] = int(m.get("video_idx", 0))
        d            = m.get("dir", "unknown") or "unknown"
        direction[i] = d
        dir_code[i]  = _DIR_CODE.get(d, _DIR_DEFAULT)
        scale[i]     = float(m.get("scale", 1.0))
        anchor_y[i]  = float(m.get("anchor_y", 0.5))

    return {
        "times":     times,
        "frames":    frames,
        "video_idx": video_idx,
        "direction": direction,
        "dir_code":  dir_code,
        "scale":     scale,
        "anchor_y":  anchor_y,
    }


class MotionMatcher:
    DEFAULT_CHUNK_SIZE     = 3000
    DEFAULT_CHUNK_OVERLAP  = 300
    DEFAULT_MAX_PER_CHUNK  = 2_000_000
    DEFAULT_MAX_TOTAL      = 30_000_000

    def __init__(self, device: str = "cuda") -> None:
        self.device = device if torch.cuda.is_available() else "cpu"

        self.k_faiss = K_FAISS

        self.chunk_size     = self.DEFAULT_CHUNK_SIZE
        self.chunk_overlap  = self.DEFAULT_CHUNK_OVERLAP
        self.max_per_chunk  = self.DEFAULT_MAX_PER_CHUNK
        self.max_total      = self.DEFAULT_MAX_TOTAL
        self.max_unique     = 2000
        self.min_match_gap  = DEFAULT_MIN_MATCH_GAP
        self.junk_ratio     = DEFAULT_JUNK_RATIO
        self.good_threshold = DEFAULT_GOOD_THRESH
        self.sim_threshold  = DEFAULT_SIM_THRESHOLD

        self._is_cuda: bool = (self.device == "cuda")
        self._usearch_index: Optional[Index] = None

        self._poses_meta_cached:   list[dict]            = []
        self._poses_tensor_cached: Optional[torch.Tensor] = None

    # ==================================================================
    # Public API
    # ==================================================================

    def find_matches(
        self,
        poses_tensor: torch.Tensor,
        poses_meta:   list[dict],
        threshold:    float = DEFAULT_SIM_THRESHOLD,
        min_gap:      float = DEFAULT_MIN_MATCH_GAP,
        use_mirror:   bool  = False,
    ) -> list[dict]:
        if poses_tensor is None or len(poses_tensor) < 10:
            return []

        if len(poses_meta) != len(poses_tensor):
            raise ValueError(
                f"poses_tensor({len(poses_tensor)}) != poses_meta({len(poses_meta)})"
            )

        n = len(poses_tensor)
        log.info(
            "[Matcher] N=%d | thr=%.2f | gap=%.1fs | mirror=%s | k_faiss=%d",
            n, threshold, min_gap, use_mirror, self.k_faiss,
        )

        V = poses_tensor.to(dtype=torch.float32, device=self.device).view(n, -1)
        V = F.normalize(V, p=2, dim=1)

        V_mirror: Optional[torch.Tensor] = None
        if use_mirror:
            V_mirror = F.normalize(mirror_vectors(V), p=2, dim=1)

        meta_arrs = _build_meta_arrays(poses_meta)

        t_start = time.time()
        candidates = self._find_candidates_usearch(
            V, V_mirror,
            meta_arrs["times"], meta_arrs["frames"], meta_arrs["video_idx"],
            meta_arrs["direction"], meta_arrs["dir_code"],
            meta_arrs["scale"], meta_arrs["anchor_y"],
            threshold, min_gap,
        )
        log.info("[Matcher] Поиск кандидатов: %.2fс", time.time() - t_start)

        if candidates is None or len(candidates) == 0:
            log.info("[Matcher] Совпадений не найдено (USearch).")
            return []

        log.info("[Matcher] Найдено кандидатов через USearch: %d", len(candidates))

        final_matches = self._build_matches_from_candidates(
            candidates, poses_meta, meta_arrs,
        )

        if len(final_matches) > 0:
            final_matches = self._dedup_pairs_torch(final_matches, n)

        log.info("[Matcher] После дедупликации пар: %d", len(final_matches))

        final_matches = self._remove_duplicates_strict(final_matches)
        log.info("[Matcher] После strict dedup: %d", len(final_matches))

        gc.collect()
        if self._is_cuda:
            torch.cuda.empty_cache()

        if not final_matches:
            log.info("[Matcher] Финальных совпадений не найдено.")
            return []

        t_dedup = time.time()
        deduplicated = self._deduplicate(final_matches, poses_meta, min_gap)
        log.info(
            "[Matcher] Финальная дедупликация: %.2fс → %d совпадений",
            time.time() - t_dedup, len(deduplicated),
        )

        return deduplicated

    # ==================================================================
    # USearch search (полностью векторизованная пост-фильтрация)
    # ==================================================================

    def _find_candidates_usearch(
        self,
        V:            torch.Tensor,
        V_mirror:     Optional[torch.Tensor],
        times_np:     np.ndarray,
        frames_np:    np.ndarray,
        vididx_np:    np.ndarray,
        direction_np: np.ndarray,
        dir_code_np:  np.ndarray,
        scale_np:     np.ndarray,
        anchor_y_np:  np.ndarray,
        threshold:    float,
        min_gap:      float,
    ) -> Optional[np.ndarray]:
        if not USEARCH_AVAILABLE:
            log.warning("[Matcher] USearch недоступен, используем fallback.")
            return self._find_candidates_fallback(
                V, V_mirror, times_np, frames_np, vididx_np,
                direction_np, dir_code_np, scale_np, anchor_y_np,
                threshold, min_gap,
            )

        n = V.shape[0]
        dim = V.shape[1]

        self._usearch_index = Index(ndim=dim, metric="cos", dtype="float32")
        V_cpu = V.cpu().numpy()
        keys = np.arange(n, dtype=np.int64)
        self._usearch_index.add(keys, V_cpu)

        matches = self._usearch_index.search(V_cpu, count=self.k_faiss)

        if matches.keys is None or matches.distances is None:
            return None

        keys_arr  = matches.keys
        dists_arr = matches.distances

        n_neighbors = self.k_faiss

        row_indices = np.repeat(np.arange(n, dtype=np.int32), n_neighbors)
        col_indices = keys_arr.ravel().astype(np.int32, copy=False)
        distances   = dists_arr.ravel()
        similarities = (1.0 - distances).astype(np.float32, copy=False)

        # ── Базовый фильтр: только верх. треугольник + threshold ──────
        valid_mask = col_indices > row_indices
        valid_mask &= (similarities >= threshold)

        # Можно сразу сократить массивы — все последующие операции
        # будут идти по уменьшенному набору, что критично для скорости.
        valid_mask = np.where(valid_mask)[0]
        if valid_mask.size == 0:
            return None

        rows_v = row_indices[valid_mask]
        cols_v = col_indices[valid_mask]
        sims_v = similarities[valid_mask]

        t1_v = times_np[rows_v]
        t2_v = times_np[cols_v]
        v1_v = vididx_np[rows_v]
        v2_v = vididx_np[cols_v]

        # ── Time-gap для одного видео ─────────────────────────────────
        same_video = (v1_v == v2_v)
        keep = ~same_video | (np.abs(t1_v - t2_v) >= min_gap)

        # ── Векторная проверка направлений ────────────────────────────
        # Несовместимы:
        #   d1∈FB ∧ d2∈LR        (или симметрично)
        #   d1∈LR ∧ d2∈LR ∧ d1≠d2 (left vs right)
        d1 = dir_code_np[rows_v]
        d2 = dir_code_np[cols_v]

        d1_fb = (d1 == 1) | (d1 == 2)
        d2_fb = (d2 == 1) | (d2 == 2)
        d1_lr = (d1 == 3) | (d1 == 4)
        d2_lr = (d2 == 3) | (d2 == 4)

        both_known = (d1 != 0) & (d2 != 0)
        bad_fb_lr  = both_known & ((d1_fb & d2_lr) | (d2_fb & d1_lr))
        bad_lr_lr  = both_known & d1_lr & d2_lr & (d1 != d2)

        keep &= ~(bad_fb_lr | bad_lr_lr)

        # ── Scene-change: anchor_y diff ───────────────────────────────
        keep &= (np.abs(anchor_y_np[rows_v] - anchor_y_np[cols_v])
                 < SCENE_CHANGE_ANCHOR_Y_DIFF)

        if not keep.any():
            return None

        rows_v = rows_v[keep]
        cols_v = cols_v[keep]
        sims_v = sims_v[keep]
        t1_v   = t1_v[keep]
        t2_v   = t2_v[keep]

        # ── Top-K cap ─────────────────────────────────────────────────
        max_candidates = min(self.max_per_chunk, len(rows_v))
        if len(rows_v) > max_candidates:
            top_k_idx = np.argpartition(-sims_v, max_candidates)[:max_candidates]
            rows_v = rows_v[top_k_idx]
            cols_v = cols_v[top_k_idx]
            sims_v = sims_v[top_k_idx]
            t1_v   = t1_v[top_k_idx]
            t2_v   = t2_v[top_k_idx]

        k = len(rows_v)
        arr = np.empty(k, dtype=_MATCH_DTYPE)
        arr["m1_idx"]         = rows_v
        arr["m2_idx"]         = cols_v
        arr["cosine_sim"]     = sims_v
        arr["sim"]            = sims_v
        arr["dtw_sim"]        = 0.0
        arr["motion_sim"]     = 0.0
        arr["appearance_sim"] = 0.0
        arr["t1"]             = t1_v
        arr["t2"]             = t2_v
        arr["f1"]             = frames_np[rows_v]
        arr["f2"]             = frames_np[cols_v]
        arr["v1_idx"]         = vididx_np[rows_v]
        arr["v2_idx"]         = vididx_np[cols_v]

        return arr

    def _find_candidates_fallback(
        self,
        V, V_mirror,
        times_np, frames_np, vididx_np,
        direction_np, dir_code_np, scale_np, anchor_y_np,
        threshold, min_gap,
    ) -> Optional[np.ndarray]:
        return self._process_chunk(
            V, V_mirror,
            times_np, frames_np, vididx_np,
            direction_np, dir_code_np, scale_np, anchor_y_np,
            0, len(V), threshold, min_gap,
        )

    def _process_chunk(
        self,
        V, V_mirror,
        times_np, frames_np, vididx_np,
        direction_np, dir_code_np, scale_np, anchor_y_np,
        start: int, end: int,
        threshold: float, min_gap: float,
    ) -> Optional[np.ndarray]:
        n = len(V)
        V_chunk = V[start:end]
        chunk_sz = end - start

        sim = torch.mm(V_chunk, V.t())
        if V_mirror is not None:
            sim_m = torch.mm(V_mirror[start:end], V.t())
            torch.maximum(sim, sim_m, out=sim)
            del sim_m

        local_idx  = torch.arange(chunk_sz, device=self.device)
        global_row = (start + local_idx).unsqueeze(1)
        col_range  = torch.arange(n, device=self.device).unsqueeze(0)
        upper_mask = (col_range > global_row) & (col_range >= start)

        valid_mask = (sim >= threshold) & upper_mask
        del upper_mask, local_idx, global_row, col_range

        cand_rows, cand_cols = torch.where(valid_mask)
        del valid_mask

        if len(cand_rows) == 0:
            del sim
            return None

        cand_sims_gpu = sim[cand_rows, cand_cols]
        del sim

        rows_np = (cand_rows + start).cpu().numpy().astype(np.int32)
        cols_np = cand_cols.cpu().numpy().astype(np.int32)
        sims_np = cand_sims_gpu.cpu().numpy().astype(np.float32)
        del cand_rows, cand_cols, cand_sims_gpu

        t1 = times_np[rows_np]
        t2 = times_np[cols_np]
        v1 = vididx_np[rows_np]
        v2 = vididx_np[cols_np]
        same_video = (v1 == v2)

        keep_mask = ~same_video | (np.abs(t1 - t2) >= min_gap)
        if not keep_mask.any():
            return None

        rows_np = rows_np[keep_mask]
        cols_np = cols_np[keep_mask]
        sims_np = sims_np[keep_mask]
        t1 = t1[keep_mask]
        t2 = t2[keep_mask]

        k = len(rows_np)
        arr = np.empty(k, dtype=_MATCH_DTYPE)
        arr["m1_idx"]         = rows_np
        arr["m2_idx"]         = cols_np
        arr["cosine_sim"]     = sims_np
        arr["sim"]            = sims_np
        arr["dtw_sim"]        = 0.0
        arr["motion_sim"]     = 0.0
        arr["appearance_sim"] = 0.0
        arr["t1"]             = t1
        arr["t2"]             = t2
        arr["f1"]             = frames_np[rows_np]
        arr["f2"]             = frames_np[cols_np]
        arr["v1_idx"]         = vididx_np[rows_np]
        arr["v2_idx"]         = vididx_np[cols_np]
        return arr

    # ==================================================================
    # Build matches — полностью векторизовано
    # ==================================================================

    def _build_matches_from_candidates(
        self,
        candidates: np.ndarray,
        poses_meta: list[dict],
        meta_arrs:  Optional[dict] = None,
    ) -> list[dict]:
        if meta_arrs is None:
            meta_arrs = _build_meta_arrays(poses_meta)

        scale_np    = meta_arrs["scale"]
        anchor_y_np = meta_arrs["anchor_y"]

        m1_idx = candidates["m1_idx"]
        m2_idx = candidates["m2_idx"]
        K = len(candidates)

        # ── (1) Motion-consistency batch ──────────────────────────────
        motion_scores = self._compute_motion_consistency_scores_vec(
            m1_idx, m2_idx, poses_meta,
        )

        # ── (2) Appearance (body-proportions) batch ───────────────────
        appearance_scores = np.full(K, 0.5, dtype=np.float32)
        # Кэш пропорций: один раз достаём по индексу
        # (compare_body_proportions делает Python-cycle на 4 ключа,
        # это ~быстро, но всё равно убираем np.array() / .get() из горячего цикла)
        props_list = [m.get("body_proportions") for m in poses_meta]

        for i in range(K):
            p1 = props_list[m1_idx[i]]
            p2 = props_list[m2_idx[i]]
            if p1 is not None and p2 is not None:
                appearance_scores[i] = compare_body_proportions(p1, p2)

        # ── (3) Penalties — векторно ──────────────────────────────────
        s1 = scale_np[m1_idx]
        s2 = scale_np[m2_idx]
        # Защита от деления на 0 / log(0)
        ratio = np.where(
            (s1 > 0) & (s2 > 0),
            s1 / np.maximum(s2, 1e-9),
            1.0,
        )
        scale_penalty = np.where(
            (s1 > 0) & (s2 > 0),
            np.abs(np.log(np.maximum(ratio, 1e-9))),
            1.0,
        ).astype(np.float32)

        anchor_penalty = np.abs(
            anchor_y_np[m1_idx] - anchor_y_np[m2_idx]
        ).astype(np.float32)

        cosine_sim = candidates["cosine_sim"]

        final_sim = (
            WEIGHT_COSINE     * cosine_sim
            + WEIGHT_MOTION     * motion_scores
            + WEIGHT_APPEARANCE * appearance_scores
        )
        final_sim = final_sim - scale_penalty * 0.5 - anchor_penalty * 0.3
        np.clip(final_sim, 0.0, 1.0, out=final_sim)

        candidates["final_sim"]      = final_sim
        candidates["scale_penalty"]  = scale_penalty
        candidates["anchor_penalty"] = anchor_penalty
        candidates["motion_sim"]     = motion_scores
        candidates["appearance_sim"] = appearance_scores

        # ── (4) Сборка списка словарей одним проходом ─────────────────
        # Здесь нельзя избежать Python-цикла (нужны kp/dir из meta),
        # но мы заранее достаём всё векторно, чтобы внутри цикла были
        # только дешёвые индексации.
        f1_arr = candidates["f1"]
        f2_arr = candidates["f2"]
        t1_arr = candidates["t1"]
        t2_arr = candidates["t2"]
        v1_arr = candidates["v1_idx"]
        v2_arr = candidates["v2_idx"]
        dtw_arr = candidates["dtw_sim"]

        # Готовые "лёгкие" lookup-кэши (вместо .get на каждом шаге)
        kp_list = [m.get("kp") for m in poses_meta]
        dir_list = [m.get("dir", "forward") for m in poses_meta]

        result: list[dict] = [None] * K  # type: ignore
        for i in range(K):
            mi1 = int(m1_idx[i])
            mi2 = int(m2_idx[i])
            result[i] = {
                "m1_idx":         mi1,
                "m2_idx":         mi2,
                "t1":             float(t1_arr[i]),
                "t2":             float(t2_arr[i]),
                "f1":             int(f1_arr[i]),
                "f2":             int(f2_arr[i]),
                "v1_idx":         int(v1_arr[i]),
                "v2_idx":         int(v2_arr[i]),
                "sim":            float(final_sim[i]),
                "sim_raw":        float(cosine_sim[i]),
                "cosine_sim":     float(cosine_sim[i]),
                "dtw_sim":        float(dtw_arr[i]),
                "motion_sim":     float(motion_scores[i]),
                "appearance_sim": float(appearance_scores[i]),
                "direction":      dir_list[mi1],
                "scale1":         float(scale_np[mi1]),
                "scale2":         float(scale_np[mi2]),
                "anchor_y1":      float(anchor_y_np[mi1]),
                "anchor_y2":      float(anchor_y_np[mi2]),
                "kp1":            kp_list[mi1],
                "kp2":            kp_list[mi2],
                "scale_penalty":  float(scale_penalty[i]),
                "anchor_penalty": float(anchor_penalty[i]),
            }

        return result

    # ==================================================================
    # Motion-consistency — векторизовано с предварительным stack-ом
    # ==================================================================

    def _compute_motion_consistency_scores_vec(
        self,
        m1_idx:     np.ndarray,
        m2_idx:     np.ndarray,
        poses_meta: list[dict],
    ) -> np.ndarray:
        """
        БЫЛО: Python-цикл с np.array(...) на каждом кандидате.
        СТАЛО: один stack всех keypoints (с lazy-конверсией) и
        векторное вычисление motion-score через einsum.
        """
        n_meta = len(poses_meta)
        K = len(m1_idx)
        if K == 0:
            return np.zeros(0, dtype=np.float32)

        # 1) Один раз готовим flat-матрицу ВСЕХ keypoints как (n_meta, D)
        #    Кэшируем на объекте matcher, чтобы при повторных вызовах
        #    не пересчитывать (если та же поза).
        kp_matrix = self._get_or_build_kp_matrix(poses_meta)
        if kp_matrix is None:
            return np.full(K, 0.5, dtype=np.float32)

        D = kp_matrix.shape[1]

        # 2) Соседние индексы (с safe-clamp на границы)
        prev1 = np.clip(m1_idx - 1, 0, n_meta - 1)
        next1 = np.clip(m1_idx + 1, 0, n_meta - 1)
        prev2 = np.clip(m2_idx - 1, 0, n_meta - 1)
        next2 = np.clip(m2_idx + 1, 0, n_meta - 1)

        kp1 = kp_matrix[m1_idx]
        kp2 = kp_matrix[m2_idx]

        d1n = kp_matrix[next1] - kp1
        d1p = kp1 - kp_matrix[prev1]
        d2n = kp_matrix[next2] - kp2
        d2p = kp2 - kp_matrix[prev2]

        eps = 1e-6

        n1n = np.linalg.norm(d1n, axis=1) + eps
        n1p = np.linalg.norm(d1p, axis=1) + eps
        nk1 = np.linalg.norm(kp1, axis=1) + eps

        n2n = np.linalg.norm(d2n, axis=1) + eps
        n2p = np.linalg.norm(d2p, axis=1) + eps
        nk2 = np.linalg.norm(kp2, axis=1) + eps

        # cosine(d, kp) построчно
        s1n = np.einsum("ij,ij->i", d1n, kp1) / (n1n * nk1)
        s1p = np.einsum("ij,ij->i", d1p, kp1) / (n1p * nk1)
        s2n = np.einsum("ij,ij->i", d2n, kp2) / (n2n * nk2)
        s2p = np.einsum("ij,ij->i", d2p, kp2) / (n2p * nk2)

        motion_sim1 = (s1n + s1p) * 0.5
        motion_sim2 = (s2n + s2p) * 0.5

        # Если у позы вообще нет движения (на краях) — fallback 0.5
        bad1 = (n1n <= eps * 1.5) | (n1p <= eps * 1.5)
        bad2 = (n2n <= eps * 1.5) | (n2p <= eps * 1.5)
        motion_sim1 = np.where(bad1, 0.5, motion_sim1)
        motion_sim2 = np.where(bad2, 0.5, motion_sim2)

        return ((motion_sim1 + motion_sim2) * 0.5).astype(np.float32)

    def _get_or_build_kp_matrix(
        self,
        poses_meta: list[dict],
    ) -> Optional[np.ndarray]:
        """
        Строит / возвращает кешированную (n_meta, D) матрицу всех keypoints.

        Используется только для motion-consistency. Если у каких-то
        кадров нет 'kp' — заполняем нулями (motion_score → 0.5 fallback).
        """
        # Простой кэш по id() списка poses_meta — сбрасывается между
        # разными запусками find_matches.
        cache_id = id(poses_meta)
        cached = getattr(self, "_kp_matrix_cache", None)
        if cached is not None and cached[0] == cache_id:
            return cached[1]

        n = len(poses_meta)
        if n == 0:
            return None

        # Определяем размерность по первому валидному kp
        D = None
        for m in poses_meta:
            kp = m.get("kp")
            if kp is None:
                continue
            try:
                arr = np.asarray(kp, dtype=np.float32)
                D = int(arr.size)
                break
            except Exception:
                continue

        if D is None:
            self._kp_matrix_cache = (cache_id, None)
            return None

        out = np.zeros((n, D), dtype=np.float32)
        for i, m in enumerate(poses_meta):
            kp = m.get("kp")
            if kp is None:
                continue
            try:
                arr = np.asarray(kp, dtype=np.float32).ravel()
                if arr.size == D:
                    out[i] = arr
                elif arr.size > D:
                    out[i] = arr[:D]
                else:
                    out[i, :arr.size] = arr
            except Exception:
                continue

        self._kp_matrix_cache = (cache_id, out)
        return out

    # ==================================================================
    # Совместимость со старым API (не используется внутри, но оставлено)
    # ==================================================================

    def _compute_motion_consistency_scores(
        self,
        candidates: np.ndarray,
        poses_meta: list[dict],
    ) -> np.ndarray:
        return self._compute_motion_consistency_scores_vec(
            candidates["m1_idx"], candidates["m2_idx"], poses_meta,
        )

    # ==================================================================
    # Strict dedup — O(n log n) вместо O(n²)
    # ==================================================================

    def _remove_duplicates_strict(self, matches: list[dict]) -> list[dict]:
        """
        Жёсткая дедупликация: если разница (по ОБОИМ временам) <
        DUPLICATE_THRESHOLD сек → дубликат. Оставляем лучший по sim.

        Старая версия: O(n²) внутри каждой видеопары.
        Новая: сортируем по sim DESC, проходим жадно;
        для каждой пары видео используем bisect-проверку по t1
        (уже занятым "якорям"), плюс фильтр по t2.
        """
        if not matches:
            return []

        # Группируем по парам видео
        by_video: dict[tuple[int, int], list[dict]] = {}
        for m in matches:
            v1 = m["v1_idx"]
            v2 = m["v2_idx"]
            key = (v1, v2) if v1 <= v2 else (v2, v1)
            by_video.setdefault(key, []).append(m)

        result: list[dict] = []
        thr = DUPLICATE_THRESHOLD

        for group in by_video.values():
            # Сортируем по sim DESC — лучший кандидат идёт первым
            group.sort(key=lambda x: x["sim"], reverse=True)

            # Список занятых t1 (отсортирован) + параллельно
            # сохраняем t2 для каждой записи. Поиск дубликата:
            # bisect по t1 → проверяем соседей (±) и их t2.
            kept_t1: list[float] = []
            kept_t2: list[float] = []  # параллельный массив

            for m in group:
                t1 = float(m["t1"])
                t2 = float(m["t2"])
                idx = bisect.bisect_left(kept_t1, t1)

                is_dup = False
                # Проверяем соседей справа
                j = idx
                while j < len(kept_t1) and (kept_t1[j] - t1) < thr:
                    if abs(kept_t2[j] - t2) < thr:
                        is_dup = True
                        break
                    j += 1
                if not is_dup:
                    # И слева
                    j = idx - 1
                    while j >= 0 and (t1 - kept_t1[j]) < thr:
                        if abs(kept_t2[j] - t2) < thr:
                            is_dup = True
                            break
                        j -= 1

                if not is_dup:
                    pos = bisect.bisect_left(kept_t1, t1)
                    kept_t1.insert(pos, t1)
                    kept_t2.insert(pos, t2)
                    result.append(m)

        return result

    # ==================================================================
    # Pair dedup
    # ==================================================================

    def _dedup_pairs_torch(self, matches: list[dict], n: int) -> list[dict]:
        """
        Дедупликация пар (m1_idx, m2_idx). Для маленьких списков
        идём через numpy (быстрее, чем перенос на GPU). Для больших —
        используем torch (если GPU). Семантика идентична.
        """
        K = len(matches)
        if K == 0:
            return matches

        if K < 4096 or self.device == "cpu":
            # Numpy путь — обычно быстрее для CPU и малых K
            m1 = np.fromiter((m["m1_idx"] for m in matches), dtype=np.int64, count=K)
            m2 = np.fromiter((m["m2_idx"] for m in matches), dtype=np.int64, count=K)
            sim = np.fromiter((m["sim"]   for m in matches), dtype=np.float64, count=K)
            keys = m1 * (n + 1) + m2
            # Хотим: для одинаковых keys оставить один с максимальным sim.
            # Сортируем по (key ASC, sim DESC) → берём первый из группы.
            order = np.lexsort((-sim, keys))
            keys_sorted = keys[order]
            unique_mask = np.empty(K, dtype=bool)
            unique_mask[0] = True
            unique_mask[1:] = keys_sorted[1:] != keys_sorted[:-1]
            chosen = order[unique_mask]
            return [matches[i] for i in chosen.tolist()]

        device = self.device
        m1_arr = torch.tensor([m["m1_idx"] for m in matches], device=device, dtype=torch.int64)
        m2_arr = torch.tensor([m["m2_idx"] for m in matches], device=device, dtype=torch.int64)
        sim_arr = torch.tensor([m["sim"]   for m in matches], device=device, dtype=torch.float32)

        keys = m1_arr * (n + 1) + m2_arr
        # Хитрость со старым кодом сохраняется: сортируем по (key,-sim)
        _, sorted_idx = torch.sort(keys * 1_000_000 - sim_arr * 1000)

        keys_sorted = keys[sorted_idx]
        unique_mask = torch.ones(len(keys_sorted), dtype=torch.bool, device=device)
        unique_mask[1:] = keys_sorted[1:] != keys_sorted[:-1]
        unique_idx = sorted_idx[unique_mask]
        return [matches[idx] for idx in unique_idx.tolist()]

    # ==================================================================
    # Финальная дедупликация (без изменений в логике; bisect уже был)
    # ==================================================================

    def _deduplicate(
        self,
        matches:    list[dict],
        poses_meta: list[dict],
        min_gap:    float,
    ) -> list[dict]:
        if len(matches) == 0:
            return []

        t_start = time.time()

        matches_sorted = sorted(matches, key=lambda m: m["sim"], reverse=True)

        good = [m for m in matches_sorted if m["sim"] >= self.good_threshold]
        junk = [m for m in matches_sorted if m["sim"] <  self.good_threshold]

        junk_take = int(len(junk) * self.junk_ratio)
        candidates = good + junk[:junk_take]

        log.info(
            "[Matcher] good=%d junk=%d junk_taken=%d candidates=%d",
            len(good), len(junk), junk_take, len(candidates),
        )

        SAME_VIDEO_GAP  = SAME_VIDEO_MIN_GAP
        CROSS_VIDEO_GAP = CROSS_VIDEO_MIN_GAP
        max_uniq = min(self.max_unique, int(len(poses_meta) * 0.05))

        used_times: dict[int, list] = {}

        def _is_close(vid: int, t: float, gap: float) -> bool:
            arr_ = used_times.get(vid)
            if arr_ is None:
                return False
            idx = bisect.bisect_left(arr_, t)
            if idx < len(arr_) and abs(arr_[idx] - t) < gap:
                return True
            if idx > 0 and abs(t - arr_[idx - 1]) < gap:
                return True
            return False

        def _mark(vid: int, t: float) -> None:
            arr_ = used_times.get(vid)
            if arr_ is None:
                used_times[vid] = [t]
            else:
                bisect.insort(arr_, t)

        unique_structs: list[dict] = []

        for m in candidates:
            v1 = int(m["v1_idx"])
            v2 = int(m["v2_idx"])
            t1 = float(m["t1"])
            t2 = float(m["t2"])

            gap = SAME_VIDEO_GAP if v1 == v2 else CROSS_VIDEO_GAP

            if _is_close(v1, t1, gap) or _is_close(v2, t2, gap):
                continue

            unique_structs.append(m)
            _mark(v1, t1)
            _mark(v2, t2)

            if len(unique_structs) >= max_uniq:
                break

        log.info(
            "[Matcher] Уникальных: %d (за %.2fс)",
            len(unique_structs), time.time() - t_start,
        )

        if not unique_structs:
            return []
        return unique_structs

    # ==================================================================
    # Config / state (без изменений)
    # ==================================================================

    def _validate_overlap(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            log.warning(
                "[Matcher] overlap(%d) >= chunk_size(%d) → сброс",
                self.chunk_overlap, self.chunk_size,
            )
            self.chunk_overlap = max(0, self.chunk_size // 10)

    def apply_state(self, state) -> None:
        def _int(name, default, lo=1):
            v = getattr(state, name, default)
            try:
                return max(lo, int(v))
            except Exception:
                return default

        def _float(name, default, lo=0.0):
            v = getattr(state, name, default)
            try:
                return max(lo, float(v))
            except Exception:
                return default

        self.chunk_size     = _int("CHUNK_SIZE",            self.chunk_size)
        self.chunk_overlap  = _int("CHUNK_OVERLAP",         self.chunk_overlap, 0)
        self.max_per_chunk  = _int("max_matches_per_chunk", self.max_per_chunk)
        self.max_total      = _int("max_total_matches",     self.max_total)
        self.max_unique     = _int("max_unique_results",    self.max_unique)
        self.min_match_gap  = _float("MIN_MATCH_GAP",       self.min_match_gap)
        self.junk_ratio     = _float("junk_ratio",          self.junk_ratio)
        self.good_threshold = _float("good_threshold",      self.good_threshold)
        self._validate_overlap()

    def apply_config(self, cfg: dict) -> None:
        def _gi(key, cur, lo=1):
            if key not in cfg:
                return cur
            try:
                return max(lo, int(cfg[key]))
            except Exception:
                return cur

        def _gf(key, cur, lo=0.0):
            if key not in cfg:
                return cur
            try:
                return max(lo, float(cfg[key]))
            except Exception:
                return cur

        self.chunk_size     = _gi("chunk_size",         self.chunk_size)
        self.chunk_overlap  = _gi("chunk_overlap",      self.chunk_overlap, 0)
        self.max_unique     = _gi("max_unique_results", self.max_unique)
        self.good_threshold = _gf("good_threshold",     self.good_threshold)
        self.min_match_gap  = _gf("match_gap",          self.min_match_gap)
        self.junk_ratio     = _gf("junk_ratio",         self.junk_ratio)
        self.k_faiss        = _gi("k_faiss",            self.k_faiss)
        self.sim_threshold  = _gf("sim_threshold",      self.sim_threshold)
        self._validate_overlap()
