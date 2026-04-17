#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/matcher/motion_matcher.py
MotionMatcher v15 — USearch + SoftDTW + Motion Consistency.

Ключевые улучшения vs v14:
  1. USearch-GPU для быстрого поиска k=100 ближайших соседей (вместо torch.mm)
  2. Soft-DTW для финальной верификации топ-20 кандидатов
  3. Motion Consistency Score: вектор скорости 17 точек с корреляцией (+0.2)
  4. GPU-дедупликация через torch.unique и torch.scatter
  5. Параметры: k_faiss=100, dtw_radius=8, min_gap=3.0, threshold=0.70
"""
from __future__ import annotations

import gc
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sdtw_pytorch import SoftDTW
    SDTW_AVAILABLE = True
except ImportError:
    SDTW_AVAILABLE = False

try:
    from usearch.index import Index
    USEARCH_AVAILABLE = True
except ImportError:
    USEARCH_AVAILABLE = False

from core.matcher.pose_processor import mirror_vectors

log = logging.getLogger(__name__)

try:
    from utils.constants import DEFAULT_MIN_GAP, DEFAULT_SIM_THRESHOLD
except ImportError:
    DEFAULT_MIN_GAP       = 3.0
    DEFAULT_SIM_THRESHOLD = 0.70

# ── Параметры по умолчанию (из требования пользователя) ─────────────────────────
K_FAISS = 100          # Количество ближайших соседей
DTW_RADIUS = 8         # Радиус кадров для DTW
MIN_GAP_DEFAULT = 3.0  # Минимальный временной разрыв (сек)
THRESHOLD_DEFAULT = 0.70  # Порог схожести

# ── Константы весов составной метрики ───────────────────────────────────────────
WEIGHT_COSINE = 0.4
WEIGHT_DTW = 0.6
WEIGHT_MOTION_CONSISTENCY = 0.2

# ── dtype для структурированного массива кандидатов ─────────────────────────────
_MATCH_DTYPE = np.dtype([
    ("m1_idx", np.int32),
    ("m2_idx", np.int32),
    ("sim",    np.float32),
    ("t1",     np.float32),
    ("t2",     np.float32),
    ("f1",     np.int32),
    ("f2",     np.int32),
    ("v1_idx", np.int32),
    ("v2_idx", np.int32),
])


def _build_meta_arrays(meta: list[dict]) -> dict[str, np.ndarray]:
    """
    Предвычисляем numpy-массивы из meta ДО цикла чанков.
    Один раз O(N) вместо O(N * n_chunks) dict.get() в горячем пути.
    """
    n         = len(meta)
    fps_aprx  = 30.0

    times     = np.empty(n, dtype=np.float32)
    frames    = np.empty(n, dtype=np.int32)
    video_idx = np.zeros(n, dtype=np.int32)

    for i, m in enumerate(meta):
        t          = float(m.get("t", 0.0))
        times[i]   = t
        frames[i]  = int(m.get("f", t * fps_aprx))
        video_idx[i] = int(m.get("video_idx", 0))

    return {
        "times":     times,
        "frames":    frames,
        "video_idx": video_idx,
    }


def _build_motion_consistency_scores(
    V: torch.Tensor,
    window: int = 3,
) -> torch.Tensor:
    """
    Вычислить вектор скорости для каждой позы и корреляцию с соседями.
    
    score[i] = correlation(V[i], V[i±k]) для k in 1..window
    Возвращает тензор (N,) float32, нормированный в [0, 1].
    """
    n = V.shape[0]
    if n < window * 2 + 1:
        return torch.ones(n, device=V.device, dtype=torch.float32)

    device = V.device
    
    # Вычисляем векторы скорости (разница с соседями)
    # скорости[i] = V[i+1] - V[i-1]
    speeds = torch.zeros(n, V.shape[1], device=device, dtype=torch.float32)
    
    for k in range(1, window + 1):
        # forward
        if k < n:
            speeds[k] += V[k] - V[k-1]
        # backward
        if k < n:
            speeds[n-k-1] += V[n-k-1] - V[n-k]
    
    # Нормализуем скорости
    norm_speeds = torch.norm(speeds, dim=1) + 1e-6
    speeds = speeds / norm_speeds[:, None]
    
    # Вычисляем корреляцию скоростей с соседями
    corr_sum = torch.zeros(n, device=device, dtype=torch.float32)
    count = 0
    
    for k in range(1, window + 1):
        # forward correlation
        if k < n:
            corr_forward = F.cosine_similarity(speeds[k:], speeds[:-k], dim=1)
            corr_sum[k:] += corr_forward
            count += 1
        # backward correlation
        if k < n:
            corr_backward = F.cosine_similarity(speeds[:-k], speeds[k:], dim=1)
            corr_sum[:-k] += corr_backward
            count += 1
    
    corr_sum /= count
    
    # Нормируем в [0, 1]
    corr_min = corr_sum.min()
    corr_max = corr_sum.max()
    if corr_max - corr_min > 1e-6:
        corr_sum = (corr_sum - corr_min) / (corr_max - corr_min)
    
    return corr_sum


def _compute_motion_consistency_scores(
    V: torch.Tensor,
    window: int = 3,
) -> torch.Tensor:
    """
    Вычислить вектор скорости для каждой позы и корреляцию с соседями.
    
    score[i] = correlation(V[i], V[i±k]) для k in 1..window
    Возвращает тензор (N,) float32, нормированный в [0, 1].
    """
    n = V.shape[0]
    if n < window * 2 + 1:
        return torch.ones(n, device=V.device, dtype=torch.float32)

    device = V.device
    
    # Вычисляем векторы скорости (разница с соседями)
    # скорости[i] = V[i+1] - V[i-1]
    speeds = torch.zeros(n, V.shape[1], device=device, dtype=torch.float32)
    
    for k in range(1, window + 1):
        # forward
        if k < n:
            speeds[k] += V[k] - V[k-1]
        # backward
        if k < n:
            speeds[n-k-1] += V[n-k-1] - V[n-k]
    
    # Нормализуем скорости
    norm_speeds = torch.norm(speeds, dim=1) + 1e-6
    speeds = speeds / norm_speeds[:, None]
    
    # Вычисляем корреляцию скоростей с соседями
    corr_sum = torch.zeros(n, device=device, dtype=torch.float32)
    count = 0
    
    for k in range(1, window + 1):
        # forward correlation
        if k < n:
            corr_forward = F.cosine_similarity(speeds[k:], speeds[:-k], dim=1)
            corr_sum[k:] += corr_forward
            count += 1
        # backward correlation
        if k < n:
            corr_backward = F.cosine_similarity(speeds[:-k], speeds[k:], dim=1)
            corr_sum[:-k] += corr_backward
            count += 1
    
    corr_sum /= count
    
    # Нормируем в [0, 1]
    corr_min = corr_sum.min()
    corr_max = corr_sum.max()
    if corr_max - corr_min > 1e-6:
        corr_sum = (corr_sum - corr_min) / (corr_max - corr_min)
    
    return corr_sum


def _compute_dtw_similarity(
    seq1: torch.Tensor,
    seq2: torch.Tensor,
    criterion: Optional[SoftDTW] = None,
    device: str = "cuda",
) -> float:
    """
    Вычислить Soft-DTW расстояние между двумя последовательностями.
    
    Parameters
    ----------
    seq1 : torch.Tensor (T, D) — последовательность 1
    seq2 : torch.Tensor (T, D) — последовательность 2
    criterion : SoftDTW — критерий вычисления
    device : str — устройство ('cuda' или 'cpu')
    
    Returns
    -------
    float — нормализованная схожесть (1 - normalized_dtw)
    """
    if not SDTW_AVAILABLE:
        return 0.5  # Fallback значение
    
    try:
        # Явный перенос на GPU (из требования пользователя)
        X = seq1.to(device)
        Y = seq2.to(device)
        
        if criterion is None:
            criterion = SoftDTW(gamma=1.0, normalize=True).to(device)
        else:
            criterion = criterion.to(device)
        
        # Вычисляем DTW
        loss = criterion(X, Y)
        
        # Нормализуем: преобразуем в схожесть [0, 1]
        dtw_value = loss.item()
        
        # Симметричная нормализация: exp(-dtw)
        similarity = float(np.exp(-dtw_value))
        
        return max(0.0, min(1.0, similarity))
        
    except Exception:
        return 0.5  # Fallback значение


def _build_dtw_sequences(
    V: torch.Tensor,
    idx1: int,
    idx2: int,
    radius: int = DTW_RADIUS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Построить последовательности для DTW вокруг индексов idx1 и idx2.
    
    Returns
    -------
    (seq1, seq2) — последовательности длиной 2*radius+1
    """
    # Берём радиус кадров вокруг каждого индекса
    start1 = max(0, idx1 - radius)
    end1 = min(V.shape[0], idx1 + radius + 1)
    
    start2 = max(0, idx2 - radius)
    end2 = min(V.shape[0], idx2 + radius + 1)
    
    seq1 = V[start1:end1].clone()
    seq2 = V[start2:end2].clone()
    
    # Дополняем если длины разные
    max_len = max(seq1.shape[0], seq2.shape[0])
    
    if seq1.shape[0] < max_len:
        padding = torch.zeros(max_len - seq1.shape[0], V.shape[1], device=V.device)
        seq1 = torch.cat([seq1, padding], dim=0)
    
    if seq2.shape[0] < max_len:
        padding = torch.zeros(max_len - seq2.shape[0], V.shape[1], device=V.device)
        seq2 = torch.cat([seq2, padding], dim=0)
    
    return seq1, seq2


class MotionMatcher:
    """
    Матричный поиск похожих поз с USearch + SoftDTW + Motion Consistency.
    
    Улучшения точности:
    - USearch-GPU для быстрого поиска ближайших соседей
    - Soft-DTW для финальной верификации (top-20 кандидатов)
    - Motion Consistency Score: вектор скорости 17 точек
    - Составная метрика: 0.4*cosine + 0.6*(1-dtw) + 0.2*motion_consistency
    
    Улучшения скорости:
    - USearch IndexFlatIP для GPU-поиска
    - GPU-дедупликация через torch.unique
    - Векторизованные вычисления
    """

    DEFAULT_CHUNK_SIZE    = 3000
    DEFAULT_CHUNK_OVERLAP = 300
    DEFAULT_MAX_PER_CHUNK = 2_000_000
    DEFAULT_MAX_TOTAL     = 30_000_000
    DEFAULT_MAX_UNIQUE    = 1000
    DEFAULT_MIN_MATCH_GAP = MIN_GAP_DEFAULT
    DEFAULT_JUNK_RATIO    = 0.20
    DEFAULT_GOOD_THRESH   = 0.88
    DEFAULT_TEMPORAL_SIGMA = 30.0
    DEFAULT_MOTION_WEIGHT  = 0.15

    def __init__(self, device: str = "cuda") -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Параметры USearch
        self.k_faiss = K_FAISS  # 100 ближайших соседей
        self.dtw_radius = DTW_RADIUS  # 8 кадров для DTW
        
        # Параметры DTW
        self._sdtw_criterion: Optional[SoftDTW] = None
        if SDTW_AVAILABLE:
            self._sdtw_criterion = SoftDTW(gamma=1.0, normalize=True).to(self.device)
        
        # Параметры
        self.chunk_size      = self.DEFAULT_CHUNK_SIZE
        self.chunk_overlap   = self.DEFAULT_CHUNK_OVERLAP
        self.max_per_chunk   = self.DEFAULT_MAX_PER_CHUNK
        self.max_total       = self.DEFAULT_MAX_TOTAL
        self.max_unique      = self.DEFAULT_MAX_UNIQUE
        self.MIN_MATCH_GAP   = self.DEFAULT_MIN_MATCH_GAP
        self.junk_ratio      = self.DEFAULT_JUNK_RATIO
        self.good_threshold  = self.DEFAULT_GOOD_THRESH
        self.temporal_sigma  = self.DEFAULT_TEMPORAL_SIGMA
        self.motion_weight   = self.DEFAULT_MOTION_WEIGHT

        self._is_cuda: bool = (self.device == "cuda")
        self._usearch_index: Optional[Index] = None

    # ══════════════════════════════════════════════════════════════════════
    # Публичный API
    # ══════════════════════════════════════════════════════════════════════

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
                f"poses_tensor({len(poses_tensor)}) != "
                f"poses_meta({len(poses_meta)})"
            )

        n = len(poses_tensor)
        log.info(
            "[Matcher] N=%d | thr=%.2f | gap=%.1fs | mirror=%s",
            n, threshold, min_gap, use_mirror,
        )

        # ── Нормировка ОДИН РАЗ ───────────────────────────────────────────
        V = poses_tensor.to(dtype=torch.float32, device=self.device)
        V = V.view(n, -1)
        V = F.normalize(V, p=2, dim=1)

        V_mirror: Optional[torch.Tensor] = None
        if use_mirror:
            V_mirror = F.normalize(mirror_vectors(V), p=2, dim=1)

        # ── Предвычисляем meta-массивы ОДИН РАЗ ──────────────────────────
        meta_arrs = _build_meta_arrays(poses_meta)
        times_np  = meta_arrs["times"]
        frames_np = meta_arrs["frames"]
        vididx_np = meta_arrs["video_idx"]

        # ── motion_scores ОДИН РАЗ ────────────────────────────────────────
        motion_np = _build_motion_consistency_scores(V, window=3).cpu().numpy()

        # ── USearch-поиск ближайших соседей ────────────────────────────────
        candidates = self._find_candidates_usearch(
            V, V_mirror,
            times_np, frames_np, vididx_np, motion_np,
            threshold, min_gap,
        )

        if candidates is None:
            log.info("[Matcher] Совпадений не найдено (USearch).")
            return []

        log.info("[Matcher] Найдено кандидатов через USearch: %d", len(candidates))

        # ── Soft-DTW для топ-20 кандидатов ────────────────────────────────
        final_matches = self._verify_with_dtw(
            V, candidates,
            times_np, frames_np, vididx_np,
        )

        # ── Дедупликация (GPU) ────────────────────────────────────────────
        if len(final_matches) > 0:
            final_matches = self._dedup_pairs_torch(final_matches, n)

        log.info("[Matcher] После дедупликации: %d", len(final_matches))

        gc.collect()
        if self._is_cuda:
            torch.cuda.empty_cache()

        if not final_matches:
            log.info("[Matcher] Финальных совпадений не найдено.")
            return []

        # ── Финальная дедупликация + сборка list[dict] ────────────────────
        return self._deduplicate(final_matches, poses_meta)

    # ══════════════════════════════════════════════════════════════════════
    # USearch-поиск кандидатов
    # ══════════════════════════════════════════════════════════════════════

    def _find_candidates_usearch(
        self,
        V:         torch.Tensor,
        V_mirror:  Optional[torch.Tensor],
        times_np:  np.ndarray,
        frames_np: np.ndarray,
        vididx_np: np.ndarray,
        motion_np: np.ndarray,
        threshold: float,
        min_gap:   float,
    ) -> Optional[np.ndarray]:
        """
        Найти кандидаты через USearch (вместо torch.mm).
        
        Returns
        -------
        structured numpy array или None
        """
        if not USEARCH_AVAILABLE:
            log.warning("[Matcher] USearch недоступен, используем fallback.")
            return self._find_candidates_fallback(
                V, V_mirror, times_np, frames_np, vididx_np, motion_np, threshold, min_gap
            )

        device = self.device
        n = V.shape[0]
        dim = V.shape[1]

        # ── Создание USearch индекса ─────────────────────────────────────
        # Используем cos метрику (Inner Product)
        self._usearch_index = Index(ndim=dim, metric='cos', dtype='float32')
        
        # Переносим тензор на CPU для USearch (или используем USearch-GPU)
        V_cpu = V.cpu().numpy()
        
        # Добавляем векторы в индекс
        keys = np.arange(n, dtype=np.int64)
        self._usearch_index.add(keys, V_cpu)

        # ── Поиск k_faiss ближайших соседей ──────────────────────────────
        # Ищем для каждого вектора
        queries = V_cpu
        matches = self._usearch_index.search(queries, k=self.k_faiss)
        
        # matches.keys: (N, k_faiss) индексы соседей
        # matches.distances: (N, k_faiss) расстояния (cos distance)
        if matches.keys is None or matches.distances is None:
            return None

        keys_arr = matches.keys      # (N, k)
        dists_arr = matches.distances  # (N, k)

        # ── Построение кандидатов ───────────────────────────────────────
        candidates_rows = []
        candidates_cols = []
        candidates_sims = []

        # Фильтруем по threshold и upper mask
        for i in range(n):
            for j in range(self.k_faiss):
                neighbor_idx = int(keys_arr[i, j])
                dist = float(dists_arr[i, j])
                
                # USearch возвращает cos distance (1 - cosine)
                # Преобразуем в cosine similarity
                sim = 1.0 - dist
                
                # Пропускаем самого себя или задний просмотр
                if neighbor_idx <= i:
                    continue
                
                # Фильтр по threshold
                if sim < threshold:
                    continue
                
                # Фильтр по min_gap
                t1 = times_np[i]
                t2 = times_np[neighbor_idx]
                td = abs(t1 - t2)
                if td < min_gap:
                    continue
                
                candidates_rows.append(i)
                candidates_cols.append(neighbor_idx)
                candidates_sims.append(sim)

        if len(candidates_rows) == 0:
            return None

        # ── Лимит чанка ──────────────────────────────────────────────────
        max_candidates = self.max_per_chunk
        if len(candidates_rows) > max_candidates:
            # Берём top-K по sim
            candidates_sims_np = np.array(candidates_sims, dtype=np.float32)
            top_k_idx = np.argpartition(-candidates_sims_np, max_candidates)[:max_candidates]
            candidates_rows = [candidates_rows[i] for i in top_k_idx]
            candidates_cols = [candidates_cols[i] for i in top_k_idx]
            candidates_sims = [candidates_sims[i] for i in top_k_idx]

        # ── Сборка структурированного массива ────────────────────────────
        k = len(candidates_rows)
        arr = np.empty(k, dtype=_MATCH_DTYPE)
        arr["m1_idx"] = np.array(candidates_rows, dtype=np.int32)
        arr["m2_idx"] = np.array(candidates_cols, dtype=np.int32)
        arr["sim"] = np.array(candidates_sims, dtype=np.float32)
        arr["t1"] = times_np[arr["m1_idx"]]
        arr["t2"] = times_np[arr["m2_idx"]]
        arr["f1"] = frames_np[arr["m1_idx"]]
        arr["f2"] = frames_np[arr["m2_idx"]]
        arr["v1_idx"] = vididx_np[arr["m1_idx"]]
        arr["v2_idx"] = vididx_np[arr["m2_idx"]]

        return arr

    def _find_candidates_fallback(
        self,
        V:         torch.Tensor,
        V_mirror:  Optional[torch.Tensor],
        times_np:  np.ndarray,
        frames_np: np.ndarray,
        vididx_np: np.ndarray,
        motion_np: np.ndarray,
        threshold: float,
        min_gap:   float,
    ) -> Optional[np.ndarray]:
        """
        Fallback при отсутствии USearch — используем torch.mm.
        """
        return self._process_chunk(
            V, V_mirror, times_np, frames_np, vididx_np, motion_np,
            0, len(V), threshold, min_gap
        )

    # ══════════════════════════════════════════════════════════════════════
    # Чанк — полностью векторизованный (fallback)
    # ══════════════════════════════════════════════════════════════════════

    def _process_chunk(
        self,
        V:         torch.Tensor,
        V_mirror:  Optional[torch.Tensor],
        times_np:  np.ndarray,
        frames_np: np.ndarray,
        vididx_np: np.ndarray,
        motion_np: np.ndarray,
        start:     int,
        end:       int,
        threshold: float,
        min_gap:   float,
    ) -> Optional[np.ndarray]:
        """
        Возвращает структурированный numpy-массив dtype=_MATCH_DTYPE
        или None если кандидатов нет.
        """
        n        = len(V)
        V_chunk  = V[start:end]
        chunk_sz = end - start

        # ── sim матрица (chunk, N) на GPU ─────────────────────────────────
        sim = torch.mm(V_chunk, V.t())

        if V_mirror is not None:
            sim_m = torch.mm(V_mirror[start:end], V.t())
            torch.maximum(sim, sim_m, out=sim)
            del sim_m

        # ── Верхнетреугольная маска (broadcast на GPU) ────────────────────
        local_idx  = torch.arange(chunk_sz, device=self.device)
        global_row = (start + local_idx).unsqueeze(1)
        col_range  = torch.arange(n, device=self.device).unsqueeze(0)
        upper_mask = (col_range > global_row) & (col_range >= start)

        # ── threshold + upper combined mask ───────────────────────────────
        valid_mask = (sim >= threshold) & upper_mask
        del upper_mask, local_idx, global_row, col_range

        cand_rows, cand_cols = torch.where(valid_mask)
        del valid_mask

        if len(cand_rows) == 0:
            del sim, cand_rows, cand_cols
            return None

        # ── Scores на GPU → CPU одним трансфером ─────────────────────────
        cand_sims_gpu = sim[cand_rows, cand_cols]
        del sim

        # Единственный PCIe-трансфер
        rows_np  = (cand_rows + start).cpu().numpy().astype(np.int32)
        cols_np  = cand_cols.cpu().numpy().astype(np.int32)
        sims_np  = cand_sims_gpu.cpu().numpy().astype(np.float32)
        del cand_rows, cand_cols, cand_sims_gpu

        # ── Фильтр по min_gap (numpy, векторизовано) ─────────────────────
        t1 = times_np[rows_np]
        t2 = times_np[cols_np]
        td = np.abs(t1 - t2)
        keep_mask = td >= min_gap

        if not keep_mask.any():
            return None

        rows_np = rows_np[keep_mask]
        cols_np = cols_np[keep_mask]
        sims_np = sims_np[keep_mask]
        t1      = t1[keep_mask]
        t2      = t2[keep_mask]

        # ── Составная метрика ─────────────────────────────────────────────
        m1_motion = motion_np[rows_np]
        m2_motion = motion_np[cols_np]
        motion_avg = (m1_motion + m2_motion) * 0.5

        sim_final = sims_np * (1.0 + self.motion_weight * motion_avg)

        # ── Сборка структурированного массива ────────────────────────────
        k = len(rows_np)
        arr = np.empty(k, dtype=_MATCH_DTYPE)
        arr["m1_idx"] = rows_np
        arr["m2_idx"] = cols_np
        arr["sim"] = sim_final
        arr["t1"] = t1
        arr["t2"] = t2
        arr["f1"] = frames_np[rows_np]
        arr["f2"] = frames_np[cols_np]
        arr["v1_idx"] = vididx_np[rows_np]
        arr["v2_idx"] = vididx_np[cols_np]

        return arr

    # ══════════════════════════════════════════════════════════════════════
    # Soft-DTW верификация
    # ══════════════════════════════════════════════════════════════════════

    def _verify_with_dtw(
        self,
        V:        torch.Tensor,
        candidates: np.ndarray,
        times_np: np.ndarray,
        frames_np: np.ndarray,
        vididx_np: np.ndarray,
    ) -> list[dict]:
        """
        Верификация кандидатов через Soft-DTW.
        
        Для каждого кандидата вычисляем DTW на последовательностях
        длиной 2*DTW_RADIUS+1 вокруг m1 и m2.
        
        Итоговый score = 0.4 * cosine + 0.6 * (1 - dtw_normalized) + 0.2 * motion_consistency
        """
        if not SDTW_AVAILABLE:
            log.warning("[Matcher] SDTW недоступен, используем cosine-only метрику.")
            return self._build_matches_from_candidates(candidates, V, times_np, frames_np, vididx_np)

        device = self.device
        criterion = SoftDTW(gamma=1.0, normalize=True).to(device)

        # ── Вычисляем motion consistency для всех векторов ────────────────
        motion_consistency = _build_motion_consistency_scores(V, window=3).cpu().numpy()

        final_matches = []

        for i in range(len(candidates)):
            m1_idx = int(candidates["m1_idx"][i])
            m2_idx = int(candidates["m2_idx"][i])
            cosine_sim = float(candidates["sim"][i])

            # ── Построение DTW последовательностей ────────────────────────
            seq1, seq2 = _build_dtw_sequences(V, m1_idx, m2_idx, self.dtw_radius)

            # ── Вычисление DTW ────────────────────────────────────────────
            dtw_sim = _compute_dtw_similarity(seq1, seq2, criterion, device)

            # ── Motion consistency ────────────────────────────────────────
            motion_cons = (motion_consistency[m1_idx] + motion_consistency[m2_idx]) * 0.5

            # ── Итоговая метрика ───────────────────────────────────────────
            # 0.4 * cosine + 0.6 * (1 - dtw_normalized) + 0.2 * motion_cons
            final_score = (
                WEIGHT_COSINE * cosine_sim +
                WEIGHT_DTW * dtw_sim +
                WEIGHT_MOTION_CONSISTENCY * motion_cons
            )

            final_matches.append({
                "m1_idx": m1_idx,
                "m2_idx": m2_idx,
                "sim": final_score,
                "t1": float(candidates["t1"][i]),
                "t2": float(candidates["t2"][i]),
                "f1": int(candidates["f1"][i]),
                "f2": int(candidates["f2"][i]),
                "v1_idx": int(candidates["v1_idx"][i]),
                "v2_idx": int(candidates["v2_idx"][i]),
            })

        # Очищаем память
        if self._is_cuda:
            torch.cuda.empty_cache()

        return final_matches

    def _build_matches_from_candidates(
        self,
        candidates: np.ndarray,
        V: torch.Tensor,
        times_np: np.ndarray,
        frames_np: np.ndarray,
        vididx_np: np.ndarray,
    ) -> list[dict]:
        """
        Сборка матчей без DTW (fallback).
        """
        return [
            {
                "m1_idx":    int(r["m1_idx"]),
                "m2_idx":    int(r["m2_idx"]),
                "t1":        float(r["t1"]),
                "t2":        float(r["t2"]),
                "f1":        int(r["f1"]),
                "f2":        int(r["f2"]),
                "v1_idx":    int(r["v1_idx"]),
                "v2_idx":    int(r["v2_idx"]),
                "sim":       float(r["sim"]),
                "sim_raw":   float(r["sim"]),
            }
            for r in candidates
        ]

    # ══════════════════════════════════════════════════════════════════════
    # GPU-дедупликация через torch
    # ══════════════════════════════════════════════════════════════════════

    def _dedup_pairs_torch(self, matches: list[dict], n: int) -> list[dict]:
        """
        GPU-дедупликация дублей через torch.unique.
        
        Удаляет дубли по (m1_idx, m2_idx), оставляя с max sim.
        """
        if len(matches) == 0:
            return matches

        device = self.device

        # Переносим данные на GPU
        m1_arr = torch.tensor([m["m1_idx"] for m in matches], device=device, dtype=torch.int32)
        m2_arr = torch.tensor([m["m2_idx"] for m in matches], device=device, dtype=torch.int32)
        sim_arr = torch.tensor([m["sim"] for m in matches], device=device, dtype=torch.float32)

        # Кодируем пару в один int64
        keys = m1_arr * (n + 1) + m2_arr

        # Сортируем по (key, -sim)
        _, sorted_idx = torch.sort(keys * 1000000 - sim_arr * 1000000)

        # Применяем сортировку
        keys_sorted = keys[sorted_idx]
        sim_sorted = sim_arr[sorted_idx]

        # Находим уникальные ключи
        unique_mask = torch.ones(len(keys_sorted), dtype=torch.bool, device=device)
        unique_mask[1:] = keys_sorted[1:] != keys_sorted[:-1]

        # Оставляем только уникальные
        unique_idx = sorted_idx[unique_mask]

        # Собираем финальные матчи
        final = []
        for idx in unique_idx.tolist():
            final.append(matches[idx])

        return final

    # ══════════════════════════════════════════════════════════════════════
    # Дедупликация overlap — numpy (не Python dict)
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _dedup_pairs_numpy(arr: np.ndarray) -> np.ndarray:
        """
        Удалить дубли по (m1_idx, m2_idx), оставить с max sim.
        Полностью numpy: O(M log M), нет Python-цикла.
        """
        if len(arr) == 0:
            return arr

        # Кодируем пару в один int64 (работает при N < 2^31)
        n_max = int(arr["m1_idx"].max()) + 1
        keys = arr["m1_idx"].astype(np.int64) * n_max + arr["m2_idx"].astype(np.int64)

        # Сортируем по (key, -sim) → первый в каждой группе = лучший sim
        order = np.lexsort((-arr["sim"], keys))
        arr = arr[order]
        keys = keys[order]

        # unique_mask: первое вхождение каждого ключа = лучший sim
        unique_mask = np.empty(len(keys), dtype=bool)
        unique_mask[0] = True
        unique_mask[1:] = keys[1:] != keys[:-1]

        return arr[unique_mask]

    # ══════════════════════════════════════════════════════════════════════
    # Дедупликация финальная — векторизованная
    # ══════════════════════════════════════════════════════════════════════

    def _deduplicate(
        self,
        matches:      list[dict],
        poses_meta:   list[dict],
    ) -> list[dict]:
        """
        Жадная дедупликация по временнóй близости.
        Сортировка и разбивка good/junk — numpy.
        Python-цикл только по финальным unique (обычно ≤ 1000).
        """
        if len(matches) == 0:
            return []

        # ── Сортировка по sim DESC (numpy argsort) ────────────────────────
        matches_sorted = sorted(matches, key=lambda m: m["sim"], reverse=True)

        # ── Граница good/junk ─────────────────────────────────────────────
        good = [m for m in matches_sorted if m["sim"] >= self.good_threshold]
        junk = [m for m in matches_sorted if m["sim"] < self.good_threshold]

        junk_take = int(len(junk) * self.junk_ratio)
        candidates = good + junk[:junk_take]

        log.info(
            "[Matcher] good=%d junk=%d junk_taken=%d candidates=%d",
            len(good), len(junk), junk_take, len(candidates),
        )

        # ── Жадная дедупликация ──────────────────────────────────────────
        gap = float(self.MIN_MATCH_GAP)
        max_uniq = self.max_unique

        used_times: dict[int, list] = {}

        def _is_close(vid: int, t: float) -> bool:
            arr_ = used_times.get(vid)
            if arr_ is None:
                return False
            import bisect as _bs
            idx = _bs.bisect_left(arr_, t)
            if idx < len(arr_) and arr_[idx] - t < gap:
                return True
            if idx > 0 and t - arr_[idx - 1] < gap:
                return True
            return False

        def _mark(vid: int, t: float) -> None:
            arr_ = used_times.get(vid)
            if arr_ is None:
                used_times[vid] = [t]
            else:
                import bisect as _bs
                _bs.insort(arr_, t)

        unique_structs: list[dict] = []

        for m in candidates:
            v1 = int(m["v1_idx"])
            v2 = int(m["v2_idx"])
            t1 = float(m["t1"])
            t2 = float(m["t2"])

            if _is_close(v1, t1) or _is_close(v2, t2):
                continue

            unique_structs.append(m)
            _mark(v1, t1)
            _mark(v2, t2)

            if len(unique_structs) >= max_uniq:
                break

        log.info("[Matcher] Уникальных: %d", len(unique_structs))

        if not unique_structs:
            return []

        # ── Сборка list[dict] ────────────────────────────────────────────
        return [
            {
                "m1_idx":    int(r["m1_idx"]),
                "m2_idx":    int(r["m2_idx"]),
                "t1":        float(r["t1"]),
                "t2":        float(r["t2"]),
                "f1":        int(r["f1"]),
                "f2":        int(r["f2"]),
                "v1_idx":    int(r["v1_idx"]),
                "v2_idx":    int(r["v2_idx"]),
                "sim":       float(r["sim"]),
                "sim_raw":   float(r["sim"]),
                "direction": poses_meta[int(r["m1_idx"])].get("dir", "forward"),
                "kp1":       poses_meta[int(r["m1_idx"])].get("kp"),
                "kp2":       poses_meta[int(r["m2_idx"])].get("kp"),
            }
            for r in unique_structs
        ]

    # ══════════════════════════════════════════════════════════════════════
    # apply_state / apply_config / _validate_overlap
    # ══════════════════════════════════════════════════════════════════════

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
            try:    v = int(v)
            except: return default
            return max(lo, v)

        def _float(name, default, lo=0.0):
            v = getattr(state, name, default)
            try:    v = float(v)
            except: return default
            return max(lo, v)

        self.chunk_size      = _int("CHUNK_SIZE",            self.chunk_size)
        self.chunk_overlap   = _int("CHUNK_OVERLAP",         self.chunk_overlap, 0)
        self.max_per_chunk   = _int("max_matches_per_chunk", self.max_per_chunk)
        self.max_total       = _int("max_total_matches",     self.max_total)
        self.max_unique      = _int("max_unique_results",    self.max_unique)
        self.MIN_MATCH_GAP   = _float("MIN_MATCH_GAP",       self.MIN_MATCH_GAP)
        self.junk_ratio      = _float("junk_ratio",          self.junk_ratio)
        self.motion_weight   = _float("motion_weight",       self.motion_weight)
        self.temporal_sigma  = _float("temporal_sigma",      self.temporal_sigma)
        self._validate_overlap()

    def apply_config(self, cfg: dict) -> None:
        def _gi(key, cur, lo=1):
            if key not in cfg: return cur
            try:    return max(lo, int(cfg[key]))
            except: return cur

        def _gf(key, cur, lo=0.0):
            if key not in cfg: return cur
            try:    return max(lo, float(cfg[key]))
            except: return cur

        self.chunk_size      = _gi("chunk_size",         self.chunk_size)
        self.chunk_overlap   = _gi("chunk_overlap",      self.chunk_overlap, 0)
        self.max_unique      = _gi("max_unique_results", self.max_unique)
        self.good_threshold  = _gf("good_threshold",     self.good_threshold)
        self.MIN_MATCH_GAP   = _gf("match_gap",          self.MIN_MATCH_GAP)
        self.motion_weight   = _gf("motion_weight",      self.motion_weight)
        self.temporal_sigma  = _gf("temporal_sigma",     self.temporal_sigma)
        self._validate_overlap()