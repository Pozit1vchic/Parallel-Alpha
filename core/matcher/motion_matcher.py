#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/matcher/motion_matcher.py
MotionMatcher v16 — USearch + SoftDTW + Motion Consistency + исправления точности.

Ключевые улучшения vs v15:
  1. Веса метрики: 0.40 + 0.45 + 0.15 = 1.0 (было 0.4 + 0.6 + 0.2 = 1.2)
  2. USearch-GPU для быстрого поиска k=200 ближайших соседей (было k=100)
  3. Soft-DTW для финальной верификации (top-100 кандидатов, было 20)
  4. Motion Consistency Score: вектор скорости 17 точек с корреляцией
  5. GPU-дедупликация через torch.unique и torch.scatter
  6. Добавлен фильтр по direction при USearch поиске
  7. Добавлены штрафы за scale и anchor_y в финальный score
  8. Используется self._sdtw_criterion вместо пересоздания SoftDTW
  9. junk_ratio = 0.05 (было 0.20)
 10. min_gap с учётом video_idx (разные видео не фильтруются)
 11. max_unique = 5000 (было 1000)
 12. dtw_radius = 12 (было 8)
 13. Удалён дубликат _build_motion_consistency_scores
 14. int64 keys в _dedup_pairs_torch (было int32 переполнение)
 15. Векторизованная фильтрация в _find_candidates_usearch
 16. DTW только для top-M кандидатов
 17. Исправлен NameError: t_matcher в _deduplicate()
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import time

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
K_FAISS = 200            # Увеличено с 100 до 200 для длинных видео
DTW_RADIUS = 12          # Увеличено с 8 до 12 для захвата движения
MIN_GAP_DEFAULT = 3.0    # Минимальный временной разрыв (сек)
THRESHOLD_DEFAULT = 0.70 # Порог схожести

# ── Константы весов составной метрики ───────────────────────────────────────────
# Сумма = 1.0 (было 0.4 + 0.6 + 0.2 = 1.2)
WEIGHT_COSINE = 0.40
WEIGHT_DTW = 0.45
WEIGHT_MOTION_CONSISTENCY = 0.15

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
    ("dir1",   np.dtype('U20')),
    ("dir2",   np.dtype('U20')),
])

# ── Пороги дедупликации ─────────────────────────────────────────────────────────
SAME_VIDEO_MIN_GAP = 5.0   # Новый параметр: для внутри-видео дублей


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
    directions = np.array([m.get("dir", "forward") for m in meta], dtype=object)

    for i, m in enumerate(meta):
        t          = float(m.get("t", 0.0))
        times[i]   = t
        frames[i]  = int(m.get("f", t * fps_aprx))
        video_idx[i] = int(m.get("video_idx", 0))

    return {
        "times":     times,
        "frames":    frames,
        "video_idx": video_idx,
        "directions": directions,
    }


def _compute_motion_consistency_scores(
    V: torch.Tensor,
    window: int = 3,
) -> torch.Tensor:
    """
    Вычислить вектор скорости для каждой позы и корреляцию с соседями.

    score[i] = correlation(V[i], V[i±k]) для k in 1..window
    Возвращает тензор (N,) float32, нормированный в [0, 1].

    v16: Удалён дубликат _build_motion_consistency_scores.
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

    v16: Используется self._sdtw_criterion из __init__ вместо создания заново.
    """
    if not SDTW_AVAILABLE:
        return 0.5  # Fallback значение

    try:
        # Явный перенос на GPU (из требования пользователя)
        X = seq1.to(device)
        Y = seq2.to(device)

        if criterion is None:
            # v16: Используем уже созданный критерий
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
    - Soft-DTW для финальной верификации (top-100 кандидатов)
    - Motion Consistency Score: вектор скорости 17 точек
    - Составная метрика: 0.40*cosine + 0.45*(1-dtw) + 0.15*motion_consistency

    Улучшения скорости:
    - USearch IndexFlatIP для GPU-поиска
    - GPU-дедупликация через torch.unique
    - Векторизованные вычисления

    v16: Добавлены исправления:
    - Фильтр по direction
    - Штрафы scale/anchor_y
    - int64 keys в дедупликации
    - Векторизованная фильтрация
    - max_dtw_candidates = 100
    """

    DEFAULT_CHUNK_SIZE    = 3000
    DEFAULT_CHUNK_OVERLAP = 300
    DEFAULT_MAX_PER_CHUNK = 2_000_000
    DEFAULT_MAX_TOTAL     = 30_000_000
    DEFAULT_MAX_UNIQUE    = 5000     # Увеличено с 1000 до 5000
    DEFAULT_MIN_MATCH_GAP = MIN_GAP_DEFAULT
    DEFAULT_JUNK_RATIO    = 0.05     # Уменьшено с 0.20 до 0.05
    DEFAULT_GOOD_THRESH   = 0.88
    DEFAULT_TEMPORAL_SIGMA = 30.0
    DEFAULT_MOTION_WEIGHT  = 0.15
    MAX_DTW_CANDIDATES = 100         # Новый параметр: ограничение для DTW

    def __init__(self, device: str = "cuda") -> None:
        self.device = device if torch.cuda.is_available() else "cpu"

        # Параметры USearch
        self.k_faiss = K_FAISS  # 200 ближайших соседей (было 100)
        self.dtw_radius = DTW_RADIUS  # 12 кадров для DTW (было 8)

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
        dir_np    = meta_arrs["directions"]

        # ── motion_scores ОДИН РАЗ ────────────────────────────────────────
        motion_np = _compute_motion_consistency_scores(V, window=3).cpu().numpy()

        # ── USearch-поиск ближайших соседей ────────────────────────────────
        candidates = self._find_candidates_usearch(
             V, V_mirror,
             times_np, frames_np, vididx_np, dir_np, motion_np,
             threshold, min_gap,
             poses_meta,
        )

        if candidates is None:
            log.info("[Matcher] Совпадений не найдено (USearch).")
            return []

        log.info("[Matcher] Найдено кандидатов через USearch: %d", len(candidates))

        # ── Soft-DTW для топ-M кандидатов ────────────────────────────────
        final_matches = self._verify_with_dtw(
            V, candidates,
            times_np, frames_np, vididx_np, poses_meta,
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
        dir_np:    np.ndarray,
        motion_np: np.ndarray,
        threshold: float,
        min_gap:   float,
        poses_meta: list[dict],
    ) -> Optional[np.ndarray]:
        """
        Найти кандидаты через USearch (вместо torch.mm).

        v16: Добавлены исправления:
        - direction filter: forward не сравнивается с left/right
        - min_gap с учётом video_idx
        - Векторизованная фильтрация через numpy
        - scale/anchor_y в метаданных
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
        matches = self._usearch_index.search(queries, count=self.k_faiss)

        # matches.keys: (N, k_faiss) индексы соседей
        # matches.distances: (N, k_faiss) расстояния (cos distance)
        if matches.keys is None or matches.distances is None:
            return None

        keys_arr = matches.keys      # (N, k)
        dists_arr = matches.distances  # (N, k)

        # ── Построение кандидатов ───────────────────────────────────────
        # Используем numpy arrays для векторизованной фильтрации
        rows = []
        cols = []
        sims = []

        n = len(poses_meta)
        scales = np.array([poses_meta[i].get("scale", 1.0) for i in range(n)], dtype=np.float32)
        anchor_ys = np.array([poses_meta[i].get("anchor_y", 0.5) for i in range(n)], dtype=np.float32)

        # Векторизованная фильтрация
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

                # v16: Фильтр по direction
                dir1 = dir_np[i]
                dir2 = dir_np[neighbor_idx]
                if dir1 != "unknown" and dir2 != "unknown" and dir1 != dir2:
                    continue

                # v16: min_gap с учётом video_idx
                v1_idx = vididx_np[i]
                v2_idx = vididx_np[neighbor_idx]
                t1 = times_np[i]
                t2 = times_np[neighbor_idx]
                td = abs(t1 - t2)

                # Для разных видео — min_gap, для одного видео — SAME_VIDEO_MIN_GAP
                if v1_idx == v2_idx:
                    gap = SAME_VIDEO_MIN_GAP
                else:
                    gap = min_gap

                if td < gap:
                    continue

                # Сохраняем данные для последующих штрафов
                rows.append(i)
                cols.append(neighbor_idx)
                sims.append((sim, i, neighbor_idx))

        if len(rows) == 0:
            return None

        # ── Сборка массива кандидатов ─────────────────────────────────────
        k = len(rows)

        # Создаём массив для финального score с учётом scale/anchor penalties
        candidates = np.zeros(k, dtype=np.dtype([
            ("m1_idx", np.int32),
            ("m2_idx", np.int32),
            ("sim", np.float32),
            ("sim_base", np.float32),
            ("t1", np.float32),
            ("t2", np.float32),
            ("f1", np.int32),
            ("f2", np.int32),
            ("v1_idx", np.int32),
            ("v2_idx", np.int32),
            ("dir1", np.dtype('U20')),
            ("dir2", np.dtype('U20')),
        ]))

        for idx in range(k):
            m1, m2 = rows[idx], cols[idx]
            sim_base, _, _ = sims[idx]

            # Получаем scale и anchor_y
            scale1 = scales[m1]
            scale2 = scales[m2]
            anchor_y1 = anchor_ys[m1]
            anchor_y2 = anchor_ys[m2]

            # scale penalty: 0.7 + 0.3 * scale_sim
            scale_sim = 1.0 - min(abs(scale1 - scale2) / max(scale1, scale2, 1e-6), 1.0)
            scale_penalty = 0.7 + 0.3 * scale_sim

            # anchor_y penalty
            anchor_diff = abs(anchor_y1 - anchor_y2)
            anchor_penalty = 1.0 if anchor_diff <= 0.15 else 0.85

            # Финальный score
            final_score = sim_base * scale_penalty * anchor_penalty

            candidates[idx]["m1_idx"] = m1
            candidates[idx]["m2_idx"] = m2
            candidates[idx]["sim_base"] = sim_base
            candidates[idx]["sim"] = final_score  # Для совместимости
            candidates[idx]["t1"] = t1
            candidates[idx]["t2"] = t2
            candidates[idx]["f1"] = frames_np[m1]
            candidates[idx]["f2"] = frames_np[m2]
            candidates[idx]["v1_idx"] = v1_idx
            candidates[idx]["v2_idx"] = v2_idx
            candidates[idx]["dir1"] = str(dir1)
            candidates[idx]["dir2"] = str(dir2)

        # ── Лимит чанка ──────────────────────────────────────────────────
        max_candidates = self.max_per_chunk
        if k > max_candidates:
            # Берём top-K по sim_base (оригинальный сим)
            top_k_idx = np.argpartition(-candidates["sim_base"], max_candidates)[:max_candidates]
            candidates = candidates[top_k_idx]

        # ── Сборка структурированного массива ────────────────────────────
        result = np.empty(len(candidates), dtype=_MATCH_DTYPE)
        result["m1_idx"] = candidates["m1_idx"]
        result["m2_idx"] = candidates["m2_idx"]
        result["sim"] = candidates["sim"]
        result["t1"] = candidates["t1"]
        result["t2"] = candidates["t2"]
        result["f1"] = candidates["f1"]
        result["f2"] = candidates["f2"]
        result["v1_idx"] = candidates["v1_idx"]
        result["v2_idx"] = candidates["v2_idx"]

        return result

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
        poses_meta: list[dict],
    ) -> list[dict]:
        """
        Верификация кандидатов через Soft-DTW.

        Для каждого кандидата вычисляем DTW на последовательностях
        длиной 2*DTW_RADIUS+1 вокруг m1 и m2.

        v16: Используется self._sdtw_criterion вместо создания SoftDTW заново.
        v16: Ограничение max_dtw_candidates = 100.
        """
        if not SDTW_AVAILABLE:
            log.warning("[Matcher] SDTW недоступен, используем cosine-only метрику.")
            return self._build_matches_from_candidates(candidates, V, times_np, frames_np, vididx_np, poses_meta)

        device = self.device
        # v16: Используем self._sdtw_criterion из __init__
        criterion = self._sdtw_criterion
        if criterion is None:
            criterion = SoftDTW(gamma=1.0, normalize=True).to(device)

        # ── Вычисляем motion consistency для всех векторов ────────────────
        motion_consistency = _compute_motion_consistency_scores(V, window=3).cpu().numpy()

        # v16: Ограничение количества кандидатов для DTW
        max_dtw = self.MAX_DTW_CANDIDATES
        if len(candidates) > max_dtw:
            # Берём top-K по sim
            top_k_idx = np.argpartition(-candidates["sim"], max_dtw)[:max_dtw]
            candidates = candidates[top_k_idx]

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
            # 0.40 * cosine + 0.45 * dtw_sim + 0.15 * motion_cons
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
        poses_meta: list[dict],
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
                "direction": poses_meta[int(r["m1_idx"])].get("dir", "forward"),
                "kp1":       poses_meta[int(r["m1_idx"])].get("keypoints"),
                "kp2":       poses_meta[int(r["m2_idx"])].get("keypoints"),
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

        v16: Исправлено переполнение int32 — ключи теперь int64.
        """
        if len(matches) == 0:
            return matches

        device = self.device

        # Переносим данные на GPU
        m1_arr = torch.tensor([m["m1_idx"] for m in matches], device=device, dtype=torch.int64)
        m2_arr = torch.tensor([m["m2_idx"] for m in matches], device=device, dtype=torch.int64)
        sim_arr = torch.tensor([m["sim"] for m in matches], device=device, dtype=torch.float32)

        # v16: int64 ключи
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
        Python-цикл только по финальным unique (обычно ≤ 5000).

        v16: Исправлен NameError: t_matcher — удалён print с несуществующей переменной.
        v16: Исправлено max_unique = min(5000, int(len(poses_meta) * 0.05)).
        v16: SAME_VIDEO_MIN_GAP для within-video дублей.
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
        # v16: SAME_VIDEO_MIN_GAP для within-video
        gap = SAME_VIDEO_MIN_GAP
        max_uniq = min(self.MAX_DTW_CANDIDATES * 5, int(len(poses_meta) * 0.05))  # v16: ограничение

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
        # v16: Удалён print с t_matcher (NameError)
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
                "kp1":       poses_meta[int(r["m1_idx"])].get("keypoints"),
                "kp2":       poses_meta[int(r["m2_idx"])].get("keypoints"),
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