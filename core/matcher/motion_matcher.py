#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/matcher/motion_matcher.py — поиск похожих поз."""
from __future__ import annotations

import bisect
import gc
from typing import TypeAlias

import numpy as np
import torch
import torch.nn.functional as F

from core.matcher.pose_processor import (
    mirror_vectors,
    preprocess_pose,
    BODY_WEIGHTS,
)

PoseDict:  TypeAlias = dict
MetaList:  TypeAlias = list[PoseDict]
MatchList: TypeAlias = list[dict]

# ── Параметры re-scoring ──────────────────────────────────────────────────────

_SCALE_RATIO_MAX = 4.5    # было 4.0
_POS_DIFF_MAX    = 0.75   # было 0.70
_SCALE_PENALTY_K = 0.65   # было 0.8
_POS_PENALTY_K   = 0.9    # было 1.2
_RESCORE_POWER   = 1.3    # оставить
_PAIR_DEDUP_COS  = 0.97   # было 0.97


def _scale_penalties_vec(scales_i: np.ndarray,
                          scales_j: np.ndarray) -> np.ndarray:
    eps    = 1e-5
    ratio  = (np.maximum(scales_i, scales_j)
               / (np.minimum(scales_i, scales_j) + eps))
    result = np.ones(len(ratio), dtype=np.float32)

    mid_mask  = (ratio > 1.5) & (ratio < _SCALE_RATIO_MAX)
    zero_mask = ratio >= _SCALE_RATIO_MAX

    if mid_mask.any():
        t = (ratio[mid_mask] - 1.5) / (_SCALE_RATIO_MAX - 1.5)
        result[mid_mask] = np.exp(-_SCALE_PENALTY_K * t).astype(np.float32)

    result[zero_mask] = 0.0
    return result


def _position_penalties_vec(anchor_y_i: np.ndarray,
                              anchor_y_j: np.ndarray,
                              scales_i:   np.ndarray,
                              scales_j:   np.ndarray) -> np.ndarray:
    diff   = np.abs(anchor_y_i - anchor_y_j)
    result = np.ones(len(diff), dtype=np.float32)

    mid_mask = (diff > 0.20) & (diff < _POS_DIFF_MAX)
    bad_mask = diff >= _POS_DIFF_MAX

    if mid_mask.any():
        t = (diff[mid_mask] - 0.20) / (_POS_DIFF_MAX - 0.20)
        result[mid_mask] = np.exp(-_POS_PENALTY_K * t).astype(np.float32)

    result[bad_mask] = 0.0
    return result


def _rescore_vec(
    sims:       np.ndarray,
    scales_i:   np.ndarray,
    scales_j:   np.ndarray,
    anchor_y_i: np.ndarray,
    anchor_y_j: np.ndarray,
) -> np.ndarray:
    sp = _scale_penalties_vec(scales_i, scales_j)
    pp = _position_penalties_vec(anchor_y_i, anchor_y_j, scales_i, scales_j)
    rescored = (sims ** _RESCORE_POWER) * sp * pp
    return np.minimum(rescored, sims).astype(np.float32)

def _calibrate_sim(sims: np.ndarray, threshold: float) -> np.ndarray:
    """
    Калибрует завышенные косинусные схожести поз.

    Косинус между нормированными векторами поз почти всегда > 0.85
    даже для разных поз — потому что все люди похожи структурно.
    Переводим диапазон [threshold..1.0] → [0.0..1.0] нелинейно.

    threshold=0.75:
      cos=0.75 → sim=0.00  (минимум)
      cos=0.85 → sim=0.40
      cos=0.92 → sim=0.70
      cos=0.97 → sim=0.88
      cos=1.00 → sim=1.00
    """
    lo   = max(0.0, threshold - 0.05)
    hi   = 1.0
    span = hi - lo

    # Линейная нормализация в [0..1]
    normed = np.clip((sims - lo) / span, 0.0, 1.0)

    # Нелинейное сжатие — верхние значения больше не задирает
    calibrated = np.power(normed, 1.8).astype(np.float32)

    return calibrated


class MotionMatcher:
    """Матричный поиск похожих поз."""

    DEFAULT_CHUNK_SIZE    = 3000
    DEFAULT_CHUNK_OVERLAP = 300
    DEFAULT_MAX_PER_CHUNK = 2_000_000
    DEFAULT_MAX_TOTAL     = 30_000_000
    DEFAULT_MAX_UNIQUE    = 3000
    DEFAULT_MIN_MATCH_GAP = 3.0
    DEFAULT_JUNK_RATIO    = 0.20

    BODY_WEIGHTS = BODY_WEIGHTS

    def __init__(self, device: str = "cuda") -> None:
        self.device = device if torch.cuda.is_available() else "cpu"

        self.chunk_size     = self.DEFAULT_CHUNK_SIZE
        self.chunk_overlap  = self.DEFAULT_CHUNK_OVERLAP
        self.max_per_chunk  = self.DEFAULT_MAX_PER_CHUNK
        self.max_total      = self.DEFAULT_MAX_TOTAL
        self.max_unique     = self.DEFAULT_MAX_UNIQUE
        self.MIN_MATCH_GAP  = self.DEFAULT_MIN_MATCH_GAP
        self.junk_ratio     = self.DEFAULT_JUNK_RATIO
        self.rescore        = True
        self.pair_dedup_cos = _PAIR_DEDUP_COS

    # ── Обратная совместимость ────────────────────────────────────────────

    @staticmethod
    def _mirror_vector(vec: torch.Tensor) -> torch.Tensor:
        if vec.dim() == 1:
            return mirror_vectors(vec.unsqueeze(0)).squeeze(0)
        return mirror_vectors(vec)

    @staticmethod
    def preprocess_pose(pose_data: PoseDict,
                        use_body_weights: bool = False) -> np.ndarray:
        return preprocess_pose(pose_data, use_body_weights=use_body_weights)

    # ── Основной поиск ────────────────────────────────────────────────────

    def find_matches(
        self,
        poses_tensor: torch.Tensor,
        poses_meta:   MetaList,
        threshold:    float = 0.75,
        min_gap:      float = 3.0,
        use_mirror:   bool  = False,
    ) -> MatchList:
        if poses_tensor is None or len(poses_tensor) < 2:
            return []

        n = len(poses_tensor)
        if len(poses_meta) != n:
            print(f"[Matcher] Несоответствие: tensor={n}, meta={len(poses_meta)}")
            return []

        V = poses_tensor.to(dtype=torch.float32, device=self.device)
        V = V.view(n, -1)
        V = F.normalize(V, p=2, dim=1)

        V_mirror: torch.Tensor | None = None
        if use_mirror:
            V_mirror = F.normalize(self._mirror_vector(V), p=2, dim=1)

        times = torch.tensor(
            [m["t"] for m in poses_meta],
            dtype=torch.float32, device=self.device)
        video_ids = torch.tensor(
            [int(m.get("video_idx", 0)) for m in poses_meta],
            dtype=torch.int32, device=self.device)

        scales    = np.array(
            [float(m.get("scale",    1.0)) for m in poses_meta],
            dtype=np.float32)
        anchor_ys = np.array(
            [float(m.get("anchor_y", 0.5)) for m in poses_meta],
            dtype=np.float32)

        step     = max(1, self.chunk_size - self.chunk_overlap)
        n_chunks = max(1, (n + step - 1) // step)
        raw_matches: MatchList = []

        for ci in range(n_chunks):
            start = ci * step
            end   = min(start + self.chunk_size, n)
            if start >= n:
                break
            if len(raw_matches) >= self.max_total:
                break

            chunk_m = self._process_chunk(
                V=V, V_mirror=V_mirror,
                times=times, video_ids=video_ids,
                meta=poses_meta,
                scales=scales, anchor_ys=anchor_ys,
                start=start, end=end,
                threshold=threshold, min_gap=min_gap,
            )
            raw_matches.extend(chunk_m)

            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        return self._deduplicate(raw_matches, min_gap=min_gap, V_cpu=V.cpu())

    # ── Обработка чанка ───────────────────────────────────────────────────

    def _process_chunk(
        self,
        V:         torch.Tensor,
        V_mirror:  torch.Tensor | None,
        times:     torch.Tensor,
        video_ids: torch.Tensor,
        meta:      MetaList,
        scales:    np.ndarray,
        anchor_ys: np.ndarray,
        start:     int,
        end:       int,
        threshold: float,
        min_gap:   float,
    ) -> MatchList:
        V_chunk  = V[start:end]
        T_chunk  = times[start:end]
        VI_chunk = video_ids[start:end]
        N        = len(V)

        sim = torch.mm(V_chunk, V.t())
        if V_mirror is not None:
            sim = torch.maximum(sim, torch.mm(V_chunk, V_mirror.t()))

        col_g = torch.arange(N,          device=self.device, dtype=torch.int32)
        row_g = torch.arange(start, end, device=self.device, dtype=torch.int32)
        upper = col_g.unsqueeze(0) > row_g.unsqueeze(1)

        td         = torch.abs(T_chunk.unsqueeze(1) - times.unsqueeze(0))
        same_video = VI_chunk.unsqueeze(1) == video_ids.unsqueeze(0)
        time_ok    = ~same_video | (td >= min_gap)

        valid   = (sim >= threshold) & upper & time_ok
        indices = torch.nonzero(valid, as_tuple=False)
        scores  = sim[valid]

        idx_np = indices.cpu().numpy()
        scr_np = scores.cpu().numpy().astype(np.float32)

        del sim, upper, td, same_video, time_ok, valid, indices, scores
        if self.device == "cuda":
            torch.cuda.empty_cache()

        K = len(idx_np)
        if K == 0:
            return []

        if K > self.max_per_chunk:
            top_idx = np.argpartition(scr_np, -self.max_per_chunk)[-self.max_per_chunk:]
            idx_np  = idx_np[top_idx]
            scr_np  = scr_np[top_idx]

        i_global = idx_np[:, 0] + start
        j_global = idx_np[:, 1]

        if self.rescore:
            final_sims = _rescore_vec(
                sims       = scr_np,
                scales_i   = scales[i_global],
                scales_j   = scales[j_global],
                anchor_y_i = anchor_ys[i_global],
                anchor_y_j = anchor_ys[j_global],
            )
        else:
            final_sims = scr_np.copy()
            
            # Калибровка
            final_sims = _calibrate_sim(final_sims, threshold)

        rescore_threshold = threshold * 0.85
        keep = final_sims >= rescore_threshold
        if not keep.any():
            return []

        i_global   = i_global[keep]
        j_global   = j_global[keep]
        final_sims = final_sims[keep]
        scr_np     = scr_np[keep]

        chunk_matches: MatchList = []
        for k in range(len(i_global)):
            i  = int(i_global[k])
            j  = int(j_global[k])
            mi = meta[i]
            mj = meta[j]

            # ── Keypoints — берём из meta["kp"] ──────────────────────────
            # meta["kp"] добавляется в analysis_backend._flush_batch
            kp1 = mi.get("kp")
            kp2 = mj.get("kp")

            # Если вдруг ndarray — конвертируем в list
            if isinstance(kp1, np.ndarray):
                kp1 = kp1.tolist()
            if isinstance(kp2, np.ndarray):
                kp2 = kp2.tolist()

            match_dict: dict = {
                "m1_idx":    i,
                "m2_idx":    j,
                "t1":        float(mi["t"]),
                "t2":        float(mj["t"]),
                "f1":        mi.get("f", int(mi["t"] * 30)),
                "f2":        mj.get("f", int(mj["t"] * 30)),
                "v1_idx":    int(mi.get("video_idx", 0)),
                "v2_idx":    int(mj.get("video_idx", 0)),
                "sim":       float(final_sims[k]),
                "sim_raw":   float(scr_np[k]),
                "direction": mi.get("dir", "forward"),
            }

            if kp1 is not None:
                match_dict["kp1"] = kp1
            if kp2 is not None:
                match_dict["kp2"] = kp2

            chunk_matches.append(match_dict)

        return chunk_matches

    # ── Дедупликация ──────────────────────────────────────────────────────

    def _deduplicate(
        self,
        matches: MatchList,
        min_gap: float,
        V_cpu:   torch.Tensor,
    ) -> MatchList:
        if not matches:
            return []

        matches.sort(key=lambda x: x["sim"], reverse=True)

        dedup_gap = max(1.0, min_gap * 0.5)
        used: dict[int, list[float]] = {}

        def _occupied(v: int, t: float) -> bool:
            arr = used.get(v)
            if not arr:
                return False
            idx = bisect.bisect_left(arr, t)
            if idx < len(arr) and abs(arr[idx] - t) < dedup_gap:
                return True
            if idx > 0 and abs(arr[idx - 1] - t) < dedup_gap:
                return True
            return False

        def _occupy(v: int, t: float) -> None:
            if v not in used:
                used[v] = []
            bisect.insort(used[v], t)

        stage1: MatchList = []
        for m in matches:
            v1, v2 = m["v1_idx"], m["v2_idx"]
            t1, t2 = m["t1"],     m["t2"]

            if _occupied(v1, t1) or _occupied(v2, t2):
                continue

            stage1.append(m)
            _occupy(v1, t1)
            _occupy(v2, t2)

            if len(stage1) >= self.max_unique * 5:
                break

        if not stage1:
            return []

        # ── Этап 2: матричный фильтр дублей ──────────────────────────────
        m1_idx = torch.tensor([m["m1_idx"] for m in stage1], dtype=torch.long)
        m2_idx = torch.tensor([m["m2_idx"] for m in stage1], dtype=torch.long)

        pair_vecs = torch.cat([V_cpu[m1_idx], V_cpu[m2_idx]], dim=1)
        pair_vecs = F.normalize(pair_vecs, p=2, dim=1)

        accepted_vecs: list[torch.Tensor] = []
        unique: MatchList = []

        for k, m in enumerate(stage1):
            pv = pair_vecs[k]

            if accepted_vecs:
                acc_mat  = torch.stack(accepted_vecs)
                cos_vals = torch.mv(acc_mat, pv)
                if (cos_vals >= self.pair_dedup_cos).any():
                    continue

            unique.append(m)
            accepted_vecs.append(pv)

            if len(unique) >= self.max_unique:
                break

        return unique

    # ── Псевдонимы — обратная совместимость ──────────────────────────────

    @property
    def CHUNK_SIZE(self) -> int:
        return self.chunk_size

    @CHUNK_SIZE.setter
    def CHUNK_SIZE(self, v: int) -> None:
        self.chunk_size = v

    @property
    def CHUNK_OVERLAP(self) -> int:
        return self.chunk_overlap

    @CHUNK_OVERLAP.setter
    def CHUNK_OVERLAP(self, v: int) -> None:
        self.chunk_overlap = v

    @property
    def max_matches_per_chunk(self) -> int:
        return self.max_per_chunk

    @max_matches_per_chunk.setter
    def max_matches_per_chunk(self, v: int) -> None:
        self.max_per_chunk = v

    @property
    def max_total_matches(self) -> int:
        return self.max_total

    @max_total_matches.setter
    def max_total_matches(self, v: int) -> None:
        self.max_total = v

    @property
    def max_unique_results(self) -> int:
        return self.max_unique

    @max_unique_results.setter
    def max_unique_results(self, v: int) -> None:
        self.max_unique = v