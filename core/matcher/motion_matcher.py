#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/matcher/motion_matcher.py
MotionMatcher v13 — Appearance-aware matching with strict deduplication.

Критические улучшения v13:
1. Добавлено сравнение пропорций тела (appearance similarity)
2. Жёсткая дедупликация: если разница < 3 сек в одном видео → дубликат
3. Temporal grouping: группировка близких совпадений
4. Увеличены веса appearance: 0.25 от финального скора
5. Уменьшен порог схожести: 0.68 (было 0.72)
6. Улучшена motion consistency с проверкой направления
7. Добавлена проверка на смену сцены (резкое изменение anchor_y)
"""

from __future__ import annotations

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

from core.matcher.pose_processor import mirror_vectors, COCO_N_KPS, ANCHOR_KPS_ARR, compare_body_proportions

log = logging.getLogger(__name__)

# ── Веса метрик ───────────────────────────────────────────────────────────────
WEIGHT_COSINE = 0.45
WEIGHT_MOTION = 0.30
WEIGHT_APPEARANCE = 0.25

# ── Параметры USearch ──────────────────────────────────────────────────────────
K_FAISS = 300

# ── Пороги ─────────────────────────────────────────────────────────────────────
DEFAULT_SIM_THRESHOLD = 0.68
DEFAULT_GOOD_THRESH = 0.75

# ── Дедупликация ───────────────────────────────────────────────────────────────
DEFAULT_MIN_MATCH_GAP = 1.0
SAME_VIDEO_MIN_GAP = 4.0
CROSS_VIDEO_MIN_GAP = 1.0
DUPLICATE_THRESHOLD = 3.0
SCENE_CHANGE_ANCHOR_Y_DIFF = 0.15
DEFAULT_JUNK_RATIO = 0.10
DEFAULT_MAX_UNIQUE = 5000

# ── Структура кандидатов ───────────────────────────────────────────────────────
_MATCH_DTYPE = np.dtype([
    ("m1_idx", np.int32),
    ("m2_idx", np.int32),
    ("cosine_sim", np.float32),
    ("sim", np.float32),
    ("dtw_sim", np.float32),
    ("motion_sim", np.float32),
    ("appearance_sim", np.float32),
    ("t1", np.float32),
    ("t2", np.float32),
    ("f1", np.int32),
    ("f2", np.int32),
    ("v1_idx", np.int32),
    ("v2_idx", np.int32),
    ("final_sim", np.float32),
    ("scale_penalty", np.float32),
    ("anchor_penalty", np.float32),
])


def _build_meta_arrays(meta: list[dict]) -> dict[str, np.ndarray]:
    n = len(meta)
    fps_aprx = 30.0
    times = np.empty(n, dtype=np.float32)
    frames = np.empty(n, dtype=np.int32)
    video_idx = np.zeros(n, dtype=np.int32)
    direction = np.empty(n, dtype=object)
    scale = np.ones(n, dtype=np.float32)
    anchor_y = np.zeros(n, dtype=np.float32)

    for i, m in enumerate(meta):
        t = float(m.get("t", 0.0))
        times[i] = t
        frames[i] = int(m.get("f", t * fps_aprx))
        video_idx[i] = int(m.get("video_idx", 0))
        direction[i] = m.get("dir", "unknown")
        scale[i] = float(m.get("scale", 1.0))
        anchor_y[i] = float(m.get("anchor_y", 0.5))

    return {
        "times": times,
        "frames": frames,
        "video_idx": video_idx,
        "direction": direction,
        "scale": scale,
        "anchor_y": anchor_y,
    }


class MotionMatcher:
    DEFAULT_CHUNK_SIZE = 3000
    DEFAULT_CHUNK_OVERLAP = 300
    DEFAULT_MAX_PER_CHUNK = 2_000_000
    DEFAULT_MAX_TOTAL = 30_000_000
    
    def __init__(self, device: str = "cuda") -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.k_faiss = K_FAISS
        
        self.chunk_size = self.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = self.DEFAULT_CHUNK_OVERLAP
        self.max_per_chunk = self.DEFAULT_MAX_PER_CHUNK
        self.max_total = self.DEFAULT_MAX_TOTAL
        self.max_unique = 2000
        self.min_match_gap = DEFAULT_MIN_MATCH_GAP
        self.junk_ratio = DEFAULT_JUNK_RATIO
        self.good_threshold = DEFAULT_GOOD_THRESH
        self.sim_threshold = DEFAULT_SIM_THRESHOLD

        self._is_cuda: bool = (self.device == "cuda")
        self._usearch_index: Optional[Index] = None
        
        self._poses_meta_cached: list[dict] = []
        self._poses_tensor_cached: Optional[torch.Tensor] = None

    def find_matches(
        self,
        poses_tensor: torch.Tensor,
        poses_meta: list[dict],
        threshold: float = DEFAULT_SIM_THRESHOLD,
        min_gap: float = DEFAULT_MIN_MATCH_GAP,
        use_mirror: bool = False,
    ) -> list[dict]:
        if poses_tensor is None or len(poses_tensor) < 10:
            return []

        if len(poses_meta) != len(poses_tensor):
            raise ValueError(f"poses_tensor({len(poses_tensor)}) != poses_meta({len(poses_meta)})")

        n = len(poses_tensor)
        log.info(
            "[Matcher] N=%d | thr=%.2f | gap=%.1fs | mirror=%s | k_faiss=%d",
            n, threshold, min_gap, use_mirror, self.k_faiss
        )

        V = poses_tensor.to(dtype=torch.float32, device=self.device)
        V = V.view(n, -1)
        V = F.normalize(V, p=2, dim=1)

        V_mirror: Optional[torch.Tensor] = None
        if use_mirror:
            V_mirror = F.normalize(mirror_vectors(V), p=2, dim=1)

        meta_arrs = _build_meta_arrays(poses_meta)
        times_np = meta_arrs["times"]
        frames_np = meta_arrs["frames"]
        vididx_np = meta_arrs["video_idx"]
        direction_np = meta_arrs["direction"]
        scale_np = meta_arrs["scale"]
        anchor_y_np = meta_arrs["anchor_y"]

        t_start = time.time()
        candidates = self._find_candidates_usearch(
            V, V_mirror, times_np, frames_np, vididx_np,
            direction_np, scale_np, anchor_y_np,
            threshold, min_gap
        )
        t_search = time.time() - t_start
        log.info("[Matcher] Поиск кандидатов: %.2fс", t_search)

        if candidates is None:
            log.info("[Matcher] Совпадений не найдено (USearch).")
            return []

        log.info("[Matcher] Найдено кандидатов через USearch: %d", len(candidates))

        final_matches = self._build_matches_from_candidates(candidates, poses_meta)

        if len(final_matches) > 0:
            final_matches = self._dedup_pairs_torch(final_matches, n)

        log.info("[Matcher] После дедупликации пар: %d", len(final_matches))

        # НОВОЕ: жёсткая дедупликация (удаление одинаковых сцен)
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
        t_dedup = time.time() - t_dedup
        log.info("[Matcher] Финальная дедупликация: %.2fс → %d совпадений", t_dedup, len(deduplicated))

        return deduplicated

    def _find_candidates_usearch(
        self,
        V: torch.Tensor,
        V_mirror: Optional[torch.Tensor],
        times_np: np.ndarray,
        frames_np: np.ndarray,
        vididx_np: np.ndarray,
        direction_np: np.ndarray,
        scale_np: np.ndarray,
        anchor_y_np: np.ndarray,
        threshold: float,
        min_gap: float,
    ) -> Optional[np.ndarray]:
        if not USEARCH_AVAILABLE:
            log.warning("[Matcher] USearch недоступен, используем fallback.")
            return self._find_candidates_fallback(
                V, V_mirror, times_np, frames_np, vididx_np,
                direction_np, scale_np, anchor_y_np, threshold, min_gap
            )

        n = V.shape[0]
        dim = V.shape[1]

        self._usearch_index = Index(ndim=dim, metric='cos', dtype='float32')
        V_cpu = V.cpu().numpy()
        keys = np.arange(n, dtype=np.int64)
        self._usearch_index.add(keys, V_cpu)

        matches = self._usearch_index.search(V_cpu, count=self.k_faiss)

        if matches.keys is None or matches.distances is None:
            return None

        keys_arr = matches.keys
        dists_arr = matches.distances

        n_neighbors = self.k_faiss
        
        row_indices = np.repeat(np.arange(n), n_neighbors)
        col_indices = keys_arr.ravel()
        distances = dists_arr.ravel()
        
        similarities = 1.0 - distances
        
        valid_mask = col_indices > row_indices
        valid_mask &= (similarities >= threshold)
        
        t1 = times_np[row_indices]
        t2 = times_np[col_indices]
        v1 = vididx_np[row_indices]
        v2 = vididx_np[col_indices]
        
        same_video = (v1 == v2)
        time_gap_ok = np.ones_like(valid_mask, dtype=bool)
        time_gap_ok[same_video] = np.abs(t1[same_video] - t2[same_video]) >= min_gap
        valid_mask &= time_gap_ok
        
        dir1 = direction_np[row_indices]
        dir2 = direction_np[col_indices]
        
        direction_mask = np.ones_like(valid_mask, dtype=bool)
        for idx in np.where(valid_mask)[0]:
            d1 = dir1[idx]
            d2 = dir2[idx]
            
            if d1 != "unknown" and d2 != "unknown":
                if (d1 in ("forward", "back") and d2 in ("left", "right")):
                    direction_mask[idx] = False
                elif (d2 in ("forward", "back") and d1 in ("left", "right")):
                    direction_mask[idx] = False
                elif d1 != d2 and d1 in ("left", "right") and d2 in ("left", "right"):
                    direction_mask[idx] = False
        
        valid_mask &= direction_mask
        
        # НОВОЕ: проверка на смену сцены (резкое изменение anchor_y)
        ay1 = anchor_y_np[row_indices]
        ay2 = anchor_y_np[col_indices]
        anchor_y_diff = np.abs(ay1 - ay2)
        scene_change_mask = anchor_y_diff < SCENE_CHANGE_ANCHOR_Y_DIFF
        valid_mask &= scene_change_mask
        
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return None

        max_candidates = min(self.max_per_chunk, len(valid_indices))
        
        if len(valid_indices) > max_candidates:
            valid_sims = similarities[valid_indices]
            top_k_idx = np.argpartition(-valid_sims, max_candidates)[:max_candidates]
            valid_indices = valid_indices[top_k_idx]

        k = len(valid_indices)
        arr = np.empty(k, dtype=_MATCH_DTYPE)
        
        arr["m1_idx"] = row_indices[valid_indices].astype(np.int32)
        arr["m2_idx"] = col_indices[valid_indices].astype(np.int32)
        arr["cosine_sim"] = similarities[valid_indices].astype(np.float32)
        arr["sim"] = arr["cosine_sim"]
        arr["dtw_sim"] = np.zeros(k, dtype=np.float32)
        arr["motion_sim"] = np.zeros(k, dtype=np.float32)
        arr["appearance_sim"] = np.zeros(k, dtype=np.float32)
        arr["t1"] = times_np[arr["m1_idx"]]
        arr["t2"] = times_np[arr["m2_idx"]]
        arr["f1"] = frames_np[arr["m1_idx"]]
        arr["f2"] = frames_np[arr["m2_idx"]]
        arr["v1_idx"] = vididx_np[arr["m1_idx"]]
        arr["v2_idx"] = vididx_np[arr["m2_idx"]]

        return arr

    def _find_candidates_fallback(
        self,
        V: torch.Tensor,
        V_mirror: Optional[torch.Tensor],
        times_np: np.ndarray,
        frames_np: np.ndarray,
        vididx_np: np.ndarray,
        direction_np: np.ndarray,
        scale_np: np.ndarray,
        anchor_y_np: np.ndarray,
        threshold: float,
        min_gap: float,
    ) -> Optional[np.ndarray]:
        return self._process_chunk(
            V, V_mirror, times_np, frames_np, vididx_np,
            direction_np, scale_np, anchor_y_np,
            0, len(V), threshold, min_gap
        )

    def _process_chunk(
        self,
        V: torch.Tensor,
        V_mirror: Optional[torch.Tensor],
        times_np: np.ndarray,
        frames_np: np.ndarray,
        vididx_np: np.ndarray,
        direction_np: np.ndarray,
        scale_np: np.ndarray,
        anchor_y_np: np.ndarray,
        start: int,
        end: int,
        threshold: float,
        min_gap: float,
    ) -> Optional[np.ndarray]:
        n = len(V)
        V_chunk = V[start:end]
        chunk_sz = end - start

        sim = torch.mm(V_chunk, V.t())
        if V_mirror is not None:
            sim_m = torch.mm(V_mirror[start:end], V.t())
            torch.maximum(sim, sim_m, out=sim)
            del sim_m

        local_idx = torch.arange(chunk_sz, device=self.device)
        global_row = (start + local_idx).unsqueeze(1)
        col_range = torch.arange(n, device=self.device).unsqueeze(0)
        upper_mask = (col_range > global_row) & (col_range >= start)

        valid_mask = (sim >= threshold) & upper_mask
        del upper_mask, local_idx, global_row, col_range

        cand_rows, cand_cols = torch.where(valid_mask)
        del valid_mask

        if len(cand_rows) == 0:
            del sim, cand_rows, cand_cols
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
        keep_mask = np.ones_like(rows_np, dtype=bool)
        keep_mask[same_video] = np.abs(t1[same_video] - t2[same_video]) >= min_gap

        if not keep_mask.any():
            return None

        rows_np = rows_np[keep_mask]
        cols_np = cols_np[keep_mask]
        sims_np = sims_np[keep_mask]
        t1 = t1[keep_mask]
        t2 = t2[keep_mask]

        k = len(rows_np)
        arr = np.empty(k, dtype=_MATCH_DTYPE)
        arr["m1_idx"] = rows_np
        arr["m2_idx"] = cols_np
        arr["cosine_sim"] = sims_np
        arr["sim"] = sims_np
        arr["dtw_sim"] = np.zeros(k, dtype=np.float32)
        arr["motion_sim"] = np.zeros(k, dtype=np.float32)
        arr["appearance_sim"] = np.zeros(k, dtype=np.float32)
        arr["t1"] = t1
        arr["t2"] = t2
        arr["f1"] = frames_np[rows_np]
        arr["f2"] = frames_np[cols_np]
        arr["v1_idx"] = vididx_np[rows_np]
        arr["v2_idx"] = vididx_np[cols_np]

        return arr

    def _build_matches_from_candidates(
        self,
        candidates: np.ndarray,
        poses_meta: list[dict],
    ) -> list[dict]:
        motion_scores = self._compute_motion_consistency_scores(candidates, poses_meta)
        
        # НОВОЕ: сравнение пропорций тела
        appearance_scores = np.zeros(len(candidates), dtype=np.float32)
        
        for i in range(len(candidates)):
            m1 = int(candidates["m1_idx"][i])
            m2 = int(candidates["m2_idx"][i])
            
            props1 = poses_meta[m1].get("body_proportions")
            props2 = poses_meta[m2].get("body_proportions")
            
            if props1 and props2:
                appearance_scores[i] = compare_body_proportions(props1, props2)
            else:
                appearance_scores[i] = 0.5
        
        candidates["appearance_sim"] = appearance_scores
        
        meta_arrs = _build_meta_arrays(poses_meta)
        scale = meta_arrs["scale"]
        anchor_y = meta_arrs["anchor_y"]
        
        scale_penalty = np.zeros(len(candidates), dtype=np.float32)
        anchor_penalty = np.zeros(len(candidates), dtype=np.float32)
        
        for i in range(len(candidates)):
            m1 = int(candidates["m1_idx"][i])
            m2 = int(candidates["m2_idx"][i])
            
            s1 = scale[m1]
            s2 = scale[m2]
            a1 = anchor_y[m1]
            a2 = anchor_y[m2]
            
            if s1 > 0:
                scale_penalty[i] = abs(np.log(s1 / s2))
            else:
                scale_penalty[i] = 1.0
            
            anchor_penalty[i] = abs(a1 - a2)
        
        cosine_sim = candidates["cosine_sim"]
        motion_sim = motion_scores
        appearance_sim = appearance_scores
        
        # НОВОЕ: веса с учётом appearance
        final_sim = (
            WEIGHT_COSINE * cosine_sim +
            WEIGHT_MOTION * motion_sim +
            WEIGHT_APPEARANCE * appearance_sim
        )
        
        final_sim = final_sim - scale_penalty * 0.5 - anchor_penalty * 0.3
        final_sim = np.clip(final_sim, 0, 1)
        
        candidates["final_sim"] = final_sim
        candidates["scale_penalty"] = scale_penalty
        candidates["anchor_penalty"] = anchor_penalty
        candidates["motion_sim"] = motion_scores
        
        return [
            {
                "m1_idx": int(r["m1_idx"]),
                "m2_idx": int(r["m2_idx"]),
                "t1": float(r["t1"]),
                "t2": float(r["t2"]),
                "f1": int(r["f1"]),
                "f2": int(r["f2"]),
                "v1_idx": int(r["v1_idx"]),
                "v2_idx": int(r["v2_idx"]),
                "sim": float(r["final_sim"]),
                "sim_raw": float(r["cosine_sim"]),
                "cosine_sim": float(r["cosine_sim"]),
                "dtw_sim": float(r["dtw_sim"]),
                "motion_sim": float(r["motion_sim"]),
                "appearance_sim": float(r["appearance_sim"]),
                "direction": poses_meta[int(r["m1_idx"])].get("dir", "forward"),
                "scale1": float(poses_meta[int(r["m1_idx"])].get("scale", 1.0)),
                "scale2": float(poses_meta[int(r["m2_idx"])].get("scale", 1.0)),
                "anchor_y1": float(poses_meta[int(r["m1_idx"])].get("anchor_y", 0.5)),
                "anchor_y2": float(poses_meta[int(r["m2_idx"])].get("anchor_y", 0.5)),
                "kp1": poses_meta[int(r["m1_idx"])].get("kp"),
                "kp2": poses_meta[int(r["m2_idx"])].get("kp"),
                "scale_penalty": float(r["scale_penalty"]),
                "anchor_penalty": float(r["anchor_penalty"]),
            }
            for r in candidates
        ]

    def _compute_motion_consistency_scores(
        self,
        candidates: np.ndarray,
        poses_meta: list[dict],
    ) -> np.ndarray:
        n_cands = len(candidates)
        motion_scores = np.ones(n_cands, dtype=np.float32) * 0.5
        
        for i in range(n_cands):
            m1 = int(candidates["m1_idx"][i])
            m2 = int(candidates["m2_idx"][i])
            
            kp1 = np.array(poses_meta[m1]["kp"], dtype=np.float32).flatten()
            kp2 = np.array(poses_meta[m2]["kp"], dtype=np.float32).flatten()

            prev1 = np.array(poses_meta[m1 - 1]["kp"], dtype=np.float32).flatten() if m1 > 0 else kp1
            next1 = np.array(poses_meta[m1 + 1]["kp"], dtype=np.float32).flatten() if m1 < len(poses_meta) - 1 else kp1

            prev2 = np.array(poses_meta[m2 - 1]["kp"], dtype=np.float32).flatten() if m2 > 0 else kp2
            next2 = np.array(poses_meta[m2 + 1]["kp"], dtype=np.float32).flatten() if m2 < len(poses_meta) - 1 else kp2
            
            delta1_next = next1 - kp1
            delta1_prev = kp1 - prev1
            
            delta2_next = next2 - kp2
            delta2_prev = kp2 - prev2
            
            norm1_next = np.linalg.norm(delta1_next) + 1e-6
            norm1_prev = np.linalg.norm(delta1_prev) + 1e-6
            norm_kp1 = np.linalg.norm(kp1) + 1e-6
            
            if norm1_next > 1e-6 and norm1_prev > 1e-6:
                sim1_next = np.dot(delta1_next, kp1) / (norm1_next * norm_kp1)
                sim1_prev = np.dot(delta1_prev, kp1) / (norm1_prev * norm_kp1)
                motion_sim1 = (sim1_next + sim1_prev) / 2.0
            else:
                motion_sim1 = 0.5
            
            norm2_next = np.linalg.norm(delta2_next) + 1e-6
            norm2_prev = np.linalg.norm(delta2_prev) + 1e-6
            norm_kp2 = np.linalg.norm(kp2) + 1e-6
            
            if norm2_next > 1e-6 and norm2_prev > 1e-6:
                sim2_next = np.dot(delta2_next, kp2) / (norm2_next * norm_kp2)
                sim2_prev = np.dot(delta2_prev, kp2) / (norm2_prev * norm_kp2)
                motion_sim2 = (sim2_next + sim2_prev) / 2.0
            else:
                motion_sim2 = 0.5
            
            motion_scores[i] = (motion_sim1 + motion_sim2) / 2.0
        
        return motion_scores

    def _remove_duplicates_strict(self, matches: list[dict]) -> list[dict]:
        """
        Жёсткая дедупликация: если разница < DUPLICATE_THRESHOLD сек → дубликат.
        
        Оставляет только лучшее совпадение из группы дубликатов.
        """
        if not matches:
            return []
        
        by_video: dict[tuple[int, int], list[dict]] = {}
        for m in matches:
            v1 = m["v1_idx"]
            v2 = m["v2_idx"]
            key = (min(v1, v2), max(v1, v2))
            if key not in by_video:
                by_video[key] = []
            by_video[key].append(m)
        
        result = []
        
        for video_pair, group in by_video.items():
            group_sorted = sorted(group, key=lambda x: x["sim"], reverse=True)
            
            kept = []
            
            for m in group_sorted:
                t1, t2 = m["t1"], m["t2"]
                
                is_duplicate = False
                for kept_m in kept:
                    kt1, kt2 = kept_m["t1"], kept_m["t2"]
                    
                    if abs(t1 - kt1) < DUPLICATE_THRESHOLD and abs(t2 - kt2) < DUPLICATE_THRESHOLD:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    kept.append(m)
            
            result.extend(kept)
        
        return result

    def _dedup_pairs_torch(self, matches: list[dict], n: int) -> list[dict]:
        if len(matches) == 0:
            return matches

        device = self.device
        m1_arr = torch.tensor([m["m1_idx"] for m in matches], device=device, dtype=torch.int64)
        m2_arr = torch.tensor([m["m2_idx"] for m in matches], device=device, dtype=torch.int64)
        sim_arr = torch.tensor([m["sim"] for m in matches], device=device, dtype=torch.float32)

        keys = m1_arr * (n + 1) + m2_arr
        _, sorted_idx = torch.sort(keys * 1000000 - sim_arr * 1000000)

        keys_sorted = keys[sorted_idx]
        unique_mask = torch.ones(len(keys_sorted), dtype=torch.bool, device=device)
        unique_mask[1:] = keys_sorted[1:] != keys_sorted[:-1]

        unique_idx = sorted_idx[unique_mask]
        return [matches[idx] for idx in unique_idx.tolist()]

    def _deduplicate(
        self,
        matches: list[dict],
        poses_meta: list[dict],
        min_gap: float,
    ) -> list[dict]:
        if len(matches) == 0:
            return []

        t_start = time.time()
        
        matches_sorted = sorted(matches, key=lambda m: m["sim"], reverse=True)

        good = [m for m in matches_sorted if m["sim"] >= self.good_threshold]
        junk = [m for m in matches_sorted if m["sim"] < self.good_threshold]

        junk_take = int(len(junk) * self.junk_ratio)
        candidates = good + junk[:junk_take]

        log.info(
            "[Matcher] good=%d junk=%d junk_taken=%d candidates=%d",
            len(good), len(junk), junk_take, len(candidates)
        )

        SAME_VIDEO_GAP = SAME_VIDEO_MIN_GAP
        CROSS_VIDEO_GAP = CROSS_VIDEO_MIN_GAP
        max_uniq = min(self.max_unique, int(len(poses_meta) * 0.05))

        used_times: dict[int, list] = {}

        def _is_close(vid: int, t: float, gap: float) -> bool:
            arr_ = used_times.get(vid)
            if arr_ is None:
                return False
            import bisect
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
                import bisect
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

        t_elapsed = time.time() - t_start
        log.info("[Matcher] Уникальных: %d (за %.2fс)", len(unique_structs), t_elapsed)

        if not unique_structs:
            return []

        return unique_structs

    def _validate_overlap(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            log.warning(
                "[Matcher] overlap(%d) >= chunk_size(%d) → сброс",
                self.chunk_overlap, self.chunk_size
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

        self.chunk_size = _int("CHUNK_SIZE", self.chunk_size)
        self.chunk_overlap = _int("CHUNK_OVERLAP", self.chunk_overlap, 0)
        self.max_per_chunk = _int("max_matches_per_chunk", self.max_per_chunk)
        self.max_total = _int("max_total_matches", self.max_total)
        self.max_unique = _int("max_unique_results", self.max_unique)
        self.min_match_gap = _float("MIN_MATCH_GAP", self.min_match_gap)
        self.junk_ratio = _float("junk_ratio", self.junk_ratio)
        self.good_threshold = _float("good_threshold", self.good_threshold)
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

        self.chunk_size = _gi("chunk_size", self.chunk_size)
        self.chunk_overlap = _gi("chunk_overlap", self.chunk_overlap, 0)
        self.max_unique = _gi("max_unique_results", self.max_unique)
        self.good_threshold = _gf("good_threshold", self.good_threshold)
        self.min_match_gap = _gf("match_gap", self.min_match_gap)
        self.junk_ratio = _gf("junk_ratio", self.junk_ratio)
        self.k_faiss = _gi("k_faiss", self.k_faiss)
        self.sim_threshold = _gf("sim_threshold", self.sim_threshold)
        self._validate_overlap()