#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

from core.matcher.pose_processor import (
    compute_body_proportions,
    compute_body_proportions_batch,
    compare_body_proportions,
)


class PhotoMatcher:
    APPEARANCE_THRESHOLD = 0.65
    POSE_THRESHOLD_STRICT = 0.70
    POSE_THRESHOLD_RELAXED = 0.55

    def __init__(self, conf_threshold: float = 0.24) -> None:
        self._conf = conf_threshold
        self._ref_vecs: list[np.ndarray] = []
        self._ref_raw_kps: list[np.ndarray] = []
        self._ref_body_props: list[dict] = []

        # (R, D) матрица референсных векторов (для batch-mm)
        self._ref_matrix: Optional[np.ndarray] = None

        # Кэш по hash от kp
        self._cache_vecs: dict[str, np.ndarray] = {}
        self._cache_props: dict[str, dict] = {}
        self._cache_max = 200_000  # лимит чтобы не съесть всю память

    # ------------------------------------------------------------------
    # Загрузка референсов
    # ------------------------------------------------------------------

    def load_references(self, photo_paths: list[str], yolo) -> bool:
        import cv2  # lazy

        self._ref_vecs.clear()
        self._ref_raw_kps.clear()
        self._ref_body_props.clear()
        self._ref_matrix = None

        for path in photo_paths[:3]:
            if not Path(path).is_file():
                print(f"[PhotoMatcher] файл не найден: {path}")
                continue

            img = cv2.imread(path)
            if img is None:
                print(f"[PhotoMatcher] не удалось прочитать: {path}")
                continue

            results = yolo.detect_batch([img])
            if not results or results[0] is None:
                print(f"[PhotoMatcher] поза не найдена: {path}")
                continue

            pose = results[0]
            vec = self._pose_to_vec(pose)
            if vec is None:
                print(f"[PhotoMatcher] не удалось векторизовать: {path}")
                continue

            self._ref_vecs.append(vec)

            kp_raw = pose.get("kp") or pose.get("keypoints")
            if kp_raw is not None:
                try:
                    kp = self._raw_to_kp(kp_raw)
                    if kp is not None:
                        self._ref_raw_kps.append(kp)
                        props = compute_body_proportions(kp, self._conf)
                        if props.get("valid"):
                            self._ref_body_props.append(props)
                            print(
                                f"[PhotoMatcher] ✓ пропорции: "
                                f"leg/torso={props['leg_to_torso']:.2f}"
                            )
                except Exception as e:
                    print(f"[PhotoMatcher] ошибка kp: {e}")

            print(f"[PhotoMatcher] ✓ поза извлечена: {Path(path).name}")

        if self._ref_vecs:
            # Стакаем в матрицу для batch-сравнения
            try:
                self._ref_matrix = np.stack(self._ref_vecs, axis=0).astype(np.float32)
            except Exception:
                self._ref_matrix = None
            print(f"[PhotoMatcher] загружено референсов: {len(self._ref_vecs)}")
            print(f"[PhotoMatcher] пропорций тела: {len(self._ref_body_props)}")

        return len(self._ref_vecs) > 0

    # ------------------------------------------------------------------
    # Фильтрация кадров
    # ------------------------------------------------------------------

    def filter_poses_by_reference(
        self,
        frames_data: list[dict],
        threshold: float = 0.55,
    ) -> list[dict]:
        if not self._ref_vecs or self._ref_matrix is None:
            return frames_data

        # ── Этап 1: собираем векторы и пропорции батчем ─────────────
        # Используем стабильный кэш по хэшу kp.
        valid_indices: list[int] = []
        vecs_list: list[np.ndarray] = []
        props_list: list[Optional[dict]] = []

        for idx, frame in enumerate(frames_data):
            kp_raw = frame.get("kp")
            if kp_raw is None:
                continue

            cache_key = self._hash_kp(kp_raw)
            vec = self._cache_vecs.get(cache_key)
            if vec is None:
                vec = self._kp_raw_to_vec(kp_raw)
                if vec is None:
                    continue
                if len(self._cache_vecs) < self._cache_max:
                    self._cache_vecs[cache_key] = vec

            props = self._cache_props.get(cache_key)
            if props is None:
                kp = self._raw_to_kp(kp_raw)
                if kp is not None:
                    props = compute_body_proportions(kp, self._conf)
                    if props.get("valid") and len(self._cache_props) < self._cache_max:
                        self._cache_props[cache_key] = props

            valid_indices.append(idx)
            vecs_list.append(vec)
            props_list.append(props if (props and props.get("valid")) else None)

        if not vecs_list:
            return []

        # ── Этап 2: батч-сходство по позам через матричное умножение ─
        vec_matrix = np.stack(vecs_list, axis=0).astype(np.float32)  # (N, D)
        # cosine = clamp01(V @ R.T).max(1) — векторы уже нормализованы
        sims = vec_matrix @ self._ref_matrix.T                        # (N, R)
        np.clip(sims, 0.0, 1.0, out=sims)
        pose_scores = sims.max(axis=1)                                # (N,)

        # ── Этап 3: appearance scores ────────────────────────────────
        # Здесь приходится прогнать Python-цикл, т.к. compare_body_proportions
        # — словарная функция; но мы её ускоряем за счёт пропуска None.
        n = len(valid_indices)
        appearance_scores = np.zeros(n, dtype=np.float32)
        if self._ref_body_props:
            ref_props = self._ref_body_props
            for i, p in enumerate(props_list):
                if p is None:
                    continue
                best = 0.0
                for rp in ref_props:
                    s = compare_body_proportions(p, rp)
                    if s > best:
                        best = s
                appearance_scores[i] = best

        # ── Этап 4: применяем двухэтапный фильтр векторно ────────────
        appearance_ok = (appearance_scores == 0.0) | (appearance_scores >= self.APPEARANCE_THRESHOLD)
        thresholds = np.where(
            appearance_scores > self.APPEARANCE_THRESHOLD,
            self.POSE_THRESHOLD_RELAXED,
            self.POSE_THRESHOLD_STRICT,
        ).astype(np.float32)
        pose_ok = pose_scores >= thresholds
        keep_mask = appearance_ok & pose_ok

        result: list[dict] = []
        skipped_appearance = int((~appearance_ok).sum())
        skipped_pose = int((appearance_ok & ~pose_ok).sum())

        for j, keep in enumerate(keep_mask):
            if not keep:
                continue
            frame = frames_data[valid_indices[j]]
            frame["photo_sim"] = float(pose_scores[j])
            frame["appearance_sim"] = float(appearance_scores[j])
            result.append(frame)

        print(
            f"[PhotoMatcher] кадры: {len(frames_data)} → {len(result)} "
            f"(внешность: -{skipped_appearance}, поза: -{skipped_pose})"
        )
        return result

    # ------------------------------------------------------------------
    # Фильтрация матчей — также векторно
    # ------------------------------------------------------------------

    def filter_matches(
        self,
        matches: list[dict],
        threshold: float = 0.70,
    ) -> list[dict]:
        if not self._ref_vecs or self._ref_matrix is None or not matches:
            return matches

        # Готовим векторы для kp1 и kp2
        K = len(matches)
        # Для каждой стороны соберём (K,D) или маски валидности
        vecs1 = np.zeros((K, self._ref_matrix.shape[1]), dtype=np.float32)
        vecs2 = np.zeros((K, self._ref_matrix.shape[1]), dtype=np.float32)
        mask1 = np.zeros(K, dtype=bool)
        mask2 = np.zeros(K, dtype=bool)
        props1: list[Optional[dict]] = [None] * K
        props2: list[Optional[dict]] = [None] * K

        for i, m in enumerate(matches):
            for side, vecs, mask, props in (
                ("kp1", vecs1, mask1, props1),
                ("kp2", vecs2, mask2, props2),
            ):
                kp_raw = m.get(side)
                if kp_raw is None:
                    continue
                key = self._hash_kp(kp_raw)
                v = self._cache_vecs.get(key)
                if v is None:
                    v = self._kp_raw_to_vec(kp_raw)
                    if v is None:
                        continue
                    if len(self._cache_vecs) < self._cache_max:
                        self._cache_vecs[key] = v
                vecs[i] = v
                mask[i] = True
                if self._ref_body_props:
                    p = self._cache_props.get(key)
                    if p is None:
                        kp = self._raw_to_kp(kp_raw)
                        if kp is not None:
                            p = compute_body_proportions(kp, self._conf)
                            if p.get("valid") and len(self._cache_props) < self._cache_max:
                                self._cache_props[key] = p
                    if p and p.get("valid"):
                        props[i] = p

        # Pose similarities — батч
        sim1 = np.zeros(K, dtype=np.float32)
        sim2 = np.zeros(K, dtype=np.float32)
        if mask1.any():
            s = vecs1[mask1] @ self._ref_matrix.T
            np.clip(s, 0.0, 1.0, out=s)
            sim1[mask1] = s.max(axis=1)
        if mask2.any():
            s = vecs2[mask2] @ self._ref_matrix.T
            np.clip(s, 0.0, 1.0, out=s)
            sim2[mask2] = s.max(axis=1)
        max_pose = np.maximum(sim1, sim2)

        # Appearance — Python-cycle (мало рефов)
        max_appearance = np.zeros(K, dtype=np.float32)
        if self._ref_body_props:
            ref_props = self._ref_body_props
            for i in range(K):
                best = 0.0
                for p in (props1[i], props2[i]):
                    if p is None:
                        continue
                    for rp in ref_props:
                        s = compare_body_proportions(p, rp)
                        if s > best:
                            best = s
                max_appearance[i] = best

        appearance_ok = (max_appearance == 0.0) | (max_appearance >= self.APPEARANCE_THRESHOLD)
        thresholds = np.where(
            max_appearance > self.APPEARANCE_THRESHOLD,
            self.POSE_THRESHOLD_RELAXED,
            self.POSE_THRESHOLD_STRICT,
        ).astype(np.float32)
        pose_ok = max_pose >= thresholds
        keep_mask = appearance_ok & pose_ok

        result: list[dict] = []
        skipped = int((~keep_mask).sum())
        for i, keep in enumerate(keep_mask):
            if not keep:
                continue
            m = matches[i]
            m["photo_sim"] = float(max_pose[i])
            m["appearance_sim"] = float(max_appearance[i])
            result.append(m)

        print(f"[PhotoMatcher] матчи: {len(matches)} → {len(result)} (отсев={skipped})")
        return result

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------

    def best_ref_sim(self, pose_dict: dict) -> float:
        if not self._ref_vecs or self._ref_matrix is None:
            return 0.0
        vec = self._pose_to_vec(pose_dict)
        if vec is None:
            return 0.0
        sims = self._ref_matrix @ vec
        return float(np.clip(sims.max(), 0.0, 1.0))

    @staticmethod
    def _hash_kp(kp_raw) -> str:
        """Стабильный хэш для кэша. Для numpy — tobytes(); для list — repr."""
        try:
            if isinstance(kp_raw, np.ndarray):
                return hashlib.md5(kp_raw.tobytes()).hexdigest()
            arr = np.asarray(kp_raw, dtype=np.float32)
            return hashlib.md5(arr.tobytes()).hexdigest()
        except Exception:
            return repr(kp_raw)[:128]

    def _raw_to_kp(self, kp_raw) -> Optional[np.ndarray]:
        try:
            kp = np.array(kp_raw, dtype=float)
            if kp.ndim == 1:
                n = len(kp)
                if n == 51:
                    kp = kp.reshape(17, 3)
                elif n == 34:
                    kp = np.hstack([kp.reshape(17, 2), np.ones((17, 1))])
                else:
                    return None
            if kp.ndim == 2 and kp.shape[1] == 2:
                kp = np.hstack([kp, np.ones((len(kp), 1))])
            if kp.ndim != 2 or kp.shape[1] < 3:
                return None
            return kp
        except Exception:
            return None

    def _pose_to_vec(self, pose: dict) -> Optional[np.ndarray]:
        kp_raw = pose.get("kp") or pose.get("keypoints")
        if kp_raw is None:
            return None
        return self._kp_raw_to_vec(kp_raw)

    def _kp_raw_to_vec(self, kp_raw) -> Optional[np.ndarray]:
        try:
            kp = np.array(kp_raw, dtype=float)

            if kp.ndim == 1:
                n = len(kp)
                if n == 51:
                    kp = kp.reshape(17, 3)
                elif n == 34:
                    kp = np.hstack([kp.reshape(17, 2), np.ones((17, 1))])
                else:
                    return None
            elif kp.ndim == 2:
                if kp.shape[1] == 2:
                    kp = np.hstack([kp, np.ones((kp.shape[0], 1))])
                elif kp.shape[1] != 3:
                    return None
            else:
                return None

            if kp.shape[0] < 17:
                return None
            kp = kp[:17]

            vis_mask = kp[:, 2] >= self._conf
            anchor_idx = [5, 6, 11, 12]
            anchor_vis = [i for i in anchor_idx if vis_mask[i]]
            if len(anchor_vis) < 2:
                return None

            xy = kp[:, :2].copy()
            anchor = xy[anchor_vis].mean(axis=0)
            centered = xy - anchor
            scale = (np.max(np.abs(centered[vis_mask])) if vis_mask.any() else 1.0) + 1e-5
            normed = centered / scale
            normed[~vis_mask] = 0.0

            vec = normed.flatten().astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                return None
            return vec / norm
        except Exception as e:
            print(f"[PhotoMatcher] _kp_raw_to_vec: {e}")
            return None

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(max(0.0, min(1.0, np.dot(a, b))))
