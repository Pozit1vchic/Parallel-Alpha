#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/photo_matcher.py
PhotoMatcher v1 — Appearance-first matching with body proportions.

Критические улучшения v1:
1. Сравнение по пропорциям тела в первую очередь
2. Кэширование эмбеддингов (векторов и пропорций)
3. Двухэтапный фильтр: внешность → поза
4. Строгий порог для внешности (0.65)
5. Адаптивный порог для позы (зависит от качества фото)
6. Fallback на частичное совпадение

Цель: находить того же человека, а не похожую позу.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from core.matcher.pose_processor import compute_body_proportions, compare_body_proportions


class PhotoMatcher:
    APPEARANCE_THRESHOLD = 0.65
    POSE_THRESHOLD_STRICT = 0.70
    POSE_THRESHOLD_RELAXED = 0.55
    
    def __init__(self, conf_threshold: float = 0.24) -> None:
        self._conf = conf_threshold
        self._ref_vecs: list[np.ndarray] = []
        self._ref_raw_kps: list[np.ndarray] = []
        self._ref_body_props: list[dict] = []
        
        self._cache_vecs: dict[int, np.ndarray] = {}
        self._cache_props: dict[int, dict] = {}

    def load_references(
        self,
        photo_paths: list[str],
        yolo,
    ) -> bool:
        self._ref_vecs.clear()
        self._ref_raw_kps.clear()
        self._ref_body_props.clear()

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
                            print(f"[PhotoMatcher] ✓ пропорции: leg/torso={props['leg_to_torso']:.2f}")
                except Exception as e:
                    print(f"[PhotoMatcher] ошибка kp: {e}")

            print(f"[PhotoMatcher] ✓ поза извлечена: {Path(path).name}")

        if self._ref_vecs:
            print(f"[PhotoMatcher] загружено референсов: {len(self._ref_vecs)}")
            print(f"[PhotoMatcher] пропорций тела: {len(self._ref_body_props)}")

        return len(self._ref_vecs) > 0

    def filter_poses_by_reference(
        self,
        frames_data: list[dict],
        threshold: float = 0.55,
    ) -> list[dict]:
        if not self._ref_vecs:
            return frames_data

        result: list[dict] = []
        skipped_appearance = 0
        skipped_pose = 0

        for frame in frames_data:
            frame_idx = id(frame)
            
            # Кэшируем векторы и пропорции
            if frame_idx not in self._cache_vecs:
                kp_raw = frame.get("kp")
                if kp_raw is None:
                    continue
                
                vec = self._kp_raw_to_vec(kp_raw)
                if vec is None:
                    continue
                
                self._cache_vecs[frame_idx] = vec
                
                kp = self._raw_to_kp(kp_raw)
                if kp is not None:
                    props = compute_body_proportions(kp, self._conf)
                    if props.get("valid"):
                        self._cache_props[frame_idx] = props
            
            vec = self._cache_vecs.get(frame_idx)
            if vec is None:
                continue
            
            props = self._cache_props.get(frame_idx)
            
            # ЭТАП 1: Сравнение по внешности (пропорциям тела)
            appearance_score = 0.0
            if props and self._ref_body_props:
                appearance_score = max(
                    compare_body_proportions(props, ref_props)
                    for ref_props in self._ref_body_props
                )
            
            if appearance_score > 0.0 and appearance_score < self.APPEARANCE_THRESHOLD:
                skipped_appearance += 1
                continue
            
            # ЭТАП 2: Сравнение по позе
            pose_score = float(max(
                self._cosine(vec, ref)
                for ref in self._ref_vecs
            ))
            
            # Адаптивный порог: если внешность совпала хорошо, ослабляем требование к позе
            effective_threshold = threshold
            if appearance_score > self.APPEARANCE_THRESHOLD:
                effective_threshold = self.POSE_THRESHOLD_RELAXED
            else:
                effective_threshold = self.POSE_THRESHOLD_STRICT
            
            if pose_score < effective_threshold:
                skipped_pose += 1
                continue

            frame["photo_sim"] = pose_score
            frame["appearance_sim"] = appearance_score
            result.append(frame)

        print(
            f"[PhotoMatcher] кадры: {len(frames_data)} → {len(result)} "
            f"(внешность: -{skipped_appearance}, поза: -{skipped_pose})"
        )
        return result

    def filter_matches(
        self,
        matches: list[dict],
        threshold: float = 0.70,
    ) -> list[dict]:
        if not self._ref_vecs:
            return matches

        result: list[dict] = []
        skipped = 0

        for m in matches:
            # Проверяем оба кадра матча
            max_appearance = 0.0
            max_pose = 0.0
            
            for kp_key in ("kp1", "kp2"):
                kp_raw = m.get(kp_key)
                if kp_raw is None:
                    continue
                
                # Векторизуем
                vec = self._kp_raw_to_vec(kp_raw)
                if vec is None:
                    continue
                
                # Сравнение по позе
                pose_sim = float(max(
                    self._cosine(vec, ref)
                    for ref in self._ref_vecs
                ))
                max_pose = max(max_pose, pose_sim)
                
                # Сравнение по внешности
                kp = self._raw_to_kp(kp_raw)
                if kp is not None:
                    props = compute_body_proportions(kp, self._conf)
                    if props.get("valid") and self._ref_body_props:
                        appearance_sim = max(
                            compare_body_proportions(props, ref_props)
                            for ref_props in self._ref_body_props
                        )
                        max_appearance = max(max_appearance, appearance_sim)
            
            # Двухэтапный фильтр
            if max_appearance > 0.0 and max_appearance < self.APPEARANCE_THRESHOLD:
                skipped += 1
                continue
            
            # Адаптивный порог для позы
            effective_threshold = threshold
            if max_appearance > self.APPEARANCE_THRESHOLD:
                effective_threshold = self.POSE_THRESHOLD_RELAXED
            else:
                effective_threshold = self.POSE_THRESHOLD_STRICT
            
            if max_pose < effective_threshold:
                skipped += 1
                continue

            m["photo_sim"] = float(max_pose)
            m["appearance_sim"] = float(max_appearance)
            result.append(m)

        print(
            f"[PhotoMatcher] матчи: {len(matches)} → {len(result)} "
            f"(отсев={skipped})"
        )
        return result

    def best_ref_sim(self, pose_dict: dict) -> float:
        if not self._ref_vecs:
            return 0.0
        vec = self._pose_to_vec(pose_dict)
        if vec is None:
            return 0.0
        return float(max(
            self._cosine(vec, ref)
            for ref in self._ref_vecs
        ))

    def _raw_to_kp(self, kp_raw) -> Optional[np.ndarray]:
        try:
            kp = np.array(kp_raw, dtype=float)
            if kp.ndim == 1:
                n = len(kp)
                if n == 51:
                    kp = kp.reshape(17, 3)
                elif n == 34:
                    kp = np.hstack([
                        kp.reshape(17, 2),
                        np.ones((17, 1))
                    ])
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
                    kp = np.hstack([
                        kp.reshape(17, 2),
                        np.ones((17, 1))
                    ])
                else:
                    return None
            elif kp.ndim == 2:
                if kp.shape[1] == 2:
                    kp = np.hstack([
                        kp, np.ones((kp.shape[0], 1))])
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
            scale = (np.max(np.abs(centered[vis_mask]))
                     if vis_mask.any() else 1.0) + 1e-5
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