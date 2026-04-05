#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/photo_matcher.py — поиск человека по фото-референсу."""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Optional


class PhotoMatcher:
    """
    Извлекает позу из фото-референса и фильтрует кадры/матчи
    по схожести с этой позой.
    """

    def __init__(self, conf_threshold: float = 0.25) -> None:
        self._conf         = conf_threshold
        self._ref_vecs:    list[np.ndarray] = []
        self._ref_raw_kps: list[np.ndarray] = []

    # ── Публичный API ─────────────────────────────────────────────────────

    def load_references(
        self,
        photo_paths: list[str],
        yolo,
    ) -> bool:
        """
        Загрузить фото-референсы и извлечь из них позы.
        Возвращает True если хотя бы одна поза извлечена.
        """
        self._ref_vecs.clear()
        self._ref_raw_kps.clear()

        for path in photo_paths[:2]:
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
            vec  = self._pose_to_vec(pose)
            if vec is None:
                print(
                    f"[PhotoMatcher] не удалось векторизовать: {path}")
                continue

            self._ref_vecs.append(vec)

            # Сохраняем сырые kp для проверки пропорций тела
            kp_raw = pose.get("kp") or pose.get("keypoints")
            if kp_raw is not None:
                try:
                    kp = np.array(kp_raw, dtype=float)
                    if kp.ndim == 1 and len(kp) == 51:
                        kp = kp.reshape(17, 3)
                    elif kp.ndim == 1 and len(kp) == 34:
                        kp = np.hstack([
                            kp.reshape(17, 2),
                            np.ones((17, 1))
                        ])
                    if kp.ndim == 2 and kp.shape[1] == 2:
                        kp = np.hstack([
                            kp, np.ones((len(kp), 1))])
                    self._ref_raw_kps.append(kp)
                except Exception as e:
                    print(f"[PhotoMatcher] ошибка kp: {e}")

            print(
                f"[PhotoMatcher] ✓ поза извлечена: "
                f"{Path(path).name}")

        if self._ref_vecs:
            print(
                f"[PhotoMatcher] загружено референсов: "
                f"{len(self._ref_vecs)}")
            ref_ratio = self._get_ref_body_ratio()
            print(
                f"[PhotoMatcher] пропорции тела: "
                f"{ref_ratio:.2f}" if ref_ratio else
                "[PhotoMatcher] пропорции тела: н/д")

        return len(self._ref_vecs) > 0

    def filter_poses_by_reference(
        self,
        frames_data: list[dict],
        threshold:   float = 0.60,
    ) -> list[dict]:
        """
        Оставить только кадры где поза похожа на референс.
        Вызывать ДО матчинга — на списке frames_data.
        Без проверки пропорций — только косинусное сходство.
        """
        if not self._ref_vecs:
            return frames_data

        result:     list[dict] = []
        skipped     = 0

        for frame in frames_data:
            kp_raw = frame.get("kp")
            if kp_raw is None:
                continue

            vec = self._kp_raw_to_vec(kp_raw)
            if vec is None:
                continue

            score = float(max(
                self._cosine(vec, ref)
                for ref in self._ref_vecs))

            if score < threshold:
                skipped += 1
                continue

            frame["photo_sim"] = score
            result.append(frame)

        print(
            f"[PhotoMatcher] кадры: "
            f"{len(frames_data)} → {len(result)} "
            f"(порог={threshold:.2f}, отсев={skipped})")
        return result

    def filter_matches(
        self,
        matches:   list[dict],
        threshold: float = 0.75,
    ) -> list[dict]:
        """
        Фильтр матчей после матчинга (дополнительный).
        Основной фильтр — filter_poses_by_reference до матчинга.
        """
        if not self._ref_vecs:
            return matches

        ref_ratio = self._get_ref_body_ratio()
        result:   list[dict] = []

        for m in matches:
            score = self._match_score(m)
            if score < threshold:
                continue

            if ref_ratio is not None:
                if not self._check_body_ratio(m, ref_ratio):
                    continue

            m["photo_sim"] = float(score)
            result.append(m)

        print(
            f"[PhotoMatcher] {len(matches)} → {len(result)} "
            f"(порог={threshold:.2f})")
        return result

    def best_ref_sim(self, pose_dict: dict) -> float:
        """Схожесть позы с лучшим референсом (0..1)."""
        if not self._ref_vecs:
            return 0.0
        vec = self._pose_to_vec(pose_dict)
        if vec is None:
            return 0.0
        return float(max(
            self._cosine(vec, ref)
            for ref in self._ref_vecs))

    # ── Пропорции тела ────────────────────────────────────────────────────

    def _get_ref_body_ratio(self) -> float | None:
        """Соотношение высоты к ширине скелета из референса."""
        if not self._ref_raw_kps:
            return None
        try:
            return self._body_ratio(self._ref_raw_kps[0])
        except Exception:
            return None

    def _body_ratio(self, kp: np.ndarray) -> float | None:
        """Высота / ширина скелета по видимым точкам."""
        try:
            if kp.ndim != 2 or kp.shape[1] < 3:
                return None
            vis = kp[kp[:, 2] >= self._conf]
            if len(vis) < 4:
                return None
            h = float(np.max(vis[:, 1]) - np.min(vis[:, 1]))
            w = float(np.max(vis[:, 0]) - np.min(vis[:, 0]))
            if w < 1e-5:
                return None
            return h / w
        except Exception:
            return None

    def _check_body_ratio(self, m: dict,
                           ref_ratio: float) -> bool:
        """
        Проверить пропорции тела матча vs референса.
        Допуск ±40%.
        """
        for kp_key in ("kp1", "kp2"):
            kp_raw = m.get(kp_key)
            if kp_raw is None:
                continue
            kp = self._raw_to_kp(kp_raw)
            if kp is None:
                continue
            ratio = self._body_ratio(kp)
            if ratio is None:
                continue
            diff = abs(ratio - ref_ratio) / (ref_ratio + 1e-5)
            if diff < 0.40:
                return True
        return False

    def _raw_to_kp(self, kp_raw) -> Optional[np.ndarray]:
        """Сырые kp → ndarray (N, 3)."""
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

    # ── Схожесть ──────────────────────────────────────────────────────────

    def _match_score(self, m: dict) -> float:
        """Максимальная схожесть kp1/kp2 с любым референсом."""
        best = 0.0
        for kp_key in ("kp1", "kp2"):
            kp_raw = m.get(kp_key)
            if kp_raw is None:
                continue
            vec = self._kp_raw_to_vec(kp_raw)
            if vec is None:
                continue
            for ref in self._ref_vecs:
                s = self._cosine(vec, ref)
                if s > best:
                    best = s
        return best

    # ── Конвертация ───────────────────────────────────────────────────────

    def _pose_to_vec(
            self, pose: dict) -> Optional[np.ndarray]:
        """Конвертировать dict позы → нормированный вектор."""
        kp_raw = pose.get("kp") or pose.get("keypoints")
        if kp_raw is None:
            return None
        return self._kp_raw_to_vec(kp_raw)

    def _kp_raw_to_vec(self, kp_raw) -> Optional[np.ndarray]:
        """
        Конвертировать сырые keypoints → нормированный вектор (34,).
        Принимает: list, ndarray (51,), (34,), (17,2), (17,3).
        """
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

            vis_mask   = kp[:, 2] >= self._conf
            anchor_idx = [5, 6, 11, 12]
            anchor_vis = [i for i in anchor_idx if vis_mask[i]]
            if len(anchor_vis) < 2:
                return None

            xy       = kp[:, :2].copy()
            anchor   = xy[anchor_vis].mean(axis=0)
            centered = xy - anchor
            scale    = (np.max(np.abs(centered[vis_mask]))
                        if vis_mask.any() else 1.0) + 1e-5
            normed   = centered / scale

            normed[~vis_mask] = 0.0

            vec  = normed.flatten().astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                return None
            return vec / norm

        except Exception as e:
            print(f"[PhotoMatcher] _kp_raw_to_vec: {e}")
            return None

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Косинусное сходство двух L2-нормированных векторов."""
        return float(max(0.0, min(1.0, np.dot(a, b))))