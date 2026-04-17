#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/motion_classifier.py — Классификация движений для EDL (Edit Decision List).
Новые категории: static, cut_point, action_peak, direction_LR, direction_RL, direction_FB.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Optional, Tuple

# COCO-17 индексы ключевых точек
_KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2,
    "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6,
    "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10,
    "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14,
    "l_ankle": 15, "r_ankle": 16,
}

# ── Константы для EDL категорий ───────────────────────────────────────────────
_STATIC_MOTION_THRESHOLD = 0.05   # Максимальная амплитуда движения для static
_CUT_SCORE_MIN = 0.6              # Минимальный cut_score для cut_point
_CUT_SCORE_MAX = 0.95             # Максимальный cut_score (идеальная поза)
_ACTION_PEAK_MOTION = 0.30        # Минимальная скорость для action_peak


def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Евклидово расстояние между двумя точками (x, y)."""
    return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))


def _normalize_keypoints(kp: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    """Нормализовать координаты ключевых точек в диапазон [0, 1]."""
    if orig_w <= 0:
        orig_w = 1
    if orig_h <= 0:
        orig_h = 1
    kp_norm = kp.copy()
    kp_norm[:, 0] = kp[:, 0] / orig_w
    kp_norm[:, 1] = kp[:, 1] / orig_h
    return kp_norm


def _compute_centroid(kp: np.ndarray) -> np.ndarray:
    """Вычислить центр кадра в нормализованных координатах."""
    return np.array([0.5, 0.5])


class MotionClassifier:
    """
    Классификатор движений для EDL (Edit Decision List).
    
    Новые категории:
    - static        — нет движения (motion_magnitude < 0.05)
    - cut_point     — идеальная поза для склейки (симметрия, центр кадра, стабильность)
    - action_peak   — пик действия (максимальная скорость движения)
    - direction_LR  — движение слева направо
    - direction_RL  — движение справа налево
    - direction_FB  — движение к камере / от камеры
    
    Старые категории удалены: dance, sports_*, exercise, medical, daily,
    emotional, falls, work, gestures.
    """

    def __init__(self, conf_threshold: float = 0.30) -> None:
        self._conf = conf_threshold
        self._orig_w = 0
        self._orig_h = 0

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def classify(self, keypoints: np.ndarray,
                 lang: str = "ru",
                 orig_w: Optional[int] = None,
                 orig_h: Optional[int] = None) -> dict:
        """
        Классифицирует позу по новым EDL категориям.
        
        Parameters
        ----------
        keypoints : np.ndarray
            Shape (17, 3) — [x, y, conf] для каждой точки COCO-17.
        lang : str
            Язык меток (ru/en).
        orig_w, orig_h : Optional[int]
            Размеры исходного кадра (для нормализации координат).
        
        Returns
        -------
        dict с ключами:
            primary_cat  : str   — ключ основной категории
            cut_score    : float 0..1 — насколько хороша поза для склейки
            motion_vec   : Tuple[float, float, float] — вектор движения
            direction    : str  — направление (forward/left/right/back)
            confidence   : float
        """
        if orig_w is not None:
            self._orig_w = orig_w
        if orig_h is not None:
            self._orig_h = orig_h

        if keypoints is None or len(keypoints) < 1:
            return self._unknown(lang)

        kp = np.array(keypoints, dtype=float)
        if kp.shape[0] < 17:
            return self._unknown(lang)

        kp = kp[:17]

        # Определяем направление
        direction = self._get_direction(kp)
        direction_conf = self._get_direction_confidence(kp)

        # Вычисляем motion magnitude
        # Если это одна поза — motion magnitude = 0 (static), иначе нужно сравнить с предыдущей
        # Для EDL-классификации одной позы считаем "движение" по смещению центра
        motion_magnitude = self._compute_motion_magnitude(kp)

        # Вычисляем cut_score
        cut_score = self.get_cut_score(kp)

        # Вычисляем motion vector (для последовательности)
        motion_vec = self.get_motion_vector(kp, kp)

        # Определяем primary category
        primary_cat = self._determine_primary_category(
            motion_magnitude, direction, cut_score
        )

        return {
            "primary_cat":  primary_cat,
            "cut_score":    float(cut_score),
            "motion_vec":   motion_vec,
            "direction":    direction,
            "confidence":   float(direction_conf),
        }

    def classify_match(self, match: dict, lang: str = "ru") -> str:
        """
        Быстрая классификация для матча.
        Возвращает строку для отображения в UI.
        """
        kp1 = match.get("kp1")
        kp2 = match.get("kp2")
        kp = kp1 if kp1 is not None else kp2

        direction = match.get("direction", "unknown")

        if kp is not None:
            result = self.classify(kp, lang)
            primary = result.get("primary_cat", "unknown")
            
            # Комбинируем категорию с направлением
            cat_label = self._get_category_label(primary, lang)
            dir_label = self._direction_label(direction, lang)
            
            return f"{cat_label} · {dir_label}"

        return self._direction_label(direction, lang)

    # ──────────────────────────────────────────────────────────────────────
    # get_cut_score() — Метод оценки качества позы для склейки
    # ──────────────────────────────────────────────────────────────────────

    def get_cut_score(self, kp_sequence: list[np.ndarray]) -> float:
        """
        Оценить, насколько хороша поза для склейки (cut point).
        
        Метрика состоит из трёх компонентов:
        1. Симметрия (0.4) — разница между левой и правой стороной тела
        2. Центрирование (0.3) — насколько нос/центр плеч близок к центру кадра
        3. Стабильность (0.3) — насколько мала дисперсия ключевых точек за 3 кадра
        
        Возвращает float в диапазоне [0, 1], где 1.0 — идеальная поза для склейки.
        
        Parameters
        ----------
        kp_sequence : list[np.ndarray]
            Последовательность ключевых точек (обычно 3 кадра).
            Если один кадр — стабильность = 1.0.
        
        Returns
        -------
        float — cut_score в [0, 1]
        """
        if not kp_sequence:
            return 0.0
        
        if len(kp_sequence) == 1:
            kp = kp_sequence[0]
            if kp.shape[0] < 17:
                return 0.0
            kp = kp[:17]
            
            # Вычисляем симметрию
            symmetry_score = self._compute_symmetry(kp)
            
            # Вычисляем центрирование
            centering_score = self._compute_centering(kp)
            
            # Для одной позы стабильность = 1.0
            stability_score = 1.0
            
            # Составная оценка
            cut_score = (
                0.4 * symmetry_score +
                0.3 * centering_score +
                0.3 * stability_score
            )
            
            return float(np.clip(cut_score, 0.0, 1.0))
        
        elif len(kp_sequence) >= 3:
            # Берём последние 3 кадра
            recent_kp = kp_sequence[-3:]
            
            # Средняя поза
            avg_kp = np.mean(np.stack(recent_kp, axis=0), axis=0)
            
            # Вычисляем симметрию для средней позы
            symmetry_score = self._compute_symmetry(avg_kp)
            
            # Вычисляем центрирование
            centering_score = self._compute_centering(avg_kp)
            
            # Вычисляем стабильность по дисперсии
            stability_score = self._compute_stability(recent_kp)
            
            # Составная оценка
            cut_score = (
                0.4 * symmetry_score +
                0.3 * centering_score +
                0.3 * stability_score
            )
            
            return float(np.clip(cut_score, 0.0, 1.0))
        
        else:
            # 2 кадра
            avg_kp = np.mean(np.stack(kp_sequence, axis=0), axis=0)
            symmetry_score = self._compute_symmetry(avg_kp)
            centering_score = self._compute_centering(avg_kp)
            stability_score = 0.8
            
            cut_score = (
                0.4 * symmetry_score +
                0.3 * centering_score +
                0.3 * stability_score
            )
            
            return float(np.clip(cut_score, 0.0, 1.0))

    def _compute_symmetry(self, kp: np.ndarray) -> float:
        """
        Вычислить симметрию позы.
        
        Сравнивает левую и правую стороны тела:
        - плечи
        - локти
        - запястья
        - бёдра
        - колени
        - лодыжки
        
        Возвращает значение в [0, 1], где 1.0 — идеальная симметрия.
        """
        # Плечи
        ls = kp[_KP["l_shoulder"]]
        rs = kp[_KP["r_shoulder"]]
        shoulder_sym = self._compute_symmetry_score(ls, rs)
        
        # Локти
        le = kp[_KP["l_elbow"]]
        re = kp[_KP["r_elbow"]]
        elbow_sym = self._compute_symmetry_score(le, re)
        
        # Запястья
        lw = kp[_KP["l_wrist"]]
        rw = kp[_KP["r_wrist"]]
        wrist_sym = self._compute_symmetry_score(lw, rw)
        
        # Бёдра
        lh = kp[_KP["l_hip"]]
        rh = kp[_KP["r_hip"]]
        hip_sym = self._compute_symmetry_score(lh, rh)
        
        # Колени
        lk = kp[_KP["l_knee"]]
        rk = kp[_KP["r_knee"]]
        knee_sym = self._compute_symmetry_score(lk, rk)
        
        # Лодыжки
        la = kp[_KP["l_ankle"]]
        ra = kp[_KP["r_ankle"]]
        ankle_sym = self._compute_symmetry_score(la, ra)
        
        # Среднее по всем симметриям
        scores = [
            shoulder_sym, elbow_sym, wrist_sym,
            hip_sym, knee_sym, ankle_sym
        ]
        
        return float(np.mean(scores))

    def _compute_symmetry_score(self, left: np.ndarray, right: np.ndarray) -> float:
        """
        Вычислить симметрию между левой и правой точками.
        
        Сравнивает высоту (y) и глубину (x) левой и правой точки.
        Возвращает значение в [0, 1], где 1.0 — идеальная симметрия.
        """
        # Если точки не видны
        if left[2] < self._conf or right[2] < self._conf:
            return 0.0
        
        # Разница по высоте
        dy = abs(left[1] - right[1])
        
        # Разница по горизонтали
        dx = abs(left[0] - right[0])
        
        # Ожидаемая разница по высоте (плечи примерно на одной высоте)
        expected_dy = 0.02  # 2% от высоты кадра
        
        # Ожидаемая разница по горизонтали (плечи симметричны)
        expected_dx = 0.01  # 1% от ширины кадра
        
        # Вычисляем симметрию
        dy_score = max(0.0, 1.0 - dy / max(expected_dy, 0.001))
        dx_score = max(0.0, 1.0 - dx / max(expected_dx, 0.001))
        
        return float(0.5 * dy_score + 0.5 * dx_score)

    def _compute_centering(self, kp: np.ndarray) -> float:
        """
        Вычислить насколько центр тела близок к центру кадра.
        
        Возвращает значение в [0, 1], где 1.0 — идеальное центрирование.
        """
        # Центр плеч
        ls = kp[_KP["l_shoulder"]]
        rs = kp[_KP["r_shoulder"]]
        
        if ls[2] < self._conf or rs[2] < self._conf:
            return 0.0
        
        shoulder_center = np.array([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2])
        
        # Центр кадра
        frame_center = np.array([0.5, 0.5])
        
        # Расстояние до центра
        dist = _distance(shoulder_center, frame_center)
        
        # Ожидаемое расстояние (чем ближе к 0, тем лучше)
        max_dist = np.sqrt(0.5**2 + 0.5**2)  # диагональ от угла до центра
        
        score = max(0.0, 1.0 - dist / max_dist)
        
        return float(score)

    def _compute_stability(self, kp_sequence: list[np.ndarray]) -> float:
        """
        Вычислить стабильность позы по дисперсии ключевых точек.
        
        Возвращает значение в [0, 1], где 1.0 — идеальная стабильность.
        """
        if len(kp_sequence) < 2:
            return 1.0
        
        # Стакаем все точки
        stacked = np.stack(kp_sequence, axis=0)  # (n_frames, 17, 3)
        
        # Вычисляем дисперсию по x и y
        var_x = np.var(stacked[:, :, 0])
        var_y = np.var(stacked[:, :, 1])
        
        # Средняя дисперсия
        avg_var = (var_x + var_y) / 2
        
        # Ожидаемая дисперсия для стабильной позы
        expected_var = 0.0001  # очень маленькая дисперсия
        
        score = max(0.0, 1.0 - avg_var / expected_var)
        
        return float(score)

    # ──────────────────────────────────────────────────────────────────────
    # get_motion_vector() — Вектор движения
    # ──────────────────────────────────────────────────────────────────────

    def get_motion_vector(self, kp1: np.ndarray, kp2: np.ndarray) -> Tuple[float, float, float]:
        """
        Вычислить вектор движения между двумя кадрами.
        
        Возвращает кортеж (dx, dy, dz_approx), где:
        - dx — движение по горизонтали (слева направо)
        - dy — движение по вертикали (вверх вниз)
        - dz_approx — приблизительное движение к/от камеры (по размеру тела)
        
        Parameters
        ----------
        kp1 : np.ndarray — Ключевые точки первого кадра (17, 3)
        kp2 : np.ndarray — Ключевые точки второго кадра (17, 3)
        
        Returns
        -------
        Tuple[float, float, float] — вектор движения
        """
        if kp1.shape[0] < 17 or kp2.shape[0] < 17:
            return (0.0, 0.0, 0.0)
        
        kp1 = kp1[:17]
        kp2 = kp2[:17]
        
        # Вычисляем центроиды
        center1 = self._compute_body_center(kp1)
        center2 = self._compute_body_center(kp2)
        
        # Вектор перемещения центроида
        dx = float(center2[0] - center1[0])
        dy = float(center2[1] - center1[1])
        
        # Оцениваем движение к/от камеры по изменению размера тела
        size1 = self._compute_body_size(kp1)
        size2 = self._compute_body_size(kp2)
        
        if size1 > 0:
            dz_approx = float((size2 - size1) / size1)
        else:
            dz_approx = 0.0
        
        return (dx, dy, dz_approx)

    def _compute_body_center(self, kp: np.ndarray) -> np.ndarray:
        """Вычислить центр тела по ключевым точкам."""
        visible = kp[kp[:, 2] >= self._conf]
        if len(visible) == 0:
            return np.array([0.5, 0.5])
        return np.mean(visible[:, :2], axis=0)

    def _compute_body_size(self, kp: np.ndarray) -> float:
        """Оценить размер тела по высоте от плеч до бёдер."""
        ls = kp[_KP["l_shoulder"]]
        rs = kp[_KP["r_shoulder"]]
        lh = kp[_KP["l_hip"]]
        rh = kp[_KP["r_hip"]]
        
        visible = []
        if ls[2] >= self._conf:
            visible.append(ls)
        if rs[2] >= self._conf:
            visible.append(rs)
        if lh[2] >= self._conf:
            visible.append(lh)
        if rh[2] >= self._conf:
            visible.append(rh)
        
        if len(visible) < 2:
            return 1.0
        
        visible = np.array(visible)
        
        # Высота (y-координаты)
        height = abs(visible[:, 1].max() - visible[:, 1].min())
        
        # Ширина (x-координаты)
        width = abs(visible[:, 0].max() - visible[:, 0].min())
        
        return float(max(height, width))

    # ──────────────────────────────────────────────────────────────────────
    # Внутренние методы
    # ──────────────────────────────────────────────────────────────────────

    def _determine_primary_category(
        self,
        motion_magnitude: float,
        direction: str,
        cut_score: float,
    ) -> str:
        """
        Определить primary категорию на основе признаков.
        """
        # 1. Static — нет движения
        if motion_magnitude < _STATIC_MOTION_THRESHOLD:
            return "static"
        
        # 2. Cut point — высокий cut_score
        if cut_score >= _CUT_SCORE_MIN:
            return "cut_point"
        
        # 3. Action peak — высокая скорость
        if motion_magnitude > _ACTION_PEAK_MOTION:
            return "action_peak"
        
        # 4. Direction categories
        if direction == "left":
            return "direction_RL"  # справа налево (видно справа)
        elif direction == "right":
            return "direction_LR"  # слева направо (видно слева)
        elif direction == "forward" or direction == "back":
            return "direction_FB"  # движение к/от камеры
        
        return "static"

    def _compute_motion_magnitude(self, kp: np.ndarray) -> float:
        """
        Вычислить амплитуду движения для одной позы.
        Для EDL-классификации одной позы считаем "движение" по смещению центра.
        """
        center = self._compute_body_center(kp)
        
        # Нормализованное расстояние от центра кадра
        frame_center = np.array([0.5, 0.5])
        dist = _distance(center, frame_center)
        
        return dist

    def _get_direction(self, kp: np.ndarray) -> str:
        """Определить направление тела."""
        ls = kp[_KP["l_shoulder"]]
        rs = kp[_KP["r_shoulder"]]
        lh = kp[_KP["l_hip"]]
        rh = kp[_KP["r_hip"]]
        
        if ls[2] < self._conf or rs[2] < self._conf:
            return "unknown"
        
        # Плечи видны
        if lh[2] < self._conf or rh[2] < self._conf:
            return "unknown"
        
        # Ширина плеч
        shoulder_w = abs(ls[0] - rs[0])
        # Ширина бёдер
        hip_w = abs(lh[0] - rh[0])
        
        if shoulder_w < 0.03:
            # Боком
            if ls[0] < rs[0]:
                return "right"
            else:
                return "left"
        
        # Нос для перед/зад
        nose = kp[_KP["nose"]]
        
        if nose[2] >= self._conf:
            center_x = (ls[0] + rs[0]) / 2
            offset = (nose[0] - center_x) / max(shoulder_w, 0.001)
            
            if abs(offset) < 0.15:
                return "forward"
            elif offset > 0:
                return "forward-right"
            else:
                return "forward-left"
        else:
            # Нос не виден — спиной
            le = kp[_KP["l_ear"]]
            re = kp[_KP["r_ear"]]
            
            if le[2] >= self._conf and re[2] >= self._conf:
                ear_center = (le[0] + re[0]) / 2
                center_x = (ls[0] + rs[0]) / 2
                offset = (ear_center - center_x) / max(shoulder_w, 0.001)
                
                if abs(offset) < 0.15:
                    return "back"
                elif offset > 0:
                    return "back-right"
                else:
                    return "back-left"
        
        return "back"

    def _get_direction_confidence(self, kp: np.ndarray) -> float:
        """Оценить уверенность в определении направления."""
        ls = kp[_KP["l_shoulder"]]
        rs = kp[_KP["r_shoulder"]]
        
        if ls[2] < self._conf or rs[2] < self._conf:
            return 0.0
        
        # Минимальная уверенность
        min_conf = min(ls[2], rs[2])
        
        return float(min_conf)

    def _get_category_label(self, category: str, lang: str) -> str:
        """Получить человекочитаемое название категории."""
        labels = {
            "static": {"ru": "Статика", "en": "Static"},
            "cut_point": {"ru": "Точка среза", "en": "Cut Point"},
            "action_peak": {"ru": "Пик действия", "en": "Action Peak"},
            "direction_LR": {"ru": "Слева направо", "en": "Left to Right"},
            "direction_RL": {"ru": "Справа налево", "en": "Right to Left"},
            "direction_FB": {"ru": "К/от камеры", "en": "To/Away Camera"},
            "unknown": {"ru": "Неизвестно", "en": "Unknown"},
        }
        
        return labels.get(category, labels["unknown"]).get(lang, category)

    def _direction_label(self, direction: str, lang: str) -> str:
        """Получить метку направления."""
        labels = {
            "forward": {"ru": "Лицом", "en": "Facing"},
            "forward-left": {"ru": "Лицом-влево", "en": "Forward-left"},
            "forward-right": {"ru": "Лицом-вправо", "en": "Forward-right"},
            "left": {"ru": "Слев", "en": "Left"},
            "right": {"ru": "Справа", "en": "Right"},
            "back": {"ru": "Спиной", "en": "Back"},
            "back-left": {"ru": "Спиной-влево", "en": "Back-left"},
            "back-right": {"ru": "Спиной-вправо", "en": "Back-right"},
            "unknown": {"ru": "Неизвестно", "en": "Unknown"},
        }
        
        return labels.get(direction, labels["unknown"]).get(lang, direction)

    def _unknown(self, lang: str) -> dict:
        """Вернуть неизвестную категорию."""
        return {
            "primary_cat": "unknown",
            "cut_score": 0.0,
            "motion_vec": (0.0, 0.0, 0.0),
            "direction": "unknown",
            "confidence": 0.0,
        }


# ──────────────────────────────────────────────────────────────────────────
# Глобальный экземпляр
# ──────────────────────────────────────────────────────────────────────────

_classifier = MotionClassifier()


def classify_match(match: dict, lang: str = "ru") -> str:
    """Быстрая классификация матча для UI."""
    return _classifier.classify_match(match, lang)


def classify_pose(keypoints: np.ndarray, lang: str = "ru") -> dict:
    """Классификация одиночной позы."""
    return _classifier.classify(keypoints, lang)


def get_cut_score(kp_sequence: list[np.ndarray]) -> float:
    """Вычислить cut_score для позы/последовательности."""
    return _classifier.get_cut_score(kp_sequence)


def get_motion_vector(kp1: np.ndarray, kp2: np.ndarray) -> Tuple[float, float, float]:
    """Вычислить вектор движения между двумя кадрами."""
    return _classifier.get_motion_vector(kp1, kp2)