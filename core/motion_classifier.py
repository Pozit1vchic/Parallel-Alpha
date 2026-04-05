#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/motion_classifier.py — Классификация движений по скелету позы."""
from __future__ import annotations

import math
import numpy as np
from typing import Optional

# ── Словарь категорий движений (ru/en) ───────────────────────────────────────

MOVEMENT_CATEGORIES: dict[str, dict[str, list[str]]] = {
    "locomotion": {
        "ru": ["ходьба", "бег", "прыжки", "ползание", "лазание",
               "шаг вперёд", "движение"],
        "en": ["walking", "running", "jumping", "crawling", "climbing",
               "stepping", "moving"],
    },
    "arms": {
        "ru": ["поднятие руки", "махи руками", "хлопки",
               "указание", "приветствие", "руки вверх"],
        "en": ["raising arm", "arm swing", "clapping",
               "pointing", "waving", "arms up"],
    },
    "legs": {
        "ru": ["приседание", "выпады", "махи ногами",
               "подъём на носки", "стойка на одной ноге"],
        "en": ["squat", "lunge", "leg swing",
               "toe raise", "one-leg stand"],
    },
    "torso": {
        "ru": ["наклон", "поворот корпуса", "скручивание",
               "прогиб", "наклон в сторону"],
        "en": ["bend", "torso rotation", "twist",
               "backbend", "side lean"],
    },
    "sports_fighting": {
        "ru": ["удар рукой", "удар ногой", "блок",
               "бросок", "захват"],
        "en": ["punch", "kick", "block",
               "throw", "grapple"],
    },
    "sports_ball": {
        "ru": ["бросок мяча", "удар по мячу", "подача", "свинг"],
        "en": ["ball throw", "ball kick", "serve", "swing"],
    },
    "sports_gymnastics": {
        "ru": ["кувырок", "стойка", "колесо", "сальто", "шпагат"],
        "en": ["roll", "handstand", "cartwheel", "flip", "split"],
    },
    "dance": {
        "ru": ["кружение", "волна", "танцевальный шаг", "вращение"],
        "en": ["spin", "body wave", "dance step", "rotation"],
    },
    "daily": {
        "ru": ["сидение", "вставание", "подъём предмета", "открывание"],
        "en": ["sitting", "standing up", "picking up", "opening"],
    },
    "work": {
        "ru": ["поднятие тяжести", "толкание", "тяга", "копание"],
        "en": ["lifting", "pushing", "pulling", "digging"],
    },
    "gestures": {
        "ru": ["рукопожатие", "объятие", "поклон", "жест рукой"],
        "en": ["handshake", "hug", "bow", "hand gesture"],
    },
    "emotional": {
        "ru": ["прыжок радости", "топанье", "отшатывание", "дрожь"],
        "en": ["joy jump", "stomping", "recoil", "trembling"],
    },
    "exercise": {
        "ru": ["отжимания", "приседания", "планка", "выпады", "пресс"],
        "en": ["push-up", "squat", "plank", "lunge", "crunch"],
    },
    "medical": {
        "ru": ["подъём ноги", "велосипед", "разведение рук"],
        "en": ["leg raise", "bicycle", "arm spread"],
    },
    "falls": {
        "ru": ["падение", "спотыкание", "восстановление равновесия"],
        "en": ["fall", "stumble", "balance recovery"],
    },
    # Направления (camera-facing — оставляем как было)
    "facing_camera": {
        "ru": ["лицом к камере"],
        "en": ["facing camera"],
    },
    "facing_right": {
        "ru": ["смотрит вправо"],
        "en": ["facing right"],
    },
    "facing_left": {
        "ru": ["смотрит влево"],
        "en": ["facing left"],
    },
    "facing_away": {
        "ru": ["спиной к камере"],
        "en": ["facing away"],
    },
    "other": {
        "ru": ["прочее"],
        "en": ["other"],
    },
}

# Человекочитаемые названия категорий
CATEGORY_LABELS: dict[str, dict[str, str]] = {
    "locomotion":         {"ru": "Передвижение",    "en": "Locomotion"},
    "arms":               {"ru": "Движения рук",    "en": "Arms"},
    "legs":               {"ru": "Движения ног",    "en": "Legs"},
    "torso":              {"ru": "Корпус",           "en": "Torso"},
    "sports_fighting":    {"ru": "Единоборства",    "en": "Fighting"},
    "sports_ball":        {"ru": "Спорт с мячом",   "en": "Ball sports"},
    "sports_gymnastics":  {"ru": "Гимнастика",      "en": "Gymnastics"},
    "dance":              {"ru": "Танец",            "en": "Dance"},
    "daily":              {"ru": "Быт",              "en": "Daily"},
    "work":               {"ru": "Работа",           "en": "Work"},
    "gestures":           {"ru": "Жесты",            "en": "Gestures"},
    "emotional":          {"ru": "Эмоции",           "en": "Emotional"},
    "exercise":           {"ru": "Упражнения",       "en": "Exercise"},
    "medical":            {"ru": "Медицина",         "en": "Medical"},
    "falls":              {"ru": "Падения",          "en": "Falls"},
    "facing_camera":      {"ru": "Лицом к камере",  "en": "Facing camera"},
    "facing_right":       {"ru": "Смотрит вправо",  "en": "Facing right"},
    "facing_left":        {"ru": "Смотрит влево",   "en": "Facing left"},
    "facing_away":        {"ru": "Спиной к камере", "en": "Facing away"},
    "other":              {"ru": "Прочее",           "en": "Other"},
}

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


def get_category_label(category_key: str, lang: str = "ru") -> str:
    """Возвращает человекочитаемое название категории."""
    return CATEGORY_LABELS.get(category_key, {}).get(lang, category_key)


def get_movement_label(category_key: str, lang: str = "ru",
                       index: int = 0) -> str:
    """Возвращает название конкретного движения из категории."""
    labels = MOVEMENT_CATEGORIES.get(category_key, {}).get(lang, [])
    if labels:
        return labels[min(index, len(labels) - 1)]
    return category_key


class MotionClassifier:
    """
    Классифицирует движение по скелетным данным COCO-17.

    Использует эвристики на основе:
    - положения суставов относительно центра тела
    - углов между суставами
    - направления взгляда / ориентации тела
    - симметрии/асимметрии позы
    """

    def __init__(self, conf_threshold: float = 0.3) -> None:
        self._conf = conf_threshold

    def classify(self, keypoints: np.ndarray,
                 lang: str = "ru") -> dict:
        """
        Классифицирует позу.

        Parameters
        ----------
        keypoints : np.ndarray
            Shape (17, 3) — [x, y, conf] для каждой точки COCO-17.
        lang : str
            Язык меток.

        Returns
        -------
        dict с ключами:
            category     : str  — ключ категории
            label        : str  — человекочитаемое название категории
            movement     : str  — конкретное движение
            direction    : str  — направление (forward/left/right/back)
            confidence   : float
        """
        if keypoints is None or len(keypoints) < 1:
            return self._unknown(lang)

        kp = np.array(keypoints, dtype=float)

        # Защита от плоского вектора (51,) → (17, 3)
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
                return self._unknown(lang)

        elif kp.ndim == 2:
            if kp.shape[1] == 2:
                kp = np.hstack([
                    kp,
                    np.ones((kp.shape[0], 1))
                ])
            elif kp.shape[1] != 3:
                return self._unknown(lang)

        else:
            return self._unknown(lang)

        if kp.shape[0] < 17:
            return self._unknown(lang)

        kp = kp[:17]

        # Направление (оставляем как primary)
        direction = self._get_direction(kp)

        # Основная категория по позе
        category, movement_idx, conf = self._classify_pose(kp)

        label    = get_category_label(category, lang)
        movement = get_movement_label(category, lang, movement_idx)

        return {
            "category":   category,
            "label":      label,
            "movement":   movement,
            "direction":  direction,
            "confidence": conf,
        }
    def classify_match(self, match: dict, lang: str = "ru") -> str:
        """
        Быстрая классификация для матча (совместимость с main_window).
        Возвращает строку для отображения.
        """
        # Пытаемся взять keypoints из матча
        kp1 = match.get("kp1")
        kp2 = match.get("kp2")
        kp  = kp1 if kp1 is not None else kp2

        direction = match.get("direction", "unknown")

        if kp is not None:
            result = self.classify(kp, lang)
            # Комбинируем: движение + направление
            dir_label = self._direction_label(direction, lang)
            if result["category"] not in (
                    "facing_camera", "facing_right",
                    "facing_left", "facing_away", "other"):
                return f"{result['movement']} · {dir_label}"
            return result["movement"]

        # Фолбек — только направление
        return self._direction_label(direction, lang)

    # ── Направление ───────────────────────────────────────────────────────

    def _get_direction(self, kp: np.ndarray) -> str:
        """Определяет ориентацию тела по плечам и бёдрам."""
        ls = kp[_KP["l_shoulder"]]
        rs = kp[_KP["r_shoulder"]]
        lh = kp[_KP["l_hip"]]
        rh = kp[_KP["r_hip"]]

        visible = (ls[2] > self._conf and rs[2] > self._conf)
        if not visible:
            return "unknown"

        # Ширина плеч
        shoulder_w = abs(ls[0] - rs[0])
        # Ширина бёдер
        hip_w = abs(lh[0] - rh[0]) if (
            lh[2] > self._conf and rh[2] > self._conf) else shoulder_w

        # Соотношение видимой ширины к ожидаемой
        ratio = shoulder_w / max(hip_w, 1e-6)

        # Наклон оси плеч
        dx = rs[0] - ls[0]
        dy = rs[1] - ls[1]
        angle = math.degrees(math.atan2(dy, dx))

        if shoulder_w < 30:
            # Боком
            if ls[0] < rs[0]:
                return "right"
            else:
                return "left"

        # Уши для определения перед/зад
        le = kp[_KP["l_ear"]]
        re = kp[_KP["r_ear"]]
        nose = kp[_KP["nose"]]

        if nose[2] > self._conf:
            # Нос виден — лицом к камере или полуоборот
            nose_x = nose[0]
            center_x = (ls[0] + rs[0]) / 2
            offset = (nose_x - center_x) / max(shoulder_w, 1)

            if abs(offset) < 0.15:
                return "forward"
            elif offset > 0:
                return "forward-right"
            else:
                return "forward-left"
        else:
            # Нос не виден — спиной
            if le[2] > self._conf and re[2] > self._conf:
                ear_center = (le[0] + re[0]) / 2
                center_x   = (ls[0] + rs[0]) / 2
                offset = (ear_center - center_x) / max(shoulder_w, 1)
                if abs(offset) < 0.15:
                    return "back"
                elif offset > 0:
                    return "back-right"
                else:
                    return "back-left"
            return "back"

    def _direction_label(self, direction: str, lang: str) -> str:
        _map = {
            "forward":      {"ru": "Лицом к камере",    "en": "Facing camera"},
            "right":        {"ru": "Смотрит вправо",    "en": "Facing right"},
            "left":         {"ru": "Смотрит влево",     "en": "Facing left"},
            "back":         {"ru": "Спиной к камере",   "en": "Facing away"},
            "forward-right":{"ru": "Пол-оборота вправо","en": "Half-turn right"},
            "forward-left": {"ru": "Пол-оборота влево", "en": "Half-turn left"},
            "back-right":   {"ru": "Спиной-вправо",     "en": "Back-right"},
            "back-left":    {"ru": "Спиной-влево",      "en": "Back-left"},
            "unknown":      {"ru": "Неизвестно",        "en": "Unknown"},
        }
        return _map.get(direction, _map["unknown"])[lang]

    # ── Классификация позы ────────────────────────────────────────────────

    def _classify_pose(self, kp: np.ndarray) -> tuple[str, int, float]:
        """
        Возвращает (category_key, movement_index, confidence).
        """
        scores: dict[str, float] = {}

        # Вычисляем признаки
        arms_raised    = self._arms_raised(kp)
        arms_asymm     = self._arms_asymmetric(kp)
        legs_bent      = self._legs_bent(kp)
        legs_wide      = self._legs_wide(kp)
        torso_lean     = self._torso_lean(kp)
        torso_twist    = self._torso_twist(kp)
        kick_pose      = self._kick_pose(kp)
        punch_pose     = self._punch_pose(kp)
        squat_deep     = self._deep_squat(kp)
        jump_pose      = self._jump_pose(kp)
        bow_pose       = self._bow_pose(kp)
        one_leg        = self._one_leg_raised(kp)

        # ── Правила (приоритет сверху вниз) ──────────────────────────────

        # Падение — экстремальный наклон корпуса
        if torso_lean > 0.8:
            scores["falls"] = torso_lean

        # Единоборства
        if kick_pose > 0.6:
            scores["sports_fighting"] = kick_pose
            return "sports_fighting", 1, kick_pose  # удар ногой
        if punch_pose > 0.6:
            scores["sports_fighting"] = punch_pose
            return "sports_fighting", 0, punch_pose  # удар рукой

        # Гимнастика
        if arms_raised > 0.8 and legs_wide > 0.6:
            return "sports_gymnastics", 0, 0.75      # стойка/колесо

        # Упражнения
        if squat_deep > 0.7:
            return "exercise", 1, squat_deep          # приседание
        if bow_pose > 0.65:
            return "exercise", 2, bow_pose            # планка/наклон

        # Танец — симметричные махи + наклон
        if arms_asymm > 0.5 and torso_lean < 0.3:
            return "dance", 0, arms_asymm

        # Руки вверх
        if arms_raised > 0.65:
            if arms_asymm < 0.2:
                return "arms", 5, arms_raised         # руки вверх
            return "arms", 0, arms_raised             # поднятие руки

        # Одна нога поднята
        if one_leg > 0.6:
            return "legs", 2, one_leg                 # мах ногой

        # Ноги широко
        if legs_wide > 0.55:
            return "legs", 1, legs_wide               # выпад

        # Ноги согнуты
        if legs_bent > 0.5:
            return "legs", 0, legs_bent               # приседание

        # Поворот корпуса
        if torso_twist > 0.5:
            return "torso", 1, torso_twist

        # Наклон
        if torso_lean > 0.4:
            return "torso", 0, torso_lean

        # Жесты — руки на уровне груди, асимметрично
        if arms_asymm > 0.3 and arms_raised < 0.4:
            return "gestures", 3, arms_asymm

        # Просто стоит / идёт
        return "locomotion", 0, 0.4

    # ── Признаки (features) ───────────────────────────────────────────────

    def _get_kp(self, kp: np.ndarray, *names: str) -> list[Optional[np.ndarray]]:
        """Возвращает точки с проверкой видимости."""
        result = []
        for name in names:
            p = kp[_KP[name]]
            result.append(p if p[2] > self._conf else None)
        return result

    def _arms_raised(self, kp: np.ndarray) -> float:
        """Насколько руки подняты выше плеч (0–1)."""
        ls, rs = kp[_KP["l_shoulder"]], kp[_KP["r_shoulder"]]
        lw, rw = kp[_KP["l_wrist"]], kp[_KP["r_wrist"]]
        lh, rh = kp[_KP["l_hip"]], kp[_KP["r_hip"]]

        torso_h = abs(
            ((ls[1] + rs[1]) / 2) - ((lh[1] + rh[1]) / 2)
        ) or 1.0

        scores = []
        if ls[2] > self._conf and lw[2] > self._conf:
            scores.append(max(0, (ls[1] - lw[1]) / torso_h))
        if rs[2] > self._conf and rw[2] > self._conf:
            scores.append(max(0, (rs[1] - rw[1]) / torso_h))
        return float(np.mean(scores)) if scores else 0.0

    def _arms_asymmetric(self, kp: np.ndarray) -> float:
        """Асимметрия рук (0–1). Высокое = руки на разной высоте."""
        lw = kp[_KP["l_wrist"]]
        rw = kp[_KP["r_wrist"]]
        ls = kp[_KP["l_shoulder"]]
        rs = kp[_KP["r_shoulder"]]
        if (lw[2] < self._conf or rw[2] < self._conf or
                ls[2] < self._conf or rs[2] < self._conf):
            return 0.0
        torso_h = abs(ls[1] - kp[_KP["l_hip"]][1]) or 1.0
        diff = abs(lw[1] - rw[1]) / torso_h
        return float(min(1.0, diff))

    def _legs_bent(self, kp: np.ndarray) -> float:
        """Согнутость ног — угол в колене (0–1)."""
        scores = []
        for side in [("l_hip","l_knee","l_ankle"),
                     ("r_hip","r_knee","r_ankle")]:
            h, k, a = [kp[_KP[s]] for s in side]
            if all(p[2] > self._conf for p in [h, k, a]):
                ang = self._angle3(h[:2], k[:2], a[:2])
                # 180° = прямая нога, 90° = сильно согнута
                score = max(0.0, (180.0 - ang) / 90.0)
                scores.append(min(1.0, score))
        return float(np.mean(scores)) if scores else 0.0

    def _legs_wide(self, kp: np.ndarray) -> float:
        """Ноги широко расставлены (0–1)."""
        la = kp[_KP["l_ankle"]]
        ra = kp[_KP["r_ankle"]]
        ls = kp[_KP["l_shoulder"]]
        rs = kp[_KP["r_shoulder"]]
        if (la[2] < self._conf or ra[2] < self._conf or
                ls[2] < self._conf or rs[2] < self._conf):
            return 0.0
        ankle_w   = abs(la[0] - ra[0])
        shoulder_w = abs(ls[0] - rs[0]) or 1.0
        return float(min(1.0, ankle_w / shoulder_w / 1.5))

    def _torso_lean(self, kp: np.ndarray) -> float:
        """Наклон корпуса от вертикали (0–1)."""
        ls = kp[_KP["l_shoulder"]]
        rs = kp[_KP["r_shoulder"]]
        lh = kp[_KP["l_hip"]]
        rh = kp[_KP["r_hip"]]
        if not all(p[2] > self._conf for p in [ls, rs, lh, rh]):
            return 0.0
        sc = np.array([(ls[0]+rs[0])/2, (ls[1]+rs[1])/2])
        hc = np.array([(lh[0]+rh[0])/2, (lh[1]+rh[1])/2])
        vec = sc - hc
        length = np.linalg.norm(vec) or 1.0
        # Угол от вертикали
        angle = abs(math.degrees(math.atan2(vec[0], -vec[1])))
        return float(min(1.0, angle / 45.0))

    def _torso_twist(self, kp: np.ndarray) -> float:
        """Скручивание корпуса — разница ширины плеч и бёдер (0–1)."""
        ls = kp[_KP["l_shoulder"]]
        rs = kp[_KP["r_shoulder"]]
        lh = kp[_KP["l_hip"]]
        rh = kp[_KP["r_hip"]]
        if not all(p[2] > self._conf for p in [ls, rs, lh, rh]):
            return 0.0
        sw = abs(ls[0] - rs[0])
        hw = abs(lh[0] - rh[0])
        diff = abs(sw - hw) / max(sw, hw, 1.0)
        return float(min(1.0, diff * 2))

    def _kick_pose(self, kp: np.ndarray) -> float:
        """Поза удара ногой (0–1)."""
        for side in [("l_hip","l_knee","l_ankle"),
                     ("r_hip","r_knee","r_ankle")]:
            h, k, a = [kp[_KP[s]] for s in side]
            opp_h = kp[_KP["r_hip" if side[0]=="l_hip" else "l_hip"]]
            if all(p[2] > self._conf for p in [h, k, a, opp_h]):
                # Нога поднята высоко
                if k[1] < opp_h[1]:
                    ang = self._angle3(h[:2], k[:2], a[:2])
                    if ang < 150:
                        return 0.8
        return 0.0

    def _punch_pose(self, kp: np.ndarray) -> float:
        """Поза удара рукой (0–1)."""
        for side in [("l_shoulder","l_elbow","l_wrist"),
                     ("r_shoulder","r_elbow","r_wrist")]:
            s, e, w = [kp[_KP[x]] for x in side]
            if all(p[2] > self._conf for p in [s, e, w]):
                ang = self._angle3(s[:2], e[:2], w[:2])
                # Вытянутая рука + горизонтально
                if ang > 160 and abs(w[1] - s[1]) < abs(w[0] - s[0]):
                    return 0.75
        return 0.0

    def _deep_squat(self, kp: np.ndarray) -> float:
        """Глубокое приседание (0–1)."""
        lh = kp[_KP["l_hip"]]
        rh = kp[_KP["r_hip"]]
        lk = kp[_KP["l_knee"]]
        rk = kp[_KP["r_knee"]]
        la = kp[_KP["l_ankle"]]
        ra = kp[_KP["r_ankle"]]
        if not all(p[2] > self._conf for p in [lh,rh,lk,rk,la,ra]):
            return 0.0
        hip_y  = (lh[1] + rh[1]) / 2
        knee_y = (lk[1] + rk[1]) / 2
        # Бёдра близко к коленям по вертикали
        dist = abs(hip_y - knee_y)
        ref  = abs(lk[1] - la[1]) or 1.0
        return float(min(1.0, max(0.0, 1.0 - dist / ref)))

    def _jump_pose(self, kp: np.ndarray) -> float:
        """Поза прыжка — ноги согнуты + корпус высоко (0–1)."""
        return self._legs_bent(kp) * 0.5

    def _bow_pose(self, kp: np.ndarray) -> float:
        """Поза наклона вперёд / поклона (0–1)."""
        lean = self._torso_lean(kp)
        ls   = kp[_KP["l_shoulder"]]
        lh   = kp[_KP["l_hip"]]
        if ls[2] > self._conf and lh[2] > self._conf:
            # Плечи ниже бёдер
            if ls[1] > lh[1]:
                return min(1.0, lean + 0.3)
        return lean * 0.6

    def _one_leg_raised(self, kp: np.ndarray) -> float:
        """Одна нога поднята (0–1)."""
        lk = kp[_KP["l_knee"]]
        rk = kp[_KP["r_knee"]]
        lh = kp[_KP["l_hip"]]
        rh = kp[_KP["r_hip"]]
        if not all(p[2] > self._conf for p in [lk,rk,lh,rh]):
            return 0.0
        hip_y  = (lh[1] + rh[1]) / 2
        l_raised = max(0.0, hip_y - lk[1])
        r_raised = max(0.0, hip_y - rk[1])
        ref = abs(lh[1] - lk[1]) or 1.0
        return float(min(1.0, max(l_raised, r_raised) / ref))

    # ── Геометрия ─────────────────────────────────────────────────────────

    @staticmethod
    def _angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Угол в точке b (градусы)."""
        v1 = a - b
        v2 = c - b
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 180.0
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(math.degrees(math.acos(cos_a)))

    @staticmethod
    def _unknown(lang: str) -> dict:
        return {
            "category":   "other",
            "label":      "Прочее" if lang == "ru" else "Other",
            "movement":   "прочее" if lang == "ru" else "other",
            "direction":  "unknown",
            "confidence": 0.0,
        }


# Глобальный экземпляр
_classifier = MotionClassifier()


def classify_match(match: dict, lang: str = "ru") -> str:
    """Быстрая классификация матча для UI."""
    return _classifier.classify_match(match, lang)


def classify_pose(keypoints: np.ndarray, lang: str = "ru") -> dict:
    """Классификация одиночной позы."""
    return _classifier.classify(keypoints, lang)