#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class YoloEngine:
    """
    Обёртка над YOLOv8/YOLO11/YOLO26-pose с поддержкой FP16 / FP32,
    динамического батч-размера и авто-прогрева на GPU.
    """

    # Поддерживаемые имена моделей YOLO-pose (от быстрой к точной)
    AVAILABLE_MODELS = [
        # ── YOLOv8 ──────────────────────────────────────────────────────────
        'yolov8n-pose.pt',      # nano    — самая быстрая, минимум VRAM (~6 MB)
        'yolov8s-pose.pt',      # small   — быстрая, хороший баланс (~22 MB)
        'yolov8m-pose.pt',      # medium  — оптимальная по умолчанию (~52 MB)
        'yolov8l-pose.pt',      # large   — точная, нужно 8+ GB VRAM (~104 MB)
        'yolov8x-pose.pt',      # xlarge  — максимальная точность (~166 MB)
        'yolov8x-pose-p6.pt',   # xlarge-p6 — высокое разрешение
        # ── YOLO11 (новое поколение, быстрее v8 при той же точности) ────────
        'yolo11n-pose.pt',      # nano
        'yolo11s-pose.pt',      # small
        'yolo11m-pose.pt',      # medium
        'yolo11l-pose.pt',      # large
        'yolo11x-pose.pt',      # xlarge
        # ── YOLO26 (следующее поколение, ещё быстрее) ──────────────────────
        'yolo26n-pose.pt',      # nano
        'yolo26s-pose.pt',      # small
        'yolo26m-pose.pt',      # medium
        'yolo26l-pose.pt',      # large
        'yolo26x-pose.pt',      # xlarge
    ]

    # Порог уверенности детекции (conf) и keypoint-видимости
    DEFAULT_CONF     = 0.25   # порог уверенности боксов
    KP_VIS_THRESHOLD = 0.3    # порог видимости кейпоинтов

    def __init__(self):
        self.device      = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model       = None
        self.use_fp16    = (self.device == 'cuda')
        self._model_path = 'yolo26m-pose.pt'

        # Кеш последнего рассчитанного batch_size
        # Пересчитывается только при смене разрешения
        self._cached_batch_size: int       = 32
        self._cached_avg_area:   float     = -1.0
        self._cached_default:    int       = 32

    # ─────────────────────────────────────────────────────────────
    # Загрузка модели
    # ─────────────────────────────────────────────────────────────

    def load(self, model_path: str = 'yolo26m-pose.pt'):
        """Загрузить YOLO-pose модель. Повторный вызов — no-op."""
        if self.model is not None:
            return

        self._model_path = model_path
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # На CPU FP16 недоступен
        if self.device == 'cpu':
            self.use_fp16 = False

        print(f"✅ YOLO модель : {model_path}")
        print(f"   Устройство  : {self.device.upper()}")
        print(f"   FP16        : {self.use_fp16}")

        # Прогрев GPU (первый инференс всегда медленнее)
        if self.device == 'cuda':
            self._warmup()

    def reload(self, model_path: str):
        """
        Перезагрузить модель (смена весов на ходу).
        Используется, если пользователь выбрал другой размер модели в UI.
        """
        self.model = None
        self.load(model_path)

    def _warmup(self, imgsz: int = 640, runs: int = 3):
        """
        Прогрев модели на GPU — снижает latency первых реальных батчей.
        Используем 3 прохода: первый компилирует граф, остальные стабилизируют.
        """
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        for _ in range(runs):
            self.model.predict(
                dummy,
                imgsz=imgsz,
                verbose=False,
                half=self.use_fp16,
                conf=self.DEFAULT_CONF,
                stream=False,
                device=self.device,
            )
        print(f"   Прогрев GPU  : {runs} прохода(ов) — готово")

    # ─────────────────────────────────────────────────────────────
    # Инференс
    # ─────────────────────────────────────────────────────────────

    def detect_batch(self, frames_batch: list) -> list:
        """
        Детектировать позы в батче кадров.

        Parameters
        ----------
        frames_batch : list[np.ndarray]
            BGR-кадры (любой размер).

        Returns
        -------
        list[dict | None]
            Один элемент на кадр. None — если поза не найдена.
            Словарь содержит:
              'keypoints'   : np.ndarray (17, 3) — x, y, conf
              'direction'   : str — 'forward' | 'left' | 'right' | 'unknown'
              'confidence'  : float — средняя уверенность по 17 точкам
              'bbox'        : [x1, y1, x2, y2] — bbox по видимым точкам
              'orig_w'      : int
              'orig_h'      : int
        """
        if not frames_batch:
            return []

        if self.model is None:
            raise RuntimeError("Модель не загружена. Вызовите load() перед detect_batch().")

        try:
            results = self.model.predict(
                frames_batch,
                imgsz=640,
                verbose=False,
                half=self.use_fp16,
                conf=self.DEFAULT_CONF,
                stream=False,
                device=self.device,
            )

            batch_poses = []
            for res in results:
                pose_data = self._parse_result(res)
                batch_poses.append(pose_data)

            return batch_poses

        except RuntimeError as exc:
            # OOM — пробуем отдельно каждый кадр
            if 'out of memory' in str(exc).lower():
                print(f"[YoloEngine] OOM в батче ({len(frames_batch)} кадров), "
                      f"переход к поштучной обработке")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return [self._detect_single(f) for f in frames_batch]
            print(f"[YoloEngine] detect_batch RuntimeError: {exc}")
            return [None] * len(frames_batch)

        except Exception as exc:
            print(f"[YoloEngine] detect_batch error: {exc}")
            return [None] * len(frames_batch)

    def _detect_single(self, frame: np.ndarray) -> 'dict | None':
        """Обработка одного кадра (fallback при OOM)."""
        try:
            results = self.model.predict(
                frame,
                imgsz=640,
                verbose=False,
                half=False,           # при OOM отключаем FP16
                conf=self.DEFAULT_CONF,
                stream=False,
                device=self.device,
            )
            if results:
                return self._parse_result(results[0])
        except Exception as exc:
            print(f"[YoloEngine] _detect_single error: {exc}")
        return None

    def _parse_result(self, res) -> 'dict | None':
        """
        Разобрать один результат YOLO → структурированный dict или None.

        Критерии валидности позы:
        - кейпоинтов >= 17
        - хотя бы 5 точек с conf >= KP_VIS_THRESHOLD
        """
        if res.keypoints is None or len(res.keypoints.data) == 0:
            return None

        # Берём самый уверенный детект (первый — YOLO уже сортирует по conf)
        kp = res.keypoints.data[0].cpu().numpy()   # (N, 3)

        if kp.shape[0] < 17:
            return None

        kp17 = kp[:17]  # гарантированно 17 точек COCO

        # Фильтр: нужно хотя бы 5 видимых точек
        visible = kp17[kp17[:, 2] >= self.KP_VIS_THRESHOLD]
        if len(visible) < 5:
            return None

        orig_h, orig_w = res.orig_shape

        # bbox по видимым точкам
        if len(visible) > 0:
            min_x = float(np.min(visible[:, 0]))
            min_y = float(np.min(visible[:, 1]))
            max_x = float(np.max(visible[:, 0]))
            max_y = float(np.max(visible[:, 1]))
            bbox = [min_x, min_y, max_x, max_y]
        else:
            bbox = [0.0, 0.0, float(orig_w), float(orig_h)]

        direction = self._classify_direction(kp17)

        return {
            'keypoints':  kp17,
            'direction':  direction,
            'confidence': float(np.mean(kp17[:, 2])),
            'bbox':       bbox,
            'orig_w':     orig_w,
            'orig_h':     orig_h,
        }

    # ─────────────────────────────────────────────────────────────
    # Классификация направления
    # ─────────────────────────────────────────────────────────────

    def _classify_direction(self, kp17: np.ndarray) -> str:
        """
        Определить направление тела по ключевым точкам COCO.

        Алгоритм (многоуровневый):
        1. Проверяем видимость плеч.
           - Оба не видны → 'unknown'.
           - Только одно видно → явный поворот в сторону невидимого плеча.
        2. Оба плеча видны — вычисляем смещение носа относительно центра плеч.
           - |offset| < THRESHOLD → 'forward'.
           - offset > 0 → 'right', offset < 0 → 'left'.
        3. Нет носа — используем уши как дополнительный признак поворота.

        Parameters
        ----------
        kp17 : np.ndarray (17, 3) — coords (x, y) + conf

        Returns
        -------
        str : 'forward' | 'left' | 'right' | 'unknown'
        """
        # Индексы COCO-17
        NOSE       = 0
        L_EAR      = 3
        R_EAR      = 4
        L_SHOULDER = 5
        R_SHOULDER = 6

        VIS = self.KP_VIS_THRESHOLD

        try:
            conf_ls   = kp17[L_SHOULDER, 2]
            conf_rs   = kp17[R_SHOULDER, 2]
            conf_nose = kp17[NOSE, 2]
            conf_le   = kp17[L_EAR, 2]
            conf_re   = kp17[R_EAR, 2]

            # ── Шаг 1: оба плеча не видны ───────────────────────────────
            if conf_ls < VIS and conf_rs < VIS:
                # Пробуем по ушам
                if conf_le >= VIS and conf_re < VIS:
                    return 'right'
                if conf_re >= VIS and conf_le < VIS:
                    return 'left'
                return 'unknown'

            # ── Шаг 2: только одно плечо видно ──────────────────────────
            # Порог с небольшим зазором, чтобы не срабатывало при частичной видимости
            if conf_ls >= VIS and conf_rs < VIS - 0.05:
                return 'right'   # видно левое → смотрит вправо
            if conf_rs >= VIS and conf_ls < VIS - 0.05:
                return 'left'    # видно правое → смотрит влево

            # ── Шаг 3: оба плеча видны — смотрим на нос ────────────────
            shoulder_cx = (kp17[L_SHOULDER, 0] + kp17[R_SHOULDER, 0]) / 2.0
            shoulder_w  = abs(kp17[L_SHOULDER, 0] - kp17[R_SHOULDER, 0]) + 1e-5

            if conf_nose >= VIS:
                head_offset = (kp17[NOSE, 0] - shoulder_cx) / shoulder_w
            else:
                # Нет носа — пробуем по ушам
                if conf_le >= VIS and conf_re >= VIS:
                    ear_cx = (kp17[L_EAR, 0] + kp17[R_EAR, 0]) / 2.0
                    head_offset = (ear_cx - shoulder_cx) / shoulder_w
                elif conf_le >= VIS:
                    head_offset = (kp17[L_EAR, 0] - shoulder_cx) / shoulder_w
                elif conf_re >= VIS:
                    head_offset = (kp17[R_EAR, 0] - shoulder_cx) / shoulder_w
                else:
                    # Фолбэк — разность уверенности плеч (слабый признак)
                    head_offset = (conf_rs - conf_ls) * 0.15

            # ── Шаг 4: пороговое решение ─────────────────────────────────
            # Адаптивный порог: чем шире плечи (= ближе к камере),
            # тем мягче порог.
            adaptive_threshold = max(0.06, min(0.15, 0.3 / (shoulder_w / 50.0 + 1e-5)))
            if abs(head_offset) < adaptive_threshold:
                return 'forward'
            return 'right' if head_offset > 0 else 'left'

        except Exception:
            return 'unknown'

    # публичный алиас (обратная совместимость)
    def classify_direction(self, kp17: np.ndarray) -> str:
        return self._classify_direction(kp17)

    # ─────────────────────────────────────────────────────────────
    # Динамический батч-размер
    # ─────────────────────────────────────────────────────────────

    def get_dynamic_batch_size(self, frame_sizes: list, default_batch: int = 32) -> int:
        """
        Адаптивный батч-размер на основе разрешения кадров и занятой VRAM.

        ВАЖНО: результат кешируется — пересчёт происходит только при изменении
        среднего разрешения или default_batch, чтобы не тормозить цикл.

        Таблица коэффициентов (средняя площадь кадра в Мпкс):
          > 2.0 Мпкс → ×0.25  (4K / high-res)
          > 1.5 Мпкс → ×0.35  (1440p)
          > 0.8 Мпкс → ×0.55  (1080p)
          > 0.4 Мпкс → ×0.80  (720p)
          > 0.2 Мпкс → ×1.10  (540p)
          <= 0.2 Мпкс → ×1.60 (360p и меньше)

        Дополнительный VRAM-фактор снижает батч при высокой загрузке памяти.

        Parameters
        ----------
        frame_sizes   : list[(h, w)]
        default_batch : int  — базовый батч из AutoTune

        Returns
        -------
        int — итоговый батч (не менее 1, не более 256)
        """
        if not frame_sizes:
            return default_batch

        avg_area = float(np.mean([h * w for h, w in frame_sizes])) / 1e6

        # Кеш: пересчитываем только при реальном изменении
        if (abs(avg_area - self._cached_avg_area) < 0.05
                and default_batch == self._cached_default):
            return self._cached_batch_size

        vram_factor = 1.0
        if torch.cuda.is_available():
            try:
                props      = torch.cuda.get_device_properties(0)
                total_vram = props.total_memory / 1e9
                used_vram  = torch.cuda.memory_reserved() / 1e9
                free_ratio = max(0.0, 1.0 - used_vram / (total_vram + 1e-6))
                # плавный коэффициент: [0.3 .. 1.0]
                vram_factor = max(0.3, free_ratio)
            except Exception:
                pass

        if avg_area > 2.0:
            coeff = 0.25
        elif avg_area > 1.5:
            coeff = 0.35
        elif avg_area > 0.8:
            coeff = 0.55
        elif avg_area > 0.4:
            coeff = 0.80
        elif avg_area > 0.2:
            coeff = 1.10
        else:
            coeff = 1.60

        batch = int(default_batch * coeff * vram_factor)
        result = max(1, min(batch, 256))

        # Сохраняем в кеш
        self._cached_batch_size = result
        self._cached_avg_area   = avg_area
        self._cached_default    = default_batch

        return result
