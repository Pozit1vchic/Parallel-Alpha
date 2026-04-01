#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
import gc
from typing import List, Tuple, Optional


class MotionMatcher:
    """
    Матричный поиск похожих поз с чанкингом, дедупликацией
    и поддержкой мульти-видео батча.

    Ключевые улучшения:
    - Правильная обработка перекрытий между чанками (chunk_overlap).
    - Дедупликация учитывает video_idx, а не только время.
    - Весовая схема по частям тела (body weights).
    - Поддержка масштабной инвариантности (уже встроена в preprocess_pose).
    - Жадная дедупликация с адаптивным порогом sim для "мусора".
    - Фиксированный размер батча при построении sim-матрицы (нет фризов).
    """

    # ── Дефолты (переопределяются через auto-tune в main_window) ────────────
    DEFAULT_CHUNK_SIZE    = 3000
    DEFAULT_CHUNK_OVERLAP = 300
    DEFAULT_MAX_PER_CHUNK = 500_000
    DEFAULT_MAX_TOTAL     = 10_000_000
    DEFAULT_MAX_UNIQUE    = 1000
    DEFAULT_MIN_MATCH_GAP = 5.0
    DEFAULT_JUNK_RATIO    = 0.20   # доля "мусора" (sim < 0.85) от общего числа

    # ── Веса частей тела (COCO-17) ───────────────────────────────────────────
    # Индексы: 0=нос, 1-2=глаза, 3-4=уши, 5-6=плечи, 7-8=локти,
    #          9-10=кисти, 11-12=бёдра, 13-14=колени, 15-16=стопы
    BODY_WEIGHTS = np.array([
        1.0,               # 0  нос
        0.6, 0.6,          # 1,2  глаза
        0.5, 0.5,          # 3,4  уши
        1.5, 1.5,          # 5,6  плечи   (опорные точки)
        1.2, 1.2,          # 7,8  локти
        1.0, 1.0,          # 9,10 кисти
        1.5, 1.5,          # 11,12 бёдра  (опорные точки)
        1.3, 1.3,          # 13,14 колени
        1.0, 1.0,          # 15,16 стопы
    ], dtype=np.float32)

    def __init__(self, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Параметры (выставляются снаружи через auto-tune)
        self.MIN_MATCH_GAP = self.DEFAULT_MIN_MATCH_GAP
        self.chunk_size    = self.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = self.DEFAULT_CHUNK_OVERLAP
        self.max_per_chunk = self.DEFAULT_MAX_PER_CHUNK
        self.max_total     = self.DEFAULT_MAX_TOTAL
        self.max_unique    = self.DEFAULT_MAX_UNIQUE
        self.junk_ratio    = self.DEFAULT_JUNK_RATIO

    # ─────────────────────────────────────────────────────────────
    # Препроцессинг позы
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def preprocess_pose(
        pose_data: dict,
        use_body_weights: bool = False,
    ) -> np.ndarray:
        """
        Нормализация позы (17 точек × 2 координаты → вектор 34).

        Шаги:
        1. Берём только (x, y), игнорируем confidence.
        2. Вычитаем центроид → центрирование.
        3. Делим на max(|x|, |y|) → нормировка в [-1, 1].
        4. Опционально — умножаем на весовой вектор частей тела.

        Parameters
        ----------
        pose_data        : dict с ключом 'keypoints' — np.ndarray (17, 3)
        use_body_weights : bool — применять ли веса частей тела

        Returns
        -------
        np.ndarray (34,) — нормированный плоский вектор позы
        """
        kps      = pose_data['keypoints'][:17, :2].astype(np.float32)  # (17, 2)
        center   = np.mean(kps, axis=0)
        centered = kps - center
        scale    = np.max(np.abs(centered)) + 1e-5
        normed   = centered / scale  # (17, 2)

        if use_body_weights:
            weights = MotionMatcher.BODY_WEIGHTS[:, None]  # (17, 1)
            normed  = normed * weights
            # Перенормируем, чтобы масштаб не уехал
            s2 = np.max(np.abs(normed)) + 1e-5
            normed = normed / s2

        return normed.flatten()  # (34,)

    # ─────────────────────────────────────────────────────────────
    # Зеркальная инвариантность
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _mirror_vector(vec: torch.Tensor) -> torch.Tensor:
        """
        Зеркальное отражение вектора позы (flip по оси X).

        Порядок ключевых точек COCO-17:
          0 – нос, 1/2 – глаз Л/П, 3/4 – ухо Л/П,
          5/6 – плечо Л/П, 7/8 – локоть Л/П, 9/10 – кисть Л/П,
          11/12 – бедро Л/П, 13/14 – колено Л/П, 15/16 – стопа Л/П

        При зеркалировании:
          - X-координаты инвертируются.
          - Левые и правые точки меняются местами.

        Parameters
        ----------
        vec : torch.Tensor (N, 34)  — плоские нормированные вектора

        Returns
        -------
        torch.Tensor (N, 34) — зеркальные вектора
        """
        # Парные точки (левая <-> правая)
        SWAP_PAIRS: List[Tuple[int, int]] = [
            (1, 2), (3, 4),
            (5, 6), (7, 8), (9, 10),
            (11, 12), (13, 14), (15, 16),
        ]

        mirrored = vec.clone()

        # Инвертируем X (чётные индексы: 0, 2, 4, … = X-компоненты)
        mirrored[:, 0::2] = -mirrored[:, 0::2]

        # Меняем местами парные точки
        for l_idx, r_idx in SWAP_PAIRS:
            lx, ly = l_idx * 2,     l_idx * 2 + 1
            rx, ry = r_idx * 2,     r_idx * 2 + 1

            tmp_x = mirrored[:, lx].clone()
            tmp_y = mirrored[:, ly].clone()
            mirrored[:, lx] = mirrored[:, rx]
            mirrored[:, ly] = mirrored[:, ry]
            mirrored[:, rx] = tmp_x
            mirrored[:, ry] = tmp_y

        return mirrored

    # ─────────────────────────────────────────────────────────────
    # Основной поиск
    # ─────────────────────────────────────────────────────────────

    def find_matches(
        self,
        poses_tensor: torch.Tensor,
        poses_meta:   list,
        threshold:    float = 0.70,
        min_gap:      float = 3.0,
        use_mirror:   bool  = False,
    ) -> list:
        """
        Поиск похожих поз среди всех пар (i, j) с i < j (верхний треугольник).

        Parameters
        ----------
        poses_tensor : torch.Tensor (N, 34)
            Нормированные вектора поз.
        poses_meta   : list[dict]
            Метаданные каждой позы (ключи: 't', 'f', 'dir', 'video_idx').
        threshold    : float [0..1]
            Минимальная косинусная схожесть.
        min_gap      : float
            Минимальный временной интервал между двумя позами одного видео (с).
        use_mirror   : bool
            Если True — дополнительно ищет совпадения с зеркальными позами.

        Returns
        -------
        list[dict] — уникальные совпадения, отсортированные по убыванию sim.
        """
        if poses_tensor is None or len(poses_tensor) < 10:
            return []

        n = len(poses_tensor)
        print(f"\n[Matcher] Поиск среди {n} поз | порог={threshold:.2f} | "
              f"мин.интервал={min_gap}с | зеркало={use_mirror}")

        # Нормируем один раз
        V = poses_tensor.to(dtype=torch.float32, device=self.device)
        V = V.view(n, -1)
        V = F.normalize(V, p=2, dim=1)   # (N, 34)

        # Зеркальная версия (опционально)
        V_mirror: Optional[torch.Tensor] = None
        if use_mirror:
            V_mirror = self._mirror_vector(V)
            V_mirror = F.normalize(V_mirror, p=2, dim=1)

        times = torch.tensor(
            [m['t'] for m in poses_meta],
            dtype=torch.float32,
            device=self.device,
        )  # (N,)

        raw_matches: list = []

        # Шаг чанка с учётом перекрытия
        effective_step = max(1, self.chunk_size - self.chunk_overlap)
        n_chunks = (n + effective_step - 1) // effective_step

        for chunk_idx in range(n_chunks):
            start = chunk_idx * effective_step
            end   = min(start + self.chunk_size, n)

            if start >= n:
                break

            if len(raw_matches) >= self.max_total:
                print(f"[Matcher] Глобальный лимит ({self.max_total:,}) достигнут, стоп.")
                break

            chunk_matches = self._process_chunk(
                V, V_mirror, times, poses_meta,
                start, end,
                threshold, min_gap,
            )
            raw_matches.extend(chunk_matches)

            # Освобождение памяти GPU
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

            print(f"[Matcher] Чанк {chunk_idx + 1}/{n_chunks} "
                  f"[{start}:{end}]: +{len(chunk_matches)} пар "
                  f"(итого {len(raw_matches):,})")

        print(f"[Matcher] Сырых совпадений: {len(raw_matches):,}")
        return self._deduplicate(raw_matches)

    # ─────────────────────────────────────────────────────────────
    # Обработка одного чанка
    # ─────────────────────────────────────────────────────────────

    def _process_chunk(
        self,
        V:         torch.Tensor,
        V_mirror:  Optional[torch.Tensor],
        times:     torch.Tensor,
        meta:      list,
        start:     int,
        end:       int,
        threshold: float,
        min_gap:   float,
    ) -> list:
        """
        Вычислить матрицу схожести для строк [start:end] и вернуть пары.

        Гарантирует верхнетреугольность (j > i глобально),
        чтобы не дублировать пары при перекрытии чанков.
        """
        V_chunk = V[start:end]       # (chunk, 34)
        T_chunk = times[start:end]   # (chunk,)
        n       = len(V)

        # Матрица схожести: (chunk, N)
        sim = torch.mm(V_chunk, V.t())

        # Опционально: поэлементный максимум sim и зеркального sim
        if V_mirror is not None:
            sim_m = torch.mm(V_chunk, V_mirror.t())
            sim   = torch.maximum(sim, sim_m)
            del sim_m

        # Временная разница: (chunk, N)
        time_diff = torch.abs(T_chunk.unsqueeze(1) - times.unsqueeze(0))

        # Только верхний треугольник (j > i глобально)
        col_idx = torch.arange(n, device=self.device).unsqueeze(0)          # (1, N)
        row_idx = torch.arange(start, end, device=self.device).unsqueeze(1) # (chunk, 1)
        upper   = col_idx > row_idx   # (chunk, N)

        valid = (sim >= threshold) & (time_diff >= min_gap) & upper

        indices = torch.nonzero(valid, as_tuple=False)  # остаётся на device
        scores  = sim[valid]

        # Переносим на CPU за один раз (быстрее, чем по одному)
        indices_np = indices.cpu().numpy()
        scores_np  = scores.cpu().numpy()

        del sim, time_diff, upper, valid, indices, scores
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        chunk_matches: list = []
        limit = min(len(indices_np), self.max_per_chunk)

        for k in range(limit):
            i_local = int(indices_np[k, 0])
            j       = int(indices_np[k, 1])
            i       = start + i_local

            chunk_matches.append({
                'm1_idx':    i,
                'm2_idx':    j,
                't1':        meta[i]['t'],
                't2':        meta[j]['t'],
                'f1':        meta[i].get('f', int(meta[i]['t'] * 30)),
                'f2':        meta[j].get('f', int(meta[j]['t'] * 30)),
                'v1_idx':    meta[i].get('video_idx', 0),
                'v2_idx':    meta[j].get('video_idx', 0),
                'sim':       float(scores_np[k]),
                'direction': meta[i].get('dir', 'forward'),
            })

        return chunk_matches

    # ─────────────────────────────────────────────────────────────
    # Дедупликация
    # ─────────────────────────────────────────────────────────────

    def _deduplicate(self, matches: list) -> list:
        """
        Жадная дедупликация.

        Алгоритм:
        1. Сортировка по убыванию sim (лучшие идут первыми).
        2. Разделение на "хорошие" (sim >= 0.85) и "мусор" (< 0.85).
        3. Из мусора берём только junk_ratio (по умолчанию 20%).
        4. Жадный фильтр: пропускаем пары, у которых t1 или t2
           слишком близки к уже выбранным (с учётом video_idx).

        Важно: конфликт — если ХОТЯ БЫ ОДНА сторона (t1 или t2) совпадает
        с уже выбранной парой (в рамках того же видео). Это предотвращает
        показ одного и того же момента в нескольких совпадениях.

        Returns
        -------
        list[dict] — не более self.max_unique уникальных совпадений.
        """
        if not matches:
            return []

        matches.sort(key=lambda x: x['sim'], reverse=True)

        good      = [m for m in matches if m['sim'] >= 0.85]
        junk      = [m for m in matches if m['sim'] <  0.85]
        junk_take = int(len(junk) * self.junk_ratio)
        selected  = good + junk[:junk_take]

        print(f"[Matcher] Хороших (>=0.85): {len(good):,} | "
              f"мусора (<0.85): {len(junk):,} (взято {junk_take:,})")
        print(f"[Matcher] На дедупликацию: {len(selected):,}")

        gap = self.MIN_MATCH_GAP

        # Используем словарь {video_idx: sorted list of used times}
        # для O(log n) поиска вместо O(n) перебора
        used_times: dict = {}   # video_idx -> list of float (sorted)
        unique: List[dict] = []

        def _is_close(vid: int, t: float) -> bool:
            """Есть ли уже использованный момент в пределах gap?"""
            arr = used_times.get(vid)
            if not arr:
                return False
            # Бинарный поиск
            import bisect
            idx = bisect.bisect_left(arr, t)
            # Проверяем соседей
            if idx < len(arr) and abs(arr[idx] - t) < gap:
                return True
            if idx > 0 and abs(arr[idx - 1] - t) < gap:
                return True
            return False

        def _mark_used(vid: int, t: float):
            import bisect
            if vid not in used_times:
                used_times[vid] = []
            bisect.insort(used_times[vid], t)

        for m in selected:
            t1 = m['t1']
            t2 = m['t2']
            v1 = m['v1_idx']
            v2 = m['v2_idx']

            # Конфликт — если хотя бы одна из сторон уже занята
            if _is_close(v1, t1) or _is_close(v2, t2):
                continue

            unique.append(m)
            _mark_used(v1, t1)
            _mark_used(v2, t2)

            if len(unique) >= self.max_unique:
                break

        print(f"[Matcher] Уникальных после дедупликации: {len(unique)}")
        return unique
