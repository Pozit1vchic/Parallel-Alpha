#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MotionMatcher — надёжный матчер поз с полной обратной совместимостью.
Совместимость со старым main_window.py:
  - MotionMatcher._mirror_vector(V)           ← статический метод (старый вызов)
  - MotionMatcher.preprocess_pose(pose, ...)  ← статический метод (старый вызов)
  - matcher.MIN_MATCH_GAP                     ← атрибут
  - matcher.chunk_size / CHUNK_SIZE           ← атрибут
  - matcher.chunk_overlap / CHUNK_OVERLAP     ← атрибут
  - matcher.max_per_chunk                     ← атрибут
  - matcher.max_total                         ← атрибут
  - matcher.max_unique                        ← атрибут
  - matcher.junk_ratio                        ← атрибут (оставлен для совместимости,
                                                          но не используется как мусор)
  - find_matches(poses_tensor, poses_meta, threshold, min_gap, use_mirror)
Исправленные баги:
  1. Совпадения строго между ОДНИМ человеком — фильтр по track_id на GPU.
  2. Дубликаты устранены: верхнетреугольный фильтр по ГЛОБАЛЬНЫМ индексам.
  3. Мусорные кадры отфильтрованы жёстко (порог good_threshold).
  4. Дедупликация учитывает (video_idx, track_id, t) — разные люди не мешают.
  5. Нормализация по опорным точкам (плечи + бёдра) — стабильнее центроид.
  6. Добавлен фильтр уверенности поз (is_pose_valid).
"""
import bisect
import gc
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
# ═══════════════════════════════════════════════════════════════════════════════
# Константы COCO-17
# ═══════════════════════════════════════════════════════════════════════════════
COCO_N_KPS = 17
# Опорные точки для центроида (плечи + бёдра)
ANCHOR_KPS = [5, 6, 11, 12]
# Парные точки (лево <-> право) для зеркального отражения
MIRROR_PAIRS: List[Tuple[int, int]] = [
    (1, 2), (3, 4),
    (5, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16),
]
# Веса частей тела (COCO-17)
BODY_WEIGHTS = np.array([
    0.8,               # 0  нос
    0.4, 0.4,          # 1,2  глаза
    0.3, 0.3,          # 3,4  уши
    1.5, 1.5,          # 5,6  плечи   ★ опорные
    1.2, 1.2,          # 7,8  локти
    1.0, 1.0,          # 9,10 кисти
    1.5, 1.5,          # 11,12 бёдра  ★ опорные
    1.3, 1.3,          # 13,14 колени
    1.0, 1.0,          # 15,16 стопы
], dtype=np.float32)
# Минимальная уверенность
MIN_KP_CONFIDENCE     = 0.40
MIN_ANCHOR_CONFIDENCE = 0.50
ANCHOR_CONF_KPS       = [5, 6, 11, 12, 13, 14]
# ═══════════════════════════════════════════════════════════════════════════════
# Публичные вспомогательные функции (для совместимости с внешним кодом)
# ═══════════════════════════════════════════════════════════════════════════════
def is_pose_valid(pose_data: dict) -> bool:
    """
    Проверка качества позы: средняя уверенность всех точек и опорных точек.
    Parameters
    ----------
    pose_data : dict с ключом 'keypoints' — np.ndarray (≥17, 3)
    Returns
    -------
    bool
    """
    kps = pose_data.get('keypoints')
    if kps is None or kps.shape[0] < COCO_N_KPS:
        return False
    conf_all    = kps[:COCO_N_KPS, 2].astype(np.float32)
    conf_anchor = kps[ANCHOR_CONF_KPS, 2].astype(np.float32)
    return (
        float(conf_all.mean())    >= MIN_KP_CONFIDENCE
        and float(conf_anchor.mean()) >= MIN_ANCHOR_CONFIDENCE
    )
def preprocess_pose(
    pose_data: dict,
    use_body_weights: bool = True,
) -> np.ndarray:
    """
    Нормализация позы → плоский вектор (34,).
    Шаги:
    1. Берём только (x, y) первых 17 точек.
    2. Вычитаем центроид по ОПОРНЫМ точкам (плечи + бёдра).
    3. Делим на max(|coords|) → [-1, 1].
    4. Умножаем на веса частей тела (опционально).
    5. Flatten → (34,).
    Parameters
    ----------
    pose_data        : dict с ключом 'keypoints' — np.ndarray (≥17, 3)
    use_body_weights : применять ли веса BODY_WEIGHTS
    Returns
    -------
    np.ndarray shape (34,) dtype float32
    """
    kps = pose_data['keypoints'][:COCO_N_KPS, :2].astype(np.float32)  # (17, 2)
    anchor_pts = kps[ANCHOR_KPS]          # (4, 2)
    center     = anchor_pts.mean(axis=0)  # (2,)
    centered   = kps - center             # (17, 2)
    scale  = np.max(np.abs(centered)) + 1e-5
    normed = centered / scale             # (17, 2) in [-1, 1]
    if use_body_weights:
        normed = normed * BODY_WEIGHTS[:, None]   # (17, 2)
        s2     = np.max(np.abs(normed)) + 1e-5
        normed = normed / s2
    return normed.flatten()              # (34,)
def mirror_vectors(vec: torch.Tensor) -> torch.Tensor:
    """
    Зеркальное отражение батча нормированных векторов поз по оси X.
    Parameters
    ----------
    vec : torch.Tensor (N, 34)
    Returns
    -------
    torch.Tensor (N, 34)
    """
    mirrored = vec.clone()
    mirrored[:, 0::2] = -mirrored[:, 0::2]   # инвертируем X-компоненты
    for l_idx, r_idx in MIRROR_PAIRS:
        lx, ly = l_idx * 2,     l_idx * 2 + 1
        rx, ry = r_idx * 2,     r_idx * 2 + 1
        tmp_x = mirrored[:, lx].clone()
        tmp_y = mirrored[:, ly].clone()
        mirrored[:, lx] = mirrored[:, rx]
        mirrored[:, ly] = mirrored[:, ry]
        mirrored[:, rx] = tmp_x
        mirrored[:, ry] = tmp_y
    return mirrored
# ═══════════════════════════════════════════════════════════════════════════════
# Основной класс
# ═══════════════════════════════════════════════════════════════════════════════
class MotionMatcher:
    """
    Матричный поиск похожих поз одного человека.
    Полностью совместим со старым main_window.py:
    ─────────────────────────────────────────────
    • MotionMatcher._mirror_vector(V)       — статический метод (псевдоним)
    • MotionMatcher.preprocess_pose(pose)   — статический метод (псевдоним)
    • matcher.junk_ratio                    — атрибут (совместимость)
    • matcher.MIN_MATCH_GAP                 — атрибут
    • matcher.chunk_size                    — атрибут
    • matcher.chunk_overlap                 — атрибут
    • matcher.max_per_chunk                 — атрибут
    • matcher.max_total                     — атрибут
    • matcher.max_unique                    — атрибут
    • matcher.find_matches(...)             — основной метод
    Принципы работы:
    ─────────────────
    • Матчинг только между позами с одинаковым track_id (если известен).
    • Верхнетреугольный фильтр по ГЛОБАЛЬНЫМ индексам — нет дубликатов.
    • Только совпадения с sim >= good_threshold (мусор не попадает).
    • Дедупликация по (video_idx, track_id, t) с бинарным поиском.
    """
    # ── Дефолты ─────────────────────────────────────────────────────────────
    DEFAULT_CHUNK_SIZE     = 3000
    DEFAULT_CHUNK_OVERLAP  = 300
    DEFAULT_MAX_PER_CHUNK  = 500_000
    DEFAULT_MAX_TOTAL      = 10_000_000
    DEFAULT_MAX_UNIQUE     = 1000
    DEFAULT_MIN_MATCH_GAP  = 5.0
    DEFAULT_JUNK_RATIO     = 0.20    # атрибут сохранён для совместимости
    DEFAULT_GOOD_THRESHOLD = 0.82    # реальная граница качества
    # ── Веса частей тела (COCO-17) — оставлены как атрибут класса ───────────
    BODY_WEIGHTS = BODY_WEIGHTS  # совместимость: MotionMatcher.BODY_WEIGHTS
    def __init__(self, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        # Параметры — полная совместимость со старым main_window.py
        self.chunk_size      = self.DEFAULT_CHUNK_SIZE
        self.chunk_overlap   = self.DEFAULT_CHUNK_OVERLAP
        self.max_per_chunk   = self.DEFAULT_MAX_PER_CHUNK
        self.max_total       = self.DEFAULT_MAX_TOTAL
        self.max_unique      = self.DEFAULT_MAX_UNIQUE
        self.MIN_MATCH_GAP   = self.DEFAULT_MIN_MATCH_GAP
        self.junk_ratio      = self.DEFAULT_JUNK_RATIO   # ← совместимость
        self.good_threshold  = self.DEFAULT_GOOD_THRESHOLD
    # ─────────────────────────────────────────────────────────────────────────
    # Статические методы — обратная совместимость со старым API
    # (main_window.py вызывает MotionMatcher._mirror_vector(V) и
    #  MotionMatcher.preprocess_pose(pose, ...))
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _mirror_vector(vec: torch.Tensor) -> torch.Tensor:
        """
        Статический псевдоним для mirror_vectors().
        Совместимость: main_window.py вызывает MotionMatcher._mirror_vector(V).
        Parameters
        ----------
        vec : torch.Tensor — (N, 34) ИЛИ (34,)
        Returns
        -------
        torch.Tensor той же формы
        """
        if vec.dim() == 1:
            return mirror_vectors(vec.unsqueeze(0)).squeeze(0)
        return mirror_vectors(vec)
    @staticmethod
    def preprocess_pose(
        pose_data: dict,
        use_body_weights: bool = False,
    ) -> np.ndarray:
        """
        Статический псевдоним для модульной функции preprocess_pose().
        Совместимость: main_window.py вызывает MotionMatcher.preprocess_pose(...).
        Parameters
        ----------
        pose_data        : dict с ключом 'keypoints' — np.ndarray (≥17, 3)
        use_body_weights : применять ли веса (по умолчанию False — старое поведение)
        Returns
        -------
        np.ndarray (34,)
        """
        return preprocess_pose(pose_data, use_body_weights=use_body_weights)
    # ─────────────────────────────────────────────────────────────────────────
    # Основной поиск
    # ─────────────────────────────────────────────────────────────────────────
    def find_matches(
        self,
        poses_tensor: torch.Tensor,
        poses_meta:   List[dict],
        threshold:    float = 0.75,
        min_gap:      float = 3.0,
        use_mirror:   bool  = False,
    ) -> List[dict]:
        """
        Найти похожие позы.
        Parameters
        ----------
        poses_tensor : torch.Tensor (N, 34)
            Нормированные вектора поз.
        poses_meta   : list[dict]
            Метаданные каждой позы. Ключи:
              't'         — время в секундах (float)  [обязательный]
              'video_idx' — индекс видео (int)         [обязательный]
              'track_id'  — id трека человека (int, -1 = неизвестен)
              'f'         — номер кадра (int)
              'dir'       — 'forward' / 'backward'
        threshold    : float
            Минимальная косинусная схожесть.
        min_gap      : float
            Минимальный временной интервал (сек) между двумя позами одного видео.
        use_mirror   : bool
            Если True — доп. проход с зеркально отражёнными позами.
        Returns
        -------
        list[dict] — уникальные совпадения, по убыванию sim.
            Ключи: sim, t1, t2, f1, f2, v1_idx, v2_idx, track_id,
                   m1_idx, m2_idx, direction.
        """
        if poses_tensor is None or len(poses_tensor) < 2:
            return []
        n = len(poses_tensor)
        if len(poses_meta) != n:
            print(f"[Matcher] ОШИБКА: poses_tensor ({n}) и poses_meta ({len(poses_meta)}) разной длины!")
            return []
        # Реальный порог — не ниже good_threshold, чтобы убрать мусор
        effective_threshold = max(threshold, self.good_threshold)
        print(
            f"\n[Matcher] N={n} | порог={effective_threshold:.3f} "
            f"(запрошен={threshold:.3f}) | мин.интервал={min_gap}с "
            f"| зеркало={use_mirror}"
        )
        # ── Подготовка тензоров ──────────────────────────────────────────────
        V = poses_tensor.to(dtype=torch.float32, device=self.device)
        V = V.view(n, -1)
        V = F.normalize(V, p=2, dim=1)           # (N, 34), L2-норм
        V_mirror: Optional[torch.Tensor] = None
        if use_mirror:
            V_mirror = F.normalize(self._mirror_vector(V), p=2, dim=1)
        times = torch.tensor(
            [m['t'] for m in poses_meta],
            dtype=torch.float32,
            device=self.device,
        )  # (N,)
        video_ids = torch.tensor(
            [int(m.get('video_idx', 0)) for m in poses_meta],
            dtype=torch.int32,
            device=self.device,
        )  # (N,)
        track_ids = torch.tensor(
            [int(m.get('track_id', -1)) for m in poses_meta],
            dtype=torch.int32,
            device=self.device,
        )  # (N,)
        # ── Чанковый перебор ─────────────────────────────────────────────────
        effective_step = max(1, self.chunk_size - self.chunk_overlap)
        n_chunks       = max(1, (n + effective_step - 1) // effective_step)
        raw_matches: List[dict] = []
        for chunk_idx in range(n_chunks):
            start = chunk_idx * effective_step
            end   = min(start + self.chunk_size, n)
            if start >= n:
                break
            if len(raw_matches) >= self.max_total:
                print(f"[Matcher] Глобальный лимит ({self.max_total:,}) достигнут, стоп.")
                break
            chunk_m = self._process_chunk(
                V=V, V_mirror=V_mirror,
                times=times,
                video_ids=video_ids,
                track_ids=track_ids,
                meta=poses_meta,
                start=start, end=end,
                threshold=effective_threshold,
                min_gap=min_gap,
            )
            raw_matches.extend(chunk_m)
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            print(
                f"[Matcher] Чанк {chunk_idx + 1}/{n_chunks} "
                f"[{start}:{end}] → +{len(chunk_m)} пар "
                f"(всего {len(raw_matches):,})"
            )
        print(f"[Matcher] Сырых совпадений: {len(raw_matches):,}")
        result = self._deduplicate(raw_matches, min_gap=min_gap)
        print(f"[Matcher] Финальных: {len(result)}")
        return result
    # ─────────────────────────────────────────────────────────────────────────
    # Обработка одного чанка
    # ─────────────────────────────────────────────────────────────────────────
    def _process_chunk(
        self,
        V:         torch.Tensor,
        V_mirror:  Optional[torch.Tensor],
        times:     torch.Tensor,
        video_ids: torch.Tensor,
        track_ids: torch.Tensor,
        meta:      List[dict],
        start:     int,
        end:       int,
        threshold: float,
        min_gap:   float,
    ) -> List[dict]:
        """
        Вычислить матрицу схожести для строк [start:end] × [0:N].
        Фильтры (все на GPU):
        1. sim >= threshold
        2. j > i   (верхний треугольник по ГЛОБАЛЬНЫМ индексам — нет дублей)
        3. Временной разрыв >= min_gap внутри одного видео
        4. Одинаковый track_id (если оба != -1)
           ИЛИ разные видео (если track_id неизвестен)
        """
        V_chunk  = V[start:end]          # (chunk, 34)
        T_chunk  = times[start:end]      # (chunk,)
        VI_chunk = video_ids[start:end]  # (chunk,) int32
        TK_chunk = track_ids[start:end]  # (chunk,) int32
        N        = len(V)
        # ── Матрица схожести (chunk × N) ────────────────────────────────────
        sim = torch.mm(V_chunk, V.t())              # (chunk, N)
        if V_mirror is not None:
            sim_m = torch.mm(V_chunk, V_mirror.t())
            sim   = torch.maximum(sim, sim_m)
            del sim_m
        # ── Маска 1: верхний треугольник (глобальные индексы) ───────────────
        col_g = torch.arange(N,          device=self.device, dtype=torch.int32)  # (N,)
        row_g = torch.arange(start, end, device=self.device, dtype=torch.int32)  # (chunk,)
        upper = col_g.unsqueeze(0) > row_g.unsqueeze(1)                          # (chunk, N)
        # ── Маска 2: временной фильтр ────────────────────────────────────────
        time_diff  = torch.abs(T_chunk.unsqueeze(1) - times.unsqueeze(0))  # (chunk, N)
        same_video = VI_chunk.unsqueeze(1) == video_ids.unsqueeze(0)       # (chunk, N)
        diff_video = ~same_video
        # Минимальный разрыв обязателен только внутри одного видео
        time_ok = diff_video | (time_diff >= min_gap)
        # ── Маска 3: фильтр по track_id ─────────────────────────────────────
        #
        # ПРАВИЛО (решает проблему разных людей):
        #   Если оба track_id != -1  →  они должны быть РАВНЫ.
        #   Если хотя бы один == -1  →  допускаем только при РАЗНЫХ видео
        #                                (нельзя отождествить людей внутри одного видео).
        #
        both_known = (TK_chunk.unsqueeze(1) != -1) & (track_ids.unsqueeze(0) != -1)
        same_track = TK_chunk.unsqueeze(1) == track_ids.unsqueeze(0)
        track_ok = (
            (both_known & same_track)          # оба известны и совпадают
            | (~both_known & diff_video)       # хотя бы один неизвестен, но разные видео
        )
        # ── Финальная маска ──────────────────────────────────────────────────
        valid = (sim >= threshold) & upper & time_ok & track_ok   # (chunk, N)
        indices   = torch.nonzero(valid, as_tuple=False)   # (K, 2)
        scores    = sim[valid]                              # (K,)
        # Перенос на CPU за один раз
        indices_np = indices.cpu().numpy()
        scores_np  = scores.cpu().numpy()
        del sim, upper, time_diff, same_video, diff_video, time_ok
        del both_known, same_track, track_ok, valid, indices, scores
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        # ── Лимит на чанк: берём топ-K по sim ───────────────────────────────
        K = len(indices_np)
        if K > self.max_per_chunk:
            top_idx    = np.argpartition(scores_np, -self.max_per_chunk)[-self.max_per_chunk:]
            indices_np = indices_np[top_idx]
            scores_np  = scores_np[top_idx]
            K          = self.max_per_chunk
        # ── Сборка результатов ───────────────────────────────────────────────
        chunk_matches: List[dict] = []
        for k in range(K):
            i_local = int(indices_np[k, 0])
            j       = int(indices_np[k, 1])
            i       = start + i_local
            mi = meta[i]
            mj = meta[j]
            # track_id второй стороны (для дедупликации)
            tk2 = int(mj.get('track_id', -1))
            chunk_matches.append({
                'm1_idx':     i,
                'm2_idx':     j,
                't1':         float(mi['t']),
                't2':         float(mj['t']),
                'f1':         mi.get('f', int(mi['t'] * 30)),
                'f2':         mj.get('f', int(mj['t'] * 30)),
                'v1_idx':     int(mi.get('video_idx', 0)),
                'v2_idx':     int(mj.get('video_idx', 0)),
                'track_id':   int(mi.get('track_id', -1)),
                'track_id_2': tk2,
                'sim':        float(scores_np[k]),
                'direction':  mi.get('dir', 'forward'),
            })
        return chunk_matches
    # ─────────────────────────────────────────────────────────────────────────
    # Дедупликация
    # ─────────────────────────────────────────────────────────────────────────
    def _deduplicate(
        self,
        matches: List[dict],
        min_gap: float,
    ) -> List[dict]:
        """
        Жадная дедупликация.
        Алгоритм:
        1. Сортировка по убыванию sim (лучшие вперёд).
        2. Итерация: пара принимается только если НИ ОДНА из её сторон
           ещё не «занята» в своём (video_idx, track_id).
        3. Принятая пара блокирует момент t1 для (v1, tk1)
           и момент t2 для (v2, tk2) — отдельно для каждого человека.
        4. Лимит: не более self.max_unique.
        Ключ занятости: (video_idx, track_id) → sorted list[float]
        При track_id == -1: (video_idx, -1) — блокирует весь поток видео.
        """
        if not matches:
            return []
        matches.sort(key=lambda x: x['sim'], reverse=True)
        # Словарь: (v_idx, tk_id) -> sorted list[float]
        used: Dict[Tuple[int, int], List[float]] = {}
        def _is_occupied(v: int, tk: int, t: float) -> bool:
            arr = used.get((v, tk))
            if not arr:
                return False
            idx = bisect.bisect_left(arr, t)
            if idx < len(arr) and abs(arr[idx]     - t) < min_gap:
                return True
            if idx > 0          and abs(arr[idx - 1] - t) < min_gap:
                return True
            return False
        def _occupy(v: int, tk: int, t: float):
            key = (v, tk)
            if key not in used:
                used[key] = []
            bisect.insort(used[key], t)
        unique: List[dict] = []
        for m in matches:
            v1  = m['v1_idx']
            v2  = m['v2_idx']
            t1  = m['t1']
            t2  = m['t2']
            tk1 = m['track_id']
            tk2 = m.get('track_id_2', tk1)
            # Принимаем пару только если обе стороны свободны
            if _is_occupied(v1, tk1, t1) or _is_occupied(v2, tk2, t2):
                continue
            unique.append(m)
            _occupy(v1, tk1, t1)
            _occupy(v2, tk2, t2)
            if len(unique) >= self.max_unique:
                break
        return unique
    # ─────────────────────────────────────────────────────────────────────────
    # Псевдонимы атрибутов для совместимости (на случай если main_window
    # обращается к альтернативным именам)
    # ─────────────────────────────────────────────────────────────────────────
    @property
    def CHUNK_SIZE(self) -> int:
        return self.chunk_size
    @CHUNK_SIZE.setter
    def CHUNK_SIZE(self, v: int):
        self.chunk_size = v
    @property
    def CHUNK_OVERLAP(self) -> int:
        return self.chunk_overlap
    @CHUNK_OVERLAP.setter
    def CHUNK_OVERLAP(self, v: int):
        self.chunk_overlap = v
    @property
    def max_matches_per_chunk(self) -> int:
        return self.max_per_chunk
    @max_matches_per_chunk.setter
    def max_matches_per_chunk(self, v: int):
        self.max_per_chunk = v
    @property
    def max_total_matches(self) -> int:
        return self.max_total
    @max_total_matches.setter
    def max_total_matches(self, v: int):
        self.max_total = v
    @property
    def max_unique_results(self) -> int:
        return self.max_unique
    @max_unique_results.setter
    def max_unique_results(self, v: int):
        self.max_unique = v
# ═══════════════════════════════════════════════════════════════════════════════
# Утилита: построить тензор и метаданные из списка frame-данных
# ═══════════════════════════════════════════════════════════════════════════════
def build_poses_tensor(
    frames_data: List[dict],
    use_body_weights: bool = True,
) -> Tuple[Optional[torch.Tensor], List[dict]]:
    """
    Из списка frame-записей построить тензор поз и список метаданных.
    Каждая запись frames_data[i] должна содержать:
      'poses'     : list[dict]  — список поз на кадре
                    Каждая поза: { 'keypoints': np.ndarray(17,3),
                                   'track_id':  int, ... }
      't'         : float       — время кадра в секундах
      'f'         : int         — номер кадра
      'video_idx' : int         — индекс видео
      'dir'       : str         — 'forward' / 'backward'
    Returns
    -------
    tensor : torch.Tensor (M, 34) или None
    meta   : list[dict] длиной M
    """
    vectors: List[np.ndarray] = []
    meta:    List[dict]       = []
    for frame in frames_data:
        t         = float(frame.get('t', 0.0))
        f         = int(frame.get('f', 0))
        video_idx = int(frame.get('video_idx', 0))
        direction = frame.get('dir', 'forward')
        for pose in frame.get('poses', []):
            if not is_pose_valid(pose):
                continue
            vec = preprocess_pose(pose, use_body_weights=use_body_weights)
            vectors.append(vec)
            meta.append({
                't':         t,
                'f':         f,
                'video_idx': video_idx,
                'track_id':  int(pose.get('track_id', -1)),
                'dir':       direction,
            })
    if not vectors:
        return None, []
    tensor = torch.from_numpy(np.stack(vectors, axis=0))  # (M, 34)
    return tensor, meta
# ═══════════════════════════════════════════════════════════════════════════════
# Smoke-test
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=== MotionMatcher smoke-test ===\n")
    # Тест 1: статические методы (старый API main_window.py)
    print("[TEST 1] Статические методы...")
    dummy_kps = np.random.rand(17, 3).astype(np.float32)
    dummy_kps[:, 2] = 0.9  # высокая уверенность
    pose_dict = {'keypoints': dummy_kps, 'track_id': 0}
    vec_np = MotionMatcher.preprocess_pose(pose_dict)
    print(f"  preprocess_pose → shape={vec_np.shape}, dtype={vec_np.dtype}")
    vec_t = torch.from_numpy(vec_np).unsqueeze(0)   # (1, 34)
    mir_t = MotionMatcher._mirror_vector(vec_t)
    print(f"  _mirror_vector  → shape={mir_t.shape}")
    # 1D вызов (иногда main_window передаёт 1D)
    mir_1d = MotionMatcher._mirror_vector(vec_t.squeeze(0))
    print(f"  _mirror_vector(1D) → shape={mir_1d.shape}")
    print("[TEST 1] OK\n")
    # Тест 2: полный пайплайн
    print("[TEST 2] Полный пайплайн...")
    rng = np.random.default_rng(42)
    N_FRAMES = 300
    N_TRACKS = 3
    N_VIDEOS = 2
    frames: List[dict] = []
    for i in range(N_FRAMES):
        vid   = i % N_VIDEOS
        t     = i * 0.5
        track = i % N_TRACKS
        kps   = rng.random((17, 3)).astype(np.float32)
        kps[:, 2] = rng.uniform(0.55, 1.0, 17)
        frames.append({
            't': t, 'f': i, 'video_idx': vid, 'dir': 'forward',
            'poses': [{'keypoints': kps, 'track_id': track}],
        })
    tensor, meta = build_poses_tensor(frames, use_body_weights=True)
    print(f"  Поз после фильтрации: {len(meta)}")
    if tensor is not None:
        matcher = MotionMatcher(device='cpu')
        matcher.max_unique = 20
        matcher.good_threshold = 0.70   # занижаем для smoke-test
        results = matcher.find_matches(
            poses_tensor=tensor,
            poses_meta=meta,
            threshold=0.70,
            min_gap=3.0,
            use_mirror=False,
        )
        print(f"\n  Найдено совпадений: {len(results)}")
        for r in results[:5]:
            print(
                f"    sim={r['sim']:.4f}  "
                f"vid({r['v1_idx']}@t={r['t1']:.1f}s tk={r['track_id']}) ↔ "
                f"vid({r['v2_idx']}@t={r['t2']:.1f}s tk={r['track_id_2']})"
            )
    print("\n[TEST 2] OK")
    print("\n=== Все тесты пройдены ===")