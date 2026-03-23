from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypedDict

import numpy as np
import torch

try:
    import faiss
except ImportError:
    faiss = None

NUM_KEYPOINTS = 17
POSE_DIM = NUM_KEYPOINTS * 2
EPS = 1e-8
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

logger = logging.getLogger(__name__)


class MotionType(IntEnum):
    STANDING = 0
    SITTING = 1
    WALKING = 2
    RUNNING = 3
    OTHER = 4


class MatchResult(TypedDict):
    t1: float
    t2: float
    duration: float
    similarity: float
    motion_type: int
    motion_label: str
    v1_idx: int
    v2_idx: int
    frame_i: int
    frame_j: int
    energy: float
    source: str


@dataclass
class MatcherConfig:
    min_pose_confidence: float = 0.25
    min_visible_keypoints: int = 6
    min_frame_gap: int = 12
        coarse_similarity_threshold: float = 0.35
    final_similarity_threshold: float = 0.30
    max_candidates_for_dtw: int = 3000
    nms_iou_threshold: float = 0.7
    diversity_min_frame_gap: int = 20
    max_results: int = 500
    fallback_top_k: int = 100
    neural_rerank_weight: float = 0.20
    max_neural_candidates: int = 100
    faiss_use_gpu: bool = True
    chunk_size: int = 4000
    log_every_stage: bool = True


@dataclass
class _PreparedData:
    features: np.ndarray
    norm_xy: np.ndarray
    frame_indices: np.ndarray
    track_ids: np.ndarray
    motion_types: np.ndarray
    motion_labels: List[str]
    energy: np.ndarray
    fps: float
    track_rows: Dict[int, np.ndarray] = field(default_factory=dict)
    track_positions: Dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class _Candidate:
    i: int
    j: int
    base_similarity: float = 0.0
    rrf_score: float = 0.0
    dtw_similarity: float = 0.0
    final_score: float = 0.0
    a_start: int = 0
    a_end: int = 0
    b_start: int = 0
    b_end: int = 0
    source: str = "fused"


class MotionMatcher2:
    def __init__(
        self,
        config: Optional[MatcherConfig] = None,
        device: Optional[str] = None,
        neural_reranker: Optional[Callable[[List[_Candidate], _PreparedData], List[float]]] = None,
    ) -> None:
        self.config = config or MatcherConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.neural_reranker = neural_reranker

    def match(self, pose_sequence: Any) -> List[MatchResult]:
        """Run the full matching cascade and return formatted matches."""
        data = self._preprocess(pose_sequence)
        
        if data.features.shape[0] < 2:
            logger.warning("Not enough valid poses: %d", data.features.shape[0])
            return []

        logger.info("Processing %d poses", data.features.shape[0])

        # Этап 1: FAISS поиск
        global_rank = self._faiss_rank_pairs(data.features, self.config.faiss_top_k_global)
        logger.info("FAISS found %d global pairs", len(global_rank))

        # Этап 2: Семантические бакеты
        bucket_rank: Dict[Tuple[int, int], Tuple[int, float]] = {}
        if self.config.use_semantic_buckets:
            bucket_rank = self._semantic_rank_pairs(data)
            logger.info("Semantic buckets found %d pairs", len(bucket_rank))

        # Этап 3: RRF слияние
        fused = self._rrf_fuse(global_rank, bucket_rank)
        logger.info("RRF fused: %d unique pairs", len(fused))

        # Этап 4: Coarse фильтрация
        coarse = self._coarse_filter(fused, data)
        logger.info("Coarse filter: %d candidates remain", len(coarse))

        # Этап 5: DTW временное сравнение
        temporal = self._temporal_rerank(coarse, data)
        logger.info("Temporal rerank: %d candidates passed", len(temporal))

        # Этап 6: Нейросетевой ранкер (опционально)
        reranked = self._apply_neural_rerank(temporal, data)
        
        # Форматирование результатов
        if reranked:
            results = self._nms_and_format(reranked, data, source="dtw")
            if results:
                logger.info("Found %d matches via DTW cascade", len(results))
                return results

        # Fallback: если ничего не нашли
        logger.info("Using FAISS fallback")
        fallback = self._fallback_from_fused(fused, data)
        return self._nms_and_format(fallback, data, source="fallback")

    def _extract_keypoints_from_frame(self, frame_data: Any) -> Optional[np.ndarray]:
        """
        Извлекает keypoints из frame_data в формате MultiPoseData.
        
        Поддерживаемые форматы:
        - MultiPoseData объект с people списком
        - Словарь с ключами 'keypoints' или 'people'
        - Прямой numpy массив keypoints
        """
        # Вариант 1: это словарь
        if isinstance(frame_data, dict):
            # Прямые keypoints
            if 'keypoints' in frame_data:
                return np.asarray(frame_data['keypoints'], dtype=np.float32)
            # Через people
            if 'people' in frame_data:
                people = frame_data['people']
                if people and len(people) > 0:
                    person = people[0]
                    if isinstance(person, dict) and 'keypoints' in person:
                        return np.asarray(person['keypoints'], dtype=np.float32)
                    if hasattr(person, 'keypoints'):
                        return np.asarray(person.keypoints, dtype=np.float32)
        
        # Вариант 2: объект с атрибутом people (MultiPoseData)
        if hasattr(frame_data, 'people'):
            people = frame_data.people
            if people and len(people) > 0:
                person = people[0]
                if hasattr(person, 'keypoints'):
                    return np.asarray(person.keypoints, dtype=np.float32)
                if isinstance(person, dict) and 'keypoints' in person:
                    return np.asarray(person['keypoints'], dtype=np.float32)
        
        # Вариант 3: прямой numpy массив
        if isinstance(frame_data, np.ndarray):
            return frame_data.astype(np.float32)
        
        # Вариант 4: объект с атрибутом keypoints
        if hasattr(frame_data, 'keypoints'):
            return np.asarray(frame_data.keypoints, dtype=np.float32)
        
        return None

    def _preprocess(self, pose_sequence: Any) -> _PreparedData:
        """
        Предобработка для формата MultiPoseData от YoloEngine.
        """
        # ДИАГНОСТИКА
        logger.info("=== PREPROCESS DEBUG ===")
        logger.info("pose_sequence type: %s", type(pose_sequence))
        
        frames = []
        fps = 30.0
        
        # Извлекаем fps если есть
        if isinstance(pose_sequence, dict):
            fps = float(pose_sequence.get('fps', pose_sequence.get('frame_rate', 30.0)))
        
        # Извлекаем список кадров
        if isinstance(pose_sequence, list):
            frames = pose_sequence
            logger.info("Direct list, frames count: %d", len(frames))
        elif isinstance(pose_sequence, dict):
            # Пробуем разные ключи
            for key in ['poses', 'frames', 'data', 'sequence']:
                if key in pose_sequence:
                    frames = pose_sequence[key]
                    logger.info("Extracted from dict['%s'], frames count: %d", key, len(frames))
                    break
        
        if not frames:
            logger.error("No frames extracted from pose_sequence")
            return self._empty_data(fps)
        
        # Логируем первый элемент для диагностики
        if len(frames) > 0:
            logger.info("First frame type: %s", type(frames[0]))
            if hasattr(frames[0], '__dict__'):
                logger.info("First frame attributes: %s", [a for a in dir(frames[0]) if not a.startswith('_')])
        
        norm_xy_list = []
        frame_idx_list = []
        track_id_list = []
        
        for frame_num, frame_data in enumerate(frames):
            # Извлекаем keypoints
            keypoints = self._extract_keypoints_from_frame(frame_data)
            
            if keypoints is None:
                continue
            
            # Проверяем форму
            if keypoints.shape == (NUM_KEYPOINTS, 3):
                xy = keypoints[:, :2]
                conf = keypoints[:, 2]
            elif keypoints.shape == (NUM_KEYPOINTS, 2):
                xy = keypoints
                conf = np.ones(NUM_KEYPOINTS, dtype=np.float32)
            else:
                logger.debug(f"Unexpected keypoints shape: {keypoints.shape}")
                continue
            
            # Проверяем уверенность
            if np.mean(conf) < self.config.min_pose_confidence:
                continue
            
            # Проверяем видимые ключевые точки
            visible = conf >= self.config.min_pose_confidence
            if visible.sum() < self.config.min_visible_keypoints:
                continue
            
            # Нормализуем позу
            norm_xy = self._normalize_single_pose(xy, visible)
            if norm_xy is None:
                continue
            
            norm_xy_list.append(norm_xy)
            frame_idx_list.append(frame_num)
            track_id_list.append(0)  # Пока один трек
        
        if not norm_xy_list:
            logger.warning("No valid poses after filtering")
            return self._empty_data(fps)
        
        logger.info("Valid poses: %d", len(norm_xy_list))
        
        # Конвертируем в массивы
        norm_xy = np.stack(norm_xy_list, axis=0)  # (N, 17, 2)
        frame_indices = np.array(frame_idx_list, dtype=np.int32)
        track_ids = np.array(track_id_list, dtype=np.int32)
        
        # Вычисляем энергию движения
        energy = self._compute_energy(norm_xy, track_ids, frame_indices)
        
        # Классифицируем типы движений
        motion_types = self._classify_motion_types(energy)
        motion_labels = [MotionType(mt).name.lower() for mt in motion_types]
        
        # Строим признаки для FAISS (позиции + энергия)
        features = norm_xy.reshape(norm_xy.shape[0], -1)  # (N, 34)
        energy_norm = energy.reshape(-1, 1)
        features = np.concatenate([features, energy_norm], axis=1)
        features = self._l2_normalize(features)
        
        # Индексы треков для оконного поиска
        track_rows = {0: np.arange(len(norm_xy))}
        track_positions = {0: np.arange(len(norm_xy))}
        
        logger.info("Preprocess complete: %d poses", len(norm_xy))
        
        return _PreparedData(
            features=features.astype(np.float32),
            norm_xy=norm_xy.astype(np.float32),
            frame_indices=frame_indices,
            track_ids=track_ids,
            motion_types=motion_types,
            motion_labels=motion_labels,
            energy=energy.astype(np.float32),
            fps=fps,
            track_rows=track_rows,
            track_positions=track_positions,
        )
    
    def _normalize_single_pose(self, xy: np.ndarray, visible: np.ndarray) -> Optional[np.ndarray]:
        """Нормализация одной позы: центрирование по тазу, масштаб по плечам."""
        # Центрирование по тазу
        if visible[LEFT_HIP] and visible[RIGHT_HIP]:
            pelvis = 0.5 * (xy[LEFT_HIP] + xy[RIGHT_HIP])
        elif visible[LEFT_SHOULDER] and visible[RIGHT_SHOULDER]:
            pelvis = 0.5 * (xy[LEFT_SHOULDER] + xy[RIGHT_SHOULDER])
        else:
            # Используем центр масс видимых точек
            pelvis = np.mean(xy[visible], axis=0)
        
        shifted = xy - pelvis
        
        # Масштабирование по плечам
        if visible[LEFT_SHOULDER] and visible[RIGHT_SHOULDER]:
            scale = np.linalg.norm(xy[LEFT_SHOULDER] - xy[RIGHT_SHOULDER])
        else:
            # Используем размах видимых точек
            min_xy = np.min(xy[visible], axis=0)
            max_xy = np.max(xy[visible], axis=0)
            scale = np.linalg.norm(max_xy - min_xy)
        
        if scale < EPS:
            return None
        
        normalized = shifted / scale
        return normalized.astype(np.float32)
    
    def _compute_energy(self, norm_xy: np.ndarray, track_ids: np.ndarray, frame_indices: np.ndarray) -> np.ndarray:
        """Вычисление энергии движения (скорости) для каждой позы."""
        energy = np.zeros(norm_xy.shape[0], dtype=np.float32)
        
        for tid in np.unique(track_ids):
            rows = np.where(track_ids == tid)[0]
            if len(rows) < 2:
                continue
            
            # Сортируем по кадрам
            sort_idx = np.argsort(frame_indices[rows])
            rows = rows[sort_idx]
            
            # Вычисляем скорости
            poses = norm_xy[rows].reshape(len(rows), -1)
            velocities = np.linalg.norm(np.diff(poses, axis=0), axis=1)
            energy[rows[0]] = velocities[0] if len(velocities) > 0 else 0
            if len(velocities) > 1:
                energy[rows[1:]] = velocities
        
        return energy
    
    def _classify_motion_types(self, energy: np.ndarray) -> np.ndarray:
        """Классификация типов движений по энергии."""
        low, high = self.config.motion_energy_thresholds
        out = np.full(energy.shape, MotionType.OTHER, dtype=np.int32)
        out[energy < low * 0.5] = MotionType.SITTING
        out[(energy >= low * 0.5) & (energy < low)] = MotionType.STANDING
        out[(energy >= low) & (energy < high)] = MotionType.WALKING
        out[energy >= high] = MotionType.RUNNING
        return out
    
    def _l2_normalize(self, x: np.ndarray) -> np.ndarray:
        """L2 нормализация векторов."""
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norms, EPS, None)
    
    def _faiss_rank_pairs(self, feats: np.ndarray, top_k: int) -> Dict[Tuple[int, int], Tuple[int, float]]:
        """FAISS поиск похожих пар."""
        n = feats.shape[0]
        if n < 2:
            return {}
        
        k = min(top_k, n - 1)
        if k <= 0:
            return {}
        
        ranks: Dict[Tuple[int, int], Tuple[int, float]] = {}
        
        # Используем FAISS если доступен
        if faiss is not None:
            try:
                x = feats.astype(np.float32)
                index = faiss.IndexFlatIP(x.shape[1])
                
                if self.config.faiss_use_gpu and torch.cuda.is_available():
                    try:
                        res = faiss.StandardGpuResources()
                        index = faiss.index_cpu_to_gpu(res, 0, index)
                    except Exception as e:
                        logger.warning("FAISS GPU init failed: %s", e)
                
                index.add(x)
                scores, ids = index.search(x, k + 1)
                
                for i in range(n):
                    rank = 0
                    for j in range(scores.shape[1]):
                        j_idx = int(ids[i, j])
                        if j_idx == i or j_idx < 0:
                            continue
                        rank += 1
                        pair = (i, j_idx) if i < j_idx else (j_idx, i)
                        score = float(scores[i, j])
                        if pair not in ranks or rank < ranks[pair][0]:
                            ranks[pair] = (rank, score)
                        if rank >= k:
                            break
                
                return ranks
            except Exception as e:
                logger.warning("FAISS search failed: %s, using dense fallback", e)
        
        # Fallback: плотный поиск
        for start in range(0, n, self.config.chunk_size):
            end = min(n, start + self.config.chunk_size)
            batch = feats[start:end]
            sim = batch @ feats.T
            
            # Исключаем себя
            for r in range(end - start):
                sim[r, start + r] = -np.inf
            
            # Получаем топ-k
            top_indices = np.argsort(-sim, axis=1)[:, :k]
            top_scores = np.take_along_axis(sim, top_indices, axis=1)
            
            for r in range(end - start):
                i = start + r
                for rank, (j, score) in enumerate(zip(top_indices[r], top_scores[r])):
                    if j == i:
                        continue
                    j = int(j)
                    pair = (i, j) if i < j else (j, i)
                    if pair not in ranks or rank + 1 < ranks[pair][0]:
                        ranks[pair] = (rank + 1, float(score))
        
        return ranks
    
    def _semantic_rank_pairs(self, data: _PreparedData) -> Dict[Tuple[int, int], Tuple[int, float]]:
        """Поиск внутри семантических групп (по типу движения)."""
        grouped: Dict[int, List[int]] = defaultdict(list)
        for i, mt in enumerate(data.motion_types):
            grouped[int(mt)].append(i)
        
        result: Dict[Tuple[int, int], Tuple[int, float]] = {}
        for rows in grouped.values():
            if len(rows) < 2:
                continue
            sub_feats = data.features[rows]
            sub_rank = self._faiss_rank_pairs(sub_feats, self.config.faiss_top_k_bucket)
            for (a, b), (r, s) in sub_rank.items():
                pair = (rows[a], rows[b])
                if pair not in result or r < result[pair][0]:
                    result[pair] = (r, s)
        return result
    
    def _rrf_fuse(
        self,
        global_rank: Dict[Tuple[int, int], Tuple[int, float]],
        bucket_rank: Dict[Tuple[int, int], Tuple[int, float]],
    ) -> List[_Candidate]:
        """Reciprocal Rank Fusion."""
        pairs = set(global_rank.keys()) | set(bucket_rank.keys())
        fused: List[_Candidate] = []
        
        for pair in pairs:
            g = global_rank.get(pair)
            b = bucket_rank.get(pair)
            
            rrf = 0.0
            sims = []
            
            if g:
                rrf += self.config.global_weight / (self.config.rrf_k + g[0])
                sims.append(g[1])
            if b:
                rrf += self.config.semantic_bucket_weight / (self.config.rrf_k + b[0])
                sims.append(b[1])
            
            if sims:
                base_sim = max(sims)
                fused.append(_Candidate(
                    i=pair[0], j=pair[1],
                    base_similarity=base_sim,
                    rrf_score=rrf,
                    final_score=base_sim
                ))
        
        fused.sort(key=lambda c: c.rrf_score, reverse=True)
        return fused
    
    def _coarse_filter(self, candidates: List[_Candidate], data: _PreparedData) -> List[_Candidate]:
        """Грубая фильтрация по порогу и временному разрыву."""
        selected: List[_Candidate] = []
        
        for c in candidates:
            # Проверяем временной разрыв
            fi = int(data.frame_indices[c.i])
            fj = int(data.frame_indices[c.j])
            if abs(fi - fj) < self.config.min_frame_gap:
                continue
            
            # Проверяем порог схожести
            if c.base_similarity >= self.config.coarse_similarity_threshold:
                c.final_score = 0.7 * c.base_similarity + 0.3 * c.rrf_score
                selected.append(c)
            
            if len(selected) >= self.config.max_candidates_for_dtw:
                break
        
        selected.sort(key=lambda c: c.final_score, reverse=True)
        return selected[:self.config.max_candidates_for_dtw]
    
    def _extract_window(self, idx: int, data: _PreparedData) -> Tuple[np.ndarray, int, int]:
        """Извлекает окно кадров вокруг указанной позиции."""
        frame = int(data.frame_indices[idx])
        half = self.config.dtw_window_frames // 2
        
        start_frame = max(0, frame - half)
        end_frame = frame + half
        
        # Находим все позы в этом диапазоне
        rows = []
        for i in range(len(data.frame_indices)):
            if start_frame <= data.frame_indices[i] <= end_frame:
                rows.append(i)
        
        if not rows:
            # Если нет кадров, возвращаем одну позу
            seq = data.norm_xy[idx:idx+1]
            seq = self._resample_sequence(seq, self.config.dtw_resample_len)
            return seq, frame, frame
        
        seq = data.norm_xy[rows]
        seq = self._resample_sequence(seq, self.config.dtw_resample_len)
        
        return seq, int(data.frame_indices[rows[0]]), int(data.frame_indices[rows[-1]])
    
    def _resample_sequence(self, seq: np.ndarray, target_len: int) -> np.ndarray:
        """Ресемплинг временного ряда до нужной длины."""
        if seq.shape[0] == target_len:
            return seq
        
        x_old = np.linspace(0, 1, seq.shape[0], dtype=np.float32)
        x_new = np.linspace(0, 1, target_len, dtype=np.float32)
        
        seq_flat = seq.reshape(seq.shape[0], -1)
        resampled = np.zeros((target_len, seq_flat.shape[1]), dtype=np.float32)
        
        for d in range(seq_flat.shape[1]):
            resampled[:, d] = np.interp(x_new, x_old, seq_flat[:, d])
        
        return resampled.reshape(target_len, seq.shape[1], seq.shape[2])
    
    def _temporal_rerank(self, candidates: List[_Candidate], data: _PreparedData) -> List[_Candidate]:
        """Soft-DTW временное сравнение."""
        if not candidates:
            return []
        
        out: List[_Candidate] = []
        
        for cand in candidates:
            try:
                a_seq, a_start, a_end = self._extract_window(cand.i, data)
                b_seq, b_start, b_end = self._extract_window(cand.j, data)
                
                cand.a_start, cand.a_end = a_start, a_end
                cand.b_start, cand.b_end = b_start, b_end
                
                # DTW стоимость
                dtw_cost = self._soft_dtw_znorm_cost(a_seq, b_seq)
                dtw_sim = math.exp(-dtw_cost)
                cand.dtw_similarity = dtw_sim
                
                # Энергетическая коррекция
                energy = 0.5 * (data.energy[cand.i] + data.energy[cand.j])
                motion_factor = 1.0
                if energy < self.config.motion_energy_thresholds[0]:
                    motion_factor = self.config.static_penalty
                elif energy > self.config.motion_energy_thresholds[1]:
                    motion_factor = self.config.dynamic_bonus
                
                # Итоговый скор
                cand.final_score = (0.35 * cand.base_similarity +
                                    0.10 * cand.rrf_score +
                                    0.55 * dtw_sim) * motion_factor
                cand.source = "dtw"
                out.append(cand)
                
            except Exception as e:
                logger.warning("DTW failed for pair (%d,%d): %s", cand.i, cand.j, e)
        
        out.sort(key=lambda c: c.final_score, reverse=True)
        return out
    
    def _soft_dtw_znorm_cost(self, a: np.ndarray, b: np.ndarray) -> float:
        """Z-нормализованная Soft-DTW стоимость."""
        # Решейп для torch
        a_flat = a.reshape(a.shape[0], -1)
        b_flat = b.reshape(b.shape[0], -1)
        
        at = torch.from_numpy(a_flat).float().to(self.device)
        bt = torch.from_numpy(b_flat).float().to(self.device)
        
        # Z-нормализация
        at = (at - at.mean(dim=0, keepdim=True)) / (at.std(dim=0, keepdim=True) + EPS)
        bt = (bt - bt.mean(dim=0, keepdim=True)) / (bt.std(dim=0, keepdim=True) + EPS)
        
        # Матрица расстояний
        dist = torch.cdist(at, bt, p=2) ** 2
        
        n, m = dist.shape
        gamma = self.config.soft_dtw_gamma
        band = max(1, int(math.ceil(max(n, m) * self.config.dtw_band_ratio)))
        
        r = torch.full((n + 1, m + 1), float('inf'), device=self.device)
        r[0, 0] = 0
        
        for i in range(1, n + 1):
            j_start = max(1, i - band)
            j_end = min(m, i + band)
            for j in range(j_start, j_end + 1):
                prev = torch.stack([r[i-1, j], r[i, j-1], r[i-1, j-1]])
                softmin = -gamma * torch.logsumexp(-prev / gamma, dim=0)
                r[i, j] = dist[i-1, j-1] + softmin
        
        cost = r[n, m] / (n + m)
        return float(cost.detach().cpu().item())
    
    def _apply_neural_rerank(self, candidates: List[_Candidate], data: _PreparedData) -> List[_Candidate]:
        """Применяет нейросетевой ранкер к топ-N кандидатам."""
        if not candidates or self.neural_reranker is None:
            return candidates
        
        top_n = min(self.config.max_neural_candidates, len(candidates))
        head = candidates[:top_n]
        tail = candidates[top_n:]
        
        try:
            neural_scores = self.neural_reranker(head, data)
            if len(neural_scores) != len(head):
                logger.warning("Neural reranker returned %d scores for %d candidates", 
                             len(neural_scores), len(head))
                return candidates
            
            alpha = self.config.neural_rerank_weight
            for c, ns in zip(head, neural_scores):
                ns = float(np.clip(ns, 0.0, 1.0))
                c.final_score = (1.0 - alpha) * c.final_score + alpha * ns
                c.source = "dtw+neural"
        except Exception as e:
            logger.warning("Neural rerank failed: %s", e)
            return candidates
        
        merged = head + tail
        merged.sort(key=lambda c: c.final_score, reverse=True)
        return merged
    
    def _fallback_from_fused(self, fused: List[_Candidate], data: _PreparedData) -> List[_Candidate]:
        """Fallback: возвращаем топ FAISS кандидатов."""
        fallback: List[_Candidate] = []
        
        for c in fused[:self.config.fallback_top_k]:
            c.final_score = 0.8 * c.base_similarity + 0.2 * c.rrf_score
            c.a_start = c.a_end = int(data.frame_indices[c.i])
            c.b_start = c.b_end = int(data.frame_indices[c.j])
            c.source = "fallback"
            fallback.append(c)
        
        return fallback
    
    def _segment_iou(self, a0: int, a1: int, b0: int, b1: int) -> float:
        """IOU двух временных отрезков."""
        inter = max(0, min(a1, b1) - max(a0, b0) + 1)
        union = max(a1, b1) - min(a0, b0) + 1
        return inter / max(union, 1)
    
    def _nms_and_format(self, candidates: List[_Candidate], data: _PreparedData, source: str) -> List[MatchResult]:
        """NMS и форматирование результатов."""
        if not candidates:
            return []
        
        kept: List[_Candidate] = []
        for c in sorted(candidates, key=lambda x: x.final_score, reverse=True):
            if c.final_score < self.config.final_similarity_threshold:
                continue
            
            # Проверяем перекрытие с уже сохраненными
            suppress = False
            for k in kept:
                iou_a = self._segment_iou(c.a_start, c.a_end, k.a_start, k.a_end)
                iou_b = self._segment_iou(c.b_start, c.b_end, k.b_start, k.b_end)
                
                if iou_a > self.config.nms_iou_threshold and iou_b > self.config.nms_iou_threshold:
                    suppress = True
                    break
                
                # Проверяем временное разнообразие
                if (abs(c.a_start - k.a_start) < self.config.diversity_min_frame_gap and
                    abs(c.b_start - k.b_start) < self.config.diversity_min_frame_gap):
                    suppress = True
                    break
            
            if not suppress:
                kept.append(c)
            
            if len(kept) >= self.config.max_results:
                break
        
        # Форматируем результаты
        fps = max(data.fps, 1.0)
        results: List[MatchResult] = []
        
        for c in kept:
            duration_frames = 0.5 * ((c.a_end - c.a_start + 1) + (c.b_end - c.b_start + 1))
            results.append(MatchResult(
                t1=float(data.frame_indices[c.i] / fps),
                t2=float(data.frame_indices[c.j] / fps),
                duration=float(duration_frames / fps),
                similarity=float(np.clip(c.final_score, 0.0, 1.0)),
                motion_type=int(data.motion_types[c.i]),
                motion_label=str(data.motion_labels[c.i]),
                v1_idx=int(data.track_ids[c.i]),
                v2_idx=int(data.track_ids[c.j]),
                frame_i=int(data.frame_indices[c.i]),
                frame_j=int(data.frame_indices[c.j]),
                energy=float(0.5 * (data.energy[c.i] + data.energy[c.j])),
                source=str(c.source or source)
            ))
        
        logger.info("NMS/format stage '%s': %d results", source, len(results))
        return results
    
    def _empty_data(self, fps: float) -> _PreparedData:
        """Возвращает пустые данные."""
        return _PreparedData(
            features=np.zeros((0, POSE_DIM + 1), dtype=np.float32),
            norm_xy=np.zeros((0, NUM_KEYPOINTS, 2), dtype=np.float32),
            frame_indices=np.zeros((0,), dtype=np.int32),
            track_ids=np.zeros((0,), dtype=np.int32),
            motion_types=np.zeros((0,), dtype=np.int32),
            motion_labels=[],
            energy=np.zeros((0,), dtype=np.float32),
            fps=fps,
            track_rows={},
            track_positions={},
        )


MotionMatcher = MotionMatcher2