"""
Neural Pose Verifier - семантическая проверка похожести поз через Ollama
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib

import numpy as np
import ollama

logger = logging.getLogger(__name__)

NUM_KEYPOINTS = 17
EPS = 1e-8

# Индексы ключевых точек (YOLO-pose format)
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

@dataclass
class PoseDescription:
    """Структурированное описание позы"""
    head_tilt: str = "neutral"
    head_turn: str = "forward"
    arms: str = "neutral"
    legs: str = "neutral"
    torso: str = "upright"
    gaze: str = "forward"
    
    def to_text(self) -> str:
        parts = []
        if self.head_tilt != "neutral":
            parts.append(f"head {self.head_tilt}")
        if self.head_turn != "forward":
            parts.append(f"looking {self.head_turn}")
        if self.torso != "upright":
            parts.append(f"torso {self.torso}")
        if self.arms != "neutral":
            parts.append(f"arms {self.arms}")
        if self.legs != "neutral":
            parts.append(f"legs {self.legs}")
        if self.gaze != "forward":
            parts.append(f"gaze {self.gaze}")
        return ", ".join(parts) if parts else "neutral pose"


class PoseVerifier:
    """Сравнивает две позы человека с помощью LLM (Ollama)."""
    
    def __init__(
        self,
        model: str = "llama3.2:1b",
        confidence_threshold: float = 0.3,
        cache_size: int = 10000,
        use_fast_mode: bool = True
    ):
        self.model = model
        self.confidence_threshold = float(confidence_threshold)
        self.use_fast_mode = use_fast_mode
        self._cache: Dict[int, float] = {}
        self._cache_maxsize = cache_size
        self._stats = {"hits": 0, "misses": 0, "fast_path": 0, "errors": 0}
        self._check_ollama()
    
    def _check_ollama(self) -> None:
        """Проверяет доступность Ollama"""
        try:
            response = ollama.list()
            models = response.get('models', [])
            if models:
                model_names = [m.get('name', m.get('model', str(m))) for m in models]
                logger.info("Ollama available. Models: %s", model_names)
            else:
                logger.warning("Ollama running but no models found. Run: ollama pull %s", self.model)
        except Exception as e:
            logger.warning("Ollama not available: %s. Will use fallback scoring.", e)
    
    def _coerce_pose(self, keypoints: Any) -> Optional[np.ndarray]:
        """Преобразует любой формат позы в (17, 3)."""
        if keypoints is None:
            return None
        
        try:
            arr = np.asarray(keypoints, dtype=np.float32)
        except Exception:
            return None
        
        if arr.ndim == 1:
            if arr.size == 34:
                arr = arr.reshape(17, 2)
            elif arr.size == 51:
                arr = arr.reshape(17, 3)
            else:
                return None
        
        elif arr.ndim == 2:
            if arr.shape[0] == NUM_KEYPOINTS:
                pass
            elif arr.shape[1] == NUM_KEYPOINTS:
                arr = arr.T
        
        elif arr.ndim == 3:
            if arr.shape[1] == NUM_KEYPOINTS:
                conf_means = arr[:, :, 2].mean(axis=1) if arr.shape[2] == 3 else np.ones(arr.shape[0])
                best_idx = int(np.argmax(conf_means))
                arr = arr[best_idx]
        
        if arr.shape == (NUM_KEYPOINTS, 2):
            conf = np.ones((NUM_KEYPOINTS, 1), dtype=np.float32)
            arr = np.concatenate([arr, conf], axis=1)
        
        if arr.shape != (NUM_KEYPOINTS, 3):
            return None
        
        return arr
    
    def _extract_pose_features(self, pose: np.ndarray) -> PoseDescription:
        """Извлекает структурированные признаки из позы."""
        desc = PoseDescription()
        xy = pose[:, :2]
        conf = pose[:, 2]
        valid = conf >= self.confidence_threshold
        
        # Анализ головы
        if valid[0] and valid[LEFT_SHOULDER] and valid[RIGHT_SHOULDER]:
            nose = xy[0]
            shoulder_center = 0.5 * (xy[LEFT_SHOULDER] + xy[RIGHT_SHOULDER])
            
            if nose[1] < shoulder_center[1] - 0.1:
                desc.head_tilt = "up"
            elif nose[1] > shoulder_center[1] + 0.05:
                desc.head_tilt = "down"
            
            if nose[0] < shoulder_center[0] - 0.05:
                desc.head_turn = "left"
            elif nose[0] > shoulder_center[0] + 0.05:
                desc.head_turn = "right"
        
        # Анализ рук
        if valid[LEFT_SHOULDER] and valid[LEFT_WRIST]:
            left_arm = xy[LEFT_WRIST][1] - xy[LEFT_SHOULDER][1]
        else:
            left_arm = 0
        
        if valid[RIGHT_SHOULDER] and valid[RIGHT_WRIST]:
            right_arm = xy[RIGHT_WRIST][1] - xy[RIGHT_SHOULDER][1]
        else:
            right_arm = 0
        
        avg_arm = (abs(left_arm) + abs(right_arm)) / 2
        if avg_arm > 0.3:
            desc.arms = "up" if left_arm < 0 or right_arm < 0 else "down"
        
        # Анализ ног
        if valid[LEFT_HIP] and valid[LEFT_ANKLE]:
            left_bent = valid[LEFT_KNEE] and xy[LEFT_KNEE][1] > xy[LEFT_HIP][1] + 0.3 * abs(xy[LEFT_ANKLE][1] - xy[LEFT_HIP][1])
        else:
            left_bent = False
        
        if valid[RIGHT_HIP] and valid[RIGHT_ANKLE]:
            right_bent = valid[RIGHT_KNEE] and xy[RIGHT_KNEE][1] > xy[RIGHT_HIP][1] + 0.3 * abs(xy[RIGHT_ANKLE][1] - xy[RIGHT_HIP][1])
        else:
            right_bent = False
        
        if left_bent or right_bent:
            desc.legs = "crouched"
        elif valid[LEFT_ANKLE] and valid[RIGHT_ANKLE]:
            if abs(xy[LEFT_ANKLE][0] - xy[RIGHT_ANKLE][0]) > 0.3:
                desc.legs = "apart"
        
        # Анализ торса
        if valid[LEFT_SHOULDER] and valid[LEFT_HIP]:
            torso_angle = abs(xy[LEFT_HIP][0] - xy[LEFT_SHOULDER][0]) / max(abs(xy[LEFT_HIP][1] - xy[LEFT_SHOULDER][1]), EPS)
            if torso_angle > 0.3:
                desc.torso = "bent"
        
        # Анализ взгляда
        if valid[0] and valid[3] and valid[4]:
            ear_center = 0.5 * (xy[3] + xy[4])
            if xy[0][0] < ear_center[0] - 0.03:
                desc.gaze = "left"
            elif xy[0][0] > ear_center[0] + 0.03:
                desc.gaze = "right"
        
        return desc
    
    def _fast_similarity(self, pose1: np.ndarray, pose2: np.ndarray) -> Optional[float]:
        """Быстрая оценка схожести без LLM."""
        if not self.use_fast_mode:
            return None
        
        p1 = self._extract_pose_features(pose1)
        p2 = self._extract_pose_features(pose2)
        
        differences = 0
        total_checks = 0
        
        for attr in ['arms', 'legs', 'torso']:
            total_checks += 1
            if getattr(p1, attr) != getattr(p2, attr):
                differences += 1
        
        if total_checks > 0:
            diff_ratio = differences / total_checks
            if diff_ratio > 0.5:
                return 0.2
            if diff_ratio > 0.3:
                return 0.4
            if differences == 0 and total_checks >= 2:
                return 0.85
        
        return None
    
    def _llm_compare(self, pose1: np.ndarray, pose2: np.ndarray) -> float:
        """Сравнивает позы через Ollama."""
        desc1 = self._extract_pose_features(pose1).to_text()
        desc2 = self._extract_pose_features(pose2).to_text()
        
        prompt = f"""Compare two human poses and rate similarity from 0 to 1.

Pose A: {desc1}
Pose B: {desc2}

Rate how similar these poses are:
- 1.0 = identical movement/position
- 0.7 = similar but some differences  
- 0.5 = somewhat similar
- 0.3 = different
- 0.0 = completely different

Return ONLY a number between 0 and 1 with two decimal places, nothing else."""

        try:
            start_time = time.time()
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 10}
            )
            elapsed = time.time() - start_time
            
            text = str(response.get("message", {}).get("content", "")).strip()
            
            match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
            if match:
                score = float(match.group(0))
            else:
                match = re.search(r"(\d+\.?\d*)", text)
                score = float(match.group(0)) if match else 0.5
            
            score = max(0.0, min(1.0, score))
            logger.debug(f"LLM compare took {elapsed:.2f}s, score: {score:.2f}")
            return score
            
        except Exception as e:
            logger.warning(f"Ollama compare failed: {e}")
            self._stats["errors"] += 1
            return 0.5
    
    def _get_cache_key(self, pose1: np.ndarray, pose2: np.ndarray) -> int:
        """Генерирует ключ кэша."""
        h1 = hashlib.blake2b(pose1.tobytes(), digest_size=8).hexdigest()
        h2 = hashlib.blake2b(pose2.tobytes(), digest_size=8).hexdigest()
        return hash((min(h1, h2), max(h1, h2)))
    
    def compare(self, pose1: Any, pose2: Any) -> float:
        """Сравнивает две позы и возвращает схожесть от 0 до 1."""
        p1 = self._coerce_pose(pose1)
        p2 = self._coerce_pose(pose2)
        
        if p1 is None or p2 is None:
            self._stats["errors"] += 1
            return 0.5
        
        cache_key = self._get_cache_key(p1, p2)
        if cache_key in self._cache:
            self._stats["hits"] += 1
            return self._cache[cache_key]
        
        self._stats["misses"] += 1
        
        fast_score = self._fast_similarity(p1, p2)
        if fast_score is not None:
            self._stats["fast_path"] += 1
            score = fast_score
        else:
            score = self._llm_compare(p1, p2)
        
        if len(self._cache) < self._cache_maxsize:
            self._cache[cache_key] = score
        
        return score
    
    def get_stats(self) -> Dict[str, int]:
        return self._stats.copy()
    
    def clear_cache(self) -> None:
        self._cache.clear()


class SimpleReranker:
    """Ранкер для интеграции с MotionMatcher."""
    
    def __init__(
        self,
        model: str = "llama3.2:1b",
        confidence_threshold: float = 0.3,
        batch_size: int = 50,
        max_candidates: int = 100
    ):
        self.verifier = PoseVerifier(
            model=model,
            confidence_threshold=confidence_threshold,
            use_fast_mode=True
        )
        self.batch_size = batch_size
        self.max_candidates = max_candidates
    
    def _extract_poses(self, data: Any) -> Optional[np.ndarray]:
        """Извлекает позы из data."""
        if hasattr(data, 'norm_xy') and data.norm_xy is not None:
            poses = data.norm_xy
            if isinstance(poses, np.ndarray) and len(poses) > 0:
                if poses.ndim == 3 and poses.shape[1] == NUM_KEYPOINTS:
                    return poses
                if poses.ndim == 2 and poses.shape[1] == NUM_KEYPOINTS * 2:
                    return poses.reshape(-1, NUM_KEYPOINTS, 2)
        
        if hasattr(data, 'features') and data.features is not None:
            feats = data.features
            if isinstance(feats, np.ndarray) and len(feats) > 0:
                if feats.shape[1] == NUM_KEYPOINTS * 2:
                    return feats.reshape(-1, NUM_KEYPOINTS, 2)
        
        return None
    
    def _extract_indices(self, candidate: Any) -> Tuple[Optional[int], Optional[int]]:
        """Извлекает индексы i, j из кандидата."""
        if hasattr(candidate, 'i') and hasattr(candidate, 'j'):
            return candidate.i, candidate.j
        if isinstance(candidate, (tuple, list)) and len(candidate) >= 2:
            return candidate[0], candidate[1]
        if isinstance(candidate, dict):
            return candidate.get('i'), candidate.get('j')
        return None, None
    
    def rerank(self, candidates: List[Any], data: Any, verbose: bool = True) -> List[float]:
        """Ранжирует кандидатов с помощью нейросети."""
        poses = self._extract_poses(data)
        if poses is None or len(poses) == 0:
            logger.error("Cannot extract poses from data")
            return [0.5] * len(candidates)
        
        if len(candidates) > self.max_candidates:
            logger.info(f"Limiting candidates from {len(candidates)} to {self.max_candidates}")
            candidates = candidates[:self.max_candidates]
        
        scores: List[float] = []
        total = len(candidates)
        
        if verbose:
            logger.info(f"Neural reranking {total} candidates...")
        
        for idx, candidate in enumerate(candidates):
            if verbose and (idx + 1) % self.batch_size == 0:
                logger.info(f"  Progress: {idx + 1}/{total}")
            
            try:
                i, j = self._extract_indices(candidate)
                if i is None or j is None:
                    scores.append(0.5)
                    continue
                
                if i < 0 or j < 0 or i >= len(poses) or j >= len(poses):
                    scores.append(0.5)
                    continue
                
                sim = self.verifier.compare(poses[i], poses[j])
                scores.append(sim)
                
            except Exception as e:
                logger.warning(f"Rerank failed at index {idx}: {e}")
                scores.append(0.5)
        
        if verbose:
            stats = self.verifier.get_stats()
            logger.info(
                f"Neural rerank complete. Avg: {np.mean(scores):.3f}, "
                f"Stats: hits={stats['hits']}, fast={stats['fast_path']}, errors={stats['errors']}"
            )
        
        return scores
    
    def clear_cache(self) -> None:
        self.verifier.clear_cache()


# Для обратной совместимости
__all__ = ["PoseVerifier", "SimpleReranker"]