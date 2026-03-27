#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import gzip
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from core.engine import YoloEngine
from core.matcher import MotionMatcher


class ProjectManager:
    """Управление проектами: сохранение, загрузка, кэширование"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.current_project_path = None
        
    def get_project_path(self, name: str) -> Path:
        """Получить путь к файлу проекта"""
        return self.cache_dir / f"{name}.pfp"
    
    def save_project(self, path: str, data: Dict[str, Any]) -> bool:
        """Сохранить проект"""
        try:
            with gzip.open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.current_project_path = path
            return True
        except Exception as e:
            print(f"Ошибка сохранения проекта: {e}")
            return False
    
    def load_project(self, path: str) -> Optional[Dict[str, Any]]:
        """Загрузить проект"""
        try:
            with gzip.open(path, 'rb') as f:
                data = pickle.load(f)
            self.current_project_path = path
            return data
        except Exception as e:
            print(f"Ошибка загрузки проекта: {e}")
            return None
    
    def get_cache_key(self, video_path: str) -> str:
        """Получить ключ кэша для видео"""
        resolved = Path(video_path).expanduser().resolve()
        stat = resolved.stat()
        payload = f"{resolved.as_posix()}::{stat.st_size}::{stat.st_mtime_ns}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()
    
    def get_cache_path(self, video_path: str) -> Path:
        """Получить путь к кэшу для видео"""
        key = self.get_cache_key(video_path)
        return self.cache_dir / f"{key}.pkl.gz"
    
    def save_poses_cache(self, video_path: str, poses_meta: List[Dict], poses_vecs: List[np.ndarray]) -> bool:
        """Сохранить позы в кэш"""
        cache_path = self.get_cache_path(video_path)
        try:
            data = {
                'poses_meta': poses_meta,
                'poses_vecs': poses_vecs,
                'video_path': video_path
            }
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"Ошибка сохранения кэша: {e}")
            return False
    
    def load_poses_cache(self, video_path: str) -> Optional[tuple]:
        """Загрузить позы из кэша"""
        cache_path = self.get_cache_path(video_path)
        if not cache_path.exists():
            return None
        
        try:
            with gzip.open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data.get('poses_meta', []), data.get('poses_vecs', [])
        except Exception as e:
            print(f"Ошибка загрузки кэша: {e}")
            return None
    
    def clear_cache(self):
        """Очистить кэш"""
        for file in self.cache_dir.glob("*.pkl.gz"):
            try:
                file.unlink()
            except:
                pass
        print(f"Кэш очищен: {self.cache_dir}")


__all__ = ['ProjectManager']