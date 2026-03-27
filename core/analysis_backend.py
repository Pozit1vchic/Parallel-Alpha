#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import hashlib
import numpy as np
import torch
import cv2
from threading import Thread
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from core.engine import YoloEngine
from core.matcher import MotionMatcher

@dataclass
class AnalysisProgress:
    percent: float
    status: str
    current_frame: int
    total_frames: int
    current_video: str


class AnalysisBackend:
    def __init__(self):
        self.yolo = YoloEngine()
        self.matcher = MotionMatcher()
        self.yolo.load()
        
        self.analysis_running = False
        self.progress_callback = None
        self.result_callback = None
        
        self.BATCH_SIZE = 64
        self.CHUNK_SIZE = 5000
        self.preview_cache_dir = "cache/previews"
        os.makedirs(self.preview_cache_dir, exist_ok=True)
    
    def start_analysis(self, video_paths: List[str], settings: Dict[str, Any],
                       progress_callback=None, result_callback=None):
        self.analysis_running = True
        self.progress_callback = progress_callback
        self.result_callback = result_callback
        
        Thread(target=self._run_analysis, args=(video_paths, settings), daemon=True).start()
    
    def stop_analysis(self):
        self.analysis_running = False
    
    def _run_analysis(self, video_paths, settings):
        try:
            all_poses_meta = []
            all_poses_vecs = []
            
            threshold = settings.get('threshold', 70) / 100.0
            min_gap = settings.get('scene_interval', 3)
            quality = settings.get('quality', 'Средне')
            
            for idx, path in enumerate(video_paths):
                if not self.analysis_running:
                    break
                
                if self.progress_callback:
                    self.progress_callback(
                        idx / len(video_paths) * 50,
                        f"Анализ {os.path.basename(path)}",
                        0, 0, path
                    )
                
                poses_meta, poses_vecs = self._extract_poses_from_video(path, quality)
                
                for meta in poses_meta:
                    meta['video_idx'] = idx
                
                all_poses_meta.extend(poses_meta)
                all_poses_vecs.extend(poses_vecs)
            
            if not self.analysis_running or not all_poses_vecs:
                return
            
            poses_tensor = torch.tensor(
                np.array(all_poses_vecs),
                dtype=torch.float16 if self.yolo.use_fp16 else torch.float32,
                device=self.yolo.device
            )
            
            if self.progress_callback:
                self.progress_callback(80, "Поиск совпадений...", 0, 0, "")
            
            matches = self.matcher.find_matches(poses_tensor, all_poses_meta, threshold, min_gap)
            
            if self.result_callback:
                self.result_callback(matches, all_poses_meta, video_paths)
            
            if self.progress_callback:
                self.progress_callback(100, "Готово!", 0, 0, "")
                
        except Exception as e:
            print(f"Ошибка анализа: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_poses_from_video(self, path, quality):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        base_fps = {'Быстро': 10, 'Средне': 15, 'Макс': 30}[quality]
        
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        pixel_load = (h * w * fps) / 1e6
        res_factor = min(2.0, pixel_load / 50)
        adaptive_skip = max(1, int((fps / base_fps) * res_factor))
        skip = min(adaptive_skip, 10)
        
        frame_idx = 0
        poses_meta = []
        poses_vecs = []
        
        batch_frames = []
        batch_meta_pre = []
        video_hash = hashlib.md5(path.encode()).hexdigest()
        
        while cap.isOpened() and self.analysis_running:
            ret = cap.grab()
            if not ret:
                break
            
            if frame_idx % skip == 0:
                ret, frame = cap.retrieve()
                if ret:
                    h, w = frame.shape[:2]
                    target_size = 640
                    scale = min(target_size / w, target_size / h, 1.0)
                    if scale < 1.0:
                        new_w, new_h = int(w * scale), int(h * scale)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    batch_frames.append(frame)
                    batch_meta_pre.append({'frame': frame_idx, 'time': frame_idx / fps})
            
            dynamic_batch = self.yolo.get_dynamic_batch_size([f.shape[:2] for f in batch_frames], self.BATCH_SIZE)
            if len(batch_frames) >= dynamic_batch:
                poses_data = self.yolo.detect_batch(batch_frames)
                for i, (pose_data, meta) in enumerate(zip(poses_data, batch_meta_pre)):
                    if pose_data:
                        processed_vec = self.matcher.preprocess_pose(pose_data)
                        poses_meta.append({
                            't': meta['time'],
                            'f': meta['frame'],
                            'dir': pose_data.get('direction', 'forward'),
                            'vec': processed_vec.reshape(17, 2)
                        })
                        poses_vecs.append(processed_vec.flatten())
                        
                        cache_path = os.path.join(self.preview_cache_dir, f"{video_hash}_{meta['frame']}.jpg")
                        if not os.path.exists(cache_path):
                            resized_frame = cv2.resize(batch_frames[i], (320, 180), interpolation=cv2.INTER_AREA)
                            cv2.imwrite(cache_path, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                batch_frames.clear()
                batch_meta_pre.clear()
            
            frame_idx += 1
            
            if frame_idx % 30 == 0 and self.progress_callback:
                progress = (frame_idx / total_frames) * 100
                self.progress_callback(progress, f"Обработка...", frame_idx, total_frames, path)
        
        if batch_frames and self.analysis_running:
            poses_data = self.yolo.detect_batch(batch_frames)
            for i, (pose_data, meta) in enumerate(zip(poses_data, batch_meta_pre)):
                if pose_data:
                    processed_vec = self.matcher.preprocess_pose(pose_data)
                    poses_meta.append({
                        't': meta['time'],
                        'f': meta['frame'],
                        'dir': pose_data.get('direction', 'forward'),
                        'vec': processed_vec.reshape(17, 2)
                    })
                    poses_vecs.append(processed_vec.flatten())
        
        cap.release()
        return poses_meta, poses_vecs