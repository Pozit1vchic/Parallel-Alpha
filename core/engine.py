#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
from ultralytics import YOLO

class YoloEngine:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.use_fp16 = self.device == 'cuda'
        
    def load(self, model_path='yolov8m-pose.pt'):
        if self.model is None:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"✅ Ускоритель: {self.device.upper()} | FP16: {self.use_fp16}")

    def detect_batch(self, frames_batch):
        try:
            results = self.model.predict(
                frames_batch,
                imgsz=640,
                verbose=False,
                half=self.use_fp16,
                conf=0.1,
                stream=False
            )

            batch_poses = []
            for res in results:
                pose_data = None
                if res.keypoints is not None and len(res.keypoints.data) > 0:
                    kp = res.keypoints.data[0].cpu().numpy()
                    if kp.shape[0] >= 17:
                        vis_kps = kp[:17][kp[:17, 2] > 0.3]
                        
                        h, w = res.orig_shape
                        if len(vis_kps) > 2:
                            min_x, min_y = np.min(vis_kps[:, :2], axis=0)
                            max_x, max_y = np.max(vis_kps[:, :2], axis=0)
                            bbox = [min_x, min_y, max_x, max_y]
                        else:
                            bbox = [0, 0, w, h]

                        direction = self.classify_direction(kp[:17])
                        
                        pose_data = {
                            'keypoints': kp[:17],
                            'direction': direction,
                            'confidence': float(np.mean(kp[:17, 2])),
                            'bbox': bbox,
                            'orig_w': w,
                            'orig_h': h
                        }
                        
                batch_poses.append(pose_data)
            return batch_poses
        except Exception:
            return [None] * len(frames_batch)
    
    def classify_direction(self, landmarks_xy_conf):
        try:
            h, w = 1.0, 1.0
            landmarks = np.copy(landmarks_xy_conf)
            landmarks[:, 0] /= w
            landmarks[:, 1] /= h

            L_SHOULDER, R_SHOULDER = 5, 6
            if landmarks[L_SHOULDER, 2] < 0.3 or landmarks[R_SHOULDER, 2] < 0.3:
                return "unknown"
            shoulder_center_x = (landmarks[L_SHOULDER, 0] + landmarks[R_SHOULDER, 0]) / 2
            head_offset = landmarks[0, 0] - shoulder_center_x
            
            if abs(head_offset) < 0.05:
                return "forward"
            elif head_offset > 0:
                return "right"
            else:
                return "left"
        except:
            return "unknown"
    
    def get_dynamic_batch_size(self, frame_sizes, default_batch=64):
        if not frame_sizes:
            return default_batch
        
        avg_area = np.mean([h * w for h, w in frame_sizes]) / 1e6
        
        vram_factor = 1.0
        if torch.cuda.is_available():
            try:
                vram_used = torch.cuda.memory_reserved() / 1e9
                vram_factor = max(0.3, (12 - vram_used) / 8.0)
            except:
                pass
        
        if avg_area > 1.5:
            batch = int(default_batch * 0.4 * vram_factor)
        elif avg_area > 0.8:
            batch = int(default_batch * 0.7 * vram_factor)
        elif avg_area < 0.3:
            batch = int(default_batch * 1.8 * vram_factor)
        else:
            batch = default_batch
        
        return max(4, min(batch, 192))