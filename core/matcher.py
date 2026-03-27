#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F

class MotionMatcher:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.MIN_MATCH_GAP = 5.0
        
    def preprocess_pose(self, pose_data):
        """Нормализация позы"""
        kps = pose_data['keypoints'][:, :2]
        center = np.mean(kps, axis=0)
        centered = kps - center
        scale = np.max(np.abs(centered)) + 1e-5
        normalized = centered / scale
        return normalized
    
    def find_matches(self, poses_tensor, poses_meta, threshold=0.7, min_gap=3):
        """
        Поиск совпадений среди поз
        poses_tensor: torch.Tensor (N, 34)
        poses_meta: list[dict] с полями 't' (время), 'video_idx'
        threshold: порог схожести
        min_gap: минимальный интервал между совпадениями (сек)
        """
        if poses_tensor is None or len(poses_tensor) < 10:
            return []
        
        print(f"Поиск среди {len(poses_tensor)} поз, порог={threshold}, интервал={min_gap}")
        
        V = poses_tensor.to(dtype=torch.float32)
        V = V.view(len(V), -1)
        V = F.normalize(V, p=2, dim=1)
        
        sim_matrix = torch.mm(V, V.t())
        
        times = [m['t'] for m in poses_meta]
        T = torch.tensor(times, device=self.device)
        time_diff = torch.abs(T.unsqueeze(1) - T.unsqueeze(0))
        
        valid_mask = (sim_matrix > threshold) & (time_diff >= min_gap)
        valid_mask = torch.triu(valid_mask, diagonal=1)
        
        indices = torch.nonzero(valid_mask).cpu().numpy()
        scores = sim_matrix[valid_mask].cpu().numpy()
        
        print(f"Найдено сырых совпадений: {len(indices)}")
        
        matches = []
        for k in range(len(indices)):
            i, j = indices[k]
            matches.append({
                'm1_idx': i, 'm2_idx': j,
                't1': poses_meta[i]['t'],
                't2': poses_meta[j]['t'],
                'f1': poses_meta[i].get('f', int(poses_meta[i]['t'] * 30)),
                'f2': poses_meta[j].get('f', int(poses_meta[j]['t'] * 30)),
                'v1_idx': poses_meta[i].get('video_idx', 0),
                'v2_idx': poses_meta[j].get('video_idx', 0),
                'sim': float(scores[k]),
                'direction': poses_meta[i].get('dir', 'forward')
            })
        
        matches.sort(key=lambda x: x['sim'], reverse=True)
        
        used_times = set()
        unique = []
        for m in matches:
            t1_key = int(m['t1'] / self.MIN_MATCH_GAP)
            t2_key = int(m['t2'] / self.MIN_MATCH_GAP)
            if t1_key not in used_times and t2_key not in used_times:
                unique.append(m)
                used_times.add(t1_key)
                used_times.add(t2_key)
        
        print(f"Уникальных после дедупликации: {len(unique)}")
        return unique[:200]