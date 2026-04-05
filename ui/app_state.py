from __future__ import annotations

import threading
from dataclasses import dataclass, field

import torch


@dataclass
class AppState:
    """Единое состояние приложения. Без tkinter виджетов."""

    # ── Очередь видео ─────────────────────────────────────────────────────
    video_queue:           list[str]         = field(default_factory=list)
    batch_mode:            bool              = False
    video_durations_cache: dict[str, float]  = field(default_factory=dict)

    # ── Результаты ────────────────────────────────────────────────────────
    matches:               list[dict]        = field(default_factory=list)
    poses_tensor:          torch.Tensor | None = None
    poses_meta:            list              = field(default_factory=list)
    current_match:         int               = 0

    # ── Категории ─────────────────────────────────────────────────────────
    found_categories:      list[str]         = field(default_factory=list)
    active_filter_cat:     str               = ""

    # ── Фото-поиск ────────────────────────────────────────────────────────
    use_photo_search:      bool              = False
    ref_photos:            list[str]         = field(default_factory=list)

    # ── Модель ────────────────────────────────────────────────────────────
    current_model_name:    str               = "yolov8n-pose.pt"

    # ── Потоки (thread-safe флаги) ────────────────────────────────────────
    analysis_event:        threading.Event   = field(
        default_factory=threading.Event)
    model_event:           threading.Event   = field(
        default_factory=threading.Event)

    # ── Batch ─────────────────────────────────────────────────────────────
    current_batch_index:   int               = 0
    total_batch_videos:    int               = 0

    # ── Производительность ────────────────────────────────────────────────
    BATCH_SIZE:            int               = 32
    CHUNK_SIZE:            int               = 3000
    CHUNK_OVERLAP:         int               = 300
    MIN_MATCH_GAP:         float             = 5.0
    max_matches_per_chunk: int               = 500_000
    max_total_matches:     int               = 10_000_000
    max_unique_results:    int               = 1000
    junk_ratio:            float             = 0.20

    # ── Timeline ──────────────────────────────────────────────────────────
    timeline_zoom:         float             = 1.0
    timeline_pan:          float             = 0.0

    # ── Properties (thread-safe) ──────────────────────────────────────────

    @property
    def analysis_running(self) -> bool:
        return self.analysis_event.is_set()

    @analysis_running.setter
    def analysis_running(self, value: bool) -> None:
        if value:
            self.analysis_event.set()
        else:
            self.analysis_event.clear()

    @property
    def model_loading(self) -> bool:
        return self.model_event.is_set()

    @model_loading.setter
    def model_loading(self, value: bool) -> None:
        if value:
            self.model_event.set()
        else:
            self.model_event.clear()

    def reset(self) -> None:
        """Сброс результатов перед новым анализом."""
        self.matches.clear()
        self.poses_tensor     = None
        self.poses_meta       = []
        self.current_match    = 0
        self.found_categories = []
        self.active_filter_cat = ""