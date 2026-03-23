"""UI package for Parallel Finder - main window and reusable widgets."""

from __future__ import annotations

from .main_window import ParallelFinderMainWindow
from .widgets import (
    AppSettingsData,
    CollapsiblePanel,
    MetricCard,
    SettingsDialog,
    SettingsRow,
    TimelineHeatmap,
    VideoPreview,
)

MainWindow = ParallelFinderMainWindow

__all__ = [
    "MainWindow",
    "ParallelFinderMainWindow",
    "SettingsRow",
    "MetricCard",
    "VideoPreview",
    "TimelineHeatmap",
    "CollapsiblePanel",
    "SettingsDialog",
    "AppSettingsData",
]
