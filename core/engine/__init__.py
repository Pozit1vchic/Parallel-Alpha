"""
Реэкспорт для обратной совместимости.
Старый код: from core.engine import YoloEngine  — работает без изменений.
"""
from core.engine.yolo_engine import YoloEngine
from core.engine.model_manager import ModelManager, MODELS_DIR, DEFAULT_MODEL_NAME, AVAILABLE_MODELS

__all__ = [
    "YoloEngine",
    "ModelManager",
    "MODELS_DIR",
    "DEFAULT_MODEL_NAME",
    "AVAILABLE_MODELS",
]