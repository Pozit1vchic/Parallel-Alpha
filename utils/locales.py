#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/locales.py — Система локализации Parallel Finder Alpha v13

Совместима с utils/__init__.py который импортирует:
  TRANSLATIONS, SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE, FALLBACK_LANGUAGE,
  Translator, check_sync, get_supported_languages, get_translator, t
"""
from __future__ import annotations

from typing import Any

# ── Константы ────────────────────────────────────────────────────────────────

DEFAULT_LANGUAGE:  str = "ru"
FALLBACK_LANGUAGE: str = "ru"
SUPPORTED_LANGUAGES: list[str] = ["ru", "en"]

# ── Строки переводов ──────────────────────────────────────────────────────────

TRANSLATIONS: dict[str, dict[str, str]] = {
    "ru": {
        # ── Общее ─────────────────────────────────────────────────────────
        "start":           "Старт",
        "stop":            "Стоп",
        "apply":           "Применить",
        "cancel":          "Отмена",
        "ok":              "ОК",
        "yes":             "Да",
        "no":              "Нет",
        "close":           "Закрыть",
        "warning":         "Предупреждение",
        "error":           "Ошибка",
        "info":            "Информация",
        "active":          "активна",
        "dont_show_again": "Больше не показывать",

        # ── Заголовки панелей ─────────────────────────────────────────────
        "source_panel":      "Источник",
        "analysis_settings": "Настройки анализа",
        "results_panel":     "Результаты",
        "progress_panel":    "Прогресс",
        "comparison_panel":  "Сравнение",
        "timeline_panel":    "Таймлайн",

        # ── Настройки ─────────────────────────────────────────────────────
        "similarity_threshold":   "Порог схожести",
        "threshold_hint":         "Насколько похожи движения (выше = строже)",
        "threshold_warning_low":  "⚠ Низкий порог — найдёт много лишнего",
        "threshold_warning_high": "⚠ Очень высокий — может ничего не найти",
        "analysis_speed":         "Скорость анализа",
        "quality_hint":           "Быстро = меньше кадров, Максимум = все кадры",
        "fast":                   "Быстро",
        "medium":                 "Средне",
        "maximum":                "Максимум",
        "advanced_settings":      "Расширенные настройки",
        "scene_interval":         "Мин. интервал сцены",
        "scene_interval_hint":    "Минимальное время между разными сценами (сек)",
        "min_gap":                "Мин. разрыв между совпадениями",
        "min_gap_hint":           "Игнорировать повторения ближе этого времени (сек)",
        "use_mirror":             "Искать зеркальные позы",
        "use_mirror_hint":        "Находить движения, зеркально отражённые по горизонтали",
        "use_body_weights":       "Веса частей тела",
        "use_body_weights_hint":  "Руки и ноги важнее туловища при сравнении",

        # ── Источник ──────────────────────────────────────────────────────
        "no_video":        "Видео не добавлены",
        "drop_hint":       "Перетащите видео сюда или нажмите кнопки ниже",
        "select_video":    "Выбрать видео",
        "select_folder":   "Выбрать папку",
        "clear":           "Очистить",
        "remove_selected": "Удалить выбранный",
        "find_by_photo":   "Найти человека по фото",
        "files":           "Файлов",
        "duration":        "Длительность",
        "frames":          "Кадров",

        # ── Статистика ────────────────────────────────────────────────────
        "stat_frames":     "Кадров",
        "stat_repeats":    "Повторов",
        "stat_duration":   "Длительность",
        "stat_similarity": "Схожесть",
        "stat_remaining":  "Осталось",
        "stat_progress":   "Прогресс",

        # ── Прогресс ──────────────────────────────────────────────────────
        "waiting":         "Ожидание",
        "extracting":      "Извлечение поз...",
        "building_tensor": "Сборка данных...",
        "searching":       "Поиск совпадений...",
        "done":            "Готово",

        # ── Результаты ────────────────────────────────────────────────────
        "no_results":   "Результатов пока нет.\nЗапустите анализ.",
        "prev":         "◀ Пред.",
        "next":         "След. ▶",
        "export_json":  "JSON",
        "export_txt":   "TXT",
        "export_edl":   "EDL",

        # ── Модели ────────────────────────────────────────────────────────
        "models":               "Модели",
        "model_selector_title": "Выбор модели",
        "choose_yolo_model":    "Выберите модель YOLO",
        "local_model":          "локальная",
        "will_download":        "будет скачана",
        "loading_model_name":   "Загрузка {name}...",
        "loading_model_status": "Загрузка модели {name}...",

        # ── Язык ──────────────────────────────────────────────────────────
        "lang_button": "RU",

        # ── Foolproof — ошибки и предупреждения ───────────────────────────
        "err_model_loading":         "Модель уже загружается",
        "err_model_during_analysis": "Остановите анализ сначала",
        "err_model_not_found":       "Модель не найдена",
        "err_model_corrupt":         "Файл модели повреждён",
        "err_no_disk_space":         "Не хватает места на диске",
        "err_no_internet":           "Проверьте подключение к интернету",
        "err_pytorch_old":           "Обновите PyTorch",
        "err_file_busy":             "Файл используется другой программой",
        "err_file_corrupt":          "Не удаётся открыть видео",
        "err_file_empty":            "Файл пуст",
        "err_no_video_stream":       "Файл не содержит видео",
        "err_no_access":             "Нет доступа к файлу",
        "err_file_too_large":        "Очень большой файл (>{size} ГБ)",
        "err_video_too_long":        "Очень длинное видео (>{hours} ч)",
        "err_video_too_short":       "Видео слишком короткое",
        "err_bad_resolution":        "Некорректное разрешение видео",
        "err_folder_empty":          "В папке нет видео",
        "err_folder_no_access":      "Нет доступа к папке",
        "err_network_folder":        "Сетевая папка — возможен медленный доступ",
        "err_too_many_videos":       "Много видео ({n}) — анализ займёт долго",
        "err_non_video_files":       "Некоторые файлы пропущены (только видео)",
        "err_no_memory":             "Мало оперативной памяти — возможны проблемы",
        "err_no_vram":               "Недостаточно VRAM, переключено на CPU",
        "err_oom":                   "Нехватка памяти, попробуйте меньшую модель",
        "err_threshold_low":         "Порог очень низкий — будет много результатов",
        "err_threshold_high":        "Порог очень высокий — результатов может не быть",
        "err_no_video_to_start":     "Сначала добавьте видео",
        "err_model_not_loaded":      "Дождитесь загрузки модели",
        "err_close_during_analysis": "Остановить анализ и выйти?",
        "err_no_results_export":     "Сначала выполните анализ",
        "err_photo_not_image":       "Выберите изображение",
        "err_photo_no_person":       "На фото не обнаружено людей",
        "err_file_overwrite":        "Файл уже существует. Перезаписать?",
        "err_path_too_long":         "Путь слишком длинный (>{n} символов)",
        "err_remove_last_video":     "Удалить последнее видео и очистить результаты?",
        "warn_many_results":         "Найдено очень много результатов ({n}). Показаны первые {max}.",
        "warn_slow_analysis":        "Анализ занимает долго. Оставшееся время: {eta}",
        "warn_quality_weak_hw":      "Режим Максимум может быть медленным на вашем железе",
        "warn_all_options_on":       "Включены все опции — анализ будет дольше",
    },

    "en": {
        # ── General ───────────────────────────────────────────────────────
        "start":           "Start",
        "stop":            "Stop",
        "apply":           "Apply",
        "cancel":          "Cancel",
        "ok":              "OK",
        "yes":             "Yes",
        "no":              "No",
        "close":           "Close",
        "warning":         "Warning",
        "error":           "Error",
        "info":            "Info",
        "active":          "active",
        "dont_show_again": "Don't show again",

        # ── Panel titles ──────────────────────────────────────────────────
        "source_panel":      "Source",
        "analysis_settings": "Analysis Settings",
        "results_panel":     "Results",
        "progress_panel":    "Progress",
        "comparison_panel":  "Comparison",
        "timeline_panel":    "Timeline",

        # ── Settings ──────────────────────────────────────────────────────
        "similarity_threshold":   "Similarity threshold",
        "threshold_hint":         "How similar movements must be (higher = stricter)",
        "threshold_warning_low":  "⚠ Low threshold — will find too many results",
        "threshold_warning_high": "⚠ Very high — may find nothing",
        "analysis_speed":         "Analysis speed",
        "quality_hint":           "Fast = fewer frames, Max = all frames",
        "fast":                   "Fast",
        "medium":                 "Medium",
        "maximum":                "Maximum",
        "advanced_settings":      "Advanced settings",
        "scene_interval":         "Min scene interval",
        "scene_interval_hint":    "Minimum time between different scenes (sec)",
        "min_gap":                "Min gap between matches",
        "min_gap_hint":           "Ignore repetitions closer than this time (sec)",
        "use_mirror":             "Find mirrored poses",
        "use_mirror_hint":        "Find horizontally mirrored movements",
        "use_body_weights":       "Body part weights",
        "use_body_weights_hint":  "Arms and legs matter more than torso",

        # ── Source ────────────────────────────────────────────────────────
        "no_video":        "No video added",
        "drop_hint":       "Drop video here or use buttons below",
        "select_video":    "Select video",
        "select_folder":   "Select folder",
        "clear":           "Clear",
        "remove_selected": "Remove selected",
        "find_by_photo":   "Find person by photo",
        "files":           "Files",
        "duration":        "Duration",
        "frames":          "Frames",

        # ── Stats ─────────────────────────────────────────────────────────
        "stat_frames":     "Frames",
        "stat_repeats":    "Repeats",
        "stat_duration":   "Duration",
        "stat_similarity": "Similarity",
        "stat_remaining":  "Remaining",
        "stat_progress":   "Progress",

        # ── Progress ──────────────────────────────────────────────────────
        "waiting":         "Waiting",
        "extracting":      "Extracting poses...",
        "building_tensor": "Building data...",
        "searching":       "Finding matches...",
        "done":            "Done",

        # ── Results ───────────────────────────────────────────────────────
        "no_results":   "No results yet.\nRun analysis.",
        "prev":         "◀ Prev",
        "next":         "Next ▶",
        "export_json":  "JSON",
        "export_txt":   "TXT",
        "export_edl":   "EDL",

        # ── Models ────────────────────────────────────────────────────────
        "models":               "Models",
        "model_selector_title": "Model selection",
        "choose_yolo_model":    "Choose YOLO model",
        "local_model":          "local",
        "will_download":        "will download",
        "loading_model_name":   "Loading {name}...",
        "loading_model_status": "Loading model {name}...",

        # ── Language ──────────────────────────────────────────────────────
        "lang_button": "EN",

        # ── Foolproof errors ──────────────────────────────────────────────
        "err_model_loading":         "Model is already loading",
        "err_model_during_analysis": "Stop analysis first",
        "err_model_not_found":       "Model not found",
        "err_model_corrupt":         "Model file is corrupted",
        "err_no_disk_space":         "Not enough disk space",
        "err_no_internet":           "Check your internet connection",
        "err_pytorch_old":           "Update PyTorch",
        "err_file_busy":             "File is used by another program",
        "err_file_corrupt":          "Cannot open video",
        "err_file_empty":            "File is empty",
        "err_no_video_stream":       "File has no video stream",
        "err_no_access":             "No access to file",
        "err_file_too_large":        "File is very large (>{size} GB)",
        "err_video_too_long":        "Video is very long (>{hours} h)",
        "err_video_too_short":       "Video is too short",
        "err_bad_resolution":        "Invalid video resolution",
        "err_folder_empty":          "No video files in folder",
        "err_folder_no_access":      "No access to folder",
        "err_network_folder":        "Network folder — may be slow",
        "err_too_many_videos":       "Many videos ({n}) — analysis will take long",
        "err_non_video_files":       "Some files skipped (video only)",
        "err_no_memory":             "Low RAM — possible issues",
        "err_no_vram":               "Not enough VRAM, switched to CPU",
        "err_oom":                   "Out of memory, try a smaller model",
        "err_threshold_low":         "Very low threshold — many results expected",
        "err_threshold_high":        "Very high threshold — may find nothing",
        "err_no_video_to_start":     "Add video first",
        "err_model_not_loaded":      "Wait for model to load",
        "err_close_during_analysis": "Stop analysis and exit?",
        "err_no_results_export":     "Run analysis first",
        "err_photo_not_image":       "Select an image",
        "err_photo_no_person":       "No people detected in photo",
        "err_file_overwrite":        "File already exists. Overwrite?",
        "err_path_too_long":         "Path is too long (>{n} chars)",
        "err_remove_last_video":     "Remove last video and clear results?",
        "warn_many_results":         "Found many results ({n}). Showing first {max}.",
        "warn_slow_analysis":        "Analysis is taking long. ETA: {eta}",
        "warn_quality_weak_hw":      "Maximum mode may be slow on your hardware",
        "warn_all_options_on":       "All options enabled — analysis will be slower",
    },
}

# ── Внутреннее состояние ─────────────────────────────────────────────────────

_current_lang: str = DEFAULT_LANGUAGE


# ── Основная функция перевода ─────────────────────────────────────────────────

def t(key: str, **kwargs: Any) -> str:
    """
    Получить перевод по ключу с подстановкой параметров.

    Примеры:
        t("start")                       → "Старт"
        t("err_file_too_large", size=5)  → "Очень большой файл (>5 ГБ)"
    """
    lang_dict = TRANSLATIONS.get(_current_lang, TRANSLATIONS[FALLBACK_LANGUAGE])
    text = lang_dict.get(key)

    # Fallback на другой язык
    if text is None:
        fallback_dict = TRANSLATIONS.get(FALLBACK_LANGUAGE, {})
        text = fallback_dict.get(key, key)

    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError, IndexError):
            pass

    return text


def set_lang(lang: str) -> None:
    """Установить текущий язык."""
    global _current_lang
    if lang in TRANSLATIONS:
        _current_lang = lang


def get_lang() -> str:
    """Получить текущий язык."""
    return _current_lang


def get_supported_languages() -> list[str]:
    """Список поддерживаемых языков."""
    return list(SUPPORTED_LANGUAGES)


# ── Класс Translator (совместимость с __init__.py) ───────────────────────────

class Translator:
    """
    Объектный интерфейс к системе локализации.
    Используется как альтернатива глобальной функции t().
    """

    def __init__(self, lang: str = DEFAULT_LANGUAGE) -> None:
        self._lang = lang if lang in TRANSLATIONS else DEFAULT_LANGUAGE

    @property
    def lang(self) -> str:
        return self._lang

    @lang.setter
    def lang(self, value: str) -> None:
        if value in TRANSLATIONS:
            self._lang = value

    def __call__(self, key: str, **kwargs: Any) -> str:
        """translator("key", param=value)"""
        return self.get(key, **kwargs)

    def get(self, key: str, **kwargs: Any) -> str:
        """Получить перевод."""
        lang_dict = TRANSLATIONS.get(self._lang, TRANSLATIONS[FALLBACK_LANGUAGE])
        text = lang_dict.get(key)
        if text is None:
            text = TRANSLATIONS.get(FALLBACK_LANGUAGE, {}).get(key, key)
        if kwargs:
            try:
                text = text.format(**kwargs)
            except (KeyError, ValueError, IndexError):
                pass
        return text

    def set_lang(self, lang: str) -> None:
        if lang in TRANSLATIONS:
            self._lang = lang

    def keys(self) -> list[str]:
        """Все доступные ключи для текущего языка."""
        return list(TRANSLATIONS.get(self._lang, {}).keys())


# ── Утилиты проверки ─────────────────────────────────────────────────────────

def check_sync() -> dict[str, list[str]]:
    """
    Проверить синхронизацию ключей между языками.

    Returns
    -------
    dict
        {lang: [список ключей которых нет в этом языке]}
    """
    all_keys: set[str] = set()
    for lang_dict in TRANSLATIONS.values():
        all_keys.update(lang_dict.keys())

    missing: dict[str, list[str]] = {}
    for lang, lang_dict in TRANSLATIONS.items():
        absent = sorted(all_keys - set(lang_dict.keys()))
        if absent:
            missing[lang] = absent

    return missing


def get_translator(lang: str = DEFAULT_LANGUAGE) -> Translator:
    """Фабричная функция — вернуть Translator для заданного языка."""
    return Translator(lang)


# ── Инициализация из конфига (опционально) ───────────────────────────────────

def _try_load_lang_from_config() -> None:
    """Попытаться загрузить язык из config.json при импорте."""
    try:
        import json
        from pathlib import Path
        cfg_path = Path(__file__).resolve().parent.parent / "config.json"
        if cfg_path.exists():
            cfg  = json.loads(cfg_path.read_text(encoding="utf-8"))
            lang = cfg.get("language", {}).get("current", DEFAULT_LANGUAGE)
            set_lang(lang)
    except Exception:
        pass


_try_load_lang_from_config()