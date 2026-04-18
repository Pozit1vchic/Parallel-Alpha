#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/project.py — ProjectManager для Parallel Finder.

Исправления относительно предыдущей версии:
- pickle.HIGHEST_PROTOCOL → protocol=4 (совместимость Python 3.4+)
- use_body_weights включён в ключ кеша поз (инвалидация при смене настроек)
- get_visible_matches → один проход вместо трёх
- AnalysisSettings.scene_interval → property-алиас min_gap
- cache_size_bytes → предупреждение если вызывается долго
- reorder_videos → понятное сообщение об ошибке
- _get_match → walrus operator убран (явная проверка читаемее)
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import os
import pickle
import shutil
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from utils.constants import (
    APP_BUILD_VERSION,
    CACHE_DIR,
    POSE_CACHE_DIR,
    PREVIEW_CACHE_DIR,
    PROJECTS_DIR,
)

log = logging.getLogger(__name__)

PROJECT_FORMAT_VERSION = "1.0"
PROJECT_SUFFIX         = ".pfp"
PICKLE_PROTOCOL        = 4   # Python 3.4+ — стабилен между версиями

# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VideoEntry:
    """Метаданные одного видео в проекте."""
    path:        str
    video_idx:   int   = 0
    duration:    float = 0.0
    fps:         float = 0.0
    frame_count: int   = 0
    cache_key:   str   = ""
    label:       str   = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = Path(self.path).name


@dataclass
class AnalysisSettings:
    """
    Параметры анализа.

    scene_interval — property-алиас для min_gap.
    Хранится только min_gap — нет рассинхрона двух полей.
    """
    threshold:        float = 0.75
    min_gap:          float = 3.0
    quality:          str   = "Средне"
    use_mirror:       bool  = False
    model_name:       str   = ""
    use_body_weights: bool  = True

    @property
    def scene_interval(self) -> float:
        return self.min_gap

    @scene_interval.setter
    def scene_interval(self, v: float) -> None:
        self.min_gap = v


@dataclass
class MatchRecord:
    """Одно совпадение из matcher.find_matches()."""
    sim:        float = 0.0
    sim_raw:    float = 0.0
    t1:         float = 0.0
    t2:         float = 0.0
    f1:         int   = 0
    f2:         int   = 0
    v1_idx:     int   = 0
    v2_idx:     int   = 0
    m1_idx:     int   = -1
    m2_idx:     int   = -1
    direction:  str   = "forward"
    hidden:     bool  = False
    excluded:   bool  = False
    confirmed:  bool  = False

    @classmethod
    def from_dict(cls, d: dict) -> "MatchRecord":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FilterState:
    """Состояние панели фильтров."""
    sort_by:             str   = "sim"
    sort_desc:           bool  = True
    min_sim:             float = 0.0
    direction_filter:    str   = "all"
    show_hidden:         bool  = False
    show_excluded:       bool  = False
    show_confirmed_only: bool  = False
    search_query:        str   = ""


@dataclass
class ProjectState:
    """Полное состояние проекта — сериализуется в .pfp."""
    project_id:        str   = field(default_factory=lambda: str(uuid.uuid4()))
    name:              str   = "Без названия"
    created_at:        float = field(default_factory=time.time)
    updated_at:        float = field(default_factory=time.time)
    format_version:    str   = PROJECT_FORMAT_VERSION
    app_version:       str   = APP_BUILD_VERSION
    videos:            list[VideoEntry]    = field(default_factory=list)
    settings:          AnalysisSettings   = field(default_factory=AnalysisSettings)
    matches:           list[MatchRecord]   = field(default_factory=list)
    motion_groups:     dict[str, list[int]] = field(default_factory=dict)
    filter_state:      FilterState         = field(default_factory=FilterState)
    language:          str                 = "ru"
    current_match_idx: int                 = 0
    extra:             dict[str, Any]      = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("Не удалось создать директорию %s: %s", path, exc)
    return path


def _atomic_write(path: Path, data: bytes) -> bool:
    """Атомарная запись: tmp → rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
        return True
    except OSError as exc:
        log.error("Ошибка записи %s: %s", path, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _pickle_gz(obj: Any) -> bytes:
    """Сериализовать в gzip-pickle bytes (protocol=4)."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=5) as gz:
        pickle.dump(obj, gz, protocol=PICKLE_PROTOCOL)
    return buf.getvalue()


def _unpickle_gz(data: bytes) -> Any:
    """Десериализовать из gzip-pickle bytes."""
    with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as gz:
        # Десериализация из gzip-pickle (сжатие для экономии места в файлах проекта)
        return pickle.load(gz)


# ═══════════════════════════════════════════════════════════════════════════════
# Основной класс
# ═══════════════════════════════════════════════════════════════════════════════

class ProjectManager:
    """
    Управление проектами Parallel Finder.

    Совместимый API:
        get_project_path(name)
        save_project(path, data)
        load_project(path)
        get_cache_key(video_path, settings=None)
        get_cache_path(video_path, kind)
        save_poses_cache(video, meta, vecs)
        load_poses_cache(video)
        clear_cache(kind)

    Новый API:
        new_project(name) / save_state / load_state
        add_video / remove_video / reorder_videos
        set_matches / get_visible_matches
        hide_match / exclude_match / confirm_match
        save_matches_cache / load_matches_cache
        get_preview_path
        get_stats / cache_size_bytes
    """

    _PROJECTS_DIR:      Path = PROJECTS_DIR
    _POSE_CACHE_DIR:    Path = POSE_CACHE_DIR
    _PREVIEW_CACHE_DIR: Path = PREVIEW_CACHE_DIR
    _MATCHES_CACHE_DIR: Path = CACHE_DIR / "matches"
    _GROUPS_CACHE_DIR:  Path = CACHE_DIR / "motion_groups"
    _FILTER_CACHE_DIR:  Path = CACHE_DIR / "filters"

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self._pose_cache_dir = _ensure(
            Path(cache_dir) if cache_dir is not None
            else self._POSE_CACHE_DIR
        )
        _ensure(self._PROJECTS_DIR)
        _ensure(self._PREVIEW_CACHE_DIR)
        _ensure(self._MATCHES_CACHE_DIR)
        _ensure(self._GROUPS_CACHE_DIR)
        _ensure(self._FILTER_CACHE_DIR)

        self.current_project_path: Path | None = None
        self._state = ProjectState()
        self.cache_dir = self._pose_cache_dir  # совместимость

    # ── § 1. Совместимый API ─────────────────────────────────────────────

    def get_project_path(self, name: str) -> Path:
        safe = "".join(
            c if c.isalnum() or c in " ._-" else "_" for c in name
        ).strip()
        return self._PROJECTS_DIR / f"{safe}{PROJECT_SUFFIX}"

    def save_project(self, path: str | Path, data: dict[str, Any]) -> bool:
        """Сохранить проект в старом dict-формате."""
        dst = Path(path)
        _ensure(dst.parent)
        try:
            ok = _atomic_write(dst, _pickle_gz(data))
            if ok:
                self.current_project_path = dst
                log.info("Проект сохранён: %s", dst)
            return ok
        except Exception as exc:
            log.error("save_project(%s): %s", dst, exc)
            return False

    def load_project(self, path: str | Path) -> dict[str, Any] | None:
        """Загрузить проект. Fallback на JSON при повреждении pickle."""
        src = Path(path)
        if not src.exists():
            log.warning("load_project: не найден %s", src)
            return None
        try:
            data = _unpickle_gz(src.read_bytes())
            self.current_project_path = src
            return data
        except Exception as exc:
            log.warning("load_project pickle failed: %s — пробуем JSON", exc)
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
            self.current_project_path = src
            return data
        except Exception as exc2:
            log.error("load_project JSON fallback failed: %s", exc2)
            return None

    def get_cache_key(
        self,
        video_path: str | Path,
        settings: "AnalysisSettings | None" = None,
    ) -> str:
        """
        SHA-1 ключ кеша.

        Включает use_body_weights из settings — инвалидация при смене
        параметра нормализации. Без settings — только путь/размер/mtime.
        """
        resolved = Path(video_path).expanduser().resolve()
        try:
            stat    = resolved.stat()
            payload = (
                f"{resolved.as_posix()}::{stat.st_size}::{stat.st_mtime_ns}"
            )
        except OSError:
            payload = resolved.as_posix()

        if settings is not None:
            payload += f"::bw={int(settings.use_body_weights)}"

        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def get_cache_path(
        self,
        video_path: str | Path,
        kind: str = "poses",
    ) -> Path:
        key = self.get_cache_key(video_path)
        match kind:
            case "poses":
                return self._pose_cache_dir / f"{key}.pkl.gz"
            case "matches":
                return self._MATCHES_CACHE_DIR / f"{key}.pkl.gz"
            case "motion_groups":
                return self._GROUPS_CACHE_DIR / f"{key}.pkl.gz"
            case "filters":
                return self._FILTER_CACHE_DIR / f"{key}.json"
            case "preview":
                return self._PREVIEW_CACHE_DIR / key
            case _:
                return self._pose_cache_dir / f"{key}_{kind}.pkl.gz"

    def save_poses_cache(
        self,
        video_path: str | Path,
        poses_meta: list[dict],
        poses_vecs: list[np.ndarray],
        settings: "AnalysisSettings | None" = None,
    ) -> bool:
        """
        Сохранить позы в кеш.
        settings передаётся для правильного ключа (с учётом use_body_weights).
        """
        key = self.get_cache_key(video_path, settings)
        dst = self._pose_cache_dir / f"{key}.pkl.gz"
        try:
            payload = {
                "poses_meta": poses_meta,
                "poses_vecs": poses_vecs,
                "video_path": str(video_path),
                "saved_at":   time.time(),
                "version":    PROJECT_FORMAT_VERSION,
            }
            ok = _atomic_write(dst, _pickle_gz(payload))
            if ok:
                log.debug("Кеш поз сохранён: %s", dst.name)
            return ok
        except Exception as exc:
            log.error("save_poses_cache: %s", exc)
            return False

    def load_poses_cache(
        self,
        video_path: str | Path,
        settings: "AnalysisSettings | None" = None,
    ) -> tuple[list[dict], list[np.ndarray]] | None:
        """
        Загрузить позы из кеша.
        settings нужен для проверки что кеш построен с теми же параметрами.
        """
        key = self.get_cache_key(video_path, settings)
        src = self._pose_cache_dir / f"{key}.pkl.gz"
        if not src.exists():
            return None
        try:
            data = _unpickle_gz(src.read_bytes())
            meta = data.get("poses_meta", [])
            vecs = data.get("poses_vecs", [])
            log.debug("Кеш поз загружен: %s (%d поз)", src.name, len(meta))
            return meta, vecs
        except Exception as exc:
            log.warning("load_poses_cache повреждён (%s): %s", src.name, exc)
            try:
                src.unlink()
            except OSError:
                pass
            return None

    def clear_cache(self, kind: str = "all") -> None:
        """Очистить кеш. kind: 'all'|'poses'|'matches'|'motion_groups'|'preview'|'filters'"""
        match kind:
            case "poses":
                targets = [(self._pose_cache_dir, "*.pkl.gz")]
            case "matches":
                targets = [(self._MATCHES_CACHE_DIR, "*.pkl.gz")]
            case "motion_groups":
                targets = [(self._GROUPS_CACHE_DIR, "*.pkl.gz")]
            case "preview":
                targets = [(self._PREVIEW_CACHE_DIR, "*")]
            case "filters":
                targets = [(self._FILTER_CACHE_DIR, "*.json")]
            case _:
                targets = [
                    (self._pose_cache_dir,    "*.pkl.gz"),
                    (self._MATCHES_CACHE_DIR, "*.pkl.gz"),
                    (self._GROUPS_CACHE_DIR,  "*.pkl.gz"),
                    (self._FILTER_CACHE_DIR,  "*.json"),
                    (self._PREVIEW_CACHE_DIR, "*"),
                ]

        removed = 0
        for directory, pattern in targets:
            for f in directory.glob(pattern):
                try:
                    if f.is_file():
                        f.unlink()
                        removed += 1
                    elif f.is_dir():
                        shutil.rmtree(f, ignore_errors=True)
                        removed += 1
                except OSError as exc:
                    log.warning("Не удалось удалить %s: %s", f, exc)

        log.info("Кеш очищен [%s]: %d объектов", kind, removed)
        print(f"[ProjectManager] Кеш очищен [{kind}]: {removed}")

    # ── § 2. ProjectState API ────────────────────────────────────────────

    def new_project(self, name: str = "Без названия") -> ProjectState:
        self._state = ProjectState(name=name)
        self.current_project_path = None
        return self._state

    @property
    def state(self) -> ProjectState:
        return self._state

    def save_state(
        self,
        path:  str | Path,
        state: ProjectState | None = None,
    ) -> bool:
        if state is None:
            state = self._state
        state.updated_at = time.time()
        dst = Path(path)
        _ensure(dst.parent)
        try:
            ok = _atomic_write(dst, _pickle_gz(state))
            if ok:
                self.current_project_path = dst
                log.info("ProjectState сохранён: %s", dst)
            return ok
        except Exception as exc:
            log.error("save_state(%s): %s", dst, exc)
            return False

    def load_state(self, path: str | Path) -> ProjectState | None:
        src = Path(path)
        if not src.exists():
            log.warning("load_state: не найден %s", src)
            return None
        try:
            data = _unpickle_gz(src.read_bytes())
        except Exception as exc:
            log.error("load_state десериализация failed %s: %s", src, exc)
            return None

        if isinstance(data, dict):
            log.info("load_state: старый dict-формат, конвертируем")
            data = self._migrate_from_dict(data)

        if not isinstance(data, ProjectState):
            log.error("load_state: неизвестный формат в %s", src)
            return None

        if data.format_version != PROJECT_FORMAT_VERSION:
            data = self._migrate_project(data)

        self._state = data
        self.current_project_path = src
        log.info("ProjectState загружен: %s (v%s)", src, data.format_version)
        return data

    def _migrate_from_dict(self, d: dict) -> ProjectState:
        state = ProjectState(name=d.get("name", "Проект"))
        for vp in d.get("video_paths", []):
            state.videos.append(VideoEntry(path=str(vp)))
        s = d.get("settings", {})
        if s:
            state.settings = AnalysisSettings(
                threshold  = s.get("threshold",  0.75),
                min_gap    = s.get("min_gap",    3.0),
                quality    = s.get("quality",    "Средне"),
                use_mirror = s.get("use_mirror", False),
            )
        for m in d.get("matches", []):
            state.matches.append(
                MatchRecord.from_dict(m) if isinstance(m, dict) else m
            )
        return state

    def _migrate_project(self, state: ProjectState) -> ProjectState:
        log.info("Миграция: %s → %s",
                 state.format_version, PROJECT_FORMAT_VERSION)
        state.format_version = PROJECT_FORMAT_VERSION
        return state

    # ── § 3. Видео ───────────────────────────────────────────────────────

    def add_video(
        self,
        path:        str | Path,
        duration:    float = 0.0,
        fps:         float = 0.0,
        frame_count: int   = 0,
        label:       str   = "",
    ) -> VideoEntry:
        entry = VideoEntry(
            path        = str(Path(path).resolve()),
            video_idx   = len(self._state.videos),
            duration    = duration,
            fps         = fps,
            frame_count = frame_count,
            cache_key   = self.get_cache_key(path),
            label       = label,
        )
        self._state.videos.append(entry)
        return entry

    def remove_video(self, video_idx: int) -> bool:
        videos = self._state.videos
        if video_idx < 0 or video_idx >= len(videos):
            return False
        videos.pop(video_idx)
        for i, v in enumerate(videos):
            v.video_idx = i
        return True

    def get_video_entries(self) -> list[VideoEntry]:
        return list(self._state.videos)

    def get_video_paths(self) -> list[str]:
        return [v.path for v in self._state.videos]

    def reorder_videos(self, new_order: list[int]) -> None:
        old      = self._state.videos
        expected = set(range(len(old)))
        given    = set(new_order)
        missing  = expected - given
        extra    = given - expected
        if missing or extra:
            raise ValueError(
                f"new_order некорректен: "
                f"отсутствуют={sorted(missing)}, лишние={sorted(extra)}"
            )
        self._state.videos = [old[i] for i in new_order]
        for i, v in enumerate(self._state.videos):
            v.video_idx = i

    # ── § 4. Совпадения и фильтрация ─────────────────────────────────────

    def set_matches(self, matches: list[dict | MatchRecord]) -> None:
        records: list[MatchRecord] = []
        for m in matches:
            if isinstance(m, MatchRecord):
                records.append(m)
            elif isinstance(m, dict):
                records.append(MatchRecord.from_dict(m))
            else:
                log.warning("set_matches: пропущен тип %s", type(m))
        self._state.matches = records

    def get_matches(self) -> list[MatchRecord]:
        return list(self._state.matches)

    def get_visible_matches(self) -> list[MatchRecord]:
        """
        Один проход по совпадениям с применением всех фильтров.
        Быстрее предыдущей версии с тремя list comprehension.
        """
        fs = self._state.filter_state

        def _passes(m: MatchRecord) -> bool:
            if not fs.show_hidden   and m.hidden:
                return False
            if not fs.show_excluded and m.excluded:
                return False
            if fs.show_confirmed_only and not m.confirmed:
                return False
            if fs.direction_filter != "all" and m.direction != fs.direction_filter:
                return False
            if m.sim < fs.min_sim:
                return False
            return True

        result = [m for m in self._state.matches if _passes(m)]

        match fs.sort_by:
            case "t1":
                result.sort(key=lambda m: m.t1, reverse=fs.sort_desc)
            case "t2":
                result.sort(key=lambda m: m.t2, reverse=fs.sort_desc)
            case "direction":
                result.sort(key=lambda m: m.direction, reverse=fs.sort_desc)
            case _:
                result.sort(key=lambda m: m.sim, reverse=fs.sort_desc)

        return result

    def _get_match(self, idx: int) -> MatchRecord | None:
        matches = self._state.matches
        if 0 <= idx < len(matches):
            return matches[idx]
        log.warning("_get_match: индекс %d вне [0, %d)", idx, len(matches))
        return None

    def hide_match(self, idx: int) -> None:
        m = self._get_match(idx)
        if m is not None:
            m.hidden = True

    def show_match(self, idx: int) -> None:
        m = self._get_match(idx)
        if m is not None:
            m.hidden = False

    def exclude_match(self, idx: int) -> None:
        m = self._get_match(idx)
        if m is not None:
            m.excluded  = True
            m.confirmed = False

    def confirm_match(self, idx: int) -> None:
        m = self._get_match(idx)
        if m is not None:
            m.confirmed = True
            m.excluded  = False
            m.hidden    = False

    def get_excluded_indices(self) -> list[int]:
        return [i for i, m in enumerate(self._state.matches) if m.excluded]

    def get_confirmed_indices(self) -> list[int]:
        return [i for i, m in enumerate(self._state.matches) if m.confirmed]

    def set_motion_groups(self, groups: dict[str, list[int]]) -> None:
        self._state.motion_groups = dict(groups)

    def get_motion_groups(self) -> dict[str, list[int]]:
        return dict(self._state.motion_groups)

    # ── § 5. Кеш совпадений / групп / превью ─────────────────────────────

    def save_matches_cache(
        self,
        cache_key: str,
        matches:   list[dict | MatchRecord],
    ) -> bool:
        dst = self._MATCHES_CACHE_DIR / f"{cache_key}.pkl.gz"
        try:
            payload = [
                m.to_dict() if isinstance(m, MatchRecord) else m
                for m in matches
            ]
            return _atomic_write(
                dst,
                _pickle_gz({"matches": payload, "saved_at": time.time()}),
            )
        except Exception as exc:
            log.error("save_matches_cache: %s", exc)
            return False

    def load_matches_cache(self, cache_key: str) -> list[dict] | None:
        src = self._MATCHES_CACHE_DIR / f"{cache_key}.pkl.gz"
        if not src.exists():
            return None
        try:
            data = _unpickle_gz(src.read_bytes())
            return data.get("matches", [])
        except Exception as exc:
            log.warning("load_matches_cache повреждён (%s): %s", src.name, exc)
            try:
                src.unlink()
            except OSError:
                pass
            return None

    def save_motion_groups_cache(
        self,
        cache_key: str,
        groups:    dict[str, list],
    ) -> bool:
        dst = self._GROUPS_CACHE_DIR / f"{cache_key}.pkl.gz"
        try:
            return _atomic_write(
                dst,
                _pickle_gz({"groups": groups, "saved_at": time.time()}),
            )
        except Exception as exc:
            log.error("save_motion_groups_cache: %s", exc)
            return False

    def load_motion_groups_cache(self, cache_key: str) -> dict | None:
        src = self._GROUPS_CACHE_DIR / f"{cache_key}.pkl.gz"
        if not src.exists():
            return None
        try:
            return _unpickle_gz(src.read_bytes()).get("groups")
        except Exception as exc:
            log.warning("load_motion_groups_cache (%s): %s", src.name, exc)
            try:
                src.unlink()
            except OSError:
                pass
            return None

    def get_preview_path(
        self,
        video_path: str | Path,
        frame_idx:  int,
    ) -> Path:
        key     = self.get_cache_key(video_path)
        sub_dir = _ensure(self._PREVIEW_CACHE_DIR / key)
        return sub_dir / f"{frame_idx:08d}.jpg"

    def preview_exists(self, video_path: str | Path, frame_idx: int) -> bool:
        return self.get_preview_path(video_path, frame_idx).exists()

    # ── § 6. Фильтры ─────────────────────────────────────────────────────

    def save_filter_state(self, path: str | Path | None = None) -> bool:
        if path is None:
            path = (
                self.current_project_path.with_suffix(".filters.json")
                if self.current_project_path
                else self._FILTER_CACHE_DIR / "default.json"
            )
        dst = Path(path)
        _ensure(dst.parent)
        try:
            raw = json.dumps(
                asdict(self._state.filter_state),
                ensure_ascii=False, indent=2,
            ).encode("utf-8")
            return _atomic_write(dst, raw)
        except Exception as exc:
            log.error("save_filter_state: %s", exc)
            return False

    def load_filter_state(
        self, path: str | Path | None = None
    ) -> FilterState | None:
        if path is None:
            path = (
                self.current_project_path.with_suffix(".filters.json")
                if self.current_project_path
                else self._FILTER_CACHE_DIR / "default.json"
            )
        src = Path(path)
        if not src.exists():
            return None
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
            fs   = FilterState(**{
                k: v for k, v in data.items()
                if k in FilterState.__dataclass_fields__
            })
            self._state.filter_state = fs
            return fs
        except Exception as exc:
            log.error("load_filter_state: %s", exc)
            return None

    # ── § 7. Настройки ───────────────────────────────────────────────────

    def get_settings(self) -> AnalysisSettings:
        return self._state.settings

    def update_settings(self, **kwargs: Any) -> None:
        s     = self._state.settings
        known = AnalysisSettings.__dataclass_fields__
        for k, v in kwargs.items():
            if k in known:
                setattr(s, k, v)
            elif k == "scene_interval":
                s.min_gap = float(v)
            else:
                log.warning("update_settings: неизвестный ключ '%s'", k)

    # ── § 8. Статистика ──────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        matches = self._state.matches
        visible = self.get_visible_matches()
        return {
            "total_matches":     len(matches),
            "visible_matches":   len(visible),
            "hidden_matches":    sum(1 for m in matches if m.hidden),
            "excluded_matches":  sum(1 for m in matches if m.excluded),
            "confirmed_matches": sum(1 for m in matches if m.confirmed),
            "total_videos":      len(self._state.videos),
            "total_duration":    sum(v.duration for v in self._state.videos),
            "motion_groups":     len(self._state.motion_groups),
            "avg_sim":           (
                float(np.mean([m.sim for m in matches]))
                if matches else 0.0
            ),
        }

    def cache_size_bytes(self, kind: str = "all") -> int:
        """
        Суммарный размер кеша в байтах.
        Предупреждение: может быть медленным при большом кеше.
        Рекомендуется вызывать из фонового потока.
        """
        match kind:
            case "poses":
                dirs = [self._pose_cache_dir]
            case "matches":
                dirs = [self._MATCHES_CACHE_DIR]
            case "preview":
                dirs = [self._PREVIEW_CACHE_DIR]
            case "motion_groups":
                dirs = [self._GROUPS_CACHE_DIR]
            case _:
                dirs = [
                    self._pose_cache_dir,
                    self._MATCHES_CACHE_DIR,
                    self._PREVIEW_CACHE_DIR,
                    self._GROUPS_CACHE_DIR,
                ]

        total = 0
        for d in dirs:
            for f in d.rglob("*"):
                if f.is_file():
                    try:
                        total += f.stat().st_size
                    except OSError:
                        pass
        return total

    def __repr__(self) -> str:
        return (
            f"<ProjectManager name={self._state.name!r} "
            f"videos={len(self._state.videos)} "
            f"matches={len(self._state.matches)} "
            f"project={self.current_project_path}>"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile

    print("=== ProjectManager smoke-test ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        pm = ProjectManager(cache_dir=tmp)

        # 1. Новый проект
        state = pm.new_project("Test")
        assert state.name == "Test"
        print(f"[1] Новый проект: {state.name!r}")

        # 2. Видео
        pm.add_video("/fake/v1.mp4", duration=120.0, fps=25.0, frame_count=3000)
        pm.add_video("/fake/v2.mp4", duration=90.0,  fps=30.0, frame_count=2700)
        assert len(pm.get_video_entries()) == 2
        print(f"[2] Видео: {[v.label for v in pm.get_video_entries()]}")

        # 3. scene_interval алиас
        pm.update_settings(scene_interval=5.0)
        assert pm.get_settings().min_gap == 5.0
        assert pm.get_settings().scene_interval == 5.0
        print(f"[3] scene_interval alias: min_gap={pm.get_settings().min_gap}")

        # 4. Совпадения
        fake = [
            {"sim": 0.91, "t1": 10.0, "t2": 45.0, "v1_idx": 0,
             "v2_idx": 1, "direction": "forward", "f1": 250, "f2": 1350},
            {"sim": 0.85, "t1": 22.0, "t2": 67.0, "v1_idx": 0,
             "v2_idx": 1, "direction": "left",    "f1": 550, "f2": 2010},
        ]
        pm.set_matches(fake)
        assert len(pm.get_matches()) == 2
        print(f"[4] Совпадений: {len(pm.get_matches())}")

        # 5. Один проход фильтрации
        pm.hide_match(1)
        visible = pm.get_visible_matches()
        assert len(visible) == 1
        print(f"[5] Видимых после hide: {len(visible)}")

        # 6. Сохранение / загрузка state
        proj_path = Path(tmp) / "test.pfp"
        assert pm.save_state(proj_path)
        pm2     = ProjectManager(cache_dir=tmp)
        loaded  = pm2.load_state(proj_path)
        assert loaded is not None
        assert loaded.name == "Test"
        assert len(loaded.matches) == 2
        print(f"[6] load_state: name={loaded.name!r} matches={len(loaded.matches)}")

        # 7. Кеш поз с инвалидацией по use_body_weights
        settings_bw   = AnalysisSettings(use_body_weights=True)
        settings_nobw = AnalysisSettings(use_body_weights=False)
        key_bw   = pm.get_cache_key("/fake/v1.mp4", settings_bw)
        key_nobw = pm.get_cache_key("/fake/v1.mp4", settings_nobw)
        assert key_bw != key_nobw, "Ключи должны различаться при разных use_body_weights!"
        print(f"[7] Ключи кеша различны: {key_bw[:8]} ≠ {key_nobw[:8]}")

        fake_meta = [{"t": 1.0, "f": 30, "video_idx": 0, "dir": "forward"}]
        fake_vecs = [np.random.rand(34).astype(np.float32)]
        ok   = pm.save_poses_cache("/fake/v1.mp4", fake_meta, fake_vecs, settings_bw)
        res  = pm.load_poses_cache("/fake/v1.mp4", settings_bw)
        none = pm.load_poses_cache("/fake/v1.mp4", settings_nobw)
        assert res  is not None
        assert none is None
        print(f"[8] Кеш поз: bw={res is not None} nobw={none is None}")

        # 8. Старый API
        old_path = str(Path(tmp) / "old.pfp")
        ok_old   = pm.save_project(old_path, {"video_paths": [], "matches": []})
        loaded_old = pm.load_project(old_path)
        assert loaded_old is not None
        print(f"[9] Старый API: saved={ok_old}")

        # 9. reorder_videos
        try:
            pm.reorder_videos([99, 0])
        except ValueError as e:
            print(f"[10] reorder_videos ошибка: {str(e)[:60]}")

        print(f"\n[OK] repr: {pm}")

    print("\n=== Все тесты пройдены ===")


__all__ = [
    "VideoEntry",
    "AnalysisSettings",
    "MatchRecord",
    "FilterState",
    "ProjectState",
    "ProjectManager",
    "PROJECT_FORMAT_VERSION",
    "PROJECT_SUFFIX",
]