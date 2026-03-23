from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import pickle
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypedDict, cast

import cv2
import numpy as np
import numpy.typing as npt

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    h5py = None  # type: ignore[assignment]

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

from core.engine import MultiPoseData, PoseData, YoloEngine
from core.matcher import MatchResult

LOGGER = logging.getLogger(__name__)
ArrayF32 = npt.NDArray[np.float32]
ArrayU8 = npt.NDArray[np.uint8]
PROJECT_VERSION = "2.0.0"


class ProjectData(TypedDict):
    """Serializable project payload used by the UI and backend."""

    version: str
    videos: list[str]
    reference_path: str
    poses: list[MultiPoseData]
    matches: list[MatchResult]
    settings: dict[str, Any]
    ui_prefs: dict[str, Any]
    created_at: str
    modified_at: str


class PreviewCache:
    """Disk-backed preview cache with LRU-like eviction and aging cleanup."""

    def __init__(self, cache_dir: Path, max_size_mb: int = 512) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max(64, int(max_size_mb)) * 1024 * 1024

    def get(self, video_path: str, frame_idx: int) -> ArrayU8 | ArrayF32 | None:
        """Load a cached preview frame if present."""
        path = self._frame_path(video_path, frame_idx)
        if not path.exists():
            return None
        try:
            frame = np.load(path, allow_pickle=False)
            path.touch(exist_ok=True)
            return cast(ArrayU8 | ArrayF32, frame)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Preview cache read failed for %s: %s", path, exc)
            try:
                path.unlink(missing_ok=True)
            except Exception:  # pragma: no cover - cleanup best effort
                pass
            return None

    def put(self, video_path: str, frame_idx: int, frame: np.ndarray) -> None:
        """Store a preview frame in the cache and enforce the size budget."""
        path = self._frame_path(video_path, frame_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        try:
            np.save(tmp, np.asarray(frame))
            tmp.replace(path)
            path.touch(exist_ok=True)
            self._enforce_size_limit()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Preview cache write failed for %s: %s", path, exc)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:  # pragma: no cover - cleanup best effort
                pass

    def clear_old(self, max_age_days: int = 7) -> None:
        """Remove files older than the configured number of days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, max_age_days))
        for file_path in self.cache_dir.rglob("*.npy"):
            try:
                modified = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                if modified < cutoff:
                    file_path.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Preview aging cleanup failed for %s: %s", file_path, exc)

    def clear(self) -> None:
        """Remove all cached preview files."""
        if not self.cache_dir.exists():
            return
        for file_path in self.cache_dir.rglob("*.npy"):
            try:
                file_path.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Preview clear failed for %s: %s", file_path, exc)

    def _frame_path(self, video_path: str, frame_idx: int) -> Path:
        digest = hashlib.sha1(Path(video_path).as_posix().encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}_{int(frame_idx):08d}.npy"

    def _enforce_size_limit(self) -> None:
        files = []
        total_size = 0
        for file_path in self.cache_dir.rglob("*.npy"):
            try:
                stat = file_path.stat()
            except FileNotFoundError:
                continue
            files.append((stat.st_mtime, stat.st_size, file_path))
            total_size += stat.st_size

        if total_size <= self.max_size_bytes:
            return

        files.sort(key=lambda item: item[0])
        for _, size, file_path in files:
            try:
                file_path.unlink(missing_ok=True)
                total_size -= size
                if total_size <= self.max_size_bytes:
                    break
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Preview cache eviction failed for %s: %s", file_path, exc)


class ProjectManager:

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            from utils.constants import CACHE_DIR
            self.cache_dir = CACHE_DIR


        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._preview_cache = PreviewCache(self.cache_dir / "previews")
        
        self.current_project_path: Path | None = None

    def new_project(self) -> ProjectData:
        now = self._now_iso()
        return ProjectData(
            version=PROJECT_VERSION,
            videos=[],
            reference_path="",
            poses=[],
            matches=[],
            settings={},
            ui_prefs={},
            created_at=now,
            modified_at=now,
        )

    def save_project(self, path: str | Path, data: ProjectData) -> bool:
        """Persist a project in JSON, pickle or HDF5 format."""
        target = Path(path)
        payload = self._normalize_project_payload(data)
        payload["modified_at"] = self._now_iso()
        payload.setdefault("created_at", payload["modified_at"])

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            suffix = target.suffix.lower()
            if suffix in {".json", ".pfproj", ".pfa"}:
                target.write_text(json.dumps(self._to_serializable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
            elif suffix in {".pkl", ".pickle"}:
                with gzip.open(target, "wb") as fh:
                    pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            elif suffix in {".h5", ".hdf5"}:
                if h5py is None:
                    raise RuntimeError("h5py is not available")
                with h5py.File(target, "w") as h5_file:  # type: ignore[union-attr]
                    json_payload = json.dumps(self._to_serializable(payload), ensure_ascii=False)
                    h5_file.create_dataset("project_json", data=np.bytes_(json_payload.encode("utf-8")))
            else:
                target.write_text(json.dumps(self._to_serializable(payload), ensure_ascii=False, indent=2), encoding="utf-8")

            self.current_project_path = target
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Project save failed for %s: %s", target, exc)
            return False

    def load_project(self, path: str | Path) -> ProjectData | None:
        """Load a project from JSON, pickle or HDF5."""
        target = Path(path)
        try:
            suffix = target.suffix.lower()
            if suffix in {".json", ".pfproj", ".pfa"}:
                raw = json.loads(target.read_text(encoding="utf-8"))
                data = self._from_serializable(raw)
            elif suffix in {".pkl", ".pickle"}:
                with gzip.open(target, "rb") as fh:
                    data = pickle.load(fh)
            elif suffix in {".h5", ".hdf5"}:
                if h5py is None:
                    raise RuntimeError("h5py is not available")
                with h5py.File(target, "r") as h5_file:  # type: ignore[union-attr]
                    blob = bytes(h5_file["project_json"][()]).decode("utf-8")
                    data = self._from_serializable(json.loads(blob))
            else:
                raw = json.loads(target.read_text(encoding="utf-8"))
                data = self._from_serializable(raw)

            project = self._normalize_project_payload(cast(dict[str, Any], data))
            self.current_project_path = target
            return project
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Project load failed for %s: %s", target, exc)
            return None

    def save_preset(self, path: str | Path, settings: dict[str, Any]) -> bool:
        """Save analysis settings as JSON or pickle."""
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.suffix.lower() in {".pkl", ".pickle"}:
                with gzip.open(target, "wb") as fh:
                    pickle.dump(settings, fh, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                target.write_text(json.dumps(self._to_serializable(settings), ensure_ascii=False, indent=2), encoding="utf-8")
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Preset save failed for %s: %s", target, exc)
            return False

    def load_preset(self, path: str | Path) -> dict[str, Any] | None:
        """Load analysis settings from JSON or pickle."""
        target = Path(path)
        try:
            if target.suffix.lower() in {".pkl", ".pickle"}:
                with gzip.open(target, "rb") as fh:
                    data = pickle.load(fh)
            else:
                data = self._from_serializable(json.loads(target.read_text(encoding="utf-8")))
            return cast(dict[str, Any], data)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Preset load failed for %s: %s", target, exc)
            return None

    def export_matches(self, matches: list[MatchResult], format: str, path: str | Path) -> bool:
        """Export matches to JSON, CSV or pickle."""
        target = Path(path)
        fmt = format.lower().strip()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if fmt == "json":
                target.write_text(
                    json.dumps([self._to_serializable(match) for match in matches], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            elif fmt == "csv":
                rows = [self._normalize_match(match) for match in matches]
                fieldnames = [
                    "t1",
                    "t2",
                    "duration",
                    "similarity",
                    "direction",
                    "context",
                    "v1_idx",
                    "v2_idx",
                    "frame_i",
                    "frame_j",
                    "energy",
                    "cluster_id",
                ]
                with target.open("w", encoding="utf-8", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            elif fmt in {"pickle", "pkl"}:
                with gzip.open(target, "wb") as fh:
                    pickle.dump(matches, fh, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Match export failed for %s: %s", target, exc)
            return False

    def import_matches(self, path: str | Path) -> list[MatchResult]:
        """Import matches from JSON, CSV or pickle."""
        target = Path(path)
        try:
            suffix = target.suffix.lower()
            if suffix == ".json":
                raw = self._from_serializable(json.loads(target.read_text(encoding="utf-8")))
                return [self._normalize_match(item) for item in cast(list[dict[str, Any]], raw)]
            if suffix == ".csv":
                with target.open("r", encoding="utf-8", newline="") as fh:
                    reader = csv.DictReader(fh)
                    return [self._normalize_match(dict(row)) for row in reader]
            if suffix in {".pkl", ".pickle"}:
                with gzip.open(target, "rb") as fh:
                    raw = pickle.load(fh)
                return [self._normalize_match(item) for item in cast(list[dict[str, Any]], raw)]
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Match import failed for %s: %s", target, exc)
        return []

    def get_preview_cache(self) -> PreviewCache:
        """Return the shared preview cache instance."""
        return self._preview_cache

    def clear_cache(self) -> None:
        """Clear all managed cache directories."""
        for child in self.cache_dir.iterdir() if self.cache_dir.exists() else []:
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Cache clear failed for %s: %s", child, exc)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._preview_cache = PreviewCache(self.cache_dir / "previews")

    def _normalize_project_payload(self, data: dict[str, Any]) -> ProjectData:
        now = self._now_iso()
        poses_raw = cast(list[dict[str, Any]], data.get("poses", []))
        matches_raw = cast(list[dict[str, Any]], data.get("matches", []))
        return ProjectData(
            version=str(data.get("version", PROJECT_VERSION)),
            videos=[str(item) for item in data.get("videos", [])],
            reference_path=str(data.get("reference_path", "")),
            poses=[self._normalize_multi_pose(item) for item in poses_raw],
            matches=[self._normalize_match(item) for item in matches_raw],
            settings=cast(dict[str, Any], data.get("settings", {})),
            ui_prefs=cast(dict[str, Any], data.get("ui_prefs", {})),
            created_at=str(data.get("created_at", now)),
            modified_at=str(data.get("modified_at", now)),
        )

    def _normalize_multi_pose(self, data: dict[str, Any]) -> MultiPoseData:
        track_ids_raw = data.get("track_ids", [])
        track_ids = []
        for value in track_ids_raw if isinstance(track_ids_raw, list) else []:
            try:
                track_ids.append(int(value))
            except Exception:
                track_ids.append(-1)

        people_raw = data.get("people", [])
        people = [self._normalize_pose(cast(dict[str, Any], person)) for person in people_raw if isinstance(person, dict)]
        return MultiPoseData(
            frame_idx=int(data.get("frame_idx", 0)),
            people=people,
            track_ids=track_ids,
        )

    def _normalize_pose(self, data: dict[str, Any]) -> PoseData:
        keypoints = self._ensure_array(data.get("keypoints"), (17, 3), dtype=np.float32)
        normalized = self._ensure_array(data.get("normalized"), (17, 2), dtype=np.float32)
        velocity = self._ensure_array(data.get("velocity"), (17, 2), dtype=np.float32)
        pose: dict[str, Any] = {
            "frame_idx": int(data.get("frame_idx", 0)),
            "keypoints": keypoints,
            "normalized": normalized,
            "confidence": float(data.get("confidence", float(keypoints[:, 2].mean()) if keypoints.size else 0.0)),
            "direction": str(data.get("direction", "unknown")),
            "velocity": velocity,
        }
        if "track_id" in data:
            try:
                pose["track_id"] = int(data.get("track_id", -1))
            except Exception:
                pose["track_id"] = -1
        return cast(PoseData, pose)

    def _normalize_match(self, data: dict[str, Any]) -> MatchResult:
        return MatchResult(
            t1=float(data.get("t1", 0.0)),
            t2=float(data.get("t2", 0.0)),
            duration=float(data.get("duration", 0.0)),
            similarity=float(data.get("similarity", 0.0)),
            direction=int(data.get("direction", -1)),
            context=int(data.get("context", 0)),
            v1_idx=int(data.get("v1_idx", 0)),
            v2_idx=int(data.get("v2_idx", 0)),
            frame_i=int(data.get("frame_i", 0)),
            frame_j=int(data.get("frame_j", 0)),
            energy=float(data.get("energy", 0.0)),
            cluster_id=int(data.get("cluster_id", -1)),
        )

    def _to_serializable(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return {
                "__ndarray__": True,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": value.tolist(),
            }
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, dict):
            return {str(key): self._to_serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_serializable(item) for item in value]
        return value

    def _from_serializable(self, value: Any) -> Any:
        if isinstance(value, dict):
            if value.get("__ndarray__"):
                array = np.asarray(value.get("data", []), dtype=np.dtype(str(value.get("dtype", "float32"))))
                shape = tuple(int(item) for item in value.get("shape", array.shape))
                return array.reshape(shape)
            return {key: self._from_serializable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._from_serializable(item) for item in value]
        return value

    @staticmethod
    def _ensure_array(value: Any, shape: tuple[int, ...], dtype: np.dtype[np.float32]) -> ArrayF32:
        array = np.zeros(shape, dtype=dtype)
        if value is None:
            return array
        try:
            candidate = np.asarray(value, dtype=dtype)
            if candidate.shape == shape:
                return candidate.astype(dtype, copy=False)
        except Exception:  # pragma: no cover - graceful fallback
            pass
        return array

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ReferencePerson:
    """Reference person helper compatible with multi-person engine outputs."""

    def __init__(self) -> None:
        self.source_path: str = ""
        self.name: str = ""
        self.keypoints: ArrayF32 | None = None
        self.normalized: ArrayF32 | None = None
        self.embedding: ArrayF32 | None = None
        self.confidence: float = 0.0
        self.frame_idx: int = 0

    def load_from_image(self, path: str, engine: YoloEngine) -> bool:
        """Load the best visible person from an image using the provided engine."""
        try:
            image = cv2.imread(path)
            if image is None:
                return False
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            people = self._infer_people(frame_rgb, engine)
            return self._assign_best_person(people, path, 0)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Reference image load failed for %s: %s", path, exc)
            return False

    def load_from_video(self, path: str, engine: YoloEngine, frame_idx: int = 0) -> bool:
        """Load the best visible person from a specific video frame."""
        capture = cv2.VideoCapture(path)
        try:
            if not capture.isOpened():
                return False
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
            ok, frame = capture.read()
            if not ok or frame is None:
                return False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            people = self._infer_people(frame_rgb, engine)
            return self._assign_best_person(people, path, frame_idx)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Reference video load failed for %s: %s", path, exc)
            return False
        finally:
            capture.release()

    def extract_embedding(self) -> ArrayF32 | None:
        """Build a confidence-aware normalized embedding for identity matching."""
        if self.keypoints is None:
            return None
        normalized = self.normalized if self.normalized is not None else self._normalize_keypoints(self.keypoints)
        visible = (self.keypoints[:, 2:3] >= 0.3).astype(np.float32)
        embedding = np.concatenate((normalized * visible, visible), axis=1).reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-8:
            return None
        self.embedding = (embedding / norm).astype(np.float32)
        return self.embedding

    def compare_with_pose(self, pose: MultiPoseData) -> float:
        """Return the best cosine similarity against all people in a frame."""
        if self.embedding is None:
            self.extract_embedding()
        if self.embedding is None:
            return 0.0

        best_score = 0.0
        for person in pose.get("people", []):
            candidate = self._embedding_from_pose(person)
            if candidate is None:
                continue
            score = float(np.clip(np.dot(self.embedding, candidate), 0.0, 1.0))
            best_score = max(best_score, score)
        return best_score

    def _assign_best_person(self, people: list[PoseData], source_path: str, frame_idx: int) -> bool:
        if not people:
            return False
        best = max(people, key=lambda item: float(item.get("confidence", 0.0)))
        self.source_path = source_path
        self.name = Path(source_path).stem
        self.frame_idx = int(frame_idx)
        self.keypoints = np.asarray(best.get("keypoints"), dtype=np.float32)
        norm = best.get("normalized")
        self.normalized = np.asarray(norm, dtype=np.float32) if norm is not None else self._normalize_keypoints(self.keypoints)
        self.confidence = float(best.get("confidence", float(self.keypoints[:, 2].mean())))
        self.extract_embedding()
        return True

    def _infer_people(self, frame_rgb: np.ndarray, engine: YoloEngine) -> list[PoseData]:
        try:
            if hasattr(engine, "_run_inference"):
                outputs = engine._run_inference([frame_rgb])  # type: ignore[attr-defined]
            elif hasattr(engine, "_pytorch_model") and getattr(engine, "_pytorch_model") is not None:
                model = getattr(engine, "_pytorch_model")
                outputs = model.predict(source=[frame_rgb], verbose=False, device=getattr(engine, "device", None))
            else:
                return []

            if not outputs:
                return []

            keypoints = self._parse_result_keypoints(outputs[0])
            people: list[PoseData] = []
            for person_idx, keypoints_xyc in enumerate(keypoints):
                normalized = self._engine_normalize(engine, keypoints_xyc)
                person: dict[str, Any] = {
                    "frame_idx": 0,
                    "keypoints": keypoints_xyc.astype(np.float32),
                    "normalized": normalized.astype(np.float32),
                    "confidence": float(keypoints_xyc[:, 2].mean()),
                    "direction": "unknown",
                    "velocity": np.zeros((17, 2), dtype=np.float32),
                    "track_id": person_idx,
                }
                people.append(cast(PoseData, person))
            return people
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Reference inference failed: %s", exc)
            return []

    def _parse_result_keypoints(self, result: object) -> ArrayF32:
        if not hasattr(result, "keypoints"):
            return np.zeros((0, 17, 3), dtype=np.float32)
        keypoints_obj = getattr(result, "keypoints")
        if not hasattr(keypoints_obj, "data"):
            return np.zeros((0, 17, 3), dtype=np.float32)
        data = getattr(keypoints_obj, "data")

        if torch is not None and isinstance(data, torch.Tensor):
            array = data.detach().float().cpu().numpy()
        else:
            array = np.asarray(data, dtype=np.float32)

        if array.ndim != 3 or array.shape[1] < 17 or array.shape[2] < 3:
            return np.zeros((0, 17, 3), dtype=np.float32)
        return array[:, :17, :3].astype(np.float32, copy=False)

    def _engine_normalize(self, engine: YoloEngine, keypoints: ArrayF32) -> ArrayF32:
        if hasattr(engine, "normalize_pose"):
            try:
                normalized = engine.normalize_pose(keypoints)  # type: ignore[attr-defined]
                candidate = np.asarray(normalized, dtype=np.float32)
                if candidate.shape == (17, 2):
                    return candidate
            except Exception:  # pragma: no cover - graceful fallback
                pass
        return self._normalize_keypoints(keypoints)

    def _embedding_from_pose(self, pose: PoseData) -> ArrayF32 | None:
        keypoints = np.asarray(pose.get("keypoints"), dtype=np.float32)
        if keypoints.shape != (17, 3):
            return None
        normalized_raw = pose.get("normalized")
        normalized = np.asarray(normalized_raw, dtype=np.float32) if normalized_raw is not None else self._normalize_keypoints(keypoints)
        if normalized.shape != (17, 2):
            return None
        visible = (keypoints[:, 2:3] >= 0.3).astype(np.float32)
        embedding = np.concatenate((normalized * visible, visible), axis=1).reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-8:
            return None
        return (embedding / norm).astype(np.float32)

    def _normalize_keypoints(self, keypoints: ArrayF32) -> ArrayF32:
        xy = np.asarray(keypoints[:, :2], dtype=np.float32)
        conf = np.asarray(keypoints[:, 2], dtype=np.float32)

        shoulders_visible = conf[[5, 6]] >= 0.3
        hips_visible = conf[[11, 12]] >= 0.3

        shoulder_center = xy[[5, 6]].mean(axis=0)
        hip_center = xy[[11, 12]].mean(axis=0)

        if hips_visible.any() and shoulders_visible.any():
            center = (shoulder_center + hip_center) * 0.5
        elif hips_visible.any():
            center = hip_center
        elif shoulders_visible.any():
            center = shoulder_center
        else:
            center = xy.mean(axis=0)

        shoulder_width = float(np.linalg.norm(xy[5] - xy[6]))
        torso_left = float(np.linalg.norm(xy[5] - xy[11]))
        torso_right = float(np.linalg.norm(xy[6] - xy[12]))
        scale = max(1e-3, np.mean([max(shoulder_width, 1e-3), max(torso_left, 1e-3), max(torso_right, 1e-3)]))
        return ((xy - center) / scale).astype(np.float32)


__all__ = [
    "PROJECT_VERSION",
    "PreviewCache",
    "ProjectData",
    "ProjectManager",
    "ReferencePerson",
]
