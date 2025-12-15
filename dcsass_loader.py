from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import imageio
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.auto import tqdm

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

from config import config

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

VIDEO_EXTS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")
SPLIT_NAMES = ("train", "val", "test")
_CACHE_VERSION = "frames_v1"

METADATA_PATH_CANDIDATES = ("path", "video", "filepath", "file", "relative_path")
METADATA_LABEL_CANDIDATES = ("label", "class", "category", "target", "y")
METADATA_SPLIT_CANDIDATES = ("split", "partition", "set")


def _ensure_ffmpeg_binary() -> None:
    """Set IMAGEIO_FFMPEG_EXE to a usable binary."""
    if "IMAGEIO_FFMPEG_EXE" in os.environ:
        return

    candidates = [
        os.environ.get("FFMPEG_PATH"),
        shutil.which("ffmpeg"),
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            os.environ["IMAGEIO_FFMPEG_EXE"] = candidate
            return

    if imageio_ffmpeg is None:
        LOGGER.warning("imageio-ffmpeg not available; ffmpeg binary could not be located.")
        return

    try:
        downloaded = imageio_ffmpeg.get_ffmpeg_exe()
        os.environ["IMAGEIO_FFMPEG_EXE"] = downloaded
    except Exception as exc:
        LOGGER.warning("Unable to download bundled ffmpeg: %s", exc)


_ensure_ffmpeg_binary()
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")


def decode_video_opencv(
    path: str,
    target_size: Sequence[int] = (224, 224),
    max_frames: int | None = None,
) -> List[np.ndarray]:
    """Decode frames with imageio and resize via OpenCV."""
    try:
        reader = imageio.get_reader(path, "ffmpeg")
    except Exception as exc:
        raise RuntimeError(f"Failed to open video {path}") from exc

    target_h, target_w = int(target_size[0]), int(target_size[1])
    frames: List[np.ndarray] = []
    try:
        for idx, frame in enumerate(reader):
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            frames.append(resized)
            if max_frames is not None and idx + 1 >= max_frames:
                break
    except Exception as exc:
        raise RuntimeError(f"Error decoding video {path}: {exc}") from exc
    finally:
        reader.close()

    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames


def make_segments(frames: Sequence[np.ndarray] | np.ndarray, T: int, stride: int) -> np.ndarray:
    """Create sliding-window segments of length T."""
    if isinstance(frames, np.ndarray):
        arr = frames
    else:
        if not frames:
            return np.empty((0, T, 224, 224, 3), dtype=np.uint8)
        arr = np.stack(frames)
    num_frames = arr.shape[0]
    if arr.size == 0 or num_frames == 0:
        return np.empty((0, T) + arr.shape[1:], dtype=arr.dtype)
    windows = []
    for start in range(0, num_frames - T + 1, stride):
        windows.append(arr[start : start + T])
    if not windows:
        return np.empty((0, T) + arr.shape[1:], dtype=arr.dtype)
    return np.stack(windows)


def list_videos(root: Path, exts: Sequence[str] = VIDEO_EXTS) -> List[Path]:
    """List all video files in a directory."""
    files = [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    files.sort()
    return files


def resolve_dcsass_root(explicit_root: Optional[str | Path]) -> Path:
    """Resolve the DCSASS dataset root directory."""
    if explicit_root:
        return Path(explicit_root).resolve()
    return config.DATA_DIR.resolve()


def _read_metadata_csv(metadata_path: Path):
    """Read metadata CSV file."""
    import pandas as pd
    df = pd.read_csv(metadata_path)
    if df.empty:
        raise ValueError(f"Metadata file {metadata_path} is empty.")
    return df


def _detect_column(df, candidates: Sequence[str]) -> Optional[str]:
    """Detect column name from candidates."""
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        match = lower_map.get(candidate.lower())
        if match is not None:
            return match
    return None


def _normalize_video_path(root: Path, raw_path: str) -> Path:
    """Normalize video path to absolute path."""
    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return path
    candidate = (root / path).resolve()
    if candidate.exists():
        return candidate
    dataset_dir = root / "DCSASS Dataset"
    if dataset_dir.exists():
        candidate = (dataset_dir / path).resolve()
        if candidate.exists():
            return candidate
    matches = list(root.rglob(path.name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Unable to resolve video path '{raw_path}' under {root}")


def _assign_splits(entries: List[Dict[str, str]], seed: int) -> None:
    """Assign train/val/test splits to entries."""
    labels = [entry["label"] for entry in entries]
    idx = np.arange(len(entries))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    
    try:
        train_idx, temp_idx = next(sss.split(idx, labels))
    except ValueError:
        LOGGER.warning("Stratified split failed (likely single class). Using random split.")
        np.random.seed(seed)
        perm = np.random.permutation(len(entries))
        split_idx = int(len(entries) * 0.8)
        train_idx = perm[:split_idx]
        temp_idx = perm[split_idx:]

    def set_split(indices: Iterable[int], name: str) -> None:
        for i in indices:
            entries[i]["split"] = name

    set_split(train_idx, "train")
    
    if len(temp_idx) > 0:
        temp_labels = [labels[i] for i in temp_idx]
        temp_idx_array = np.array(temp_idx)
        try:
            sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
            val_idx, test_idx = next(sss_val.split(temp_idx_array, temp_labels))
        except ValueError:
            mid = len(temp_idx) // 2
            val_idx = np.arange(mid)
            test_idx = np.arange(mid, len(temp_idx))
            
        set_split(temp_idx_array[val_idx], "val")
        set_split(temp_idx_array[test_idx], "test")


def _load_metadata(root: Path, seed: int = 1337) -> List[Dict[str, str]]:
    """Load metadata from CSV or scan directory."""
    metadata_path = root / "metadata.csv"
    entries: List[Dict[str, str]] = []

    if metadata_path.exists():
        df = _read_metadata_csv(metadata_path)
        path_col = _detect_column(df, METADATA_PATH_CANDIDATES)
        label_col = _detect_column(df, METADATA_LABEL_CANDIDATES)
        
        if path_col is None or label_col is None:
            LOGGER.warning("Metadata CSV found but columns missing. Falling back to directory scan.")
        else:
            split_col = _detect_column(df, METADATA_SPLIT_CANDIDATES)
            
            for _, row in df.iterrows():
                try:
                    video_path = _normalize_video_path(root, str(row[path_col]))
                    entries.append({
                        "path": str(video_path),
                        "label": str(row[label_col]),
                        "split": str(row.get(split_col, "train")).lower() if split_col else "train"
                    })
                except FileNotFoundError:
                    continue
            
            if split_col is None:
                _assign_splits(entries, seed)
            return entries

    videos = list_videos(root, VIDEO_EXTS)
    if not videos:
        if (root / "DCSASS Dataset").exists():
            videos = list_videos(root / "DCSASS Dataset", VIDEO_EXTS)
        
    if not videos:
        LOGGER.warning(f"No video files found under {root}.")
        return []

    for video in videos:
        label = video.parent.name
        if label.lower().endswith(VIDEO_EXTS):
            label = video.parent.parent.name
        entries.append({"path": str(video), "label": label})
    
    unique_labels = sorted(list(set(entry["label"] for entry in entries)))
    if hasattr(config, 'SELECTED_LABELS') and config.SELECTED_LABELS:
        LOGGER.info(f"Limiting to selected labels: {config.SELECTED_LABELS}")
        entries = [e for e in entries if e["label"] in config.SELECTED_LABELS]
    elif config.MAX_LABELS is not None and len(unique_labels) > config.MAX_LABELS:
        keep_labels = set(unique_labels[:config.MAX_LABELS])
        LOGGER.info(f"Limiting to first {config.MAX_LABELS} labels: {sorted(list(keep_labels))}")
        entries = [e for e in entries if e["label"] in keep_labels]
    else:
        LOGGER.info(f"Using all {len(unique_labels)} labels: {unique_labels}")

    _assign_splits(entries, seed)
    
    for entry in entries:
        entry["binary_label"] = 0 if entry["label"].lower() == "normal" else 1
        
    return entries


def load_split_entries(root: str | Path, split: str, seed: int = 1337) -> List[Dict[str, str]]:
    """Load entries for a specific split."""
    dataset_root = resolve_dcsass_root(root)
    entries = _load_metadata(dataset_root, seed=seed)
    filtered = [entry for entry in entries if entry.get("split") == split]
    if not filtered:
        LOGGER.warning(f"No entries found for split '{split}'.")
    return filtered


def _frames_to_array(frames: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Convert frames to numpy array."""
    if isinstance(frames, np.ndarray):
        return frames
    if not frames:
        return np.empty((0, 224, 224, 3), dtype=np.uint8)
    return np.stack(frames)


def _resolve_cache_dir(root: Path, cache_dir: Optional[str | Path]) -> Path:
    """Resolve cache directory path."""
    if cache_dir is None:
        return (root / "cache" / "frames").resolve()
    cache_dir = Path(cache_dir)
    if cache_dir.is_absolute():
        return cache_dir.resolve()
    return (root / cache_dir).resolve()


def _frame_cache_path(cache_root: Path, video_path: Path, image_size: Tuple[int, int]) -> Path:
    """Generate cache path for video frames."""
    key = f"{_CACHE_VERSION}|{video_path.resolve()}|{image_size[0]}x{image_size[1]}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return cache_root / f"{digest}.npz"


def _load_or_cache_frames(
    video_path: Path,
    image_size: Tuple[int, int],
    cache_decoded: bool,
    cache_dir: Optional[Path],
    max_frames: Optional[int] = None,
) -> np.ndarray:
    """Load frames from cache or decode and cache."""
    if not cache_decoded:
        return _frames_to_array(decode_video_opencv(str(video_path), target_size=image_size, max_frames=max_frames))
        
    if cache_dir is None:
        raise ValueError("cache_dir must be provided when cache_decoded is True")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _frame_cache_path(cache_dir, video_path, image_size)
    
    if cache_path.exists():
        try:
            with np.load(cache_path) as data:
                return data["frames"]
        except Exception:
            try:
                cache_path.unlink()
            except OSError:
                pass
                
    frames = _frames_to_array(decode_video_opencv(str(video_path), target_size=image_size, max_frames=max_frames))
    if frames.size > 0:
        try:
            np.savez_compressed(cache_path, frames=frames.astype(np.uint8))
        except Exception as exc:
            LOGGER.warning("Failed to write frame cache: %s", exc)
    return frames


def make_bag_dataset(
    root: str | Path,
    split: str,
    T: int = 32,
    stride: int = 3,
    batch_size: int = 1,
    image_size: Tuple[int, int] = (224, 224),
    seed: int = 1337,
    cache_decoded: bool = False,
    cache_dir: Optional[str | Path] = None,
    max_frames: int = 1000,
) -> "tf.data.Dataset":
    """Create a TensorFlow dataset of video bags for MIL training."""
    dataset_root = resolve_dcsass_root(root)
    entries_list = load_split_entries(dataset_root, split, seed=seed)
    cache_root = _resolve_cache_dir(dataset_root, cache_dir) if cache_decoded else None

    def generator() -> Iterator[Tuple[np.ndarray, np.int32, bytes]]:
        for entry in entries_list:
            video_path = Path(entry["path"])
            try:
                frames = _load_or_cache_frames(
                    video_path,
                    image_size,
                    cache_decoded,
                    cache_root,
                    max_frames=max_frames,
                )
            except (FileNotFoundError, RuntimeError):
                continue
            
            if frames.size == 0:
                continue
                
            segments = make_segments(frames, T=T, stride=stride)
            if segments.size == 0:
                continue
                
            segments = segments.astype(np.float32) / 255.0
            yield segments, np.int32(entry["binary_label"]), str(video_path).encode("utf-8")

    output_signature = (
        tf.TensorSpec(shape=(None, T, image_size[0], image_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.string),
    )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    
    ds = ds.map(
        lambda seg, label, vid: (tf.RaggedTensor.from_tensor(seg), label, vid),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    ds = make_bag_dataset(None, "train", batch_size=1)
    print("Dataset created.")
