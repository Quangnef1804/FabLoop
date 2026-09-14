"""Dataset adapters shared by photometric-stereo benchmark methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .diligent_validate import DiligentSample, import_cv2, load_diligent_sample, normalize_object_name


@dataclass(frozen=True)
class BenchmarkInput:
    object_name: str
    images: np.ndarray  # N,H,W,3 RGB float32, radiometrically calibrated
    raw_images: np.ndarray  # N,H,W,3 RGB float32 before light-intensity correction
    light_directions: np.ndarray  # N,3
    mask: np.ndarray  # H,W bool
    normal_gt: np.ndarray  # H,W,3
    indices: tuple[int, ...]
    image_names: tuple[str, ...]


def _read_rgb(path: Path) -> np.ndarray:
    cv2 = import_cv2()
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read observation image: {path}")
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Unsupported observation shape {image.shape}: {path}")
    rgb = image[:, :, :3][:, :, ::-1]
    if np.issubdtype(rgb.dtype, np.integer):
        scale = float(np.iinfo(rgb.dtype).max)
    else:
        scale = float(np.nanmax(rgb)) if float(np.nanmax(rgb)) > 1.0 else 1.0
    return rgb.astype(np.float32) / max(scale, 1.0)


def load_benchmark_input(sample: DiligentSample, indices: Sequence[int]) -> BenchmarkInput:
    chosen = tuple(int(index) for index in indices)
    if len(chosen) < 3 or len(set(chosen)) != len(chosen):
        raise ValueError("A benchmark subset needs at least three unique indices")
    if min(chosen) < 0 or max(chosen) >= len(sample.image_paths):
        raise IndexError("A selected image index is outside the dataset")

    images: list[np.ndarray] = []
    raw_images: list[np.ndarray] = []
    for index in chosen:
        rgb = _read_rgb(sample.image_paths[index])
        if rgb.shape[:2] != sample.mask.shape:
            raise ValueError(f"Image and mask shapes differ: {sample.image_paths[index]}")
        rgb[~sample.mask] = 0.0
        raw_images.append(rgb.astype(np.float32, copy=True))
        if sample.light_intensities is not None:
            rgb = rgb / np.maximum(sample.light_intensities[index].reshape(1, 1, 3), 1.0e-8)
        images.append(rgb.astype(np.float32, copy=False))

    return BenchmarkInput(
        object_name=sample.object_name,
        images=np.stack(images),
        raw_images=np.stack(raw_images),
        light_directions=sample.light_directions[np.asarray(chosen)].astype(np.float32),
        mask=sample.mask.copy(),
        normal_gt=sample.normal_gt.astype(np.float32),
        indices=chosen,
        image_names=tuple(sample.image_paths[index].name for index in chosen),
    )


def discover_diligent_objects(root: Path, requested: Sequence[str] | None = None) -> list[DiligentSample]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DiLiGenT pmsData directory does not exist: {root}")
    wanted = {name.lower() for name in requested or []}
    samples: list[DiligentSample] = []
    for folder in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name.lower()):
        if wanted and normalize_object_name(folder.name) not in wanted:
            continue
        try:
            sample = load_diligent_sample(folder)
        except FileNotFoundError:
            continue
        if not wanted or sample.object_name in wanted:
            samples.append(sample)
    missing = wanted - {sample.object_name for sample in samples}
    if missing:
        raise FileNotFoundError(f"Requested DiLiGenT objects were not found: {sorted(missing)}")
    if not samples:
        raise FileNotFoundError(f"No DiLiGenT objects found under {root}")
    return samples
