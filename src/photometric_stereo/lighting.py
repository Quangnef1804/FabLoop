"""Deterministic, geometry-aware DiLiGenT light subset selection."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .diligent_validate import light_geometry, normalize_normals


CARDINAL_NAMES = ("right", "front", "left", "back")
CARDINAL_AZIMUTH_DEG = (0.0, 90.0, 180.0, -90.0)


def _wrapped_angle_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def select_cardinal_four(light_directions: np.ndarray) -> list[int]:
    """Pick unique lights closest in azimuth to four image-plane cardinal directions.

    Names describe calibrated image-plane azimuths. They are an approximation to a
    physical F/B/L/R box and are deliberately reported with their actual vectors.
    """
    unit = normalize_normals(np.asarray(light_directions, dtype=np.float64))
    if unit.shape[0] < 4:
        raise ValueError("At least four calibrated light directions are required")
    azimuth = np.degrees(np.arctan2(unit[:, 1], unit[:, 0]))
    targets = np.asarray(CARDINAL_AZIMUTH_DEG)[:, None]
    cost = _wrapped_angle_distance(targets, azimuth[None, :])
    target_rows, selected_cols = linear_sum_assignment(cost)
    ordered = [int(selected_cols[np.where(target_rows == row)[0][0]]) for row in range(4)]
    light_geometry(unit[np.asarray(ordered)])
    return ordered


def _farthest_point_indices(unit: np.ndarray, initial: Sequence[int], count: int) -> list[int]:
    selected = list(dict.fromkeys(int(index) for index in initial))
    while len(selected) < count:
        remaining = np.asarray([index for index in range(len(unit)) if index not in selected])
        similarity = unit[remaining] @ unit[np.asarray(selected)].T
        min_angle = np.min(np.arccos(np.clip(similarity, -1.0, 1.0)), axis=1)
        best_score = float(np.max(min_angle))
        tied = remaining[np.isclose(min_angle, best_score, rtol=0.0, atol=1.0e-12)]
        selected.append(int(np.min(tied)))
    return selected


def select_light_subsets(
    light_directions: np.ndarray, counts: Sequence[int] = (4, 8, 16, 32, 96)
) -> dict[int, list[int]]:
    lights = normalize_normals(np.asarray(light_directions, dtype=np.float64))
    unique_counts = sorted(set(int(count) for count in counts))
    if not unique_counts or unique_counts[0] < 4 or unique_counts[-1] > len(lights):
        raise ValueError(f"Light counts must lie in [4, {len(lights)}]")
    cardinal = select_cardinal_four(lights)
    selected = _farthest_point_indices(lights, cardinal, unique_counts[-1])
    subsets = {
        count: (list(range(len(lights))) if count == len(lights) else selected[:count])
        for count in unique_counts
    }
    for indices in subsets.values():
        light_geometry(lights[np.asarray(indices)])
    return subsets


def describe_subset(light_directions: np.ndarray, indices: Sequence[int]) -> dict[str, object]:
    geometry = light_geometry(np.asarray(light_directions)[np.asarray(indices)])
    geometry["indices_zero_based"] = [int(index) for index in indices]
    if len(indices) == 4:
        geometry["cardinal_target_order"] = list(CARDINAL_NAMES)
        geometry["cardinal_target_azimuth_deg"] = list(CARDINAL_AZIMUTH_DEG)
    return geometry
