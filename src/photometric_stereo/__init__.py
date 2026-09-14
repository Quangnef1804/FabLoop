"""Photometric stereo preparation, validation and benchmark tools."""

from .datasets import BenchmarkInput, discover_diligent_objects, load_benchmark_input
from .lighting import select_cardinal_four, select_light_subsets
from .metrics import normal_error_metrics

__all__ = [
    "BenchmarkInput",
    "discover_diligent_objects",
    "load_benchmark_input",
    "normal_error_metrics",
    "select_cardinal_four",
    "select_light_subsets",
]

