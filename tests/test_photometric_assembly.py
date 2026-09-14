from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photometric_stereo.assemble import merge_passes
from photometric_stereo.benchmark import write_benchmark_tables
from photometric_stereo.visualization import coverage_note, is_complete


def _accuracy(obj, method, lights, status="ok", mae=5.0):
    record = {"object": obj, "method": method, "lights": lights, "status": status}
    if status == "ok":
        record.update({
            "mae_deg": mae, "median_deg": mae, "p95_deg": mae,
            "runtime_sec": 9.0, "runtime_median_sec": 9.0, "runtime_runs_sec": [9.0],
            "end_to_end_sec": 9.5, "repeat_outputs_identical": True,
            "mae_deg_std_over_runs": 0.0, "peak_gpu_memory_mb": 100.0,
        })
    return record


def _runtime(obj, method, lights, status="ok", mae=5.0, median=2.0):
    record = {"object": obj, "method": method, "lights": lights, "status": status}
    if status == "ok":
        record.update({
            "mae_deg": mae, "runtime_sec": median, "runtime_median_sec": median,
            "runtime_iqr_sec": 0.1, "runtime_std_sec": 0.05, "runtime_runs_sec": [2.1, 2.0, 1.9],
            "runtime_measured_runs": 3, "end_to_end_sec": 2.5,
            "repeat_outputs_identical": True, "mae_deg_std_over_runs": 0.0,
        })
    return record


class MergePassesTests(unittest.TestCase):
    def test_cited_runtime_comes_only_from_the_repeated_pass(self):
        merged = merge_passes([_accuracy("ball", "l1", 4)], [_runtime("ball", "l1", 4)])[0]
        self.assertEqual(merged["runtime_sec"], 2.0)
        self.assertEqual(merged["runtime_iqr_sec"], 0.1)
        self.assertEqual(merged["runtime_runs_sec"], [2.1, 2.0, 1.9])
        self.assertEqual(merged["runtime_source"], "runtime_pass")
        self.assertEqual(merged["accuracy_pass_single_run_sec"], 9.0)
        self.assertTrue(merged["runtime_pass_mae_matches"])
        self.assertEqual(merged["mae_deg"], 5.0)

    def test_unmeasured_case_carries_no_runtime_that_could_be_cited(self):
        merged = merge_passes([_accuracy("cat", "l1", 8)], [])[0]
        self.assertIsNone(merged["runtime_sec"])
        self.assertEqual(merged["runtime_source"], "not_measured")
        self.assertNotIn("runtime_median_sec", merged)
        self.assertNotIn("runtime_runs_sec", merged)
        self.assertNotIn("repeat_outputs_identical", merged)
        self.assertEqual(merged["accuracy_pass_single_run_sec"], 9.0)
        self.assertEqual(merged["peak_gpu_memory_mb"], 100.0)

    def test_mae_disagreement_between_passes_is_flagged(self):
        merged = merge_passes([_accuracy("ball", "l1", 4, mae=5.0)], [_runtime("ball", "l1", 4, mae=5.5)])[0]
        self.assertFalse(merged["runtime_pass_mae_matches"])

    def test_oom_case_keeps_its_row_without_any_accuracy_value(self):
        merged = merge_passes([_accuracy("ball", "sdm_unips", 96, status="unsupported_oom")], [])[0]
        self.assertEqual(merged["status"], "unsupported_oom")
        self.assertNotIn("mae_deg", merged)
        self.assertIsNone(merged["runtime_sec"])

    def test_runtime_pass_oom_is_recorded_instead_of_a_runtime(self):
        merged = merge_passes(
            [_accuracy("ball", "sdm_unips", 32)], [_runtime("ball", "sdm_unips", 32, status="unsupported_oom")]
        )[0]
        self.assertIsNone(merged["runtime_sec"])
        self.assertEqual(merged["runtime_source"], "runtime_pass_unsupported_oom")


class GpuMemorySpillTests(unittest.TestCase):
    """Measured SDM-UniPS Ball x 16 peaked at 4977 MB on a 4096 MB card: the driver
    spilled into system memory, so accuracy is valid but runtime measures PCIe."""

    def _merged(self, peak):
        record = _accuracy("ball", "sdm_unips", 16)
        record["peak_gpu_memory_mb"] = peak
        record["gpu_total_memory_mb"] = 4095.5
        return merge_passes([record], [])[0]

    def test_peak_above_physical_memory_is_flagged(self):
        self.assertTrue(self._merged(4977.06)["gpu_memory_exceeds_physical"])

    def test_peak_within_physical_memory_is_not_flagged(self):
        self.assertFalse(self._merged(1448.84)["gpu_memory_exceeds_physical"])

    def test_cpu_and_oom_cases_are_not_flagged(self):
        self.assertFalse(self._merged(0.0)["gpu_memory_exceeds_physical"])
        oom = merge_passes([_accuracy("ball", "sdm_unips", 96, status="unsupported_oom")], [])[0]
        self.assertFalse(oom["gpu_memory_exceeds_physical"])


class RuntimeCaveatTests(unittest.TestCase):
    def test_caveat_travels_with_the_case_without_changing_its_numbers(self):
        caveats = {("pot1", "l1", 4): {
            "object": "pot1", "method": "l1", "lights": 4,
            "caveat": "screen_locked_e_core_throttle", "evidence": "x4.65 vs pass 1",
        }}
        merged = merge_passes([_accuracy("pot1", "l1", 4)], [_runtime("pot1", "l1", 4, median=468.0)], caveats)[0]
        self.assertEqual(merged["runtime_sec"], 468.0)
        self.assertEqual(merged["runtime_caveat"], "screen_locked_e_core_throttle")
        self.assertEqual(merged["runtime_caveat_evidence"], "x4.65 vs pass 1")

    def test_uncaveated_case_has_no_caveat_field(self):
        merged = merge_passes([_accuracy("ball", "l1", 4)], [_runtime("ball", "l1", 4)], {})[0]
        self.assertNotIn("runtime_caveat", merged)


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            _accuracy("ball", "l2", 96), _accuracy("cat", "l2", 96),
            _accuracy("ball", "sdm_unips", 96), _accuracy("cat", "sdm_unips", 96, status="unsupported_oom"),
        ]

    def test_partial_coverage_is_not_averaged_against_complete_methods(self):
        self.assertTrue(is_complete(self.records, "l2", 96))
        self.assertFalse(is_complete(self.records, "sdm_unips", 96))

    def test_coverage_note_states_count_and_reason(self):
        note = coverage_note(self.records, "sdm_unips", 96)
        self.assertIn("1/2", note)
        self.assertIn("unsupported_oomx1", note)


class SummaryColumnOrderTests(unittest.TestCase):
    def test_leading_fields_come_first_in_csv(self):
        records = [_accuracy("ball", "l2", 4)]
        with tempfile.TemporaryDirectory() as temporary:
            write_benchmark_tables(Path(temporary), records, leading_fields=("object", "method", "lights", "status", "mae_deg"))
            with (Path(temporary) / "benchmark_summary.csv").open(encoding="utf-8") as handle:
                header = next(csv.reader(handle))
        self.assertEqual(header[:5], ["object", "method", "lights", "status", "mae_deg"])


if __name__ == "__main__":
    unittest.main()
