from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photometric_stereo.research_questions import (
    holm_adjust,
    rq1_accuracy_loss,
    rq2_versus_l2_at_four,
    rq3_tradeoff,
    sign_test_p,
)

OBJECTS = [f"obj{i}" for i in range(10)]


def _ok(obj, method, lights, mae, p95=None, runtime=None, peak=0.0):
    return {
        "object": obj, "method": method, "lights": lights, "status": "ok",
        "mae_deg": mae, "p95_deg": mae * 2 if p95 is None else p95,
        "runtime_sec": runtime, "peak_gpu_memory_mb": peak,
    }


class StatisticsTests(unittest.TestCase):
    def test_sign_test_matches_exact_binomial(self):
        # 9 of 10 in one direction: 2 * (C(10,0) + C(10,1)) / 2**10
        self.assertAlmostEqual(sign_test_p([-1.0] * 9 + [1.0]), 22 / 1024)

    def test_sign_test_drops_ties_and_handles_empty(self):
        self.assertIsNone(sign_test_p([0.0, 0.0]))
        self.assertAlmostEqual(sign_test_p([0.0, -1.0, -1.0]), 0.5)

    def test_holm_is_monotone_and_keeps_untestable_as_none(self):
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": None})
        self.assertAlmostEqual(adjusted["a"], 0.02)
        self.assertAlmostEqual(adjusted["b"], 0.04)
        self.assertIsNone(adjusted["c"])


class ResearchQuestionTests(unittest.TestCase):
    def _records(self):
        records = []
        for i, obj in enumerate(OBJECTS):
            records.append(_ok(obj, "l2", 4, 10.0 + i, runtime=0.1))
            records.append(_ok(obj, "l2", 96, 9.0 + i))
            # Consistently better than L2 with distinct gaps, so the exact test is decisive.
            records.append(_ok(obj, "ps_fcn", 4, 10.0 + i - (1.0 + 0.1 * i), runtime=1.0, peak=400.0))
            records.append(_ok(obj, "ps_fcn", 96, 7.0 + i))
            # Mixed signs around L2: no evidence of a difference.
            records.append(_ok(obj, "l1", 4, 10.0 + i + (0.5 if i % 2 else -0.5) * (1 + 0.1 * i)))
            records.append(_ok(obj, "sdm_unips", 4, 5.0 + i, runtime=10.0, peak=1400.0))
            status = "ok" if i else "unsupported_oom"
            if status == "ok":
                records.append(_ok(obj, "sdm_unips", 96, 4.0 + i))
            else:
                records.append({"object": obj, "method": "sdm_unips", "lights": 96, "status": status})
        return records

    def test_rq2_consistent_gain_is_significant_and_mixed_signs_are_not(self):
        result = rq2_versus_l2_at_four(self._records())["comparisons"]
        self.assertEqual(result["ps_fcn"]["objects_better_than_l2"], 10)
        self.assertEqual(result["ps_fcn"]["verdict"], "significantly better than L2")
        self.assertEqual(result["l1"]["verdict"], "no significant difference from L2")
        self.assertLess(result["ps_fcn"]["delta_mae_deg"], 0)

    def test_rq1_never_compares_against_an_incompletely_covered_light_count(self):
        result = rq1_accuracy_loss(self._records())
        self.assertNotIn(96, result["sdm_unips"]["supported_light_counts"])
        self.assertNotIn("vs_96_lights", result["sdm_unips"])
        self.assertEqual(result["sdm_unips"]["unsupported_light_counts"], [96])
        self.assertAlmostEqual(result["l2"]["vs_96_lights"]["mae_loss_deg"], 1.0)
        self.assertEqual(result["l2"]["vs_96_lights"]["objects_worse_at_4"], 10)

    def test_rq3_pareto_excludes_methods_without_repeated_runtime(self):
        result = rq3_tradeoff(self._records())
        front = result["pareto_front_mae_runtime_memory"]
        self.assertNotIn("l1", front)  # no runtime measured
        self.assertIn("l2", front)  # fastest and lightest
        self.assertIn("sdm_unips", front)  # most accurate
        self.assertIsNone(result["methods"]["l1"]["runtime_sec_at_4_median"])
        self.assertIn("GPL", result["methods"]["l1"]["licence"])
        self.assertIn("non-commercial", result["methods"]["sdm_unips"]["licence"])


class RuntimeCaveatReportingTests(unittest.TestCase):
    def test_rq3_reports_runtime_with_and_without_caveated_cases(self):
        records = [_ok(obj, "l1", 4, 10.0, runtime=seconds) for obj, seconds in (("a", 10.0), ("b", 20.0), ("c", 90.0))]
        records[2]["runtime_caveat"] = "screen_locked_e_core_throttle"
        row = rq3_tradeoff(records)["methods"]["l1"]
        self.assertEqual(row["runtime_sec_at_4_median"], 20.0)
        self.assertEqual(row["runtime_sec_at_4_median_excluding_caveats"], 15.0)
        self.assertEqual(row["runtime_caveats_at_4"], ["c"])


if __name__ == "__main__":
    unittest.main()
