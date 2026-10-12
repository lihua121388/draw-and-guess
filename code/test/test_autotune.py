"""Check throughput ranking, failed probes and saved-profile compatibility."""

import json
import itertools
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tuning_utils import continuous_batches, load_profile, rank_configurations


def row(batch, workers, ips, memory=.5, status="ok"):
    return {"batch": batch, "workers": workers, "images_per_second": ips,
            "peak_memory_fraction": memory, "status": status}


class TuningTests(unittest.TestCase):
    def test_probe_crosses_epoch_boundary_and_rejects_empty_loader(self):
        self.assertEqual(list(itertools.islice(continuous_batches([1, 2]), 5)), [1, 2, 1, 2, 1])
        with self.assertRaises(ValueError):
            next(continuous_batches([]))

    def test_images_per_second_selects_larger_batch(self):
        best = rank_configurations([row(8, 4, 8 * 9), row(16, 8, 16 * 6.83)])[0]
        self.assertEqual((best["batch"], best["workers"]), (16, 8))

    def test_memory_limit_and_oom_rejected(self):
        best = rank_configurations([row(16, 4, 100), row(32, 4, 150, .9), row(64, 4, 0, status="oom")])[0]
        self.assertEqual(best["batch"], 16)

    def test_near_tie_prefers_fewer_workers(self):
        best = rank_configurations([row(16, 4, 98), row(16, 16, 100)])[0]
        self.assertEqual(best["workers"], 4)

    def test_repeat_uses_median_and_failure_disqualifies(self):
        ranked = rank_configurations([row(8, 4, 500), row(8, 4, 50), row(8, 4, 51), row(16, 4, 100),
                                      row(32, 8, 200), row(32, 8, 0, status="oom")])
        self.assertEqual(ranked[0]["batch"], 16)
        self.assertNotIn(32, [r["batch"] for r in ranked])

    def test_no_safe_candidates_is_error(self):
        with self.assertRaises(RuntimeError):
            rank_configurations([row(64, 4, 0, status="oom")])

    def test_profile_requires_context_and_experiment_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            payload = {"status": "complete", "signature": {"imgsz": 640},
                       "verified_experiments": ["E0", "E4"], "recommendation": {"batch": 16, "workers": 8}}
            path.write_text(json.dumps(payload))
            self.assertEqual(load_profile(path, {"imgsz": 640}, ["E0"])["batch"], 16)
            with self.assertRaises(ValueError):
                load_profile(path, {"imgsz": 320}, ["E0"])
            with self.assertRaises(ValueError):
                load_profile(path, {"imgsz": 640}, ["E6"])
            payload["status"] = "in_progress"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_profile(path, {"imgsz": 640}, ["E0"])

    def test_active_training_detection_excludes_validation(self):
        from auto_tune import active_training_processes

        class Process:
            def __init__(self, pid, command):
                self.info = {"pid": pid, "cmdline": command}

        processes = [Process(1, ["python", "train_test.py", "ablate"]),
                     Process(2, ["python", "train_test.py", "val"]),
                     Process(3, ["python", "train_test.py", "--help"])]
        with patch("psutil.process_iter", return_value=processes):
            self.assertEqual([p["pid"] for p in active_training_processes()], [1])


if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
