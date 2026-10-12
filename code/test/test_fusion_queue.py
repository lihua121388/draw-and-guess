"""A changed formal budget requires a fresh, first-position F0 comparison control."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from run_fusion import check_reference, check_resume_request


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        reference = root / "M0_seed42"
        (reference / "evaluation").mkdir(parents=True)
        (reference / "summary.json").write_text(json.dumps({"epochs": 100}))
        (reference / "evaluation/evaluation.json").write_text("{}")
        data = root / "dataset.yaml"
        data.write_text("path: unused\n")
        manifest = dict(arguments=dict(data=str(data.resolve()), effective_training_arguments=dict(
            batch=8, workers=4, imgsz=640, device="0")), checkpoint_sha256="same",
            dataset_files={}, dataset_labels_sha256={})
        (reference / "experiment.json").write_text(json.dumps(manifest))
        self.args = SimpleNamespace(epochs=100, batch=16, workers=4, imgsz=640, device="0", seeds=[42],
            experiments=["F2_GLOBAL", "F2"], data=data, weights=root / "weights.pt",
            baseline_project=root, project=root / "new_queue")

    @patch("nscr_experiments.sha256", return_value="same")
    def test_changed_batch_without_new_baseline_rejected(self, _):
        with self.assertRaisesRegex(ValueError, "run F0 first"):
            check_reference(self.args)

    @patch("nscr_experiments.sha256", return_value="same")
    def test_changed_batch_with_first_F0_accepted(self, _):
        self.args.experiments.insert(0, "F0")
        result = check_reference(self.args)
        self.assertEqual(result["status"], "new_F0_control")
        self.assertEqual(result["settings_differences"], {"batch": [8, 16]})
        self.assertEqual(Path(result["seed42_control"]), self.args.project / "F0_seed42")

    @patch("nscr_experiments.sha256", return_value="same")
    def test_baseline_later_in_queue_rejected(self, _):
        self.args.experiments.append("F0")
        with self.assertRaisesRegex(ValueError, "Run F0 first"):
            check_reference(self.args)

    @patch("nscr_experiments.sha256", return_value="same")
    def test_mixed_resume_uses_completed_historical_control(self, _):
        self.args.batch = 8
        self.args.experiments = ["F0", "F2"]
        self.args.resume_first = self.args.project / "F0_seed42/weights/last.pt"
        result = check_reference(self.args)
        self.assertEqual(result["status"], "mixed_history_resume")
        self.assertEqual(Path(result["seed42_control"]), self.args.baseline_project / "M0_seed42")

    def prepare_resume(self):
        import torch
        self.args.batch = 8
        self.args.experiments = ["F0", "F2"]
        self.args.resume_first = self.args.project / "F0_seed42/weights/last.pt"
        self.args.resume_first.parent.mkdir(parents=True)
        (self.args.project / "queue.json").write_text(json.dumps(dict(status="stopped_by_user", supervisor_pid=99999999)))
        saved = dict(epochs=100, workers=4, imgsz=640, seed=42, device="0", batch=16,
                     project=str(self.args.project), name="F0_seed42")
        torch.save(dict(epoch=21, optimizer={"state": {}}, train_args=saved,
                        ema=SimpleNamespace(yaml={"nscr": {"experiment": "F0"}})), self.args.resume_first)

    def test_resume_preserves_epoch_and_marks_batch_transition(self):
        self.prepare_resume()
        info = check_resume_request(self.args)
        self.assertEqual(info["completed_epochs"], 22)
        self.assertEqual((info["old_batch"], info["new_batch"]), (16, 8))
        self.assertTrue(info["excluded_as_formal_control"])

    def test_resume_rejects_already_started_pending_run(self):
        self.prepare_resume()
        (self.args.project / "F2_seed42").mkdir()
        with self.assertRaisesRegex(ValueError, "Pending run already exists"):
            check_resume_request(self.args)

    def test_mixed_history_excluded_even_when_final_arguments_match(self):
        from analyze_fusion_results import comparison
        reference = self.args.baseline_project / "M0_seed42"
        manifest = json.loads((reference / "experiment.json").read_text())
        manifest["architecture"] = dict(backbone=[], head=[[0], [1]])
        (reference / "experiment.json").write_text(json.dumps(manifest))
        run = self.args.project / "F0_seed42"
        run.mkdir(parents=True)
        (run / "experiment.json").write_text(json.dumps(manifest))
        self.assertTrue(comparison(run, reference)["matched"])
        (run / "batch_transition.json").write_text(json.dumps(dict(mixed_training_history=True,
            old_batch=16, new_batch=8, completed_epochs=22)))
        result = comparison(run, reference)
        self.assertFalse(result["matched"])
        self.assertIn("batch_history", result["differences"])

    def test_train_one_passes_explicit_checkpoint_instead_of_latest_lookup(self):
        import train_test
        self.args.data.write_text("names:\n  0: object\n")
        values = vars(self.args).copy()
        values.update(resume=self.args.project / "F0_seed42/weights/last.pt", name=None,
            spacing_weight=0., tau=2., temperature=.5, small_size=32., density_sigma=2.,
            density_threshold=2., density_temperature=.5, patience=101, no_amp=False,
            no_plots=False, cache="none", max_det=500, save_period=10)
        args = SimpleNamespace(**values)
        with patch.object(train_test, "ResearchTrainer", side_effect=RuntimeError("captured")) as trainer:
            with self.assertRaisesRegex(RuntimeError, "captured"):
                train_test.train_one(args, args.data, "F0", 42,
                    architecture_writer=lambda *a, **k: Path("model.yaml"))
        self.assertEqual(trainer.call_args.kwargs["overrides"]["resume"], str(args.resume))


if __name__ == "__main__":
    unittest.main()
