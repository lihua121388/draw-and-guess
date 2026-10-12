"""Mechanism checks: identical starts, actual gradients, isolation, checkpoint and YOLO loss."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train_test import YOLO
import torch
from fusion_experiments import EXPERIMENTS, architecture
from candidate_experiments import architecture as candidate_architecture
from nscr_experiments import transfer_pretrained
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        (ROOT / "runs/fusion").mkdir(parents=True, exist_ok=True)

    def test_same_common_parameters_and_initial_function(self):
        torch.manual_seed(42)
        baseline = DetectionModel(candidate_architecture("M0"), verbose=False).eval()
        image = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            reference = baseline(image)[0]
        for experiment in EXPERIMENTS:
            torch.manual_seed(42)
            model = DetectionModel(architecture(experiment), verbose=False).eval()
            self.assertEqual(model.stride.tolist(), [4., 8., 16., 32.])
            for key, tensor in baseline.state_dict().items():
                torch.testing.assert_close(model.state_dict()[key], tensor, rtol=0, atol=0)
            with torch.no_grad():
                torch.testing.assert_close(model(image)[0], reference, rtol=0, atol=0)

    def test_gradient_and_loss_and_architecture_isolation(self):
        for experiment in EXPERIMENTS:
            cfg = architecture(experiment)
            base = candidate_architecture("M0")
            self.assertEqual(cfg["backbone"], base["backbone"])
            self.assertEqual(cfg["head"][:-2], base["head"][:-2])
            self.assertEqual(cfg["head"][-1], base["head"][-1])
            model = DetectionModel(cfg, verbose=False)
            model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
            batch = dict(img=torch.rand(2, 3, 64, 64), cls=torch.tensor([[0.], [3.], [9.]]),
                bboxes=torch.tensor([[.3, .3, .12, .12], [.6, .6, .2, .2], [.4, .4, .15, .15]]),
                batch_idx=torch.tensor([0, 0, 1]))
            loss, terms = model(batch)
            self.assertTrue(torch.isfinite(loss).all())
            self.assertEqual(len(terms), 4)
            self.assertEqual(float(terms[-1]), 0.)
            loss.sum().backward()
            fusion = model.model[23]
            for name in ("semantic_gate", "scale_logits", "detail"):
                if hasattr(fusion, name):
                    value = getattr(fusion, name)
                    parameter = value.weight if hasattr(value, "weight") else value
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    # Detail-gate gradients start at zero because its DW residual
                    # starts at zero. Verify gate can learn once detail is nonzero.
                    if name != "semantic_gate" or experiment != "F3":
                        self.assertGreater(float(parameter.grad.abs().sum()), 0.)
            if experiment == "F3":
                with torch.no_grad():
                    fusion.detail.weight.fill_(.01)
                model.zero_grad()
                model(batch)[0].sum().backward()
                self.assertGreater(float(fusion.semantic_gate.weight.grad.abs().sum()), 0.)

    def test_pretrained_remap_and_checkpoint(self):
        source = YOLO(str(ROOT / "model/yolo11n.pt")).model.state_dict()
        for experiment in EXPERIMENTS:
            model = DetectionModel(architecture(experiment), verbose=False)
            report = transfer_pretrained(model, ROOT / "model/yolo11n.pt")
            self.assertTrue(report["detect_branch_remap"])
            torch.testing.assert_close(source["model.23.cv2.0.0.conv.weight"].float(),
                model.state_dict()["model.24.cv2.1.0.conv.weight"])
            model.args = dict(DEFAULT_CFG_DICT)
            model.eval()
            with tempfile.TemporaryDirectory(dir=ROOT / "runs/fusion") as folder:
                path = Path(folder) / "model.pt"
                torch.save({"model": model, "train_args": dict(DEFAULT_CFG_DICT)}, path)
                loaded = YOLO(str(path)).model.eval()
                image = torch.rand(1, 3, 64, 64)
                with torch.no_grad():
                    torch.testing.assert_close(loaded(image)[0], model(image)[0])

    def test_training_monitor_and_inference_cache(self):
        from train_fusion import FusionTrainer
        import csv
        model = DetectionModel(architecture("F2"), verbose=False)
        model(torch.rand(2, 3, 64, 64))
        self.assertIsNotNone(model.model[23].last_weights)
        with tempfile.TemporaryDirectory(dir=ROOT / "runs/fusion") as folder:
            trainer = SimpleNamespace(model=model, epoch=0, save_dir=Path(folder))
            FusionTrainer.record_weights(None, trainer)
            self.assertIsNone(model.model[23].last_weights)
            with (Path(folder) / "fusion_weights.csv").open(encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row["mean"]), .5)
        model.eval()
        with torch.no_grad():
            model(torch.rand(1, 3, 64, 64))
        self.assertIsNone(model.model[23].last_weights)


if __name__ == "__main__":
    unittest.main()
