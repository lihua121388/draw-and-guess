"""Geometry, direction semantics, gradient and detection integration checks."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs/nscr/.config"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/nscr/.matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.dcr import DCRFusion, spatial_sample
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.dcr_loss import directional_clearance, clearance_distribution, dcr_auxiliary
from dcr_experiments import architecture
from nscr_experiments import transfer_pretrained

torch.set_num_threads(2)


def batch(boxes, indices=None, classes=None):
    return {"img": torch.zeros(2, 3, 64, 64), "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "batch_idx": torch.tensor(indices if indices is not None else [0] * len(boxes)),
            "cls": torch.tensor(classes if classes is not None else [0] * len(boxes)).reshape(-1, 1)}


class GeometryTests(unittest.TestCase):
    def test_invalid_configuration_rejected(self):
        for temperature in (0., float("nan")):
            with self.assertRaises(ValueError):
                clearance_distribution(torch.zeros(1, 4), temperature=temperature)
        with self.assertRaises(ValueError):
            directional_clearance(torch.empty(0, 4), (64, 64), cap=float("nan"))
        model = DetectionModel(architecture("D3"), verbose=False)
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        model.yaml["dcr"]["small_size"] = -1
        with self.assertRaises(ValueError):
            model.init_criterion()

    def test_exact_normalized_boundary_gaps_and_corridors(self):
        boxes = torch.tensor([[20., 20., 4., 4.], [32., 20., 4., 4.], [21., 50., 4., 4.], [22., 40., 4., 4.]])
        q = directional_clearance(boxes, (100, 100), cap=20, chunk_size=1)
        # Right gap=(30-22)/4=2; downward closest gap=(38-22)/4=4.
        torch.testing.assert_close(q[0], torch.tensor([2., 4.5, 4., 4.5]))
        torch.testing.assert_close(q, directional_clearance(boxes, (100, 100), cap=20, chunk_size=128))
        torch.testing.assert_close(q, directional_clearance(boxes * 3, (300, 300), cap=20))

    def test_overlap_and_self_exclusion(self):
        boxes = torch.tensor([[30., 30., 8., 8.], [30., 30., 8., 8.]])
        torch.testing.assert_close(directional_clearance(boxes, (100, 100)), torch.zeros(2, 4))
        q = directional_clearance(boxes[:1], (100, 100))
        self.assertGreater(q.min().item(), 0)
        self.assertEqual(directional_clearance(boxes[:0], (100, 100)).shape, (0, 4))

    def test_reflection_equivariance(self):
        boxes = torch.tensor([[20., 30., 6., 8.], [31., 32., 4., 4.], [50., 52., 8., 6.]])
        q = directional_clearance(boxes, (80, 100))
        reflected = boxes.clone()
        reflected[:, 0] = 100 - reflected[:, 0]
        torch.testing.assert_close(directional_clearance(reflected, (80, 100)), q[:, [1, 0, 2, 3]])
        reflected = boxes.clone()
        reflected[:, 1] = 80 - reflected[:, 1]
        torch.testing.assert_close(directional_clearance(reflected, (80, 100)), q[:, [0, 1, 3, 2]])

    def test_perpendicular_nonoverlap_does_not_block(self):
        boxes = torch.tensor([[30., 30., 4., 4.], [35., 35., 4., 4.]])
        q = directional_clearance(boxes, (100, 100))
        torch.testing.assert_close(q, torch.full((2, 4), 6.))

    def test_sample_direction_and_no_wrap(self):
        x = torch.arange(12.).reshape(1, 1, 3, 4)
        right = spatial_sample(x, 1, 0)
        torch.testing.assert_close(right[..., :-1], x[..., 1:])
        self.assertEqual(right[..., -1].sum().item(), 0)
        up = spatial_sample(x, 0, -1)
        self.assertEqual(up[..., 0, :].sum().item(), 0)
        torch.testing.assert_close(up[..., 1:, :], x[..., :-1, :])

    def test_label_controls_preserve_context_mass(self):
        q = torch.tensor([[0., 1., 3., 6.], [6., 0., 4., 1.]])
        target = clearance_distribution(q)
        iso = clearance_distribution(q, supervision="isotropic")
        torch.manual_seed(42)
        shuffled = clearance_distribution(q, supervision="shuffle")
        torch.testing.assert_close(target.sum(1), torch.ones(2))
        torch.testing.assert_close(iso[:, 1:], iso[:, 1:2].expand(-1, 4))
        torch.testing.assert_close(target[:, 0], iso[:, 0])
        torch.testing.assert_close(target[:, 0], shuffled[:, 0])
        torch.testing.assert_close(target[:, 1:].sort(1).values, shuffled[:, 1:].sort(1).values)
        self.assertFalse(torch.equal(target, shuffled))

    def test_masks_single_invalid_large_and_cross_image(self):
        cfg = dict(nc=10, supervision="directional", small_size=4)
        cases = [batch([]), batch([[.5, .5, .02, .02]]),
                 batch([[.3, .3, .2, .2], [.4, .3, .2, .2]]),
                 batch([[.3, .3, .02, .02], [.4, .3, .02, .02]], indices=[0, 1]),
                 batch([[.3, .3, .02, .02], [.4, .3, .02, .02]], classes=[0, -1]),
                 batch([[.3, .3, .02, .02], [float("nan"), .3, .02, .02]])]
        ignored = batch([[.3, .3, .02, .02], [.4, .3, .02, .02]])
        ignored["spacing_valid"] = torch.tensor([True, False])
        cases.append(ignored)
        for data in cases:
            logits = torch.zeros(2, 5, 16, 16, requires_grad=True)
            loss = dcr_auxiliary(logits, data, cfg)
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_supervised_gradient_prefers_clear_direction(self):
        data = batch([[.45, .5, .05, .05], [.5, .5, .05, .05]])
        logits = torch.zeros(2, 5, 16, 16, requires_grad=True)
        loss = dcr_auxiliary(logits, data, dict(nc=10, supervision="directional"))
        loss.backward()
        # Opposing horizontal blockers make horizontal channels less desirable
        # than the empty vertical corridor in both instances.
        gradient = logits.grad.sum((0, 2, 3))
        self.assertGreater(gradient[1].item(), gradient[3].item())
        self.assertGreater(gradient[2].item(), gradient[4].item())
        self.assertEqual(logits.grad[1].abs().sum().item(), 0)


class IntegrationTests(unittest.TestCase):
    def test_matched_ablations_backward_and_parameter_count(self):
        counts = []
        data = batch([[.3, .3, .15, .15], [.35, .3, .15, .15]])
        data["img"] = torch.rand_like(data["img"])
        for experiment in ("D1", "D2", "D3", "D4"):
            torch.manual_seed(42)
            model = DetectionModel(architecture(experiment), verbose=False)
            counts.append(sum(p.numel() for p in model.parameters()))
            model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
            model.train()
            loss, items = model(data)
            self.assertEqual(len(items), 4)
            self.assertTrue(torch.isfinite(loss).all())
            loss.sum().backward()
            router = next(m for m in model.modules() if isinstance(m, DCRFusion))
            self.assertIsNone(router.gate_logits)
            self.assertGreater(router.fusion.conv.weight.grad.abs().sum().item(), 0)
            if experiment == "D1":
                self.assertEqual(items[-1].item(), 0)
            else:
                self.assertGreater(items[-1].item(), 0)
                self.assertGreater(router.gate[-1].weight.grad.abs().sum().item(), 0)
            self.assertEqual(model.stride.tolist(), [4., 8., 16., 32.])
        self.assertEqual(len(set(counts)), 1)

    def test_backbone_transfer_and_rgb_only_reload(self):
        cfg = architecture("D3")
        self.assertEqual(cfg["backbone"], architecture("E0")["backbone"])
        self.assertEqual(cfg["head"][:-2], architecture("E1")["head"][:-2])
        model = DetectionModel(cfg, verbose=False)
        report = transfer_pretrained(model, ROOT / "model/yolo11n.pt")
        self.assertTrue(report["detect_branch_remap"])
        model.args = dict(DEFAULT_CFG_DICT)
        model.eval()
        with patch("ultralytics.utils.dcr_loss.directional_clearance", side_effect=AssertionError("GT at inference")):
            with torch.no_grad():
                output = model(torch.zeros(1, 3, 64, 64))[0]
        self.assertEqual(output.shape[1], 14)
        with tempfile.TemporaryDirectory(dir=ROOT / "runs/nscr") as directory:
            path = Path(directory) / "dcr.pt"
            torch.save({"model": model, "train_args": dict(DEFAULT_CFG_DICT)}, path)
            loaded = YOLO(str(path))
            with torch.no_grad():
                torch.testing.assert_close(loaded.model(torch.zeros(1, 3, 64, 64))[0], output, rtol=.01, atol=.01)


if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
