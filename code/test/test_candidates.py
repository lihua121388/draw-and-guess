"""Tests of distinct teacher geometry and train/inference/checkpoint compatibility."""
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
from ultralytics.nn.modules.candidates import CandidateFusion
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.candidate_loss import (boundary_samples, candidate_auxiliary,
    grid_phase_targets, max_cell_overlap, points_outside_boxes, ring_targets)
from candidate_experiments import architecture
from nscr_experiments import transfer_pretrained

torch.set_num_threads(2)


def batch(boxes, indices=None, classes=None):
    return {"img": torch.rand(2, 3, 64, 64),
        "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "batch_idx": torch.tensor(indices if indices is not None else [0]*len(boxes)),
        "cls": torch.tensor(classes if classes is not None else [0]*len(boxes)).reshape(-1, 1)}


class TeacherTests(unittest.TestCase):
    def test_grid_phase_changes_at_fixed_size(self):
        boxes = torch.tensor([[3., 3., 2., 2.], [4., 4., 2., 2.], [40., 40., 32., 32.]])
        target = grid_phase_targets(boxes)
        torch.testing.assert_close(target, torch.tensor([[.8, .2], [.5, .5], [.5, .5]]))
        torch.testing.assert_close(target.sum(1), torch.ones(3))

    def test_lattice_overlap_matches_brute_force(self):
        lo = torch.tensor([-1., 0., 2., 3., 7., 10.])
        hi = lo + torch.tensor([2., 8., 3., 2., 18., 1.])
        expected = torch.stack([((torch.minimum(b, torch.arange(-2, 10.)*8+8) -
            torch.maximum(a, torch.arange(-2, 10.)*8)).clamp_min(0)).max() for a, b in zip(lo, hi)])
        torch.testing.assert_close(max_cell_overlap(lo, hi, 8), expected)

    def test_ring_collision_reduces_outer_context(self):
        sparse = torch.tensor([[40., 40., 4., 4.], [90., 90., 4., 4.]])
        crowded = torch.tensor([[40., 40., 4., 4.], [40., 40., 48., 48.]])
        a, b = ring_targets(sparse, (128, 128)), ring_targets(crowded, (128, 128))
        self.assertGreater(a[0, 3].item(), b[0, 3].item())
        torch.testing.assert_close(a.sum(1), torch.ones(2))
        torch.testing.assert_close(a, ring_targets(sparse, (128, 128), chunk_size=1))
        torch.testing.assert_close(a, ring_targets(sparse*2, (256, 256), stride=8))

    def test_ring_inside_own_box_is_disabled(self):
        boxes = torch.tensor([[40., 40., 24., 24.], [90., 90., 4., 4.]])
        target = ring_targets(boxes, (128, 128))
        self.assertEqual(target[0, 1].item(), 0)
        self.assertGreater(target[0, 0].item(), 0)

    def test_boundary_negatives_never_enter_other_object(self):
        own = torch.tensor([[16., 16., 4., 4.]])
        all_boxes = torch.cat((own, torch.tensor([[22., 16., 8., 8.]])))
        points, labels = boundary_samples(own, all_boxes, (64, 64))
        self.assertEqual(int((labels == 1).sum()), 4)
        self.assertTrue(points_outside_boxes(points[labels == 0] * 64, all_boxes).all())
        self.assertLess(int((labels == 0).sum()), 4)

    def test_empty_invalid_and_ignore_masks(self):
        config = dict(nc=10, weight=.1, small_size=32., tau=2., temperature=.5, cap=6.)
        for kind, channels in (("dcr", 5), ("csr", 4), ("brr", 1), ("gpr", 2), ("pcr", 1)):
            for data in (batch([]), batch([[.3, .3, .1, .1]], classes=[-1])):
                logits = torch.zeros(2, channels, 16, 16, requires_grad=True)
                state = dict(logits=logits, embedding=torch.randn(2, 16, 16, 16, requires_grad=True))
                loss = candidate_auxiliary(kind, state, data, config)
                self.assertEqual(loss.item(), 0)
                loss.backward()
                self.assertEqual(logits.grad.abs().sum().item(), 0)
            data = batch([[.3, .3, .1, .1], [.4, .3, .1, .1]])
            data["spacing_valid"] = torch.zeros(2, dtype=torch.bool)
            self.assertEqual(candidate_auxiliary(kind, state, data, config).item(), 0)

    def test_pcr_embedding_receives_pair_gradients(self):
        config = dict(nc=10, weight=.1, small_size=32.)
        data = batch([[.3, .3, .1, .1], [.4, .3, .1, .1], [.5, .3, .1, .1]], classes=[0, 0, 1])
        embedding = torch.randn(2, 16, 16, 16, requires_grad=True)
        logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
        loss = candidate_auxiliary("pcr", dict(logits=logits, embedding=embedding), data, config)
        loss.backward()
        self.assertGreater(embedding.grad.abs().sum().item(), 0)
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        self.assertEqual(embedding.grad[1].abs().sum().item(), 0)


class IntegrationTests(unittest.TestCase):
    def test_all_candidates_backward_and_unchanged_backbone(self):
        data = batch([[.3, .3, .15, .15], [.42, .3, .1, .1], [.52, .3, .1, .1]], classes=[0, 0, 1])
        for experiment in ("M0", "M1", "M2", "M3", "M4", "M5"):
            with self.subTest(experiment=experiment):
                cfg = architecture(experiment)
                self.assertEqual(cfg["backbone"], architecture("M0")["backbone"])
                self.assertEqual(cfg["head"][:-2], architecture("M0")["head"][:-2])
                model = DetectionModel(cfg, verbose=False)
                model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
                model.train()
                loss, items = model(data)
                self.assertTrue(torch.isfinite(loss).all())
                self.assertEqual(len(items), 4)
                loss.sum().backward()
                self.assertEqual(model.stride.tolist(), [4., 8., 16., 32.])
                if experiment != "M0":
                    router = next(m for m in model.modules() if isinstance(m, CandidateFusion))
                    self.assertIsNone(router.aux_state)
                    self.assertGreater(items[-1].item(), 0)
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in router.parameters() if p.grad is not None))
                    head = router.impl.gate if experiment == "M1" else router.head
                    self.assertGreater(head[-1].weight.grad.abs().sum().item(), 0)

    def test_zero_weight_same_architecture_and_zero_aux(self):
        for experiment in ("M1", "M2", "M3", "M4", "M5"):
            torch.manual_seed(42)
            a = DetectionModel(architecture(experiment, weight=.1), verbose=False)
            torch.manual_seed(42)
            b = DetectionModel(architecture(experiment, weight=0.), verbose=False)
            for key in a.state_dict():
                torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key])
            b.args = SimpleNamespace(**DEFAULT_CFG_DICT)
            loss, items = b(batch([[.3, .3, .1, .1], [.5, .3, .1, .1]]))
            self.assertEqual(items[-1].item(), 0)
            loss.sum().backward()

    def test_rgb_only_inference_and_all_checkpoints_reload(self):
        for experiment in ("M1", "M2", "M3", "M4", "M5"):
            model = DetectionModel(architecture(experiment), verbose=False)
            model.args = dict(DEFAULT_CFG_DICT)
            model.eval()
            with patch("ultralytics.utils.candidate_loss.candidate_auxiliary", side_effect=AssertionError("GT inference")):
                with torch.no_grad():
                    output = model(torch.rand(1, 3, 64, 64))[0]
            router = next(m for m in model.modules() if isinstance(m, CandidateFusion))
            self.assertIsNone(router.aux_state)
            with tempfile.TemporaryDirectory(dir=ROOT / "runs/candidates") as directory:
                path = Path(directory) / "model.pt"
                torch.save({"model": model, "train_args": dict(DEFAULT_CFG_DICT)}, path)
                loaded = YOLO(str(path))
                with torch.no_grad():
                    self.assertEqual(loaded.model(torch.rand(1, 3, 64, 64))[0].shape, output.shape)

    def test_pretrained_head_remap(self):
        source = YOLO(str(ROOT / "model/yolo11n.pt")).model
        for experiment in ("M1", "M2", "M3", "M4", "M5"):
            model = DetectionModel(architecture(experiment), verbose=False)
            report = transfer_pretrained(model, ROOT / "model/yolo11n.pt")
            self.assertTrue(report["detect_branch_remap"])
            torch.testing.assert_close(source.state_dict()["model.23.cv2.0.0.conv.weight"].float(),
                                       model.state_dict()["model.24.cv2.1.0.conv.weight"])


if __name__ == "__main__":
    (ROOT / "runs/candidates").mkdir(parents=True, exist_ok=True)
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
