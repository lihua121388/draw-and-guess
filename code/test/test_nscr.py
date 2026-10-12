"""Focused checks for spacing math, masks, gradient flow and checkpoint compatibility."""

import os
import json
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
from ultralytics.nn.modules.nscr import NSCRFusion
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.nscr_loss import gaussian_density_targets, normalized_spacing, spacing_auxiliary

from nscr_experiments import EXPERIMENTS, architecture, transfer_pretrained
from prepare_data import clip_box, inside_ignore, prepare_aitod, write_splits

torch.set_num_threads(2)


def batch(boxes, indices=None, classes=None):
    return {"img": torch.zeros(2, 3, 64, 64), "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "batch_idx": torch.tensor(indices if indices is not None else [0] * len(boxes)),
            "cls": torch.tensor(classes if classes is not None else [0] * len(boxes)).reshape(-1, 1)}


class SpacingTests(unittest.TestCase):
    def test_normalized_minimum_and_scale_invariance(self):
        boxes = torch.tensor([[0., 0., 2., 2.], [3., 0., 2., 2.], [6., 0., 20., 20.]])
        q = normalized_spacing(boxes, chunk_size=1)
        self.assertAlmostEqual(q[0].item(), 6 / 11, places=5)
        torch.testing.assert_close(q, normalized_spacing(boxes * 3))

    def test_dense_sparse_gradient_direction(self):
        dense = batch([[.3, .3, .15, .15], [.35, .3, .15, .15]])
        sparse = batch([[.1, .1, .1, .1], [.9, .9, .1, .1]])
        cfg = {"nc": 10, "small_size": 32, "supervision": "spacing", "tau": 2., "temperature": .5}
        for data, sign in ((dense, 1), (sparse, -1)):
            logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
            loss = spacing_auxiliary(logits, data, cfg)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(sign * logits.grad.sum().item(), 0)

    def test_single_empty_invalid_and_large_masks(self):
        cfg = {"nc": 10, "small_size": 4, "supervision": "spacing"}
        cases = [batch([]), batch([[.5, .5, .02, .02]]),
                 batch([[.1, .1, .2, .2], [.9, .9, .2, .2]]),
                 batch([[.3, .3, .02, .02], [.4, .4, .02, .02]], classes=[0, -1]),
                 batch([[.3, .3, .02, .02], [float("nan"), .4, .02, .02]])]
        for data in cases:
            logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
            loss = spacing_auxiliary(logits, data, cfg)
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_images_not_mixed_and_ignore_mask(self):
        cfg = {"nc": 10, "supervision": "spacing"}
        data = batch([[.3, .3, .1, .1], [.4, .4, .1, .1]], indices=[0, 1])
        logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
        self.assertEqual(spacing_auxiliary(logits, data, cfg).item(), 0)
        data["batch_idx"][:] = 0
        data["spacing_valid"] = torch.tensor([True, False])
        self.assertEqual(spacing_auxiliary(logits, data, cfg).item(), 0)

    def test_shuffle_breaks_correspondence(self):
        data = batch([[.1, .1, .1, .1], [.12, .1, .1, .1], [.9, .9, .1, .1]])
        cfg = {"nc": 10, "supervision": "spacing"}
        logits = torch.linspace(-3, 3, 256).reshape(1, 1, 16, 16).repeat(2, 1, 1, 1).requires_grad_()
        regular = spacing_auxiliary(logits, data, cfg)
        torch.manual_seed(42)
        shuffled = spacing_auxiliary(logits, data, dict(cfg, supervision="shuffle"))
        self.assertGreater(abs((regular - shuffled).item()), 0.01)

    def test_bad_temperature_rejected(self):
        with self.assertRaises(ValueError):
            spacing_auxiliary(torch.zeros(2, 1, 16, 16), batch([]), {"nc": 10, "temperature": 0})

    def test_density_supervision_sparse_prefers_wide(self):
        dense = torch.tensor([[.3, .3], [.31, .3], [.32, .3]])
        sparse = torch.tensor([[.1, .1], [.9, .9]])
        d = gaussian_density_targets(dense, dense, (160, 160))
        s = gaussian_density_targets(sparse, sparse, (160, 160))
        self.assertGreater(s.min().item(), .5)
        self.assertLess(d.max().item(), s.min().item())


class IntegrationTests(unittest.TestCase):
    def test_all_ablations_backward(self):
        data = batch([[.3, .3, .15, .15], [.35, .3, .15, .15]])
        data["img"] = torch.rand_like(data["img"])
        for experiment in EXPERIMENTS:
            with self.subTest(experiment=experiment):
                model = DetectionModel(architecture(experiment), verbose=False)
                model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
                model.train()
                loss, items = model(data)
                self.assertTrue(torch.isfinite(loss).all())
                self.assertEqual(len(items), 3 if experiment == "E0" else 4)
                loss.sum().backward()
                self.assertEqual(model.stride.tolist(), [8., 16., 32.] if experiment == "E0" else [4., 8., 16., 32.])
                if experiment != "E0":
                    router = next(m for m in model.modules() if isinstance(m, NSCRFusion))
                    self.assertIsNone(router.gate_logits)
                    self.assertTrue(torch.isfinite(router.fusion.conv.weight.grad).all())
                if experiment in ("E4", "E5", "E6"):
                    self.assertGreater(items[-1].item(), 0)
                    self.assertGreater(router.gate[-1].weight.grad.abs().sum().item(), 0)
                elif experiment != "E0":
                    self.assertEqual(items[-1].item(), 0)

    def test_inference_has_no_gt_and_checkpoint_reloads(self):
        model = DetectionModel(architecture("E4"), verbose=False).eval()
        model.args = dict(DEFAULT_CFG_DICT)
        with patch("ultralytics.utils.nscr_loss.normalized_spacing", side_effect=AssertionError("GT during inference")):
            with torch.no_grad():
                prediction = model(torch.zeros(1, 3, 64, 64))[0]
        self.assertEqual(prediction.shape[1], 14)
        router = next(m for m in model.modules() if isinstance(m, NSCRFusion))
        self.assertIsNone(router.gate_logits)
        with tempfile.TemporaryDirectory(dir=ROOT / "runs/nscr") as directory:
            path = Path(directory) / "model.pt"
            torch.save({"model": model, "train_args": dict(DEFAULT_CFG_DICT)}, path)
            loaded = YOLO(str(path))
            with torch.no_grad():
                self.assertEqual(loaded.model(torch.zeros(1, 3, 64, 64))[0].shape, prediction.shape)

    def test_backbone_and_pretrained_head_mapping(self):
        from ultralytics.utils import YAML

        source_yaml = YAML.load(ROOT / "ultralytics/cfg/models/11/yolo11.yaml")
        self.assertEqual(architecture("E4")["backbone"], source_yaml["backbone"])
        self.assertEqual(architecture("E4")["head"][:-2], source_yaml["head"][:-1])
        source = YOLO(str(ROOT / "model/yolo11n.pt")).model
        model = DetectionModel(architecture("E4"), verbose=False)
        transfer_pretrained(model, ROOT / "model/yolo11n.pt")
        for old, new in (("model.0.conv.weight", "model.0.conv.weight"),
                         ("model.23.cv2.0.0.conv.weight", "model.24.cv2.1.0.conv.weight")):
            torch.testing.assert_close(source.state_dict()[old].float(), model.state_dict()[new])

    def test_clip_and_ignored_regions(self):
        self.assertEqual(clip_box([-2, -2, 5, 5], 10, 10), [0, 0, 3., 3.])
        self.assertIsNone(clip_box([1, 1, -1, 2], 10, 10))
        self.assertTrue(inside_ignore([1, 1, 2, 2], [[0, 0, 4, 4]]))

    def test_single_branch_parameter_budget(self):
        single = NSCRFusion([64, 64], 64, "single")
        gated = NSCRFusion([64, 64], 64, "gated")
        difference = abs(sum(p.numel() for p in single.parameters()) - sum(p.numel() for p in gated.parameters()))
        self.assertLessEqual(difference, 64)

    def test_subsets_preserve_full_manifests(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs/nscr") as directory:
            output = Path(directory)
            images = [output / f"{i}.jpg" for i in range(20)]
            full = write_splits(output, ["target"], {"train": images, "val": images[:10]})
            original = (output / "train.txt").read_bytes()
            subset = write_splits(output, ["target"], {"train": images, "val": images[:10]}, 3, 2)
            self.assertEqual((output / "train.txt").read_bytes(), original)
            self.assertNotEqual(full, subset)
            self.assertEqual(len((output / "train-subset-3-2-s42.txt").read_text().splitlines()), 3)

    def test_aitod_preserves_category_ids_and_splits(self):
        from PIL import Image

        with tempfile.TemporaryDirectory(dir=ROOT / "runs/nscr") as directory:
            base = Path(directory)
            paths = {}
            for split in ("train", "val", "test"):
                image = base / f"{split}.jpg"
                Image.new("RGB", (32, 32)).save(image)
                payload = {"images": [{"id": 1, "file_name": image.name, "width": 32, "height": 32}],
                    "categories": [{"id": 7, "name": "vehicle"}, {"id": 3, "name": "person"}],
                    "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [1, 1, 4, 4], "area": 16, "iscrowd": 0}]}
                path = base / f"{split}.json"
                path.write_text(json.dumps(payload))
                paths[split] = path
            result = prepare_aitod(base, paths, base / "prepared")
            self.assertTrue(result.is_file())
            self.assertTrue((base / "prepared/train/labels/train.txt").read_text().startswith("1 "))
            coco = json.loads((base / "prepared/test_coco.json").read_text())
            self.assertEqual(coco["annotations"][0]["category_id"], 7)
            self.assertEqual(len((base / "prepared/test.txt").read_text().splitlines()), 1)

    def test_group_matching_rejects_duplicates_and_wrong_classes(self):
        import numpy as np
        from evaluate_nscr import match_instances

        boxes = np.array([[0, 0, 4, 4], [10, 10, 14, 14]], dtype=float)
        classes = np.array([0, 1])
        predictions = np.array([[0, 0, 4, 4, .9, 0], [0, 0, 4, 4, .8, 0], [10, 10, 14, 14, .9, 0]])
        matched, _ = match_instances(boxes, classes, predictions)
        self.assertEqual(matched.tolist(), [1., 0.])

    def test_coco_small_metrics_perfect_and_empty_predictions(self):
        from evaluate_nscr import coco_metrics

        with tempfile.TemporaryDirectory(dir=ROOT / "runs/nscr") as directory:
            path = Path(directory) / "gt.json"
            path.write_text(json.dumps({"info": {}, "images": [{"id": 1, "width": 64, "height": 64, "file_name": "a.jpg"}],
                "categories": [{"id": 3, "name": "tiny"}], "annotations": [{"id": 1, "image_id": 1,
                "category_id": 3, "bbox": [5, 5, 4, 4], "area": 16, "iscrowd": 0}]}))
            perfect = coco_metrics(path, [{"image_id": 1, "category_id": 3, "bbox": [5, 5, 4, 4], "score": .9}], [1], 500)
            empty = coco_metrics(path, [], [1], 500)
            self.assertAlmostEqual(perfect["AP_small"], 1.)
            self.assertAlmostEqual(perfect["AR_small"], 1.)
            self.assertEqual(empty["AP_small"], 0.)
            self.assertEqual(empty["AR_small"], 0.)


if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
