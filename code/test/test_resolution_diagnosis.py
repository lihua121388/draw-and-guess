"""Validate error partitions with hand-computed detections, including duplicated GT and ignore boxes."""
import unittest
import numpy as np
from resolution_diagnosis import diagnose, spacing, ERRORS


class DiagnosisTests(unittest.TestCase):
    def test_disjoint_errors_and_one_to_one_competition(self):
        xs = [0, 30, 60, 90, 120, 150, 180, 180]
        gt = dict(images=[dict(id=1, width=220, height=40)],
                  categories=[dict(id=1, name="a"), dict(id=2, name="b")],
                  annotations=[dict(image_id=1, bbox=[x, 0, 10, 10], category_id=1, iscrowd=0) for x in xs])
        predictions = [dict(image_id=1, bbox=[x, 0, 10, 10], category_id=cid, score=score)
                      for x, cid, score in [(0, 1, .1), (30, 1, .02), (60, 2, .9), (95, 1, .9), (150, 1, .9), (180, 1, .9)]]
        result = diagnose(gt, predictions, 640)
        self.assertEqual(result["all"]["targets"], 8)
        self.assertEqual(result["all"]["matched"], 2)
        self.assertEqual(result["all"]["missed"], 6)
        self.assertEqual(result["all"]["errors"], {name: 1 for name in ERRORS})
        self.assertEqual(sum(v["targets"] for v in result["density_size"]["8-16"].values()), 8)

    def test_single_instance_and_no_prediction(self):
        gt = dict(images=[dict(id=1, width=100, height=100)], categories=[dict(id=1, name="a")],
                  annotations=[dict(image_id=1, bbox=[0, 0, 4, 4], category_id=1, iscrowd=0)])
        result = diagnose(gt, [], 640)
        self.assertEqual(result["density_size"]["<8"]["undefined"]["targets"], 1)
        self.assertEqual(result["all"]["errors"]["no_qualifying_output"], 1)
        self.assertTrue(np.isnan(spacing(np.array([[0, 0, 4, 4]]))).all())

    def test_ignore_prediction_and_empty_gt(self):
        gt = dict(images=[dict(id=1, width=100, height=100)], categories=[dict(id=1, name="a")],
                  annotations=[dict(image_id=1, bbox=[0, 0, 20, 20], category_id=1, iscrowd=1)])
        result = diagnose(gt, [dict(image_id=1, bbox=[2, 2, 4, 4], category_id=1, score=.9)], 960)
        self.assertEqual(result["ignored_prediction_count"], 1)
        self.assertEqual(result["all"]["targets"], 0)


if __name__ == "__main__":
    unittest.main()
