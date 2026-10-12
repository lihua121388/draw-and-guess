"""Hand-computed cases covering ignored-region union, match priority and AP interpolation."""
import unittest
import numpy as np
from visdrone_protocol import ignore_integral, drop_in_ignore, match, voc_ap


class ProtocolTests(unittest.TestCase):
    def test_region_union_and_matlab_coordinates(self):
        gt = np.array([[2, 2, 2, 4, 0, 0, -1, -1], [4, 2, 2, 4, 0, 0, -1, -1]], dtype=float)
        integral = ignore_integral(gt, 12, 12)
        # The union covers >50%; no single rectangle alone needs to cover >50%.
        boxes = np.array([[2, 2, 5, 4, .9], [8, 8, 2, 2, .8]])
        np.testing.assert_array_equal(drop_in_ignore(boxes, integral), boxes[1:])
        self.assertIsNone(ignore_integral(np.array([[2, 2, 2, 2, 1, 11, 0, 0]]), 12, 12))

    def test_positive_first_and_ignored_multiple_match(self):
        gt = np.array([[0, 0, 5, 5, 0], [0, 0, 10, 10, 1]], dtype=float)
        dt = np.array([[0, 0, 5, 5, .8], [5, 5, 2, 2, .6], [7, 7, 2, 2, .9], [20, 20, 1, 1, .5]])
        order, flags = match(gt, dt, .5)
        np.testing.assert_array_equal(order, [2, 0, 1, 3])
        np.testing.assert_array_equal(flags, [-1, 1, -1, 0])

    def test_duplicates_and_last_overlap_tie(self):
        gt = np.array([[0, 0, 5, 5, 0]], dtype=float)
        dt = np.array([[0, 0, 5, 5, .9], [0, 0, 5, 5, .8]])
        np.testing.assert_array_equal(match(gt, dt, .5)[1], [1, 0])
        self.assertAlmostEqual(voc_ap(np.array([.5, 1.]), np.array([1., .5])), .75)
        self.assertEqual(voc_ap(np.array([]), np.array([])), 0.)

    def test_empty_ground_truth(self):
        dt = np.array([[0, 0, 2, 2, .5]])
        np.testing.assert_array_equal(match(np.empty((0, 5)), dt, .5)[1], [0])


if __name__ == "__main__":
    unittest.main()
