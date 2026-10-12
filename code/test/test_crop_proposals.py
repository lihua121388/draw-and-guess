import unittest
import numpy as np
from crop_proposals import weighted_roi_windows,predict_roi_windows


class ProposalChecks(unittest.TestCase):
    def test_confidence_mass_not_just_count(self):
        boxes=np.array([[10+i*10,10,14+i*10,14] for i in range(3)]+[[1400,10,1404,14]])
        rois,info=weighted_roi_windows(2000,800,boxes,[.1,.1,.1,.9],1)
        self.assertGreater(rois[0][0],700)
        self.assertAlmostEqual(info['new_confidence_mass'][0],.9)

    def test_nested_budgets_no_repeat(self):
        b=[[10,10,14,14],[1000,1000,1004,1004],[1800,10,1804,14]]
        two,_=weighted_roi_windows(2000,1400,b,[.5,.6,.7],2)
        one,_=weighted_roi_windows(2000,1400,b,[.5,.6,.7],1)
        self.assertEqual(one,two[:1]);self.assertEqual(len(two),len(set(two)))

    def test_empty_and_low_conf_do_not_force_window(self):
        self.assertEqual(predict_roi_windows(1000,700,[],2)[0],[])
        self.assertEqual(predict_roi_windows(1000,700,[[0,0,5,5,.05,0]],2)[0],[])

    def test_proposal_dedup_does_not_change_detection_inputs(self):
        p=np.array([[10,10,14,14,.8,0],[10,10,14,14,.6,1],[100,100,200,200,.9,0]],dtype=np.float32)
        before=p.copy();r,info=predict_roi_windows(1000,700,p,1)
        self.assertEqual(len(r),1);self.assertEqual(info['eligible_before_dedup'],2)
        self.assertEqual(info['eligible_after_dedup'],1);np.testing.assert_array_equal(p,before)

    def test_positive_score_validation(self):
        with self.assertRaises(ValueError):weighted_roi_windows(1000,700,[[0,0,4,4]],[-1],1)


if __name__=='__main__':unittest.main()
