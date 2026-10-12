import unittest
import numpy as np
import torch
from instance_quality import extract_features,FEATURES,feature_mask,quality_loss,rescore,QualityMLP


class QualityChecks(unittest.TestCase):
    def test_full_image_has_no_internal_boundary(self):
        p=np.array([[0,0,4,8,.5,0]],np.float32)
        x=extract_features(p,[p],[[0,0,640,640]],640,640)
        self.assertEqual(x[0,FEATURES.index('boundary_available')],0)
        self.assertEqual(x[0,FEATURES.index('relative_distance')],0)
        self.assertTrue(np.isfinite(x).all())

    def test_observation_availability_and_support_separate(self):
        p=np.array([[50,50,70,70,.5,0],[400,50,420,70,.5,0],[500,50,520,70,.5,0]],np.float32)
        q=np.array([[400,50,420,70,.5,0]],np.float32)
        x=extract_features(p,[p,q],[[0,0,640,640],[320,0,960,640]],960,640)
        self.assertEqual(x[:3,FEATURES.index('observation_available')].tolist(),[0,1,1])
        self.assertEqual(x[:3,FEATURES.index('supported')].tolist(),[0,1,0])
        self.assertEqual(x[:3,FEATURES.index('observed_no_support')].tolist(),[0,0,1])

    def test_masks_keep_capacity_and_gain_bound(self):
        self.assertEqual(feature_mask('C2').shape,feature_mask('C4').shape)
        logits=torch.zeros(2,2,requires_grad=True);targets=torch.tensor([[0.,0.],[1.,0.]])
        loss=quality_loss(logits,targets,torch.ones(2));loss.backward()
        self.assertEqual(float(logits.grad[0,1]),0.)
        with self.assertRaises(ValueError):quality_loss(logits,torch.tensor([[0.,1.],[1.,0.]]),torch.ones(2))

    def test_gain_floor_preserves_reliable_nonnew(self):
        scores=torch.tensor([.8]);logits=torch.tensor([[20.,-20.]])
        self.assertAlmostEqual(float(rescore(scores,logits,beta=.5,gain_enabled=True)),.4,places=5)
        self.assertAlmostEqual(float(rescore(scores,logits,beta=1.,gain_enabled=True)),.8,places=5)
        self.assertEqual(QualityMLP()(torch.zeros(1,len(FEATURES))).shape,(1,2))


if __name__=='__main__':unittest.main()
