"""Observation opportunity must not be conflated with failed detection."""
import unittest
import numpy as np
from check_quality_prerequisites import observation_states


class ObservationChecks(unittest.TestCase):
    def test_three_states_and_internal_margin(self):
        p=np.array([[50,50,70,70,.5,0],[400,50,420,70,.5,0],[500,50,520,70,.5,0],[325,50,340,70,.5,0]],np.float32)
        q=np.array([[400,50,420,70,.5,0],[325,50,340,70,.5,0]],np.float32)
        states=observation_states([p,q],[[0,0,640,640],[320,0,960,640]],960,640)
        self.assertEqual(states[:4].tolist(),[0,2,1,0])

    def test_same_roi_copies_not_observations(self):
        p=np.array([[400,50,420,70,.5,0],[400,50,420,70,.8,0]],np.float32)
        self.assertEqual(observation_states([p],[[0,0,640,640]],960,640).tolist(),[0,0])

    def test_wrong_class_and_low_conf_not_support(self):
        p=np.array([[400,50,420,70,.5,0]],np.float32)
        q=np.array([[400,50,420,70,.5,1],[400,50,420,70,.05,0]],np.float32)
        self.assertEqual(observation_states([p,q],[[0,0,640,640],[320,0,960,640]],960,640)[0],1)

    def test_image_edge_exempt_and_empty(self):
        p=np.array([[0,0,10,10,.5,0]],np.float32)
        states=observation_states([p,p],[[0,0,640,640],[0,0,500,500]],960,640)
        self.assertEqual(states.tolist(),[2,2])
        self.assertEqual(len(observation_states([],[],960,640)),0)


if __name__=='__main__':unittest.main()
