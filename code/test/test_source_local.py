"""Invariants for source recovery, GT-free filtering and boundary-label safety."""
import unittest
import numpy as np
from source_fusion import source_fuse,edge_distance,split_sources
from prepare_local_rois import crop_labels,group_split
from run_mru_fusion_control import mru_fuse


class Checks(unittest.TestCase):
    def test_source_recovery(self):
        g=[[0,0,10,10,.8,0]];l=[[20,20,30,30,.7,1]]
        r=dict(candidates=g+l,global_predictions=g,rois=[[0,0,640,640]],stages=[{},dict(mapped_candidates=1)])
        gg,ll,rid=split_sources(r);self.assertEqual(rid.tolist(),[0]);self.assertTrue(np.allclose(ll,l))
        r['stages'][1]['mapped_candidates']=2
        with self.assertRaises(ValueError):split_sources(r)

    def test_calibration_changes_competition_only(self):
        g=np.array([[20,20,40,40,.8,0]],np.float32);l=np.array([[20,20,40,40,.9,0]],np.float32);original=l.copy()
        _,chosen,_=source_fuse(g,l,np.array([0]),[[0,0,640,640]],1000,1000,'B0');self.assertEqual(chosen.tolist(),[1])
        _,chosen,_=source_fuse(g,l,np.array([0]),[[0,0,640,640]],1000,1000,'B1');self.assertEqual(chosen.tolist(),[0])
        self.assertTrue(np.array_equal(l,original))

    def test_boundary_image_edge_exemption(self):
        p=np.array([[0,50,20,70,.8,0],[620,50,640,70,.8,0]],np.float32)
        d=edge_distance(p,np.array([0,0]),[[0,0,640,640]],1000,1000)
        self.assertEqual(d.tolist(),[570,0])

    def test_protection_same_class_large_only(self):
        g=np.array([[20,20,80,80,.6,0]],np.float32)
        l=np.array([[20,20,80,80,.9,0],[20,20,80,80,.8,1]],np.float32)
        p,chosen,s=source_fuse(g,l,np.array([0,0]),[[0,0,640,640]],1000,1000,'B3')
        self.assertEqual(s['protected_local_removed'],1);self.assertEqual(set(chosen),{0,2})

    def test_empty_and_maxdet(self):
        empty=np.empty((0,6),np.float32)
        p,_,_=source_fuse(empty,empty,np.array([],int),[],1000,1000,'B4');self.assertEqual(len(p),0)

    def test_mru_size_in_val_coordinates_and_class_protection(self):
        g=np.array([[0,0,200,200,.3,0]],np.float32)
        l=np.array([[0,0,200,200,.9,0],[0,0,120,120,.8,0],[0,0,120,120,.7,1]],np.float32)
        # Large local >128 dropped; second local IoU=.36 not protected.
        p=mru_fuse(g,l,.5);self.assertEqual(len(p),3)
        l=np.array([[0,0,128,128,.9,0]],np.float32)
        g=np.array([[0,0,128,128,.3,0]],np.float32)
        # 128 native*.5=64 <96, no protection; local retains higher score.
        p=mru_fuse(g,l,.5);self.assertAlmostEqual(float(p[0,4]),.81,places=6)
        p=mru_fuse(g,l,1.);self.assertAlmostEqual(float(p[0,4]),.3,places=6)

    def test_ignore_mask_removes_dominated_positive(self):
        image=np.full((640,640,3),180,np.uint8)
        crop,labels,s=crop_labels(image,[0,0,640,640],[dict(class_id=0,bbox=[10,10,20,20])],[[10,10,15,20]])
        self.assertEqual(labels,[]);self.assertEqual(s['discarded_mask_overlap'],1);self.assertTrue((crop[10:30,10:30]==114).all())

    def test_visible_targets_and_unknown_neutral_fill(self):
        image=np.full((800,800,3),200,np.uint8)
        positives=[dict(class_id=0,bbox=[95,150,20,20]),dict(class_id=1,bbox=[80,200,25,25]),dict(class_id=2,bbox=[150,150,10,10])]
        crop,labels,s=crop_labels(image,[100,100,740,740],positives,[[300,300,20,20]])
        self.assertEqual([c for c,_ in labels],[0,2]);self.assertTrue((crop[100:125,:5]==114).all())
        self.assertTrue((crop[200:220,200:220]==114).all());self.assertEqual(s['retained_partial'],1)

    def test_empty_background_padding_preserves_pixels(self):
        image=np.full((500,500,3),197,np.uint8)
        crop,labels,_=crop_labels(image,[0,0,500,500],[],[])
        self.assertEqual(labels,[]);self.assertTrue((crop[70:570,70:570]==197).all());self.assertTrue((crop[:70]==114).all())

    def test_sequence_holdout(self):
        m=[dict(file_name=f'{i:07}_001_d_{j}.jpg') for i in range(20) for j in range(3)]
        held,split=group_split(m);self.assertEqual(len(held),2);self.assertEqual(held,group_split(m)[0])


if __name__=='__main__':unittest.main()
