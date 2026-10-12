"""Geometry, candidate budgets, padding and class-aware duplicate handling."""
import unittest
import numpy as np
from crop_geometry import axis_starts, fixed_windows, restore_boxes, oracle_windows, coverage_matrix, union_area
from crop_infer import fuse_predictions, rect_shapes, boundary_flags
from crop_evaluate import fixed_conf_quality


class CropChecks(unittest.TestCase):
    def test_non_square_complete_coverage(self):
        rois=fixed_windows(1501,731)
        self.assertEqual(axis_starts(1501),[0,512,861])
        self.assertEqual(union_area(rois),1501*731)
        self.assertTrue(all(0<=x<xe<=1501 and 0<=y<ye<=731 for x,y,xe,ye in rois))

    def test_small_image_no_outside(self):
        self.assertEqual(fixed_windows(301,211),[(0,0,301,211)])

    def test_padding_scale_offset_inverse(self):
        original=np.array([[10,20,110,70]],dtype=np.float32)
        padded=original*2+[0,120,0,120]
        mapped=restore_boxes(padded,2,(0,120),(512,91),(731,1501))
        np.testing.assert_allclose(mapped,original+[512,91,512,91])

    def test_coordinate_clipping_empty(self):
        self.assertEqual(restore_boxes([],1,(0,0),(0,0),(80,90)).shape,(0,4))
        np.testing.assert_allclose(restore_boxes([[-5,-2,100,100]],1,(0,0),(0,0),(80,90)),[[0,0,90,80]])

    def test_class_aware_nms_and_final_cap(self):
        p=np.array([[0,0,10,10,.9,0],[0,0,10,10,.8,0],[0,0,10,10,.7,1],
                    [20,20,30,30,.6,0]],dtype=np.float32)
        out,s=fuse_predictions(p,.7,2)
        self.assertEqual(out[:,5].tolist(),[0,1])
        self.assertEqual(s['final_cap_removed'],1)
        self.assertEqual(fuse_predictions([], .7)[0].shape,(0,6))

    def test_oracle_nested_unique_and_margin(self):
        boxes=np.array([[10,10,14,14],[1000,1000,1004,1004],[1800,10,1804,14]])
        r,s=oracle_windows(2000,1400,boxes,4)
        self.assertLessEqual(len(r),4)
        self.assertEqual(len(r),len(set(r)))
        self.assertTrue(coverage_matrix(r,boxes,2000,1400).any(0).all())
        for k in [1,2]:
            self.assertEqual(oracle_windows(2000,1400,boxes,k)[0],r[:k])
        self.assertEqual(oracle_windows(100,100,[],4)[0],[])
        self.assertFalse(coverage_matrix([(50,0,690,640)],[[49,20,55,26]],1000,700)[0,0])

    def test_oracle_image_edge_not_internal_margin(self):
        self.assertTrue(coverage_matrix([(0,0,100,90)],[[0,0,4,4]],100,90).all())
        self.assertEqual(union_area([(0,0,10,10),(5,0,15,10)]),150)

    def test_val_shapes_have_padding_and_stride(self):
        metas=[dict(id=1,width=1000,height=500),dict(id=2,width=1000,height=600)]
        self.assertEqual(rect_shapes(metas,960)[1],(608,992))

    def test_boundary_interface_excludes_original_edges(self):
        p=np.array([[0,10,5,20,.9,0],[100,10,110,20,.8,0],[700,10,710,20,.7,0]],dtype=np.float32)
        copy=p.copy()
        self.assertFalse(boundary_flags(p, (0,0,1000,500),(500,1000)).any())
        self.assertEqual(boundary_flags(p[1:], (100,0,740,500),(500,1000)).tolist(),[True,False])
        np.testing.assert_array_equal(p,copy)

    def test_oracle_prioritizes_small_not_large_count(self):
        boxes=np.array([[x,10,x+100,110] for x in range(0,300,30)]+[[1400,20,1404,24],[1450,20,1454,24]])
        rois,_=oracle_windows(2000,800,boxes,1)
        covered=coverage_matrix(rois,boxes,2000,800)[0]
        self.assertTrue(covered[-2:].all())
        self.assertFalse(covered[:-2].any())

    def test_budget_stops_with_no_new_small_coverage(self):
        boxes=[[100,100,104,104],[200,200,206,206]]
        rois,_=oracle_windows(640,640,boxes,4)
        self.assertEqual(rois,[(0,0,640,640)])

    def test_retrospective_fp_categories_do_not_edit_outputs(self):
        gt={'images':[{'id':1}], 'annotations':[{'image_id':1,'bbox':[0,0,10,10],'category_id':1}]}
        preds=[dict(image_id=1,bbox=b,category_id=c,score=s) for b,c,s in [
            ([0,0,10,10],1,.9),([0,0,10,10],1,.8),([0,0,10,10],2,.7),
            ([5,0,10,10],1,.6),([30,30,10,10],1,.5)]]
        import copy
        before=copy.deepcopy(preds)
        q=fixed_conf_quality(gt,preds)
        self.assertEqual([q[k] for k in ('TP','FP','duplicates_proxy','wrong_class_proxy','localization_proxy','background_proxy')],
                         [1,4,1,1,1,1])
        self.assertEqual(preds,before)


if __name__=='__main__':
    unittest.main()
