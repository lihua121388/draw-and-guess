"""MRU published hybrid fusion control on FROZEN S2 candidates, not full MRU-YOLO.

Independent implementation of the documented numerical rules, with attribution.
Original source: JiajunChen223/MRU-YOLO, dla_yolo/mru/validator.py and
configs/reproduction/visdrone.yaml (AGPL-3.0-only). ROI selector is NOT imported.
"""
import numpy as np
from train_test import ROOT
import torch
from source_fusion import split_sources
from crop_infer import fuse_predictions
from evaluate_nscr import box_iou_numpy
from crop_evaluate import evaluate_saved,fixed_conf_quality
from run_crop_study import load,save,context,to_coco
from run_source_study import DEFAULT,records
import time


def mru_fuse(g,l,global_gain):
    l=l.copy();size=np.sqrt((l[:,2:4]-l[:,:2]).prod(1));l=l[size<=128]
    l[:,4]*=.9
    gs=np.sqrt((g[:,2:4]-g[:,:2]).prod(1))*global_gain
    stable=g[(g[:,4]>=.25)&(gs>=96)]
    keep=np.ones(len(l),bool)
    for start in range(0,len(l),1024):
        p=l[start:start+1024]
        keep[start:start+len(p)]&=~((box_iou_numpy(p[:,:4],stable[:,:4])>=.5)&(p[:,None,5]==stable[None,:,5])).any(1)
    return fuse_predictions(np.concatenate((g,l[keep])),.7,500)[0]


def main():
    torch.set_num_threads(4);a,gt,images=context(DEFAULT);meta={m['id']:m for m in images}
    d=DEFAULT/'B_MRU';d.mkdir(exist_ok=False)
    from nscr_experiments import sha256
    save(d/'config.json',dict(local_size_gate_original=128,score_scale=.9,global_conf=.25,
        global_size_threshold_val_space=96,protection_iou=.5,final_iou=.7,maxdet=500,
        frozen_rois_sha256=sha256(DEFAULT/'frozen_s2_rois.json'),GT_input=False,
        source='https://github.com/JiajunChen223/MRU-YOLO/blob/main/dla_yolo/mru/validator.py',
        note='Published HYBRID fusion transplanted to S2 raw multi-label pools, native640 ROIs and max500. NOT full MRU reproduction; same detector/ROI/evaluation as B0.'))
    predictions=[];times=[]
    for r in records(__import__('pathlib').Path(a['source_study'])):
        g,l,rid=split_sources(r);t=time.perf_counter();p=mru_fuse(g,l,r['global_stage_stats']['gain']);times.append((time.perf_counter()-t)*1000)
        predictions.extend(to_coco(p,r['image_id']))
    save(d/'coco_predictions.json',predictions);save(d/'evaluated_image_ids.json',list(meta))
    save(d/'timing.json',dict(fusion_ms=dict(mean=float(np.mean(times)),p50=float(np.median(times)),p95=float(np.percentile(times,95))),
        measured_e2e_ms=None,cost_scope='Offline CPU fusion only; no model executed',
        mean_rois=load(__import__('pathlib').Path(a['source_study'])/'S2/timing.json')['mean_rois']))
    evaluate_saved(d,gt,__import__('pathlib').Path(a['data_root'])/'val_coco.json',a['original_annotations'])
    save(d/'fixed_conf_quality.json',fixed_conf_quality(gt,predictions))


if __name__=='__main__':main()
