"""Later fresh complete-pipeline GPU timing with frozen ROIs. Not run in offline B study."""
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from train_test import ROOT
import torch
from run_source_study import DEFAULT
from run_crop_study import load,save,context,to_coco
from crop_infer import CropDetector,rect_shapes,fuse_predictions
from crop_proposals import predict_roi_windows
from source_fusion import source_fuse,METHODS
from nscr_experiments import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,default=DEFAULT);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--local-weights',type=Path);p.add_argument('--fusion',choices=list(METHODS),default='B0');p.add_argument('--device',default='0');args=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(2);args.output.mkdir(parents=True,exist_ok=False)
    a,gt,images=context(args.study);expected={r['image_id']:r['rois'] for r in load(args.study/'frozen_s2_rois.json')['images']}
    global_model=CropDetector(a['checkpoint'],args.device)
    local=CropDetector(args.local_weights,args.device) if args.local_weights else global_model
    if [type(m).__name__ for m in global_model.model.model]!=[type(m).__name__ for m in local.model.model]:raise RuntimeError('Local structure differs')
    shapes=rect_shapes(gt['images'],640);global_model.warm_shapes(list(shapes.values()));local.warm_shapes([(640,640)])
    rows=[];predictions=[]
    for m in images:
        global_model.sync();start=time.perf_counter();image=cv2.imread(m['path'])
        g,s=global_model.infer(image,imgsz=640,global_shape=shapes[m['id']]);g,_=fuse_predictions(g,.7,500)
        select_start=time.perf_counter();rois,_=predict_roi_windows(m['width'],m['height'],g,2);selection_ms=(time.perf_counter()-select_start)*1000
        if [list(r) for r in rois]!=expected[m['id']]:raise RuntimeError('Global model generated different ROI coordinates')
        pieces=[];stats=[s]
        for roi in expected[m['id']]:
            pp,ss=local.infer(image,roi);pieces.append(pp);stats.append(ss)
        l=np.concatenate(pieces) if pieces else np.empty((0,6),np.float32);rid=np.repeat(np.arange(len(pieces)),[len(pp) for pp in pieces])
        fstart=time.perf_counter();pred,chosen,fs=source_fuse(g,l,rid,rois,m['width'],m['height'],args.fusion);global_model.sync();end=time.perf_counter()
        rows.append(dict(image_id=m['id'],e2e_ms=(end-start)*1000,fusion_ms=(end-fstart)*1000,selection_ms=selection_ms,rois=len(rois),
            GFLOPs_actual_shape_estimate=sum(11.4582016*t['input_shape'][0]*t['input_shape'][1]/640**2 for t in stats)))
        predictions.extend(to_coco(pred,m['id']))
    save(args.output/'coco_predictions.json',predictions);save(args.output/'timing_per_image.json',rows)
    save(args.output/'timing.json',dict(images=548,frozen_rois_sha256=sha256(args.study/'frozen_s2_rois.json'),
        global_checkpoint_sha256=sha256(a['checkpoint']),local_checkpoint_sha256=sha256(args.local_weights or a['checkpoint']),
        fusion=args.fusion,mean_rois=float(np.mean([r['rois'] for r in rows])),
        mean_GFLOPs_actual_shape_estimate=float(np.mean([r['GFLOPs_actual_shape_estimate'] for r in rows])),
        e2e_ms=dict(mean=float(np.mean([r['e2e_ms'] for r in rows])),p50=float(np.median([r['e2e_ms'] for r in rows])),p95=float(np.percentile([r['e2e_ms'] for r in rows],95))),
        FPS=1000/np.mean([r['e2e_ms'] for r in rows]),
        scope='Fresh synchronized GPU whole640 + unchanged selection verification + fixed native ROI640 + source fusion; disk decode included. Warmups/loading/scoring/serialization excluded.'))


if __name__=='__main__':main()
