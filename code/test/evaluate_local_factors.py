"""Frozen S2 ROI A/B interface. E2/E3 wait for an authorized trained local model.

Cached mode does no inference. --local-weights executes LOCAL-ONLY inference;
the global predictions and frozen ROI coordinates are read from historical S2.
Fresh full GPU end-to-end cost is deliberately not fabricated from cached mode.
"""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from train_test import ROOT
import torch
from run_source_study import DEFAULT,SOURCE,records
from run_crop_study import load,save,context,to_coco
from source_fusion import METHODS,PRESET,split_sources,source_fuse
from crop_infer import CropDetector
from crop_evaluate import evaluate_saved,fixed_conf_quality
from nscr_experiments import sha256


def local_inference(study,weights,destination,device):
    a,gt,images=context(study);frozen=load(study/'frozen_s2_rois.json');meta={m['id']:m for m in images}
    if destination.exists():raise FileExistsError('Local cache exists')
    detector=CropDetector(weights,device)
    # Matching module types and four detection strides, not merely nc.
    from ultralytics import YOLO
    global_model=YOLO(a['checkpoint']).model
    if [type(m).__name__ for m in detector.model.model]!=[type(m).__name__ for m in global_model.model] or detector.model.stride.tolist()!=global_model.stride.tolist():raise RuntimeError('Local architecture differs from M0')
    with destination.open('w',encoding='utf-8') as f:
        f.write(json.dumps(dict(header=True,frozen_rois_sha256=sha256(study/'frozen_s2_rois.json'),checkpoint_sha256=sha256(weights),checkpoint=str(weights)))+'\n')
        for row in frozen['images']:
            m=meta[row['image_id']];image=cv2.imread(m['path']);pieces=[];stages=[]
            start=time.perf_counter()
            for roi in row['rois']:
                pred,s=detector.infer(image,roi);pieces.append(pred);stages.append(s)
            f.write(json.dumps(dict(image_id=row['image_id'],rois=row['rois'],local_predictions=[p.tolist() for p in pieces],
                stages=stages,local_pipeline_ms=(time.perf_counter()-start)*1000))+'\n')
    del detector;torch.cuda.empty_cache()


def evaluate(study,destination,fusion,local_cache=None):
    destination.mkdir(parents=True,exist_ok=False)
    a,gt,images=context(study);source=Path(a['source_study']);meta={m['id']:m for m in images}
    frozen=load(study/'frozen_s2_rois.json');frozen_hash=sha256(study/'frozen_s2_rois.json')
    expected={r['image_id']:r['rois'] for r in frozen['images']};cache={};header=None
    if local_cache:
        with local_cache.open(encoding='utf-8') as f:
            header=json.loads(next(f))
            if header.get('frozen_rois_sha256')!=frozen_hash:raise RuntimeError('Cache ROI hash differs')
            for line in f:
                row=json.loads(line)
                if row['image_id'] in cache:raise RuntimeError('Duplicate cache image')
                cache[row['image_id']]=row
        if set(cache)!=set(expected):raise RuntimeError('Cache must contain all548 images')
    names=('E2','E3') if local_cache else ('E0','E1');predictions={n:[] for n in names};timing={n:[] for n in names}
    for r in records(source):
        g,l,rid=split_sources(r);m=meta[r['image_id']]
        if r['rois']!=expected[r['image_id']]:raise RuntimeError('S2 ROI changed')
        if local_cache:
            row=cache[r['image_id']]
            if row['rois']!=r['rois'] or len(row['local_predictions'])!=len(r['rois']):raise RuntimeError('Local ROI order changed')
            pieces=[np.asarray(p,np.float32).reshape(-1,6) for p in row['local_predictions']]
            l=np.concatenate(pieces) if pieces else np.empty((0,6),np.float32);rid=np.repeat(np.arange(len(pieces)),[len(p) for p in pieces])
            if not np.isfinite(l).all() or (l[:,4]<0).any() or (l[:,4]>1).any() or not np.isin(l[:,5],range(10)).all():raise RuntimeError('Invalid local candidates')
            for i,p in enumerate(pieces):
                x,y,xe,ye=r['rois'][i]
                if len(p) and ((p[:,:2]<[x-.001,y-.001]).any() or (p[:,2:4]>[xe+.001,ye+.001]).any() or (p[:,2:4]<=p[:,:2]).any()):raise RuntimeError('Local cache coordinates outside frozen ROI')
        for name,method in zip(names,('B0',fusion)):
            t=time.perf_counter();p,chosen,s=source_fuse(g,l,rid,r['rois'],m['width'],m['height'],method)
            timing[name].append(dict(image_id=m['id'],fusion_ms=(time.perf_counter()-t)*1000,**s));predictions[name].extend(to_coco(p,m['id']))
    for name in names:
        d=destination/name;d.mkdir();save(d/'coco_predictions.json',predictions[name]);save(d/'fusion_per_image.json',timing[name])
        save(d/'evaluated_image_ids.json',list(meta));save(d/'config.json',dict(A=bool(local_cache),B=name in ('E1','E3'),
            fusion_family='mature source-aware preset',fusion='B0' if name in ('E0','E2') else fusion,parameters=PRESET,
            global_checkpoint_sha256=a['immutable']['checkpoint_sha256'],local_checkpoint=header,
            frozen_rois_sha256=frozen_hash,mean_rois=sum(len(r['rois']) for r in frozen['images'])/548,
            measured_e2e_ms=None,latency_scope='Cached global; local inference cost if newly executed saved separately. Fresh E2E requires a dedicated later complete pipeline benchmark.',
            GFLOPs_estimate=load(source/'S2/timing.json')['mean_GFLOPs_actual_shape_estimate']))
        evaluate_saved(d,gt,Path(a['data_root'])/'val_coco.json',a['original_annotations'])
        save(d/'fixed_conf_quality.json',fixed_conf_quality(gt,predictions[name]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,default=DEFAULT);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fusion',choices=['B1','B2','B3','B4'],default='B4');p.add_argument('--local-cache',type=Path);p.add_argument('--local-weights',type=Path);p.add_argument('--device',default='0')
    args=p.parse_args();torch.set_num_threads(4)
    if args.local_weights:
        if args.local_cache:p.error('Choose local-weights OR existing local-cache')
        args.local_cache=args.output.parent/(args.output.name+'_local_cache.jsonl')
        local_inference(args.study,args.local_weights,args.local_cache,args.device)
    evaluate(args.study,args.output,args.fusion,args.local_cache)


if __name__=='__main__':main()
