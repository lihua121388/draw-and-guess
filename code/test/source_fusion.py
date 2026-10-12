"""Preset detection-level baselines. No GT, fitting, ROI selection or model changes."""
import numpy as np
from train_test import ROOT
import torch
import torchvision
from evaluate_nscr import box_iou_numpy

PRESET = dict(local_scale=.85, boundary_margin=8., boundary_floor=.25,
              stable_global_conf=.50, stable_global_size=32., protection_iou=.50,
              fusion_iou=.7, max_det=500, output_conf=.001)
METHODS = {'B0':(), 'B1':('calibration',), 'B2':('boundary',),
           'B3':('protection',), 'B4':('calibration','boundary','protection')}


def split_sources(record):
    pool=np.asarray(record['candidates'],np.float32).reshape(-1,6)
    g=np.asarray(record['global_predictions'],np.float32).reshape(-1,6)
    if not np.array_equal(pool[:len(g)],g):raise ValueError('Global pool prefix differs')
    sizes=[s['mapped_candidates'] for s in record['stages'][1:]]
    if len(pool)!=len(g)+sum(sizes) or len(sizes)!=len(record['rois']):raise ValueError('Source accounting mismatch')
    rid=np.repeat(np.arange(len(sizes)),sizes)
    return g,pool[len(g):],rid


def edge_distance(local,roi_ids,rois,width,height):
    """Distance to internal crop edges; image edges and padded areas are exempt."""
    distance=np.full(len(local),np.inf,np.float32)
    for i,(x,y,xe,ye) in enumerate(rois):
        ids=np.flatnonzero(roi_ids==i)
        for column,boundary,internal,sign in ((0,x,x>0,1),(1,y,y>0,1),(2,xe,xe<width,-1),(3,ye,ye<height,-1)):
            if internal:distance[ids]=np.minimum(distance[ids],sign*(local[ids,column]-boundary))
    return np.maximum(distance,0)


def overlap_features(global_pred,local):
    n=len(local)
    result={k:np.zeros(n,np.float32) for k in ('same_iou','any_iou','same_conf','any_conf','class_agrees','score_delta')}
    if not len(global_pred) or not n:return result
    for start in range(0,n,1024):
        p=local[start:start+1024];iou=box_iou_numpy(p[:,:4],global_pred[:,:4])
        same=p[:,None,5]==global_pred[None,:,5]
        masked=np.where(same,iou,0)
        a=iou.argmax(1);s=masked.argmax(1);rows=np.arange(len(p));sl=slice(start,start+len(p))
        result['same_iou'][sl]=masked[rows,s];result['any_iou'][sl]=iou[rows,a]
        result['same_conf'][sl]=np.where(masked[rows,s]>0,global_pred[s,4],0)
        result['any_conf'][sl]=np.where(iou[rows,a]>0,global_pred[a,4],0)
        result['class_agrees'][sl]=(p[:,5]==global_pred[a,5])&(iou[rows,a]>0)
        result['score_delta'][sl]=p[:,4]-result['same_conf'][sl]
    return result


def source_fuse(global_pred,local,roi_ids,rois,width,height,method='B0',preset=None):
    """Score scaling, soft edge control and stable LARGE global protection.

    Protection removes conflicting SAME-class local boxes only. It never promotes
    scores or guarantees a global box survives unrelated global NMS/top500.
    Returned indices refer to the ORIGINAL global+local candidate array.
    """
    cfg=PRESET if preset is None else preset
    enabled=METHODS[method];g=np.asarray(global_pred,np.float32).reshape(-1,6)
    l=np.asarray(local,np.float32).reshape(-1,6).copy();valid=np.ones(len(l),bool)
    if 'calibration' in enabled:l[:,4]*=cfg['local_scale']
    if 'boundary' in enabled:
        distance=edge_distance(l,roi_ids,rois,width,height)
        l[:,4]*=cfg['boundary_floor']+(1-cfg['boundary_floor'])*np.clip(distance/cfg['boundary_margin'],0,1)
    protected=0
    if 'protection' in enabled and len(g):
        size=np.sqrt(np.maximum(0,g[:,2:4]-g[:,:2]).prod(1))
        stable=g[(g[:,4]>=cfg['stable_global_conf'])&(size>=cfg['stable_global_size'])]
        for start in range(0,len(l),1024):
            p=l[start:start+1024]
            hit=((box_iou_numpy(p[:,:4],stable[:,:4])>cfg['protection_iou'])&(p[:,None,5]==stable[None,:,5])).any(1)
            valid[start:start+len(p)]&=~hit;protected+=int(hit.sum())
    valid&=l[:,4]>=cfg['output_conf']
    ids=np.r_[np.arange(len(g)),len(g)+np.flatnonzero(valid)]
    p=np.concatenate((g,l[valid]))
    if len(p):
        t=torch.from_numpy(np.ascontiguousarray(p))
        keep=torchvision.ops.batched_nms(t[:,:4],t[:,4],t[:,5],cfg['fusion_iou']).numpy()
        after=len(keep);keep=keep[:cfg['max_det']];out=p[keep];chosen=ids[keep]
    else:after=0;out=p;chosen=ids
    return out,chosen,dict(input=len(g)+len(l),protected_local_removed=protected,
        low_score_removed=int((~valid).sum())-protected,after_nms=after,final=len(out),
        cap_removed=max(0,after-cfg['max_det']),retained_global=int((chosen<len(g)).sum()),
        retained_local=int((chosen>=len(g)).sum()))
