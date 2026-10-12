"""Prediction-only, bounded ROI selection. No GT/class labels needed by this selector."""
import numpy as np
from crop_geometry import fixed_windows, coverage_matrix


def weighted_roi_windows(width,height,boxes,scores,budget,window=640,margin=8):
    """Greedy new confidence-mass coverage; full containment; deterministic nested budgets."""
    boxes=np.asarray(boxes,dtype=float).reshape(-1,4)
    scores=np.asarray(scores,dtype=float).reshape(-1)
    if len(boxes)!=len(scores) or budget<0 or not np.isfinite(boxes).all() or not np.isfinite(scores).all() or (scores<0).any():
        raise ValueError('Finite boxes/nonnegative matching scores and nonnegative budget required')
    if not len(boxes) or budget==0:
        return [],dict(candidate_count=0,new_confidence_mass=[],new_prediction_counts=[])
    rw,rh=min(width,window),min(height,window)
    candidates=set(fixed_windows(width,height,window,.2))
    for b in boxes:
        cx,cy=(b[:2]+b[2:])/2
        for x,y in ((cx-rw/2,cy-rh/2),(b[0]-margin,b[1]-margin),(b[2]+margin-rw,b[3]+margin-rh)):
            x=int(np.clip(round(x),0,width-rw));y=int(np.clip(round(y),0,height-rh))
            candidates.add((x,y,x+rw,y+rh))
    candidates=sorted(candidates)
    covers=coverage_matrix(candidates,boxes,width,height,margin)
    remaining=np.ones(len(boxes),bool)
    selected,mass,counts=[],[],[]
    for _ in range(budget):
        new=covers&remaining
        gains=(new*scores).sum(1)
        maximum=float(gains.max())
        if maximum<=0:
            break
        ties=np.flatnonzero(np.isclose(gains,maximum,atol=1e-12,rtol=0))
        best=int(ties[np.argmax(new[ties].sum(1))])
        selected.append(candidates[best]);mass.append(float(gains[best]));counts.append(int(new[best].sum()))
        remaining&=~covers[best]
    return selected,dict(candidate_count=len(candidates),new_confidence_mass=mass,new_prediction_counts=counts,
        objective='Maximum NEW confidence sum of fully contained predicted small objects',margin=margin)


def predict_roi_windows(width,height,predictions,budget,conf=.10,small_size=32.,window=640,margin=8):
    """Global640 FINAL predictions -> proposal-only dedup -> GT-free ROI geometry.

    Class-agnostic .7 NMS removes multi-label copies ONLY in the selection signals;
    it never removes/changes global or ROI detection outputs used for evaluation.
    No eligible predictions / no new confidence mass -> fewer than the maximum K.
    """
    import torch
    import torchvision
    p=np.asarray(predictions,dtype=np.float32).reshape(-1,6)
    size=np.sqrt(np.maximum(0,p[:,2:4]-p[:,:2]).prod(1))
    p=p[(p[:,4]>=conf)&(size>0)&(size<small_size)]
    eligible=len(p)
    if eligible:
        t=torch.from_numpy(np.ascontiguousarray(p))
        keep=torchvision.ops.nms(t[:,:4],t[:,4],.7).numpy()
        p=p[keep]
    rois,info=weighted_roi_windows(width,height,p[:,:4],p[:,4],budget,window,margin)
    info.update(eligible_before_dedup=eligible,eligible_after_dedup=len(p),
        conf_threshold=conf,small_size=small_size,proposal_only_class_agnostic_nms=.7,
        signal_source='Global640 final max500 model predictions only; no GT',predicted_boxes=p.tolist())
    return rois,info
