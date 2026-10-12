"""Frozen-detector instance quality components. No fitting or dataset side effects."""
import numpy as np
from train_test import ROOT
import torch
from torch import nn
from torch.nn import functional as F
from crop_geometry import coverage_matrix
from source_fusion import edge_distance,overlap_features
from evaluate_nscr import box_iou_numpy

LOCAL=['score_logit','log_width','log_height','log_aspect']+[f'class_{i}' for i in range(10)]
BOUNDARY=['relative_distance','distance','boundary_available','left_nearest','top_nearest','right_nearest','bottom_nearest']
GLOBAL=['same_iou','same_conf','any_iou','any_conf','class_agrees','score_delta','global_support_available']
CROSS=['observation_available','support_iou','support_conf','supported','observed_no_support']
FEATURES=LOCAL+BOUNDARY+GLOBAL+CROSS
GROUPS={k:[FEATURES.index(n) for n in v] for k,v in [('local',LOCAL),('boundary',BOUNDARY),('global',GLOBAL),('cross',CROSS)]}
VARIANTS=['C2','C3a','C3','C4','C4_no_boundary','C4_no_global','C4_no_cross','C4_no_gain_loss']


def extract_features(global_pred,pieces,rois,width,height):
    pieces=[np.asarray(p,np.float32).reshape(-1,6) for p in pieces]
    local=np.concatenate(pieces) if pieces else np.empty((0,6),np.float32)
    rid=np.repeat(np.arange(len(pieces)),[len(p) for p in pieces]);n=len(local)
    result=np.zeros((n,len(FEATURES)),np.float32)
    def put(name,value):result[:,FEATURES.index(name)]=value
    wh=np.maximum(local[:,2:4]-local[:,:2],1e-6);scale=np.sqrt(wh.prod(1))
    score=np.clip(local[:,4],1e-6,1-1e-6);put('score_logit',np.log(score/(1-score)))
    put('log_width',np.log(wh[:,0]));put('log_height',np.log(wh[:,1]));put('log_aspect',np.log(wh[:,0]/wh[:,1]))
    if n:result[np.arange(n),[FEATURES.index(f'class_{int(c)}') for c in local[:,5]]]=1
    distance=edge_distance(local,rid,rois,width,height);available=np.isfinite(distance)
    put('boundary_available',available);put('distance',np.where(available,np.log1p(distance),0))
    put('relative_distance',np.where(available,np.log1p(np.minimum(distance/(scale+1e-6),1e4)),0))
    offset=0
    for i,(x,y,xe,ye) in enumerate(rois):
        p=pieces[i];d=np.full((len(p),4),np.inf)
        if x>0:d[:,0]=p[:,0]-x
        if y>0:d[:,1]=p[:,1]-y
        if xe<width:d[:,2]=xe-p[:,2]
        if ye<height:d[:,3]=ye-p[:,3]
        ids=np.flatnonzero(np.isfinite(d).any(1));nearest=d.argmin(1)
        for j,label in enumerate(['left_nearest','top_nearest','right_nearest','bottom_nearest']):
            result[offset+ids,FEATURES.index(label)]=(nearest[ids]==j)
        opportunity=np.zeros(len(p),bool);support_iou=np.zeros(len(p));support_conf=np.zeros(len(p))
        for k,q in enumerate(pieces):
            if k==i:continue
            covered=coverage_matrix([rois[k]],p[:,:4],width,height,8)[0];opportunity|=covered
            q=q[q[:,4]>=.10]
            if not len(q):continue
            for start in range(0,len(p),1024):
                pp=p[start:start+1024];ious=box_iou_numpy(pp[:,:4],q[:,:4])
                ious=np.where(pp[:,None,5]==q[None,:,5],ious,0);idx=ious.argmax(1);best=ious[np.arange(len(pp)),idx]
                update=covered[start:start+len(pp)]&(best>support_iou[start:start+len(pp)])
                rows=start+np.flatnonzero(update);support_iou[rows]=best[update];support_conf[rows]=q[idx[update],4]
        supported=opportunity&(support_iou>=.5)
        for name,v in [('observation_available',opportunity),('support_iou',support_iou),('support_conf',support_conf),('supported',supported),('observed_no_support',opportunity&~supported)]:
            result[offset:offset+len(p),FEATURES.index(name)]=v
        offset+=len(p)
    context=overlap_features(np.asarray(global_pred,np.float32).reshape(-1,6),local)
    for name,values in context.items():put(name,values)
    put('global_support_available',context['same_iou']>0)
    if not np.isfinite(result).all():raise ValueError('Nonfinite features')
    return result


def feature_mask(variant):
    if variant not in VARIANTS:raise ValueError('Unknown ablation')
    mask=np.ones(len(FEATURES),np.float32)
    disabled=[]
    if variant=='C2':disabled=['global','cross']
    elif variant in ('C3a','C3'):disabled=['cross']
    elif variant.startswith('C4_no_') and variant!='C4_no_gain_loss':disabled=[variant[len('C4_no_'):]]
    for group in disabled:mask[GROUPS[group]]=0
    return mask


class QualityMLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(len(FEATURES),32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,2))

    def forward(self,x):return self.net(x)


def quality_loss(logits,targets,weights,gain_enabled=True):
    """Targets = multi-IoU validity and JOINT gain; head2 estimates conditional gain."""
    valid=targets[:,0];conditional=targets[:,1]/valid.clamp_min(1e-8)
    if (targets[:,1]>valid+1e-6).any():raise ValueError('Joint gain cannot exceed validity')
    first=F.binary_cross_entropy_with_logits(logits[:,0],valid,reduction='none')
    second=F.binary_cross_entropy_with_logits(logits[:,1],conditional.clamp(0,1),reduction='none')*valid
    return ((first+second*float(gain_enabled))*weights).sum()/weights.sum().clamp_min(1)


def rescore(scores,logits,beta=1.,gain_enabled=False,temperatures=(1.,1.)):
    if not 0<=beta<=1 or min(temperatures)<=0:raise ValueError('Invalid calibration settings')
    probabilities=torch.sigmoid(logits/torch.tensor(temperatures,device=logits.device))
    factor=probabilities[:,0]
    if gain_enabled:factor=factor*(beta+(1-beta)*probabilities[:,1])
    return scores*factor
