"""Extra local-only error/source relations; retrospective GT diagnostics ONLY."""
import collections
import csv
from pathlib import Path
import cv2
import numpy as np
from train_test import ROOT
from run_source_study import DEFAULT,records,label_candidates
from run_crop_study import load,save,context
from source_fusion import split_sources,edge_distance,source_fuse
from evaluate_nscr import box_iou_numpy
from resolution_diagnosis import match_indices


def main():
    a,gt,images=context(DEFAULT);d=DEFAULT/'diagnosis_detail';d.mkdir(exist_ok=False)
    anns=collections.defaultdict(list);meta={m['id']:m for m in images};counts=collections.Counter();relations=collections.Counter()
    for x in gt['annotations']:anns[x['image_id']].append(x)
    for r in records(Path(a['source_study'])):
        g,l,rid=split_sources(r);m=meta[r['image_id']]
        labels,blocked,summary,chosen=label_candidates(anns[m['id']],r,m)
        high=np.flatnonzero(l[:,4]>=.25);pp=l[high]
        overlaps=box_iou_numpy(pp[:,:4],pp[:,:4]);np.fill_diagonal(overlaps,0)
        same=pp[:,None,5]==pp[None,:,5];same_roi=rid[high,None]==rid[None,high]
        counts['local_raw_highconf']+=len(high)
        counts['same_ROI_sameclass_overlap_gt07']+=int(((overlaps>.7)&same&same_roi).any(1).sum())
        counts['cross_ROI_sameclass_overlap_ge05']+=int(((overlaps>=.5)&same&~same_roi).any(1).sum())
        counts['same_ROI_class_conflict_overlap_ge05']+=int(((overlaps>=.5)&~same&same_roi).any(1).sum())
        edge=edge_distance(l,rid,r['rois'],m['width'],m['height'])
        positives=[x for x in anns[m['id']] if not x.get('iscrowd')]
        boxes=np.array([x['bbox'] for x in positives],float).reshape(-1,4);boxes[:,2:]+=boxes[:,:2];classes=np.array([x['category_id']-1 for x in positives])
        p,idx,st=source_fuse(g,l,rid,r['rois'],m['width'],m['height'],'B0')
        iou=box_iou_numpy(boxes,p[:,:4]);matches=match_indices(iou,classes,p,conf=.25,blocked=blocked[idx])
        fp=np.ones(len(p),bool);fp[matches[matches>=0]]=False;same=classes[:,None]==p[None,:,5]
        categories=np.full(len(p),'background',object);categories[((iou>=.1)&same).any(0)]='localization'
        categories[((iou>=.5)&~same).any(0)]='wrong_class';categories[((iou>=.5)&same).any(0)]='duplicate';categories[~fp]='TP'
        selected=(idx>=len(g))&(p[:,4]>=.25)&~blocked[idx]
        for k in np.flatnonzero(selected):
            label=str(categories[k]);distance=float(edge[idx[k]-len(g)])
            counts['final_local_'+label]+=1;counts['final_local_boundary8_'+label]+=int(distance<8);counts['final_local_clipped2_'+label]+=int(distance<=2)
    with (DEFAULT/'diagnosis/local_features.csv').open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            iou=float(row['same_iou']);delta=float(row['score_delta']);label=row['label']
            ibin=('0-.1','.1-.5','.5-.7','.7-1')[np.searchsorted([.1,.5,.7],iou,side='right')]
            dbin=('<-.2','-.2-0','0-.2','>=.2')[np.searchsorted([-.2,0,.2],delta,side='right')]
            relations[(ibin,dbin,label)]+=1
    save(d/'summary.json',dict(counts=dict(counts),iou_score_delta_histogram=[dict(same_global_iou=k[0],local_minus_global_score=k[1],label=k[2],count=v) for k,v in sorted(relations.items())],
        note='Pair overlap counts count predictions having at least one partner, not unique duplicate pairs; subset raw conf>=.25. '
        'Final local FP categories are mutually exclusive proxies; boundary flags overlap categories and do not establish actual truncation.'))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    labels=['TP','duplicate','wrong_class','localization','background'];base=[counts['final_local_'+k] for k in labels];edge=[counts['final_local_boundary8_'+k] for k in labels]
    axes[0].bar(labels,base,label='All final local');axes[0].bar(labels,edge,label='Within 8px internal edge');axes[0].tick_params(axis='x',rotation=30);axes[0].legend(fontsize=8)
    matrix=np.zeros((4,4));ibs=['0-.1','.1-.5','.5-.7','.7-1'];dbs=['<-.2','-.2-0','0-.2','>=.2']
    for i,ib in enumerate(ibs):
        for j,db in enumerate(dbs):
            tp=relations[(ib,db,'TP')];den=sum(relations[(ib,db,l)] for l in labels);matrix[i,j]=tp/max(1,den)
    im=axes[1].imshow(matrix,vmin=0,vmax=1);axes[1].set_xticks(range(4),dbs);axes[1].set_yticks(range(4),ibs);axes[1].set_xlabel('Local - nearest same-class global score');axes[1].set_ylabel('Same-class global max IoU');fig.colorbar(im,ax=axes[1],label='Raw candidate TP fraction (.001)')
    fig.tight_layout()
    for suffix in ['png','svg','pdf']:fig.savefig(d/('source_relations.'+suffix),dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
