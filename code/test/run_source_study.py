"""Audit/freeze S2, detection diagnostics, and preset offline fusion. No YOLO calls."""
import argparse
import collections
import csv
import datetime
import json
from pathlib import Path
import time
import cv2
import numpy as np
from train_test import ROOT
import torch
from nscr_experiments import sha256
from run_crop_study import load,save,context,audit_fingerprint,to_coco
from crop_proposals import predict_roi_windows
from crop_infer import fuse_predictions
from crop_evaluate import evaluate_saved,fixed_conf_quality
from source_fusion import PRESET,METHODS,split_sources,source_fuse,edge_distance,overlap_features
from evaluate_nscr import box_iou_numpy
from resolution_diagnosis import match_indices

DEFAULT=ROOT/'runs/crop_study/source_local_20261012'
SOURCE=ROOT/'runs/crop_study/s0_s4_640_20261012'


def records(source):
    with (source/'S2/raw_candidates.jsonl').open(encoding='utf-8') as f:
        for line in f:yield json.loads(line)


def setup(output,source):
    output.mkdir(parents=True,exist_ok=False)
    a,gt,images=context(source)
    if load(source/'verification.json')['status']!='complete':raise RuntimeError('S study incomplete')
    if audit_fingerprint(a,images)!=a['immutable']:raise RuntimeError('Historical inputs changed')
    save(output/'preset.json',dict(parameters=PRESET,methods=METHODS,
        chosen_before_evaluation=True,validation_GT_tuning=False,
        note='MRU-inspired controls adapted to native640 windows and max500; NOT an exact reproduction of MRU full pipeline.'))
    frozen=[];meta={m['id']:m for m in images}
    for r in records(source):
        m=meta[r['image_id']];g,l,rid=split_sources(r)
        rois,info=predict_roi_windows(m['width'],m['height'],g,2)
        if [list(x) for x in rois]!=r['rois']:raise RuntimeError('S2 selector replay differs')
        frozen.append(dict(image_id=m['id'],file_name=m['file_name'],width=m['width'],height=m['height'],
            rois=r['rois'],global_predictions_sha256=__import__('hashlib').sha256(g.tobytes()).hexdigest()))
    if len(frozen)!=548 or {x['image_id'] for x in frozen}!=set(meta):raise RuntimeError('548 unique images required')
    save(output/'frozen_s2_rois.json',dict(source=str(source),checkpoint_sha256=a['immutable']['checkpoint_sha256'],
        selector=a['selector'],images=frozen,uses_validation_GT=False))
    a.update(source_study=str(source),training_authorized=False,local_training='prepare/dry-run only',
        frozen_rois_sha256=sha256(output/'frozen_s2_rois.json'),created=datetime.datetime.now().isoformat())
    save(output/'audit.json',a);save(output/'image_manifest.json',images)
    source_files=[source/'S2/raw_candidates.jsonl',source/'S2/coco_predictions.json',source/'S2/config.json',
        source/'S2/evaluation.json',source/'verification.json']
    save(output/'historical_artifacts_sha256.json',{str(p):sha256(p) for p in source_files})
    import psutil
    jobs=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            command=' '.join(p.info['cmdline'] or [])
            if p.info['name'] and p.info['name'].lower().startswith('python') and any(x in command for x in
                ['run_fusion.py','train_fusion.py','run_five.py','train_candidate.py','run_formal.py']):jobs.append(p.info)
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    save(output/'training_state.json',dict(active_training=jobs,queue=load(ROOT/'runs/fusion/formal_100e_b16_s42/queue.json')['status']))
    (output/'AUDIT.md').write_text('# S2工程审计\n\n'
        f"实际权重：`{a['checkpoint']}`；SHA256 `{a['immutable']['checkpoint_sha256']}`。\n\n"
        'M0保持25层YOLO11n+P2，原COCO YOLO11n预训练、100轮、batch8、seed42；本阶段只准备从M0最佳权重初始化局部模型。\n\n'
        'ROI逐图从S2原始记录冻结，548张的选区用原函数重新验证完全一致。候选池的前缀是已finalize的全局max500，'
        '后续按每个stage mapped_candidates切分局部候选，局部最多10000。原图xyxy，零起始类ID；逆gain/pad并加ROI偏移、裁剪到ROI/图像。'
        '最终类感知NMS .7、全图max500。所有来源可恢复，跨ROI统一NMS。\n\n'
        '主AP沿用converted-positive Ultralytics无额外忽略过滤；COCO small采用classwise crowd近似；'
        '固定conf .25/IoU .5误检分析使用单独ignore区域IoA>=.5屏蔽；VisDrone Python union-raster协议单列，未经MATLAB交叉验证。\n\n'
        '训练队列与实际进程保存在training_state.json。冻结清单、预设参数和历史文件哈希独立保存。'
        '融合函数不接收GT；诊断GT仅供事后分析。离线融合不会重新运行检测器。\n',encoding='utf-8')
    print('Audit complete; frozen 548 S2 images',flush=True)


def refusion(output,source):
    _,gt,images=context(output);meta={m['id']:m for m in images}
    predictions={k:[] for k in METHODS};times={k:[] for k in METHODS};stats={k:[] for k in METHODS}
    old=collections.defaultdict(list)
    for p in load(source/'S2/coco_predictions.json'):old[p['image_id']].append(p)
    for r in records(source):
        g,l,rid=split_sources(r);m=meta[r['image_id']]
        for method in METHODS:
            t=time.perf_counter();p,chosen,s=source_fuse(g,l,rid,r['rois'],m['width'],m['height'],method)
            elapsed=(time.perf_counter()-t)*1000;times[method].append(elapsed)
            result=to_coco(p,m['id']);predictions[method].extend(result);stats[method].append(dict(image_id=m['id'],fusion_ms=elapsed,**s))
            if method=='B0' and result!=old[m['id']]:raise RuntimeError('B0 must reproduce original S2 exactly')
    prior=load(source/'S2/timing.json')
    for method in METHODS:
        d=output/method;d.mkdir(exist_ok=False)
        save(d/'coco_predictions.json',predictions[method]);save(d/'evaluated_image_ids.json',list(meta))
        save(d/'fusion_per_image.json',stats[method])
        save(d/'config.json',dict(method=method,enabled=METHODS[method],parameters=PRESET,GT_input=False,
            frozen_rois_sha256=sha256(output/'frozen_s2_rois.json'),inference='Cached predictions ONLY'))
        save(d/'timing.json',dict(fusion_ms=dict(mean=float(np.mean(times[method])),p50=float(np.median(times[method])),p95=float(np.percentile(times[method],95))),
            measured_e2e_ms=None,historical_S2_e2e_ms=prior['e2e_ms'],
            estimated_e2e_ms=prior['e2e_ms']['mean']-__import__('pandas').read_csv(source/'S2/timing_per_image.csv')['fusion_ms'].mean()+np.mean(times[method]),
            cost_scope='Actual measured offline CPU fusion; E2E estimate replaces historical final fusion only. Not a fresh GPU E2E measurement.',
            mean_rois=prior['mean_rois'],mean_forward_calls=prior['mean_forward_calls'],
            mean_GFLOPs_actual_shape_estimate=prior['mean_GFLOPs_actual_shape_estimate']))
    print('B0 exactly reproduces S2; B1-B4 predictions saved, preset locked',flush=True)


def label_candidates(gt,record,meta):
    g,l,rid=split_sources(record);d=np.concatenate((g,l));n=len(d)
    a=[x for x in gt if not x.get('iscrowd')];boxes=np.array([x['bbox'] for x in a],float).reshape(-1,4);boxes[:,2:]+=boxes[:,:2]
    cls=np.array([x['category_id']-1 for x in a]);iou=box_iou_numpy(boxes,d[:,:4]);same=cls[:,None]==d[None,:,5]
    ignore=np.unique(np.array([x['bbox'] for x in gt if x.get('iscrowd')],float).reshape(-1,4),axis=0)
    blocked=np.zeros(n,bool)
    if len(ignore) and n:
        inter=np.maximum(0,np.minimum(d[:,None,2:4],ignore[None,:,:2]+ignore[None,:,2:4])-np.maximum(d[:,None,:2],ignore[None,:,:2])).prod(2)
        blocked=(inter/np.maximum((d[:,2:4]-d[:,:2]).prod(1)[:,None],1e-9)).max(1)>=.5
    matches=match_indices(iou,cls,d,conf=.001,blocked=blocked)
    labels=np.full(n,'background',object);labels[((iou>=.1)&same).any(0)]='localization'
    labels[((iou>=.5)&~same).any(0)]='wrong_class';labels[((iou>=.5)&same).any(0)]='duplicate'
    labels[matches[matches>=0]]='TP';labels[blocked]='ignored'
    global_matches=match_indices(iou[:,:len(g)],cls,g,conf=.25,blocked=blocked[:len(g)])
    p,chosen,_=source_fuse(g,l,rid,record['rois'],meta['width'],meta['height'],'B0')
    final_matches=match_indices(iou[:,chosen],cls,p,conf=.25,blocked=blocked[chosen])
    final_global=np.zeros(len(cls),bool);final_local=np.zeros(len(cls),bool)
    hit=final_matches>=0;chosen_hit=chosen[final_matches[hit]]
    final_global[hit]=chosen_hit<len(g);final_local[hit]=chosen_hit>=len(g)
    final_fp=np.ones(len(p),bool);final_fp[final_matches[hit]]=False
    final_good=(p[:,4]>=.25)&~blocked[chosen]
    counts=dict(global_TP=int((global_matches>=0).sum()),fused_TP=int(hit.sum()),
        newly_recovered_GT=int((hit&(global_matches<0)).sum()),lost_global_GT=int((~hit&(global_matches>=0)).sum()),
        local_retained_TP=int(final_local.sum()),local_retained_FP=int((final_fp&final_good&(chosen>=len(g))).sum()),
        global_retained_TP=int(final_global.sum()),global_retained_FP=int((final_fp&final_good&(chosen<len(g))).sum()))
    return labels,blocked,counts,chosen


def diagnose(output,source):
    a,gt,images=context(output);meta={m['id']:m for m in images};anns=collections.defaultdict(list)
    for x in gt['annotations']:anns[x['image_id']].append(x)
    histogram=collections.Counter();aggregate=collections.Counter();per_image=[];examples=[]
    d=output/'diagnosis';d.mkdir(exist_ok=False)
    with (d/'local_features.csv').open('w',newline='',encoding='utf-8') as f:
        names=['image_id','roi_id','confidence','class_id','size','aspect_ratio','edge_distance','same_iou','any_iou','same_conf','any_conf','class_agrees','score_delta','label','retained_B0']
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for j,r in enumerate(records(source)):
            m=meta[r['image_id']];g,l,rid=split_sources(r)
            labels,blocked,counts,chosen=label_candidates(anns[m['id']],r,m);aggregate.update(counts)
            per_image.append(dict(image_id=m['id'],**counts))
            feat=overlap_features(g,l);edge=edge_distance(l,rid,r['rois'],m['width'],m['height'])
            size=np.sqrt((l[:,2:4]-l[:,:2]).prod(1));retained=set(chosen[chosen>=len(g)]-len(g))
            for k,p in enumerate(l):
                confbin=('.001-.05','.05-.10','.10-.25','.25-.50','.50-1')[np.searchsorted([.05,.1,.25,.5],p[4],side='right')]
                sizebin=('<8','8-16','16-32','>=32')[np.searchsorted([8,16,32],size[k],side='right')]
                label=str(labels[len(g)+k]);histogram[('confidence',confbin,label)]+=1;histogram[('size',sizebin,label)]+=1;histogram[('class',str(int(p[5])+1),label)]+=1
                histogram[('edge',str(edge[k]<8),label)]+=1
                writer.writerow(dict(image_id=m['id'],roi_id=int(rid[k]),confidence=float(p[4]),class_id=int(p[5])+1,size=float(size[k]),
                    aspect_ratio=float((p[2]-p[0])/max(p[3]-p[1],1e-6)),edge_distance=float(edge[k]) if np.isfinite(edge[k]) else '',
                    **{x:float(v[k]) for x,v in feat.items()},label=label,retained_B0=k in retained))
            if j in (0,100,250,400):examples.append((r,labels,m))
            if (j+1)%50==0:print(f'Diagnostics {j+1}/548',flush=True)
    save(d/'summary.json',dict(fixed_conf25_source_counts=dict(aggregate),
        histogram=[dict(group=k[0],bin=k[1],label=k[2],count=v) for k,v in sorted(histogram.items())],
        protocol='Raw candidate TP/FP labels: all-source confidence-greedy one-to-one .001/.5, individual ignore IoA.5. '
        'New/lost GT and final source TP/FP: B0 output conf.25/.5, compared with global-only. '
        'Boundary is a geometric flag overlapping TP/FP categories, not evidence of actual truncation.'))
    save(d/'per_image.json',per_image)
    for r,labels,m in examples:
        image=cv2.imread(m['path']);g,l,rid=split_sources(r)
        for roi in r['rois']:cv2.rectangle(image,tuple(roi[:2]),tuple(roi[2:]),(255,200,0),3)
        for p,label in zip(l,labels[len(g):]):
            if p[4]<.25 or label=='ignored':continue
            color=(0,200,0) if label=='TP' else (0,0,255)
            cv2.rectangle(image,tuple(p[:2].astype(int)),tuple(p[2:4].astype(int)),color,1)
        cv2.imwrite(str(d/(Path(m['file_name']).stem+'.jpg')),image)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,group,bins in zip(axes,['confidence','size'],[['.001-.05','.05-.10','.10-.25','.25-.50','.50-1'],['<8','8-16','16-32','>=32']]):
        bottoms=np.zeros(len(bins))
        for label in ['TP','duplicate','wrong_class','localization','background','ignored']:
            values=np.array([histogram[(group,b,label)] for b in bins]);ax.bar(bins,values,bottom=bottoms,label=label);bottoms+=values
        ax.set_title(group);ax.set_ylabel('Raw local candidates')
    axes[1].legend(fontsize=7);fig.tight_layout()
    for suffix in ['png','svg','pdf']:fig.savefig(d/('candidate_errors.'+suffix),dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--stage',choices=['audit','fuse','diagnose','score'],required=True);p.add_argument('--methods',nargs='+',choices=list(METHODS),default=list(METHODS));args=p.parse_args()
    torch.set_num_threads(4)
    if args.stage=='audit':setup(args.output,args.source)
    elif args.stage=='fuse':refusion(args.output,args.source)
    elif args.stage=='diagnose':diagnose(args.output,args.source)
    else:
        a,gt,_=context(args.output)
        for method in args.methods:
            evaluate_saved(args.output/method,gt,Path(a['data_root'])/'val_coco.json',a['original_annotations'])
            save(args.output/method/'fixed_conf_quality.json',fixed_conf_quality(gt,load(args.output/method/'coco_predictions.json')))


if __name__=='__main__':main()
