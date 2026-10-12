"""S0-S4: a common whole640 baseline, prediction-guided ROI budgets and GT Oracle controls. NO TRAINING."""
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
from crop_geometry import oracle_windows,coverage_matrix,union_area
from crop_proposals import predict_roi_windows
from crop_infer import CropDetector,rect_shapes,fuse_predictions
from crop_evaluate import ultra_metrics
from run_crop_study import load,save,context,audit_fingerprint,EXPECTED_SHA,to_coco,summarize_timing,score
from nscr_experiments import sha256

SCHEMES={'S0':('none',0),'S1':('predicted',1),'S2':('predicted',2),'S3':('oracle',1),'S4':('oracle',2)}


def setup(output,source):
    if output.exists():raise FileExistsError('Use a NEW experiment directory; old outputs must be preserved')
    old,gt,images=context(source)
    if load(source/'verification.json')['status']!='complete':raise RuntimeError('Source study not complete')
    if audit_fingerprint(old,images)!=old['immutable']:raise RuntimeError('M0/data changed since source study')
    a=dict(old)
    a.update(study='S0-S4 whole640 + limited ROIs',source_study=str(source),training=False,
        fixed_global_imgsz=640,roi_imgsz=640,roi_size_original=640,budgets=[1,2],
        selector=dict(conf=.10,small_size=32.,margin=8,proposal_only_class_agnostic_nms=.7,
            candidate_rule='fixed20% grid plus predicted-box center/corner anchored windows',
            objective='greedy NEW full-containment confidence mass; secondary new object count, then lexicographic coordinates',
            zero_gain='stop; actual ROI count can be less than K; no duplicated/filler windows',
            source='S0 final max500 predictions only, no validation GT for S1/S2'),
        fusion='Each global branch is finalized exactly as S0 (max500) before ROI selection AND fusion. '
            'ROI intermediate max10000; final class-aware NMS.7/global max500. No boxes/score corrections.',
        decision='Fresh S0 val-compatible 640 replication; all five methods share offline scoring and global outputs.',
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    output.mkdir(parents=True)
    save(output/'audit.json',a);save(output/'image_manifest.json',images)
    (output/'AUDIT.md').write_text('# S0–S4 工程审计\n\n'
        f'继承已完成实验的模型/数据哈希，并在启动前重新检查：`{source}`。M0 SHA256：`{EXPECTED_SHA}`。\n\n'
        '548张验证图像、同一M0、FP32、同一640全局分支、同一原图GT与离线评价。仅推理，不训练。\n\n'
        'S1/S2使用全局最终预测(conf>=0.10、原图sqrt(wh)<32)，选区信号先做类无关0.7去重，'
        '贪心最大新增置信度总和，最多1/2个原图640窗口，内部边界安全距离8px。没有新增信号时少选ROI。'
        '信号去重不改变用于评分的检测输出。S3/S4使用GT Oracle几何，仅作诊断，不参与部署模型排名。\n\n'
        '先复现S0主AP后继续；主AP、COCO small、VisDrone Python参考协议分别报告。'
        '所有方案实际执行完整流程，保留每图延迟、候选截断、ROI和原始预测。\n',encoding='utf-8')


def validate_s0(output):
    a,gt,_=context(output)
    preds=load(output/'S0/coco_predictions.json')
    u=ultra_metrics(gt,preds)
    legacy=load(Path(a['historical960']).parent/'R0_640/evaluation.json')
    differences={k:u['metrics'][k]-legacy['ultralytics'][k] for k in ('metrics/mAP50(B)','metrics/mAP50-95(B)')}
    passed=bool(all(abs(d)<.0015 for d in differences.values()))
    save(output/'S0/control_reproduction.json',dict(passed=passed,metrics=u['metrics'],historical=legacy['ultralytics'],
        differences=differences,tolerance_percentage_points=.15,
        note='Current APsmall is scored on the SAME val-compatible predictions as main AP; legacy small was predict outputs.'))
    save(output/'S0/ultralytics_offline.json',u)
    if not passed:raise RuntimeError('S0 control failed; stop S1-S4 comparison')


def run_scheme(output,name,detector):
    a,gt,images=context(output);policy,budget=SCHEMES[name]
    d=output/name;d.mkdir(exist_ok=False)
    shapes=rect_shapes(gt['images'],640)
    detector.warm_shapes([(640,640)]+list(shapes.values()))
    anns=collections.defaultdict(list)
    for g in gt['annotations']:
        if not g.get('iscrowd'):anns[g['image_id']].append(g)
    import ultralytics,sys
    cfg=dict(scheme=name,global_imgsz=640,roi_imgsz=640,budget=budget,selection=policy,GT_Oracle=policy=='oracle',
        images=548,immutable=a['immutable'],selector=a['selector'],fusion=a['fusion'],postprocess=a['postprocess'],
        start=datetime.datetime.now(datetime.timezone.utc).isoformat(),timing=a['timing'],
        environment=dict(python=sys.version,torch=str(torch.__version__),ultralytics=ultralytics.__version__,
            opencv=cv2.__version__,numpy=np.__version__,device=str(detector.device),torch_CPU_threads=torch.get_num_threads()),
        warmup='30 square640 on model load, then 3 repeats per unique tensor shape before each scheme, excluded')
    save(d/'config.json',cfg)
    rows,predictions,coverage=[],[],[]
    with (d/'raw_candidates.jsonl').open('w',encoding='utf-8') as raw:
        for i,m in enumerate(images):
            detector.sync();start=time.perf_counter();image=cv2.imread(m['path'])
            if image is None or image.shape[:2]!=(m['height'],m['width']):raise RuntimeError('Image shape/decode mismatch')
            decoded=time.perf_counter()
            g,info=detector.infer(image,imgsz=640,global_shape=shapes[m['id']]);info['kind']='global'
            tf=time.perf_counter();global_pred,global_stats=fuse_predictions(g,.7,500);global_post_ms=(time.perf_counter()-tf)*1000
            selection_start=time.perf_counter()
            if policy=='predicted':
                rois,selection=predict_roi_windows(m['width'],m['height'],global_pred,budget)
            elif policy=='oracle':
                boxes=np.array([x['bbox'] for x in anns[m['id']]],dtype=float).reshape(-1,4);boxes[:,2:]+=boxes[:,:2]
                rois,selection=oracle_windows(m['width'],m['height'],boxes,budget)
            else:rois=[];selection={}
            selection_ms=(time.perf_counter()-selection_start)*1000
            pieces=[global_pred];stages=[info]
            for roi in rois:
                p,s=detector.infer(image,roi);s['kind']='local';pieces.append(p);stages.append(s)
            pool=np.concatenate(pieces)
            merge_start=time.perf_counter()
            if rois:pred,fusion=fuse_predictions(pool,.7,500)
            else:pred=global_pred;fusion=global_stats
            detector.sync();end=time.perf_counter();fusion_ms=(end-merge_start)*1000
            row=dict(image_id=m['id'],file_name=m['file_name'],width=m['width'],height=m['height'],rois=len(rois),
                forward_calls=1+len(rois),decode_ms=(decoded-start)*1000,selection_ms=selection_ms,
                global_finalize_ms=global_post_ms,fusion_ms=fusion_ms,merge_ms=global_post_ms+fusion_ms,e2e_ms=(end-start)*1000,
                GFLOPs_square_estimate=11.4582016*(1+len(rois)),
                GFLOPs_actual_shape_estimate=sum(11.4582016*s['input_shape'][0]*s['input_shape'][1]/640**2 for s in stages),
                stage1_max_nms_removed=sum(s['max_nms_removed'] for s in stages),
                stage1_intermediate_removed=sum(s['intermediate_cap_removed'] for s in stages),
                global_finalize_cap_removed=global_stats['final_cap_removed'],
                boundary_candidates=sum(s['internal_boundary_candidates'] for s in stages[1:]),
                boundary_high_conf=sum(s['internal_boundary_high_conf'] for s in stages[1:]),**fusion)
            for key in ('crop','preprocess','forward','roi_nms','map'):row[key+'_ms']=sum(s['time_ms'][key] for s in stages)
            if len(pred)>500 or len(rois)>budget or not np.isfinite(pred).all():raise RuntimeError('Budget/output invariant violated')
            rows.append(row);predictions.extend(to_coco(pred,m['id']))
            # GT enters S0/S1/S2 only HERE, after final detections and timed inference.
            boxes=np.array([x['bbox'] for x in anns[m['id']]],dtype=float).reshape(-1,4);size=np.sqrt(boxes[:,2:].prod(1));boxes[:,2:]+=boxes[:,:2]
            covered=coverage_matrix(rois,boxes,m['width'],m['height'],8).any(0) if rois else np.zeros(len(boxes),bool)
            masks={'all':np.ones(len(boxes),bool),'small':size<32,'<8':size<8,'8-16':(size>=8)&(size<16),'16-32':(size>=16)&(size<32)}
            coverage.append(dict(image_id=m['id'],rois=rois,selection=selection,
                groups={k:dict(total=int(mask.sum()),covered=int((mask&covered).sum())) for k,mask in masks.items()},
                crop_area_ratio=sum((r[2]-r[0])*(r[3]-r[1]) for r in rois)/(m['width']*m['height']),
                unique_crop_area_ratio=union_area(rois)/(m['width']*m['height'])))
            raw.write(json.dumps(dict(image_id=m['id'],rois=rois,selection=selection,candidates=pool.tolist(),
                global_predictions=global_pred.tolist(),global_stage_stats=info,global_finalize=global_stats,
                stages=stages,fusion=fusion),separators=(',',':'))+'\n')
            if (i+1)%20==0 or i+1==len(images):print(f'{name} {i+1}/548 ROIs={len(rois)} e2e={row["e2e_ms"]:.1f}ms',flush=True)
    save(d/'coco_predictions.json',predictions);save(d/'evaluated_image_ids.json',[m['id'] for m in images])
    save(d/'coverage_per_image.json',coverage);save(d/'timing.json',summarize_timing(rows))
    with (d/'timing_per_image.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    cfg.update(status='inference_complete',finish=datetime.datetime.now(datetime.timezone.utc).isoformat());save(d/'config.json',cfg)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'runs/crop_study/s0_s4_640_20261012')
    p.add_argument('--source',type=Path,default=ROOT/'runs/crop_study/c0_c1_c2_20261012')
    p.add_argument('--stage',choices=['all','audit','infer','score','report'],default='all')
    p.add_argument('--schemes',nargs='+',choices=list(SCHEMES))
    p.add_argument('--threads',type=int,default=4);p.add_argument('--device',default='0')
    args=p.parse_args();args.output=args.output.resolve();args.source=args.source.resolve()
    if args.schemes and args.stage!='score':p.error('--schemes only selects independent offline scoring groups')
    if args.stage in ('all','audit'):setup(args.output,args.source)
    a,_,_=context(args.output)
    if sha256(Path(a['checkpoint']))!=EXPECTED_SHA:raise RuntimeError('M0 checkpoint changed')
    torch.set_num_threads(args.threads)
    if args.stage in ('all','infer'):
        detector=CropDetector(a['checkpoint'],args.device)
        for name in SCHEMES:
            run_scheme(args.output,name,detector)
            if name=='S0':validate_s0(args.output)
        del detector;torch.cuda.empty_cache()
    if args.stage in ('all','score'):
        for name in args.schemes or SCHEMES:score(args.output,name)
    if args.stage in ('all','report'):
        from report_crop_budget import make_report
        make_report(args.output)
        a,_,images=context(args.output);fresh=audit_fingerprint(a,images)
        if fresh!=a['immutable']:raise RuntimeError('Weights/data changed during experiment')
        save(args.output/'verification.json',dict(status='complete',unchanged_weights_and_data=True,immutable=fresh,
            all_schemes=list(SCHEMES),source_sha256={f:sha256(ROOT/'code/test'/f) for f in ('crop_proposals.py','run_crop_budget.py','report_crop_budget.py',
                'crop_geometry.py','crop_infer.py','crop_evaluate.py')},finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__=='__main__':main()
