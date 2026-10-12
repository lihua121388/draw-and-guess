"""C0/C1/C2 feasibility experiment. Inference ONLY; immutable M0/data; fresh outputs."""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
import time
import cv2
import numpy as np
from train_test import ROOT
import torch
import yaml
from nscr_experiments import sha256
from crop_geometry import fixed_windows, oracle_windows, coverage_matrix, union_area
from crop_infer import CropDetector, rect_shapes, fuse_predictions
from crop_evaluate import evaluate_saved, ultra_metrics

EXPECTED_SHA='ea7a1f071cfc5e0b4b3dc4016c38f36727b1f0302a9a9bc5c0192da39df1d16f'
SCHEMES={'C0':(960,None,None), 'C1-A':(None,'fixed',None),
         'C1-B':(960,'fixed',None), 'C1-C':(640,'fixed',None),
         **{f'C2-{branch}-K{k}':(None if branch=='A' else 960,'oracle',k)
            for branch in ('A','B') for k in (1,2,4)}}


def save(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(output, history):
    provenance=load(history/'R1_960/resolution_provenance.json')
    checkpoint=Path(provenance['checkpoint'])
    if not checkpoint.is_file() or sha256(checkpoint)!=EXPECTED_SHA:
        raise RuntimeError('Actual historical M0 checkpoint missing or SHA mismatch')
    data=Path(provenance['data'])
    cfg=yaml.safe_load(data.read_text(encoding='utf-8'))
    root=Path(cfg['path'])
    manifest=root/cfg['val']
    images=[Path(s) for s in manifest.read_text(encoding='utf-8').splitlines() if s.strip()]
    gt=load(root/'val_coco.json')
    if len(images)!=548 or len(set(images))!=548:
        raise RuntimeError('Expected identical 548 validation images')
    if {p.name for p in images}!={m['file_name'] for m in gt['images']}:
        raise RuntimeError('COCO/manifest image membership mismatch')
    for name,p in [('dataset_yaml_sha256',data),('val_manifest_sha256',manifest),
                   ('annotation_sha256',root/'val_coco.json')]:
        if sha256(p)!=provenance[name]:
            raise RuntimeError(f'Changed data: {name}')
    prep=load(root/'preparation.json')
    positives=sum(not a.get('iscrowd') for a in gt['annotations'])
    if positives!=38587:
        raise RuntimeError('Converted positive GT changed')
    original_annotations=Path(prep['splits']['val']['original_annotations'])
    digest=hashlib.sha256()
    original_digest=hashlib.sha256()
    image_digest=hashlib.sha256()
    for path in sorted(images):
        for d,f in ((digest,path.parent.parent/'labels'/(path.stem+'.txt')),
                    (original_digest,original_annotations/(path.stem+'.txt')),
                    (image_digest,path)):
            d.update(path.name.encode());d.update(bytes.fromhex(sha256(f)))
    current=dict(checkpoint_sha256=sha256(checkpoint),label_content_sha256=digest.hexdigest(),
                 original_annotations_sha256=original_digest.hexdigest(),image_content_sha256=image_digest.hexdigest(),
                 annotation_sha256=sha256(root/'val_coco.json'),val_manifest_sha256=sha256(manifest))
    result=dict(checkpoint=str(checkpoint),data=str(data),data_root=str(root),original_annotations=str(original_annotations),
        historical960=str(history/'R1_960'),immutable=current,images=548,positives=positives,
        model='Unchanged YOLO11n+P2 M0; layers 2/16/19/22, fusion23, Detect24; COCO pretraining unchanged',
        categories=gt['categories'],class_count=10,precision='FP32',training=False,
        historical_saved_predictions_complete=True,
        historical_prediction_branch='predict: best-class-only, auto rectangular letterbox; historical main AP from val: multi-label/batch rectangle',
        historical_predict_rescored_AP95=0.24139501714122347,
        decision='Re-run C0 with val-compatible resizing/multi-label decoder, save original-coordinate candidates and score ALL methods offline.',
        postprocess=dict(conf=.001,roi_nms_iou=.7,fusion_main_iou=.7,fusion_sensitivity=[.5,.6],
            max_nms=30000,intermediate_max_det=10000,final_max_det=500,multi_label=True,
            edge_suppression='disabled; geometry-only hook for future studies; edge candidates counted at 2px'),
        gt_rules='YOLO positives already filter invalid/category0/11 and centers inside ignored regions. Main AP has no extra ignore filter. '
          'COCO small uses classwise crowd approximation; size diagnosis ignores predictions by individual-region IoA>=.5; '
          'VisDrone Python union-raster scoring is separate and not MATLAB cross-validated.',
        timing='Sequential image end-to-end, synchronized GPU, includes disk decode, resize/ROI crop, forward, both NMS stages, '
          'mapping and CPU transfer. Excludes model load/initial 30 warmups, JSON serialization, scoring. '
          'Oracle selection geometry INCLUDED and separately timed. No reused/cached-forward cost masquerading as measured latency.',
        global_preprocessing='Exact val resize ceil(INTER_LINEAR), sorted dimension-only batch8 rectangles/pad.5, '
          'but each image forward measured independently; local square640 LetterBox scaleupTrue.',
        roi_policy=dict(size=640,overlap=.2,margin_oracle=8,budgets=[1,2,4],seed=42),
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    output.mkdir(parents=True,exist_ok=False)
    save(output/'audit.json',result)
    save(output/'image_manifest.json',[dict(path=str(p),**next(m for m in gt['images'] if m['file_name']==p.name)) for p in images])
    (output/'AUDIT.md').write_text('# 裁剪实验工程审计\n\n'
        f'模型 SHA256：`{EXPECTED_SHA}`，548 张验证图像，38587 个有效目标。仅推理，不训练。\n\n'
        '历史主 AP 与已保存 COCO 预测来自不同接口；保存预测重新计算主 AP 为 24.14%，不能将历史 24.68% '
        '直接当作该 JSON 的指标。因此重新运行 C0，并对各组保存预测使用统一离线评分。\n\n'
        '主融合 NMS=0.7，保持与 C0 一致；另外计算 0.5/0.6。两级 NMS、候选截断、原图坐标、忽略区域和计时范围详见 audit.json。\n',encoding='utf-8')
    return result


def context(output):
    a=load(output/'audit.json')
    gt=load(Path(a['data_root'])/'val_coco.json')
    images=load(output/'image_manifest.json')
    return a,gt,images


def validate_history_predictions(output):
    a,gt,_=context(output)
    d=Path(a['historical960'])
    predictions=load(d/'coco_predictions.json')
    ids=load(d/'evaluated_image_ids.json')
    meta={m['id']:m for m in gt['images']}
    counts=collections.Counter()
    zero_area=0
    for p in predictions:
        m=meta[p['image_id']]
        x,y,w,h=p['bbox']
        if p['category_id'] not in range(1,11) or not np.isfinite([x,y,w,h,p['score']]).all():
            raise RuntimeError('Invalid historical saved prediction')
        if x<-.001 or y<-.001 or w<0 or h<0 or x+w>m['width']+.001 or y+h>m['height']+.001:
            raise RuntimeError('Historical prediction coordinates outside original image')
        counts[p['image_id']]+=1;zero_area+=int(w*h==0)
    if set(ids)!=set(meta) or len(ids)!=548 or max(counts.values())>500:
        raise RuntimeError('Historical image list/maxDet invalid')
    result=dict(status='complete',images=len(ids),predictions=len(predictions),original_coordinates=True,
        category_ids=list(range(1,11)),max_predictions_per_image=max(counts.values()),zero_area_predictions=zero_area,
        predictions_sha256=sha256(d/'coco_predictions.json'),image_ids_sha256=sha256(d/'evaluated_image_ids.json'),
        saved_configuration=str(d/'resolution_provenance.json'),historical_evaluation=str(d/'evaluation.json'),
        reuse_note='Artifacts complete, but predict branch differs from val main AP; retained as separate legacy reference, not mixed into new scoring.')
    save(output/'history_prediction_validation.json',result)
    return result


def to_coco(pred,image_id):
    return [dict(image_id=image_id,category_id=int(p[5])+1,
            bbox=[float(p[0]),float(p[1]),float(p[2]-p[0]),float(p[3]-p[1])],score=float(p[4])) for p in pred]


def summarize_timing(rows):
    fields=['e2e_ms','decode_ms','selection_ms','crop_ms','preprocess_ms','forward_ms','roi_nms_ms','map_ms','merge_ms']
    stats={k:dict(mean=float(np.mean([r[k] for r in rows])),p50=float(np.median([r[k] for r in rows])),
            p95=float(np.percentile([r[k] for r in rows],95))) for k in fields}
    stats.update(images=len(rows),mean_rois=float(np.mean([r['rois'] for r in rows])),
        total_rois=sum(r['rois'] for r in rows),forward_calls=sum(r['forward_calls'] for r in rows),
        mean_forward_calls=float(np.mean([r['forward_calls'] for r in rows])),
        mean_GFLOPs_square_estimate=float(np.mean([r['GFLOPs_square_estimate'] for r in rows])),
        mean_GFLOPs_actual_shape_estimate=float(np.mean([r['GFLOPs_actual_shape_estimate'] for r in rows])))
    return stats


def run_scheme(output,name,detector,smoke_ids=None):
    a,gt,images=context(output)
    global_size,policy,budget=SCHEMES[name]
    directory=output/('smoke20' if smoke_ids is not None else name)
    directory.mkdir(exist_ok=False)
    anns=collections.defaultdict(list)
    for g in gt['annotations']:
        if not g.get('iscrowd'):
            anns[g['image_id']].append(g)
    shapes={s:rect_shapes(gt['images'],s) for s in (640,960)}
    detector.warm_shapes([(640,640)]+(list(shapes[global_size].values()) if global_size else []))
    if smoke_ids is not None:
        images=[m for m in images if m['id'] in smoke_ids]
    rows,predictions,coverage=[],[],[]
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    cfg=dict(scheme=name,GT_Oracle=policy=='oracle',global_imgsz=global_size,roi_imgsz=640,roi_policy=policy,
        budget=budget,images=len(images),immutable=a['immutable'],postprocess=a['postprocess'],
        start=started,timing=a['timing'],smoke=smoke_ids is not None,
        warmup='30 square640 at model load; 3 forward repeats per unique shape before each scheme; excluded from timed/costed calls')
    import ultralytics
    cfg['environment']=dict(python=sys.version,torch=str(torch.__version__),ultralytics=ultralytics.__version__,
        opencv=cv2.__version__,numpy=np.__version__,device=str(detector.device),torch_CPU_threads=torch.get_num_threads(),
        gpu=torch.cuda.get_device_name(detector.device) if detector.device.type=='cuda' else None)
    save(directory/'config.json',cfg)
    with (directory/'raw_candidates.jsonl').open('w',encoding='utf-8') as raw_file:
        for i,m in enumerate(images):
            detector.sync();start=time.perf_counter()
            image=cv2.imread(m['path'])
            if image is None or image.shape[:2]!=(m['height'],m['width']):
                raise RuntimeError('Image dimensions/decode mismatch')
            decoded=time.perf_counter()
            boxes=np.array([g['bbox'] for g in anns[m['id']]],dtype=float).reshape(-1,4)
            boxes[:,2:]+=boxes[:,:2]
            if policy=='oracle':
                rois,selection=oracle_windows(m['width'],m['height'],boxes,budget)
            elif policy=='fixed':
                rois=fixed_windows(m['width'],m['height'])
                selection={'candidate_count':len(rois)}
            else:
                rois=[];selection={}
            selected=time.perf_counter()
            pieces,details=[],[]
            if global_size:
                pred,info=detector.infer(image,imgsz=global_size,global_shape=shapes[global_size][m['id']])
                info['kind']='global';pieces.append(pred);details.append(info)
            for roi in rois:
                pred,info=detector.infer(image,roi=roi)
                info['kind']='local';pieces.append(pred);details.append(info)
            pool=np.concatenate(pieces) if pieces else np.empty((0,6),dtype=np.float32)
            merged_start=time.perf_counter()
            pred,fusion=fuse_predictions(pool,.7,500)
            detector.sync();end=time.perf_counter()
            row=dict(image_id=m['id'],file_name=m['file_name'],width=m['width'],height=m['height'],rois=len(rois),
                forward_calls=len(details),decode_ms=(decoded-start)*1000,selection_ms=(selected-decoded)*1000,
                merge_ms=(end-merged_start)*1000,e2e_ms=(end-start)*1000,
                GFLOPs_square_estimate=(25.7809536 if global_size==960 else 11.4582016 if global_size==640 else 0)+len(rois)*11.4582016,
                GFLOPs_actual_shape_estimate=sum(11.4582016*d['input_shape'][0]*d['input_shape'][1]/640**2 for d in details),
                stage1_max_nms_removed=sum(d['max_nms_removed'] for d in details),
                stage1_intermediate_removed=sum(d['intermediate_cap_removed'] for d in details),
                boundary_candidates=sum(d['internal_boundary_candidates'] for d in details if d['kind']=='local'),
                boundary_high_conf=sum(d['internal_boundary_high_conf'] for d in details if d['kind']=='local'),**fusion)
            for key in ('crop','preprocess','forward','roi_nms','map'):
                row[key+'_ms']=sum(d['time_ms'][key] for d in details)
            rows.append(row);predictions.extend(to_coco(pred,m['id']))
            # Coverage is measured AFTER inference; for C0/C1 never affects predictions.
            covered=coverage_matrix(rois,boxes,m['width'],m['height'],margin=8 if policy=='oracle' else 0).any(0) if rois else np.zeros(len(boxes),bool)
            size=np.sqrt((boxes[:,2:]-boxes[:,:2]).prod(1))
            groups={'all':np.ones(len(boxes),bool),'small':size<32,'<8':size<8,
                    '8-16':(size>=8)&(size<16),'16-32':(size>=16)&(size<32)}
            coverage.append(dict(image_id=m['id'],rois=rois,selection=selection,
                crop_area_ratio=sum((r[2]-r[0])*(r[3]-r[1]) for r in rois)/(m['width']*m['height']),
                unique_crop_area_ratio=union_area(rois)/(m['width']*m['height']),
                groups={key:dict(total=int(mask.sum()),covered=int((covered&mask).sum())) for key,mask in groups.items()}))
            raw_file.write(json.dumps(dict(image_id=m['id'],rois=rois,global_size=global_size,
                candidates=pool.tolist(),stages=details,fusion=fusion),separators=(',',':'))+'\n')
            if len(pred)>500 or not np.isfinite(pred).all() or (len(pred) and ((pred[:,:2]<0).any() or
                    (pred[:,[0,2]]>m['width']+.01).any() or (pred[:,[1,3]]>m['height']+.01).any())):
                raise RuntimeError('Mapping/final cap integrity check failed')
            if (i+1)%20==0 or i+1==len(images):
                print(f'{name} {i+1}/{len(images)} ROI={len(rois)} e2e={row["e2e_ms"]:.1f}ms',flush=True)
    save(directory/'coco_predictions.json',predictions)
    save(directory/'evaluated_image_ids.json',[m['id'] for m in images])
    save(directory/'coverage_per_image.json',coverage)
    with (directory/'timing_per_image.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(directory/'timing.json',summarize_timing(rows))
    if smoke_ids is not None:
        save(directory/'verification.json',dict(status='passed',images=len(images),seed=42,
            ROI_bounds_valid=True,final_max500=True,finite_original_coordinates=True,
            fixed_full_image_coverage=all(union_area(r['rois'])==m['width']*m['height'] for r,m in zip(coverage,images))))
    cfg.update(status='inference_complete',finish=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(directory/'config.json',cfg)
    return directory


def score(output,name,reference=True):
    a,gt,_=context(output)
    directory=output/name
    if (directory/'evaluation.json').exists():
        return load(directory/'evaluation.json')
    return evaluate_saved(directory,gt,Path(a['data_root'])/'val_coco.json',a['original_annotations'],reference)


def c0_check(output,result,history):
    legacy=load(history/'R1_960/evaluation.json')
    actual=result['ultralytics']
    deviations={k:actual[k]-v for k,v in legacy['ultralytics'].items() if k.startswith('metrics/')}
    # Independent batch1 execution/offline native-coordinate GT introduces tiny numerical differences.
    passed=bool(abs(deviations['metrics/mAP50-95(B)'])<.0015 and abs(deviations['metrics/mAP50(B)'])<.0015)
    save(output/'C0/reproduction.json',dict(passed=passed,tolerance_AP_percentage_points=.15,
        legacy=legacy['ultralytics'],current=actual,differences=deviations,
        legacy_small=legacy['coco_supplemental'],current_small=result['coco_supplemental'],
        small_note='Historical small metrics were predict outputs; current small metrics use the SAME saved val-compatible predictions as main AP. '
          'Branch mismatch is disclosed; no mixing historical small with new main predictions.'))
    if not passed:
        raise RuntimeError('C0 failed historical AP reproduction; do not continue C1/C2')


def sensitivity(output,schemes=None):
    a,gt,_=context(output)
    from evaluate_nscr import coco_metrics
    for name in schemes or ('C0','C1-A','C1-B','C1-C'):
        for threshold in (.5,.6):
            d=output/'sensitivity'/f'{name}_iou{threshold}'
            if d.exists():
                continue
            d.mkdir(parents=True)
            preds=[];times=[];trunc=0
            with (output/name/'raw_candidates.jsonl').open(encoding='utf-8') as f:
                for line in f:
                    record=json.loads(line);t=time.perf_counter()
                    merged,s=fuse_predictions(record['candidates'],threshold,500)
                    times.append((time.perf_counter()-t)*1000);trunc+=s['final_cap_removed']
                    preds.extend(to_coco(merged,record['image_id']))
            save(d/'coco_predictions.json',preds)
            u=ultra_metrics(gt,preds)
            c=coco_metrics(Path(a['data_root'])/'val_coco.json',preds,[m['id'] for m in gt['images']],500)
            save(d/'evaluation.json',dict(ultralytics=u['metrics'],per_class=u['per_class'],coco_supplemental=c,
                fusion_iou=threshold,changed_only='offline fusion NMS; no new inference or parameter search',
                mean_merge_ms=float(np.mean(times)),final_cap_removed=trunc))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/crop_study/c0_c1_c2_20261012')
    parser.add_argument('--history',type=Path,default=ROOT/'runs/fusion/resolution_m0_20261011')
    parser.add_argument('--stage',choices=['all','audit','c0','smoke','c1','c2','score','sensitivity','report'],default='all')
    parser.add_argument('--device',default='0')
    parser.add_argument('--schemes',nargs='+',choices=list(SCHEMES),help='Score selected completed schemes only (separate output groups can be scored independently)')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();args.output=args.output.resolve();args.history=args.history.resolve()
    if args.schemes and args.stage not in ('score','sensitivity'):
        parser.error('--schemes applies only to independent offline score/sensitivity groups')
    if args.threads<1:
        parser.error('--threads must be positive')
    if args.stage in ('all','audit'):
        audit(args.output,args.history)
    a,gt,images=context(args.output)
    if sha256(Path(a['checkpoint']))!=EXPECTED_SHA:
        raise RuntimeError('Checkpoint changed')
    torch.set_num_threads(args.threads)
    detector=None
    if args.stage in ('all','c0','smoke','c1','c2'):
        detector=CropDetector(a['checkpoint'],args.device)
        if list(detector.names.values())!=[c['name'] for c in gt['categories']]:
            raise RuntimeError('Class mismatch')
    if args.stage in ('all','c0'):
        run_scheme(args.output,'C0',detector)
        result=score(args.output,'C0');c0_check(args.output,result,args.history)
    if args.stage in ('all','smoke'):
        if not load(args.output/'C0/reproduction.json')['passed']:
            raise RuntimeError('C0 guard missing')
        rng=np.random.default_rng(42);ids=rng.choice([m['id'] for m in images],20,replace=False).tolist()
        run_scheme(args.output,'C1-B',detector,ids)
    if args.stage in ('all','c1','c2'):
        if not load(args.output/'C0/reproduction.json')['passed'] or not load(args.output/'smoke20/verification.json')['status']=='passed':
            raise RuntimeError('C0 and smoke guards required')
    if args.stage in ('all','c1'):
        for name in ('C1-A','C1-B','C1-C'):
            run_scheme(args.output,name,detector)
    if args.stage in ('all','c2'):
        for name in (n for n in SCHEMES if n.startswith('C2')):
            run_scheme(args.output,name,detector)
    # Release GPU before offline metrics. No formal training is ever called.
    if detector is not None:
        del detector;torch.cuda.empty_cache()
    if args.stage in ('all','score'):
        for name in args.schemes or SCHEMES:
            score(args.output,name)
    if args.stage in ('all','sensitivity'):
        sensitivity(args.output,args.schemes)
    if args.stage in ('all','report'):
        validate_history_predictions(args.output)
        from report_crop_study import make_report
        make_report(args.output)
    if args.stage in ('all','report'):
        fresh=audit_fingerprint(a,images)
        if fresh!=a['immutable']:
            raise RuntimeError('Weights/data changed during experiment')
        save(args.output/'verification.json',dict(status='complete',all_schemes=list(SCHEMES),
            unchanged_weights_and_data=True,immutable=fresh,
            source_sha256={f:sha256(ROOT/'code/test'/f) for f in ('crop_geometry.py','crop_infer.py','crop_evaluate.py','run_crop_study.py','report_crop_study.py')},
            finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))


def audit_fingerprint(a,images):
    digests=[hashlib.sha256() for _ in range(3)]
    for m in sorted(images,key=lambda x:Path(x['path'])):
        path=Path(m['path'])
        for d,f in zip(digests,(path.parent.parent/'labels'/(path.stem+'.txt'),
                               Path(a['original_annotations'])/(path.stem+'.txt'),path)):
            d.update(path.name.encode());d.update(bytes.fromhex(sha256(f)))
    root=Path(a['data_root'])
    return dict(checkpoint_sha256=sha256(Path(a['checkpoint'])),label_content_sha256=digests[0].hexdigest(),
        original_annotations_sha256=digests[1].hexdigest(),image_content_sha256=digests[2].hexdigest(),
        annotation_sha256=sha256(root/'val_coco.json'),val_manifest_sha256=sha256(root/'val.txt'))


if __name__=='__main__':
    main()
