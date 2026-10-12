"""Frozen S2 ROI E2/E3 inference, real complete GPU timing and offline scoring."""
import argparse
import collections
import csv
import datetime
from pathlib import Path
import time
import cv2
import numpy as np
import torch
import yaml
from train_test import ROOT
from run_source_study import DEFAULT,records
from run_crop_study import load,save,context,to_coco,audit_fingerprint
from crop_infer import CropDetector,rect_shapes,fuse_predictions
from source_fusion import PRESET,split_sources,source_fuse
from crop_evaluate import evaluate_saved,fixed_conf_quality
from nscr_experiments import sha256

ROI_SHA='659d22fd9f68d1f1639b07a20dc8eabb94222c9363935d8d60606dbaf0f54b01'


def audit(study,output):
    output.mkdir(parents=True,exist_ok=False)
    a,gt,images=context(study);source=Path(a['source_study']);local=study/'local_training/A1_seed42/weights/best.pt'
    if sha256(study/'frozen_s2_rois.json')!=ROI_SHA:raise RuntimeError('Frozen ROI SHA changed')
    if audit_fingerprint(a,images)!=a['immutable']:raise RuntimeError('M0/validation changed')
    frozen=load(study/'frozen_s2_rois.json')['images']
    if len(frozen)!=548 or len({x['image_id'] for x in frozen})!=548 or sum(len(x['rois']) for x in frozen)!=935:raise RuntimeError('ROI membership/count changed')
    cfg=yaml.safe_load((study/'local_training/A1_seed42/args.yaml').read_text())
    with (study/'local_training/A1_seed42/results.csv').open() as f:rows=list(csv.DictReader(f))
    best=max(rows,key=lambda r:.1*float(r['metrics/mAP50(B)'])+.9*float(r['metrics/mAP50-95(B)']))
    if len(rows)!=30 or cfg['batch']!=8 or cfg['lr0']!=.0001:raise RuntimeError('Actual training configuration differs')
    checkpoint=torch.load(local,map_location='cpu',weights_only=False)
    history=checkpoint.get('train_results',{})
    evidence=dict(args=cfg,best_epoch_by_csv=int(best['epoch']),best_csv_metrics=best,
        best_selection_rule='0.1*AP50+0.9*AP95 (local Ultralytics fitness)',
        checkpoint_metrics_match_selected_epoch=True,
        checkpoint_epoch=checkpoint.get('epoch'),checkpoint_best_fitness=checkpoint.get('best_fitness'),
        checkpoint_train_metrics=checkpoint.get('train_metrics'),
        note='Final stripped checkpoint epoch may be -1. Best epoch inferred from CSV fitness; full train_results in checkpoint cross-checked.',
        checkpoint_history_rows=len(history.get('epoch',[])))
    if history and list(map(int,history['epoch']))!=list(map(int,[r['epoch'] for r in rows[:len(history['epoch'])]])):raise RuntimeError('Checkpoint training history differs')
    metrics=checkpoint.get('train_metrics',{})
    if any(abs(float(best[k])-metrics[k])>1e-6 for k in ('metrics/mAP50(B)','metrics/mAP50-95(B)')):raise RuntimeError('best.pt metrics do not match best fitness CSV epoch')
    save(output/'training_evidence.json',evidence)
    old={r['image_id']:r for r in records(source)};expect={r['image_id']:r for r in frozen};meta={m['id']:m for m in images}
    controls={m:collections.defaultdict(list) for m in ('E0','E1')}
    for name,branch in [('E0','B0'),('E1','B2')]:
        for p in load(study/branch/'coco_predictions.json'):controls[name][p['image_id']].append(p)
    for m in images:
        r=old[m['id']]
        if r['rois']!=expect[m['id']]['rois']:raise RuntimeError('Frozen/history ROI differs')
        g,l,rid=split_sources(r)
        for name,method in [('E0','B0'),('E1','B2')]:
            p,_,_=source_fuse(g,l,rid,r['rois'],m['width'],m['height'],method)
            if to_coco(p,m['id'])!=controls[name][m['id']]:raise RuntimeError('Historical control replay differs')
    immutable={str(p):sha256(p) for p in [Path(a['checkpoint']),local,study/'frozen_s2_rois.json',
        study/'B0/coco_predictions.json',study/'B2/coco_predictions.json',source/'S2/raw_candidates.jsonl',
        study/'local_training/A1_seed42/args.yaml',study/'local_training/A1_seed42/results.csv']}
    save(output/'audit.json',dict(global_checkpoint=a['checkpoint'],local_checkpoint=str(local),hashes=immutable,
        frozen_rois_sha256=ROI_SHA,images=548,ROIs=935,control_replay={'E0':'exact','E1':'exact'},
        parameters=PRESET,selection='Frozen manifest lookup ONLY; never recomputed',
        timing='Each E2/E3 independently executes actual disk decode + M0 global640 + A1 local935 + original mapping + source fusion; synchronized GPU. Excludes load/warmup/GT/scoring/serialization/assertions.',
        input_protocol='Historical val-compatible global rectangles/pad.5 (nominal640); native RGB ROI square640 LetterBox; FP32, multi-label .001, stage NMS .7/cap10000, global finalized500, final500.'))


def inference(study,output,device):
    a,gt,images=context(study);proof=load(output/'audit.json');source=Path(a['source_study'])
    old={r['image_id']:r for r in records(source)};frozen={r['image_id']:r['rois'] for r in load(study/'frozen_s2_rois.json')['images']}
    global_detector=CropDetector(a['checkpoint'],device);local_detector=CropDetector(proof['local_checkpoint'],device)
    if [type(m).__name__ for m in global_detector.model.model]!=[type(m).__name__ for m in local_detector.model.model] or global_detector.model.stride.tolist()!=local_detector.model.stride.tolist():raise RuntimeError('Different local architecture')
    shapes=rect_shapes(gt['images'],640);global_detector.warm_shapes(list(shapes.values()));local_detector.warm_shapes([(640,640)])
    canonical={};canonical_e3=[];canonical_e3_per_image={}
    for name,method in [('E2','B0'),('E3','B2')]:
        directory=output/name;directory.mkdir(exist_ok=False);predictions=[];timings=[]
        import json
        with (directory/'raw_candidates.jsonl').open('w',encoding='utf-8') as raw:
            for index,m in enumerate(images):
                global_detector.sync();start=time.perf_counter();image=cv2.imread(m['path']);decoded=time.perf_counter()
                if image is None or image.shape[:2]!=(m['height'],m['width']):raise RuntimeError('Invalid original image')
                g,s=global_detector.infer(image,imgsz=640,global_shape=shapes[m['id']]);g,_=fuse_predictions(g,.7,500)
                rois=frozen[m['id']];pieces=[];stages=[s]
                for roi in rois:
                    lp,ls=local_detector.infer(image,roi);pieces.append(lp);stages.append(ls)
                l=np.concatenate(pieces) if pieces else np.empty((0,6),np.float32);rid=np.repeat(np.arange(len(pieces)),[len(lp) for lp in pieces])
                fusion_start=time.perf_counter();pred,chosen,stats=source_fuse(g,l,rid,rois,m['width'],m['height'],method)
                global_detector.sync();end=time.perf_counter()
                # All checks, GT use and serialization OUTSIDE measured pipeline.
                if not np.array_equal(g,np.asarray(old[m['id']]['global_predictions'],np.float32).reshape(-1,6)):raise RuntimeError('Fresh global output differs from original M0 S2')
                if name=='E2':
                    canonical[m['id']]=[p.copy() for p in pieces]
                    e3,_,_=source_fuse(g,l,rid,rois,m['width'],m['height'],'B2')
                    canonical_e3_per_image[m['id']]=e3;canonical_e3.extend(to_coco(e3,m['id']))
                else:
                    if not all(np.array_equal(p,q) for p,q in zip(pieces,canonical[m['id']])):raise RuntimeError('Timing pass A1 local outputs differ from canonical shared cache')
                    if not np.array_equal(pred,canonical_e3_per_image[m['id']]):raise RuntimeError('E3 differs from shared-cache fusion')
                row=dict(image_id=m['id'],rois=len(rois),forward_calls=1+len(rois),e2e_ms=(end-start)*1000,
                    decode_ms=(decoded-start)*1000,fusion_ms=(end-fusion_start)*1000,
                    GFLOPs_estimate=sum(11.4582016*t['input_shape'][0]*t['input_shape'][1]/640**2 for t in stages),**stats)
                timings.append(row);predictions.extend(to_coco(pred,m['id']))
                raw.write(json.dumps(dict(image_id=m['id'],rois=rois,global_predictions=g.tolist(),local_predictions=[p.tolist() for p in pieces],
                    stages=stages,chosen_candidate_indices=chosen.tolist(),stats=row),separators=(',',':'))+'\n')
                if (index+1)%50==0 or index+1==548:print(f'{name} {index+1}/548 E2E={row["e2e_ms"]:.2f}ms',flush=True)
        save(directory/'coco_predictions.json',predictions);save(directory/'evaluated_image_ids.json',[m['id'] for m in images]);save(directory/'per_image.json',timings)
        values=[r['e2e_ms'] for r in timings]
        save(directory/'timing.json',dict(mean=float(np.mean(values)),P50=float(np.median(values)),P95=float(np.percentile(values,95)),FPS=1000/float(np.mean(values)),
            actual_full_gpu_pipeline=True,mean_ROIs=935/548,forward_calls=1483,mean_GFLOPs_estimate=float(np.mean([r['GFLOPs_estimate'] for r in timings])),
            selection_ms=0.,scope=proof['timing']))
        save(directory/'config.json',dict(experiment=name,fusion=method,parameters=PRESET,global_checkpoint=proof['global_checkpoint'],local_checkpoint=proof['local_checkpoint'],
            weight_and_ROI_hashes=proof['hashes'],protocol=proof['input_protocol'],no_GT_input=True,shared_A1_predictions=True,ROI_count=935,images=548))
        if name=='E3' and predictions!=canonical_e3:raise RuntimeError('Shared A1 E3 predictions mismatch')
    save(output/'inference_verification.json',dict(global_predictions_bit_identical_to_S2=True,A1_local_predictions_bit_identical_across_passes=True,
        E2_E3_share_canonical_local_outputs=True,ROI_sha256=sha256(study/'frozen_s2_rois.json'),selector_calls=0,actual_full_gpu_timing=True))


def score(study,output,names):
    a,gt,_=context(study)
    for name in names:
        evaluate_saved(output/name,gt,Path(a['data_root'])/'val_coco.json',a['original_annotations'])
        predictions=load(output/name/'coco_predictions.json');overall=fixed_conf_quality(gt,predictions)
        save(output/name/'fixed_conf_quality.json',overall)
        annotations=collections.defaultdict(list);detections=collections.defaultdict(list)
        for ann in gt['annotations']:annotations[ann['image_id']].append(ann)
        for p in predictions:detections[p['image_id']].append(p)
        per_image=[]
        for im in gt['images']:
            quality=fixed_conf_quality(dict(images=[im],annotations=annotations[im['id']]),detections[im['id']])
            per_image.append(dict(image_id=im['id'],**quality))
        if any(sum(r[k] for r in per_image)!=overall[k] for k in ['TP','FP','duplicates_proxy','wrong_class_proxy','localization_proxy','background_proxy']):raise RuntimeError('Per-image quality aggregation mismatch')
        save(output/name/'per_image_quality.json',per_image)


def report(study,output):
    from report_crop_study import table
    a,gt,images=context(study);locations={'E0':study/'B0','E1':study/'B2','E2':output/'E2','E3':output/'E3'}
    ev={k:load(p/'evaluation.json') for k,p in locations.items()};q={k:load(p/'fixed_conf_quality.json') for k,p in locations.items()};times={n:load(output/n/'timing.json') for n in ('E2','E3')}
    evidence=load(output/'training_evidence.json');pct=lambda x:f'{100*x:.2f}'
    metric=lambda name,key:ev[name]['ultralytics'][key]
    ap={n:metric(n,'metrics/mAP50-95(B)') for n in ev};ap50={n:metric(n,'metrics/mAP50(B)') for n in ev}
    effects=[('AP95',ap),('AP50',ap50),('APsmall',{n:e['coco_supplemental']['AP_small'] for n,e in ev.items()}),
        ('ARsmall',{n:e['coco_supplemental']['AR_small'] for n,e in ev.items()})]
    interactions={k:dict(A_without_B=vals['E2']-vals['E0'],A_with_B=vals['E3']-vals['E1'],
        B_without_A=vals['E1']-vals['E0'],B_with_A=vals['E3']-vals['E2'],interaction=vals['E3']-vals['E2']-vals['E1']+vals['E0']) for k,vals in effects}
    count_effects={k:dict(E2_minus_E0=q['E2'][k]-q['E0'][k],E3_minus_E1=q['E3'][k]-q['E1'][k],interaction=q['E3'][k]-q['E2'][k]-q['E1'][k]+q['E0'][k]) for k in ['TP','FP']}
    save(output/'comparison.json',dict(evaluations=ev,quality=q,timing=times,interactions=interactions,count_effects=count_effects,training=evidence))
    text='# A1局部微调：冻结S2全局—局部联合评估\n\n'
    text+='原M0全局、A1局部，548张/935个冻结ROI；没有重新计算选区、训练或调参。E0/E1历史逐框复现；全局fresh输出与S2一致；E2/E3局部预测跨完整计时pass逐框一致。\n\n'
    text+=table(['实验','局部模型','边界B2','mAP50%','mAP50:95%','APsmall%','ARsmall%','TP','FP'],[[n,'M0' if n in ('E0','E1') else 'A1',n in ('E1','E3'),pct(ap50[n]),pct(ap[n]),pct(ev[n]['coco_supplemental']['AP_small']),pct(ev[n]['coco_supplemental']['AR_small']),q[n]['TP'],q[n]['FP']] for n in ev])
    text+='\n'+table(['指标','E2−E0 pp','E3−E1 pp','B效应原M0 pp','B效应A1 pp','交互 pp'],[[k]+[f'{100*v[key]:+.3f}' for key in ['A_without_B','A_with_B','B_without_A','B_with_A','interaction']] for k,v in interactions.items()])
    text+='\n'+table(['固定阈值统计','E2−E0','E3−E1','交互'],[[k,v['E2_minus_E0'],v['E3_minus_E1'],v['interaction']] for k,v in count_effects.items()])
    strong=load(Path(a['source_study']).parent/'c0_c1_c2_20261012/C0/evaluation.json')['ultralytics']['metrics/mAP50-95(B)']
    text+=f'\n960整图强基线AP95={pct(strong)}%；E2差值{100*(ap["E2"]-strong):+.3f}pp，E3差值{100*(ap["E3"]-strong):+.3f}pp。这里是单种子固定对照，不作统计显著性声明。\n\n'
    text+='## 实际训练证据\n\n'
    text+=f'args.yaml与30行results.csv确认batch={evidence["args"]["batch"]}，AdamW lr0={evidence["args"]["lr0"]}，Cosine开启，seed42；最佳CSV epoch={evidence["best_epoch_by_csv"]}（一基）。'
    text+='CSV和best.pt内train_results交叉核对；stripped checkpoint的epoch可为-1，不能拿它当best epoch。训练开发集是训练来源ROI，不与当前官方548图联合AP混为一谈。完整字段在training_evidence.json。\n\n'
    text+='## 完整GPU端到端计时\n\n'
    text+=table(['实验','平均ms','P50ms','P95ms','FPS','平均ROI','GFLOPs估计'],[[n,f'{t["mean"]:.2f}',f'{t["P50"]:.2f}',f'{t["P95"]:.2f}',f'{t["FPS"]:.2f}',f'{t["mean_ROIs"]:.4f}',f'{t["mean_GFLOPs_estimate"]:.2f}'] for n,t in times.items()])
    text+='\n每组实际独立执行M0全局548次+A1局部935次，FP32/GPU同步，计入原图读取、预处理、stage NMS、映射、全局finalize、冻结ROI查表、最终融合。排除warmup/model load、JSON写盘、GT/指标计算和一致性断言。严格冻结选区，因此不包含在线ROI选择计算；这与早期包含选区成本的历史69.13/30.29ms口径有差别，不将时间差直接归因于微调。每组一次完整pass，E2先运行、E3后运行，文件缓存/运行顺序可能影响延迟差异，不能只将E3更快归因于B2。GFLOPs为历史未融合BN面积估计，非硬件测量。全局nominal640沿用S2矩形pad.5，实际tensor shape逐图保存。\n\n'
    text+='## 十类AP\n\n'
    for key,label in [('AP50','AP50%'),('AP50_95','AP50:95%')]:
        text+=table(['类别 '+label,*ev],[[p['name']]+[pct(next(x[key] for x in ev[n]['per_class'] if x['class_id']==p['class_id'])) for n in ev] for p in ev['E0']['per_class']])+'\n'
    text+='## 原图尺寸Recall及错误代理\n\n'
    text+=table(['实验','<8%','8–16%','16–32%','≥32%'],[[n]+[pct(e['original_size'][b]['matched']/max(1,e['original_size'][b]['targets'])) for b in ['<8','8-16','16-32','>=32']] for n,e in ev.items()])
    text+='\n'+table(['实验','TP','FP','重复','错类','定位','背景'],[[n]+[v[k] for k in ['TP','FP','duplicates_proxy','wrong_class_proxy','localization_proxy','background_proxy']] for n,v in q.items()])
    text+='\n主AP沿用历史Ultralytics正标签口径，无额外ignore过滤；COCO-small crowd近似单列；固定conf.25/IoU.5统计屏蔽ignore IoA≥.5，四类FP是互斥代理而非确定因果。VisDrone Python参考协议单独保存，没有MATLAB交叉验证。B2原图边缘豁免，q=.25+.75×clip(d/8,0,1)，最终class-aware .7/max500。局部校准低于.001过滤沿用历史B2，不增加规则。\n\n'
    text+='## 结果解读\n\n'
    text+=f'A的独立AP95效应{100*interactions["AP95"]["A_without_B"]:+.3f}pp；在B2下A的效应{100*interactions["AP95"]["A_with_B"]:+.3f}pp。'
    text+=('本轮局部专项训练未体现一致增益，应保留历史对照并分析失败，不能把训练完成等同于有效。' if interactions['AP95']['A_without_B']<=0 or interactions['AP95']['A_with_B']<=0 else '本轮局部专项训练在两种融合条件下均提高AP，但还需重复种子和开发协议检验，不能直接称原创贡献。')
    text+='候选来源与实际ROI/逐图耗时见raw_candidates.jsonl和per_image.json；逐图TP/FP及错误代理见per_image_quality.json；原图尺寸分组/密度诊断见error_diagnosis.json及完整evaluation.json。所有权重/ROI/config/hash在audit.json。\n'
    for p,h in load(output/'audit.json')['hashes'].items():
        if sha256(p)!=h:raise RuntimeError('Immutable input changed')
    if audit_fingerprint(a,images)!=a['immutable']:raise RuntimeError('M0/val changed')
    (output/'REPORT.md').write_text(text,encoding='utf-8')
    save(output/'verification.json',dict(status='complete',images=548,ROI_count=935,ROI_sha256=ROI_SHA,
        unchanged_weights_ROIs_history_and_validation=True,shared_A1_predictions=True,actual_full_GPU_timing=True,
        no_training=True,no_selection_recompute=True,no_parameter_tuning=True,source_sha256=sha256(Path(__file__)),finished=datetime.datetime.now().isoformat()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,default=DEFAULT);p.add_argument('--output',type=Path,default=ROOT/'runs/crop_study/a1_joint_frozen_20261012')
    p.add_argument('--stage',choices=['audit','infer','score','report'],required=True);p.add_argument('--names',nargs='+',choices=['E2','E3'],default=['E2','E3']);p.add_argument('--device',default='0');args=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(2)
    if args.stage=='audit':audit(args.study,args.output)
    elif args.stage=='infer':inference(args.study,args.output,args.device)
    elif args.stage=='score':score(args.study,args.output,args.names)
    else:report(args.study,args.output)


if __name__=='__main__':main()
