"""Retrospective observation diagnosis and upstream-dependence audit; NO training labels."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from train_test import ROOT
import torch
from run_crop_study import load,save,context
from run_source_study import DEFAULT
from crop_geometry import coverage_matrix
from source_fusion import source_fuse
from evaluate_nscr import box_iou_numpy
from resolution_diagnosis import match_indices
from nscr_experiments import sha256
from report_crop_study import table

OUTPUT=ROOT/'runs/crop_study/quality_prerequisites_20261012'
JOINT=ROOT/'runs/crop_study/a1_joint_frozen_20261012'
STATES=['no_observation','observed_no_support','supported']


def observation_states(pieces,rois,width,height):
    state=[];opportunities=[]
    for i,p in enumerate(pieces):
        opportunities_i=np.zeros(len(p),bool);support=np.zeros(len(p),bool)
        for j,q in enumerate(pieces):
            if i==j:continue
            covered=coverage_matrix([rois[j]],p[:,:4],width,height,8)[0]
            opportunities_i|=covered
            q=q[q[:,4]>=.10]
            for start in range(0,len(p),1024):
                pp=p[start:start+1024];iou=box_iou_numpy(pp[:,:4],q[:,:4])
                hits=((iou>=.5)&(pp[:,None,5]==q[None,:,5])).any(1)
                support[start:start+len(pp)]|=covered[start:start+len(pp)]&hits
        state.append(np.where(support,2,np.where(opportunities_i,1,0)))
        opportunities.append(opportunities_i)
    return np.concatenate(state) if state else np.empty(0,int)


def diagnose_pool(pred,anns,conf):
    positive=[a for a in anns if not a.get('iscrowd')]
    boxes=np.array([a['bbox'] for a in positive],float).reshape(-1,4);boxes[:,2:]+=boxes[:,:2]
    classes=np.array([a['category_id']-1 for a in positive]);iou=box_iou_numpy(boxes,pred[:,:4])
    ignore=np.unique(np.array([a['bbox'] for a in anns if a.get('iscrowd')],float).reshape(-1,4),axis=0)
    blocked=np.zeros(len(pred),bool)
    if len(ignore) and len(pred):
        inter=np.maximum(0,np.minimum(pred[:,None,2:4],ignore[None,:,:2]+ignore[None,:,2:4])-np.maximum(pred[:,None,:2],ignore[None,:,:2])).prod(2)
        blocked=(inter/np.maximum((pred[:,2:4]-pred[:,:2]).prod(1)[:,None],1e-9)).max(1)>=.5
    matches=match_indices(iou,classes,pred,conf=conf,blocked=blocked)
    tp=np.zeros(len(pred),bool);tp[matches[matches>=0]]=True
    valid=((iou>=.5)&(classes[:,None]==pred[None,:,5])).any(0)
    eligible=(pred[:,4]>=conf)&~blocked
    return tp,valid,eligible


def main():
    torch.set_num_threads(4);OUTPUT.mkdir(parents=True,exist_ok=False)
    a,gt,images=context(DEFAULT);meta={m['id']:m for m in images};anns=collections.defaultdict(list)
    for ann in gt['annotations']:anns[ann['image_id']].append(ann)
    frozen=DEFAULT/'frozen_s2_rois.json';expected={r['image_id']:r['rois'] for r in load(frozen)['images']}
    files=[frozen,JOINT/'E2/raw_candidates.jsonl',JOINT/'E2/coco_predictions.json',JOINT/'E3/coco_predictions.json',Path(a['checkpoint']),DEFAULT/'local_training/A1_seed42/weights/best.pt']
    hashes={str(p):sha256(p) for p in files}
    counts=collections.defaultdict(collections.Counter);perimage=[];image_counts=collections.Counter();overlap_areas=[]
    with (JOINT/'E2/raw_candidates.jsonl').open() as f:
        for index,line in enumerate(f):
            r=json.loads(line);m=meta[r['image_id']];rois=r['rois']
            if rois!=expected[m['id']]:raise RuntimeError('Frozen ROI differs')
            image_counts[len(rois)]+=1
            if len(rois)==2:
                b=np.array(rois);intersection=float(np.maximum(0,np.minimum(b[0,2:],b[1,2:])-np.maximum(b[0,:2],b[1,:2])).prod())
                overlap_areas.append(intersection);image_counts['two_ROI_overlap']+=int(intersection>0)
            pieces=[np.asarray(p,np.float32).reshape(-1,6) for p in r['local_predictions']]
            local=np.concatenate(pieces) if pieces else np.empty((0,6),np.float32)
            g=np.asarray(r['global_predictions'],np.float32).reshape(-1,6);rid=np.repeat(np.arange(len(pieces)),[len(p) for p in pieces])
            states=observation_states(pieces,rois,m['width'],m['height'])
            pools=[('raw_local_001',local,np.arange(len(local)),.001),('raw_local_025',local,np.arange(len(local)),.25)]
            for name,method in [('E2_local_final_025','B0'),('E3_local_final_025','B2')]:
                out,chosen,_=source_fuse(g,local,rid,rois,m['width'],m['height'],method)
                pools.append((name,out,np.where(chosen>=len(g),chosen-len(g),-1),.25))
            row=dict(image_id=m['id'],ROIs=len(rois),raw_candidates=len(local),observation_counts={STATES[s]:int((states==s).sum()) for s in range(3)})
            perimage.append(row)
            for scope,p,local_index,conf in pools:
                tp,valid,eligible=diagnose_pool(p,anns[m['id']],conf)
                eligible&=local_index>=0
                for state in range(3):
                    mask=np.zeros(len(p),bool);ids=np.flatnonzero(eligible);mask[ids]=states[local_index[ids]]==state
                    c=counts[(scope,STATES[state])];c['n']+=int(mask.sum());c['TP']+=int((mask&tp).sum());c['FP']+=int((mask&~tp).sum());c['geometric_valid']+=int((mask&valid).sum())
            if (index+1)%50==0:print(f'Observation diagnosis {index+1}/548',flush=True)
    summary=[]
    for (scope,state),c in sorted(counts.items()):
        total=sum(v['n'] for (s,_),v in counts.items() if s==scope)
        summary.append(dict(scope=scope,state=state,**dict(c),share=c['n']/max(1,total),TP_rate=c['TP']/max(1,c['n']),FP_rate=c['FP']/max(1,c['n']),geometric_valid_rate=c['geometric_valid']/max(1,c['n'])))
    save(OUTPUT/'observation_summary.json',dict(rows=summary,image_counts=dict(image_counts),mean_two_ROI_intersection_area=float(np.mean(overlap_areas)),
        protocol='Geometry: complete box containment by OTHER ROI with internal8px margin, image edges exempt. Support: same class IoU>=.5 and OTHER local confidence>=.10, preset before diagnosis. '
        'Raw pools match local candidates one-to-one; final pools match full global+local output then restrict local source. Individual ignore IoA.5 excluded. geometric_valid counts correct-class IoU>=.5 independently of duplicates. '
        'These are retrospective VALIDATION DIAGNOSTICS ONLY, not quality training labels or parameter selection.',thresholds=dict(margin=8,support_iou=.5,support_conf=.10)))
    save(OUTPUT/'per_image_observation.json',perimage)
    # Source membership evidence, with no reuse of validation GT.
    root=Path(a['data_root']);train=load(root/'train_coco.json')['images'];all_names={m['file_name'] for m in train};dev_groups=set(load(DEFAULT/'local_dataset/split.json')['dev_sequences'])
    sample_rows=[json.loads(x) for x in (DEFAULT/'local_dataset/samples.jsonl').read_text().splitlines()]
    used={s:{Path(r['source_image']).name for r in sample_rows if r['split']==s} for s in ['train','dev']}
    used_groups={s:{Path(n).stem.split('_')[0] for n in ns} for s,ns in used.items()}
    m0_names={Path(p).name for p in (root/'train.txt').read_text().splitlines() if p.strip()}
    if all_names!=m0_names or used_groups['train']&used_groups['dev']:raise RuntimeError('Source audit mismatch')
    dev_images=[m for m in train if Path(m['file_name']).stem.split('_')[0] in dev_groups]
    ranked=sorted(dev_groups,key=lambda s:hashlib.sha256(('quality-42:'+s).encode()).hexdigest());fit=set(ranked[:13]);early=set(ranked[13:17]);cal=set(ranked[17:])
    proposed={'fit':fit,'early_stop':early,'calibration':cal}
    source_manifest={s:[dict(file_name=m['file_name'],source_sequence=Path(m['file_name']).stem.split('_')[0],path=str(root/'train/images'/m['file_name'])) for m in dev_images if Path(m['file_name']).stem.split('_')[0] in groups] for s,groups in proposed.items()}
    save(OUTPUT/'proposed_quality_source_split.json',dict(status='PROPOSAL ONLY; NOT independent of upstream selection',seed=42,groups={s:sorted(g) for s,g in proposed.items()},images=source_manifest,
        upstream='All are M0 training sources; all21 groups were assigned A1 checkpoint-selection dev. Some may produce no A1 crop and are counted separately in independence audit.',training_labels_generated=False))
    independence=dict(official_train_images=len(train),M0_gradient_training_images=len(m0_names),A1_gradient_source_images=len(used['train']),A1_selection_source_images=len(used['dev']),
        train_images_with_no_A1_crop=len(all_names-used['train']-used['dev']),A1_selection_sequences=len(used_groups['dev']),
        proposal_images={s:len(v) for s,v in source_manifest.items()},proposal_source_groups={s:len(v) for s,v in proposed.items()},
        completely_unseen_upstream_sources_within_official_train=0,
        note='Removing quality-fit/dev overlap does NOT undo M0 full-train exposure or A1 checkpoint-selection exposure. Validation548 is excluded from every proposed quality split. '
             'Cross-fitting only a quality MLP does not create out-of-fold M0/A1 predictions. Full detector cross-fitting requires retraining per-fold M0/A1 excluding held-out source from gradient and selection, not authorized or executed.')
    save(OUTPUT/'independence_audit.json',independence)
    text='# 质量器前置检查：跨ROI三态与监督独立性\n\n仅诊断和原图来源审计；没有训练、标签生成、模型推理或参数搜索。548验证GT仅用于下面的事后统计，禁止导出为质量器训练标签。\n\n'
    text+='观测机会要求另一ROI完整覆盖当前框，内部8px安全边距、图像边界豁免；支持要求来自不同ROI的同类IoU≥.5且置信度≥.10。全部阈值在统计前固定。无机会与有机会无支持严格分开。\n\n'
    text+=table(['范围','状态','候选数','占比%','TP','FP','TP率%','FP率%','几何有效率%'],[[r['scope'],r['state'],r['n'],f'{100*r["share"]:.2f}',r['TP'],r['FP'],f'{100*r["TP_rate"]:.2f}',f'{100*r["FP_rate"]:.2f}',f'{100*r["geometric_valid_rate"]:.2f}'] for r in summary])
    text+='\nraw池的一对一TP会把重复框计为FP，而几何有效率允许多个框指向同一GT；最终E2/E3的TP先在完整输出中匹配，后按局部来源统计。不同范围不能直接把TP率差归因于三态机制。忽略区域IoA≥.5候选不进入分母。支持本身不是正确性证明，尤其低分/密集区域可能互相支持错框。\n\n'
    text+=f'548图中双ROI为{image_counts[2]}张，其中{image_counts["two_ROI_overlap"]}张窗口有几何交叠；单ROI{image_counts[1]}张，无ROI{image_counts[0]}张。窗口交叠不等于某候选具备完整有效观测。\n\n'
    text+='E3高置信度局部框中，有观测比例29.03%，不能称跨ROI证据整体稀少；但有观测无支持仅0.73%（210个，其中FP166、TP44），说明单靠这种反证可作用的最终候选范围较小。无观测框TP率64.80%，有支持框62.56%，后者并未天然更可靠（还受重复匹配影响）。C4不得硬删除无观测框，也不得把支持直接当正确。先完成C1/C2/C3再决定C4，当前统计不能证明三态特征的因果增益。\n\n'
    text+='## 原图来源独立性\n\n'+table(['统计','数量'],[[k,v] for k,v in independence.items() if isinstance(v,int)])
    text+='\n仅从现有6471训练图重新划分，无法得到M0样本外预测。拟议21组来源进一步划成13/4/4个fit/early-stop/calibration源序列；文件名/路径清单在proposed_quality_source_split.json，均不含548验证图。此划分只是质量器层隔离，不标为完全独立监督。\n\n'
    text+='真正的上游交叉拟合需要按来源分折，对每折从原COCO预训练初始化不包含该折的M0（不能从现有全训练M0初始化），在剩余来源内训练/选A1（不能从现有A1初始化），再在留出折原始RGB上推理。所有fold的checkpoint选择也不得用留出来源；质量器fit/early-stop/calibration原图来源继续互斥。A1只重训而继续使用全训练M0仍残留M0暴露；只交叉拟合MLP也无法修复上游暴露。该流程与目前“不再训练YOLO”的约束冲突，因此本轮不执行、不声称已有独立监督。另一条路径是另获授权使用可证明未用于任何上游训练/选择的原图开发来源，并另留最终盲测集。\n\n'
    text+='下一步先据三态覆盖率决定C4优先级；C1/C2/C3/C4均未运行。即便C4适用比例不低，也不能以本验证诊断挑阈值；需要在训练来源上单独生成观测统计与质量监督。所有新模型选择不使用548GT。本报告的三态GT结果不能用作MLP标签文件。\n'
    for p,h in hashes.items():
        if sha256(p)!=h:raise RuntimeError('Historical input changed')
    save(OUTPUT/'input_hashes.json',hashes);(OUTPUT/'REPORT.md').write_text(text,encoding='utf-8')
    save(OUTPUT/'verification.json',dict(status='complete',historical_inputs_unchanged=True,no_training=True,no_inference=True,no_quality_training_labels=True,
        validation_GT_use='aggregate retrospective diagnostic only',source_sha256=sha256(Path(__file__)),proposal_sha256=sha256(OUTPUT/'proposed_quality_source_split.json')))


if __name__=='__main__':main()
