"""Export full C0/C1/C2 tables, scientific plots and seeded success/failure visualizations."""
import collections
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from train_test import ROOT
import matplotlib.pyplot as plt
from crop_geometry import coverage_matrix
from resolution_diagnosis import spacing, match_indices
from evaluate_nscr import box_iou_numpy
from crop_evaluate import fixed_conf_quality

NAMES={'C0':'960 whole image','C1-A':'fixed tiles only','C1-B':'960 + fixed tiles','C1-C':'640 + fixed tiles',
       **{f'C2-{b}-K{k}':f'GT Oracle {k} ROI'+(' + 960' if b=='B' else '') for b in ('A','B') for k in (1,2,4)}}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def coverage_summary(records):
    out={}
    for key in ('all','small','<8','8-16','16-32'):
        total=sum(r['groups'][key]['total'] for r in records)
        hits=sum(r['groups'][key]['covered'] for r in records)
        out[key]=dict(total=total,covered=hits,coverage=hits/total if total else None)
    out.update(mean_rois=float(np.mean([len(r['rois']) for r in records])),
        crop_area_ratio=float(np.mean([r['crop_area_ratio'] for r in records])),
        unique_crop_area_ratio=float(np.mean([r['unique_crop_area_ratio'] for r in records])))
    return out


def pct(x):
    return f'{100*x:.2f}' if x is not None else 'N/A'


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
                     ['|'+'|'.join(map(str,r))+'|' for r in rows])+'\n\n'


def plots(output,results,coverage):
    figures=output/'figures';figures.mkdir(exist_ok=True)
    budgets=[1,2,4]
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for key,label in [('small','sqrt(wh)<32'),('<8','sqrt(wh)<8'),('8-16','8-16'),('16-32','16-32')]:
        axs[0].plot(budgets,[100*coverage[f'C2-B-K{k}'][key]['coverage'] for k in budgets],marker='o',label=label)
    axs[0].set(xlabel='Maximum GT Oracle ROI budget',ylabel='Full GT coverage (%)',ylim=(0,100),xticks=budgets)
    axs[0].legend(fontsize=8)
    for b in ('A','B'):
        axs[1].plot(budgets,[100*results[f'C2-{b}-K{k}']['ultralytics']['metrics/mAP50-95(B)'] for k in budgets],marker='o',label=f'GT Oracle {b}')
    axs[1].axhline(100*results['C0']['ultralytics']['metrics/mAP50-95(B)'],color='black',ls='--',label='C0 960')
    axs[1].set(xlabel='Maximum GT Oracle ROI budget',ylabel='mAP50:95 (%)',xticks=budgets)
    axs[1].legend(fontsize=8)
    fig.suptitle('GT Oracle ONLY: diagnostic potential, not deployable system accuracy')
    fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(figures/f'oracle_budget.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for name,e in results.items():
        latency=read(output/name/'timing.json')['e2e_ms']['mean']
        ap=100*e['ultralytics']['metrics/mAP50-95(B)']
        ax.scatter(latency,ap,marker='x' if name.startswith('C2') else 'o')
        ax.annotate(name,(latency,ap),fontsize=8,xytext=(3,3),textcoords='offset points')
    ax.set(xlabel='Measured image end-to-end latency (ms)',ylabel='mAP50:95 (%)',title='Circles: deployable; crosses: GT Oracle diagnostic')
    fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(figures/f'accuracy_cost.{ext}',dpi=180)
    plt.close(fig)


def visualizations(output,gt,manifest):
    directory=output/'visualizations';directory.mkdir(exist_ok=True)
    anns=collections.defaultdict(list)
    for a in gt['annotations']:
        if not a.get('iscrowd'):anns[a['image_id']].append(a)
    predictions={}
    for name in ('C0','C1-A','C1-B','C2-B-K4'):
        by=collections.defaultdict(list)
        for p in read(output/name/'coco_predictions.json'):by[p['image_id']].append(p)
        predictions[name]=by
    fixed={r['image_id']:r['rois'] for r in read(output/'C1-B/coverage_per_image.json')}
    oracle={r['image_id']:r['rois'] for r in read(output/'C2-B-K4/coverage_per_image.json')}
    rng=np.random.default_rng(42)
    order=rng.permutation(len(manifest)).tolist()
    descriptions=[]
    for idx in order:
        m=manifest[idx];a=anns[m['id']]
        boxes=np.array([x['bbox'] for x in a],dtype=float).reshape(-1,4)
        sizes=np.sqrt(boxes[:,2:].prod(1));boxes[:,2:]+=boxes[:,:2]
        q=spacing(boxes)
        tiny=sizes<16
        values={};matched={}
        for name in ('C0','C1-A','C1-B'):
            pp=np.array([[*p['bbox'],p['score'],p['category_id']] for p in predictions[name][m['id']]],dtype=float).reshape(-1,6)
            pp[:,2:4]+=pp[:,:2]
            matches=match_indices(box_iou_numpy(boxes,pp[:,:4]),np.array([x['category_id'] for x in a]),pp)
            matched[name]=matches>=0
            values[name]=int((tiny&(matches>=0)).sum())
        crosses=False
        for r in fixed[m['id']]:
            intersect=((boxes[:,:2]<r[2:]).all(1)&(boxes[:,2:]>r[:2]).all(1))
            contained=coverage_matrix([r],boxes,m['width'],m['height'],0)[0]
            crosses|=bool((tiny&intersect&~contained).any())
        missed=matched['C0']&~matched['C1-A']
        descriptions.append(dict(image=m,dense=int((tiny&(q<2)).sum()),sparse=int((tiny&(q>=2)).sum()),
            mixed=bool(tiny.any() and (sizes>=32).any()),failure=bool(missed.any()),boundary=crosses,
            GT_boxes_missed_by_tiles=boxes[missed].tolist(),tiny_hits=values))
    selected=[];used=set()
    # Pick first eligible image in ONE fixed random ordering; no hand-picked visual success.
    for label,key,predicate in [('tiny_dense','dense',lambda v:v>=20),('tiny_sparse','sparse',lambda v:v>=5),
                                ('mixed_sizes','mixed',bool),('crop_failure','failure',bool),('ROI_boundary','boundary',bool)]:
        eligible=next((d for d in descriptions if d['image']['id'] not in used and predicate(d[key])),None)
        if eligible is None:
            selected.append(dict(case=label,status='no eligible image under declared rule'));continue
        used.add(eligible['image']['id'])
        selected.append(dict(case=label,**eligible))
        m=eligible['image'];image=cv2.imread(m['path']);scale=min(1.,1000/m['width'])
        base=cv2.resize(image,(round(m['width']*scale),round(m['height']*scale)))
        panels=[]
        for title,mode in [('Original RGB','raw'),('Fixed 640 ROIs','fixed'),('GT ORACLE ROIs (K<=4)','oracle'),
                           ('C0: whole960','C0'),('C1-B: whole960 + tiles','C1-B'),('C2-B-K4: GT ORACLE ONLY','C2-B-K4')]:
            panel=base.copy()
            if mode in ('fixed','oracle'):
                for i,r in enumerate((fixed if mode=='fixed' else oracle)[m['id']]):
                    xy=np.round(np.array(r)*scale).astype(int)
                    cv2.rectangle(panel,tuple(xy[:2]),tuple(xy[2:]),(0,180,255),2)
                    cv2.putText(panel,str(i+1),tuple(xy[:2]+[4,18]),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,100,255),2)
                if mode=='fixed' and label=='crop_failure':
                    title='Fixed ROIs + separate GT diagnostic'
                    for b in eligible['GT_boxes_missed_by_tiles']:
                        xy=np.round(np.array(b)*scale).astype(int)
                        cv2.rectangle(panel,tuple(xy[:2]),tuple(xy[2:]),(0,0,255),3)
                        cv2.putText(panel,'GT missed by tiles',tuple(xy[:2]+[0,-3]),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
            elif mode!='raw':
                for p in predictions[mode][m['id']]:
                    if p['score']<.25:continue
                    b=np.array(p['bbox']);b[2:]+=b[:2];xy=np.round(b*scale).astype(int)
                    cid=p['category_id'];color=tuple(int(c) for c in np.random.default_rng(cid).integers(50,240,3))
                    cv2.rectangle(panel,tuple(xy[:2]),tuple(xy[2:]),color,1)
            panel=cv2.copyMakeBorder(panel,35,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
            cv2.putText(panel,title,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,180) if 'ORACLE' in title else (0,0,0),2)
            panels.append(panel)
        collage=np.concatenate([np.concatenate(panels[:3],axis=1),np.concatenate(panels[3:],axis=1)],axis=0)
        collage=cv2.copyMakeBorder(collage,0,60,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        for i,c in enumerate(gt['categories']):
            color=tuple(int(v) for v in np.random.default_rng(c['id']).integers(50,240,3))
            xpos=10+(i%5)*(collage.shape[1]//5);ypos=collage.shape[0]-35+(i//5)*25
            cv2.putText(collage,c['name'],(xpos,ypos),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
        cv2.imwrite(str(directory/f'{label}_{Path(m["file_name"]).stem}.jpg'),collage)
    (directory/'selection.json').write_text(json.dumps(dict(seed=42,selection_rule='First eligible in seeded permutation; crop failure = valid GT detected by C0 but missed by C1-A, conf.25/IoU.5; '
        'boundary uses GT crossing fixed ROI edge, solely retrospective visualization; GT never enters C0/C1 selection/fusion.',cases=selected),indent=2),encoding='utf-8')


def make_report(output):
    output=Path(output)
    if (output/'REPORT.md').exists():
        raise FileExistsError('Completed report exists; preserve existing experiment outputs')
    audit=read(output/'audit.json')
    results={n:read(output/n/'evaluation.json') for n in NAMES}
    timing={n:read(output/n/'timing.json') for n in NAMES}
    coverage={n:coverage_summary(read(output/n/'coverage_per_image.json')) for n in NAMES}
    stats={}
    for n in NAMES:
        with (output/n/'timing_per_image.csv').open(encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        stats[n]={key:sum(int(r[key]) for r in rows) for key in ['candidates','after_nms','final','final_cap_removed',
            'stage1_max_nms_removed','stage1_intermediate_removed','boundary_candidates','boundary_high_conf']}
    (output/'comparison.json').write_text(json.dumps(dict(results=results,timing=timing,coverage=coverage,truncation=stats),indent=2),encoding='utf-8')
    lines=['# VisDrone C0/C1/C2 原图裁剪可行性实验\n',
        '同一份 M0-640 权重、548 张验证图像、38587 个有效目标。未训练、未修改模型/损失/依赖或原始图像标注。\n',
        '主结果均从原图坐标预测离线评分。阶段一 conf=0.001、多类别候选、NMS=0.7；主跨 ROI 融合 NMS=0.7，'
        '最终每图最多500。阶段一最大候选30000、中间输出10000；任何截断均记录。固定 ROI640、重叠20%、原图贴边。\n',
        '## C0 复现与历史口径修正\n',
        '历史主 AP 来自 val，多类别候选与 batch8 矩形形状；历史 small 指标来自 predict，其保存预测重新评分主 AP 为24.14%。'
        '本次使用 val 一致的缩放/多类别候选，顺序 batch1 执行，主 AP 与 small 均对应同一份输出。'
        '不能将历史主 AP 与另一套预测的 small 指标拼接为本次结果。差异与容差详见 C0/reproduction.json。\n',
        '补充 small 指标使用原图面积<=32²的 COCO 协议，忽略区域近似 classwise crowds；'
        '原图尺寸 Recall 使用 sqrt(wh)、conf>=0.25、同类一对一 IoU>=0.5，保持历史离线忽略规则。'
        'VisDrone Python 参考协议单独报告，尚未与 MATLAB 逐项对齐，不能称官方成绩。\n',
        '历史C0（val主指标 / predict small）：41.19 / 24.68 / 15.40 / 28.24；'
        '本次统一C0指标以下表为准。small差异来自候选输出与预处理变化，并非模型改进。\n',
        '## 表1：可部署方案精度（%）\n']
    headers=['实验','方案','mAP50','mAP50:95','AP-small','AR-small']
    def acc_row(n):
        e=results[n];u=e['ultralytics'];c=e['coco_supplemental']
        return [n,NAMES[n],pct(u['metrics/mAP50(B)']),pct(u['metrics/mAP50-95(B)']),pct(c['AP_small']),pct(c['AR_small'])]
    lines.append(table(headers,[acc_row(n) for n in ('C0','C1-A','C1-B','C1-C')]))
    lines.append('## GT Oracle：仅作潜力诊断，不参与真实系统排名\n')
    lines.append('使用验证集 GT 几何位置选 ROI；不把 GT 写入预测、不修正预测类别/置信度。候选来自固定网格及小目标中心/角点，'
        '贪心最大新增完整覆盖 sqrt(wh)<32 的目标，内部边缘安全距离8px；无新增小目标时提前终止。'
        '这不是数学最优上界，也不是可部署成绩。\n')
    lines.append(table(headers,[acc_row(n) for n in NAMES if n.startswith('C2')]))
    lines.append('## 表2：实际精度与计算成本\n')
    rows=[]
    for n in NAMES:
        t=timing[n]
        rows.append([n,f'{t["mean_rois"]:.2f}',f'{t["mean_forward_calls"]:.2f}',
            f'{t["mean_GFLOPs_square_estimate"]:.2f}',f'{t["mean_GFLOPs_actual_shape_estimate"]:.2f}',
            f'{t["e2e_ms"]["mean"]:.2f}',f'{t["e2e_ms"]["p50"]:.2f}',f'{t["e2e_ms"]["p95"]:.2f}',
            pct(results[n]['ultralytics']['metrics/mAP50-95(B)'])])
    lines.append(table(['实验','ROI/图','前向次数/图','方形GFLOPs/图','实际矩形GFLOPs估计','平均ms','P50ms','P95ms','mAP50:95%'],rows))
    lines.append('每方案均实际独立执行整张原图完整流程，不以缓存前向时间相加充当端到端延迟。'
        '包括文件读取/解码、ROI选择/裁剪、预处理、GPU前向、阶段一NMS、映射和最终NMS；'
        '不含模型加载、预热、JSON写入、离线评分。Oracle GT几何选区耗时包含在端到端时间中。'
        '原图尺寸不同，全局 val 矩形还包含 stride32/pad.5；GFLOPs 分别给出方形预算和实际张量形状估计。'
        'CPU/GPU同步计时会有系统负载噪声，本次单次548图不能证明毫秒级差异稳定。\n')
    lines.append(table(['实验','解码ms','选区ms','裁剪ms','预处理ms','前向ms','ROI内NMSms','映射ms','最终融合ms'],
        [[n]+[f'{timing[n][k+"_ms"]["mean"]:.2f}' for k in ('decode','selection','crop','preprocess','forward','roi_nms','map','merge')] for n in NAMES]))
    lines.append('## 表3：固定原图尺寸分组 Recall（%）\n')
    lines.append(table(['实验','<8','8–16','16–32','≥32'],[[n]+[pct(results[n]['original_size'][k]['recall']) for k in ('<8','8-16','16-32','>=32')] for n in NAMES]))
    lines.append('## 表4：Oracle区域覆盖与效率\n')
    rows=[];base=results['C0']['ultralytics']['metrics/mAP50-95(B)'];prev_ap=base;prev_roi=0
    for k in (1,2,4):
        n=f'C2-B-K{k}';c=coverage[n];ap=results[n]['ultralytics']['metrics/mAP50-95(B)']
        marginal=100*(ap-prev_ap)/(c['mean_rois']-prev_roi) if c['mean_rois']>prev_roi else None
        rows.append([k,f'{c["mean_rois"]:.2f}']+[pct(c[g]['coverage']) for g in ('all','small','<8','8-16','16-32')]+
            [f'{c["crop_area_ratio"]:.3f}',f'{c["unique_crop_area_ratio"]:.3f}',f'{100*(ap-base):+.2f}',f'{marginal:+.3f}' if marginal is not None else 'N/A'])
        prev_ap,prev_roi=ap,c['mean_rois']
    lines.append(table(['最多ROI','实际ROI/图','全部GT%','small%','<8%','8–16%','16–32%','累加面积/原图','并集面积/原图','AP增益pp','每新增ROI边际pp'],rows))
    lines.append('覆盖率按全数据目标计数加权；面积比和ROI数按图平均。累加面积可以超过1，显示重复计算；并集面积最多1。\n')
    lines.append('![Oracle预算与覆盖/AP](figures/oracle_budget.png)\n\n![精度与端到端代价](figures/accuracy_cost.png)\n')
    lines.append('## 候选截断、边界和重复计算\n')
    lines.append(table(['实验','融合输入候选','跨ROI NMS后','最终保留','最终500截断','阶段一30000截断','阶段一10000截断','边界候选','边界高置信候选'],
                       [[n]+list(stats[n].values()) for n in NAMES]))
    lines.append('边界候选指触及内部 ROI 边缘2px的输出，可能含真实截断目标和伪框；仅计数，不依据 GT 抑制任何预测。'
        '跨 ROI NMS 可能同时去除重复框与密集邻近真实框，不能把其去除数全部当成重复目标。\n')
    lines.append('## NMS敏感性（固定预设0.5/0.6/0.7，不选最佳值替换主结果）\n')
    sens=[]
    for n in ('C0','C1-A','C1-B','C1-C'):
        for th in (.5,.6,.7):
            e=results[n] if th==.7 else read(output/'sensitivity'/f'{n}_iou{th}'/'evaluation.json')
            sens.append([n,th,pct(e['ultralytics']['metrics/mAP50-95(B)']),pct(e['coco_supplemental']['AP_small'])])
    lines.append(table(['实验','融合IoU','mAP50:95%','AP-small%'],sens))
    lines.append('同阈值对照：'+ '；'.join(f'IoU={th}时，C1-B相对C0为 '
        f'{100*(read(output/"sensitivity"/f"C1-B_iou{th}"/"evaluation.json")["ultralytics"]["metrics/mAP50-95(B)"]-read(output/"sensitivity"/f"C0_iou{th}"/"evaluation.json")["ultralytics"]["metrics/mAP50-95(B)"]):+.2f}pp' for th in (.5,.6))+'。\n')
    lines.append('未运行可选的0%重叠/960原图ROI窗口扩展；优先完成预设640窗口20%重叠全量实验，不根据验证集结果反复调窗口。\n')
    lines.append('## 10类 AP（AP50 / AP50:95，%）\n')
    lines.append(table(['类别']+list(NAMES),[[results['C0']['per_class'][i]['name']]+[
        f'{pct(results[n]["per_class"][i]["AP50"])} / {pct(results[n]["per_class"][i]["AP50_95"])}' for n in NAMES] for i in range(10)]))
    lines.append('## VisDrone Python参考协议（独立口径，非MATLAB运行）\n')
    lines.append(table(['实验','参考AP50%','参考AP50:95%','macro AP50:95%'],[[n,pct(results[n]['visdrone_reference_python']['AP50']),
        pct(results[n]['visdrone_reference_python']['AP50_95']),pct(results[n]['visdrone_reference_python']['macro_AP50_95'])] for n in NAMES]))
    b=results['C1-B']['ultralytics']['metrics/mAP50-95(B)'];c=results['C1-C']['ultralytics']['metrics/mAP50-95(B)']
    legacy1280=read(Path(audit['historical960']).parent/'R2_1280/evaluation.json')['ultralytics']['metrics/mAP50-95(B)']
    c1cost=timing['C1-B']['e2e_ms']['mean']/timing['C0']['e2e_ms']['mean']
    fixedcov=coverage['C1-B'];dup=fixedcov['crop_area_ratio']/fixedcov['unique_crop_area_ratio']
    lines.append('## 结果分析与下一阶段建议\n')
    lines.append(f'C1-B 相对 C0 为 {100*(b-base):+.2f}pp，'+('超过' if b>base else '未超过')+
        f'C0；相对历史1280参考25.22%为 {100*(b-legacy1280):+.2f}pp。平均端到端延迟是C0的 {c1cost:.2f}倍。'
        f'C1-C 相对C0为 {100*(c-base):+.2f}pp，平均延迟为C0的 {timing["C1-C"]["e2e_ms"]["mean"]/timing["C0"]["e2e_ms"]["mean"]:.2f}倍，'
        '需同时比较精度和成本，不能因全局输入较小就称总体低成本。\n')
    lines.append(f'固定切片平均每图 {fixedcov["mean_rois"]:.2f} 个，累加面积是原图的 {fixedcov["crop_area_ratio"]:.2f}倍，'
        f'累加面积/覆盖并集为 {dup:.2f}，存在明确重叠计算。Oracle4小目标完整覆盖率为 '
        f'{pct(coverage["C2-B-K4"]["small"]["coverage"])}%，极小目标<8为{pct(coverage["C2-B-K4"]["<8"]["coverage"])}%。'
        '覆盖不等于可检出，预算上限和零新增收益早停也会使实际ROI少于K。\n')
    for k in ('<8','8-16','16-32','>=32'):
        gain=results['C1-B']['original_size'][k]['recall']-results['C0']['original_size'][k]['recall']
        lines.append(f'原图尺寸{k}：C1-B Recall变化 {100*gain:+.2f}pp。\n')
    oracle_ap={k:results[f'C2-B-K{k}']['ultralytics']['metrics/mAP50-95(B)'] for k in (1,2,4)}
    worthwhile=[k for k in (1,2,4) if oracle_ap[k]-base>=.005]
    lines.append('本轮用至少+0.5pp主AP作为探索性可行性筛选，不能把它当成统计显著性或论文创新判据。\n')
    if worthwhile:
        target_budget=worthwhile[0];target=coverage[f'C2-B-K{target_budget}']
        lines.append(f'达到该筛选收益的最小Oracle预算是K={target_budget}，对应small完整覆盖 '
            f'{pct(target["small"]["coverage"])}%、<8覆盖{pct(target["<8"]["coverage"])}%、8–16覆盖{pct(target["8-16"]["coverage"])}%。'
            '如果研究真实选区器，可将该预算及覆盖量级作为首个几何目标；只有无GT预测AP与端到端时延也达标才有部署意义，'
            '覆盖率本身不足以保证收益。\n')
    else:
        lines.append('K=1/2/4均未取得+0.5pp的主AP收益，因此目前无法给出支持训练前置网络的覆盖率门槛。\n')
    if b-base>=.005 and max(oracle_ap[k]-base for k in (1,2))>=.005:
        advice='局部精检有继续研究依据，但固定全图切片的额外成本需要压缩。建议下一阶段先定义最多1–2个ROI的部署预算和可接受延迟，'
        advice+='用训练集产生选区监督，做无GT的区域选择验证；不把本次Oracle当成部署性能。'
    else:
        advice='当前证据不足以直接投入ROI前置网络训练。先保持现有系统，避免新增选区网络；若继续，应先用廉价无GT选区方法验证能否接近预算Oracle收益，'
        advice+='并复测端到端成本，而不是继续增加注意力或无预算裁剪。'
    lines.append(advice+'\n')
    if c1cost>2:
        lines.append(f'当前固定全图切片消耗{c1cost:.2f}倍端到端时间，默认仍建议保留960整图，'
            '本次固定切片不作为整体精度升级方案。若业务只重极小目标召回，需另行定义误检容忍度和时延预算；'
            '不能以当前结果宣称额外成本已取得整体性能收益。\n')
    if oracle_ap[4]<=oracle_ap[2]:
        lines.append('2→4 ROI没有提高整体AP，说明增加GT覆盖并不保证预测融合收益；边界框、密集误抑制及误检需进一步分析。\n')
    else:
        lines.append(f'2→4 ROI额外AP收益为{100*(oracle_ap[4]-oracle_ap[2]):.2f}pp，结合表4的边际收益判断递减程度。\n')
    lines.append('前置预测器所需覆盖率不能仅由GT几何Oracle唯一确定，因为候选分布、误选ROI与误检会影响精度。'
        '可把取得可接受收益的最小Oracle预算对应 small/<8覆盖作为必要的研发参考目标，但不能保证达到相同覆盖就复现Oracle AP。\n')
    gt=read(Path(audit['data_root'])/'val_coco.json')
    quality={n:fixed_conf_quality(gt,read(output/n/'coco_predictions.json')) for n in ('C0','C1-A','C1-B','C1-C','C2-B-K1','C2-B-K2','C2-B-K4')}
    (output/'fixed_conf_quality.json').write_text(json.dumps(quality,indent=2),encoding='utf-8')
    lines.append('## 为什么召回提高而整体AP下降\n')
    lines.append(table(['实验','固定阈值Precision%','Recall%','TP','FP','重复代理','错类代理','定位代理','背景代理'],
        [[n,pct(q['precision']),pct(q['recall'])]+[q[k] for k in ('TP','FP','duplicates_proxy','wrong_class_proxy','localization_proxy','background_proxy')] for n,q in quality.items()]))
    lines.append('以上固定conf=0.25、IoU=0.5，同一逐图一对一匹配，与AP完整置信度曲线不同。'
        '重复代理指额外预测与同类GT的IoU>=0.5但未获一对一匹配；错类代理指IoU>=0.5的异类框；'
        '定位代理指剩余同类框IoU>=0.1；背景代理指不满足前述条件。它们只用于事后分析，不过滤检测输出，不能直接解释全部AP变化。\n')
    lines.append(f'C1-B比C0增加 {quality["C1-B"]["TP"]-quality["C0"]["TP"]} 个TP，同时增加 '
        f'{quality["C1-B"]["FP"]-quality["C0"]["FP"]} 个FP；需同时评估召回与误检代价，不能仅凭小目标Recall上升宣称有效。\n')
    lines.append('## 可视化与复现\n\n可视化按seed42固定随机顺序，分别取第一个满足极小密集、极小稀疏、混合大小、裁剪失败、边界截断条件的图像。'
        '同色同坐标尺度；Oracle有明显标签，GT未混入预测。选择规则和原始预测完整保存。\n')
    plots(output,results,coverage)
    manifest=read(output/'image_manifest.json');gt=read(Path(audit['data_root'])/'val_coco.json')
    visualizations(output,gt,manifest)
    for f in sorted((output/'visualizations').glob('*.jpg')):
        lines.append(f'![{f.stem}](visualizations/{f.name})\n')
    lines.append('```bat\nC:\\Users\\FXQJDYTH\\anaconda3\\envs\\yolo\\python.exe -u code\\test\\run_crop_study.py --output runs\\crop_study\\new_repeat --stage all\n```\n\n'
        '必须使用新输出目录。可按audit/c0/smoke/c1/c2/score/sensitivity/report分阶段执行；已有推理组拒绝覆盖。'
        '原始预测为各组coco_predictions.json；中间候选/坐标恢复/两级NMS见raw_candidates.jsonl；'
        '每图时间和ROI数见timing_per_image.csv；Oracle几何覆盖见coverage_per_image.json。\n')
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Report complete:',output/'REPORT.md',flush=True)
