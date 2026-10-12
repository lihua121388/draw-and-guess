"""S0-S4 results with a strict common640 control and separately labelled GT Oracle diagnostics."""
import collections
import json
from pathlib import Path
import cv2
import numpy as np
from train_test import ROOT
import matplotlib.pyplot as plt
from run_crop_study import context,load,save
from report_crop_study import table,pct,coverage_summary
from crop_evaluate import fixed_conf_quality

NAMES={'S0':'640整图','S1':'640整图+预测ROI≤1','S2':'640整图+预测ROI≤2',
       'S3':'640整图+GT Oracle Top-1','S4':'640整图+GT Oracle Top-2'}


def verify_common_globals(output):
    def records(name):
        with (output/name/'raw_candidates.jsonl').open(encoding='utf-8') as f:
            return {r['image_id']:r for r in (json.loads(line) for line in f)}
    base=records('S0');checks={};regions={}
    for n in NAMES:
        data=base if n=='S0' else records(n)
        if set(data)!=set(base) or len(data)!=548:raise RuntimeError('Different image membership')
        mismatch=sum(r['global_predictions']!=base[k]['global_predictions'] for k,r in data.items())
        checks[n]=dict(images=len(data),different_global_predictions=mismatch)
        if mismatch:raise RuntimeError(f'{n} global outputs differ from S0; inspect before comparison')
        regions[n]={k:r['rois'] for k,r in data.items()}
    for first,second in [('S1','S2'),('S3','S4')]:
        if any(regions[first][i]!=regions[second][i][:len(regions[first][i])] for i in base):
            raise RuntimeError('Budgets not nested prefixes')
    save(output/'common_global_verification.json',dict(status='passed',groups=checks,
        nested_predicted_budgets=True,nested_oracle_budgets=True,
        note='Each scheme actually re-executed global inference; all saved final global predictions are bit-identical to S0.'))


def figures(output,results,coverage,timing):
    d=output/'figures';d.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ns,label,style in [(['S0','S1','S2'],'Prediction-only','o-'),(['S0','S3','S4'],'GT Oracle ONLY','x--')]:
        axs[0].plot([0,1,2],[100*results[n]['ultralytics']['metrics/mAP50-95(B)'] for n in ns],style,label=label)
        axs[1].plot([0,1,2],[timing[n]['e2e_ms']['mean'] for n in ns],style,label=label)
    axs[0].set(xlabel='Maximum added ROI budget',ylabel='mAP50:95 (%)',xticks=[0,1,2]);axs[0].legend()
    axs[1].set(xlabel='Maximum added ROI budget',ylabel='Measured image end-to-end latency (ms)',xticks=[0,1,2]);axs[1].legend()
    fig.suptitle('Same whole640 M0; GT Oracle is diagnostic, not deployable accuracy');fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(d/f'budget_accuracy_cost.{ext}',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for names,label,style in [(['S1','S2'],'Prediction-only','o-'),(['S3','S4'],'GT Oracle ONLY','x--')]:
        for ax,key in zip(axs,('small','<8')):
            ax.plot([1,2],[100*coverage[n][key]['coverage'] for n in names],style,label=label)
            ax.set(xlabel='Maximum ROI budget',ylabel=f'{key} full GT coverage (%)',ylim=(0,100),xticks=[1,2]);ax.legend()
    fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(d/f'coverage.{ext}',dpi=180)
    plt.close(fig)


def visuals(output,images):
    d=output/'visualizations';d.mkdir(exist_ok=True)
    rng=np.random.default_rng(42);chosen=rng.choice(len(images),4,replace=False).tolist()
    predictions={};regions={}
    for n in NAMES:
        pp=collections.defaultdict(list)
        for p in load(output/n/'coco_predictions.json'):pp[p['image_id']].append(p)
        predictions[n]=pp;regions[n]={r['image_id']:r['rois'] for r in load(output/n/'coverage_per_image.json')}
    selection=[]
    for index in chosen:
        m=images[index];source=cv2.imread(m['path']);scale=min(1.,1000/m['width'])
        base=cv2.resize(source,(round(m['width']*scale),round(m['height']*scale)))
        panels=[]
        for mode,title in [('original','Original RGB'),('S0','S0 whole640'),('S1','S1 whole640 + predicted ROI<=1'),
                           ('S2','S2 whole640 + predicted ROI<=2'),('S3','S3 GT ORACLE Top-1 ONLY'),('S4','S4 GT ORACLE Top-2 ONLY')]:
            panel=base.copy()
            if mode!='original':
                for p in predictions[mode][m['id']]:
                    if p['score']<.25:continue
                    b=np.array(p['bbox']);b[2:]+=b[:2];xy=np.round(b*scale).astype(int)
                    color=tuple(int(v) for v in np.random.default_rng(p['category_id']).integers(50,240,3))
                    cv2.rectangle(panel,tuple(xy[:2]),tuple(xy[2:]),color,1)
                for i,r in enumerate(regions[mode][m['id']]):
                    xy=np.round(np.array(r)*scale).astype(int)
                    cv2.rectangle(panel,tuple(xy[:2]),tuple(xy[2:]),(0,0,255) if mode in ('S3','S4') else (0,180,255),2)
                    cv2.putText(panel,f'ROI{i+1}',tuple(xy[:2]+[3,18]),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
            panel=cv2.copyMakeBorder(panel,35,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
            cv2.putText(panel,title,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,180) if mode in ('S3','S4') else (0,0,0),2)
            panels.append(panel)
        collage=np.concatenate([np.concatenate(panels[:3],axis=1),np.concatenate(panels[3:],axis=1)],axis=0)
        cv2.imwrite(str(d/(Path(m['file_name']).stem+'.jpg')),collage);selection.append(m)
    save(d/'selection.json',dict(seed=42,rule='4 images sampled without replacement BEFORE any visual success/failure filtering',images=selection))


def make_report(output):
    output=Path(output)
    if (output/'REPORT.md').exists():raise FileExistsError('Completed report exists; preserve old experiment')
    a,gt,images=context(output);verify_common_globals(output)
    results={n:load(output/n/'evaluation.json') for n in NAMES}
    timing={n:load(output/n/'timing.json') for n in NAMES}
    coverage={n:coverage_summary(load(output/n/'coverage_per_image.json')) for n in NAMES}
    quality={n:fixed_conf_quality(gt,load(output/n/'coco_predictions.json')) for n in NAMES}
    save(output/'comparison.json',dict(results=results,timing=timing,coverage=coverage,quality=quality))
    base=results['S0']['ultralytics']['metrics/mAP50-95(B)'];base_ms=timing['S0']['e2e_ms']['mean']
    headers=['实验','方案','mAP50%','mAP50:95%','相对S0 pp','AP-small%','AR-small%']
    def accuracy(n):
        e=results[n];u=e['ultralytics'];c=e['coco_supplemental']
        return [n,NAMES[n],pct(u['metrics/mAP50(B)']),pct(u['metrics/mAP50-95(B)']),
            f'{100*(u["metrics/mAP50-95(B)"]-base):+.2f}',pct(c['AP_small']),pct(c['AR_small'])]
    lines=['# S0–S4：640整图与有限ROI预算实验\n',
        '所有组使用同一M0-640最佳权重、同一548张验证图像、同一38587个有效GT、FP32和统一评分。仅推理，没有训练/结构修改。\n',
        '## 选区与后处理预设\n',
        'S1/S2依据S0最终预测，保留conf>=0.10、原图sqrt(wh)<32的框；选区信号做类无关IoU0.7去重以避免多类别候选重复计权。'
        '候选来自固定20%网格和预测框中心/角点；贪心新增完整覆盖置信度总和最大、再按新增框数和坐标打破平局。'
        '窗口原图640，内部边界安全距离8px，预算最多1/2；无新增信号时提前终止，因此实际ROI可能少于预算。'
        '这些参数在测试前固定，没有通过验证集GT调ROI位置、阈值或窗口。\n',
        'S3/S4使用有效GT几何选区，最大新增完整覆盖的小目标数量，最多1/2。仅作GT Oracle潜力诊断，不作为可部署成绩，也不是数学最优AP上界。\n',
        '全局分支为val兼容640（历史batch8矩形形状，逐图batch1计时）。阶段一conf0.001、多类别候选、NMS0.7、候选上限30000、中间上限10000。'
        '各组先形成与S0完全相同的最终全局max500输出，用于选区与融合；局部来自原图RGB，640输入，最终全图类别感知NMS0.7/max500。'
        '选区信号去重不修改检测输出，GT不参与S0/S1/S2任何预测保留或置信度修正。\n',
        '主AP为统一离线Ultralytics匹配/AP积分；small为补充COCO原图面积<=32²协议；VisDrone Python参考协议单独报告，非官方MATLAB运行。'
        '历史small来自predict而主AP来自val，本次二者都使用同一份保存预测，不混用历史口径。S0复现与逐图共同全局输出核验已通过。\n',
        '## 可部署方案主结果\n',table(headers,[accuracy(n) for n in ('S0','S1','S2')]),
        '## GT Oracle诊断，单独报告\n',table(headers,[accuracy(n) for n in ('S3','S4')]),
        '## 计算成本与逐图实际时延\n']
    lines.append(table(['实验','类型','实际ROI/图','前向/图','总前向次数','方形GFLOPs/图','矩形GFLOPs估计','平均ms','P50ms','P95ms','时延/S0'],
        [[n,'GT Oracle' if n in ('S3','S4') else '预测',f'{timing[n]["mean_rois"]:.3f}',f'{timing[n]["mean_forward_calls"]:.3f}',
          timing[n]['forward_calls'],f'{timing[n]["mean_GFLOPs_square_estimate"]:.2f}',f'{timing[n]["mean_GFLOPs_actual_shape_estimate"]:.2f}',
          f'{timing[n]["e2e_ms"]["mean"]:.2f}',f'{timing[n]["e2e_ms"]["p50"]:.2f}',f'{timing[n]["e2e_ms"]["p95"]:.2f}',
          f'{timing[n]["e2e_ms"]["mean"]/base_ms:.2f}'] for n in NAMES]))
    lines.append('每方案实际独立运行，不复用缓存前向合成时延。包含读取/解码、全局预处理/前向/NMS、选区、原图裁剪、局部前向、坐标恢复和最终融合；'
        '不含模型加载、预热、JSON写入和评分。预测选区时间与Oracle几何选择时间均计入端到端。矩形GFLOPs按实际张量面积估计。\n')
    lines.append(table(['实验','解码ms','选区ms','裁剪ms','预处理ms','前向ms','阶段一NMSms','映射ms','全局定稿+融合ms'],
        [[n]+[f'{timing[n][k+"_ms"]["mean"]:.2f}' for k in ('decode','selection','crop','preprocess','forward','roi_nms','map','merge')] for n in NAMES]))
    lines.append('## 固定原图尺寸Recall（conf0.25、同类一对一IoU0.5）\n')
    lines.append(table(['实验','<8%','8–16%','16–32%','≥32%'],[[n]+[pct(results[n]['original_size'][k]['recall']) for k in ('<8','8-16','16-32','>=32')] for n in NAMES]))
    lines.append('## ROI完整覆盖与面积\n')
    lines.append(table(['实验','全部GT%','small%','<8%','8–16%','16–32%','累加面积/原图','面积并集/原图'],
        [[n]+[pct(coverage[n][k]['coverage']) for k in ('all','small','<8','8-16','16-32')]+[
            f'{coverage[n]["crop_area_ratio"]:.3f}',f'{coverage[n]["unique_crop_area_ratio"]:.3f}'] for n in NAMES]))
    lines.append('GT覆盖只在推理结束后计算，不能用来决定S1/S2窗口或筛选预测。覆盖按目标计数加权，面积按图平均。\n')
    lines.append('## 固定阈值误检诊断\n')
    lines.append(table(['实验','Precision%','Recall%','TP','FP','重复代理','错类代理','定位代理','背景代理'],
        [[n,pct(quality[n]['precision']),pct(quality[n]['recall'])]+[quality[n][k] for k in ('TP','FP','duplicates_proxy','wrong_class_proxy','localization_proxy','background_proxy')] for n in NAMES]))
    lines.append('上述FP分类是事后代理统计，不是确定的因果诊断，不改变任何预测；固定阈值Precision/Recall与AP完整置信度曲线不同。\n')
    lines.append('## 十类AP（AP50 / AP50:95，%）\n')
    lines.append(table(['类别']+list(NAMES),[[results['S0']['per_class'][i]['name']]+[
        f'{pct(results[n]["per_class"][i]["AP50"])} / {pct(results[n]["per_class"][i]["AP50_95"])}' for n in NAMES] for i in range(10)]))
    lines.append('## VisDrone Python参考（非MATLAB成绩）\n')
    lines.append(table(['实验','参考AP50%','参考AP50:95%','macro AP50:95%'],[[n,pct(results[n]['visdrone_reference_python']['AP50']),
        pct(results[n]['visdrone_reference_python']['AP50_95']),pct(results[n]['visdrone_reference_python']['macro_AP50_95'])] for n in NAMES]))
    lines.append('## 收益与下一阶段判断\n')
    for n in ('S1','S2','S3','S4'):
        delta=100*(results[n]['ultralytics']['metrics/mAP50-95(B)']-base)
        ratio=timing[n]['e2e_ms']['mean']/base_ms
        lines.append(f'{n}：相对S0主AP {delta:+.2f}pp，平均端到端时延{ratio:.2f}倍，small完整覆盖率{pct(coverage[n]["small"]["coverage"])}%。\n')
    diff=100*(results['S2']['ultralytics']['metrics/mAP50-95(B)']-results['S1']['ultralytics']['metrics/mAP50-95(B)'])
    extra=timing['S2']['e2e_ms']['mean']-timing['S1']['e2e_ms']['mean']
    lines.append(f'增加预测ROI预算1→2的边际收益为{diff:+.2f}pp，额外平均时延{extra:+.2f}ms。\n')
    for real,oracle in [('S1','S3'),('S2','S4')]:
        gap=100*(results[oracle]['ultralytics']['metrics/mAP50-95(B)']-results[real]['ultralytics']['metrics/mAP50-95(B)'])
        lines.append(f'同预算 {real} vs {oracle}：AP差值（GT Oracle−预测选区）为{gap:+.2f}pp；GT greedy覆盖不是AP最优选区，不应视为严格可部署性能上界。\n')
    ref=load(Path(a['source_study'])/'C0/evaluation.json');rt=load(Path(a['source_study'])/'C0/timing.json')
    refap=ref['ultralytics']['metrics/mAP50-95(B)'];refms=rt['e2e_ms']['mean']
    lines.append(f'此前同权重960整图参考为{100*refap:.2f}%、平均{refms:.2f}ms（不同时间测量，只作成本参考）。\n')
    deploy=['S1','S2'];best=max(deploy,key=lambda n:results[n]['ultralytics']['metrics/mAP50-95(B)'])
    bestap=results[best]['ultralytics']['metrics/mAP50-95(B)'];bestms=timing[best]['e2e_ms']['mean']
    if bestap<=refap and bestms>=refms:
        lines.append('即使相对640基准存在收益，当前最佳预测选区方案仍未在精度或平均时延上优于已有960整图。'
            '本轮不建议直接训练ROI前置网络，优先保留960整图；需先证明ROI预算在实际部署限制下有相称收益。\n')
    elif bestap-base>=.005:
        lines.append('预测选区相对640显示可复现收益，可考虑进一步验证在目标部署预算下的价值；'
            '先做无GT独立测试和延迟复测，再决定是否需要训练选区网络。本任务未启动任何训练。\n')
    else:
        lines.append('当前预测选区相对640的整体收益不足，不建议直接训练前置网络；Oracle结果仅作诊断，不能作为部署性能。\n')
    lines.append('## 图表与固定seed可视化\n\n![预算精度与成本](figures/budget_accuracy_cost.png)\n\n![覆盖率](figures/coverage.png)\n')
    figures(output,results,coverage,timing);visuals(output,images)
    for f in sorted((output/'visualizations').glob('*.jpg')):lines.append(f'![{f.stem}](visualizations/{f.name})\n')
    lines.append('## 完整复现\n\n```bat\nC:\\Users\\FXQJDYTH\\anaconda3\\envs\\yolo\\python.exe -u code\\test\\run_crop_budget.py --output runs\\crop_study\\s0_s4_new_repeat --stage all\n```\n\n'
        '必须使用新目录。可按audit/infer/score/report分阶段运行；score可用--schemes选择独立组。'
        '每组保存原图坐标coco_predictions.json、raw_candidates.jsonl、ROI和选区依据、coverage_per_image.json、timing_per_image.csv及全评价JSON。'
        'common_global_verification.json核验五组全局预测逐图完全相同和两个预算的嵌套性；verification.json校验权重及数据未变化。\n')
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('S report complete:',output/'REPORT.md',flush=True)
