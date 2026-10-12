"""Assemble reviewable A/B preparation deliverables; no training or tuning."""
import argparse
import collections
import datetime
import json
from pathlib import Path
import random
import shutil
import cv2
import numpy as np
from train_test import ROOT
import torch
from run_source_study import DEFAULT
from run_crop_study import load,save,context,audit_fingerprint
from nscr_experiments import sha256
from report_crop_study import table


def percent(x):return f'{100*x:.2f}'


def make_report(output):
    a,gt,images=context(output);source=Path(a['source_study']);methods=['B0','B1','B2','B3','B4','B_MRU']
    evaluations={m:load(output/m/'evaluation.json') for m in methods};quality={m:load(output/m/'fixed_conf_quality.json') for m in methods}
    times={m:load(output/m/'timing.json') for m in methods};d=output/'local_dataset';dataset=load(d/'statistics.json');verified=load(d/'verification.json');dry=load(output/'local_dry_run.json')
    diagnosis=load(output/'diagnosis/summary.json');details=load(output/'diagnosis_detail/summary.json')
    if (output/'REPORT.md').exists():raise FileExistsError('Completed report must not be overwritten')
    names=dict(B0='原S2统一NMS',B1='仅局部置信度×.85',B2='仅内部ROI边界软抑制',B3='仅可靠较大全局保护',B4='三项联合+统一NMS',B_MRU='MRU公开hybrid融合移植')
    t=load(source/'S2/timing.json');control=load(Path(a['source_study']).parent/'c0_c1_c2_20261012/C0/evaluation.json')
    rows=[]
    for m,e in evaluations.items():
        small=e['coco_supplemental'];q=quality[m]
        rows.append([m,names[m],percent(e['ultralytics']['metrics/mAP50(B)']),percent(e['ultralytics']['metrics/mAP50-95(B)']),
            percent(small['AP_small']),percent(small['AR_small']),q['TP'],q['FP'],percent(q['precision']),percent(q['recall'])])
    # Fixed image IDs from the diagnostic sampling, no choosing successful cases.
    visual=output/'visualizations';visual.mkdir(exist_ok=False)
    grouped={m:collections.defaultdict(list) for m in ('B0','B2')}
    for method in grouped:
        for p in load(output/method/'coco_predictions.json'):grouped[method][p['image_id']].append(p)
    frozen={r['image_id']:r['rois'] for r in load(output/'frozen_s2_rois.json')['images']}
    visual_ids=[images[j]['id'] for j in (0,100,250,400)]
    save(visual/'selection.json',dict(image_ids=visual_ids,rule='Same fixed positions as source diagnostics; no performance filtering'))
    for m in (x for x in images if x['id'] in visual_ids):
        image=cv2.imread(m['path']);panels=[]
        for method in ('B0','B2'):
            panel=image.copy()
            for roi in frozen[m['id']]:cv2.rectangle(panel,tuple(roi[:2]),tuple(roi[2:]),(255,200,0),2)
            for p in grouped[method][m['id']]:
                if p['score']<.25:continue
                x,y,w,h=p['bbox'];cv2.rectangle(panel,(round(x),round(y)),(round(x+w),round(y+h)),(0,255,0),1)
            cv2.putText(panel,method+' conf>=.25',(12,32),cv2.FONT_HERSHEY_SIMPLEX,1.,(0,255,255),2)
            panels.append(cv2.resize(panel,(960,round(960*m['height']/m['width']))))
        cv2.imwrite(str(visual/(Path(m['file_name']).stem+'.jpg')),np.concatenate(panels,axis=1))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].bar(methods,[100*evaluations[m]['ultralytics']['metrics/mAP50-95(B)'] for m in methods]);axes[0].axhline(100*control['ultralytics']['metrics/mAP50-95(B)'],color='red',linestyle='--',label='Whole960 reference');axes[0].set_ylim(22,25.3);axes[0].set_ylabel('AP50:95 (%)');axes[0].legend(fontsize=8)
    axes[1].scatter([quality[m]['FP'] for m in methods],[quality[m]['TP'] for m in methods])
    for m in methods:axes[1].annotate(m,(quality[m]['FP'],quality[m]['TP']),fontsize=8)
    axes[1].set_xlabel('FP at fixed conf .25');axes[1].set_ylabel('TP at fixed conf .25');fig.tight_layout()
    for suffix in ('png','svg','pdf'):fig.savefig(visual/('fusion_ablation.'+suffix),dpi=180)
    plt.close(fig)
    content='# YOLO11n+P2全局—局部协同：来源融合与局部训练准备\n\n'
    content+='全部B实验完成；本阶段没有YOLO正式训练和质量预测器训练。548张ROI、全局M0和评价冻结。B0逐框精确复现S2，历史文件保留。\n\n'
    content+=table(['实验','方法','mAP50%','mAP50:95%','APsmall%','ARsmall%','TP','FP','固定P%','固定R%'],rows)
    content+='\nTP/FP固定conf≥.25、IoU≥.5、类别感知一对一及单独ignore IoA.5；AP使用未额外ignore过滤的历史Ultralytics主口径。不能把固定阈值FP下降单独当成AP提高的证据。\n\n![融合消融](visualizations/fusion_ablation.png)\n\n'
    base=evaluations['B0']['ultralytics']['metrics/mAP50-95(B)'];edge=evaluations['B2']['ultralytics']['metrics/mAP50-95(B)'];strong=control['ultralytics']['metrics/mAP50-95(B)']
    content+=f'预固定边界规则B2增加AP95 {100*(edge-base):.3f}个百分点，FP减少{quality["B0"]["FP"]-quality["B2"]["FP"]}，TP减少{quality["B0"]["TP"]-quality["B2"]["TP"]}。'
    content+=f'与960整图强基线{percent(strong)}%相比仍低{100*(strong-edge):.3f}个百分点。B1/B3及联合B4没有胜过B2，不能声称叠加必然改善。'
    content+='B2是在预设单次消融中的机制线索，未经独立开发集/多种子确认，不能把验证集赢家直接当成已确定最优部署规则。\n\n'
    content+='## 冻结及融合规则\n\n'
    content+=f'`frozen_s2_rois.json`：548张、935个ROI、平均{t["mean_rois"]:.4f}个；SHA256 `{sha256(output/"frozen_s2_rois.json")}`。'
    content+='从S2记录读取且原函数逐图复算一致；全局已finalize max500，其余候选按ROI stage计数恢复来源，原图xyxy。选区没有改动。\n\n'
    content+='预设在首次评分前写入preset.json：B1局部分数×.85；B2对内部边界最小距离d采用q=.25+.75×clip(d/8,0,1)，局部分数×q，图像边界豁免；'
    content+='B3全局conf≥.50且sqrt(area)≥32的框，屏蔽同类局部IoU>.50冲突；B4按上述顺序联合。最终同类NMS .7、max500，局部校准后<.001丢弃，全局不改分。'
    content+='B3不保证所有可靠全局框免于全局自身NMS/最终top500。所有融合函数不接受GT，也不根据本次验证曲线调参数。\n\n'
    content+='[MRU官方配置](https://github.com/JiajunChen223/MRU-YOLO/blob/main/configs/reproduction/visdrone.yaml)对应额外B_MRU移植：原图局部等效尺寸≤128，局部分数×.9，'
    content+='全局conf≥.25且验证输入坐标等效尺寸≥96，同类IoU≥.5冲突保护。保留本项目ROI/max500/多标签候选，所以不是完整MRU复现；原代码AGPL-3.0。'
    content+='它用于固定ROI下的融合技术对照，不能与原论文不同结构和评价上限的公开AP直接排名。B1—B4均为成熟思想。\n\n'
    content+='## 局部误检诊断\n\n'
    s=diagnosis['fixed_conf25_source_counts']
    content+=table(['统计','数量'],[[k,v] for k,v in s.items()])+'\n'
    content+='新增GT5363，丢失全局GT49，净TP增加5314。局部最终TP包含替换/重复观察到的已有目标，不能全部称为新增TP。\n\n'
    content+=table(['实验','重复代理','错误类别代理','定位代理','背景代理'],[[m,quality[m]['duplicates_proxy'],quality[m]['wrong_class_proxy'],quality[m]['localization_proxy'],quality[m]['background_proxy']] for m in methods])
    content+='\n局部来源进一步分类/边界标志：\n\n'+table(['诊断','数量'],[[k,v] for k,v in details['counts'].items()])
    content+='\n边界标志与错误类型重叠，无法仅凭接近边缘确认真实截断。原始池多标签、跨ROI重复和低分候选很多，不能把全部原始候选FP数量直接当最终检测FP。'
    content+='置信度/类别/原图预测尺寸分布在diagnosis/summary.json；逐候选最大全局同类/任意IoU、对应全局置信度、分差、类别一致性和保留标志在local_features.csv；'
    content+='diagnosis_detail/summary.json给出同/跨ROI重叠及IoU—分差统计。诊断图只供解释，不进入融合。\n\n'
    dc=details['counts'];boundary_tp=dc['final_local_boundary8_TP'];boundary_fp=sum(dc['final_local_boundary8_'+k] for k in ('duplicate','wrong_class','localization','background'))
    content+=f'固定阈值下，距离内部边缘<8px的局部最终框TP={boundary_tp}、FP={boundary_fp}，精度{percent(boundary_tp/(boundary_tp+boundary_fp))}%；'
    content+=f'其余局部框精度{percent((s["local_retained_TP"]-boundary_tp)/(s["local_retained_TP"]+s["local_retained_FP"]-boundary_tp-boundary_fp))}%。'
    content+='边缘候选质量较低，与B2的改善方向相符，但不能由此确定所有误检的因果机制；没有据此改写8px或缩放参数。\n\n'
    content+='![局部候选错误分布](diagnosis/candidate_errors.png)\n\n![来源关系](diagnosis_detail/source_relations.png)\n\n'
    content+='## 10类AP与原图尺寸Recall\n\n'
    class_names={p['class_id']:p['name'] for p in evaluations['B0']['per_class']}
    class_rows=[]
    for c in class_names:
        class_rows.append([class_names[c]]+[percent(next(p['AP50_95'] for p in evaluations[m]['per_class'] if p['class_id']==c)) for m in methods])
    content+=table(['类别AP50:95%',*methods],class_rows)+'\n'
    content+=table(['实验','<8 Recall%','8–16 Recall%','16–32 Recall%','≥32 Recall%'],[[m]+[percent(evaluations[m]['original_size'][b]['matched']/max(1,evaluations[m]['original_size'][b]['targets'])) for b in ['<8','8-16','16-32','>=32']] for m in methods])+'\n'
    content+='每类AP50同样保存在evaluation.json/per_class。COCO small和VisDrone Python reference为各自单独协议，全部使用当前同一份最终预测；Python reference没有MATLAB交叉校验。\n\n'
    content+='## 计算量与延迟\n\n'
    content+=table(['实验','平均ROI','实际forward/图','GFLOPs估计','离线融合ms','完整GPU E2Ems'],[[m,f'{t["mean_rois"]:.4f}',f'{t["mean_forward_calls"]:.4f}',f'{t["mean_GFLOPs_actual_shape_estimate"]:.2f}',f'{times[m]["fusion_ms"]["mean"]:.2f}','未重新测量'] for m in methods])+'\n'
    content+='本次严格复用S2原始预测，不重新运行模型；只能实际测CPU融合。B0历史完整GPU流程69.13ms，960历史30.29ms，仅作为成本参考。'
    content+='timing.json内的替换融合估计不是新的E2E实测，不能报为FPS。GFLOPs由固定原始输入张量面积和同结构M0推导，含BN未融合近似；加分/几何运算未算进NN FLOPs。'
    content+='`benchmark_local_factors.py`提供未来完整流程GPU计时接口，本阶段没有执行。\n\n'
    content+='## 局部数据集\n\n'
    content+=f'仅官方6471训练图。无548验证GT参与训练样本生成。生成{verified["crops"]}个640×640 lossless PNG，'
    content+=f'训练{verified["splits"]["train"]}、开发{verified["splits"]["dev"]}，标签检查图{verified["overlays"]}张；全部图像/标签哈希、边界和归一化检查通过。\n\n'
    content+=table(['数据统计','数量'],[[k,v] for k,v in dataset['counts'].items()])+'\n'
    content+=table(['split','class_id','目标数'],[[r['split'],r['class_id'],r['count']] for r in dataset['class_counts']])+'\n'
    content+=table(['split','等效尺寸','目标数'],[[r['split'],r['size'],r['count']] for r in dataset['size_counts']])+'\n'
    content+='实际样本使用冻结M0训练图预测和完全相同S2选区；补充样本单独标注来源，按训练GT补未覆盖极小/稀有/遮挡截断类；每第四源图尝试选零正目标交叠且M0置信度最高的网格负窗，没有合格窗不伪造负样本。'
    content+='原图窗直接裁剪，不缩放，小于640的居中填114，与局部推理padding位置对齐。标签保留条件：原中心在窗内、可见面积≥70%、裁剪边长≥2；保留框坐标裁剪、减ROI偏移并加训练padding。\n\n'
    if (d/'padding_alignment.json').exists():
        pad=load(d/'padding_alignment.json');content+=f'首批生成副本经一次padding对齐后重新更新PNG/标签/hash，见padding_alignment.json；双边尺寸都不足640的裁剪有{pad["both_dimensions_below640"]}个。此类保持原生像素不放大，而S2推理LetterBox会放大，若计数非零须单独评估此分布差异。\n\n'
    content+='忽略/丢弃残缺目标不能被当成正常背景。现有YOLO没有区域ignore loss mask，本阶段在训练PNG上中性填充这些区域，并移除≥50%被遮挡的正标签到固定点；'
    content+='这是明确的像素屏蔽近似，未改变检测损失。未屏蔽区域原始RGB细节完全保留，验证图和原始图未改。'
    content+='ignore_regions计数是每窗对应源图忽略条目数（包括不交叠者），实际遮挡量由masked_pixels报告；retained_full/partial为mask传播前计数，最终标签以retained_final为准。\n\n'
    content+='按文件首字段源序列划分train/dev，同源所有ROI同组，无组交叠。M0已经训练过官方全部train，因此此dev只隔离局部微调的优化/选checkpoint，并非M0未见数据。'
    content+='最终效果仍在冻结548图上评估，不能用dev来夸大泛化。标签图与样本来源/ROI/hash见local_dataset/checks、samples.jsonl。\n\n'
    content+='![局部数据分布](local_dataset/label_distribution.png)\n\n## 局部微调配置与dry-run\n\n'
    content+='local_train.yaml：M0 best.pt初始化，完全相同25层四尺度结构；30epochs、batch8、workers4、640、AdamW、lr0=1e-4、cosLR、seed42；'
    content+='第一轮关闭mosaic/mixup，只做轻量尺度/翻转/颜色增广，独立local_training目录，不resume全局优化器。全局M0权重文件保持冻结，局部实例可训练。\n\n'
    content+=f'dry-run使用真实loader、增广、原损失和一次前后向，batch={dry["batch_size"]}、增强后实例={dry["objects"]}，loss有限、梯度有限；'
    content+=f'没有optimizer.step，没有执行epoch，没有保存训练权重。此批GPU峰值allocated={dry["peak_allocated_GiB"]:.2f}GiB。'
    content+='该数字不含AdamW状态/完整epoch最大密度峰值，不能当稳定资源上限。建议RTX3090上batch8先监控首epoch；不自动扩大batch或改ROI。\n\n'
    content+=f'本数据目录占用约{sum(p.stat().st_size for p in d.rglob("*") if p.is_file())/1024**3:.2f}GiB；剩余磁盘约{shutil.disk_usage(output).free/1024**3:.2f}GiB。正式微调还需日志/每5轮权重空间。训练总时长尚未实际测量，不凭dry-run外推30轮承诺。\n\n'
    content+='## 后续启动条件及统一接口\n\n'
    content+='A0=原全局M0+原局部M0+B0；A1=原全局M0+专项微调局部M0+B0。当前A1没有正式权重，不能给结果。'
    content+='E0/E1是A0配B0/成熟融合；E2/E3是A1配相同B0/成熟融合；学习质量机制另标Q，尚未训练。'
    content+='配置、dry-run及数据准备完成后，必须等待用户明确授权30轮局部训练，并先接受上述像素屏蔽近似和开发集限制。建议先按单一固定配置运行A1，'
    content+='使用已锁B0/B2/B4分别评价，不因本验证集反复挑参数，再决定是否投入质量器训练。任何正式训练不得覆盖M0或历史S0—S4。\n\n'
    content+='候选质量机制、标签、loss及容量匹配消融详见[设计文档](../../../code/test/COMPLEMENTARY_QUALITY_DESIGN.md)。通用候选重评分、全局关系特征及MRU增量价值都有先例，当前不存在已证明的新颖性或新机制收益。\n\n'
    content+='后续命令和源码见[使用说明](../../../code/test/SOURCE_LOCAL_STUDY.md)。所有B实验预测/每类结果/忽略协议、冻结hash和源文件hash均已保存；正式训练仍停止。\n'
    # Cross-source history and all immutable model/val assets are rechecked at delivery.
    for p,h in load(output/'historical_artifacts_sha256.json').items():
        if sha256(p)!=h:raise RuntimeError('Historical result changed')
    if audit_fingerprint(a,images)!=a['immutable']:raise RuntimeError('M0 or official validation changed')
    (output/'REPORT.md').write_text(content,encoding='utf-8')
    files=['source_fusion.py','run_source_study.py','prepare_local_rois.py','train_local_detector.py','evaluate_local_factors.py',
        'benchmark_local_factors.py','run_mru_fusion_control.py','analyze_source_detail.py','report_source_local.py','test_source_local.py','local_dataset_checks.py','align_local_padding.py']
    save(output/'verification.json',dict(status='complete',historical_predictions_unchanged=True,M0_and_validation_unchanged=True,
        frozen_rois_sha256=sha256(output/'frozen_s2_rois.json'),B0_exact_S2=True,fusion_methods=methods,
        dataset_crops=verified['crops'],dataset_verified=True,dry_run='passed',formal_training_started=False,quality_predictor_trained=False,
        source_sha256={p:sha256(ROOT/'code/test'/p) for p in files},finished=datetime.datetime.now().isoformat()))
    print('Reviewable report complete; training remains unauthorized',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT);args=p.parse_args();torch.set_num_threads(4);make_report(args.output)
