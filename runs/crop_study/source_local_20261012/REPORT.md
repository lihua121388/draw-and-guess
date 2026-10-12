# YOLO11n+P2全局—局部协同：来源融合与局部训练准备

全部B实验完成；本阶段没有YOLO正式训练和质量预测器训练。548张ROI、全局M0和评价冻结。B0逐框精确复现S2，历史文件保留。

|实验|方法|mAP50%|mAP50:95%|APsmall%|ARsmall%|TP|FP|固定P%|固定R%|
|---|---|---|---|---|---|---|---|---|---|
|B0|原S2统一NMS|39.70|23.66|16.63|35.18|23465|22037|51.57|60.81|
|B1|仅局部置信度×.85|39.47|23.48|16.21|34.89|22564|18700|54.68|58.48|
|B2|仅内部ROI边界软抑制|40.64|24.39|17.02|35.31|23390|20307|53.53|60.62|
|B3|仅可靠较大全局保护|39.82|23.48|16.69|35.20|23453|21530|52.14|60.78|
|B4|三项联合+统一NMS|40.31|23.83|16.52|35.02|22478|17026|56.90|58.25|
|B_MRU|MRU公开hybrid融合移植|39.67|23.65|16.37|35.06|22872|19778|53.63|59.27|


TP/FP固定conf≥.25、IoU≥.5、类别感知一对一及单独ignore IoA.5；AP使用未额外ignore过滤的历史Ultralytics主口径。不能把固定阈值FP下降单独当成AP提高的证据。

![融合消融](visualizations/fusion_ablation.png)

预固定边界规则B2增加AP95 0.724个百分点，FP减少1730，TP减少75。与960整图强基线24.67%相比仍低0.288个百分点。B1/B3及联合B4没有胜过B2，不能声称叠加必然改善。B2是在预设单次消融中的机制线索，未经独立开发集/多种子确认，不能把验证集赢家直接当成已确定最优部署规则。

## 冻结及融合规则

`frozen_s2_rois.json`：548张、935个ROI、平均1.7062个；SHA256 `659d22fd9f68d1f1639b07a20dc8eabb94222c9363935d8d60606dbaf0f54b01`。从S2记录读取且原函数逐图复算一致；全局已finalize max500，其余候选按ROI stage计数恢复来源，原图xyxy。选区没有改动。

预设在首次评分前写入preset.json：B1局部分数×.85；B2对内部边界最小距离d采用q=.25+.75×clip(d/8,0,1)，局部分数×q，图像边界豁免；B3全局conf≥.50且sqrt(area)≥32的框，屏蔽同类局部IoU>.50冲突；B4按上述顺序联合。最终同类NMS .7、max500，局部校准后<.001丢弃，全局不改分。B3不保证所有可靠全局框免于全局自身NMS/最终top500。所有融合函数不接受GT，也不根据本次验证曲线调参数。

[MRU官方配置](https://github.com/JiajunChen223/MRU-YOLO/blob/main/configs/reproduction/visdrone.yaml)对应额外B_MRU移植：原图局部等效尺寸≤128，局部分数×.9，全局conf≥.25且验证输入坐标等效尺寸≥96，同类IoU≥.5冲突保护。保留本项目ROI/max500/多标签候选，所以不是完整MRU复现；原代码AGPL-3.0。它用于固定ROI下的融合技术对照，不能与原论文不同结构和评价上限的公开AP直接排名。B1—B4均为成熟思想。

## 局部误检诊断

|统计|数量|
|---|---|
|global_TP|18151|
|fused_TP|23465|
|newly_recovered_GT|5363|
|lost_global_GT|49|
|local_retained_TP|16831|
|local_retained_FP|12980|
|global_retained_TP|6634|
|global_retained_FP|9057|


新增GT5363，丢失全局GT49，净TP增加5314。局部最终TP包含替换/重复观察到的已有目标，不能全部称为新增TP。

|实验|重复代理|错误类别代理|定位代理|背景代理|
|---|---|---|---|---|
|B0|5436|6971|4651|4979|
|B1|4719|6137|3909|3935|
|B2|5030|6791|3853|4633|
|B3|5018|6936|4599|4977|
|B4|4185|5984|3175|3682|
|B_MRU|4965|6408|4134|4271|


局部来源进一步分类/边界标志：

|诊断|数量|
|---|---|
|local_raw_highconf|43685|
|same_ROI_sameclass_overlap_gt07|40|
|cross_ROI_sameclass_overlap_ge05|16641|
|same_ROI_class_conflict_overlap_ge05|6841|
|final_local_TP|16831|
|final_local_boundary8_TP|710|
|final_local_clipped2_TP|485|
|final_local_wrong_class|4175|
|final_local_boundary8_wrong_class|261|
|final_local_clipped2_wrong_class|185|
|final_local_localization|2829|
|final_local_boundary8_localization|812|
|final_local_clipped2_localization|788|
|final_local_background|3203|
|final_local_boundary8_background|391|
|final_local_clipped2_background|341|
|final_local_duplicate|2773|
|final_local_boundary8_duplicate|328|
|final_local_clipped2_duplicate|295|


边界标志与错误类型重叠，无法仅凭接近边缘确认真实截断。原始池多标签、跨ROI重复和低分候选很多，不能把全部原始候选FP数量直接当最终检测FP。置信度/类别/原图预测尺寸分布在diagnosis/summary.json；逐候选最大全局同类/任意IoU、对应全局置信度、分差、类别一致性和保留标志在local_features.csv；diagnosis_detail/summary.json给出同/跨ROI重叠及IoU—分差统计。诊断图只供解释，不进入融合。

固定阈值下，距离内部边缘<8px的局部最终框TP=710、FP=1792，精度28.38%；其余局部框精度59.03%。边缘候选质量较低，与B2的改善方向相符，但不能由此确定所有误检的因果机制；没有据此改写8px或缩放参数。

![局部候选错误分布](diagnosis/candidate_errors.png)

![来源关系](diagnosis_detail/source_relations.png)

## 10类AP与原图尺寸Recall

|类别AP50:95%|B0|B1|B2|B3|B4|B_MRU|
|---|---|---|---|---|---|---|
|pedestrian|23.43|22.69|23.61|23.36|22.80|23.00|
|people|14.56|14.38|14.59|14.56|14.39|14.43|
|bicycle|5.02|5.14|5.03|5.01|5.14|5.08|
|car|56.94|56.98|59.57|57.00|58.04|57.22|
|van|30.98|30.69|32.34|30.68|31.38|30.80|
|truck|21.39|21.93|22.34|21.44|22.44|21.95|
|tricycle|14.54|14.53|14.70|14.60|14.82|14.63|
|awning-tricycle|10.13|9.98|10.24|10.14|10.05|10.10|
|bus|39.10|38.16|40.50|37.47|38.58|38.88|
|motor|20.54|20.34|20.95|20.57|20.63|20.44|


|实验|<8 Recall%|8–16 Recall%|16–32 Recall%|≥32 Recall%|
|---|---|---|---|---|
|B0|26.70|47.23|59.79|79.13|
|B1|22.75|42.83|57.56|78.60|
|B2|26.66|47.11|59.66|78.78|
|B3|26.70|47.23|59.72|79.12|
|B4|22.71|42.71|57.39|78.20|
|B_MRU|24.06|44.34|58.31|78.81|


每类AP50同样保存在evaluation.json/per_class。COCO small和VisDrone Python reference为各自单独协议，全部使用当前同一份最终预测；Python reference没有MATLAB交叉校验。

## 计算量与延迟

|实验|平均ROI|实际forward/图|GFLOPs估计|离线融合ms|完整GPU E2Ems|
|---|---|---|---|---|---|
|B0|1.7062|2.7062|26.77|8.62|未重新测量|
|B1|1.7062|2.7062|26.77|7.62|未重新测量|
|B2|1.7062|2.7062|26.77|8.29|未重新测量|
|B3|1.7062|2.7062|26.77|11.36|未重新测量|
|B4|1.7062|2.7062|26.77|10.44|未重新测量|
|B_MRU|1.7062|2.7062|26.77|9.15|未重新测量|


本次严格复用S2原始预测，不重新运行模型；只能实际测CPU融合。B0历史完整GPU流程69.13ms，960历史30.29ms，仅作为成本参考。timing.json内的替换融合估计不是新的E2E实测，不能报为FPS。GFLOPs由固定原始输入张量面积和同结构M0推导，含BN未融合近似；加分/几何运算未算进NN FLOPs。`benchmark_local_factors.py`提供未来完整流程GPU计时接口，本阶段没有执行。

## 局部数据集

仅官方6471训练图。无548验证GT参与训练样本生成。生成16993个640×640 lossless PNG，训练15136、开发1857，标签检查图66张；全部图像/标签哈希、边界和归一化检查通过。

|数据统计|数量|
|---|---|
|train_actual|9579|
|train_images|15136|
|train_empty|468|
|ignore_regions|29077|
|intersecting_targets|495625|
|discarded_boundary_or_tiny|29876|
|retained_partial|11793|
|retained_full|453956|
|retained_final|464424|
|masked_pixels|97500633|
|padded_pixels|31532800|
|train_supplemental|5179|
|discarded_mask_overlap|1325|
|dev_actual|1157|
|dev_images|1857|
|dev_empty|55|
|dev_supplemental|655|
|dev_negative|45|
|train_negative|378|


|split|class_id|目标数|
|---|---|---|
|dev|1|6207|
|dev|2|2658|
|dev|3|990|
|dev|4|20511|
|dev|5|2266|
|dev|6|2654|
|dev|7|559|
|dev|8|148|
|dev|9|658|
|dev|10|2390|
|train|1|105355|
|train|2|36000|
|train|3|12394|
|train|4|168978|
|train|5|30637|
|train|6|14436|
|train|7|6318|
|train|8|4282|
|train|9|7682|
|train|10|39301|


|split|等效尺寸|目标数|
|---|---|---|
|dev|16-32|14051|
|dev|8-16|9025|
|dev|<8|2215|
|dev|>=32|13750|
|train|16-32|160781|
|train|8-16|100692|
|train|<8|25722|
|train|>=32|138188|


实际样本使用冻结M0训练图预测和完全相同S2选区；补充样本单独标注来源，按训练GT补未覆盖极小/稀有/遮挡截断类；每第四源图尝试选零正目标交叠且M0置信度最高的网格负窗，没有合格窗不伪造负样本。原图窗直接裁剪，不缩放，小于640的居中填114，与局部推理padding位置对齐。标签保留条件：原中心在窗内、可见面积≥70%、裁剪边长≥2；保留框坐标裁剪、减ROI偏移并加训练padding。

首批生成副本经一次padding对齐后重新更新PNG/标签/hash，见padding_alignment.json；双边尺寸都不足640的裁剪有1个。此类保持原生像素不放大，而S2推理LetterBox会放大，若计数非零须单独评估此分布差异。

忽略/丢弃残缺目标不能被当成正常背景。现有YOLO没有区域ignore loss mask，本阶段在训练PNG上中性填充这些区域，并移除≥50%被遮挡的正标签到固定点；这是明确的像素屏蔽近似，未改变检测损失。未屏蔽区域原始RGB细节完全保留，验证图和原始图未改。ignore_regions计数是每窗对应源图忽略条目数（包括不交叠者），实际遮挡量由masked_pixels报告；retained_full/partial为mask传播前计数，最终标签以retained_final为准。

按文件首字段源序列划分train/dev，同源所有ROI同组，无组交叠。M0已经训练过官方全部train，因此此dev只隔离局部微调的优化/选checkpoint，并非M0未见数据。最终效果仍在冻结548图上评估，不能用dev来夸大泛化。标签图与样本来源/ROI/hash见local_dataset/checks、samples.jsonl。

![局部数据分布](local_dataset/label_distribution.png)

## 局部微调配置与dry-run

local_train.yaml：M0 best.pt初始化，完全相同25层四尺度结构；30epochs、batch8、workers4、640、AdamW、lr0=1e-4、cosLR、seed42；第一轮关闭mosaic/mixup，只做轻量尺度/翻转/颜色增广，独立local_training目录，不resume全局优化器。全局M0权重文件保持冻结，局部实例可训练。

dry-run使用真实loader、增广、原损失和一次前后向，batch=8、增强后实例=1234，loss有限、梯度有限；没有optimizer.step，没有执行epoch，没有保存训练权重。此批GPU峰值allocated=7.54GiB。该数字不含AdamW状态/完整epoch最大密度峰值，不能当稳定资源上限。建议RTX3090上batch8先监控首epoch；不自动扩大batch或改ROI。

本数据目录占用约9.35GiB；剩余磁盘约42.80GiB。正式微调还需日志/每5轮权重空间。训练总时长尚未实际测量，不凭dry-run外推30轮承诺。

## 后续启动条件及统一接口

A0=原全局M0+原局部M0+B0；A1=原全局M0+专项微调局部M0+B0。当前A1没有正式权重，不能给结果。E0/E1是A0配B0/成熟融合；E2/E3是A1配相同B0/成熟融合；学习质量机制另标Q，尚未训练。配置、dry-run及数据准备完成后，必须等待用户明确授权30轮局部训练，并先接受上述像素屏蔽近似和开发集限制。建议先按单一固定配置运行A1，使用已锁B0/B2/B4分别评价，不因本验证集反复挑参数，再决定是否投入质量器训练。任何正式训练不得覆盖M0或历史S0—S4。

候选质量机制、标签、loss及容量匹配消融详见[设计文档](../../../code/test/COMPLEMENTARY_QUALITY_DESIGN.md)。通用候选重评分、全局关系特征及MRU增量价值都有先例，当前不存在已证明的新颖性或新机制收益。

后续命令和源码见[使用说明](../../../code/test/SOURCE_LOCAL_STUDY.md)。所有B实验预测/每类结果/忽略协议、冻结hash和源文件hash均已保存；正式训练仍停止。
