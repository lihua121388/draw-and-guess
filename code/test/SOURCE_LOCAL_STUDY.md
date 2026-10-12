# 固定S2：融合消融和局部训练准备

输出目录：`runs/crop_study/source_local_20261012`。当前只执行工程审计、ROI冻结、离线B消融/诊断、训练图ROI生成、数据检查和dry-run，未启动正式训练。最终状态以该目录`verification.json`为准。

## 当前融合结果

|实验|mAP50%|mAP50:95%|AP-small%|固定TP|固定FP|
|---|---:|---:|---:|---:|---:|
|B0 原S2|39.70|23.66|16.63|23465|22037|
|B1 仅预设局部缩放|39.47|23.48|16.21|22564|18700|
|B2 仅ROI边界质量|40.64|24.39|17.02|23390|20307|
|B3 仅可靠全局保护|39.82|23.48|16.69|23453|21530|
|B4 成熟来源联合|40.31|23.83|16.52|22478|17026|
|MRU hybrid融合移植|39.67|23.65|16.37|22872|19778|

B2相对S2 AP95+0.724个百分点，FP−1730、TP−75。960整图仍为41.21/24.67%，历史完整E2E30.29ms；S2历史69.13ms。此次不重新运行模型，B耗时只有实际离线CPU融合，各组**没有新的GPU E2E实测值**。不能把cached时长标成真实FPS。参数先固定、后评分，未用验证GT调参。

所有ROI坐标冻结在`frozen_s2_rois.json`；B0逐框复现原S2，平均1.7062ROI/图。主AP、COCO small、带ignore的固定Recall/误检和VisDrone Python reference分列，口径与前期一致。B_MRU是同M0/同ROI/同max500的融合移植，**不是完整MRU模型复现**。成熟思想来源与候选原创性边界见`COMPLEMENTARY_QUALITY_DESIGN.md`。

## 源码接口

- `source_fusion.py`：无GT融合；来源恢复；边界距离/全局关系特征。
- `run_source_study.py`：独立新目录审计/冻结、B0—B4离线重融合和评分、诊断。
- `run_mru_fusion_control.py`：公开MRU数值规则的额外固定ROI融合对照；当前路径指向本研究目录。
- `analyze_source_detail.py`：局部最终FP分类及同/跨ROI关系诊断；当前路径指向本研究目录。
- `prepare_local_rois.py`：6471官方train，预测实际ROI、GT补充、真正空背景/困难负窗；从不调用训练。
- `local_dataset_checks.py`：seed42额外随机30标签图和类别/尺寸曲线。
- `train_local_detector.py`：配置、dry-run和后续显式正式训练入口。
- `evaluate_local_factors.py`：统一E0/E1和后续E2/E3评价。
- `benchmark_local_factors.py`：未来完整GPU流程计时，当前未执行。

数据采用lossless PNG，原图裁剪，不做缩小，尺寸不足640时居中填114；标签加相同padding偏移。若双边均不足640，则仍保持原生像素，与S2推理上采样有差异，数量在padding_alignment.json单列。标签保留中心在ROI、可见比例≥.70、边长≥2px的目标。原图忽略区域及被舍弃的残缺目标在训练副本中中性填114，避免直接监督为背景；与这些mask重叠≥50%的正框也移除并屏蔽。现有损失不支持真正区域ignore mask，所以这是披露的像素屏蔽近似；未mask部分原RGB保留。正式授权前需要接受此训练策略，或者另授权改造loss支持ignore mask。

源序列按文件首字段分组划分train/dev，不能同源跨组。M0已看过官方全部train，因此dev只用于局部训练/质量器优化隔离，不是未见数据的独立检测泛化测试。最终评估548官方val；验证GT不做训练裁剪，不做融合调参。

## 已准备的训练方案

配置`runs/crop_study/source_local_20261012/local_train.yaml`：M0 best.pt初始化，同样25层P2/P3/P4/P5结构；30epochs、batch8、workers4、imgsz640、AdamW、lr0=1e-4、AMP、seed42。关闭mosaic/mixup，轻量增强。局部模型全参数微调，全局M0文件和全局推理保持冻结。没有新Backbone/Neck模块。

dry-run检查真实loader/增强/原损失的一次forward/backward；没有optimizer.step，没有epoch循环。`local_dry_run.json`记录finite loss/gradient、模型结构、M0文件hash和单批峰值显存。它不是正式训练结果，也不能推断30epochs收益或完整资源峰值。

## 运行命令（仓库当前终端）

现有已完成输出不能重复覆盖。以下只读/单批验证可运行：

```bat
C:\Users\FXQJDYTH\anaconda3\envs\yolo\python.exe code\test\test_source_local.py
C:\Users\FXQJDYTH\anaconda3\envs\yolo\python.exe code\test\train_local_detector.py --mode dry-run
```

**下面的正式训练命令仅在用户明确授权30轮微调后执行，本阶段没有执行：**

```bat
C:\Users\FXQJDYTH\anaconda3\envs\yolo\python.exe -u code\test\train_local_detector.py --mode train --execute-training --device 0
```

默认输出`runs/crop_study/source_local_20261012/local_training/A1_seed42`，目录存在时拒绝启动，防止覆盖。改配置后必须重新dry-run，权重SHA不一致拒绝执行。先检查GPU空闲和磁盘空间；实际显存/每轮时间需正式首轮监控，当前不承诺估计。

E0/E1无需训练可由统一接口复现，输出必须是新目录：

```bat
C:\Users\FXQJDYTH\anaconda3\envs\yolo\python.exe -u code\test\evaluate_local_factors.py --output runs\crop_study\ab_fixed_s2_new --fusion B4
```

后续已授权训练并产生A1最佳权重后，E2/E3依旧使用同ROI。以下路径是**未来产物**，当前不存在：

```bat
C:\Users\FXQJDYTH\anaconda3\envs\yolo\python.exe -u code\test\evaluate_local_factors.py --output runs\crop_study\ab_local_new --fusion B4 --local-weights runs\crop_study\source_local_20261012\local_training\A1_seed42\weights\best.pt
```

全局直接读取S2缓存，只跑新局部模型；导出带ROI hash的局部cache，可以用`--local-cache 文件`重复对比B2/B4，避免重复检测。同一ROI顺序/坐标不一致拒绝评估。学习式质量器没有实现拟合入口，E1-Q/E3-Q尚无结果。

未来需要实测完整GPU端到端延迟时单独运行（本阶段遵守离线融合不重复模型推理）：

```bat
C:\Users\FXQJDYTH\anaconda3\envs\yolo\python.exe -u code\test\benchmark_local_factors.py --output runs\crop_study\ab_timing_new --fusion B2
```

该命令实际跑全局640、相同选区验证、固定ROI局部、融合，计入disk decode/selection/GPU同步/原图映射。后续加`--local-weights ...`可测E2/E3。不自动启动训练。

原研究重做时先选新目录：`run_source_study.py --output 新目录 --stage audit`，再fuse/diagnose/score，`prepare_local_rois.py --output 新目录 --stage build/verify`。当前额外MRU、detail、checks入口默认当前研究路径，改研究根时需要显式更新其DEFAULT或调用对应函数，不能误写旧目录。已完成REPORT拒绝覆盖。
