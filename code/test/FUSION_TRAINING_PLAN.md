# P2/P3 最小融合实验：审核与执行计划

> 最新执行：用户要求将batch改为8并继续。已停止batch16队列，保留前22轮检查点、优化器与旧参数日志；F0从第23轮恢复，总轮数仍为100。后续F2_GLOBAL → F2 → F3_PLAIN → F3 → F1从头以batch8、workers4、640、seed42训练。为保留断点与日志路径，继续使用 `runs/fusion/formal_100e_b16_s42/`，目录名保留b16但实际batch已为8，配置以queue.json/args.yaml为准。

F0具有前22轮batch16、后78轮batch8的混合训练历史，不能作为纯batch8正式基准。后续五组使用原 `runs/candidates/formal_100e_b8_s42_v2/M0_seed42` 的完整100轮batch8基准进行公平比较。旧状态、配置、CSV、best/last及日志归档到队列目录 `history/`；切换信息另存F0的 `batch_transition.json`。恢复逻辑、明确指定检查点、混合历史排除和基准保护共8项测试通过，完整dry-run通过。已核对实际加载检查点的SHA256与原22轮检查点一致，且确认日志从第23轮进入训练；验证记录为 `batch8_resume_verification.json`。

> 执行更新（2026-10-11）：用户已明确授权“开始训练吧batch=16”。本次执行配置更新为batch=16，其余100轮、workers=4、imgsz=640、seed42不变。为避免与旧batch8的M0混比，先重新训练F0，再运行F2_GLOBAL → F2 → F3_PLAIN → F3 → F1。新目录为 `runs/fusion/formal_100e_b16_s42/`。下文batch8与待授权内容保留为原审核计划；当前执行以此更新及新目录queue.json为准。

已通过新基准保护的3项测试及batch16完整dry-run。启动器在设置改变且未安排F0首组时仍拒绝正式对照；包含F0的队列在最终分析时使用本目录F0基准。

当前队列查看进度（不会启动第二个训练）：

```bat
C:\Users\FXQJDYTH\anaconda3\envs\yolo\python.exe -u code\test\watch_training.py --project runs\fusion\formal_100e_b16_s42
```

日期：2026-10-11。本计划尚未执行。用户本轮附件明确要求“需要训练时先给出完整计划，获得用户明确授权后启动”，因此完成审计及固定权重诊断后保持训练停止。

## 1. 研究依据与当前状态

固定 M0 最佳权重完成同一548张验证图像的640/960/1280推理对照，详见 [RESOLUTION_STUDY.md](RESOLUTION_STUDY.md)。已有 M1—M3 没有足够机制收益证据，不继续叠加；F 系列尚无正式精度结果。现有结构、梯度、初始化、预训练映射与保存加载检查已通过。本轮没有更改融合模块结构。

第一轮只做已有最小融合对照，统一在640训练和评估；不把分辨率收益混入模块收益。F0复用同配置、同seed42、100轮完成的M0基准。所有新实验从本地COCO预训练迁移开始，**不从已训练M0 best.pt继续微调**，保证训练预算一致。

## 2. 顺序、假设与结构

|顺序|实验|第23层改动|新增参数|主要比较|
|---|---|---|---:|---|
|1|F2_GLOBAL|两枚全局标量logits，softmax后分别加权P2/P3|2|相对M0，检查重新参数化和优化影响|
|2|F2|P3产生两通道空间softmax权重，逐位置加权P2/P3|130|相对F2_GLOBAL与M0，检查空间变化的附加价值|
|3|F3_PLAIN|P2高通残差经零初始化DW3卷积，弱残差系数0.1|576|相对M0，检查细节分支本身|
|4|F3|对相同细节分支加入P3语义门控|641|相对F3_PLAIN与M0，检查语义选择|
|5|F1|P3产生弱语义门控，P2乘性调制范围0.9–1.1|65|相对M0，检查弱语义校准|

0—22层、第24层Detect、原始YOLO损失、P2/P3/P4/P5分支、数据划分保持一致。所有模型起始共享权重、Detect初始化及融合输出一致；额外层构造不消耗共享随机序列。

F2_GLOBAL 的全局系数可被原1×1投影吸收，因此是优化参数化对照，不是表达能力相等的空间门控对照。F2与F2_GLOBAL的参数量、输入依赖和函数能力不同，正收益仍需要进一步机制验证。F1失败也只否定当前弱门控及调制范围，不能否定全部语义校准方法。

## 3. 固定训练配置及预训练

|项目|值|
|---|---|
|数据|`datasets/nscr_visdrone/dataset.yaml`，官方train6471 / val548；test1610不参与模型选择|
|输入与预算|epochs=100，imgsz=640，batch=8，workers=4，device=0，seed=42|
|优化|AdamW，lr0=0.001，lrf=0.01，cos_lr=True，nbs=64，weight_decay=0.0005|
|预热与复现|warmup_epochs=3，deterministic=True，AMP开启|
|增强|mosaic=1，最后10轮关闭；mixup/cutmix/copy_paste=0，其余沿用M0有效参数|
|保存与停止|patience=101，save_period=10，缓存关闭，失败后停止队列|
|评估|每组best.pt统一548张val，imgsz=640，NMS IoU=.7，预测conf=.001，maxDet=500；诊断conf=.25|
|预训练|本地`model/yolo11n.pt` SHA256=`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`|
|迁移规则|448个兼容张量；原P3/P4/P5头重映射；不迁移COCO80类输出到VisDrone10类；P2按原策略初始化|
|辅助损失|无，保留与M0相同的零第四项用于日志兼容|

`train_fusion.py`调用共享`train_one`，有效训练参数写入每组`experiment.json`。启动前再次核对完整有效参数，尤其依赖框架默认值的lrf/nbs/weight_decay/warmup；若与M0不一致应停止排查，不自行替换配置。数据名单与标签内容SHA256、本地权重SHA256预检查已经通过，dry-run保存于 `runs/fusion/fusion_plan_dry_run.log`。

## 4. 时间、显存与存储预算

使用同一RTX3090单卡串行运行，最多一个训练进程；不与本次验证或其他训练并发。M0训练CSV累计10089.2秒，约2.80小时/100轮；5组按基准耗时约14.0小时。已有M1—M3耗时3.66–4.28小时/组，其中包含不同模块、辅助监督等成本，不能直接当作F训练速度。预留评估、日志及系统波动后，预计约15–20小时，实际以运行日志为准。

本机24GiB显存可完成M0原配置，F仅增加2–641参数；预计仍可沿用batch8。F的正式训练峰值尚未实测，本次推理峰值不能代替训练估计；若OOM应保留日志并停止，不自动降batch或改变AMP继续。训练开始后的首轮记录显存及耗时，更新剩余预计时间。

每组保留best/last和10轮周期检查点、预测、曲线及可视化，预留20GB存储预算；本轮审计时H盘剩余约36.9GiB，正式开始前再次检查实际剩余空间。M0基准只读复用，旧M4检查点保留，不恢复M4或启动M5。

## 5. 已审核的终端命令

在仓库根目录、Anaconda yolo 环境运行。以下完整解释器路径可以避免调用系统Python38。

只检查，不训练（已执行）：

```bat
C:\Users\FXQJDYTH\anaconda3\envs\yolo\python.exe -u code\test\run_fusion.py --data datasets\nscr_visdrone\dataset.yaml --weights model\yolo11n.pt --project runs\fusion\formal_100e_b8_s42 --experiments F2_GLOBAL F2 F3_PLAIN F3 F1 --epochs 100 --batch 8 --workers 4 --imgsz 640 --device 0 --seeds 42 --dry-run
```

**获得用户明确授权后**执行下列正式命令；本轮没有执行：

```bat
C:\Users\FXQJDYTH\anaconda3\envs\yolo\python.exe -u code\test\run_fusion.py --data datasets\nscr_visdrone\dataset.yaml --weights model\yolo11n.pt --project runs\fusion\formal_100e_b8_s42 --experiments F2_GLOBAL F2 F3_PLAIN F3 F1 --epochs 100 --batch 8 --workers 4 --imgsz 640 --device 0 --seeds 42
```

项目目录当前不存在。若后来已存在，不删除、不覆盖，按实际状态选择新目录或显式单组resume。启动器遇到训练失败、未完成100轮或评估失败即停队列；不自动修改设置绕过失败。训练进度会在当前终端显示，同时保存日志。

## 6. 输出与机制检查

输出根目录：`runs/fusion/formal_100e_b8_s42/`。

|文件/目录|内容|
|---|---|
|`queue.json`、`logs/`|队列状态、进程、参数和逐组训练评估日志|
|`F*_seed42/experiment.json`|实际配置、数据及代码SHA256、权重迁移报告|
|`F*_seed42/results.csv`、`weights/`|100轮曲线、best/last与周期检查点|
|`F*_seed42/fusion_weights.csv`|最后一个训练batch的权重均值、标准差及极端值比例；不是全数据平均|
|`F*_seed42/evaluation/`|统一best.pt指标、COCO预测、548张原始预测文本、MATLAB评估桥接文件|
|`analysis/`|与M0比较的指标、复杂度、分组及机制诊断|

训练期权重日志只作异常提示；正式判定门控塌缩必须使用相同固定验证子集的空间分布。训练后在GT内部与去除GT/忽略区/填充的背景计算门控均值、方差、极端比例，按原图尺寸及q密度分组。F2记录两尺度权重和空间图；同时说明门控由低分辨率P3生成，未自动解决细粒度对齐。

F3训练后比对DW权重初始零值与最终值、周期变化，使用固定batch诊断DW和门控梯度；F3初始DW为零时门控梯度为零是设计性质，后续必须检查有效梯度。对比GT区域与背景的残差能量及原特征能量，不能只看整图增强幅度。任何诊断反向只用于检查，禁止optimizer.step，不能改变best.pt。

统一报告mAP50/mAP50:95、补充AP-small/AR-small、Precision/Recall、10类AP、固定原图尺寸及密度Recall、Params/GFLOPs、同设备同精度实测延迟和可视化。官方VisDrone工具尚需对齐，Python参考成绩单独列，不当作官方成绩。

## 7. 结果筛选与后续决策

第一轮主要基准为M0 mAP50:95=20.3682%。约+0.5pp且AP-small/AR-small与成本合理，可列为多种子复验候选；这只是预算筛选条件，不能称为统计显著。F2同时比较M0与F2_GLOBAL，F3同时比较M0与F3_PLAIN，不能只展示相对较弱对照的收益。

如果出现有效机制，另行制定并获得授权的seed42/43/44配对训练，43/44包括同配置F0；报告均值、样本标准差、配对差值与尺寸/类别/密度收益。三个种子本身也不保证统计显著，更不能替代成熟方法对照与查新。

若F系列无明显收益，停止继续堆叠第23层，优先考虑960匹配训练、小目标裁剪及类别采样；根据当前分辨率诊断制定新基线预算并单独申请授权。若F2或F3确有稳定独立收益，先做空间权重/背景增强解释和成熟机制对照，再讨论原创性。

**授权范围待确认：按上述顺序启动5组100轮、batch8、seed42的正式融合对照；不包括960匹配训练、多种子扩展或新模块设计。**
