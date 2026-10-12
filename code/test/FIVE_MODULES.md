# VisDrone：五个独立候选模块及正式训练

更新日期：2026-10-11。当前数据集为 VisDrone2019-DET，完整 train 6471 / val 548 / test-dev 1610 张。本轮先在官方训练/验证划分筛选五个方向，test-dev 不参与选择。AI-TOD-v2 尚未实验。

旧 `runs/dcr/pilot256_e15` 训练已按用户要求停止，保留部分权重和日志。不能把中断的初筛写成完成实验。

## 文献依据与贡献边界

以下参考的是原论文和出版社页面，当前没有复现它们的完整网络。不同骨干、输入分辨率、预训练、测试划分及推理策略的公开 AP 不能直接作为本轮公平对照。

|使用 VisDrone 的论文|已提出的方法方向|本轮吸收的问题|
|---|---|---|
|[DMNet，CVPRW 2020](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Li_Density_Map_Guided_Object_Detection_in_Aerial_Images_CVPRW_2020_paper.pdf)、[CDMNet，ICCVW 2021](https://openaccess.thecvf.com/content/ICCV2021W/VisDrone/papers/Duan_Coarse-Grained_Density_Map_Guided_Object_Detection_in_Aerial_Images_ICCVW_2021_paper.pdf)|以密度图指导区域裁图和检测|标量密度不能充分表达哪个方向、哪一圈邻居会干扰|
|[TPH-YOLOv5，ICCVW 2021](https://openaccess.thecvf.com/content/ICCV2021W/VisDrone/papers/Zhu_TPH-YOLOv5_Improved_YOLOv5_Based_on_Transformer_Prediction_Head_for_Object_ICCVW_2021_paper.pdf)、[QueryDet，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.pdf)|高分辨率检测头；以粗查询引导稀疏高分辨率计算|P2 是已知手段，新增计算成本必须实测|
|[VistrongerDet，ICCVW 2021](https://openaccess.thecvf.com/content/ICCV2021W/VisDrone/papers/Wan_VistrongerDet_Stronger_Visual_Information_for_Object_Detection_in_VisDrone_Images_ICCVW_2021_paper.pdf)|增强前景视觉信息和多尺度融合|辅助特征监督已经有先例|
|[SARC-DETR，Remote Sensing 2026](https://www.mdpi.com/2072-4292/18/10/1670)|相似性自适应卷积、方向条带上下文和动态融合|方向卷积、相似性注意力本身不能作为原创贡献|
|[ECF-YOLO，Discover Computing 2026](https://link.springer.com/article/10.1007/s10791-026-10109-w)|YOLO11n 的边缘保留、方向上下文、加权跨尺度融合|边缘增强与融合已有工作；研究可验证的实例几何监督|
|[Unmasking the Tiny，Image and Vision Computing 2026](https://www.sciencedirect.com/science/article/pii/S0262885626001332)|稀疏前景选择、语义邻居聚合，恢复弱前景|同类邻居可能帮助弱目标，也可能造成类别串扰|

补充相关工作：[BGS 的边界辅助监督](https://pmc.ncbi.nlm.nih.gov/articles/PMC9142905/)研究堆叠纸箱，[Repulsion Loss](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Repulsion_Loss_Detecting_CVPR_2018_paper.html)研究拥挤行人；二者不能列成 VisDrone 实验论文，但必须承认边界监督和利用邻近框已有先例。

本轮五个名称指本项目的候选实现，不是已获认可的新算法。候选贡献集中在“增强后的实例几何/类别关系如何监督具体的特征选择”。Softmax、池化、平移采样、高通残差、对比学习和 P2 本身都不是创新点。检索尚不足以证明首次提出；方向监督、边界监督、尺度融合和语义邻居是重叠风险较高的领域。论文投稿前仍需系统查新和与最接近机制的实证对照。

## 五个可检验的候选

它们单独插入同一个 P2/P3 融合位置，不在一个模型中叠加。

|编号|名称|候选监督机制|可能的优势及风险|
|---|---|---|---|
|M1|DCR，方向净空路由|从实例边界提取四方向归一化净空，监督局部/右/左/下/上五路选择|可能避开被邻居占据的方向；轴对齐框净空不是真实像素遮挡|
|M2|CSR，环域碰撞抑制路由|按不同环域中的邻居框碰撞面积监督局部/三圈上下文选择|可能选择有上下文而较少碰撞的范围；面积和只是碰撞代理|
|M3|BRR，目标边界残差恢复|用框边界内外成对采样监督高通细节门控|可能保留弱目标几何差异；框边缘不一定等于物体轮廓|
|M4|GPR，网格相位可靠性融合|用目标框对 stride=4/8 网格的最大单格覆盖率监督 P2/P3 融合|可能适应目标相对于网格的位置变化；覆盖率不等于真实抗混叠可靠性|
|M5|PCR，邻居共识弱目标恢复|前景信任监督加邻近同类/异类实例对监督，约束语义邻居聚合|可能恢复弱小目标、减少类别串扰；同类相邻目标仍可能被过度平滑|

### M1：方向净空

采用 [DCR_RESEARCH.md](DCR_RESEARCH.md) 的准确规则。邻居需要处于对应半平面，并在垂直方向的框区间相交。边界间隙除以两框尺度均值，重叠净空为 0，图像边界裁限，cap=6。

`t_i = normalize([1, sigmoid((q_R-2)/0.5), sigmoid((q_L-2)/0.5), sigmoid((q_D-2)/0.5), sigmoid((q_U-2)/0.5)])`

共享 DW3×3 特征在 2、4 个 P2 格偏移采样形成方向上下文；越界采样不参与均值，不循环回绕。增强后单目标图不计算邻居监督。

### M2：环域碰撞

共享局部 DW3×3，使用半径 1、3、5 个 P2 格的外方窗减内方窗得到三圈特征；按有效格数归一化。

对实例 i，`A_i,r` 为该圈在图像内、排除自身框后的上下文面积，`C_i,r` 为其他实例框与该区域相交面积的和除以 `A_i,r`，裁到 `[0,1]`。自身排除，重叠面积可能重复计数，因此不是精确并集或真实可见比例。

`u_i,r = log(1 + 4r/s_i) exp(-4 C_i,r)`，局部分支 utility=0.5；上下文面积为空的圈被屏蔽，四路 softmax 形成软标签。监督门控 soft cross entropy。不同尺度的圈在局部特征后进行空间汇聚，因此整体有效感受野还包含 DW3×3 支持。

### M3：边界内外细节

`H=F-AvgPool3(F)`，输出 `F + gamma Projection(DW3(F)+sigmoid(gate(F))*H)`。

每个微小框取四条边中点附近的内侧正样本和外侧负样本。外侧距离 `max(2,0.1*s_i)` 像素，内侧偏移受到框四分之一宽高约束。外侧点出图像或落入任何有效目标框时屏蔽；在 P2 门控图上双线性采样，类别平衡 BCE。监督的是框内外分离代理，不使用实例分割轮廓。

### M4：网格相位

令 `a_i,s` 为增强后的框与某个 stride=s 网格单元的最大交叠面积；`o_i,s=a_i,s/s²`，标签 `t_i=normalize([o_i,4,o_i,8])`。

两路分别投影 P2 和上采样 P3，由 softmax 门控融合后加到基础融合特征。相同尺寸的框在网格中心与网格交界处可能产生不同标签。例如 2×2 框中心为 (3,3) 时为 `[0.8,0.2]`，在 (4,4) 时为 `[0.5,0.5]`。当两种网格都存在被目标完全覆盖的格时趋于均匀，而不是预设所有小目标永远选择 P2。

这里实现的是“网格覆盖代理监督”，没有改写下采样器，没有证明精确的采样可靠性或抗混叠能力。需进一步检验相位平移后的性能变化。

### M5：同类邻居共识

学习 16 维归一化嵌入和单通道前景信任。在八个邻近 P2 格偏移（轴向和对角，各 2 格）采样局部特征。

邻居路由分数 `cos(e_p,e_neighbor)/0.3 + log(trust_neighbor)`；自身分数为 0，越界邻居屏蔽。九路 softmax 决定汇聚权重。

微小实例中心为前景正样本，16×16 规则点阵中落在所有有效框之外的位置为背景负样本。16 像素内的微小实例对，同类为 1、异类为 0，采用 cosine/0.2 的平衡 BCE；嵌入监督和信任监督取均值。实例对仅在同图计算，分块避免完整大矩阵。

## 公共实现和可复现设置

- 模块：[candidates.py](../../ultralytics/nn/modules/candidates.py)；监督：[candidate_loss.py](../../ultralytics/utils/candidate_loss.py)。
- 架构：[candidate_experiments.py](candidate_experiments.py)。保持原 YOLO11n 层 0–22，层 23 输入 `[2,16]`，检测层 24 输入 `[23,16,19,22]`，stride `[4,8,16,32]`。
- 权重：本地 `model/yolo11n.pt`。原检测头的 P3/P4/P5 参数映射到四尺度头的对应分支；类别含义不同的分类输出层不迁移。P2 和新模块初始化。
- 总损失：原 YOLO box/cls/dfl 三项加 `0.1 * L_candidate`，主损失和标签分配器不变。γ 初始 0.1 可学习。
- 教师从图像增强后的 `batch.bboxes` 计算，不使用验证标注训练。微小目标定义为增强输入像素中 `sqrt(w*h)<=32`。无效框、空图合理屏蔽；邻居型 M1/M2 屏蔽单有效实例图。
- `spacing_valid` 如由调用方提供会屏蔽相应实例。当前 YOLO 数据增强没有传递 VisDrone ignored-region 掩膜：忽略区内的原正样本在预处理时剔除，但 M3/M5 背景点无法完整屏蔽忽略区域。这是当前候选监督的限制。
- 推理仅 RGB。真实框教师不参与前向推理，训练张量在 loss 后清除，支持 EMA、保存/加载。
- 正式默认：100 epoch，batch=8，640，workers=4，GPU0，AMP，seed42，AdamW lr0=0.001，cosine lr，nbs=64，deterministic；最后 10 轮关闭 Mosaic。cache=none；每 10 轮保存额外权重。
- 实测 PyTorch 提示 CUDA `grid_sample` 的反向没有确定性实现，上游设置为 `warn_only=True`。固定 seed/数据/参数可重跑，但当前不能保证逐位一致；稳定收益需要多个种子统计，不能用“deterministic=True”替代实际重复实验。
- M0 为同设置 YOLO11n+P2，没有新候选模块。此前 E0 使用 batch16 的结果只能作为背景，不能代替本轮 M0。
- 各模型独立进程顺序运行，单卡不并行训练。失败会记录并继续其他候选，不自动减少 batch 或轮数，不把失败写成完成。
- `experiment.json` 记录实际训练参数、数据清单与标签、预训练权重和源码 SHA256；数据和模型来自相同全量划分。

## 检查与正式命令

必须使用 Anaconda yolo 环境，避免系统 Python 缺少 yaml/torch。CPU 测试 `test_candidates.py` 验证梯度、教师几何、无监督开关、检测头迁移、RGB 权重加载；`stress_candidates.py` 检查 batch8、640、每图512目标的 CUDA AMP 前向/反向，真实数据 smoke 再检查全部六个模型的训练/验证/保存/额外评估。

2026-10-11 检查已完成：新模块11项、NSCR16项、DCR11项测试通过；六个模型均通过 batch8 CUDA AMP 压力测试；六组真实 VisDrone 16训练/8验证图的一轮训练、权重加载、补充评价和结果汇总通过。峰值压力测试显存约5.83–6.50 GiB。证据见 [一轮流程核对](../../runs/candidates/smoke_b8/validation.json)、[GPU压力结果](../../runs/candidates/stress_b8.json)。这些是功能验证，不是精度提升结论。检查结束时按用户要求提供正式命令；此后的实际训练状态以各项目的 `queue.json`、进程和日志为准。

在项目根目录的 PowerShell 执行：

```powershell
& "C:\Users\FXQJDYTH\anaconda3\envs\yolo\python.exe" -u .\code\test\run_five.py --data .\datasets\nscr_visdrone\dataset.yaml --weights .\model\yolo11n.pt --project .\runs\candidates\formal_100e_b8_s42 --experiments M1 M2 M3 M4 M5 M0 --epochs 100 --batch 8 --workers 4 --imgsz 640 --device 0 --seeds 42
```

该命令先训练 M1–M5，再训练公平比较用的 M0，每组100轮。监督器自动选择 yolo 环境并保存独立日志、`best.pt` / `last.pt`、epoch CSV 和各组评估。已有队列目录拒绝覆盖；重复运行需明确选新 project。单模型恢复示例（已有 `last.pt` 时）：

```powershell
& "C:\Users\FXQJDYTH\anaconda3\envs\yolo\python.exe" .\code\test\train_candidate.py --experiment M1 --project .\runs\candidates\formal_100e_b8_s42 --epochs 100 --batch 8 --workers 4 --resume .\runs\candidates\formal_100e_b8_s42\M1_seed42\weights\last.pt
```

上例只恢复该单模型，不自动恢复后续队列；完成后可给剩余模型新 project 重新建队列，避免重复已完成模型。

查询、停止：

```powershell
& "C:\Users\FXQJDYTH\anaconda3\envs\yolo\python.exe" .\code\test\status_candidates.py
& "C:\Users\FXQJDYTH\anaconda3\envs\yolo\python.exe" .\code\test\stop_candidates.py --after-current
# 如需立即停止当前训练及整个队列，去掉 --after-current，已有权重保留。
```

`queue.json` 记录状态、进程、开始/结束时间和返回码。`logs/M1_seed42.log` 等记录实时进度，`ablation_summary.csv` 汇总完成训练，队列结束生成 `review/review.md`。

队列启动后默认将训练/验证输出同时显示在终端并写入独立日志，保留进度条的回车刷新。修改前已启动的进程不会自动应用新代码；可在另一个终端查看现有队列，无需中断训练：

```cmd
python code\test\watch_training.py --project runs\candidates\formal_100e_b8_s42_v2
```

查看器会跟随各模型及评估任务的切换；关闭查看器只停止显示，不会停止训练。

2026-10-11 已按用户要求安排当前正式队列在 M3 的100轮训练及评估完成后，先运行 M0，再运行 M4、M5。交接记录见项目 `handoff.json`，原监督器在当前任务结束后退出，后续监督器使用 `--resume-queue` 跳过完成的 M1/M2/M3，复用原数据和训练参数。训练与权重均不中断、不覆盖；STOP 标记由交接器创建并在成功交接时清除。恢复队列会拒绝已有活动监督器、改变参数或直接覆盖中断训练。

## 有效性的判断

每个完成训练自动计算验证集 mAP50/mAP50:95、Precision/Recall、COCO 补充 AP-small/AR-small、目标密度分组召回和定位误差、Params/GFLOPs、GPU fused forward 延迟/FPS，并给出候选教师诊断。

COCO 小目标以原图面积≤32²、maxDet=500统计，忽略区域按 crowd 近似处理，不等于官方 VisDrone 得分。已导出官方评估格式，但官方工具评估需另行执行。GPU 延迟为 batch1 FP32 的同步网络前向，不含图像读取、预处理或 NMS。GFLOPs 工具不完整覆盖访存和逐元素操作，不能直接等同实际耗时。不同教师的诊断 loss 数值不能用于候选之间的精度排名。

本轮是单 seed 的完整训练筛选；先看同设置 M0 的 mAP50:95 和小目标指标，再看算力代价。若有优胜候选，下一轮至少需要：相同结构 `--aux-weight 0`、打乱/去除监督信息、seed42/43/44、相应场景分组和另一数据集。没有完成这些实验之前，不能声称监督机制成立或论文原创性已经证明。零权重入口已经实现，但本次队列不会偷偷扩展为额外15组实验。
