# VisDrone 文献查新与 DCR 候选模块

查阅日期：2026-10-10。当前实际数据集是 **VisDrone2019-DET**，已有全量划分为 train 6471 / val 548 / test-dev 1610 张。AI-TOD-v2 是后续独立数据集验证，不能把本次结果写成已完成双数据集验证。

## 使用 VisDrone 的代表性方法

以下是原论文/出版社页面支持的方法概述。表内是设计参考，本项目没有声称已复现这些论文。

|论文|主要改进|对本项目的启发及区别|来源|
|---|---|---|---|
|DMNet，CVPR Workshops 2020|预测密度图，由密度决定裁图区域，再检测|空间分布信息有价值；DCR 在特征内选择上下文方向，当前不裁图|[原论文](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Li_Density_Map_Guided_Object_Detection_in_Aerial_Images_CVPRW_2020_paper.pdf)|
|CDMNet，ICCV Workshops 2021|粗粒度密度图、连通区域裁图及检测|可作为密度指导思想的对照；DCR 使用实例边界净空而非密度图强度|[原论文](https://openaccess.thecvf.com/content/ICCV2021W/VisDrone/papers/Duan_Coarse-Grained_Density_Map_Guided_Object_Detection_in_Aerial_Images_ICCVW_2021_paper.pdf)|
|TPH-YOLOv5，ICCV Workshops 2021|增加高分辨率预测头、Transformer 预测头、CBAM，并使用多尺度测试/集成等策略|P2 和注意力本身已有充分先例；DCR 不把增加 P2 当原创贡献|[原论文](https://openaccess.thecvf.com/content/ICCV2021W/VisDrone/papers/Zhu_TPH-YOLOv5_Improved_YOLOv5_Based_on_Transformer_Prediction_Head_for_Object_ICCVW_2021_paper.pdf)|
|VistrongerDet，ICCV Workshops 2021|加强 FPN 融合和前景视觉信息，包括 mask 分支|辅助特征监督已有先例；DCR 的监督对象是方向选择|[原论文](https://openaccess.thecvf.com/content/ICCV2021W/VisDrone/papers/Wan_VistrongerDet_Stronger_Visual_Information_for_Object_Detection_in_VisDrone_Images_ICCVW_2021_paper.pdf)|
|QueryDet，CVPR 2022|低分辨率粗定位查询，指导高分辨率的级联稀疏计算|说明高分辨率收益伴随计算成本；DCR 当前仍为稠密 P2，需要实测速度|[原论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.pdf)|
|Cascaded Zoom-In，CVPR Workshops 2023|复用检测器进行密度指导的级联放大检测|密度指导也可不依赖单独密度网络；DCR 当前单次前向、不进行级联放大|[原论文](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/html/Meethal_Cascaded_Zoom-In_Detector_for_High_Resolution_Aerial_Images_CVPRW_2023_paper.html)|
|Unmasking the Tiny，Image and Vision Computing 2026|稀疏前景 token 选择、语义邻居聚合，恢复弱小目标前景得分|邻居上下文与前景增强已有工作；DCR 选择目标边界较空的方向，而非学习语义相似性聚合|[出版社页面](https://www.sciencedirect.com/science/article/pii/S0262885626001332)|
|SARC-DETR，Remote Sensing 2026|相似性自适应卷积，以及水平/垂直条带方向卷积和内容自适应融合|最需要区分的已有工作：方向感受野和动态融合本身不能作为 DCR 原创点|[出版社页面](https://www.mdpi.com/2072-4292/18/10/1670)|

TPH-YOLOv5 使用的 VisDrone2021 延续了相同数据集，论文还包含不同测试划分和测试策略；不能把其挑战赛 AP 当作当前 640 分辨率、单模型验证集 AP 的直接对照。Unmasking the Tiny 使用 YOLOX-X、训练 960、VisDrone 测试 1504。本项目 YOLO11n、640 与这些公开表格的网络规模、输入、推理次数和协议不同，公开 AP 只可作背景。

另外，[LSKNet，ICCV 2023](https://arxiv.org/abs/2303.09030) 已有大核动态选择，其原文实验为 HRSC/DOTA/FAIR1M，不能列成已在 VisDrone 上验证的方法。[Repulsion Loss，CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Repulsion_Loss_Detecting_CVPR_2018_paper.html) 已经使用邻近真实框约束拥挤检测的框回归，因此“利用邻近框”也不是新概念；DCR 保持框回归损失不变，监督的是特征方向路由。

部分出版社/会议页面直接打开受到 403/429 限制；2026 条目基于出版社检索结果提供的摘要/实验片段，未声称完整逐页审查。此次检索不是穷尽式系统综述，也不是专利检索。

## 科学问题和候选贡献

原 NSCR 使用一个归一化最近邻中心间距，决定局部/宽域上下文。这个标量无法区分“邻居在右边、上下空闲”和“四周都拥挤”。进一步的研究假设是：**在相同标量密度下，避开被邻近实例占据的方向、保留较空方向的上下文，可能更适合密集微小目标。**

提出 **DCR：Directional Clearance Routing，方向净空路由**。候选贡献限定为：

> 从增强后的实例框边界，构造尺度归一化的四方向净空分布，对图像特征产生的五路上下文路由进行实例级辅助监督。

共享卷积、平移采样、Softmax、多分支和残差均为常规组件。有限检索暂未识别到与上述监督规则完全相同的论文，不能由此断言“首次提出”。若存在同类实例边界/方向监督，需要继续对比并调整贡献表述。若实验不支持方向监督收益，则应作为未成立候选报告。

检索词包括 `VisDrone density guided detection`、`VisDrone dynamic receptive field`、`VisDrone directional routing supervision`、`object detection directional supervision context neighbor`、`receptive field ground truth neighbor routing detection`、`directional clearance object detection`。也检查了新的 VisDrone 方向感受野工作，避免仅对比旧 YOLO 改进文章。

## 已实现的网络

YOLO11n 原主干和原 P3/P4/P5 颈部（层 0–22）保持原结构。层 23 输入原 P2（层 2）和 P3（层 16），上采样 P3、拼接、1×1 融合。检测层 24 接收 `[23,16,19,22]`，四尺度 stride 为 `[4,8,16,32]`。

```text
P2 ─────────────────┐
                    ├─ concat → 1×1 → F → shared 3×3 DW → local
P3 → nearest resize ┘                └→ gate → Softmax(local,R,L,D,U)
local → directional samples at 2,4 P2 cells → 4 context features
local + 4 context features → weighted sum → 1×1 → residual with F
```

令 `Z=DW3x3(F)`，方向单位向量 `e_d` 按右/左/下/上排列。`C_d(p)` 是 `Z(p+2e_d)` 与 `Z(p+4e_d)` 的有效采样均值，越过边界的采样不参与平均；两个都无效时退回 `Z(p)`，无循环边界。`C_0=Z`。

`P(p)=softmax(gate(F(p)))`

`F_out(p)=F(p)+γ φ(Σ[k=0..4] P_k(p) C_k(p))`

γ 初始 0.1、可学习。五路 gate 输出层初始为零，初始路由均匀。平移采样共享局部卷积参数，四个方向不新增各自的卷积参数。D1–D4 参数量完全相等。P2 的整体成本仍然需要对照 E1 实测，不能只因参数少就称其无开销。

## 净空监督的准确规则

增强后第 i 个像素坐标框边界为 `(l_i,r_i,t_i,b_i)`，尺度 `s_i=sqrt(w_i h_i)`。方向邻居满足：

- 右/左：两框的 y 区间有严格正长度相交，邻居中心处于相应 x 半平面。
- 下/上：两框的 x 区间有严格正长度相交，邻居中心处于相应 y 半平面。
- 相同中心的另一实例视为四方向遮挡；自身从候选中排除。

右方向示例：

`q_i,R = min(cap, (W-r_i)/s_i, min[j in N_R(i)] max(l_j-r_i,0)/(0.5(s_i+s_j)+ε))`

其余方向对称。无有效邻居时内层最小值为 `+∞`，最终受到图像边界和 cap=6 限制。重叠实例净空为零。该量是轴对齐框的保守间隙代理，不等同于真实遮挡比例、精确像素安全距离或旋转框几何。大对象可以作为小对象的邻居。

`a_i,d = sigmoid((q_i,d-τ)/T)`，默认 τ=2、T=0.5。

`target_i = normalize([1,a_i,R,a_i,L,a_i,D,a_i,U])`

局部分支始终有正质量，净空方向得到更高上下文质量。全部方向净空大时趋于均匀混合；全部拥挤时偏向 local。不能断言净空方向一定有有用语义，该映射是待验证假设。

在 GT 中心双线性采样 **gate logits**，随后 Softmax：

`L_DCR = mean_over_eligible_images(mean_over_tiny_instances(-Σ target_i,k log P_i,k))`

`L_total = L_YOLO + λ L_DCR`，默认 λ=0.1。YOLO 原 box/cls/DFL 损失及 TaskAlignedAssigner 不变。

不同监督分布的标签熵不同，因此跨组 CE 辅助 loss 的绝对数值不能用来证明谁更好。验证集方向诊断使用 `KL(target || prediction)` 扣除标签熵；最终有效性仍以检测 AP 和对照为依据。

训练期通过 YOLO 已完成 mosaic、裁剪、缩放、翻转的 `batch['bboxes']` 现算；不读取预计算原图净空。无有限坐标、负/越界类别、非正尺寸、可选 `spacing_valid=False` 均屏蔽。有效实例不足两个的图像不计算辅助监督；只监督增强后 `sqrt(area)<=32` 像素的小目标，邻居使用该图全部有效目标。不同 batch 图像互不混合。分块 pairwise 运算工作内存 `O(chunk_size*N)`，运算仍为 `O(N²)`，只发生在训练。

VisDrone 原转换流程已剔除 ignore 类别/无效框，并排除中心处在 ignore 区域内的正框。这能保证这些框不参与辅助监督，但目前没有完整地将 ignore 区域从 YOLO 背景损失中屏蔽。补充 COCO 评价的 ignored-region 近似不能宣称等价于官方 VisDrone 工具。

**推理输入只有 RGB。** 不调用净空构造，不需要 GT、密度标注、新传感器或手工方向标签。验证时使用 GT 的门控诊断发生在预测之后，只用于解释实验。

## 对照设计

|编号|网络/监督|检验目的|
|---|---|---|
|E0|原 YOLO11n|完整基线；正式对照须用匹配训练设置重新运行|
|E1|YOLO11n + P2|P2 收益/成本|
|E3|原双感受野门控，无辅助监督|旧常规动态融合对照|
|E4|原 NSCR 最近邻中心间距监督|与已有候选作匹配比较|
|D1|DCR 网络，无辅助监督|排除增加方向上下文/参数的影响|
|D2|DCR，四方向标签取均值后广播|保留同实例 local/context 总质量，去除方向信息|
|D3|DCR，真实四方向净空监督|完整候选|
|D4|DCR，逐实例非恒等循环置换四方向标签|保持实例标签直方图和 local/context 质量，破坏方向对应关系|

D4 的通道顺序为 R/L/D/U，因此循环置换是标签置换，不能称为真实罗盘旋转。对称目标的标签可能置换后仍相同，须单独分析各向异性目标。D2 是无方向的净空监督，不能称为常规 Gaussian 密度图对照；Gaussian 密度思想的旧 E6 仍由 `run_formal.py` 提供。

预先指定主指标为 `mAP50:95`，要求 D3 对 D1/D2/D4 的同种子差值提供正向证据，并检查 E1/E4、AP-small、AR-small、密集/稀疏召回和定位误差、方向 KL 与方向概率差。不把辅助 loss 下降作为检测精度改进证据。短训初筛可能只能否定或提示机制，不足以确认其有效性。

## 运行和已验证状态

实现：`ultralytics/nn/modules/dcr.py`、`ultralytics/utils/dcr_loss.py`。实验配置由 `dcr_experiments.py` 生成；正式入口 `run_dcr.py` 会优先选择 Anaconda yolo，防止误用系统 Python。

功能测试命令：

```powershell
& "$env:USERPROFILE\anaconda3\envs\yolo\python.exe" code/test/test_dcr.py
& "$env:USERPROFILE\anaconda3\envs\yolo\python.exe" code/test/test_nscr.py
```

已通过 11 项 DCR 测试和 16 项旧 NSCR 测试：框边界净空/反射对应/尺度归一化/分块一致性、局部与上下文质量匹配、梯度方向、单实例和无效框屏蔽、无效配置拒绝、D1–D4 反传及相同参数量、权重映射、RGB 推理不调用 GT 函数、保存加载兼容。它们验证实现，不能证明检测指标提升。

另在两张真实验证图上通过 CPU 评估流程检查（`smoke_dcr_evaluation.py`，结果 `runs/dcr/evaluation_smoke/smoke.json`）：未训练、均匀门控的 KL 等于 uniform 参考 KL，方向概率差为零，符合预期；小目标 COCO 评价、官方预测导出及坐标映射流程可执行。该未训练检查不属于精度实验，也不属于 GPU 性能结果。

旧初筛计划：固定 hash 划分 seed=42，train=256、val=128；模型 seeds=42/43/44；E1/E4/D1/D2/D3/D4；15 epoch、640、batch=4、workers=4、AdamW lr0=0.001、AMP、相同增强。**2026-10-11 已按用户要求停止，只有部分训练日志/权重，不构成完成的对照实验。** 新的五候选、batch8/100轮队列见 [FIVE_MODULES.md](FIVE_MODULES.md)。test-dev 不参与模型选择。

独立的标注几何统计见 `runs/dcr/geometry/geometry.json`：训练子集增强前有 12139 个缩放到 640 后的小目标，其中 10487 个按原 NSCR 标量间距定义为密集，约 74.68% 的这些密集目标满足 `max(q_d)-min(q_d)>=1`。这个结果只证明方向标签有变化，不证明空方向必然更有用。统计没有用于拟合或调整超参数。

静态复杂度见 `runs/dcr/model_complexity.json`：E1 为 2688760 Params / 11.458 GFLOPs，D1–D4 均为 2694830 Params / 11.783 GFLOPs。相比 E1 新增 6070 参数，名义 GFLOPs 增加约 2.84%；相比原 E0 6.448 GFLOPs，四尺度结构总成本明显增加。平移与访存开销需结合实测延迟判断。

正式完整实验命令（下面包括 E0/E3，共 24 个实验；删去 E0/E3 为 18 个 E1/E4/D1–D4 实验）：

```powershell
python code/test/run_dcr.py --data datasets/nscr_visdrone/dataset.yaml --project runs/dcr/formal --epochs 100 --imgsz 640 --batch 4 --workers 4 --experiments E0 E1 E3 E4 D1 D2 D3 D4 --seeds 42 43 44
```

正式原 NSCR 的 E0–E6 命令和矩阵保持不变。为可比性须固定 batch，而不是对不同模型各自 AutoBatch。`experiment.json` 记录权重、数据划分、标签与代码哈希；避免重用不完整实验目录。`--skip-complete` 仅允许参数和哈希匹配的完整实验。

全指标评估和配对报告：

```powershell
& "$env:USERPROFILE\anaconda3\envs\yolo\python.exe" code/test/evaluate_dcr.py --project runs/dcr/formal --data datasets/nscr_visdrone/dataset.yaml --experiments E0 E1 E3 E4 D1 D2 D3 D4 --seeds 42 43 44 --batch 4 --workers 4 --no-plots
& "$env:USERPROFILE\anaconda3\envs\yolo\python.exe" code/test/review_dcr.py --project runs/dcr/formal
```

评估输出包含 Ultralytics AP/P/R、补充 COCO AP-small/AR-small、原始小目标密集/稀疏 recall/IoU/归一化中心误差、DCR 验证集方向 KL 和方向概率、Params/GFLOPs、RTX3090 单图 FP32 融合前向延迟与 FPS，以及官方 VisDrone 预测文件/MATLAB 调用桥接。前向 FPS 排除 IO、预处理和 NMS，不是端到端应用 FPS；官方 MATLAB 工具未执行前不能把补充 COCO 结果称为官方指标。

## 进一步研究的限制

1. 清空方向未必包含有用上下文；同类邻居有时能提供强语义，应通过实验判断。
2. 固定 2/4 P2 格偏移和按目标尺度归一化的标签并不严格一一对应，需要正式做偏移、τ、T、λ 灵敏度对照。初筛后调参属于探索阶段，应另设确认实验。
3. 全图边界包含 letterbox/mosaic 背景，当前没有额外建模各 mosaic tile 的有效图像边界。
4. 五路采样增加访存，THOP GFLOPs 可能不完整统计平移/Softmax/逐元素操作，必须结合实际 GPU 延迟。
5. 单数据集、短训、三个种子的正向均值都不足以证明通用有效或统计显著，更不能保证 EI 录用。完整数据、多种子、跨数据集和机制诊断才决定最终贡献是否成立。
