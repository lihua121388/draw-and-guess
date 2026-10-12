# NSCR-YOLO 使用说明

> 最新执行为batch8续训：F0从第23轮恢复，后续五组从头训练，均100轮、workers4、640、seed42。目录沿用 `runs/fusion/formal_100e_b16_s42/`，实际batch以配置为准，混合batch的F0结果单独标注。

> 最新固定权重分辨率诊断见 [RESOLUTION_STUDY.md](RESOLUTION_STUDY.md)。用户已授权batch16正式训练，先运行新F0基准，再运行5组最小融合对照；新目录 `runs/fusion/formal_100e_b16_s42/`。执行配置和进度查看命令见 [FUSION_TRAINING_PLAN.md](FUSION_TRAINING_PLAN.md)，以下停止/待授权内容为历史记录。

> 2026-10-11 当前研究路线、M0–M3审计、最小融合实现及停止状态见 [FUSION_STUDY.md](FUSION_STUDY.md)。现有正式训练已停止，新融合正式实验尚未启动。本页保留原NSCR工程的复现说明。

本目录实现 `readme.md` 的模型、监督和可复现实验框架。默认数据集为本地 **VisDrone2019-DET**，使用官方 train/val/test-dev 划分（6471/548/1610 张），不使用红外或其他模态。默认预训练权重为项目根目录 `model/yolo11n.pt`；ONNX 文件不能用于训练。

## 环境与启动

在项目根目录打开 Anaconda Prompt 或支持 conda 的 PowerShell：

```powershell
conda activate yolo
python code/test/train_test.py prepare
python code/test/test_nscr.py
python code/test/train_test.py train --experiment E4 --epochs 100 --imgsz 640 --batch 8 --device 0
```

本机解释器：`C:\Users\FXQJDYTH\anaconda3\envs\yolo\python.exe`。IDE 的 Python 解释器也应选择该文件；默认系统 Python 不是此环境。直接运行 `train_test.py` 等价于启动 E4 正式训练，默认 100 epochs、640 分辨率、batch=8、workers=4、AMP。

RTX 3090 24GB 已检测可用。若数据加载出现 Windows 多进程问题可设 `--workers 0`。显存不足时降低 batch；改变分辨率后必须重新跑对应基线。`--device cpu` 可做功能检查，正式速度报告使用 GPU。

小规模流程检查（结果不能当作论文精度）：

```powershell
python code/test/train_test.py prepare --train-limit 64 --val-limit 16
python code/test/train_test.py ablate --data datasets/nscr_visdrone/dataset-subset-64-16-s42.yaml --epochs 1 --imgsz 320 --batch 4 --workers 0 --project runs/nscr/check --no-plots
```

适中训练子集选项：`prepare --train-limit 2048` 生成固定哈希选取的 2048 张训练图，仍保留完整 548 张验证图。先用该子集调试；论文主结果必须使用完整官方训练划分，不能与全量数据结果直接比较。子集名单与完整名单分开存储，避免覆盖。

## 实现内容

- 主干和原 neck 的 0–22 层保持原 YOLO11n 架构。
- 从 backbone P2（层 2）和 neck P3（层 16）产生额外 P2 特征；1×1 融合后为 64 通道。
- 原 P3/P4/P5 路径不改，Detect 使用 P2/P3/P4/P5，stride 为 4/8/16/32。
- NSCR 使用 dilation=1、3 的两个 3×3 深度卷积分支，单通道空间 Sigmoid 门控和 1×1 输出投影；可学习残差系数初值 0.1。
- 原 box、cls、DFL 损失及 TaskAlignedAssigner 不改。第四个日志项为已乘 λ 的辅助损失。
- 原检测头从层 23 移到层 24，预训练 P3/P4/P5 的回归分支相应映射到新头索引 1/2/3。兼容的分类特征权重保留，类别输出重新初始化，P2 头随机初始化。
- E3/E4/E5/E6 结构和参数量完全相同。E2 使用单路径 1×1→3×3 深度卷积→1×1，以瓶颈宽度近似匹配 E3 模块参数预算；nano 通道下两模块相差约 50 参数，必须按实际 Params/GFLOPs 报告。

| 实验 | P2 模块 | 辅助监督 |
|---|---|---|
| E0 | 原始三尺度 YOLO11n | 无 |
| E1 | P2/P3 的 1×1 融合 | 无 |
| E2 | 参数预算接近的单路径卷积 | 无 |
| E3 | 双分支门控 | 无 |
| E4 | 双分支门控 | 归一化实例间距 |
| E5 | 同 E4 | 打乱间距标签 |
| E6 | 同 E4 | 固定高斯中心密度 |

E6 是控制监督信息的密度对照，不是 DMNet/CDMNet 整个网络的复现；不引入裁图、多模态或其他增强模块。

## 监督定义与屏蔽

训练标注来自数据增强后的 batch，归一化 xywh 乘当前图像宽高得到像素框。对每张图独立计算：

```text
s_i = sqrt(w_i h_i)
q_i = min_(j != i) ||c_i-c_j|| / (0.5(s_i+s_j) + 1e-6)
t_i = sigmoid((q_i - tau) / temperature)
L_spacing = mean_over_eligible_images(mean_over_eligible_instances(BCEWithLogits(g_i, t_i)))
L_total = L_YOLO + lambda * L_spacing
```

这里严格采用需求公式的“归一化比值最小”，与“先选欧氏最近邻，再除尺度”不同。大邻居可能改变最近邻排序，见 `RESEARCH_NOTES.md` 的风险分析。所有有效类别的目标均可作邻居，辅助监督只用于增强后 `sqrt(area) <= 32` 像素的目标。阈值按输入尺度定义，评价 AP-small 则按原图面积定义，两者不能混淆。

默认 λ=0.1、τ=2、T=0.5。门控在目标中心双线性采样；落在同一网格的实例分别参与损失，不覆盖标签。每图均值后再跨图均值，避免极密集图像单方面主导辅助损失。只有一个有效框、无有效框、没有微小目标时辅助项为可微零；不跨图计算邻居。非有限数值、无效类别、非正尺寸以及可选 `spacing_valid` 屏蔽的实例不参与邻居构造。

E5 在一个 batch 的有效标签上构造随机置换的循环错排，保留标签分布和屏蔽位置；只有一个标签时屏蔽此项，相同软标签也可能在错排后数值不变。E6 在 P2 坐标下叠加固定 σ=2 的高斯中心密度，取 `sigmoid((density_threshold - density)/density_temperature)`，默认 2/0.5；与 E4 使用相同监督位置和损失。可用 `--density-sigma`、`--density-threshold`、`--density-temperature` 调整，并记录配置。

Mosaic 会创造合成相邻关系，这是增强后监督的定义，论文应报告 Mosaic 设置，并补充关闭 Mosaic 的稳定性检查。无标签泄漏：验证/推理不构造间距监督，验证日志第四项为零；推理输入仅为 RGB。

## 数据准备

`prepare_data.py` 识别本地单层、双层解压目录，用硬链接或复制建立 `datasets/nscr_visdrone`，保留原始数据。剪裁越界框，排除零尺寸/负尺寸/非有限框及类别 0/11、score≤0 标注。正样本中心落入忽略区域时也排除，并记录计数。原 YOLO 背景损失不增加忽略区域掩码，这一训练限制必须公开说明。

`preparation.json` 记录转换规则和原始 annotation 路径，`train.txt`/`val.txt`/`test.txt` 记录固定划分。正式数据已经准备完毕。训练日志旁的 `experiment.json` 包含实际训练参数、架构、解释器/CUDA/GPU、权重 SHA256、名单/标签/实现源码 SHA256。所有实验共享相同数据和增强设置，不覆盖已有训练目录。

AI-TOD-v2 暂未在本地发现对应图像和 v2 标注。接口接受官方 COCO 格式划分，禁止混用 AI-TOD-v1 标注：

```powershell
python code/test/prepare_data.py --dataset aitod --source H:/path/to/AI-TOD-images --coco-train H:/path/to/aitodv2_train.json --coco-val H:/path/to/aitodv2_val.json --coco-test H:/path/to/aitodv2_test.json
python code/test/train_test.py ablate --data datasets/nscr_aitod/dataset.yaml --epochs 100 --seeds 42 43 44
```

`--source` 需包含 COCO `file_name` 对应图像（也支持 train/val/test 或这些目录中的 images）。保留官方类别 ID 与划分；检查类别映射一致和跨划分图像重叠，不重新随机划分。

## 正式消融与恢复

### 加速选项

#### 内置 AutoBatch

本地 Ultralytics 8.3.163 支持 `--batch -1`，按默认约 60% CUDA 显存利用率估算 batch；它不是按训练吞吐量寻找最快配置。正式启动器及直接训练入口均支持：

```powershell
python code/test/run_formal.py --experiment E0 --seeds 42 --batch -1 --workers 8 --project runs/nscr/formal_autobatch
```

`workers=-1` 在这一版本没有自动选择含义，负数会导致 DataLoader 报错，入口会提前给出说明。workers 需要设为非负整数；要实测选择它，使用下面的可选联合调优。AutoBatch 会为每个实验分别估算 batch；正式消融需核对 `experiment.json` 中的实际 batch，并固定共同可用的配置以保证公平。

#### 自动选择 batch 与 workers

推荐先调优一次，检查报告，再启动正式训练。启动器自动选择 Anaconda `yolo` 解释器：

```powershell
# 测速及显存检查；不会启动 100 轮正式训练
python code/test/run_formal.py --auto-tune --tune-only --project runs/nscr/formal_auto

# 复用测出的配置，统一运行 E0–E6、种子 42/43/44
python code/test/run_formal.py --tuning-profile runs/nscr/formal_auto/tuning.json --project runs/nscr/formal_auto

# 只训练基准，一个种子；仍可复用上述覆盖全部实验的报告
python code/test/run_formal.py --experiment E0 --seeds 42 --tuning-profile runs/nscr/formal_auto/tuning.json --project runs/nscr/formal_auto
```

省略 `--tune-only` 时，`--auto-tune` 会在调优成功后直接训练。已有训练进程运行时调优会报错退出，不停止该进程，也不修改它的 batch/workers。等当前训练结束，或自行在保存检查点后停止，再测速。

默认候选 batch 为 `4/8/16/24/32`，workers 为 `0/4/8/12/16`，使用完整训练名单、640 输入和当前 cache/AMP 设置。每组候选单独创建进程，从同一份预训练权重开始，预热后测 20 个 batch 的真实增强、加载等待、GPU 传输、前向、反向、AdamW 更新和 EMA；不会保存探测模型为正式权重。按 **images/s** 比较，领先的三组再次独立测速，以中位数排序；速度相差 3% 以内优先较少的 workers 和较小的 batch。

默认以 E4 为测速模型，并对选出的配置检查所有请求的消融模型。高标注数量训练图片与两倍 batch 的验证前向/检测损失也参与显存检查；CUDA 峰值分配显存不得超过总显存的 80%。探测进程还对标签分配器的稠密临时张量做保守空间预估，超预算提前退出，避免 Windows 共享显存导致长时间停顿；这可能排除一些实际能够运行的组合。这个余量和抽样检查降低 OOM 风险，但不保证训练中所有增强组合的内存上限。结果是**候选范围内满足显存预算的优选配置**，不保证所有 batch/workers 中的全局最快；正式运行仍应观察完整 epoch 吞吐量。

默认扫描需要创建多个 Windows 子进程，可能花费数分钟至十几分钟。可扩大候选范围或增加计时步数，也可以先缩小范围：

```powershell
python code/test/run_formal.py --auto-tune --tune-only --tune-batches 8 16 24 --tune-workers 4 8 --tune-steps 40 --project runs/nscr/formal_auto
```

`tuning.json` 保存候选结果、速度、显存、硬件/代码/数据签名和验证过的实验；同目录的 `tuning_probes` 保存逐次日志。数据名单、权重、相关实现、分辨率、设备或 AMP/cache 设置改变后，启动器拒绝复用旧报告。正式实验记录包含调优报告路径及 SHA256。共同配置保证消融的 batch/workers 一致，但它是以参考模型测得的配置，各模型单独追求极限速度可能得到不同选择。

保持 `imgsz=640` 和 AMP，先尝试 `--batch 16 --workers 8`。batch=32 也可作为进一步测量选项，但 P2 模型及密集目标会增加显存使用，不能根据 E0 的显存直接保证 E4 同样可用。GPU 显存有余量不代表加大 batch 一定提速，应比较完整 epoch 耗时和每秒图像数（batch×it/s），并排除第一轮启动开销。

```powershell
# E0，尝试更大的 batch 和更多数据加载进程
python code/test/run_formal.py --experiment E0 --seeds 42 --batch 16 --workers 8 --cache none --no-plots

# 全部实验统一设置；新的结果目录与原 batch=8 实验分开
python code/test/run_formal.py --batch 16 --workers 8 --cache none --no-plots --project runs/nscr/formal_b16_w8
```

`--cache none|disk|ram` 在两个训练入口均可使用，默认 none。AMP 默认开启；`--no-plots` 仅省去图表输出，不改变每轮验证。batch 改变会影响 BatchNorm、warmup 和部分增强，nbs=64 的梯度累积只能尽量维持有效 batch，不能保证训练轨迹与 batch=8 相同。正式比较须将 E0–E6 的设置统一，或者将不同 batch 作为独立实验。

磁盘缓存减少重复 JPEG 解码，第一次建缓存需要时间和额外磁盘空间；本机完整 train+val 原图 `.npy` 估算约 29.92 GiB，H 盘剩余约 39 GiB，且上游磁盘容量检查还要求安全余量，所以当前不建议默认启用磁盘缓存。数据换到空间充足的 SSD 后可尝试 `--cache disk`。

本机 256 GiB RAM 可尝试 `--cache ram` 进行吞吐测量，但本版本对 RAM cache 给出非确定性提示；它也改变 Mosaic 从缓冲区或全数据池选择图像的路径。Windows 多进程还会增加内存占用。重视严格复现的正式实验优先使用 none 或空间充足时的 disk。

已经运行的训练不会因修改命令或脚本自动改变设置。本版本恢复训练只允许部分参数覆盖：`--batch` 可以覆盖，`workers/cache` 会恢复检查点内的设置。若仅提高 batch，可在保存检查点后使用：

```powershell
python code/test/train_test.py train --resume runs/nscr/formal/E0_seed42/weights/last.pt --batch 16
```

中途改变 batch 的运行属于混合配置，应在实验记录中说明。要统一应用新的 workers/cache 设置并进行公平消融，使用单独目录从头训练。

正式训练启动器 `run_formal.py` 优先使用用户目录下 `anaconda3/envs/yolo/python.exe`，即使用系统 Python 启动它，也会将训练交给 yolo 环境。其他安装位置可用 `--python` 指定解释器。默认完整 VisDrone、全部七项实验、三个种子、100 epochs，输出与测试目录分开。

若直接运行 `train_test.py`，必须使用 yolo 解释器，IDE 也应选择它。出现 `No module named yaml` 时，先核对命令开头的解释器路径。

```text
C:\Users\FXQJDYTH\anaconda3\envs\yolo\python.exe code/test/run_formal.py --experiment E0 --seeds 42
```

```powershell
# 全部基准与消融，共 21 次正式训练
python code/test/run_formal.py

# 分别跑一个种子，也可把 --seeds 42 改成 --seeds 42 43 44
python code/test/run_formal.py --experiment E0 --seeds 42
python code/test/run_formal.py --experiment E1 --seeds 42
python code/test/run_formal.py --experiment E2 --seeds 42
python code/test/run_formal.py --experiment E3 --seeds 42
python code/test/run_formal.py --experiment E4 --seeds 42
python code/test/run_formal.py --experiment E5 --seeds 42
python code/test/run_formal.py --experiment E6 --seeds 42
```

也可直接使用以下等价入口：

```powershell
python code/test/train_test.py ablate --experiments E0 E1 E2 E3 E4 E5 E6 --seeds 42 43 44 --epochs 100 --imgsz 640 --batch 8
python code/test/train_test.py train --resume runs/nscr/E4_seed42/weights/epoch10.pt
```

训练输出：`runs/nscr/E*_seed*/weights/best.pt`、`last.pt`、每 10 epoch 保存的检查点、`results.csv`、`summary.json`、`experiment.json`。完整结束时 Ultralytics 会去掉 best/last 的 optimizer；中断恢复用尚未被 strip 的 last 或周期检查点。此入口只支持单 GPU。

`ablation_summary.csv` 汇总一次调用中的运行；每个运行另有独立 summary，跨调用不会丢失。所有种子同等预算；如使用早停，应报告实际完成轮数，不能只报请求轮数。超参数只根据训练/验证决定，最终 test-dev 不用于选模型或选择超参数。

## 评价与推理

```powershell
python code/test/train_test.py evaluate --checkpoint runs/nscr/E4_seed42/weights/best.pt --data datasets/nscr_visdrone/dataset.yaml --split val --imgsz 640 --name E4_val
python code/test/train_test.py predict --checkpoint runs/nscr/E4_seed42/weights/best.pt --predict-source H:/path/to/image.jpg
```

评价需 `pycocotools==2.0.11`。本次已安装在项目内 `.research-deps`，未更改 yolo 环境全局依赖。其他机器可安装：

```powershell
python -m pip install --target .research-deps --no-deps -r code/test/requirements-research.txt
```

`evaluation.json` 分别报告 Ultralytics 的 Precision/Recall/mAP、补充 COCO AP-small/AR-small、Params/GFLOPs、按 q<τ / q≥τ 分组的原图微小目标 Recall、匹配 IoU、归一化中心误差和平均宽域门控。分组诊断采用置信度 0.25、IoU=0.5、同类一对一匹配；缺少有效样本时输出 null，不填假数值。

GPU 测速为融合模型、FP32、batch=1、30 次预热、100 次 CUDA Event 同步计时，报告 mean/p50/p95/FPS；输入为固定尺寸合成 RGB，仅测 forward。验证阶段另有预处理、forward、NMS 各阶段耗时，不能把 forward FPS 称为端到端视频 FPS。

**Ultralytics/COCO 补充指标不等于 VisDrone 官方评价。** COCO 忽略区域以各类别 crowd 框近似处理。`visdrone_official_predictions` 导出原图坐标的官方八列 txt；必须配合原始 annotations 和 [官方 DET 工具包](https://github.com/VisDrone/VisDrone2018-DET-toolkit) 的 `saveAnnoRes/calcAccuracy` 跑官方 AP50、AP50:95 和 AR500。评价目录另生成 `run_visdrone_official.m`，设置工具包路径后在 MATLAB 执行；不修改官方代码。AI-TOD-v2 的 tiny 子分组结果也需使用其官方评价扩展另行复核。

本实现不宣称已获得论文级精度提升，也不把一轮小样本运行当作消融结论。完整多种子训练、官方评价和 AI-TOD-v2 对照完成后再填结果章节。
