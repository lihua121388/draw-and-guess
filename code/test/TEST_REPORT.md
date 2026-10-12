# 测试记录

日期：2026-10-10。解释器为 Anaconda `yolo`，PyTorch 2.12.0+cu126，GPU 为 NVIDIA RTX 3090 24GB。

## 已执行

- 16 项专项测试全部通过：间距公式、尺度不变性、密集/稀疏梯度方向、单目标/空目标/无效框/忽略掩码、跨图隔离、随机标签、密度对照、E0–E6 反向传播、推理无需真实框、权重保存加载、原主干一致、预训练检测头映射、E2 参数预算、完整与子集名单隔离、AI-TOD-v2 转换、COCO AP-small/AR-small 的完美/空预测、分组匹配。
- E0–E6 均在真实 VisDrone 数据上完成：64 张 train、16 张 val、1 epoch、imgsz=320、batch=4、AMP。权重、CSV、配置、哈希记录和摘要正常保存。
- 正式启动器 `run_formal.py` 用 E4、workers=4、默认绘图参数完成一轮实际训练。Windows 多进程数据加载与 `results.png` 正常。绘图使用 Agg 后端保存文件。
- E4 的 COCO 补充评价、密度分组统计、官方 txt/MATLAB 桥接文件导出、CUDA Event 延迟测试完成。
- 完整正式数据名单再次核对：train=6471、val=548、test-dev=1610。正式默认 YAML 指向完整名单。

## 运行位置

| 内容 | 位置 |
|---|---|
| 专项测试日志 | `runs/nscr/tests.log` |
| 七项训练日志 | `runs/nscr/verified.log` |
| 七项汇总 | `runs/nscr/verified/ablation_summary.csv` |
| 各项权重 | `runs/nscr/verified/E*_seed42/weights/best.pt` |
| 评价报告 | `runs/nscr/verified/E4_evaluation/evaluation.json` |
| 正式启动器验证日志 | `runs/nscr/launcher_verified.log` |
| 启动器验证图表 | `runs/nscr/launcher_verified/E4_seed42/results.png` |

## 参数量核对

按未融合训练模型统计，10 类、nano 宽度：E0 为 2,591,790；E1 为 2,688,760；E2 为 2,695,516；E3/E4/E5/E6 均为 2,695,466。E2 与 E3 相差 50 参数。融合 BN 后参数量会变化，评价报告将 fused 与未融合统计分开。

## 测试结论的范围

上述一轮小样本运行的 mAP 均为 0，不能证明算法精度或科学假设。这里只确认训练、梯度、存储、加载、推理和评价流程可执行。正式多种子训练没有在本次测试中运行，`runs/nscr/formal` 为正式结果目录。

VisDrone 官方 MATLAB 评价只完成桥接文件生成，尚未执行官方工具包；COCO 指标只作为补充。AI-TOD-v2 已验证转换接口，真实 v2 数据尚未在本机提供，没有生成该数据集的正式实验结果。

## Batch/workers 自动选择入口

- `test_autotune.py` 的 8 项检查通过：images/s 排序、显存预算和失败候选排除、复测中位数、近似并列优先较少 workers、报告兼容性、正在训练进程识别、跨 epoch 连续采样和空加载器处理。
- 原 NSCR 的 16 项专项测试再次通过。
- CPU 上 E4、64 张训练子集、imgsz=64、batch=8、workers=0 的真实预热/反向/EMA/密集图片及双倍 batch 验证探测通过；预热超过一个 epoch 的场景正常。
- `run_formal.py --experiment E0 --seeds 42 --batch -1 --workers 8 --dry-run` 正常转发原生 AutoBatch，并选择 Anaconda yolo。`workers=-1` 提前拒绝，说明该版本没有此自动模式。这是参数传递检查，没有启动 100 轮训练。
- 可选联合调优入口在完整训练名单、imgsz=320、单一候选 batch=8/workers=4 上完成 GPU 端到端验证：E4 两次独立测速、E0 配置检查、密集图片/双倍验证、完整 JSON 报告和正式启动器报告复用检查通过。报告位于 `runs/nscr/autotune_functional.json`；单一候选、每次仅 3 个计时 batch，只用于功能检查，不是最佳配置搜索。
- 完整 VisDrone、640 输入的部分压力测试测得 E4 的 batch=8/workers=0 约 23.8 images/s，workers=4 约 58.4 images/s。batch=16 的密集双倍验证触及 Windows 共享显存，测试已停止；这份不完整报告没有给出推荐值，不可用于正式训练。探测现加入标签分配临时张量的保守预算检查，超预算提前退出。
- 上述短时测速用于验证入口和发现内存风险，不代表已经确定本机 640 输入下的最佳参数。原生 `batch=-1` 也只是显存估算，不保证吞吐量最快；正式消融应固定共同可用的 batch/workers。
