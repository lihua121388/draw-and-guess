# VisDrone 原图区域裁剪可行性实验

本阶段只执行推理、验证和分析，所有训练保持停止。M0 权重、模型结构、原图及标注均不修改。

工程审计、10组全量主评估及8组固定NMS敏感性分析均已完成，目录：`runs/crop_study/c0_c1_c2_20261012/`。[完整报告](../../runs/crop_study/c0_c1_c2_20261012/REPORT.md)包含原始预测、每图耗时、覆盖曲线、十类AP和五类案例。12项测试通过，权重和图像/标注哈希保持不变。

当前固定切片C1-B主AP为23.72%，低于960整图C0的24.67%，平均135.85ms对30.29ms；GT Oracle1/2/4的主AP为24.86/24.75/24.62%。当前不建议直接训练ROI前置网络。后续按用户要求开展统一640的[S0–S4有限ROI实验](BUDGET_CROP_STUDY.md)，不覆盖本次输出。

## 实验与可复用代码

- `crop_geometry.py`：动态尺寸固定切片、贴边窗口、逆缩放/padding/ROI偏移、完整包含覆盖、预算贪心Oracle、面积并集。
- `crop_infer.py`：未改动M0的FP32推理、显式阶段一NMS、原图坐标类别感知融合、GPU同步计时。
- `crop_evaluate.py`：保存预测统一评分，Ultralytics主指标、COCO small指标、固定原图尺寸诊断、单独的VisDrone Python参考协议。
- `run_crop_study.py`：审计→C0复现→固定20图检查→C1三组→C2的1/2/4 ROI→评分→NMS敏感性→报告。
- `report_crop_study.py`：整体/十类/尺寸/覆盖/计算成本表，预算曲线和固定seed案例。
- `test_crop_study.py`：几何、边界、空输出、预算、类别感知NMS和矩形输入测试。

C0：960整图。C1-A：固定640切片。C1-B：960整图+固定切片。C1-C：640整图+固定切片。C2-A/B：仅Oracle局部 / 960整图+Oracle，均验证预算1、2、4；仅作GT选区潜力诊断。

## 口径与成本

历史960主AP来自`val`；保存的COCO预测及small指标来自`predict`。二者多类别候选、缩放和padding不同。历史保存预测重新计算主AP为24.14%，不能直接将它对应到历史`val`的24.68%。新实验重新复现`val`的预处理/候选策略，各组都对同一套保存预测进行离线评分。small口径差异会明确披露。

主融合IoU=0.7，保持C0口径；附加0.5、0.6只作固定敏感性分析，不替换主结果。所有区域conf=0.001，ROI内IoU=0.7；阶段一最大NMS候选30000，中间输出10000，映射和融合之后每张原图最多500。候选截断、边界预测和各阶段耗时逐图保存，不将各ROI的maxDet当成整图限制。

推理全局分支沿用历史`val`排序后的batch8矩形尺寸（只由图像尺寸决定），逐图batch1实际执行，以测量端到端延迟。ROI从原始图像裁剪，局部输入固定640平方，重叠20%。Oracle只依据有效GT几何选区，内部边界安全距离8px，不更改检测框、类别、置信度或根据GT过滤预测。GT覆盖率在推理完成后统计，C0/C1选区和融合不读取GT决策。

逐图端到端时间包含解码、选区、裁剪、预处理、前向、ROI内NMS、坐标恢复、跨ROI融合。模型加载、预热、预测序列化和评分不计入延迟。每方案独立实际执行，不使用共享推理缓存合成延迟。全量运行前后校验权重、图像、有效标签、原始标注和验证名单哈希。

## 新目录完整复现

```bat
C:\Users\FXQJDYTH\anaconda3\envs\yolo\python.exe code\test\test_crop_study.py
C:\Users\FXQJDYTH\anaconda3\envs\yolo\python.exe -u code\test\run_crop_study.py --output runs\crop_study\new_repeat --stage all
```

也可以按`--stage audit/c0/smoke/c1/c2/score/sensitivity/report`分阶段执行。已有推理组拒绝覆盖；C1/C2要求C0复现和20图检查已经通过。可选ROI尺寸/重叠扩展不在本次主实验中进行，以优先完成固定预设全量比较。
