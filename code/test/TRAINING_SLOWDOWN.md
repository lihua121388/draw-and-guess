# batch16 训练卡顿诊断

检查日期：2026-10-11。对象：`runs/fusion/formal_100e_b16_s42/F0_seed42`，训练PID18020，队列PID1812。本轮只读诊断训练进程，未修改batch、模型、增强或标签分配器。

## 结论

主要证据指向：密集标注与P2使训练期间的临时张量需求超过24GiB显存，Windows/NVIDIA驱动使用共享系统内存维持运行，产生严重卡顿。已直接确认共享内存占用；标签分配器是代码中明显的内存放大环节，但尚未对正在运行的进程做逐算子GPU profiling，不能声称18分钟全部发生于某个已测量函数。

## 实测证据

|项目|检查结果|
|---|---|
|GPU|RTX3090，物理显存24576MiB，nvidia-smi占用约24034MiB|
|训练进程独立显存|Windows GPUProcessMemory计数器：24426913792字节，约22.75GiB|
|训练进程共享系统内存|9852420096字节，约9.18GiB|
|GPU提交总量|34279333888字节，约31.93GiB|
|训练器日志GPU_mem|卡顿批次30.7G，历史最高39.3G；这是PyTorch memory_reserved统计，不能当作显卡物理容量|
|最明显停顿|第23轮第114个batch附近，Instances=2054，累计进度时间从00:43跳至19:05，约1102秒|
|普通批次速度|卡顿前约5.4–5.8batch/s；不是所有batch持续缓慢|
|其他卡顿例子|第15轮Instances=2419时153秒，第3轮Instances=1981时163秒|
|显式CPU标签分配回退|日志中无`CUDA OutOfMemoryError ... using CPU`提示|
|AMP / Mosaic|AMP已通过检查；mosaic=1，最后10轮关闭|
|本机系统RAM|约256GiB，检查时可用约212GiB，未见系统RAM不足证据|
|GPU状态|约71°C、SM时钟1905MHz；没有明显严重降频证据|
|已有batch8速度参照|M0完成22轮CSV累计2287.78秒；本次F0为7447.35秒，约3.25倍耗时|

计数器命令（读取可能需要沙箱外权限；本轮已成功只读查询）：

```powershell
Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | Where-Object { $_.Name -match 'pid_18020_' } | Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted
```

共享系统内存是供GPU使用的主机内存，与Ultralytics收到CUDA OOM后把整个标签分配计算移到CPU的回退路径不同。本次有前者的计数器证据，没有后者的日志证据。NVIDIA说明，接近GPU显存上限时可使用共享系统内存保持程序运行，但速度会下降：[NVIDIA官方说明](https://nvidia.custhelp.com/app/answers/detail/a_id/5490)。WDDM的GPU虚拟地址可对应显存或系统内存：[Microsoft驱动文档](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-virtual-memory-in-wddm-2-0)。

## 为什么此任务会产生大内存需求

1. 640输入下，P2/P3/P4/P5共有`160²+80²+40²+20²=34000`个候选位置；原P3/P4/P5只有8400个，增加约4.05倍。新增参数少，不代表训练期间标签分配内存少。
2. `ultralytics/utils/loss.py:226`把每张图的GT补齐到当前batch的单图最大GT数`G_max`。内存放大取决于最密集的一张增强后图像，不只取决于日志中整个batch的Instances总数。
3. `ultralytics/utils/tal.py:173`创建`[B,G_max,A]`的IoU与分类分数张量，后续还创建对齐指标和多个mask；`:298`创建`[B,G_max,A,4]`的位置差临时张量。主要规模与`B×G_max×A`相关。GT与部分mask使用FP32，AMP不自动把这一切压到FP16。
4. batch从8增至16使B加倍；Mosaic叠加多个图像的标注，使某些图像的G_max更大。在密集场景中因此会出现少数批次的尖峰。

仅作数量级示例，**不是已测得该批次G_max=800**：若B=16、G_max=800、A=34000，一份FP32 `[B,G_max,A]`张量约1.62GiB，四分量临时张量约6.48GiB；标签分配中有多个同时存活的张量，还要加模型激活、反向、优化器与缓存。

F0没有新增融合门控，没有间距监督，spacing_loss为0；当前卡顿不能归因于尚未开始训练的F2/F3模块。图像访问抽样约1–1.9GB/s，普通批次仍可达到5batch/s，现有证据不指向磁盘读取或workers数量是此次长停顿的主要原因。

## 可行处理方向

- 恢复batch8：已有完整100轮M0的实际速度依据。需要保留本次F0结果并建立一致的batch8对照队列，不能混用batch16与batch8结果归因模块收益。
- 保留batch16：先实现标签分配按图像或小块计算，保持分配规则与损失归一化不变；验证输出、损失与梯度一致后，再以统一实现运行所有对照。当前代码没有进行这项修改。
- 增加workers或图像cache不能解决标签分配显存尖峰。自动batch测试也应覆盖稠密增强标注，单凭普通图像或纯前向测试容易高估可用batch。
- 关闭驱动共享内存回退不能降低显存需求，可能使超量批次直接OOM；应先降低内存需求，而不是把驱动开关当作提速方法。

当前队列仍按用户指定batch16运行。任何训练配置或实现调整均未在本轮自动执行。
