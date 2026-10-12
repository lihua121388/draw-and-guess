# S2工程审计

实际权重：`H:\JZ\code\ultralytics-8.3.163\runs\candidates\formal_100e_b8_s42_v2\M0_seed42\weights\best.pt`；SHA256 `ea7a1f071cfc5e0b4b3dc4016c38f36727b1f0302a9a9bc5c0192da39df1d16f`。

M0保持25层YOLO11n+P2，原COCO YOLO11n预训练、100轮、batch8、seed42；本阶段只准备从M0最佳权重初始化局部模型。

ROI逐图从S2原始记录冻结，548张的选区用原函数重新验证完全一致。候选池的前缀是已finalize的全局max500，后续按每个stage mapped_candidates切分局部候选，局部最多10000。原图xyxy，零起始类ID；逆gain/pad并加ROI偏移、裁剪到ROI/图像。最终类感知NMS .7、全图max500。所有来源可恢复，跨ROI统一NMS。

主AP沿用converted-positive Ultralytics无额外忽略过滤；COCO small采用classwise crowd近似；固定conf .25/IoU .5误检分析使用单独ignore区域IoA>=.5屏蔽；VisDrone Python union-raster协议单列，未经MATLAB交叉验证。

训练队列与实际进程保存在training_state.json。冻结清单、预设参数和历史文件哈希独立保存。融合函数不接收GT；诊断GT仅供事后分析。离线融合不会重新运行检测器。
