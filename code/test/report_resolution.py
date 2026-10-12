"""Build frozen M0 diagnosis tables, exportable figures and deterministic detection examples.

Uses saved predictions only: never trains or runs fresh model inference.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("R0_640", "R1_960", "R2_1280")
BINS = ("<8", "8-16", "16-32", ">=32")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pct(value):
    return "—" if value is None else f"{100*value:.2f}"


def table(headers, rows):
    return "\n".join(["|"+"|".join(headers)+"|", "|"+"|".join("---" for _ in headers)+"|",
                      *["|"+"|".join(map(str, row))+"|" for row in rows]])


def link(path, label):
    return f"[{label}]({path.relative_to(ROOT).as_posix()})"


def save_figure(fig, base):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(base.with_suffix("."+extension), dpi=180, bbox_inches="tight")
    plt.close(fig)


def figures(project, records):
    output = project / "figures"
    output.mkdir(exist_ok=True)
    sizes = [r["provenance"]["imgsz"] for r in records]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for key, label in (("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50:95")):
        axes[0].plot(sizes, [100*r["evaluation"]["ultralytics"][key] for r in records], "o-", label=label)
    for key, label in (("AP_small", "AP-small"), ("AR_small", "AR-small")):
        axes[1].plot(sizes, [100*r["evaluation"]["coco_supplemental"][key] for r in records], "o-", label=label)
    axes[2].plot(sizes, [r["evaluation"]["forward_benchmark"]["mean_ms"] for r in records], "o-", label="FP32 forward, batch1")
    for i, ax in enumerate(axes):
        ax.set_xlabel("Inference imgsz (weights trained at 640)")
        ax.set_ylabel("Percent" if i < 2 else "Milliseconds")
        ax.legend(); ax.grid(alpha=.25); ax.set_xticks(sizes)
    axes[0].set_title("Ultralytics main protocol")
    axes[1].set_title("Supplemental COCO, original area")
    axes[2].set_title("RTX 3090, excludes NMS / IO")
    fig.tight_layout()
    save_figure(fig, output / "accuracy_cost")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for record in records:
        d = record["diagnosis"]
        label = str(record["provenance"]["imgsz"])
        axes[0].plot(BINS, [100*d["original_size"][b]["recall"] for b in BINS], "o-", label=label)
        axes[1].plot(BINS, [100*(d["density_size"][b]["sparse"]["recall"]-d["density_size"][b]["dense"]["recall"]) for b in BINS], "o-", label=label)
    axes[0].set_ylabel("Recall (%) at conf .25 / IoU .5")
    axes[1].set_ylabel("Sparse minus dense recall (pp)")
    for ax in axes:
        ax.set_xlabel("Original sqrt(box area), pixels")
        ax.legend(title="Inference imgsz"); ax.grid(alpha=.25)
    axes[1].axhline(0, color="grey", linewidth=.7)
    fig.suptitle("Frozen M0; converted GT; descriptive grouping, not a causal ablation")
    fig.tight_layout()
    save_figure(fig, output / "size_density")


def examples(project, records, gt):
    """Same hash-selected image and crop across sizes; no selection based on improvements."""
    output = project / "visualizations"
    output.mkdir(exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    heading_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    positives = collections.defaultdict(list)
    for a in gt["annotations"]:
        if not a.get("iscrowd"):
            positives[a["image_id"]].append(a)
    candidates = [m for m in gt["images"] if any(a["bbox"][2]*a["bbox"][3] < 32**2 for a in positives[m["id"]])]
    candidates.sort(key=lambda m: hashlib.sha256(f"42:{m['file_name']}".encode()).hexdigest())
    selected = candidates[:4]
    predictions = []
    for record in records:
        by_image = collections.defaultdict(list)
        for p in read(record["directory"] / "coco_predictions.json"):
            if p["score"] >= .25:
                by_image[p["image_id"]].append(p)
        predictions.append(by_image)
    selection = []
    palette = ((255, 180, 40), (250, 95, 120), (60, 210, 255), (110, 240, 80), (200, 130, 255),
               (255, 220, 140), (160, 210, 255), (255, 160, 210), (140, 250, 210), (240, 220, 70))
    for meta in selected:
        image = Image.open(ROOT / "datasets/nscr_visdrone/val/images" / meta["file_name"]).convert("RGB")
        # Anchor on the smallest positive GT, independent of predicted improvements.
        target = min(positives[meta["id"]], key=lambda a: a["bbox"][2]*a["bbox"][3])
        x, y, w, h = target["bbox"]
        edge = min(256, image.width, image.height)
        left = max(0, min(image.width-edge, round(x+w/2-edge/2)))
        top = max(0, min(image.height-edge, round(y+h/2-edge/2)))
        crop = (left, top, left+edge, top+edge)
        selection.append(dict(image=meta["file_name"], image_id=meta["id"], crop_original_xyxy=crop,
                              anchor_gt_id=target["id"], selection="first4 seed42 hash-order; smallest GT anchor"))
        full_panels, roi_panels = [], []
        for record, predicted in zip(records, predictions):
            canvas = image.copy(); draw = ImageDraw.Draw(canvas)
            for a in positives[meta["id"]]:
                bx, by, bw, bh = a["bbox"]
                draw.rectangle((bx, by, bx+bw, by+bh), outline=(160, 160, 160), width=1)
            for p in predicted[meta["id"]]:
                bx, by, bw, bh = p["bbox"]
                color = palette[(p["category_id"]-1) % len(palette)]
                draw.rectangle((bx, by, bx+bw, by+bh), outline=color, width=2)
                # Numeric class id and confidence; full-resolution files retain readable text.
                if bw >= 40 and p["score"] >= .7:
                    draw.text((bx, max(0, by-16)), f"{p['category_id']}:{p['score']:.2f}", fill=color, font=font,
                              stroke_width=1, stroke_fill=(0, 0, 0))
            size = record["provenance"]["imgsz"]
            directory = output / str(size); directory.mkdir(exist_ok=True)
            canvas.save(directory / meta["file_name"])
            def panel(picture):
                picture = picture.copy(); picture.thumbnail((600, 380))
                tile = Image.new("RGB", (600, 420), (22, 25, 30))
                tile.paste(picture, ((600-picture.width)//2, 40+(380-picture.height)//2))
                ImageDraw.Draw(tile).text((12, 9), f"imgsz={size}   conf>=0.25   gray=GT", fill="white", font=heading_font)
                return tile
            full_panels.append(panel(canvas))
            roi = canvas.crop(crop).resize((380, 380))
            roi_panels.append(panel(roi))
        sheet = Image.new("RGB", (600*len(records), 876), (22, 25, 30))
        for column, (full, roi) in enumerate(zip(full_panels, roi_panels)):
            sheet.paste(full, (600*column, 0)); sheet.paste(roi, (600*column, 420))
        legend = ImageDraw.Draw(sheet)
        for index, category in enumerate(gt["categories"]):
            left = index * 180
            legend.rectangle((left+5, 851, left+17, 865), fill=palette[index])
            legend.text((left+22, 850), f"{index+1}:{category['name']}", font=font, fill="white")
        sheet.save(output / (Path(meta["file_name"]).stem+"_comparison.jpg"), quality=94)
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return selection


def audit(project, records):
    candidates = ROOT / "runs/candidates/formal_100e_b8_s42_v2"
    reference = read(candidates / "M0_seed42/experiment.json")
    configuration = reference["architecture"]
    layers = configuration["backbone"] + configuration["head"]
    profile = read(ROOT / "runs/fusion/profile.json")["models"]
    models = {}
    for name in ("F0", "F1", "F2_GLOBAL", "F2", "F3_PLAIN", "F3"):
        path = Path(profile[name]["config"])
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        modified = cfg["backbone"] + cfg["head"]
        differences = [i for i, (a, b) in enumerate(zip(layers, modified)) if a != b]
        assert len(modified) == len(layers) == 25 and differences == ([] if name == "F0" else [23]), (name, differences)
        assert cfg["nscr"]["weight"] == 0 and cfg["nscr"]["supervision"] == "none"
        models[name] = dict(config=str(path), sha256=digest(path), changed_layers=differences,
                            params=profile[name]["params"], GFLOPs=profile[name]["GFLOPs"],
                            transferred_tensors=profile[name]["weight_transfer"]["transferred_tensors"])
    shared_keys = ("checkpoint_sha256", "dataset_yaml_sha256", "val_manifest_sha256", "annotation_sha256", "image_paths")
    for record in records:
        for key in shared_keys:
            assert record["provenance"][key] == records[0]["provenance"][key], key
        assert digest(Path(record["provenance"]["checkpoint"])) == record["provenance"]["checkpoint_sha256"]
        assert len(read(record["directory"] / "evaluated_image_ids.json")) == 548
        d = record["diagnosis"]
        assert d["images"] == 548 and d["all"]["targets"] == 38587
        for name in BINS:
            assert sum(row["targets"] for row in d["density_size"][name].values()) == d["original_size"][name]["targets"]
    historical = read(candidates / "M0_seed42/evaluation/evaluation.json")
    assert historical["ultralytics"] == records[0]["evaluation"]["ultralytics"]
    assert historical["coco_supplemental"]["AP_small"] == records[0]["evaluation"]["coco_supplemental"]["AP_small"]
    durations = {}
    for name in ("M0", "M1", "M2", "M3"):
        rows = list(csv.DictReader((candidates / f"{name}_seed42/results.csv").open(encoding="utf-8")))
        durations[name] = dict(epochs=int(rows[-1]["epoch"]), total_seconds=float(rows[-1]["time"]))
    queue = read(candidates / "queue.json")
    code_paths = ("ultralytics/nn/modules/p2_fusion.py", "code/test/fusion_experiments.py",
                  "code/test/train_fusion.py", "code/test/run_fusion.py", "code/test/train_test.py",
                  "code/test/evaluate_nscr.py", "code/test/resolution_diagnosis.py", "code/test/diagnose_fusion.py")
    result = dict(reference_project=str(candidates), formal_fusion_started=(ROOT / "runs/fusion/formal_100e_b8_s42").exists(),
                  old_queue_status=queue["status"], models=models, durations=durations,
                  pretrained_checkpoint_sha256=digest(ROOT / "model/yolo11n.pt"),
                  expected_pretrained_sha256=reference["checkpoint_sha256"],
                  current_code_sha256={p: digest(ROOT / p) for p in code_paths},
                  all_resolution_hashes_and_membership_equal=True, historical_640_metrics_exactly_reproduced=True,
                  dry_run_log=str(ROOT / "runs/fusion/fusion_plan_dry_run.log"))
    assert result["pretrained_checkpoint_sha256"] == result["expected_pretrained_sha256"]
    (project / "engineering_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def report(project, records, features, audit_data, selection):
    evaluations = [r["evaluation"] for r in records]
    sizes = [r["provenance"]["imgsz"] for r in records]
    lines = ["# M0 固定权重：分辨率与误差来源诊断", "", "日期：2026-10-11。正式训练保持停止；本次仅验证、诊断和形成待授权计划。",
        "", "## 1. 工程审计与可比性", "",
        "已读取实际代码、YAML、实验元数据、训练 CSV 和队列。M0—M3 均完成 100 轮；M4 停在 18 轮，M5 未开始。旧队列为 `stopped_by_user`；没有启动正式融合队列。",
        "M0 第 23 层实际是 `Conv1×1/BN/SiLU(Concat(P2, nearest-U(P3)))`，不是 Concat+C3k2。第 2/16/19/22 层分别提供 P2/P3/P4/P5；第 24 层 Detect 接收 23/16/19/22 四尺度。",
        "F1—F3 及其对照配置逐层比对：仅第 23 层发生变化，0—22 层与 Detect 完全相同；F0 全部结构与 M0 相同。没有新增辅助监督。现有形状、梯度、预训练映射、保存加载检查保持有效。本轮仅扩充评估输出和诊断脚本，没有改动融合模块结构。",
        "COCO 预训练仍使用本地 `model/yolo11n.pt`：448 个兼容张量迁移，原 P3/P4/P5 检测分支映射到新头的 1/2/3 分支，P2 与 10 类输出按原策略初始化。`pretrained=False` 是禁用框架自动加载，不代表没有执行项目的显式迁移。",
        f"M0 best.pt SHA256：`{records[0]['provenance']['checkpoint_sha256']}`。",
        "三组均为同一份 640 训练权重、同一批 548 张验证图像和相同标注；所有权重、数据 YAML、验证名单及 COCO 标注哈希完全一致。640 主指标与历史评估精确复现。验证 batch=8/workers=4/device=0，未因显存调整 batch。",
        "保持原有矩形 letterbox、FP32 独立评估、预测 conf=0.001、NMS IoU=0.7、maxDet=500；仅 imgsz 改变。诊断另固定 conf=0.25、IoU=0.5。AP 使用完整置信度曲线，不通过降低诊断阈值宣称 AP 提升。",
        "Ultralytics 主指标、补充 COCO small 指标和 VisDrone Python 参考指标分别解释。补充 COCO ignored region 使用 classwise crowd 近似；Python 参考实现尚未与 MATLAB 原工具逐项对齐。本报告不称这些为官方 VisDrone 成绩。",
        "", "## 2. 分辨率结果", ""]
    lines += [table(["推理 imgsz", "mAP50 %", "mAP50:95 %", "AP-small %", "AR-small %", "P %", "R %"],
        [[size, pct(e["ultralytics"]["metrics/mAP50(B)"]), pct(e["ultralytics"]["metrics/mAP50-95(B)"]),
          pct(e["coco_supplemental"]["AP_small"]), pct(e["coco_supplemental"]["AR_small"]),
          pct(e["ultralytics"]["metrics/precision(B)"]), pct(e["ultralytics"]["metrics/recall(B)"])] for size, e in zip(sizes, evaluations)]), "",
        table(["imgsz", "参数", "GFLOPs 未融合 / 融合", "forward ms / FPS", "P95 ms", "Torch 峰值 allocated / reserved GiB"],
        [[size, e["params"], f"{e['GFLOPs']:.2f} / {e['fused_GFLOPs']:.2f}",
          f"{e['forward_benchmark']['mean_ms']:.2f} / {e['forward_benchmark']['FPS']:.1f}",
          f"{e['forward_benchmark']['p95_ms']:.2f}",
          f"{r['provenance']['peak_allocated_GiB']:.3f} / {r['provenance']['peak_reserved_GiB']:.3f}"] for size, e, r in zip(sizes, evaluations, records)]), "",
        "延迟：RTX3090，FP32，batch1，方形零 RGB 输入，模型融合后纯前向，30 次预热、100 次 CUDA Event 测量；不含预处理、NMS、磁盘和主机传输。640与960本次纯前向接近，可能受小模型算子启动开销及测量噪声影响，不代表提高分辨率没有成本；960/1280理论运算量分别为640的2.25/4倍。验证器批量处理的逐图平均速度如下，不能与 batch1 延迟互换。Torch 峰值是本进程张量分配/保留显存，不是 GPU 总占用，也不代表训练峰值。",
        table(["imgsz", "val预处理 ms/图", "val前向 ms/图", "val后处理 ms/图"],
              [[s, f"{e['validation_speed_ms']['preprocess']:.3f}", f"{e['validation_speed_ms']['inference']:.3f}",
                f"{e['validation_speed_ms']['postprocess']:.3f}"] for s, e in zip(sizes, evaluations)]),
        f"![精度与成本](../../{(project/'figures/accuracy_cost.png').relative_to(ROOT).as_posix()})", "",
        "这是跨分辨率推理诊断，不是 960/1280 匹配训练。没有 TTA、新模型或切片融合。提高 imgsz 会改变目标像素数、计算量及上下文采样；结果支持尺度敏感性，不能证明输入尺度是唯一瓶颈，也不能作为新的融合模块收益。",
        "", "## 3. 十类 AP", "", "以下均为 Ultralytics 主评估口径，每格 AP50 / AP50:95（%）。", ""]
    maps = [{c["name"]: c for c in e["ultralytics_per_class"]} for e in evaluations]
    classes = list(maps[0])
    lines += [table(["类别", *map(str, sizes)], [[c, *[f"{pct(m[c]['AP50'])} / {pct(m[c]['AP50_95'])}" for m in maps]] for c in classes]),
        "", "## 4. 尺寸、类别与密度", "",
        "使用固定原图 GT 分组，等效尺寸 s=sqrt(w×h)，区间采用半开区间 [0,8)、[8,16)、[16,32)、[32,+∞)。以下 Recall 为同类一对一 IoU≥0.5 且 conf≥0.25；不等同于 AP，也不等同于主评估自动选取工作点的 Recall。", "",
        table(["原图尺寸", "GT 数", *[f"{s} Recall %" for s in sizes]],
              [[b, records[0]["diagnosis"]["original_size"][b]["targets"], *[pct(r["diagnosis"]["original_size"][b]["recall"]) for r in records]] for b in BINS]), "",
        "输入尺寸变化会让同一目标跨组；因此分辨率收益主要依据固定原图分组判断。输入空间尺寸分布仅用于描述像素预算：", "",
        table(["imgsz", "输入 <8 像素 GT", "输入 8–16 像素 GT", "输入 <16 占比"],
              [[s, r["diagnosis"]["input_size"]["<8"]["targets"], r["diagnosis"]["input_size"]["8-16"]["targets"],
                pct(sum(r["diagnosis"]["input_size"][b]["targets"] for b in BINS[:2])/38587)] for s, r in zip(sizes, records)]), "",
        "小目标原图 s<32 的分类召回（每格 GT数 / Recall%）：", ""]
    small_rows = []
    for c in classes:
        cells = []
        for r in records:
            rows = [r["diagnosis"]["class_size"][c][b] for b in BINS[:3]]
            count = sum(v["targets"] for v in rows)
            cells.append(f"{count} / {pct(sum(v['matched'] for v in rows)/count if count else None)}")
        small_rows.append([c, *cells])
    lines += [table(["类别", *map(str, sizes)], small_rows), "",
        "密度：在原图所有有效实例之间计算尺度归一化最近邻间距 q；q<2 为密集，q≥2 为稀疏，单目标归未定义组。每个原图尺寸区间内分别比较，仍不能消除区间内尺寸、类别及场景差异，因此不作为密度的因果证据。每格 dense / sparse Recall%。", "",
        table(["原图尺寸", "dense / sparse GT数", *map(str, sizes)],
              [[b, f"{records[0]['diagnosis']['density_size'][b]['dense']['targets']} / {records[0]['diagnosis']['density_size'][b]['sparse']['targets']}",
                *[f"{pct(r['diagnosis']['density_size'][b]['dense']['recall'])} / {pct(r['diagnosis']['density_size'][b]['sparse']['recall'])}" for r in records]] for b in BINS]), "",
        f"![尺寸与密度](../../{(project/'figures/size_density.png').relative_to(ROOT).as_posix()})", "",
        "## 5. 漏检的互斥诊断", "",
        "按 conf≥0.25 未召回的 GT 计比例。逐步归类：降至0.05能恢复 → 降至0.001能恢复 → 已有高置信同类框但一对一分配竞争 → 未用于正确匹配的高置信错类框 → 未用于匹配的高置信同类 IoU∈[0.1,0.5)框 → 无上述输出。各类互斥且总和等于漏检数。", ""]
    labels = dict(low_conf_005_025="低置信（0.05–0.25）", very_low_conf_0001_005="极低置信（0.001–0.05）",
                  assignment_competition="同类分配竞争", wrong_class_proxy="错类匹配代理", localization_proxy="定位不足代理", no_qualifying_output="无上述合格输出")
    lines += [table(["漏检类型", *[f"{s} 数量 / 占漏检 %" for s in sizes]],
              [[label, *[f"{r['diagnosis']['all']['errors'][key]} / {pct(r['diagnosis']['all']['error_fraction_of_misses'][key])}" for r in records]] for key, label in labels.items()]), "",
        table(["conf", *[f"{s} Recall %" for s in sizes]],
              [[c, *[pct(r["diagnosis"]["thresholds"][c]["recall"]) for r in records]] for c in ("0.001", "0.05", "0.1", "0.25")]), "",
        "这些是预测输出的代理诊断，分类优先级会影响占比，不是互相独立的真实成因分解。低阈值增加假阳性，不能据召回上涨宣称性能提升；NMS和maxDet截断也可能影响漏检。错类框排除已匹配正确 GT 的预测，避免把旁边已检测目标误算为当前目标混淆。ignored regions 沿用历史单矩形 IoA≥0.5 过滤，非官方并集栅格规则。",
        "", "## 6. P2/P3 输入贡献与特征统计", "",
        "64 张 seed42 按文件名哈希固定抽样；只在640推理时关闭P2头或置零融合输入。每组使用完全相同的图像和 GT。它们会产生分布偏移和 NMS 变化，是敏感性检查，不是重新训练的机制消融。", ""]
    lines += [table(["推理干预", *[f"{b} Recall %" for b in BINS]],
              [[name, *[pct(row["size_bins"][b]["recall"]) for b in BINS]] for name, row in features["scenarios"].items()]), "",
        table(["特征层", "small RMS", "background RMS", "small/background"],
              [[i, f"{row['small_rms_mean']:.4f}", f"{row['background_rms_mean']:.4f}", f"{row['small_to_background_ratio']:.4f}"] for i, row in features["feature_energy"].items()]), "",
        "特征RMS的small沿用历史原图面积≤32²定义，尺寸分组采用半开区间，两者在恰好32像素边界处略有差别。P2 与 P3 输入都参与当前训练模型的有效预测，但不能由置零结果断定某个尺度应该全局增权。不同层激活能量分布不同；RMS 比值不能证明融合后语义退化，也不能单凭它设计新增模块。",
        "", "## 7. 检测可视化与复现", "",
        "固定前4张含小目标图像的 seed42 文件名哈希序，不按结果好坏挑选。三组使用相同原图和 ROI；灰框为转换后的有效 GT，彩框为类别颜色预测（conf≥0.25），底部提供类别图例。为避免密集小框标签重叠，仅较大且高置信的框显示类别1–10及置信度；不会因此移除低置信框。原图高分辨率版本也分别保存，ROI 以最小 GT 为锚点，不是切片推理。", ""]
    for item in selection:
        image = project / "visualizations" / (Path(item["image"]).stem+"_comparison.jpg")
        lines.append(f"[{item['image']} 对比](../../{image.relative_to(ROOT).as_posix()})")
    lines += ["", "```bat", "python code\\test\\validate_resolution.py --imgsz 960 --output runs\\fusion\\resolution_m0_repeat\\R1_960",
              "python code\\test\\resolution_diagnosis.py --project runs\\fusion\\resolution_m0_20261011",
              "python code\\test\\diagnose_fusion.py --limit 64 --output runs\\fusion\\resolution_m0_20261011\\feature_size_640",
              "python code\\test\\report_resolution.py", "```", "",
        "复验需使用新的输出目录；图表脚本只读取保存预测，不再次推理。完整 per-class/per-size/密度及错误数据在各组 `error_diagnosis.json`；环境与成员、哈希和显存在 `resolution_provenance.json`。工程审计为 `engineering_audit.json`。",
        "", "## 8. 当前结论与下一步", "",
        "已完成的 M1—M3 只有 -0.08、+0.11、+0.04 pp 主 mAP50:95 变化，没有证明几何监督机制有效，不继续堆叠这些模块。当前固定权重分辨率结果显示：增加目标输入像素是一条有明确验证收益的方向，需要同时承担更高计算和延迟。",
        "640→960 的主mAP50:95提高4.31pp，补充AP-small提高4.68pp；1280相对960主mAP仅再提高0.55pp，运算量增加约78%。960因此是更优先的下一轮尺度基线候选，但具体选择仍取决于部署预算，不能保证960匹配训练一定进一步提升。",
        "尺度证据主要来自小目标：原图<8像素Recall从5.47%升至17.11%/24.92%，8–16从23.92%升至37.67%/44.58%；而≥32组从960的76.51%降到1280的75.55%。输入<16像素目标占比从640的69.78%降至960的46.75%。分辨率变化并非对所有尺寸和类别单调有利。",
        "640无上述合格输出的漏检代理有9744个，960降为6742个；低置信0.05–0.25代理数量约6690–6710个，几乎不变。优先解决目标输入像素与弱表征问题，比依据少数错类/定位代理直接改损失更有证据。分类混淆、定位不足与无输出类别均受归类优先级影响，不能据其比例否定分类或定位问题。",
        "同原图尺寸区间内，密集组Recall仍比稀疏组低约4–8pp，两组都受益于提高分辨率。这个差距值得进一步控制类别、区间内尺寸及场景后分析，但目前不能支持新增几何监督。bicycle与awning-tricycle即使960下主AP50:95也只有5.81%和10.18%，需审查训练样本数量、类间相似性、标注质量及类别采样；不先凭低AP改分类头。",
        "仍未完成 F 系列正式训练，所以无法判断空间门控是否优于全局权重、语义细节残差是否优于普通残差。应先完成最小对照，按数据决定是否继续研究融合。F2 若优于 F2_GLOBAL 与 M0，再分析目标/背景及尺寸间门控差异；F3 若优于 F3_PLAIN，再检查 DW 参数、门控梯度和背景增强；若均无明显收益，停止第23层继续堆叠，优先另行申请 960 匹配训练与小目标裁剪/采样对照。",
        "这些都是实验路线，当前没有可宣称成立的原创模块。+0.5 pp 仅用于第一轮预算筛选，后续需42/43/44与同配置M0配对复验，报告均值、样本标准差及逐种子差值。",
        "", "正式训练顺序、命令、资源与判据见 [FUSION_TRAINING_PLAN.md](FUSION_TRAINING_PLAN.md)。正式队列尚未创建，等待用户明确授权。", ""]
    (ROOT / "code/test/RESOLUTION_STUDY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/fusion/resolution_m0_20261011")
    args = parser.parse_args()
    project = args.project.resolve()
    records = []
    for name in NAMES:
        directory = project / name
        record = dict(directory=directory, provenance=read(directory / "resolution_provenance.json"),
                      evaluation=read(directory / "evaluation.json"), diagnosis=read(directory / "error_diagnosis.json"))
        assert record["provenance"]["status"] == "complete"
        records.append(record)
    features = read(project / "feature_size_640/diagnosis.json")
    audit_data = audit(project, records)
    figures(project, records)
    selection = examples(project, records, read(ROOT / "datasets/nscr_visdrone/val_coco.json"))
    report(project, records, features, audit_data, selection)
    print(json.dumps(dict(report=str(ROOT / "code/test/RESOLUTION_STUDY.md"),
                         visualizations=4, resolutions=3, images_per_resolution=548,
                         formal_fusion_started=audit_data["formal_fusion_started"])))


if __name__ == "__main__":
    main()
