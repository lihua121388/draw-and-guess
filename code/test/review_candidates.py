"""Report actual completed candidate results and matching-baseline differences."""
import argparse
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs/nscr/.config"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/nscr/.matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
from candidate_experiments import EXPERIMENTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.project.glob("M*_seed*/summary.json")):
        row = json.loads(path.read_text())
        if row.get("status") != "training_complete" or row.get("epochs") != row.get("requested_epochs"):
            continue
        evaluation = path.parent / "evaluation/evaluation.json"
        row["evaluation"] = json.loads(evaluation.read_text()) if evaluation.is_file() else None
        row["manifest"] = json.loads((path.parent / "experiment.json").read_text())
        rows.append(row)
    if not rows:
        raise RuntimeError("No completed runs")
    baseline = {r["seed"]: r for r in rows if r["experiment"] == "M0"}
    metric = "metrics/mAP50-95(B)"
    for row in rows:
        reference = baseline.get(row["seed"])
        if reference:
            a, b = row["manifest"], reference["manifest"]
            keys = ("epochs", "batch", "imgsz", "workers", "device", "cache", "no_amp")
            same_data = a["dataset_files"] == b["dataset_files"] and a["dataset_labels_sha256"] == b["dataset_labels_sha256"]
            same_weights = a["checkpoint_sha256"] == b["checkpoint_sha256"]
            if not same_data or not same_weights or any(a["arguments"][k] != b["arguments"][k] for k in keys):
                raise ValueError("Unmatched baseline settings")
            row["delta_mAP50_95_vs_M0"] = row[metric] - reference[metric]
    directory = args.project / "review"
    directory.mkdir(exist_ok=True)
    (directory / "review.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 五个候选模块：实际结果", "", "只有完成的训练列入表格；空白项表示评价尚未完成。AP、AR 和 mAP 以百分数显示。",
        "", "|实验|模块|seed|epoch|mAP50|mAP50:95|对 M0 差值(pp)|AP-small|AR-small|Params|GFLOPs|GPU forward ms|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    if any(row["epochs"] == 1 for row in rows):
        lines[2] += " **本表含一轮流程测试，仅用于检查运行，不用于评判模块精度收益。**"
    for row in sorted(rows, key=lambda r: (r["experiment"], r["seed"])):
        ev = row["evaluation"] or {}
        small = ev.get("coco_supplemental", {})
        ap = f"{small['AP_small']*100:.3f}" if small.get("AP_small") is not None else ""
        ar = f"{small['AR_small']*100:.3f}" if small.get("AR_small") is not None else ""
        latency = ev.get("forward_benchmark", {}).get("mean_ms")
        delta = row.get("delta_mAP50_95_vs_M0")
        lines.append("|" + "|".join([row["experiment"], EXPERIMENTS[row["experiment"]]["name"], str(row["seed"]),
            str(row["epochs"]), f"{row['metrics/mAP50(B)']*100:.3f}", f"{row[metric]*100:.3f}",
            f"{delta*100:+.3f}" if delta is not None else "", ap, ar, str(row["params"]),
            f"{row['GFLOPs']:.3f}", f"{latency:.3f}" if latency is not None else ""]) + "|")
    lines += ["", "主指标 mAP50:95；M0 是相同训练设置的 P2 基准。正差值是该次训练的观察，不是原创性或统计显著性证明。",
        "只有一个 seed 的筛选不证明稳定收益；候选需要后续同参数无辅助监督、打乱/去除信息、多个种子及另一数据集验证。",
        "COCO 小目标指标为补充协议、原图面积 ≤32²、maxDet=500，ignored-region 处理是近似；官方 VisDrone 工具输出另行报告。",
        "GPU timing 为 batch=1 FP32 fused forward，排除 IO、预处理和 NMS；GFLOPs 不完整计入逐元素/访存开销。"]
    (directory / "review.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(directory / "review.md")


if __name__ == "__main__":
    main()
