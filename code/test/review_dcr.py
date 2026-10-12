"""Summarize completed matched runs without claiming a positive result in advance."""
import argparse
import csv
import json
import statistics
from pathlib import Path

METRICS = ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)")


def stats(values):
    values = list(values)
    return dict(n=len(values), mean=statistics.mean(values),
                sample_std=statistics.stdev(values) if len(values) > 1 else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    rows, groups = [], {}
    for run in sorted(args.project.glob("*_seed*")):
        if not (run / "summary.json").is_file():
            continue
        row = json.loads((run / "summary.json").read_text())
        manifest = json.loads((run / "experiment.json").read_text())
        row["configuration"] = {k: manifest["arguments"][k] for k in
            ("epochs", "imgsz", "batch", "workers", "cache", "spacing_weight", "tau", "temperature", "small_size", "cap")}
        row["dataset_files"] = manifest["dataset_files"]
        with (run / "results.csv").open(encoding="utf-8-sig") as stream:
            curve = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(stream)]
        row["last_epoch_metrics"] = {k: float(curve[-1][k]) for k in METRICS}
        row["best_csv_epoch"] = int(max(curve, key=lambda r: float(r["metrics/mAP50-95(B)"]))["epoch"])
        evaluation = run / "evaluation/evaluation.json"
        if evaluation.is_file():
            evaluated = json.loads(evaluation.read_text())
            row["evaluation"] = evaluated
        rows.append(row)
        groups.setdefault(row["experiment"], {})[row["seed"]] = row
    if not rows:
        raise RuntimeError("No completed runs")
    for row in rows[1:]:
        if row["configuration"] != rows[0]["configuration"] or row["dataset_files"] != rows[0]["dataset_files"]:
            raise ValueError("Unmatched training/data settings; do not aggregate these runs")
    aggregate = {name: {metric: stats(row[metric] for row in seeds.values()) for metric in METRICS}
                 for name, seeds in groups.items()}
    pairs = {}
    for control in ("E1", "E4", "D1", "D2", "D4"):
        common = sorted(set(groups.get("D3", {})) & set(groups.get(control, {})))
        if common:
            differences = [groups["D3"][s][METRICS[1]] - groups[control][s][METRICS[1]] for s in common]
            pairs[control] = dict(seeds=common, differences=differences, **stats(differences))
    report = dict(aggregate=aggregate, D3_minus_control_mAP50_95=pairs, runs=rows)
    output = args.project / "review"
    output.mkdir(exist_ok=True)
    (output / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# DCR 实际对照结果", "", "数值为百分数；±为种子间样本标准差。主指标：mAP50:95。", "",
             "|实验|种子数|mAP50|mAP50:95|Precision|Recall|Params|GFLOPs|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, metric in aggregate.items():
        seed_rows = list(groups[name].values())
        values = [f"{metric[k]['mean']*100:.3f} ± {metric[k]['sample_std']*100:.3f}" if metric[k]['sample_std'] is not None
                  else f"{metric[k]['mean']*100:.3f}" for k in METRICS]
        lines.append(f"|{name}|{len(seed_rows)}|" + "|".join(values) + f"|{seed_rows[0]['params']}|{seed_rows[0]['GFLOPs']:.3f}|")
    lines += ["", "## 配对差值", "", "D3 减对应对照的 mAP50:95（百分点），同种子比较：", ""]
    for name, pair in pairs.items():
        lines.append(f"- D3 − {name}: 平均 {pair['mean']*100:+.3f}；各种子 " +
                     ", ".join(f"{s}: {v*100:+.3f}" for s, v in zip(pair['seeds'], pair['differences'])))
    positive = all(control in pairs and pairs[control]["mean"] > 0 for control in ("D1", "D2", "D4"))
    lines += ["", "## 当前结论", "", ("方向监督在此初筛的平均主指标上优于三个内部对照；仅作为探索性正向证据。" if positive else
                "此轮初筛未同时证明方向监督优于无监督、无方向监督及打乱标签三个对照，不能宣称核心机制有效。"),
              "", "功能测试证明程序可训练、可推理，不等同于检测效果验证。短训/子集结果不能作为完整 VisDrone 论文结果，"
              "不能与旧的全量 E0 100 epoch 结果直接比较。3 个种子不足以稳定证明统计显著性。"
              "正式结论需要全量 6471/548 图、同训练设置、100 epoch 对照，并在 AI-TOD-v2 复现。",
              "", "## 训练设置", "", "```json", json.dumps(rows[0]["configuration"], indent=2), "```"]
    if any("evaluation" in row for row in rows):
        lines += ["", "## 补充小目标评价与速度", "", "COCO 补充协议；原图面积 ≤32²，maxDet=500；不是官方 VisDrone 工具输出。",
                  "", "|实验|AP-small (%)|AR-small (%)|forward ms|FPS|", "|---|---:|---:|---:|---:|"]
        for name, seeds in groups.items():
            evaluated = [r["evaluation"] for r in seeds.values() if "evaluation" in r]
            if not evaluated:
                continue
            values = [statistics.mean(e["coco_supplemental"][k] for e in evaluated)*100 for k in ("AP_small", "AR_small")]
            timing = statistics.mean(e["forward_benchmark"]["mean_ms"] for e in evaluated)
            fps = statistics.mean(e["forward_benchmark"]["FPS"] for e in evaluated)
            lines.append(f"|{name}|{values[0]:.3f}|{values[1]:.3f}|{timing:.3f}|{fps:.1f}|")
        lines += ["", "速度为 RTX3090 上 batch=1、FP32、fused model forward，排除解码、预处理及 NMS，不能解释为应用端到端 FPS。"]
    (output / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    names = list(aggregate)
    ax.bar(names, [aggregate[n][METRICS[1]]["mean"]*100 for n in names],
           yerr=[(aggregate[n][METRICS[1]]["sample_std"] or 0)*100 for n in names], capsize=5)
    ax.set_ylabel("mAP50:95 (%)"); ax.set_title("Controlled VisDrone pilot (mean ± sample SD)")
    fig.savefig(output / "ablation.png", dpi=180)
    plt.close(fig)
    print(output / "review.md")


if __name__ == "__main__":
    main()
