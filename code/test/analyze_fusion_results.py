"""Offline, auditable comparisons and error diagnosis from saved predictions; no GPU or training."""
import argparse
import collections
import contextlib
import io
import json
from pathlib import Path
import numpy as np
from train_test import ROOT
from evaluate_nscr import box_iou_numpy, match_instances


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def comparison(run, baseline):
    a, b = read(run / "experiment.json"), read(baseline / "experiment.json")
    ignored = {"model", "project", "name", "save_dir"}
    aa, bb = a["arguments"]["effective_training_arguments"], b["arguments"]["effective_training_arguments"]
    differences = {k: [bb.get(k), aa.get(k)] for k in set(aa) | set(bb)
                   if k not in ignored and aa.get(k) != bb.get(k)}
    identities = {k: a[k] == b[k] for k in ("checkpoint_sha256", "dataset_files", "dataset_labels_sha256")}
    identities["layers_0_to_22"] = (a["architecture"]["backbone"] == b["architecture"]["backbone"] and
        a["architecture"]["head"][:-2] == b["architecture"]["head"][:-2])
    identities["detection_inputs"] = a["architecture"]["head"][-1] == b["architecture"]["head"][-1]
    transition = run / "batch_transition.json"
    if transition.is_file():
        history = read(transition)
        if history.get("mixed_training_history"):
            differences["batch_history"] = ["single training protocol", history]
    return dict(matched=not differences and all(identities.values()), differences=differences, identities=identities)


def error_diagnosis(annotation, predictions, imgsz=640, conf=.25):
    gt = read(annotation)
    by_gt, by_pred, by_ignore = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for a in gt["annotations"]:
        (by_ignore if a.get("iscrowd") else by_gt)[a["image_id"]].append(a)
    for p in predictions:
        by_pred[p["image_id"]].append(p)
    bins = [(0, 8), (8, 16), (16, 32), (32, float("inf"))]
    sizes = {coordinate: {f"{lo}-{hi:g}": dict(targets=0, matched=0) for lo, hi in bins}
             for coordinate in ("original", "input640")}
    cls = {c["id"]: dict(name=c["name"], targets=0, matched=0, wrong_class_overlap=0) for c in gt["categories"]}
    confusion = collections.Counter()
    counts = dict(targets=0, matched_iou50=0, matched_iou75=0, predictions=0,
                  unmatched_wrong_class_overlap=0, ignored_predictions=0, background_predictions_iou_lt_01=0)
    thresholds = {str(t): dict(targets=0, matched=0) for t in (.05, .1, .25)}
    for meta in gt["images"]:
        anns, ps = by_gt[meta["id"]], by_pred[meta["id"]]
        boxes = np.array([a["bbox"] for a in anns], dtype=float).reshape(-1, 4)
        classes = np.array([a["category_id"] for a in anns])
        detections = np.array([[*p["bbox"], p["score"], p["category_id"]] for p in ps], dtype=float).reshape(-1, 6)
        # Supplemental diagnosis: remove detections >50% covered by any ignore box.
        # Official region-union pixel handling remains a separate protocol.
        regions = np.unique(np.array([a["bbox"] for a in by_ignore[meta["id"]]], dtype=float).reshape(-1, 4), axis=0)
        if len(detections) and len(regions):
            inter = np.maximum(0, np.minimum(detections[:, None, :2] + detections[:, None, 2:4],
                regions[None, :, :2] + regions[None, :, 2:]) - np.maximum(detections[:, None, :2], regions[None, :, :2])).prod(2)
            keep = (inter / np.maximum(detections[:, 2:4].prod(1)[:, None], 1e-9)).max(1) < .5
            counts["ignored_predictions"] += int((~keep).sum())
            detections = detections[keep]
        size = np.sqrt(boxes[:, 2:].prod(1))
        boxes[:, 2:] += boxes[:, :2]
        detections[:, 2:4] += detections[:, :2]
        hits, _ = match_instances(boxes, classes, detections, conf=conf)
        hits75, _ = match_instances(boxes, classes, detections, conf=conf, iou_threshold=.75)
        for threshold in thresholds:
            result, _ = match_instances(boxes, classes, detections, conf=float(threshold))
            thresholds[threshold]["targets"] += len(boxes)
            thresholds[threshold]["matched"] += int((result > 0).sum())
        gain = imgsz / max(meta["width"], meta["height"])
        for coordinate, s in (("original", size), ("input640", size * gain)):
            for lo, hi in bins:
                mask = (s >= lo) & (s < hi)
                row = sizes[coordinate][f"{lo}-{hi:g}"]
                row["targets"] += int(mask.sum())
                row["matched"] += int((mask & (hits > 0)).sum())
        p = detections[detections[:, 4] >= conf]
        iou = box_iou_numpy(boxes, p[:, :4]) if len(boxes) and len(p) else np.zeros((len(boxes), len(p)))
        wrong = np.zeros(len(boxes), dtype=bool)
        if len(p):
            for k in np.flatnonzero(hits == 0):
                options = np.where(p[:, 5] != classes[k], iou[k], -1)
                index = int(options.argmax())
                if options[index] >= .5:
                    wrong[k] = True
                    confusion[(int(classes[k]), int(p[index, 5]))] += 1
        for cid in cls:
            mask = classes == cid
            cls[cid]["targets"] += int(mask.sum())
            cls[cid]["matched"] += int((mask & (hits > 0)).sum())
            cls[cid]["wrong_class_overlap"] += int((mask & wrong).sum())
        counts["targets"] += len(boxes)
        counts["matched_iou50"] += int((hits > 0).sum())
        counts["matched_iou75"] += int((hits75 > 0).sum())
        counts["unmatched_wrong_class_overlap"] += int(wrong.sum())
        counts["predictions"] += len(p)
        counts["background_predictions_iou_lt_01"] += int((iou.max(0) < .1).sum()) if len(boxes) else len(p)
    for container in (*sizes.values(), cls, thresholds):
        for row in container.values():
            row["recall"] = row["matched"] / row["targets"] if row["targets"] else None
    return dict(protocol="Supplemental converted positives; original-image boxes, class-aware one-to-one IoU=0.5, conf=0.25. "
                "Ignore rectangles filtered at individual IoA>=0.5, not the official region-union raster protocol. "
                "Wrong-class overlaps suggest confusion; not a causal error attribution. Size=sqrt(w*h), half-open bins.",
                sizes=sizes, classes=cls, counts=counts, confidence_thresholds=thresholds,
                confusion=[dict(gt=g, prediction=p, count=n) for (g, p), n in confusion.most_common(15)])


def per_class_ap(annotation, predictions):
    import sys
    if (ROOT / ".research-deps").is_dir():
        sys.path.insert(0, str(ROOT / ".research-deps"))
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(annotation))
        dt = gt.loadRes(predictions)
        evaluation = COCOeval(gt, dt, "bbox")
        evaluation.params.maxDets = [1, 10, 500]
        evaluation.evaluate()
        evaluation.accumulate()
    precision = evaluation.eval["precision"]
    def mean(values):
        valid = values[values >= 0]
        return float(valid.mean()) if len(valid) else None
    return [dict(category=gt.cats[cid]["name"], AP50=mean(precision[0, :, i, 0, -1]),
                 AP50_95=mean(precision[:, :, i, 0, -1]), APsmall=mean(precision[:, :, i, 1, -1]))
            for i, cid in enumerate(evaluation.params.catIds)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/candidates/formal_100e_b8_s42_v2")
    parser.add_argument("--baseline-project", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-diagnosis", action="store_true")
    args = parser.parse_args()
    base_project = args.baseline_project or args.project
    baseline = base_project / "M0_seed42"
    if not baseline.is_dir():
        baseline = base_project / "F0_seed42"
    base = read(baseline / "evaluation/evaluation.json")
    rows = []
    for run in sorted(args.project.glob("*_seed*")):
        if not (run / "summary.json").exists() or not (run / "evaluation/evaluation.json").exists():
            continue
        summary, e = read(run / "summary.json"), read(run / "evaluation/evaluation.json")
        if summary.get("epochs") != summary.get("requested_epochs"):
            continue
        m = e["ultralytics"]
        seed = summary["seed"]
        same_seed_baseline = base_project / f"M0_seed{seed}"
        if not same_seed_baseline.is_dir():
            same_seed_baseline = args.project / f"F0_seed{seed}"
        if not (same_seed_baseline / "evaluation/evaluation.json").is_file():
            raise ValueError(f"Missing completed same-seed baseline for {run.name}")
        same_seed_metrics = read(same_seed_baseline / "evaluation/evaluation.json")
        row = dict(run=run.name, epochs=summary["epochs"],
            AP50=m["metrics/mAP50(B)"], AP95=m["metrics/mAP50-95(B)"],
            delta_AP95_pp=100*(m["metrics/mAP50-95(B)"]-same_seed_metrics["ultralytics"]["metrics/mAP50-95(B)"]),
            APsmall=e["coco_supplemental"]["AP_small"], ARsmall=e["coco_supplemental"]["AR_small"],
            dense_recall=e["groups"]["dense"]["recall"], sparse_recall=e["groups"]["sparse"]["recall"],
            params=e["params"], GFLOPs=e["GFLOPs"], forward_ms=e["forward_benchmark"]["mean_ms"],
            baseline=str(same_seed_baseline), fair_comparison=comparison(run, same_seed_baseline))
        reference = run / "evaluation/visdrone_reference_python.json"
        if reference.is_file():
            row["visdrone_reference_python"] = read(reference)
        rows.append(row)
    report = dict(baseline=str(baseline), results=rows,
        evaluation_note="Ultralytics AP and supplemental COCO small AP have different protocols. Original official toolkit has not run. "
                        "Historical timings were measured at different times, not paired interleaved benchmarks. One seed cannot prove repeatability.")
    grouped = collections.defaultdict(list)
    for row in rows:
        if row["fair_comparison"]["matched"] and row["epochs"] == 100:
            grouped[row["run"].rsplit("_seed", 1)[0]].append(row)
    report["repeats"] = {name: dict(seeds=len(values), AP95_mean=float(np.mean([v["AP95"] for v in values])),
        AP95_sample_std=float(np.std([v["AP95"] for v in values], ddof=1)) if len(values) > 1 else None,
        paired_delta_mean_pp=float(np.mean([v["delta_AP95_pp"] for v in values]))) for name, values in grouped.items()}
    if not args.skip_diagnosis:
        data = read(baseline / "experiment.json")["arguments"]["data"]
        import yaml
        cfg = yaml.safe_load(Path(data).read_text(encoding="utf-8"))
        annotation = Path(cfg["path"]) / "val_coco.json"
        predictions = read(baseline / "evaluation/coco_predictions.json")
        report["M0_errors"] = error_diagnosis(annotation, predictions)
        report["M0_per_class_COCO"] = per_class_ap(annotation, predictions)
    output = args.output or args.project / "review/fusion_audit"
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 已完成结果与融合研究审计", "", report["evaluation_note"], "",
        "|实验|mAP50|mAP50:95|Δ pp|APsmall|ARsmall|Params|GFLOPs|前向 ms|配置一致|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"|{r['run']}|{100*r['AP50']:.2f}|{100*r['AP95']:.2f}|{r['delta_AP95_pp']:+.3f}|"
            f"{100*r['APsmall']:.2f}|{100*r['ARsmall']:.2f}|{r['params']}|{r['GFLOPs']:.3f}|{r['forward_ms']:.2f}|"
            f"{r['fair_comparison']['matched']}|")
    reference_rows = [r for r in rows if "visdrone_reference_python" in r]
    if reference_rows:
        lines += ["", "## 原始标注的参考规则复现（尚未与MATLAB对齐，不能称官方实跑）", "",
            "保留公开工具重复类别加权约定；另列去重类别macro AP。不会覆盖上面的历史Ultralytics指标。", "",
            "|实验|参考AP50|参考AP50:95|macro AP50:95|", "|---|---:|---:|---:|"]
        for row in reference_rows:
            ref = row["visdrone_reference_python"]
            lines.append(f"|{row['run']}|{100*ref['AP50']:.3f}|{100*ref['AP50_95']:.3f}|{100*ref['macro_AP50_95']:.3f}|")
    if "M0_errors" in report:
        lines += ["", "## M0 尺寸与漏检诊断", "", report["M0_errors"]["protocol"], "",
                  "|尺寸口径|sqrt(wh) 像素|目标数|Recall@conf0.25,IoU0.5|", "|---|---|---:|---:|"]
        for coordinate, bins in report["M0_errors"]["sizes"].items():
            for name, r in bins.items():
                lines.append(f"|{coordinate}|{name}|{r['targets']}|{100*r['recall']:.2f}%|" if r['recall'] is not None
                             else f"|{coordinate}|{name}|0|NA|")
        lines += ["", "## 每类 AP（补充 COCO 协议，不能与 Ultralytics mAP 直接混算）", "",
                  "|类别|AP50|AP50:95|APsmall|", "|---|---:|---:|---:|"]
        for r in report["M0_per_class_COCO"]:
            values = [f"{100*r[k]:.2f}" if r[k] is not None else "NA" for k in ("AP50", "AP50_95", "APsmall")]
            lines.append("|" + "|".join([r["category"], *values]) + "|")
    lines += ["", "完整配置差异、类别混淆、阈值诊断及密度分组见 analysis.json。",
        "", "M1–M3 未提供明确稳定收益，不继续调参或组合。F1–F3 是成熟机制诊断，完整100轮及对照未完成前不选优、不声明原创。"]
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "completed": len(rows), "matched": [r['fair_comparison']['matched'] for r in rows]}))


if __name__ == "__main__":
    main()
