"""Size/class/density and disjoint error proxies from frozen-weight predictions; no training."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from train_test import ROOT
from evaluate_nscr import box_iou_numpy

SIZE_BINS = ((0, 8, "<8"), (8, 16, "8-16"), (16, 32, "16-32"), (32, np.inf, ">=32"))
ERRORS = ("low_conf_005_025", "very_low_conf_0001_005", "assignment_competition",
          "wrong_class_proxy", "localization_proxy", "no_qualifying_output")


def match_indices(iou, classes, pred, conf=.25, eligible=None, blocked=None, threshold=.5, wrong=False):
    """Stable confidence-ordered, one-to-one matching. Return indices into input predictions."""
    matched = np.full(len(classes), -1, dtype=int)
    eligible = np.ones(len(classes), dtype=bool) if eligible is None else eligible.copy()
    blocked = np.zeros(len(pred), dtype=bool) if blocked is None else blocked
    for p in np.argsort(-pred[:, 4], kind="stable"):
        if pred[p, 4] < conf or blocked[p]:
            continue
        same = classes == pred[p, 5]
        valid = eligible & (matched < 0) & (~same if wrong else same)
        if not valid.any():
            continue
        scores = np.where(valid, iou[:, p], -1)
        g = int(scores.argmax())
        if scores[g] >= threshold:
            matched[g] = p
    return matched


def spacing(boxes):
    if len(boxes) < 2:
        return np.full(len(boxes), np.nan)
    size = np.sqrt((boxes[:, 2:] - boxes[:, :2]).prod(1))
    center = (boxes[:, 2:] + boxes[:, :2]) / 2
    distance = np.linalg.norm(center[:, None] - center[None, :], axis=2)
    q = distance / (.5 * (size[:, None] + size[None, :]) + 1e-6)
    np.fill_diagonal(q, np.inf)
    return q.min(1)


def empty_row():
    return dict(targets=0, matched=0, errors={name: 0 for name in ERRORS})


def update(row, mask, hits, categories):
    row["targets"] += int(mask.sum())
    row["matched"] += int((mask & hits).sum())
    for name in ERRORS:
        row["errors"][name] += int((mask & (categories == name)).sum())


def diagnose(gt, predictions, imgsz):
    anns, preds, regions = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for a in gt["annotations"]:
        (regions if a.get("iscrowd") else anns)[a["image_id"]].append(a)
    for p in predictions:
        preds[p["image_id"]].append(p)
    report = dict(all=empty_row(), original_size={b: empty_row() for _, _, b in SIZE_BINS},
        input_size={b: empty_row() for _, _, b in SIZE_BINS},
        class_size={c["name"]: {b: empty_row() for _, _, b in SIZE_BINS} for c in gt["categories"]},
        density_size={b: {d: empty_row() for d in ("dense", "sparse", "undefined")} for _, _, b in SIZE_BINS},
        thresholds={str(c): dict(targets=0, matched=0) for c in (.001, .05, .1, .25)},
        images=0, ignored_prediction_count=0, confusion=collections.Counter())
    category_names = {c["id"]: c["name"] for c in gt["categories"]}
    for meta in gt["images"]:
        boxes = np.array([a["bbox"] for a in anns[meta["id"]]], dtype=float).reshape(-1, 4)
        classes = np.array([a["category_id"] for a in anns[meta["id"]]])
        pred = np.array([[*p["bbox"], p["score"], p["category_id"]] for p in preds[meta["id"]]], dtype=float).reshape(-1, 6)
        ignore = np.unique(np.array([a["bbox"] for a in regions[meta["id"]]], dtype=float).reshape(-1, 4), axis=0)
        if len(pred) and len(ignore):
            intersection = np.maximum(0, np.minimum(pred[:, None, :2]+pred[:, None, 2:4],
                ignore[None, :, :2]+ignore[None, :, 2:4])-np.maximum(pred[:, None, :2], ignore[None, :, :2])).prod(2)
            keep = (intersection/np.maximum(pred[:, 2:4].prod(1)[:, None], 1e-9)).max(1) < .5
            report["ignored_prediction_count"] += int((~keep).sum())
            pred = pred[keep]
        size = np.sqrt(boxes[:, 2:].prod(1))
        boxes[:, 2:] += boxes[:, :2]
        pred[:, 2:4] += pred[:, :2]
        iou = box_iou_numpy(boxes, pred[:, :4])
        matches = {str(c): match_indices(iou, classes, pred, conf=c) for c in (.001, .05, .1, .25)}
        high = matches["0.25"] >= 0
        categories = np.full(len(boxes), "detected", dtype=object)
        remaining = ~high
        low = remaining & (matches["0.05"] >= 0)
        categories[low] = "low_conf_005_025"
        remaining &= ~low
        very_low = remaining & (matches["0.001"] >= 0)
        categories[very_low] = "very_low_conf_0001_005"
        remaining &= ~very_low
        if len(pred):
            good_same = (classes[:, None] == pred[None, :, 5]) & (pred[None, :, 4] >= .25) & (iou >= .5)
            competition = remaining & good_same.any(1)
        else:
            competition = np.zeros(len(boxes), dtype=bool)
        categories[competition] = "assignment_competition"
        remaining &= ~competition
        blocked = np.zeros(len(pred), dtype=bool)
        blocked[matches["0.001"][matches["0.001"] >= 0]] = True
        wrong = match_indices(iou, classes, pred, eligible=remaining, blocked=blocked, wrong=True)
        wrong_mask = wrong >= 0
        categories[wrong_mask] = "wrong_class_proxy"
        for k in np.flatnonzero(wrong_mask):
            report["confusion"][(int(classes[k]), int(pred[wrong[k], 5]))] += 1
        remaining &= ~wrong_mask
        blocked[wrong[wrong_mask]] = True
        location = match_indices(iou, classes, pred, eligible=remaining, blocked=blocked, threshold=.1)
        location_mask = location >= 0
        categories[location_mask] = "localization_proxy"
        remaining &= ~location_mask
        categories[remaining] = "no_qualifying_output"
        update(report["all"], np.ones(len(boxes), dtype=bool), high, categories)
        q = spacing(boxes)
        gain = imgsz/max(meta["width"], meta["height"])
        for lo, hi, name in SIZE_BINS:
            mask = (size >= lo) & (size < hi)
            update(report["original_size"][name], mask, high, categories)
            update(report["input_size"][name], (size*gain >= lo) & (size*gain < hi), high, categories)
            for cid, category in category_names.items():
                update(report["class_size"][category][name], mask & (classes == cid), high, categories)
            for density, dmask in (("dense", q < 2), ("sparse", q >= 2), ("undefined", ~np.isfinite(q))):
                update(report["density_size"][name][density], mask & dmask, high, categories)
        for conf, match in matches.items():
            report["thresholds"][conf]["targets"] += len(boxes)
            report["thresholds"][conf]["matched"] += int((match >= 0).sum())
        report["images"] += 1
    def finish(row):
        row["recall"] = row["matched"]/row["targets"] if row["targets"] else None
        row["missed"] = row["targets"]-row["matched"]
        if "errors" in row:
            assert sum(row["errors"].values()) == row["missed"]
            row["error_fraction_of_misses"] = {k: v/row["missed"] if row["missed"] else None for k, v in row["errors"].items()}
    finish(report["all"])
    for container in (report["original_size"], report["input_size"], report["thresholds"],
                      *report["class_size"].values(), *report["density_size"].values()):
        for row in container.values():
            finish(row)
    report["confusion"] = [dict(gt=category_names[g], prediction=category_names[p], count=n)
        for (g, p), n in report["confusion"].most_common(20)]
    report["protocol"] = "Historical converted positives, original-image boxes, conf.25 IoU.5 one-to-one class-aware recall. " \
        "Ignore rectangles filtered at individual IoA>=.5 like previous offline diagnosis, not official union-raster scoring. " \
        "sqrt(wh) half-open size bins, spacing q<2 dense, >=2 sparse within each original-size bin. " \
        "Disjoint miss proxies, in priority: recovered at conf.05, recovered at conf.001, same-class assignment competition, " \
        "unused high-conf wrong-class IoU>=.5, unused high-conf same-class IoU>=.1, otherwise no qualifying output. " \
        "Wrong/poor-localization are observable proxies, not proven causes; post-NMS max500 output may suppress/truncate candidates."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/fusion/resolution_m0_20261011")
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets/nscr_visdrone")
    args = parser.parse_args()
    gt = json.loads((args.data_root / "val_coco.json").read_text(encoding="utf-8"))
    for directory in sorted(args.project.glob("R*_*")):
        provenance = json.loads((directory / "resolution_provenance.json").read_text(encoding="utf-8"))
        if provenance.get("status") != "complete":
            continue
        predictions = json.loads((directory / "coco_predictions.json").read_text(encoding="utf-8"))
        report = diagnose(gt, predictions, provenance["imgsz"])
        (directory / "error_diagnosis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"resolution": provenance["imgsz"], "all": report["all"], "original_size": report["original_size"]}))


if __name__ == "__main__":
    main()
