"""Python reproduction of the public VisDrone DET MATLAB scoring conventions.

Reference: https://github.com/VisDrone/VisDrone2018-DET-toolkit/tree/master/utils
Functions inspected: dropObjectsInIgr, createIntImg, saveAnnoRes, evalRes,
compOas, calcAccuracy, VOCap. This port has synthetic checks, but has NOT
been cross-validated against a MATLAB execution. Never label it toolkit-run.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def ignore_integral(gt, height, width):
    regions = gt[gt[:, 5] == 0, :4]
    if not len(regions):
        return None
    mask = np.zeros((height, width), dtype=np.int32)
    for x, y, w, h in np.maximum(1, regions).astype(int):
        # MATLAB inclusive, one-based slices. Raw annotations are integral.
        mask[y-1:min(height, y+h), x-1:min(width, x+w)] = 1
    return mask.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)


def drop_in_ignore(rows, integral):
    if integral is None or not len(rows):
        return rows
    height, width = integral.shape
    # For nonnegative coordinates, MATLAB round is floor(x+0.5), not bankers rounding.
    p = np.maximum(1, np.floor(rows[:, :4] + .5)).astype(int)
    x, y = np.clip(p[:, 0], 1, width), np.clip(p[:, 1], 1, height)
    xe, ye = np.minimum(width, x + p[:, 2]), np.minimum(height, y + p[:, 3])
    count = integral[y-1, x-1] + integral[ye-1, xe-1] - integral[y-1, xe-1] - integral[ye-1, x-1]
    return rows[count / (p[:, 2] * p[:, 3]) < .5]


def overlap(dt, gt):
    intersection = np.maximum(0, np.minimum(dt[:, None, :2]+dt[:, None, 2:4],
        gt[None, :, :2]+gt[None, :, 2:4]) - np.maximum(dt[:, None, :2], gt[None, :, :2])).prod(2)
    da, ga = dt[:, 2:4].prod(1), gt[:, 2:4].prod(1)
    union = da[:, None] + ga[None] - intersection
    union[:, gt[:, 4] == 1] = da[:, None]
    return intersection / np.maximum(union, 1e-12)


def match(gt, dt, threshold):
    """Return confidence-sorted detections and -1/0/1 flags; positives preferred."""
    gt = gt[np.argsort(gt[:, 4], kind="stable")]
    order = np.argsort(-dt[:, 4], kind="stable")
    dt = dt[order]
    matched = np.zeros(len(gt), dtype=bool)
    flags = np.zeros(len(dt), dtype=np.int8)
    oa = overlap(dt, gt)
    for i in range(len(dt)):
        valid = (gt[:, 4] == 0) & ~matched & (oa[i] >= threshold)
        if valid.any():
            # evalRes selects the last GT when overlaps tie exactly.
            options = np.flatnonzero(valid)
            index = options[::-1][np.argmax(oa[i, options[::-1]])]
            matched[index] = True
            flags[i] = 1
        elif ((gt[:, 4] == 1) & (oa[i] >= threshold)).any():
            flags[i] = -1
    return order, flags


def voc_ap(recall, precision):
    rec = np.r_[0., recall, 1.]
    pre = np.r_[0., precision, 0.]
    pre = np.maximum.accumulate(pre[::-1])[::-1]
    changes = np.flatnonzero(rec[1:] != rec[:-1]) + 1
    return float(((rec[changes] - rec[changes-1]) * pre[changes]).sum())


def evaluate(annotations, prediction_dir, metadata, output):
    pairs = []
    presence = np.zeros(10, dtype=int)
    for meta in metadata:
        stem = Path(meta["file_name"]).stem
        gt = np.loadtxt(Path(annotations) / (stem + ".txt"), delimiter=",", ndmin=2)
        pred_path = Path(prediction_dir) / (stem + ".txt")
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        dt = np.loadtxt(pred_path, delimiter=",", ndmin=2) if pred_path.stat().st_size else np.empty((0, 8))
        integral = ignore_integral(gt, meta["height"], meta["width"])
        gt = drop_in_ignore(gt[gt[:, 5] != 0], integral)
        dt = drop_in_ignore(dt, integral)
        if len(gt) and not np.isin(gt[:, 4], [0, 1]).all():
            raise ValueError("Original VisDrone GT scores must be 0/1")
        gt[:, 4] = 1-gt[:, 4]
        if len(dt) > 1 and np.any(np.diff(dt[:, 4]) > 0):
            raise ValueError("Predictions must be exported in descending confidence order, like toolkit inputs")
        pairs.append((gt, dt[:500]))
        for cid in range(1, 11):
            presence[cid-1] += int((gt[:, 5] == cid).any())
    ap = np.zeros((10, 10))
    ar = np.zeros((10, 10, 4))
    for cid in range(1, 11):
        print(f"VisDrone reference conventions: class {cid}/10", flush=True)
        for t, threshold in enumerate(np.linspace(.5, .95, 10)):
            score_flags = []
            denominator = 0
            for gt, dt in pairs:
                g = gt[gt[:, 5] == cid, :5]
                ranks = np.flatnonzero(dt[:, 5] == cid)
                d = dt[ranks, :5]
                order, flags = match(g, d, threshold)
                denominator += len(g)  # faithfully retains toolkit's ignored-GT denominator
                if len(d):
                    score_flags.append(np.c_[d[order, 4], flags, ranks[order]])
            all_dt = np.concatenate(score_flags) if score_flags else np.empty((0, 3))
            for k, limit in enumerate((1, 10, 100, 500)):
                ranked = all_dt[all_dt[:, 2] < limit]
                ranked = ranked[np.argsort(-ranked[:, 0], kind="stable")]
                tp = np.cumsum(ranked[:, 1] == 1)
                recall = tp / max(1, denominator)
                if len(recall):
                    ar[cid-1, t, k] = recall[-1]
                if limit == 500:
                    fp = np.cumsum(ranked[:, 1] == 0)
                    ap[cid-1, t] = voc_ap(recall, tp / np.maximum(1, fp+tp))
    valid = presence > 0
    # Public calcAccuracy appends a class once per image containing it and uses
    # AP(evalClass,:), rather than unique(evalClass). Report both to expose this convention.
    weighted = np.average(ap[valid], axis=0, weights=presence[valid])
    report = dict(protocol="Python reproduction of public MATLAB toolkit conventions; NOT MATLAB cross-validated",
        source="https://github.com/VisDrone/VisDrone2018-DET-toolkit",
        images=len(pairs), AP50_95=float(weighted.mean()), AP50=float(weighted[0]), AP75=float(weighted[5]),
        macro_AP50_95=float(ap[valid].mean()), macro_AP50=float(ap[valid, 0].mean()),
        category_image_presence=presence.tolist(), per_category_AP50_95=ap.mean(1).tolist(),
        per_category_AP50=ap[:, 0].tolist(),
        recall={f"AR{n}": float(np.average(ar[valid, :, k], axis=0, weights=presence[valid]).mean())
                for k, n in enumerate((1, 10, 100, 500))},
        conventions="Category-0 union raster filtering, MATLAB rounding and integral indexing; "
        "score=0 GT ignored in matching; per-image global top-k before category filtering; continuous VOC precision envelope. "
        "Category11 is outside evaluated classes, not a category0 region. Toolkit repeated-class aggregation and "
        "GT denominator reproduced literally; macro aggregate also given. APsmall is not defined by this toolkit.")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "datasets/nscr_visdrone")
    parser.add_argument("--experiments", nargs="+", default=["M0", "M1", "M2", "M3"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = json.loads((args.data_root / "preparation.json").read_text(encoding="utf-8"))
    meta = json.loads((args.data_root / "val_coco.json").read_text(encoding="utf-8"))["images"]
    manifest = {Path(p).name for p in (args.data_root / "val.txt").read_text().splitlines() if p}
    meta = [m for m in meta if m["file_name"] in manifest]
    for experiment in args.experiments:
        directory = args.project / f"{experiment}_seed{args.seed}/evaluation"
        result = evaluate(cfg["splits"]["val"]["original_annotations"], directory / "visdrone_official_predictions",
                          meta, directory / "visdrone_reference_python.json")
        print(experiment, result["AP50"], result["AP50_95"])


if __name__ == "__main__":
    main()
