"""RGB inference only: feature/background statistics and M0 branch removal sensitivity."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from train_test import ROOT, YOLO
import torch
import yaml
from evaluate_nscr import match_instances


def grid_mask(boxes, shape, gain, pad, stride):
    h, w = shape
    yy, xx = np.meshgrid((np.arange(h) + .5) * stride, (np.arange(w) + .5) * stride, indexing="ij")
    mask = np.zeros((h, w), dtype=bool)
    for box in boxes:
        x, y, bw, bh = np.asarray(box) * gain
        x += pad[0]; y += pad[1]
        mask |= (xx >= x) & (xx < x + bw) & (yy >= y) & (yy < y + bh)
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/candidates/formal_100e_b8_s42_v2/M0_seed42/weights/best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/nscr_visdrone/dataset.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/fusion/diagnosis_m0")
    parser.add_argument("--limit", type=int, default=64, help="0 means all val images")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    payload = json.loads((root / "val_coco.json").read_text(encoding="utf-8"))
    by_name = {m["file_name"]: m for m in payload["images"]}
    gt, ignore = {}, {}
    for meta in payload["images"]:
        gt[meta["id"]], ignore[meta["id"]] = [], []
    for a in payload["annotations"]:
        (ignore if a.get("iscrowd") else gt)[a["image_id"]].append(a)
    images = [Path(p) for p in (root / cfg["val"]).read_text().splitlines() if p]
    images.sort(key=lambda p: hashlib.sha256(f"42:{p.name}".encode()).hexdigest())
    if args.limit:
        images = images[:args.limit]
    model = YOLO(str(args.checkpoint))
    # Initialize and fuse once before attaching hooks; avoids profiling/warmup contamination.
    model.predict(str(images[0]), imgsz=640, device=args.device, verbose=False)
    module = model.model.model
    cache, handles = {}, []
    for idx in (2, 16, 23):
        def capture(layer, inputs, output, idx=idx):
            cache[idx] = output.detach().float().square().mean(1)[0].sqrt().cpu().numpy()
        handles.append(module[idx].register_forward_hook(capture))
    stats = {str(i): dict(small_energy_sum=0., small_cells=0, background_energy_sum=0., background_cells=0)
             for i in (2, 16, 23)}
    scenarios = {name: dict(targets=0, small_targets=0, matched=0, small_matched=0,
        size_bins={label: dict(targets=0, matched=0) for label in ("<8", "8-16", "16-32", ">=32")})
                 for name in ("full", "disable_P2_head", "zero_P2_fusion_input", "zero_P3_fusion_input")}
    feature_shapes = {}
    try:
        for image in images:
            meta = by_name[image.name]
            annotations = gt[meta["id"]]
            xywh = np.array([a["bbox"] for a in annotations], dtype=float).reshape(-1, 4)
            classes = np.array([a["category_id"]-1 for a in annotations])
            small = xywh[:, 2:].prod(1) <= 32**2
            boxes = xywh.copy(); boxes[:, 2:] += boxes[:, :2]
            for name, row in scenarios.items():
                temporary = None
                if name == "disable_P2_head":
                    def disable(layer, inputs, output):
                        decoded, raw = output
                        count = raw[0].shape[-2] * raw[0].shape[-1]
                        decoded[:, 4:, :count] = 0.
                        return decoded, raw
                    temporary = module[24].register_forward_hook(disable)
                elif name.startswith("zero_"):
                    branch = 0 if name == "zero_P2_fusion_input" else 1
                    def zero(layer, inputs, branch=branch):
                        features = list(inputs[0])
                        features[branch] = torch.zeros_like(features[branch])
                        return (features,)
                    temporary = module[23].register_forward_pre_hook(zero)
                try:
                    result = model.predict(str(image), imgsz=640, device=args.device, conf=.001,
                                           iou=.7, max_det=500, verbose=False)[0]
                finally:
                    if temporary is not None:
                        temporary.remove()
                pred = result.boxes.data.detach().float().cpu().numpy()
                hits, _ = match_instances(boxes, classes, pred, conf=.25)
                row["targets"] += len(boxes)
                row["small_targets"] += int(small.sum())
                row["matched"] += int((hits > 0).sum())
                row["small_matched"] += int(((hits > 0) & small).sum())
                size = np.sqrt(xywh[:, 2:].prod(1))
                for lo, hi, label in ((0, 8, "<8"), (8, 16, "8-16"), (16, 32, "16-32"), (32, np.inf, ">=32")):
                    mask = (size >= lo) & (size < hi)
                    row["size_bins"][label]["targets"] += int(mask.sum())
                    row["size_bins"][label]["matched"] += int((mask & (hits > 0)).sum())
                if name != "full":
                    continue
                input_shape = np.array(cache[2].shape) * 4
                gain = min(input_shape[1] / meta["width"], input_shape[0] / meta["height"])
                pad = (round((input_shape[1] - round(meta["width"] * gain))/2-.1),
                       round((input_shape[0] - round(meta["height"] * gain))/2-.1))
                ignored = [a["bbox"] for a in ignore[meta["id"]]]
                for idx, stride in ((2, 4), (16, 8), (23, 4)):
                    energy = cache[idx]
                    feature_shapes[str(idx)] = list(energy.shape)
                    foreground = grid_mask(xywh[small], energy.shape, gain, pad, stride)
                    occupied = grid_mask([*xywh.tolist(), *ignored], energy.shape, gain, pad, stride)
                    h, w = energy.shape
                    yy, xx = np.meshgrid((np.arange(h)+.5)*stride, (np.arange(w)+.5)*stride, indexing="ij")
                    valid = (xx >= pad[0]) & (xx < pad[0]+round(meta["width"]*gain)) & (
                        yy >= pad[1]) & (yy < pad[1]+round(meta["height"]*gain))
                    background = valid & ~occupied
                    stat = stats[str(idx)]
                    stat["small_energy_sum"] += float(energy[foreground].sum())
                    stat["small_cells"] += int(foreground.sum())
                    stat["background_energy_sum"] += float(energy[background].sum())
                    stat["background_cells"] += int(background.sum())
    finally:
        for handle in handles:
            handle.remove()
    for row in scenarios.values():
        row["recall"] = row["matched"] / row["targets"] if row["targets"] else None
        row["small_recall"] = row["small_matched"] / row["small_targets"] if row["small_targets"] else None
        for subgroup in row["size_bins"].values():
            subgroup["recall"] = subgroup["matched"] / subgroup["targets"] if subgroup["targets"] else None
    for stat in stats.values():
        stat["small_rms_mean"] = stat["small_energy_sum"] / max(1, stat["small_cells"])
        stat["background_rms_mean"] = stat["background_energy_sum"] / max(1, stat["background_cells"])
        stat["small_to_background_ratio"] = stat["small_rms_mean"] / max(1e-9, stat["background_rms_mean"])
    report = dict(checkpoint=str(args.checkpoint.resolve()), images=len(images), image_names=[p.name for p in images],
        scenarios=scenarios, feature_energy=stats, last_image_spatial_shapes=feature_shapes,
        protocol="seed42 hash-selected val images; RGB-only frozen-model inference, conf=.25 IoU=.5. "
          "Small means original-image area<=32^2. RMS across channels then pixel-weighted pooling; feature-cell centers "
          "inside GT/ignore rectangles; padding excluded from background. Inference head removal/zero inputs cause "
          "distribution shifts and NMS changes, not retrained ablations or a proof of fusion bottleneck. "
          "Energy alone is not semantic quality. Positive GT is the historical converted set, not official scoring.")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "diagnosis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
