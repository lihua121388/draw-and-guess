"""Supplemental COCO small-object metrics, density groups, official exports and GPU timing."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ultralytics import YOLO
from ultralytics.nn.modules.nscr import NSCRFusion
from ultralytics.utils.nscr_loss import normalized_spacing, sample_gate
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parents[2]


def box_iou_numpy(a, b):
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                              np.maximum(a[:, None, :2], b[None, :, :2])).prod(2)
    area_a = np.maximum(0, a[:, 2:] - a[:, :2]).prod(1)
    area_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(1)
    return intersection / np.maximum(area_a[:, None] + area_b[None] - intersection, 1e-9)


def match_instances(gt_boxes, gt_classes, predictions, iou_threshold=0.5, conf=0.25):
    """Greedy confidence-ordered, class-aware one-to-one matching for grouped diagnostics."""
    matched_iou = np.zeros(len(gt_boxes))
    matched_centers = np.full((len(gt_boxes), 2), np.nan)
    if not len(predictions) or not len(gt_boxes):
        return matched_iou, matched_centers
    predictions = predictions[predictions[:, 4] >= conf]
    if not len(predictions):
        return matched_iou, matched_centers
    ious = box_iou_numpy(gt_boxes, predictions[:, :4])
    for index in np.argsort(-predictions[:, 4], kind="stable"):
        valid = (gt_classes == predictions[index, 5]) & (matched_iou == 0)
        candidates = np.where(valid, ious[:, index], -1)
        best = candidates.argmax()
        if candidates[best] >= iou_threshold:
            matched_iou[best] = candidates[best]
            matched_centers[best] = (predictions[index, :2] + predictions[index, 2:4]) / 2
    return matched_iou, matched_centers


def benchmark(model, imgsz, device, warmup=30, repeats=100):
    """Measure synchronized batch-one forward latency, excluding preprocessing and NMS."""
    target = torch.device("cpu" if str(device) == "cpu" else f"cuda:{device}")
    model = model.to(target).float().eval()
    tensor = torch.zeros(1, 3, imgsz, imgsz, device=target)
    timings = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(tensor)
        for _ in range(repeats):
            if target.type == "cuda":
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                model(tensor)
                stop.record()
                stop.synchronize()
                timings.append(start.elapsed_time(stop))
            else:
                start = time.perf_counter()
                model(tensor)
                timings.append((time.perf_counter() - start) * 1000)
    mean = float(np.mean(timings))
    return {"device": str(target), "precision": "FP32", "batch": 1, "imgsz": imgsz,
            "warmup": warmup, "repeats": repeats, "input": "synthetic zero RGB",
            "scope": "fused model forward only; excludes preprocessing, NMS and IO",
            "mean_ms": mean, "p50_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
            "FPS": 1000 / mean, "gpu": torch.cuda.get_device_name(target) if target.type == "cuda" else None}


def coco_metrics(annotations, predictions, image_ids, max_det):
    """Use upstream pycocotools at original image coordinates; preserve official COCO category IDs."""
    local_dependencies = ROOT / ".research-deps"
    if local_dependencies.is_dir():
        sys.path.insert(0, str(local_dependencies))
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError("Install pycocotools in yolo or .research-deps; see IMPLEMENTATION.md") from error
    gt = COCO(str(annotations))
    if predictions:
        dt = gt.loadRes(predictions)
    else:
        dt = COCO()
        dt.dataset = {"images": gt.dataset["images"], "categories": gt.dataset["categories"], "annotations": []}
        dt.createIndex()
    evaluator = COCOeval(gt, dt, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.params.maxDets = [1, 10, max_det]
    evaluator.evaluate()
    evaluator.accumulate()
    precision, recall = evaluator.eval["precision"], evaluator.eval["recall"]

    def average(values):
        valid = values[values >= 0]
        return float(valid.mean()) if len(valid) else None

    return {"mAP50_95": average(precision[:, :, :, 0, -1]), "mAP50": average(precision[0, :, :, 0, -1]),
            "AP_small": average(precision[:, :, :, 1, -1]), "AR_small": average(recall[:, :, 1, -1]),
            "AP_medium": average(precision[:, :, :, 2, -1]), "AP_large": average(precision[:, :, :, 3, -1]),
            "max_det": max_det, "small_area": "original-image area <= 32^2 pixels",
            "protocol": "supplemental standard COCO; VisDrone ignored regions approximated as classwise crowds",
            "per_class": [{"category_id": int(cid), "name": gt.cats[cid]["name"],
                "AP50": average(precision[0, :, k, 0, -1]),
                "AP50_95": average(precision[:, :, k, 0, -1]),
                "AP_small": average(precision[:, :, k, 1, -1]),
                "AR_small": average(recall[:, k, 1, -1])}
                for k, cid in enumerate(evaluator.params.catIds)]}


def write_official_matlab(output, original_annotations, predictions, images):
    """Create a bridge to the unmodified VisDrone toolkit using only the evaluated image list."""
    output = Path(output).resolve()
    ground_truth = Path(original_annotations).resolve()
    metadata = {"ground_truth": ground_truth.as_posix(),
                "images": (ground_truth.parent / "images").as_posix(),
                "predictions": Path(predictions).resolve().as_posix()}
    metadata_path = output / "visdrone_paths.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    names = output / "visdrone_evaluated_names.txt"
    names.write_text("\n".join(p.stem + ".txt" for p in images) + "\n", encoding="utf-8")
    def quoted(path):
        return str(path).replace("'", "''").replace("\\", "/")
    script = f"""% NSCR bridge; download the official toolkit and set its path below.
% https://github.com/VisDrone/VisDrone2018-DET-toolkit
toolkit = 'SET_PATH_TO_VisDrone2018-DET-toolkit';
assert(isfolder(toolkit), 'Set toolkit path before running this script.');
addpath(fullfile(toolkit, 'utils'));
paths = jsondecode(fileread('{quoted(metadata_path)}'));
names = splitlines(string(fileread('{quoted(names)}')));
names = cellstr(names(strlength(names) > 0));
count = numel(names);
[groundTruth, predictions] = saveAnnoRes(paths.ground_truth, paths.predictions, paths.images, count, names);
[ap, ap50, ap75, ar1, ar10, ar100, ar500] = calcAccuracy(count, groundTruth, predictions);
result = struct('AP50_95_percent', ap, 'AP50_percent', ap50, 'AP75_percent', ap75, ...
    'AR1_percent', ar1, 'AR10_percent', ar10, 'AR100_percent', ar100, 'AR500_percent', ar500, 'images', count);
disp(result);
file = fopen('{quoted(output / 'visdrone_official_metrics.json')}', 'w');
fprintf(file, '%s', jsonencode(result));
fclose(file);
"""
    (output / "run_visdrone_official.m").write_text(script, encoding="utf-8")


def evaluate(checkpoint, data_path, output, args):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(data_path).read_text(encoding="utf-8"))
    directory = Path(cfg["path"])
    manifest = directory / cfg[args.split]
    if manifest.suffix != ".txt":
        raise ValueError("Research evaluation expects a prepared split manifest (.txt)")
    images = [Path(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    annotation_path = directory / f"{args.split}_coco.json"
    gt = json.loads(annotation_path.read_text(encoding="utf-8"))
    lookup = {img["file_name"]: img for img in gt["images"]}
    categories = sorted(gt["categories"], key=lambda c: c["id"])
    preparation = json.loads((directory / "preparation.json").read_text(encoding="utf-8"))
    is_visdrone = preparation["dataset"] == "VisDrone2019-DET"
    model = YOLO(str(checkpoint))
    if list(model.names.values()) != [c["name"] for c in categories]:
        raise ValueError("Checkpoint classes differ from the prepared evaluation dataset")
    original_params = sum(p.numel() for p in model.model.parameters())
    original_flops = get_flops(model.model, args.imgsz)
    metrics = model.val(data=str(data_path), imgsz=args.imgsz, device=args.device, batch=args.batch,
                        workers=args.workers, split=args.split, max_det=args.max_det,
                        project=str(output), name="ultralytics", plots=not args.no_plots)
    report = {"checkpoint": str(Path(checkpoint).resolve()), "dataset": preparation["dataset"],
              "split": args.split, "images": len(images), "ultralytics": metrics.results_dict,
              "validation_speed_ms": metrics.speed, "params": original_params, "GFLOPs": original_flops,
              "fused_params": sum(p.numel() for p in model.model.parameters()),
              "fused_GFLOPs": get_flops(model.model, args.imgsz),
              "ultralytics_per_class": [{"class_id": int(cid), "name": model.names[int(cid)],
                  "Precision": float(metrics.box.p[k]), "Recall": float(metrics.box.r[k]),
                  "AP50": float(metrics.box.ap50[k]), "AP50_95": float(metrics.box.ap[k])}
                  for k, cid in enumerate(metrics.box.ap_class_index)]}
    groups = {name: {"targets": 0, "matched": 0, "iou_sum": 0.0, "center_error_sum": 0.0,
                     "gate_sum": 0.0, "gate_count": 0} for name in ("dense", "sparse")}
    gate_cache, handles = {}, []
    router = next((m for m in model.model.modules() if isinstance(m, NSCRFusion) and m.mode == "gated"), None)
    if router is not None:
        def capture_gate(module, inputs, logits):
            gate_cache["logits"] = logits.detach().float().cpu()

        handles.append(router.gate.register_forward_hook(capture_gate))
    detections, image_ids = [], []
    official_dir = output / "visdrone_official_predictions"
    if is_visdrone:
        official_dir.mkdir(exist_ok=True)
    try:
        for path in images:
            result = model.predict(str(path), imgsz=args.imgsz, device=args.device,
                                   conf=0.001, iou=0.7, max_det=args.max_det, verbose=False)[0]
            pred = result.boxes.data.detach().float().cpu().numpy()
            meta = lookup[path.name]
            image_ids.append(meta["id"])
            official_lines = []
            for x1, y1, x2, y2, confidence, cls in pred:
                category = categories[int(cls)]["id"]
                box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                detections.append({"image_id": meta["id"], "category_id": category, "bbox": box, "score": float(confidence)})
                official_lines.append(",".join(f"{v:.6f}" for v in (*box, confidence)) + f",{category},-1,-1")
            if is_visdrone:
                (official_dir / f"{path.stem}.txt").write_text("\n".join(official_lines) + ("\n" if official_lines else ""), encoding="utf-8")
            # Diagnostics use exactly the positive boxes seen by the YOLO validation loader.
            label_file = path.parent.parent / "labels" / f"{path.stem}.txt"
            labels = np.array([list(map(float, row.split())) for row in label_file.read_text().splitlines()], dtype=np.float32).reshape(-1, 5)
            if len(labels) < 2:
                continue
            width, height = meta["width"], meta["height"]
            pixel = labels[:, 1:] * np.array([width, height, width, height])
            q = normalized_spacing(torch.tensor(pixel, dtype=torch.float32)).numpy()
            size = np.sqrt(pixel[:, 2] * pixel[:, 3])
            boxes = np.concatenate((pixel[:, :2] - pixel[:, 2:] / 2, pixel[:, :2] + pixel[:, 2:] / 2), 1)
            matched_iou, matched_centers = match_instances(boxes, labels[:, 0], pred, conf=args.conf)
            gate_values = None
            if "logits" in gate_cache:
                logits = gate_cache["logits"]
                input_h, input_w = logits.shape[-2] * 4, logits.shape[-1] * 4
                gain = min(input_w / width, input_h / height)
                pad = np.array([round((input_w - round(width * gain)) / 2 - 0.1),
                                round((input_h - round(height * gain)) / 2 - 0.1)])
                centers = (pixel[:, :2] * gain + pad) / np.array([input_w, input_h])
                gate_values = sample_gate(logits, torch.tensor(centers, dtype=torch.float32)).sigmoid().numpy()
            for name, mask in (("dense", q < args.tau), ("sparse", q >= args.tau)):
                mask &= (size <= args.small_size) & np.isfinite(q)
                hits = mask & (matched_iou > 0)
                group = groups[name]
                group["targets"] += int(mask.sum())
                group["matched"] += int(hits.sum())
                group["iou_sum"] += float(matched_iou[hits].sum())
                group["center_error_sum"] += float((np.linalg.norm(matched_centers[hits] - pixel[hits, :2], axis=1) / np.maximum(size[hits], 1e-6)).sum())
                if gate_values is not None:
                    group["gate_sum"] += float(gate_values[mask].sum())
                    group["gate_count"] += int(mask.sum())
    finally:
        for handle in handles:
            handle.remove()
    (output / "coco_predictions.json").write_text(json.dumps(detections), encoding="utf-8")
    (output / "evaluated_image_ids.json").write_text(json.dumps(image_ids), encoding="utf-8")
    report["coco_supplemental"] = coco_metrics(annotation_path, detections, image_ids, args.max_det)
    report["groups"] = {}
    for name, group in groups.items():
        report["groups"][name] = {"targets": group["targets"], "matched": group["matched"],
            "recall": group["matched"] / group["targets"] if group["targets"] else None,
            "mean_iou_matched": group["iou_sum"] / group["matched"] if group["matched"] else None,
            "mean_normalized_center_error_matched": group["center_error_sum"] / group["matched"] if group["matched"] else None,
            "mean_wide_gate": group["gate_sum"] / group["gate_count"] if group["gate_count"] else None}
    report["group_protocol"] = {"spacing_tau": args.tau, "small_size_original_pixels": args.small_size,
                                "confidence": args.conf, "IoU": 0.5, "matching": "class-aware one-to-one"}
    report["forward_benchmark"] = benchmark(model.model, args.imgsz, args.device)
    if is_visdrone:
        write_official_matlab(output, preparation["splits"][args.split]["original_annotations"], official_dir, images)
        report["visdrone_official"] = {"status": "exported; official toolkit must be run separately",
            "predictions": str(official_dir.resolve()),
            "ground_truth": preparation["splits"][args.split]["original_annotations"],
            "toolkit": "https://github.com/VisDrone/VisDrone2018-DET-toolkit"}
    else:
        report["aitod_official"] = {"status": "COCO predictions exported; run official AI-TOD-v2 tiny-object evaluator separately",
            "toolkit": "https://github.com/Chasel-Tsui/AI-TOD-v2"}
    (output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report
