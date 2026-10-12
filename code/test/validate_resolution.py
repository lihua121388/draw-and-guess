"""Validate fixed M0 weights at one input size; never trains or overwrites an existing evaluation."""
import argparse
import datetime
import json
from pathlib import Path
from train_test import ROOT
import torch
import yaml
from nscr_experiments import sha256
from evaluate_nscr import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/candidates/formal_100e_b8_s42_v2/M0_seed42/weights/best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/nscr_visdrone/dataset.yaml")
    parser.add_argument("--imgsz", type=int, required=True, choices=(640, 960, 1280))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.checkpoint, args.data, args.output = args.checkpoint.resolve(), args.data.resolve(), args.output.resolve()
    if args.output.exists():
        parser.error("Output exists: use a fresh directory; existing evaluation must be preserved")
    if args.batch < 1 or args.workers < 0:
        parser.error("Positive validation batch and nonnegative workers required")
    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    images = [p for p in (root / cfg["val"]).read_text().splitlines() if p]
    if len(images) != 548 or len(set(images)) != 548:
        parser.error("Use the unchanged 548-image official validation membership")
    args.split, args.max_det, args.conf = "val", 500, .25
    args.tau, args.small_size, args.no_plots = 2., 32., True
    provenance = dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint), data=str(args.data),
        dataset_yaml_sha256=sha256(args.data), val_manifest_sha256=sha256(root / cfg["val"]),
        annotation_sha256=sha256(root / "val_coco.json"), images=548, image_paths=images,
        imgsz=args.imgsz, requested_batch=args.batch, actual_batch=args.batch, workers=args.workers,
        device=args.device, train_resolution=640, max_det=500, NMS_iou=.7, prediction_conf=.001,
        diagnostic_conf=.25, protocol="Exactly the existing evaluate_nscr pipeline: standalone Ultralytics val "
          "and single-image predictions for supplemental COCO. No training, TTA or architecture change. "
          "FP32 standalone validation/prediction; rectangular letterbox policy unchanged across sizes.")
    args.output.mkdir(parents=True)
    path = args.output / "resolution_provenance.json"
    path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if args.device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    report = evaluate(args.checkpoint, args.data, args.output, args)
    provenance.update(status="complete", finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3 if args.device != "cpu" else None,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3 if args.device != "cpu" else None)
    if sha256(args.checkpoint) != provenance["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed during validation")
    path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "imgsz": args.imgsz, "ultralytics": report["ultralytics"],
                      "small": report["coco_supplemental"]["AP_small"], "peak_reserved_GiB": provenance["peak_reserved_GiB"]}))


if __name__ == "__main__":
    main()
