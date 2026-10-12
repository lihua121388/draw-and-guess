"""Supplemental detection metrics and held-out teacher diagnostics for each candidate."""
import argparse
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs/nscr/.config"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/nscr/.matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import torch
import yaml
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.nn.modules.candidates import CandidateFusion
from ultralytics.utils.candidate_loss import candidate_auxiliary
from evaluate_nscr import evaluate


def teacher_diagnostics(checkpoint, data, args):
    model = YOLO(str(checkpoint))
    router = next((m for m in model.model.modules() if isinstance(m, CandidateFusion)), None)
    if router is None:
        return {"status": "P2 control, no candidate teacher"}
    config = dict(model.model.yaml["candidate"], nc=len(model.names), weight=1.)
    cfg = yaml.safe_load(Path(data).read_text(encoding="utf-8"))
    images = [Path(x) for x in (Path(cfg["path"]) / cfg[args.split]).read_text().splitlines() if x]
    cache, handles, values = {}, [], []

    def capture_logits(module, inputs, output):
        cache["logits"] = output.detach().float().cpu()

    head = router.impl.gate if router.kind == "dcr" else router.head
    handles.append(head.register_forward_hook(capture_logits))
    if router.kind == "pcr":
        def capture_embedding(module, inputs, output):
            cache["embedding"] = F.normalize(output.detach().float().cpu(), dim=1, eps=1e-6)
        handles.append(router.embedding.register_forward_hook(capture_embedding))
    try:
        for path in images:
            cache.clear()
            prediction = model.predict(str(path), imgsz=args.imgsz, device=args.device,
                max_det=args.max_det, conf=.001, verbose=False)[0]
            labels = torch.tensor(np.array([list(map(float, line.split())) for line in
                (path.parent.parent / "labels" / f"{path.stem}.txt").read_text().splitlines()],
                dtype=np.float32).reshape(-1, 5))
            if not len(labels):
                continue
            height, width = prediction.orig_shape
            input_h, input_w = cache["logits"].shape[-2] * 4, cache["logits"].shape[-1] * 4
            gain = min(input_w / width, input_h / height)
            pad = torch.tensor([round((input_w-round(width*gain))/2-.1), round((input_h-round(height*gain))/2-.1)])
            pixel = labels[:, 1:] * torch.tensor([width, height, width, height]) * gain
            pixel[:, :2] += pad
            batch = dict(img=torch.zeros(1, 3, input_h, input_w),
                bboxes=pixel / torch.tensor([input_w, input_h, input_w, input_h]),
                cls=labels[:, :1], batch_idx=torch.zeros(len(labels), dtype=torch.long))
            value = float(candidate_auxiliary(router.kind, cache, batch, config))
            if np.isfinite(value):
                values.append(value)
    finally:
        for handle in handles:
            handle.remove()
    return {"status": "evaluated", "kind": router.kind, "images": len(values),
        "mean_unweighted_auxiliary": float(np.mean(values)) if values else None,
        "protocol": "Held-out GT teacher computed after RGB-only prediction, no augmentation, letterboxed coordinates. "
                    "Unweighted loss is a mechanism diagnostic; different teachers/loss entropies cannot be compared as detection quality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", default=["M1", "M2", "M3", "M4", "M5", "M0"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--tau", type=float, default=2.)
    parser.add_argument("--small-size", type=float, default=32.)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    for experiment in args.experiments:
        for seed in args.seeds:
            run = args.project / f"{experiment}_seed{seed}"
            if not (run / "summary.json").is_file():
                raise FileNotFoundError(f"Training not complete: {run}")
            checkpoint = run / "weights/best.pt"
            report = evaluate(checkpoint, args.data, run / "evaluation", args)
            report["candidate_diagnostics"] = teacher_diagnostics(checkpoint, args.data, args)
            (run / "evaluation/evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
