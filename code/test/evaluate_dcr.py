"""Full supplemental evaluation plus GT-based direction diagnostics on held-out images."""
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
from ultralytics.nn.modules.dcr import DCRFusion
from ultralytics.utils.dcr_loss import directional_clearance, clearance_distribution
from evaluate_nscr import evaluate


def direction_diagnostics(checkpoint, data_path, args):
    model = YOLO(str(checkpoint))
    router = next((m for m in model.model.modules() if isinstance(m, DCRFusion)), None)
    if router is None:
        return {"status": "not a DCR model"}
    cfg = yaml.safe_load(Path(data_path).read_text(encoding="utf-8"))
    images = [Path(x) for x in (Path(cfg["path"]) / cfg[args.split]).read_text().splitlines() if x]
    cache, records = {}, []
    config = model.model.yaml["dcr"]

    def capture(module, inputs, output):
        cache["logits"] = output.detach().float().cpu()

    handle = router.gate.register_forward_hook(capture)
    try:
        for path in images:
            cache.clear()
            result = model.predict(str(path), imgsz=args.imgsz, device=args.device, conf=.001,
                                   max_det=args.max_det, verbose=False)[0]
            labels = torch.tensor(np.array([list(map(float, row.split())) for row in
                (path.parent.parent / "labels" / f"{path.stem}.txt").read_text().splitlines()],
                dtype=np.float32).reshape(-1, 5))
            if len(labels) < 2:
                continue
            logits = cache["logits"]
            height, width = result.orig_shape
            input_h, input_w = logits.shape[-2] * 4, logits.shape[-1] * 4
            gain = min(input_w / width, input_h / height)
            pad = torch.tensor([round((input_w - round(width * gain)) / 2 - .1),
                                round((input_h - round(height * gain)) / 2 - .1)])
            pixels = labels[:, 1:] * torch.tensor([width, height, width, height]) * gain
            pixels[:, :2] += pad
            small = pixels[:, 2:].prod(1).sqrt() <= config["small_size"]
            if not small.any():
                continue
            q = directional_clearance(pixels, (input_h, input_w), cap=config["cap"])[small]
            target = clearance_distribution(q, config["tau"], config["temperature"], "directional")
            centers = pixels[small, :2] / torch.tensor([input_w, input_h])
            sampled = F.grid_sample(logits, (centers * 2 - 1).reshape(1, -1, 1, 2),
                                    mode="bilinear", padding_mode="border", align_corners=False)[0, :, :, 0].T
            log_p = sampled.log_softmax(1)
            p = log_p.exp()
            kl = (target * (target.clamp_min(1e-9).log() - log_p)).sum(1)
            uniform_kl = (target * (target.clamp_min(1e-9).log() + np.log(5))).sum(1)
            anisotropic = q.max(1).values - q.min(1).values >= 1.
            best = q.argmax(1)
            worst = q.argmin(1)
            for i in range(len(q)):
                records.append(dict(kl=float(kl[i]), uniform_kl=float(uniform_kl[i]),
                    local_prob=float(p[i, 0]), anisotropic=bool(anisotropic[i]),
                    direction_hit=bool(p[i, 1:].argmax() == best[i]),
                    clear_minus_blocked=float(p[i, 1 + best[i]] - p[i, 1 + worst[i]])))
    finally:
        handle.remove()
    selected = [r for r in records if r["anisotropic"]]
    return {"status": "evaluated", "targets": len(records), "anisotropic_targets": len(selected),
            "mean_KL_target_to_prediction": float(np.mean([r["kl"] for r in records])) if records else None,
            "mean_KL_target_to_uniform": float(np.mean([r["uniform_kl"] for r in records])) if records else None,
            "mean_local_probability": float(np.mean([r["local_prob"] for r in records])) if records else None,
            "best_direction_accuracy": float(np.mean([r["direction_hit"] for r in selected])) if selected else None,
            "clear_minus_blocked_probability": float(np.mean([r["clear_minus_blocked"] for r in selected])) if selected else None,
            "protocol": "Training-target geometry on letterboxed held-out RGB; GT used only after prediction. "
                        "tiny sqrt-area<=configured pixels in model input; anisotropic gap range>=1. "
                        "Argmax ties use first R/L/D/U direction. Mechanism diagnostic, not AP."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", default=["E1", "E4", "D1", "D2", "D3", "D4"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
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
            checkpoint = run / "weights/best.pt"
            if not checkpoint.is_file() or not (run / "summary.json").is_file():
                raise FileNotFoundError(f"Complete training first: {run}")
            output = run / "evaluation"
            report = evaluate(checkpoint, args.data, output, args)
            report["directions"] = direction_diagnostics(checkpoint, args.data, args)
            (output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
