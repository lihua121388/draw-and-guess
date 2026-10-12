"""Describe existing box geometry; do not use validation geometry to tune training."""
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
from PIL import Image
import torch
import yaml
from ultralytics.utils.dcr_loss import directional_clearance, clearance_distribution
from ultralytics.utils.nscr_loss import normalized_spacing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    report, points = {}, {}
    for split in ("train", "val"):
        images = [Path(x) for x in (Path(cfg["path"]) / cfg[split]).read_text().splitlines() if x]
        entries, singleton = [], 0
        for path in images:
            with Image.open(path) as image:
                width, height = image.size
            rows = np.array([list(map(float, row.split())) for row in
                            (path.parent.parent / "labels" / f"{path.stem}.txt").read_text().splitlines()],
                            dtype=np.float32).reshape(-1, 5)
            if len(rows) < 2:
                singleton += 1
                continue
            boxes = torch.tensor(rows[:, 1:]) * torch.tensor([width, height, width, height])
            small = boxes[:, 2:].prod(1).sqrt() * args.imgsz / max(width, height) <= 32
            if not small.any():
                continue
            q = directional_clearance(boxes, (height, width))[small]
            spacing = normalized_spacing(boxes)[small]
            target = clearance_distribution(q)
            entropy = -(target * target.log()).sum(1)
            entries.append(torch.stack((spacing, q.max(1).values - q.min(1).values, entropy), 1).numpy())
        data = np.concatenate(entries) if entries else np.zeros((0, 3))
        points[split] = data
        dense = data[:, 0] < 2
        anisotropic = data[:, 1] >= 1
        report[split] = dict(images=len(images), excluded_single_or_empty_images=singleton,
            tiny_instances=len(data), anisotropic_fraction=float(anisotropic.mean()) if len(data) else None,
            dense_instances=int(dense.sum()),
            anisotropic_fraction_in_dense=float(anisotropic[dense].mean()) if dense.any() else None,
            mean_target_entropy=float(data[:, 2].mean()) if len(data) else None)
    report["protocol"] = "Descriptive positive-GT geometry only, no augmentation; original-image boundary, "
    report["protocol"] += "tiny if sqrt(area)*640/max(width,height)<=32. Anisotropic if max(q_d)-min(q_d)>=1; dense if old NSCR center spacing<2. "
    report["protocol"] += "This establishes label variation, not useful semantic context or detection benefit. No hyperparameters fitted."
    (args.output / "geometry.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, (split, data) in zip(axes, points.items()):
        ax.hist2d(np.clip(data[:, 0], 0, 6), data[:, 1], bins=40, cmap="viridis")
        ax.axvline(2, color="white", linestyle="--", linewidth=1)
        ax.set_xlabel("Scalar center spacing (clipped at 6)")
        ax.set_ylabel("Directional clearance range")
        ax.set_title(split)
    fig.savefig(args.output / "geometry.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
