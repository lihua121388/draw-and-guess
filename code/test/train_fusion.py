"""Train one fusion diagnosis with exactly the existing M0 protocol."""
import argparse
import csv
import json
from pathlib import Path
import train_test
from train_test import ROOT, train_one
from fusion_experiments import EXPERIMENTS, write_architecture
from ultralytics.nn.modules.p2_fusion import P2Fusion


class FusionTrainer(train_test.ResearchTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_end", self.record_weights)

    def record_weights(self, trainer):
        module = trainer.model.model[23]
        if not isinstance(module, P2Fusion) or module.last_weights is None:
            return
        weights = module.last_weights.float().cpu()
        module.last_weights = None
        row = dict(epoch=trainer.epoch + 1, scope="last training batch; not dataset average",
                   mean=float(weights[:, :1].mean()), std=float(weights[:, :1].std(unbiased=False)),
                   below_005=float((weights[:, :1] < .05).float().mean()),
                   above_095=float((weights[:, :1] > .95).float().mean()))
        path = trainer.save_dir / "fusion_weights.csv"
        existed = path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not existed:
                writer.writeheader()
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/nscr_visdrone/dataset.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "model/yolo11n.pt")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/fusion/formal_100e_b8_s42")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", choices=("none", "disk", "ram"), default="none")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.workers < 0 or args.imgsz < 32 or args.imgsz % 32:
        parser.error("Positive epochs/batch, nonnegative workers, imgsz multiple of 32")
    args.data, args.weights, args.project = args.data.resolve(), args.weights.resolve(), args.project.resolve()
    if not args.data.is_file() or not args.weights.is_file():
        parser.error("Prepared dataset and pretrained weights must exist")
    if (args.project / f"{args.experiment}_seed{args.seed}").exists() and not args.resume:
        parser.error("Run exists; use another project or explicitly resume last.pt")
    args.name, args.tuning_report, args.spacing_weight = None, None, 0.
    args.tau, args.temperature, args.small_size = 2., .5, 32.
    args.density_sigma, args.density_threshold, args.density_temperature = 2., 2., .5
    args.patience, args.save_period, args.max_det = args.epochs + 1, 10, 500
    train_test.ResearchTrainer = FusionTrainer
    result = train_one(args, args.data, args.experiment, args.seed, architecture_writer=write_architecture)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
