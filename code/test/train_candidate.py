"""Train exactly one candidate, with the same base training protocol as NSCR."""
import argparse
import json
from pathlib import Path
from train_test import ROOT, train_one
from candidate_experiments import EXPERIMENTS, write_architecture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/nscr_visdrone/dataset.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "model/yolo11n.pt")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/candidates/formal_100e_b8_s42")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aux-weight", dest="spacing_weight", type=float, default=.1)
    parser.add_argument("--cache", choices=("none", "disk", "ram"), default="none")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.workers < 0 or args.imgsz < 32 or args.imgsz % 32:
        parser.error("epochs/batch positive, workers nonnegative, imgsz multiple of 32")
    args.data, args.weights, args.project = args.data.resolve(), args.weights.resolve(), args.project.resolve()
    if not args.data.is_file() or not args.weights.is_file():
        parser.error("Dataset and local pretrained model must exist")
    if (args.project / f"{args.experiment}_seed{args.seed}").exists() and not args.resume:
        parser.error("Run already exists; use a new project or explicitly resume last.pt")
    args.name, args.tuning_report = None, None
    args.tau, args.temperature, args.small_size = 2., .5, 32.
    args.density_sigma, args.density_threshold, args.density_temperature = 2., 2., .5
    args.patience, args.save_period, args.max_det = args.epochs + 1, 10, 500
    result = train_one(args, args.data, args.experiment, args.seed, architecture_writer=write_architecture)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
