"""NSCR-YOLO training, ablations, validation and RGB-only inference.

Run in the Anaconda yolo environment. Default: E4, official VisDrone train/val,
local model/yolo11n.pt, 100 epochs. Paths are relative to this repository.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs/nscr/.config"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/nscr/.matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import torch
    import yaml
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing dependency: {error.name}\nCurrent Python: {sys.executable}\n"
        "Use the Anaconda yolo interpreter, or run code/test/run_formal.py "
        "which selects the local yolo environment automatically."
    ) from None

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops

from nscr_experiments import EXPERIMENTS, save_manifest, transfer_pretrained, write_architecture
from prepare_data import prepare_visdrone


class ResearchTrainer(DetectionTrainer):
    """Use local pretraining with detection-head remapping and ordinary YOLO training."""

    def __init__(self, checkpoint, **kwargs):
        self.checkpoint = Path(checkpoint)
        self.transfer_report = {}
        super().__init__(**kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        model.names = self.data["names"]
        if weights is not None:
            model.load(weights)
            self.transfer_report = {"resumed_checkpoint": str(self.args.model)}
        else:
            self.transfer_report = transfer_pretrained(model, self.checkpoint)
            print(f"Local pretraining: {self.transfer_report}")
        return model

    def plot_metrics(self):
        """Plot all four loss terms without relying on upstream fixed CSV column offsets."""
        if not any(key in self.model.yaml for key in ("nscr", "dcr", "candidate")):
            return super().plot_metrics()
        import matplotlib.pyplot as plt
        import pandas as pd

        data = pd.read_csv(self.csv)
        data.columns = data.columns.str.strip()
        columns = [f"train/{name}" for name in self.loss_names] + ["metrics/precision(B)", "metrics/recall(B)"]
        columns += [f"val/{name}" for name in self.loss_names] + ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]
        figure, axes = plt.subplots(2, 6, figsize=(16, 7), constrained_layout=True)
        for axis, name in zip(axes.flat, columns):
            axis.plot(data["epoch"], data[name], marker=".")
            axis.set_title(name)
            axis.set_xlabel("epoch")
        path = self.save_dir / "results.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        self.on_plot(path)


def dataset_path(args):
    if args.data:
        path = Path(args.data).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    directory = ROOT / "datasets/nscr_visdrone"
    suffix = f"-subset-{args.train_limit}-{args.val_limit}-s{args.seed}" if args.train_limit or args.val_limit else ""
    path = directory / f"dataset{suffix}.yaml"
    if not path.exists() or args.command == "prepare":
        path = prepare_visdrone(args.source, directory, args.train_limit, args.val_limit, args.seed)
    return path


def train_one(args, data, experiment, seed, architecture_writer=write_architecture):
    names = yaml.safe_load(data.read_text(encoding="utf-8"))["names"]
    cfg = architecture_writer(experiment, ROOT / "code/test/configs/generated", nc=len(names),
                             weight=args.spacing_weight, tau=args.tau, temperature=args.temperature,
                             small_size=args.small_size, density_sigma=args.density_sigma,
                             density_threshold=args.density_threshold, density_temperature=args.density_temperature)
    name = args.name or f"{experiment}_seed{seed}"
    overrides = dict(model=str(args.resume or cfg), data=str(data), task="detect", mode="train",
                     epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                     workers=args.workers, project=str(args.project), name=name, exist_ok=False,
                     seed=seed, deterministic=True, optimizer="AdamW", lr0=0.001, cos_lr=True,
                     patience=args.patience, amp=not args.no_amp, plots=not args.no_plots,
                     cache=False if args.cache == "none" else args.cache,
                     close_mosaic=min(10, max(args.epochs - 1, 0)), max_det=args.max_det,
                     pretrained=False, resume=str(args.resume) if args.resume else False, save_period=args.save_period,
                     mosaic=1.0, mixup=0.0, cutmix=0.0, copy_paste=0.0)
    trainer = ResearchTrainer(args.weights, overrides=overrides)

    def record_manifest(active_trainer):
        manifest_args = argparse.Namespace(**vars(args))
        manifest_args.seed = active_trainer.args.seed
        metadata = active_trainer.model.yaml.get("candidate", active_trainer.model.yaml.get("dcr", active_trainer.model.yaml.get("nscr", {})))
        manifest_args.experiment = metadata.get("experiment", "E0")
        manifest_args.effective_training_arguments = vars(active_trainer.args)
        save_manifest(active_trainer.save_dir / "experiment.json", manifest_args, data,
                      active_trainer.model.yaml, active_trainer.transfer_report)

    trainer.add_callback("on_pretrain_routine_end", record_manifest)
    trainer.train()
    model = YOLO(str(trainer.best if trainer.best.exists() else trainer.last))
    metrics = dict(trainer.metrics)
    metadata = model.model.yaml.get("candidate", model.model.yaml.get("dcr", model.model.yaml.get("nscr", {})))
    metrics.update(experiment=metadata.get("experiment", "E0"),
                   seed=trainer.args.seed, run=str(trainer.save_dir),
                   params=sum(p.numel() for p in model.model.parameters()),
                   GFLOPs=get_flops(model.model, args.imgsz),
                   status="training_complete", epochs=trainer.epoch + 1, requested_epochs=args.epochs)
    (trainer.save_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("prepare", "train", "ablate", "val", "predict", "evaluate"), default="train")
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), default="E4")
    parser.add_argument("--experiments", nargs="+", choices=tuple(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/VisDrone2019-DET")
    parser.add_argument("--weights", type=Path, default=ROOT / "model/yolo11n.pt")
    parser.add_argument("--checkpoint", type=Path, help="Trained best.pt for val/predict/evaluate")
    parser.add_argument("--resume", type=Path, help="Resumable last.pt or periodic checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8, help="Positive batch size, or -1 for Ultralytics AutoBatch")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", choices=("none", "disk", "ram"), default="none",
                        help="Image cache: disk preserves the original decode path; RAM may affect determinism")
    parser.add_argument("--tuning-report", type=Path, help="Provenance report supplied by the formal launcher")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--spacing-weight", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--small-size", type=float, default=32.0)
    parser.add_argument("--density-sigma", type=float, default=2.0)
    parser.add_argument("--density-threshold", type=float, default=2.0)
    parser.add_argument("--density-temperature", type=float, default=0.5)
    parser.add_argument("--train-limit", type=int, default=0, help="0=all official training images")
    parser.add_argument("--val-limit", type=int, default=0, help="0=full official validation; subsets for smoke checks")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/nscr")
    parser.add_argument("--name")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--predict-source", help="Local image, directory or video")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    if (args.batch < 1 and args.batch != -1) or args.workers < 0:
        parser.error("batch must be positive or -1 for AutoBatch; workers must be nonnegative (no workers=-1 mode)")
    if args.epochs <= 0 or args.temperature <= 0 or args.small_size <= 0 or args.spacing_weight < 0:
        parser.error("epochs, temperature, small-size must be positive; spacing-weight must be nonnegative")
    if args.train_limit < 0 or args.val_limit < 0:
        parser.error("Subset limits must be nonnegative")
    if "," in args.device:
        parser.error("This experiment runner currently supports one GPU or --device cpu")
    if args.command != "prepare" and args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable in {sys.executable}; activate the Anaconda yolo environment")
    if args.command in ("train", "ablate") and not args.weights.is_file():
        raise FileNotFoundError(f"Local pretrained model missing: {args.weights}")
    if args.command in ("val", "predict", "evaluate") and args.checkpoint and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.resume and not args.resume.is_file():
        raise FileNotFoundError(args.resume)
    if args.tuning_report and not args.tuning_report.is_file():
        raise FileNotFoundError(args.tuning_report)
    if args.resume and args.command != "train":
        parser.error("--resume is supported for train only")
    print(f"Python: {sys.executable}\nPyTorch: {torch.__version__}\nCUDA: {torch.cuda.is_available()}")
    if args.command == "predict":
        if not args.checkpoint or not args.predict_source:
            parser.error("predict requires --checkpoint and --predict-source")
        YOLO(str(args.checkpoint)).predict(source=args.predict_source, imgsz=args.imgsz, device=args.device,
            conf=args.conf, max_det=args.max_det, save=True, project=str(args.project), name=args.name or "predict")
        return
    data = dataset_path(args)
    if args.command == "prepare":
        print(data)
        return
    if args.command in ("val", "evaluate"):
        if not args.checkpoint:
            parser.error("val/evaluate requires --checkpoint")
        if args.command == "evaluate":
            from evaluate_nscr import evaluate

            evaluate(args.checkpoint, data, args.project / (args.name or "evaluation"), args)
        else:
            YOLO(str(args.checkpoint)).val(data=str(data), imgsz=args.imgsz, device=args.device,
                batch=args.batch, workers=args.workers, split=args.split, max_det=args.max_det,
                project=str(args.project), name=args.name or "validation", plots=not args.no_plots)
        return
    experiments = args.experiments if args.command == "ablate" else [args.experiment]
    if args.name and (len(experiments) > 1 or len(args.seeds or [args.seed]) > 1):
        parser.error("Omit --name when running multiple experiments/seeds")
    results = []
    for experiment in experiments:
        for seed in args.seeds or [args.seed]:
            results.append(train_one(args, data, experiment, seed))
            keys = list(dict.fromkeys(key for result in results for key in result))
            args.project.mkdir(parents=True, exist_ok=True)
            with (args.project / "ablation_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
    print(f"Completed {len(results)} run(s): {args.project}")


if __name__ == "__main__":
    main()
