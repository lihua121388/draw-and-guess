"""Train DCR and matched controls in the local Anaconda yolo environment."""
import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from run_formal import resolve_python

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", choices=("E0", "E1", "E3", "E4", "D1", "D2", "D3", "D4"),
                        default=["E1", "E4", "D1", "D2", "D3", "D4"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/nscr_visdrone/dataset.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "model/yolo11n.pt")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/dcr/formal")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cache", choices=("none", "disk", "ram"), default="none")
    parser.add_argument("--spacing-weight", type=float, default=.1, help="Auxiliary weight shared by supervised controls")
    parser.add_argument("--tau", type=float, default=2.)
    parser.add_argument("--temperature", type=float, default=.5)
    parser.add_argument("--small-size", type=float, default=32.)
    parser.add_argument("--cap", type=float, default=6.)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-complete", action="store_true", help="Reuse only runs with complete matching arguments and hashes")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    executable = resolve_python(args.python)
    if executable != Path(sys.executable).resolve():
        command = [str(executable), str(Path(__file__).resolve()), *sys.argv[1:]]
        print(subprocess.list2cmdline(command), flush=True)
        raise SystemExit(subprocess.call(command, cwd=str(ROOT)))
    if not args.data.is_file() or not args.weights.is_file():
        parser.error("Dataset YAML and local pretrained checkpoint must exist")
    if args.epochs < 1 or args.batch < 1 or args.workers < 0 or args.imgsz < 32 or args.imgsz % 32:
        parser.error("epochs/batch positive, workers nonnegative, imgsz a positive multiple of 32")
    if args.temperature <= 0 or args.cap <= 0 or args.small_size <= 0 or args.spacing_weight < 0:
        parser.error("temperature/cap/small-size positive and auxiliary weight nonnegative")
    if len(set(args.experiments)) != len(args.experiments) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate experiment/seed would overwrite a comparison")
    if args.dry_run:
        print(json.dumps(vars(args), indent=2, default=str))
        return
    from train_test import train_one
    from dcr_experiments import write_architecture
    from nscr_experiments import sha256
    args.name, args.resume, args.tuning_report = None, None, None
    args.density_sigma, args.density_threshold, args.density_temperature = 2., 2., .5
    args.patience, args.save_period, args.max_det = args.epochs + 1, -1, 500
    args.project.mkdir(parents=True, exist_ok=True)
    results = []

    def writer(experiment, directory, **kwargs):
        return write_architecture(experiment, directory, cap=args.cap, **kwargs)

    for experiment in args.experiments:
        for seed in args.seeds:
            existing = args.project / f"{experiment}_seed{seed}"
            if existing.exists():
                if not args.skip_complete or not (existing / "summary.json").is_file():
                    raise FileExistsError(f"Refusing duplicate/incomplete run: {existing}; choose another --project")
                summary = json.loads((existing / "summary.json").read_text())
                manifest = json.loads((existing / "experiment.json").read_text())
                recorded = manifest["arguments"]
                for key in ("epochs", "imgsz", "batch", "workers", "device", "cache", "spacing_weight", "tau", "temperature", "small_size", "cap", "no_amp"):
                    if recorded[key] != vars(args)[key]:
                        raise ValueError(f"Completed run differs in {key}: {existing}")
                if summary["epochs"] != args.epochs or manifest["checkpoint_sha256"] != sha256(args.weights):
                    raise ValueError(f"Completed run/checkpoint differs: {existing}")
                for path, digest in manifest["dataset_files"].items():
                    if sha256(path) != digest:
                        raise ValueError(f"Dataset manifest changed: {path}")
                for path, digest in manifest["implementation_files"].items():
                    if sha256(ROOT / path) != digest:
                        raise ValueError(f"Source changed: {path}; use a new project")
                results.append(summary)
            else:
                results.append(train_one(args, args.data.resolve(), experiment, seed, architecture_writer=writer))
            keys = list(dict.fromkeys(key for row in results for key in row))
            with (args.project / "ablation_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                csv_writer = csv.DictWriter(stream, fieldnames=keys)
                csv_writer.writeheader()
                csv_writer.writerows(results)
            aggregate = {}
            for label in args.experiments:
                rows = [r for r in results if r["experiment"] == label]
                if rows:
                    aggregate[label] = {"n": len(rows), **{metric: {"mean": statistics.mean(r[metric] for r in rows),
                        "sample_std": statistics.stdev(r[metric] for r in rows) if len(rows) > 1 else None}
                        for metric in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")}}
            (args.project / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"Completed {len(results)} controlled runs in {args.project}")


if __name__ == "__main__":
    main()
