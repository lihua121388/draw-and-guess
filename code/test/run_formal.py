"""Formal baseline/ablation launcher, preferring the local Anaconda yolo interpreter."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve_python(explicit=None):
    """Use the user's yolo environment even if the launcher runs under system Python."""
    local_yolo = Path.home() / "anaconda3/envs/yolo/python.exe"
    executable = Path(explicit).expanduser() if explicit else local_yolo if local_yolo.is_file() else Path(sys.executable)
    if not executable.is_file():
        raise FileNotFoundError(f"Python interpreter missing: {executable}")
    return executable.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("all", "E0", "E1", "E2", "E3", "E4", "E5", "E6"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/nscr_visdrone/dataset.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "model/yolo11n.pt")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/nscr/formal")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8, help="Positive batch size, or -1 for Ultralytics AutoBatch")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", choices=("none", "disk", "ram"), default="none")
    parser.add_argument("--device", default="0")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--python", type=Path, help="Explicit training interpreter; default prefers Anaconda yolo")
    parser.add_argument("--dry-run", action="store_true", help="Check paths and print the exact training command")
    tuning = parser.add_mutually_exclusive_group()
    tuning.add_argument("--auto-tune", action="store_true", help="Measure batch/workers before training")
    tuning.add_argument("--tuning-profile", type=Path, help="Reuse a compatible, completed tuning report")
    parser.add_argument("--tune-only", action="store_true", help="Tune and save the report without starting formal training")
    parser.add_argument("--tune-batches", nargs="+", type=int, default=[4, 8, 16, 24, 32])
    parser.add_argument("--tune-workers", nargs="+", type=int, default=[0, 4, 8, 12, 16])
    parser.add_argument("--tune-steps", type=int, default=20)
    parser.add_argument("--tune-warmup", type=int, default=5)
    parser.add_argument("--tune-memory-fraction", type=float, default=0.8)
    args = parser.parse_args()
    try:
        executable = resolve_python(args.python)
    except FileNotFoundError as error:
        parser.error(str(error))
    if not args.data.is_file() or not args.weights.is_file():
        parser.error("Prepared dataset YAML and local yolo11n.pt must exist; run prepare_data.py first")
    if args.epochs < 1 or args.imgsz < 32 or args.imgsz % 32 or (args.batch < 1 and args.batch != -1) or args.workers < 0:
        parser.error("epochs must be positive; batch positive or -1 for AutoBatch; imgsz a multiple of 32; workers nonnegative (no workers=-1 mode)")
    experiments = [f"E{i}" for i in range(7)] if args.experiment == "all" else [args.experiment]
    if args.tune_only and not args.auto_tune:
        parser.error("--tune-only requires --auto-tune")
    profile = args.tuning_profile
    if args.auto_tune:
        profile = args.project.resolve() / "tuning.json"
        tuning_command = [str(executable), str(ROOT / "code/test/auto_tune.py"),
            "--data", str(args.data.resolve()), "--weights", str(args.weights.resolve()),
            "--imgsz", str(args.imgsz), "--device", args.device, "--cache", args.cache,
            "--output", str(profile), "--steps", str(args.tune_steps), "--warmup", str(args.tune_warmup),
            "--memory-fraction", str(args.tune_memory_fraction),
            "--batches", *map(str, args.tune_batches), "--workers", *map(str, args.tune_workers),
            "--experiments", *experiments, "--python", str(executable)]
        if args.no_amp:
            tuning_command.append("--no-amp")
        print(subprocess.list2cmdline(tuning_command), flush=True)
        if args.dry_run:
            print("After tuning, the measured batch/workers will be used for all selected runs.", flush=True)
            return
        completed = subprocess.run(tuning_command, cwd=str(ROOT))
        if completed.returncode:
            raise SystemExit(completed.returncode)
        if args.tune_only:
            return
    if profile:
        from tuning_utils import load_profile, signature

        try:
            expected = signature(args.data, args.weights, args.imgsz, args.device, args.cache, not args.no_amp, executable)
            chosen = load_profile(profile, expected, experiments)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        args.batch, args.workers = chosen["batch"], chosen["workers"]
        print(f"Tuned batch={args.batch}, workers={args.workers}; profile={profile}", flush=True)
    command = [str(executable), str(ROOT / "code/test/train_test.py"), "ablate",
               "--data", str(args.data.resolve()), "--weights", str(args.weights.resolve()),
               "--project", str(args.project.resolve()), "--epochs", str(args.epochs),
               "--imgsz", str(args.imgsz), "--batch", str(args.batch), "--workers", str(args.workers),
               "--cache", args.cache,
               "--device", args.device, "--experiments", *experiments, "--seeds", *map(str, args.seeds)]
    if args.no_amp:
        command.append("--no-amp")
    if args.no_plots:
        command.append("--no-plots")
    if profile:
        command.extend(["--tuning-report", str(profile.resolve())])
    print(f"Training interpreter: {executable}", flush=True)
    print(f"{len(experiments) * len(args.seeds)} run(s): {', '.join(experiments)}; seeds={args.seeds}", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    if not args.dry_run:
        completed = subprocess.run(command, cwd=str(ROOT))
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
