"""Explicit sequential launcher; never starts automatically after an old queue stops."""
import argparse
import json
import os
import subprocess
from pathlib import Path
from run_formal import ROOT, resolve_python
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs/nscr/.config"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/nscr/.matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
from run_five import now, write_json
from progress_console import child_environment, wait_with_console

EXPERIMENTS = ("F0", "F1", "F2_GLOBAL", "F2", "F3_PLAIN", "F3")


def check_reference(args):
    """Prevent a full-budget queue from silently comparing against a different protocol."""
    if args.epochs != 100:
        return {"status": "functional_check_only", "accuracy_comparison": False}
    from nscr_experiments import sha256
    reference = args.baseline_project / "M0_seed42"
    manifest = json.loads((reference / "experiment.json").read_text(encoding="utf-8"))
    summary = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    if summary.get("epochs") != 100 or not (reference / "evaluation/evaluation.json").exists():
        raise ValueError("M0 must finish the full training/evaluation budget")
    if "F0" in args.experiments and args.experiments[0] != "F0":
        raise ValueError("Run F0 first to establish same-seed controls")
    settings_differences = {
        name: [manifest["arguments"]["effective_training_arguments"][name], getattr(args, name)]
        for name in ("batch", "workers", "imgsz", "device")
        if getattr(args, name) != manifest["arguments"]["effective_training_arguments"][name]
    }
    if settings_differences and "F0" not in args.experiments:
        raise ValueError(f"Formal settings differ from M0: {settings_differences}; run F0 first in this queue")
    if str(args.data.resolve()) != manifest["arguments"]["data"]:
        raise ValueError("Use the same prepared data YAML as M0")
    if sha256(args.weights) != manifest["checkpoint_sha256"]:
        raise ValueError("Pretraining checkpoint content changed")
    for path, checksum in manifest["dataset_files"].items():
        if sha256(path) != checksum:
            raise ValueError(f"Dataset manifest changed: {path}")
    # Label files, not just image lists, also belong to the matched protocol.
    import hashlib
    import yaml
    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    for split, checksum in manifest["dataset_labels_sha256"].items():
        digest = hashlib.sha256()
        for image in (Path(cfg["path"]) / cfg[split]).read_text(encoding="utf-8").splitlines():
            label = Path(image).parent.parent / "labels" / (Path(image).stem + ".txt")
            digest.update(label.name.encode()); digest.update(label.read_bytes())
        if digest.hexdigest() != checksum:
            raise ValueError(f"Labels changed in {split}")
    if any(s != 42 for s in args.seeds) and "F0" not in args.experiments:
        raise ValueError("For seeds43/44 include F0 as a same-seed control")
    resumed = getattr(args, "resume_first", None) is not None
    if resumed and settings_differences:
        raise ValueError("A mixed-history resume requires pending experiments to match the historical M0 settings")
    return {"status": "mixed_history_resume" if resumed else "new_F0_control" if "F0" in args.experiments else "matched_M0",
            "historical_reference": str(reference), "settings_differences": settings_differences,
            "seed42_control": str(args.project / "F0_seed42") if "F0" in args.experiments and not resumed else str(reference)}


def check_resume_request(args):
    """Explicitly resume a stopped first F0; never rerun completed or partially started candidates."""
    if args.experiments[0] != "F0" or args.seeds != [42]:
        raise ValueError("--resume-first requires F0 first and seed42 only")
    checkpoint = args.resume_first.resolve()
    run = args.project / "F0_seed42"
    if checkpoint != run / "weights/last.pt" or not checkpoint.is_file():
        raise ValueError("Resume the existing project's F0_seed42/weights/last.pt")
    previous = json.loads((args.project / "queue.json").read_text(encoding="utf-8"))
    if previous["status"] not in {"stopped_by_user", "failed", "stopped_before_next_run"}:
        raise ValueError("Existing queue must be stopped before resuming")
    import psutil
    try:
        command = psutil.Process(previous["supervisor_pid"]).cmdline()
        if any(Path(part).name.lower() == "run_fusion.py" for part in command):
            raise ValueError("Previous supervisor is still running")
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    for experiment in args.experiments[1:]:
        if (args.project / f"{experiment}_seed42").exists():
            raise ValueError(f"Pending run already exists: {experiment}; refuse duplicate training")
    import torch
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    completed = int(ckpt["epoch"]) + 1
    if not 0 < completed < args.epochs or ckpt.get("optimizer") is None:
        raise ValueError("Checkpoint must be incomplete and contain resumable optimizer state")
    saved = ckpt["train_args"]
    for key in ("epochs", "workers", "imgsz", "seed", "device"):
        requested = args.seeds[0] if key == "seed" else getattr(args, key)
        if saved[key] != requested:
            raise ValueError(f"Resume changes unsupported setting: {key}")
    if Path(saved["project"]).resolve() != args.project or saved["name"] != "F0_seed42":
        raise ValueError("Checkpoint belongs to another run")
    model = ckpt.get("ema") or ckpt.get("model")
    if model.yaml.get("nscr", {}).get("experiment") != "F0":
        raise ValueError("Checkpoint must be F0")
    from nscr_experiments import sha256
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                completed_epochs=completed, old_batch=saved["batch"], new_batch=args.batch,
                mixed_training_history=saved["batch"] != args.batch, excluded_as_formal_control=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS,
                        default=["F1", "F2_GLOBAL", "F2", "F3_PLAIN", "F3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/nscr_visdrone/dataset.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "model/yolo11n.pt")
    parser.add_argument("--baseline-project", type=Path, default=ROOT / "runs/candidates/formal_100e_b8_s42_v2")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/fusion/formal_100e_b8_s42")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-first", type=Path,
                        help="Explicitly resume stopped F0 last.pt in this project; use historical M0 for formal comparisons")
    args = parser.parse_args()
    python = resolve_python(args.python)
    args.project = args.project.resolve()
    if len(set(args.experiments)) != len(args.experiments) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate experiments/seeds")
    if args.epochs < 1 or args.batch < 1 or args.workers < 0 or args.imgsz % 32 or args.imgsz < 32:
        parser.error("Invalid training budget")
    if not args.data.is_file() or not args.weights.is_file():
        parser.error("Data and weights are required")
    try:
        reference_check = check_reference(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    resume_info = None
    if args.resume_first:
        try:
            resume_info = check_resume_request(args)
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
    if args.project.exists() and not args.dry_run and not args.resume_first:
        parser.error("Use a new project; interrupted checkpoints require train_fusion.py --resume")
    jobs = []
    for experiment in args.experiments:
        for seed in args.seeds:
            command = [str(python), "-u", str(ROOT / "code/test/train_fusion.py"),
                "--experiment", experiment, "--data", str(args.data.resolve()), "--weights", str(args.weights.resolve()),
                "--project", str(args.project), "--epochs", str(args.epochs), "--batch", str(args.batch),
                "--workers", str(args.workers), "--imgsz", str(args.imgsz), "--device", args.device, "--seed", str(seed)]
            if args.no_plots:
                command.append("--no-plots")
            job = dict(experiment=experiment, seed=seed, status="pending", command=command)
            if resume_info and not jobs:
                command.extend(["--resume", resume_info["checkpoint"]])
                job["resume"] = resume_info
            jobs.append(job)
    if args.dry_run:
        for job in jobs:
            print(subprocess.list2cmdline(job["command"]))
        return
    if resume_info:
        # Archive the original phase before the trainer writes new args/manifest/checkpoints.
        import shutil
        archive = args.project / "history" / now().replace(":", "-")
        archive.mkdir(parents=True)
        for source in (args.project / "queue.json", args.project / "F0_seed42/experiment.json",
                       args.project / "F0_seed42/args.yaml", Path(resume_info["checkpoint"]),
                       args.project / "F0_seed42/weights/best.pt", args.project / "F0_seed42/results.csv",
                       args.project / "logs/F0_seed42.log"):
            shutil.copy2(source, archive / source.name)
        stop = args.project / "STOP"
        if stop.exists():
            stop.replace(archive / "STOP")
        write_json(archive / "batch_transition.json", resume_info)
        write_json(args.project / "F0_seed42/batch_transition.json", resume_info)
    # Reuse M0 only after confirming its full budget and identities in the analysis report.
    logs = args.project / "logs"
    logs.mkdir(parents=True, exist_ok=bool(resume_info))
    state = dict(status="running", supervisor_pid=os.getpid(), started_at=now(),
                 arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 reference_check=reference_check, resume=resume_info, jobs=jobs)
    queue = args.project / "queue.json"
    write_json(queue, state)
    for job in jobs:
        if (args.project / "STOP").exists():
            state["status"] = "stopped_before_next_run"
            write_json(queue, state)
            return
        log = logs / f"{job['experiment']}_seed{job['seed']}.log"
        job.update(status="training", log=str(log), started_at=now())
        with log.open("w", encoding="utf-8") as stream:
            child = subprocess.Popen(job["command"], cwd=ROOT, stdout=stream,
                                     stderr=subprocess.STDOUT, env=child_environment())
            job["pid"] = child.pid
            write_json(queue, state)
            code = wait_with_console(child, log)
        job["training_return_code"] = code
        run = args.project / f"{job['experiment']}_seed{job['seed']}"
        summary_path = run / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if code or summary.get("epochs") != args.epochs:
            job["status"] = "training_failed"
            state["status"] = "failed"
            write_json(queue, state)
            raise SystemExit(code or 1)
        job["status"] = "evaluating"
        write_json(queue, state)
        command = [str(python), "-u", str(ROOT / "code/test/evaluate_candidates.py"),
            "--project", str(args.project), "--data", str(args.data.resolve()), "--experiments", job["experiment"],
            "--seeds", str(job["seed"]), "--batch", str(args.batch), "--workers", str(args.workers),
            "--imgsz", str(args.imgsz), "--device", args.device, "--no-plots"]
        evaluation_log = logs / f"{job['experiment']}_seed{job['seed']}_evaluation.log"
        with evaluation_log.open("w", encoding="utf-8") as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                                     stderr=subprocess.STDOUT, env=child_environment())
            job["evaluation_pid"] = child.pid
            write_json(queue, state)
            code = wait_with_console(child, evaluation_log)
        job.update(status="complete" if code == 0 else "evaluation_failed", evaluation_return_code=code,
                   finished_at=now())
        write_json(queue, state)
        if code:
            state["status"] = "failed"
            write_json(queue, state)
            raise SystemExit(code)
        # Re-score exported predictions from raw GT without overwriting historical YOLO/COCO metrics.
        protocol_log = logs / f"{job['experiment']}_seed{job['seed']}_visdrone_reference.log"
        import yaml
        data_config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
        command = [str(python), "-u", str(ROOT / "code/test/visdrone_protocol.py"),
            "--project", str(args.project), "--data-root", data_config["path"],
            "--experiments", job["experiment"], "--seed", str(job["seed"])]
        with protocol_log.open("w", encoding="utf-8") as stream:
            code = subprocess.call(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=child_environment())
        job["reference_evaluation_return_code"] = code
        if code:
            job["status"] = "reference_evaluation_failed"
            state["status"] = "failed"
            write_json(queue, state)
            raise SystemExit(code)
    analysis_baseline = args.project if "F0" in args.experiments and not args.resume_first else args.baseline_project.resolve()
    command = [str(python), str(ROOT / "code/test/analyze_fusion_results.py"),
               "--project", str(args.project), "--baseline-project", str(analysis_baseline)]
    with (logs / "analysis.log").open("w", encoding="utf-8") as stream:
        code = subprocess.call(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=child_environment())
    state.update(status="complete" if code == 0 else "analysis_failed", analysis_return_code=code, finished_at=now())
    write_json(queue, state)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
