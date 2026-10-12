"""Durable isolated queue: five candidates, then a matching P2 baseline.

Each child uses fixed batch (default 8), 100 epochs and the same official
VisDrone train/val. Status, child PID and separate logs are saved continuously.
An individual failure is recorded and does not silently change batch or epochs.
"""
import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from run_formal import ROOT, resolve_python
from progress_console import child_environment, wait_with_console


def queue_process_alive(pid):
    import psutil
    try:
        command = psutil.Process(pid).cmdline()
        return any(Path(part).name.lower() in {"run_five.py", "run_fusion.py"} for part in command)
    except psutil.NoSuchProcess:
        return False


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, data):
    # Same-directory atomic replace avoids partially written queue state.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def summarize(project, rows):
    results = []
    for row in rows:
        path = project / f"{row['experiment']}_seed{row['seed']}" / "summary.json"
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("status") == "training_complete" and result.get("epochs") == result.get("requested_epochs"):
                results.append(result)
    if results:
        keys = list(dict.fromkeys(k for row in results for k in row))
        with (project / "ablation_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader(); writer.writerows(results)
        write_json(project / "completed_results.json", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", choices=("M0", "M1", "M2", "M3", "M4", "M5"),
                        default=["M1", "M2", "M3", "M4", "M5", "M0"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/nscr_visdrone/dataset.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "model/yolo11n.pt")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/candidates/formal_100e_b8_s42")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--aux-weight", type=float, default=.1)
    parser.add_argument("--cache", choices=("none", "disk", "ram"), default="none")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true", help="Skip supplemental metrics after each completed model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-queue", action="store_true", help="Continue completed/pending jobs in an existing stopped queue")
    args = parser.parse_args()
    python = resolve_python(args.python)
    if args.batch < 1 or args.epochs < 1 or args.workers < 0 or args.imgsz < 32 or args.imgsz % 32:
        parser.error("Positive batch/epochs, nonnegative workers, imgsz multiple of 32")
    if len(set(args.experiments)) != len(args.experiments) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate experiment/seed")
    args.data, args.weights, args.project = args.data.resolve(), args.weights.resolve(), args.project.resolve()
    if not args.data.is_file() or not args.weights.is_file():
        parser.error("Prepared full dataset YAML and local yolo11n.pt required")
    jobs = []
    for experiment in args.experiments:
        for seed in args.seeds:
            command = [str(python), "-u", str(ROOT / "code/test/train_candidate.py"),
                "--experiment", experiment, "--data", str(args.data), "--weights", str(args.weights),
                "--project", str(args.project), "--epochs", str(args.epochs), "--batch", str(args.batch),
                "--workers", str(args.workers), "--imgsz", str(args.imgsz), "--device", args.device,
                "--seed", str(seed), "--aux-weight", str(args.aux_weight), "--cache", args.cache]
            for flag in ("no_amp", "no_plots"):
                if getattr(args, flag):
                    command.append("--" + flag.replace("_", "-"))
            jobs.append(dict(experiment=experiment, seed=seed, status="pending", command=command))
    previous = None
    queue = args.project / "queue.json"
    if args.resume_queue:
        if not queue.is_file():
            parser.error("--resume-queue requires an existing queue.json")
        previous = json.loads(queue.read_text(encoding="utf-8"))
        for key in ("data", "weights", "epochs", "batch", "workers", "imgsz", "device", "aux_weight", "cache", "no_amp", "no_plots", "no_evaluate"):
            value = str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key)
            if value != previous["arguments"][key]:
                parser.error(f"Cannot change existing training setting {key} while resuming")
        old = {(j["experiment"], j["seed"]): j for j in previous["jobs"]}
        if set(old) != {(j["experiment"], j["seed"]) for j in jobs}:
            parser.error("Resume must preserve all experiment/seed pairs; only their order may change")
        if not args.dry_run and queue_process_alive(previous["supervisor_pid"]):
            parser.error("Existing queue is still running; wait for its safe handoff")
        for job in jobs:
            saved = old[(job["experiment"], job["seed"])]
            directory = args.project / f"{job['experiment']}_seed{job['seed']}"
            summary = directory / "summary.json"
            completed = json.loads(summary.read_text(encoding="utf-8")) if summary.is_file() else {}
            if completed.get("status") == "training_complete" and completed.get("epochs") == args.epochs:
                job.update(saved)
                evaluation_done = (directory / "evaluation/evaluation.json").is_file() and saved.get("evaluation_return_code") == 0
                job["status"] = "complete" if evaluation_done else "training_complete"
            elif directory.exists() and not args.dry_run:
                parser.error(f"Interrupted training requires explicit checkpoint recovery: {directory}")
        if (args.project / "STOP").exists() and not args.dry_run:
            parser.error("STOP is present; a handoff must clear its own marker before resuming")
    elif not args.dry_run and (queue.exists() or any((args.project / f"{j['experiment']}_seed{j['seed']}").exists() for j in jobs)):
        parser.error("Refusing duplicate queue/run directory; choose a new --project")
    if args.dry_run:
        for job in jobs:
            if job["status"] == "pending":
                print(subprocess.list2cmdline(job["command"]))
            else:
                print(f"Skip completed training: {job['experiment']} seed={job['seed']}")
        return
    args.project.mkdir(parents=True, exist_ok=True)
    logs = args.project / "logs"
    logs.mkdir(exist_ok=True)
    state = dict(status="running", supervisor_pid=os.getpid(), started_at=now(), arguments={k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()}, jobs=jobs)
    if previous:
        history = list(previous.get("supervisor_history", []))
        history.append({"supervisor_pid": previous["supervisor_pid"], "started_at": previous["started_at"],
                        "status_at_handoff": previous["status"]})
        state["supervisor_history"] = history
    write_json(queue, state)
    failures = []
    for job in jobs:
        if job["status"] == "complete":
            print(f"Already complete: {job['experiment']} seed={job['seed']}", flush=True)
            continue
        if (args.project / "STOP").exists():
            state["status"] = "stopped_before_next_run"
            write_json(queue, state)
            return
        log = logs / f"{job['experiment']}_seed{job['seed']}.log"
        if job["status"] == "training_complete":
            code = 0
        else:
            job.update(status="training", started_at=now(), log=str(log))
            with log.open("w", encoding="utf-8") as stream:
                child = subprocess.Popen(job["command"], cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT,
                                         env=child_environment())
                job["pid"] = child.pid
                write_json(queue, state)
                print(f"Training {job['experiment']} seed={job['seed']} PID={child.pid}; log={log}", flush=True)
                code = wait_with_console(child, log)
        job.update(training_return_code=code, training_finished_at=now())
        summary = args.project / f"{job['experiment']}_seed{job['seed']}" / "summary.json"
        completed = json.loads(summary.read_text()) if summary.is_file() else {}
        if code or completed.get("epochs") != args.epochs:
            job["status"] = "training_failed"
            failures.append(job["experiment"])
            write_json(queue, state)
            continue
        job["status"] = "training_complete"
        summarize(args.project, jobs)
        write_json(queue, state)
        if not args.no_evaluate:
            job["status"] = "evaluating"
            write_json(queue, state)
            evaluation_log = logs / f"{job['experiment']}_seed{job['seed']}_evaluation.log"
            job["evaluation_log"] = str(evaluation_log)
            command = [str(python), "-u", str(ROOT / "code/test/evaluate_candidates.py"),
                "--project", str(args.project), "--data", str(args.data), "--experiments", job["experiment"],
                "--seeds", str(job["seed"]), "--batch", str(args.batch), "--workers", str(args.workers),
                "--imgsz", str(args.imgsz), "--device", args.device, "--no-plots"]
            with evaluation_log.open("w", encoding="utf-8") as stream:
                child = subprocess.Popen(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT,
                                         env=child_environment())
                job["evaluation_pid"] = child.pid
                write_json(queue, state)
                print(f"Evaluating {job['experiment']} seed={job['seed']}; log={evaluation_log}", flush=True)
                code = wait_with_console(child, evaluation_log)
            job.update(evaluation_return_code=code, evaluation_log=str(evaluation_log))
            job["status"] = "complete" if code == 0 else "evaluation_failed"
            if code:
                failures.append(job["experiment"] + "_evaluation")
        job["finished_at"] = now()
        write_json(queue, state)
    with (logs / "review.log").open("w", encoding="utf-8") as stream:
        code = subprocess.call([str(python), str(ROOT / "code/test/review_candidates.py"),
                                "--project", str(args.project)], cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT)
    state.update(status="completed_with_failures" if failures or code else "complete", failures=failures,
                 review_return_code=code, finished_at=now())
    write_json(queue, state)
    print(state["status"], flush=True)
    if failures or code:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
