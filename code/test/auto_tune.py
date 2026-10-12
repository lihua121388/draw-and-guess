"""Measure real augmented training throughput to select a safe batch/worker combination.

Candidates run in separate processes. No formal checkpoint is touched. The result is
best among tested candidates under the stated memory budget, not a global optimum.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs/nscr/.config"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/nscr/.matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

from tuning_utils import continuous_batches, rank_configurations, signature


class MemoryBudgetExceeded(RuntimeError):
    """A candidate completed a step but exceeded the configured allocation budget."""


def active_training_processes():
    """Detect repository training jobs without interrupting them."""
    import psutil

    processes = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = process.info["cmdline"] or []
            script = next((p for p in command if Path(p).name.lower() == "train_test.py"), None)
            if script and "--help" not in command and not any(mode in command for mode in ("prepare", "val", "predict", "evaluate")):
                processes.append({"pid": process.info["pid"], "script": script})
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return processes


def stop_owned_process(process):
    """Clean up only the probe and its workers on timeout or cancellation."""
    import psutil

    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    for child in reversed(descendants):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.kill()
    process.wait()


def probe(args):
    """Benchmark forward/backward, optimizer, loader wait and transfer, then dense validation."""
    import numpy as np
    import psutil
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import ModelEMA, autocast, init_seeds, select_device
    from nscr_experiments import architecture, transfer_pretrained

    init_seeds(42, deterministic=True)
    device = select_device(args.device, args.probe_batch)
    total_memory = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0
    amp = not args.no_amp and device.type == "cuda"
    cfg = get_cfg(overrides={"task": "detect", "imgsz": args.imgsz, "batch": args.probe_batch,
        "workers": args.probe_workers, "cache": False if args.cache == "none" else args.cache,
        "deterministic": True, "mosaic": 1.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0})
    data = check_det_dataset(str(args.data), autodownload=False)
    dataset = build_yolo_dataset(cfg, data["train"], args.probe_batch, data, mode="train", rect=False, stride=32)
    if len(dataset) < args.probe_batch:
        raise ValueError(f"Dataset contains fewer than {args.probe_batch} images; use smaller candidates")
    model = DetectionModel(architecture(args.reference, nc=data["nc"]), ch=data["channels"], verbose=False)
    model.names, model.args = data["names"], cfg
    transfer_pretrained(model, args.weights)
    model = model.to(device).train()
    for name, parameter in model.named_parameters():
        if ".dfl" in name:
            parameter.requires_grad_(False)
    accumulate = max(round(64 / args.probe_batch), 1)
    optimizer = DetectionTrainer.build_optimizer(None, model, name="AdamW", lr=0.001,
        momentum=0.937, decay=0.0005 * args.probe_batch * accumulate / 64, iterations=1000)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = ModelEMA(model)
    model.criterion = model.init_criterion()

    def guard_assignment(_assigner, inputs):
        # TAL creates several padded [batch, max_gt, anchors] grids, including
        # FP32 normalization buffers. Avoid driving WDDM into shared GPU memory
        # or accepting the assigner's slow CPU fallback as a valid GPU result.
        scores, _, _, _, boxes, _ = inputs
        if not total_memory:
            return
        workspace = scores.shape[0] * boxes.shape[1] * scores.shape[1] * 12 * 4
        projected = torch.cuda.memory_allocated(device) + workspace
        if projected > total_memory * args.memory_fraction:
            raise MemoryBudgetExceeded("Conservative TAL workspace estimate exceeds CUDA memory budget")

    model.criterion.assigner.register_forward_pre_hook(guard_assignment)
    optimizer.zero_grad(set_to_none=True)
    loader = build_dataloader(dataset, args.probe_batch, args.probe_workers, shuffle=True)
    iterator = continuous_batches(loader)
    warmup = max(args.warmup, accumulate + 1)
    durations, seen, max_targets, calls = [], 0, 0, 0

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def check_memory():
        if total_memory and torch.cuda.max_memory_allocated(device) > total_memory * args.memory_fraction:
            raise MemoryBudgetExceeded(f"Peak allocated CUDA memory exceeded {args.memory_fraction:.0%} budget")

    def step(batch):
        nonlocal calls, max_targets
        batch["img"] = batch["img"].to(device, non_blocking=True).float() / 255
        if len(batch["batch_idx"]):
            max_targets = max(max_targets, int(torch.bincount(batch["batch_idx"].long()).max()))
        with autocast(amp):
            loss, _ = model(batch)
            total = loss.sum()
        check_memory()
        if not torch.isfinite(total).item():
            raise RuntimeError("Nonfinite loss during throughput probe")
        scaler.scale(total).backward()
        calls += 1
        if calls % accumulate == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
        check_memory()
        return batch

    for _ in range(warmup):
        step(next(iterator))
    sync()
    print(f"Warmup finished: {warmup} steps", flush=True)
    for _ in range(args.steps):
        start = time.perf_counter()
        processed = step(next(iterator))
        sync()
        durations.append(time.perf_counter() - start)
        seen += len(processed["img"])
    print(f"Timed loop finished: {seen / sum(durations):.1f} images/s", flush=True)
    # Exercise high-label-count images, retaining the same training augmentation pipeline.
    # Multiprocess loaders mutate worker-local buffers; populate the parent's Mosaic
    # pool consistently so a cold buffer does not repeat the densest image four times.
    for index in np.random.default_rng(42).permutation(len(dataset))[:dataset.max_buffer_length]:
        dataset.load_image(int(index))
    dense_indices = sorted(range(len(dataset)), key=lambda i: len(dataset.labels[i]["cls"]), reverse=True)
    for offset in (0, args.probe_batch):
        chosen = [dense_indices[(offset + i) % len(dataset)] for i in range(args.probe_batch)]
        dense = step(dataset.collate_fn([dataset[i] for i in chosen]))
        sync()
        print(f"Dense batch checked: offset={offset}, max targets={max_targets}", flush=True)
    # The framework validates at 2x training batch and also computes detection losses.
    doubled = {"img": torch.cat((dense["img"], dense["img"])),
               "bboxes": torch.cat((dense["bboxes"], dense["bboxes"])),
               "cls": torch.cat((dense["cls"], dense["cls"])),
               "batch_idx": torch.cat((dense["batch_idx"], dense["batch_idx"] + args.probe_batch))}
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()  # framework also clears its cache before validation
    with torch.inference_mode(), autocast(amp):
        model(doubled)
    sync()
    check_memory()
    print("Double-batch validation checked", flush=True)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    result = {"status": "ok", "batch": args.probe_batch, "workers": loader.num_workers,
              "experiment": args.reference, "images_per_second": seen / sum(durations),
              "median_step_ms": float(np.median(durations) * 1000),
              "p95_step_ms": float(np.percentile(durations, 95) * 1000), "measured_images": seen,
              "warmup_steps": warmup, "measured_steps": args.steps, "effective_batch": args.probe_batch * accumulate,
              "max_augmented_targets_per_image": max_targets, "dense_and_double_batch_validation": True,
              "conservative_assignment_workspace_check": True,
              "peak_memory_GiB": peak / 2 ** 30, "peak_reserved_GiB": reserved / 2 ** 30,
              "peak_memory_fraction": peak / total_memory if total_memory else 0,
              "parent_RAM_GiB": psutil.Process().memory_info().rss / 2 ** 30,
              "cache_effective": dataset.cache, "amp_effective": amp}
    Path(args.result).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


def run_probe(args, batch, workers, experiment, number):
    directory = args.output.parent / f"{args.output.stem}_probes"
    directory.mkdir(parents=True, exist_ok=True)
    result_path, log_path = directory / f"{number:03d}.json", directory / f"{number:03d}.log"
    result_path.unlink(missing_ok=True)  # never mistake an earlier probe's result for this run
    command = [sys.executable, str(Path(__file__).resolve()), "--probe", "--data", str(args.data),
               "--weights", str(args.weights), "--imgsz", str(args.imgsz), "--device", args.device,
               "--cache", args.cache, "--reference", experiment, "--probe-batch", str(batch),
               "--probe-workers", str(workers), "--warmup", str(args.warmup), "--steps", str(args.steps),
               "--result", str(result_path), "--python", sys.executable,
               "--memory-fraction", str(args.memory_fraction)]
    if args.no_amp:
        command.append("--no-amp")
    print(f"Probe {number}: {experiment}, batch={batch}, workers={workers}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
        try:
            process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            stop_owned_process(process)
            return {"status": "timeout", "batch": batch, "workers": workers, "log": str(log_path)}
        except BaseException:
            stop_owned_process(process)
            raise
    if not result_path.exists():
        return {"status": "error", "batch": batch, "workers": workers, "log": str(log_path), "exit_code": process.returncode}
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["log"] = str(log_path)
    if result.get("status") == "ok":
        print(f"  {result['images_per_second']:.1f} images/s, peak={result['peak_memory_GiB']:.2f} GiB", flush=True)
    else:
        print(f"  {result.get('status')}: {result.get('error', '')[:160]}", flush=True)
    return result


def tune(args):
    jobs = active_training_processes()
    if jobs:
        raise RuntimeError(f"Training is still running (PIDs {[p['pid'] for p in jobs]}). Tune after it ends; no process was stopped")
    workers = sorted(set(w for w in args.workers if w <= (os.cpu_count() or 1)))
    if args.device == "cpu":
        raise ValueError("Automatic GPU tuning requires --device 0; --probe supports CPU for functional checks")
    context = signature(args.data, args.weights, args.imgsz, args.device, args.cache, not args.no_amp, sys.executable)
    reference = args.reference or ("E4" if "E4" in args.experiments else args.experiments[0])
    rows = []
    report = {"status": "in_progress", "signature": context, "reference": reference,
              "memory_fraction": args.memory_fraction, "candidate_batches": args.batches,
              "candidate_workers": workers, "results": rows, "verified_experiments": [],
              "scope": "augmented training loop incl. loader/transfer/backward/AdamW/EMA; dense train and 2x validation memory checks"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save()
    for batch in sorted(set(args.batches)):
        for worker_count in workers:
            row = run_probe(args, batch, worker_count, reference, len(rows) + 1)
            rows.append(row)
            save()
            if row.get("status") == "oom":
                break  # worker count cannot repair a GPU OOM for the same batch
    ranking = rank_configurations(rows, args.memory_fraction)
    # Repeat the leading three pairs; use the median of independent measurements.
    for candidate in ranking[:3]:
        rows.append(run_probe(args, candidate["batch"], candidate["workers"], reference, len(rows) + 1))
        save()
    ranking = rank_configurations(rows, args.memory_fraction)
    for candidate in ranking:
        verified = []
        for experiment in args.experiments:
            if experiment == reference:
                verified.append(experiment)
                continue
            row = run_probe(args, candidate["batch"], candidate["workers"], experiment, len(rows) + 1)
            rows.append(row)
            save()
            if row.get("status") != "ok" or row.get("peak_memory_fraction", 0) > args.memory_fraction:
                break
            verified.append(experiment)
        if len(verified) == len(args.experiments):
            report.update(status="complete", recommendation=candidate, verified_experiments=verified)
            save()
            print(f"Selected batch={candidate['batch']}, workers={candidate['workers']} "
                  f"({candidate['images_per_second']:.1f} images/s on {reference}); report: {args.output}", flush=True)
            return
    report["status"] = "no_common_safe_configuration"
    save()
    raise RuntimeError("No common configuration passed all requested experiments; try smaller batch candidates")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/nscr_visdrone/dataset.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "model/yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cache", choices=("none", "disk", "ram"), default="none")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--batches", nargs="+", type=int, default=[4, 8, 16, 24, 32])
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 4, 8, 12, 16])
    parser.add_argument("--experiments", nargs="+", choices=[f"E{i}" for i in range(7)], default=[f"E{i}" for i in range(7)])
    parser.add_argument("--reference", choices=[f"E{i}" for i in range(7)])
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/nscr/tuning.json")
    parser.add_argument("--python", type=Path, help="Override the preferred Anaconda yolo interpreter")
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe-batch", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--probe-workers", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    from run_formal import resolve_python

    executable = resolve_python(args.python)
    if Path(sys.executable).resolve() != executable:
        raise SystemExit(subprocess.call([str(executable), str(Path(__file__).resolve()), *sys.argv[1:]]))
    if args.imgsz < 32 or args.imgsz % 32 or args.steps < 1 or args.warmup < 0 or args.timeout < 1:
        parser.error("imgsz must be a multiple of 32; steps/timeout positive; warmup nonnegative")
    if any(b < 1 for b in args.batches) or any(w < 0 for w in args.workers) or not 0 < args.memory_fraction < 1:
        parser.error("batch candidates positive, workers nonnegative, memory-fraction between 0 and 1")
    if args.reference is None and args.probe:
        args.reference = "E4"
    if args.device != "cpu" and not args.device.isdigit():
        parser.error("Use one numeric CUDA device, e.g. --device 0")
    if args.probe:
        if args.result is None:
            parser.error("--probe requires --result")
        if args.probe_batch < 1 or args.probe_workers < 0:
            parser.error("probe batch must be positive and workers nonnegative")
        try:
            probe(args)
        except Exception as error:
            status = "memory_limit" if isinstance(error, MemoryBudgetExceeded) else (
                "oom" if "out of memory" in str(error).lower() else "error")
            result = {"status": status,
                      "batch": args.probe_batch, "workers": args.probe_workers, "error": str(error)}
            args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
            raise
    else:
        tune(args)


if __name__ == "__main__":
    main()
