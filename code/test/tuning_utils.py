"""Pure selection and configuration checks for reproducible batch/worker tuning."""

import hashlib
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def continuous_batches(loader):
    """Continue over epoch boundaries without rebuilding persistent loader workers."""
    while True:
        iterator = iter(loader)
        try:
            first = next(iterator)
        except StopIteration:
            raise ValueError("Cannot benchmark an empty data loader") from None
        yield first
        yield from iterator


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(data, weights, imgsz, device, cache, amp, python):
    """Reject stale recommendations after data, code, resolution, precision or hardware changes."""
    code = ("ultralytics/nn/modules/nscr.py", "ultralytics/utils/nscr_loss.py",
            "code/test/nscr_experiments.py", "code/test/configs/yolo11n-nscr.yaml",
            "code/test/auto_tune.py", "code/test/train_test.py", "code/test/tuning_utils.py",
            "ultralytics/data/build.py", "ultralytics/data/augment.py")
    inspection = (
        "import json,os,sys,torch,yaml; from pathlib import Path; "
        f"d=yaml.safe_load(Path({str(Path(data).resolve())!r}).read_text(encoding='utf-8')); "
        "p=Path(d['path'])/d['train']; "
        "import hashlib; "
        "print(json.dumps({'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'cpus':os.cpu_count(),"
        f"'gpu':torch.cuda.get_device_name(int({str(device)!r})) if {str(device)!r}!='cpu' else 'cpu',"
        "'training_list':hashlib.sha256(p.read_bytes()).hexdigest()}))"
    )
    hardware = json.loads(subprocess.check_output([str(python), "-c", inspection], text=True).strip())
    return {"data": str(Path(data).resolve()), "data_sha256": file_hash(data),
            "weights_sha256": file_hash(weights), "imgsz": imgsz, "device": str(device),
            "cache": cache, "amp": bool(amp), "python": str(Path(python).resolve()), "runtime": hardware,
            "code": {path: file_hash(ROOT / path) for path in code}}


def rank_configurations(results, memory_fraction=0.8):
    """Rank measured configurations by median images/s; prefer fewer workers within a 3% tie."""
    grouped = {}
    rejected = {(r["batch"], r["workers"]) for r in results
                if r.get("status") != "ok" or r.get("peak_memory_fraction", 0) > memory_fraction}
    for row in results:
        if row.get("status") != "ok" or row.get("peak_memory_fraction", 0) > memory_fraction:
            continue
        if (row["batch"], row["workers"]) in rejected:
            continue
        if row.get("images_per_second", 0) <= 0:
            continue
        grouped.setdefault((row["batch"], row["workers"]), []).append(row)
    summary = []
    for (batch, workers), rows in grouped.items():
        summary.append({"batch": batch, "workers": workers,
                        "images_per_second": statistics.median(r["images_per_second"] for r in rows),
                        "peak_memory_fraction": max(r.get("peak_memory_fraction", 0) for r in rows),
                        "measurements": len(rows)})
    if not summary:
        raise RuntimeError("No successful configuration fits the memory budget; inspect candidate logs or try smaller batches")
    ranked = []
    while summary:
        fastest = max(r["images_per_second"] for r in summary)
        tied = [r for r in summary if r["images_per_second"] >= fastest * 0.97]
        chosen = min(tied, key=lambda r: (r["workers"], r["batch"]))
        ranked.append(chosen)
        summary.remove(chosen)
    return ranked


def load_profile(path, expected_signature, experiments):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("signature") != expected_signature:
        raise ValueError("Tuning profile is incomplete or no longer matches this dataset/model/runtime; tune again")
    missing = set(experiments) - set(payload.get("verified_experiments", []))
    if missing:
        raise ValueError(f"Profile has not checked these experiments: {sorted(missing)}; tune with --experiments")
    recommendation = payload["recommendation"]
    if recommendation["batch"] <= 0 or recommendation["workers"] < 0:
        raise ValueError("Invalid batch/workers in tuning profile")
    return recommendation
