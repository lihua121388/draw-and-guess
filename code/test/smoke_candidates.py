"""Real batch-eight training/evaluation integration check on a disjoint manifest."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from prepare_data import NAMES, ROOT, write_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Audit an already finished smoke queue")
    args = parser.parse_args()
    project = ROOT / "runs/candidates/smoke_b8"
    if args.verify_only:
        verify(project)
        return
    output = ROOT / "datasets/nscr_visdrone"
    splits = {split: [Path(p) for p in (output / f"{split}.txt").read_text().splitlines() if p]
              for split in ("train", "val", "test")}
    data = write_splits(output, NAMES, splits, train_limit=16, val_limit=8, seed=42)
    log = ROOT / "runs/candidates/smoke_queue.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(ROOT / "code/test/run_five.py"),
               "--data", str(data), "--project", str(log.parent / "smoke_b8"),
               "--epochs", "1", "--batch", "8", "--workers", "4", "--no-plots"]
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    print(f"Smoke queue return code: {result.returncode}")
    print(log.read_text(encoding="utf-8")[-6000:])
    if result.returncode:
        raise SystemExit(result.returncode)
    verify(project)


def verify(project):
    state = json.loads((project / "queue.json").read_text(encoding="utf-8"))
    if state["status"] != "complete" or len(state["jobs"]) != 6:
        raise AssertionError("Six-model smoke queue did not complete")
    checks = []
    for job in state["jobs"]:
        if job["status"] != "complete" or job.get("training_return_code") != 0 or job.get("evaluation_return_code") != 0:
            raise AssertionError(f"Failed job: {job}")
        run = project / f"{job['experiment']}_seed{job['seed']}"
        summary = json.loads((run / "summary.json").read_text())
        manifest = json.loads((run / "experiment.json").read_text())
        evaluation = json.loads((run / "evaluation/evaluation.json").read_text())
        if summary["epochs"] != 1 or manifest["arguments"]["batch"] != 8:
            raise AssertionError("Wrong smoke training settings")
        if not all(math.isfinite(v) for k, v in summary.items() if k.startswith("metrics/") and isinstance(v, (float, int))):
            raise AssertionError("Nonfinite detection metric")
        if not (run / "weights/best.pt").is_file() or not (run / "weights/last.pt").is_file():
            raise AssertionError("Missing saved checkpoint")
        diagnostic = evaluation["candidate_diagnostics"]
        if job["experiment"] != "M0" and (diagnostic["status"] != "evaluated" or diagnostic["images"] != 8):
            raise AssertionError("Held-out teacher diagnostics did not evaluate eight images")
        checks.append(dict(experiment=job["experiment"], status="passed", epochs=1, batch=8,
                           params=summary["params"], GFLOPs=summary["GFLOPs"]))
    if not (project / "review/review.md").is_file():
        raise AssertionError("Missing matching-control report")
    report = dict(status="passed", train_images=16, val_images=8, jobs=checks,
                  limitation="One-epoch integration check, not evidence of improved detection accuracy.")
    (project / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
