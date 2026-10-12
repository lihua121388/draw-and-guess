"""Read actual queue/epoch progress without importing CUDA or training models."""
import argparse
import csv
import json
from pathlib import Path
import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2] /
                        "runs/candidates/formal_100e_b8_s42")
    args = parser.parse_args()
    state = json.loads((args.project / "queue.json").read_text(encoding="utf-8"))
    pid = state["supervisor_pid"]
    try:
        command = psutil.Process(pid).cmdline()
        alive = any(Path(s).name.lower() in {"run_five.py", "run_fusion.py"} for s in command)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        alive = False
    print(f"Queue: {state['status']}; supervisor PID={pid}, alive={alive}")
    handoff_path = args.project / "handoff.json"
    if handoff_path.is_file():
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        if handoff.get("status") == "waiting_for_current" and not handoff.get("cancelled"):
            print(f"Scheduled after {handoff['after']}: {' -> '.join(handoff['order'])}")
    for job in state["jobs"]:
        name = f"{job['experiment']}_seed{job['seed']}"
        results = args.project / name / "results.csv"
        epoch, metric = 0, ""
        if results.is_file():
            with results.open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream, skipinitialspace=True))
            if rows:
                row = {k.strip(): v.strip() for k, v in rows[-1].items()}
                epoch, metric = row["epoch"], row.get("metrics/mAP50-95(B)", "")
        print(f"{name}: {job['status']}; completed epochs={epoch}/{state['arguments']['epochs']}; "
              f"latest mAP50:95={metric}")
    if not alive and state["status"] == "running":
        print("Supervisor no longer running; saved status is stale. Inspect logs/checkpoints before resuming.")


if __name__ == "__main__":
    main()
