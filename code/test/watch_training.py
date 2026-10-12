"""Display progress for an already running queue without restarting training."""
import argparse
import json
import time
from pathlib import Path
import psutil
from progress_console import LogConsole


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    state_path = project / "queue.json"
    if not state_path.is_file():
        parser.error(f"Queue not found: {state_path}")
    console, active_path = None, None
    try:
        while True:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            job = next((j for j in state["jobs"] if j["status"] in ("training", "evaluating")), None)
            if job:
                name = f"{job['experiment']}_seed{job['seed']}"
                filename = f"{name}_evaluation.log" if job["status"] == "evaluating" else f"{name}.log"
                path = project / "logs" / filename
                if path != active_path and path.is_file():
                    if console:
                        console.close()
                    print(f"\nWatching {name}: {job['status']}; log={path}", flush=True)
                    if job["status"] == "training":
                        print("      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  candidate_loss  Instances  Size", flush=True)
                    console, active_path = LogConsole(path, tail_bytes=2048), path
            if console:
                console.pump()
            try:
                command = psutil.Process(state["supervisor_pid"]).cmdline()
                alive = any(Path(part).name.lower() in {"run_five.py", "run_fusion.py"} for part in command)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                alive = False
            transferring = False
            handoff_path = project / "handoff.json"
            if handoff_path.is_file():
                handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
                transferring = (handoff.get("status") in ("waiting_for_current", "launching", "running")
                                and not handoff.get("cancelled") and psutil.pid_exists(handoff.get("helper_pid", 0)))
            if (state["status"] != "running" or not alive) and not transferring:
                print(f"\nQueue status: {state['status']}; supervisor alive={alive}")
                break
            time.sleep(.2)
    except KeyboardInterrupt:
        print("\nProgress viewer stopped; training continues.")
    finally:
        if console:
            console.close()


if __name__ == "__main__":
    main()
