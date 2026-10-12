"""Stop only the selected workspace queue, preserving all results/checkpoints."""
import argparse
import datetime
import json
from pathlib import Path
import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2] /
                        "runs/candidates/formal_100e_b8_s42")
    parser.add_argument("--after-current", action="store_true", help="Finish the current run before stopping the queue")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    project = args.project.resolve()
    if not project.is_relative_to(root / "runs"):
        parser.error("Queue must be within this workspace's runs directory")
    path = project / "queue.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    handoff = project / "handoff.json"
    if handoff.is_file():
        request = json.loads(handoff.read_text(encoding="utf-8"))
        if request.get("status") in ("waiting_for_current", "launching"):
            request["cancelled"] = True
            temporary_handoff = handoff.with_suffix(".json.tmp")
            temporary_handoff.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary_handoff.replace(handoff)
    (project / "STOP").touch()
    if args.after_current:
        print("STOP saved: queue will finish the current run and its evaluation, then stop.")
        return
    stopped = []
    try:
        parent = psutil.Process(state["supervisor_pid"])
        command = parent.cmdline()
        runners = {root / "code/test/run_five.py", root / "code/test/run_fusion.py"}
        if not any(Path(x).resolve() in runners for x in command if x.lower().endswith(".py")):
            raise RuntimeError("PID does not match this workspace's queue supervisor; refusing to terminate")
        index = command.index("--project") if "--project" in command else None
        actual_project = Path(command[index+1]).resolve() if index is not None else root / "runs/candidates/formal_100e_b8_s42"
        if actual_project != project:
            raise RuntimeError("PID refers to a different queue; refusing to terminate")
        parent.suspend()
        children = parent.children(recursive=True)
        for process in reversed(children):
            try:
                process.terminate()
                stopped.append(process.pid)
            except psutil.NoSuchProcess:
                pass
        parent.terminate()
        stopped.append(parent.pid)
        psutil.wait_procs(children + [parent], timeout=10)
    except psutil.NoSuchProcess:
        print("Supervisor already stopped")
    state.update(status="stopped_by_user", stopped_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 stopped_pids=stopped)
    for job in state["jobs"]:
        if job["status"] in ("training", "evaluating"):
            job["status"] = "interrupted"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(f"Stopped {len(stopped)} processes; checkpoints retained in {project}")


if __name__ == "__main__":
    main()
