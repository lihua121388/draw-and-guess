"""Finish the current job, then continue the same queue in a new order."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
import psutil
from run_five import now, write_json, queue_process_alive
from run_formal import ROOT, resolve_python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--after", default="M3")
    parser.add_argument("--experiments", nargs="+", required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    if not project.is_relative_to(ROOT / "runs"):
        parser.error("Queue must be in this workspace's runs directory")
    queue = project / "queue.json"
    state = json.loads(queue.read_text(encoding="utf-8"))
    current = [j for j in state["jobs"] if j["status"] in ("training", "evaluating")]
    if len(current) != 1 or current[0]["experiment"] != args.after:
        parser.error(f"Expected exactly one current {args.after} job")
    if not queue_process_alive(state["supervisor_pid"]):
        parser.error("Current supervisor is not running")
    old_order = state["arguments"]["experiments"]
    if len(args.experiments) != len(set(args.experiments)) or set(args.experiments) != set(old_order):
        parser.error("New order must contain every original experiment exactly once")
    finished_or_current = {j["experiment"] for j in state["jobs"] if j["status"] != "pending"}
    prefix = args.experiments[:args.experiments.index(args.after)+1]
    if set(prefix) != finished_or_current:
        parser.error("Completed/current jobs must precede the remaining jobs")
    path = project / "handoff.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("status") in ("waiting_for_current", "launching", "running"):
            parser.error("A handoff is already scheduled")
    stop = project / "STOP"
    if stop.exists():
        parser.error("A stop was already requested; refusing to override it")
    record = dict(status="waiting_for_current", helper_pid=os.getpid(), previous_supervisor_pid=state["supervisor_pid"],
                  requested_at=now(), after=args.after, order=args.experiments, arguments=state["arguments"])
    write_json(path, record)
    stop.write_text(f"handoff:{os.getpid()}", encoding="utf-8")
    print(f"Handoff scheduled after {args.after}: {args.experiments}", flush=True)
    previous_pid = state["supervisor_pid"]
    while queue_process_alive(previous_pid):
        latest = json.loads(path.read_text(encoding="utf-8"))
        if latest.get("cancelled"):
            record.update(status="cancelled", finished_at=now())
            write_json(path, record)
            return
        time.sleep(1)
    state = json.loads(queue.read_text(encoding="utf-8"))
    current = [j for j in state["jobs"] if j["experiment"] == args.after]
    if state["status"] != "stopped_before_next_run" or not all(j["status"] == "complete" for j in current):
        record.update(status="handoff_failed", reason="Current job did not complete with the expected safe stop", finished_at=now())
        write_json(path, record)
        raise SystemExit(1)
    latest = json.loads(path.read_text(encoding="utf-8"))
    if latest.get("cancelled"):
        record.update(status="cancelled", finished_at=now())
        write_json(path, record)
        return
    if not stop.exists() or stop.read_text(encoding="utf-8") != f"handoff:{os.getpid()}":
        raise RuntimeError("Handoff STOP marker changed; refusing to resume")
    saved = record["arguments"]
    python = resolve_python(saved.get("python"))
    command = [str(python), "-u", str(ROOT / "code/test/run_five.py"), "--resume-queue",
               "--project", str(project), "--experiments", *args.experiments, "--seeds", *map(str, saved["seeds"])]
    for key in ("data", "weights", "epochs", "batch", "workers", "imgsz", "device", "aux_weight", "cache"):
        command.extend(["--"+key.replace("_", "-"), str(saved[key])])
    for key in ("no_amp", "no_plots", "no_evaluate"):
        if saved[key]:
            command.append("--"+key.replace("_", "-"))
    # This removes only this helper's marker; no checkpoint or training log is deleted.
    stop.unlink()
    record.update(status="launching", command=command)
    write_json(path, record)
    child = subprocess.Popen(command, cwd=str(ROOT))
    record.update(status="running", new_supervisor_pid=child.pid, resumed_at=now())
    write_json(path, record)
    code = child.wait()
    record.update(status="complete" if code == 0 else "queue_failed", return_code=code, finished_at=now())
    write_json(path, record)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
