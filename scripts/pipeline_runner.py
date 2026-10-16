"""Generic sequential runner with resume (the earlier watchdog generalised): runs the steps of a JSON file one after another, retries a failed step, adopts an already running
instance of a step's script, commits between steps, and writes the same status file as pipeline_watchdog.py (read by watchdog_report.py).

Step fields: name; either "commit" (commit message) or "script" + "args" + "log" + "done_files" (paths relative to the project root that must exist when the step has finished;
a step whose done_files all exist before it starts is skipped) + optional "stall_min". After MAX_RETRIES failures the state becomes BLOCKED and the runner stops.
Usage: py -I -u scripts/pipeline_runner.py scripts/chains/<file>.json
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TMP = Path(r"D:\tmp_pip")
STATUS, EVENTS = TMP / "watchdog_status.json", TMP / "watchdog.log"
MAX_RETRIES, POLL_S = 3, 30
state = {"state": "STARTING", "step": "", "attempt": 0, "progress": "", "last_log_line": "", "updated": "", "recent_events": []}


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save():
    state["updated"] = now()
    STATUS.write_text(json.dumps(state, indent=1), encoding="utf-8")


def event(msg):
    line = f"{now()} {msg}"
    print(line, flush=True)
    with open(EVENTS, "a", encoding="utf-8") as f:
        f.write(line + chr(10))
    state["recent_events"] = (state["recent_events"] + [line])[-6:]
    save()


def procs(script):
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match '" + script + "' } | ForEach-Object { $_.ProcessId }"],
                         capture_output=True, text=True).stdout.split()
    return [int(x) for x in out]


def tail(path):
    try:
        lines = [l for l in Path(path).read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        return lines[-1][:200] if lines else ""
    except Exception:
        return ""


def git_commit(msg):
    subprocess.run(["git", "add", "-A"], cwd=ROOT)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0:
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT)
    return subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip() == ""


def done(st):
    return all((ROOT / f).exists() for f in st["done_files"])


def run_step(st, i, n):
    name = st["name"]
    state.update(step=f"{i}/{n} {name}", attempt=0, progress="")
    if "commit" in st:
        event(f"step {name}: committing")
        if not git_commit(st["commit"]):
            state["state"] = "BLOCKED"
            event(f"step {name}: working tree still not clean after the commit")
            return False
        return True
    if done(st):
        event(f"step {name}: outputs already exist, skipped")
        return True
    log = TMP / st.get("log", f"{name}.log")
    for attempt in range(1, MAX_RETRIES + 2):
        state.update(attempt=attempt, state="RUNNING")
        running = procs(st["script"])
        child = None
        if running:
            event(f"step {name}: adopting the running process {running}")
        else:
            event(f"step {name}: starting (attempt {attempt})")
            child = subprocess.Popen([PY, "-I", "-u", str(ROOT / "scripts" / st["script"])] + st.get("args", []), cwd=ROOT, stdout=open(log, "a", encoding="utf-8"), stderr=subprocess.STDOUT)
        last_change, last_size = time.time(), -1
        while True:
            time.sleep(POLL_S)
            alive = bool(procs(st["script"]))
            size = log.stat().st_size if log.exists() else 0
            if size != last_size:
                last_size, last_change = size, time.time()
            state["last_log_line"] = tail(log)
            save()
            if not alive:
                break
            if time.time() - last_change > st.get("stall_min", 90) * 60:
                event(f"step {name}: log unchanged for {st.get('stall_min', 90)} minutes")
                if child is not None:
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/F"], capture_output=True)
                    event(f"step {name}: killed the stalled process; will retry")
                    time.sleep(5)
                    break
                state["state"] = "BLOCKED"
                event(f"step {name}: an adopted process looks stalled; a person should look")
                return False
        if done(st):
            event(f"step {name}: finished")
            return True
        if attempt > MAX_RETRIES:
            break
        state["state"] = "RETRY"
        event(f"step {name}: ended without its outputs (attempt {attempt}); last log line: {tail(log)}; retrying in {60 * attempt} s")
        time.sleep(60 * attempt)
    state["state"] = "BLOCKED"
    event(f"step {name}: BLOCKED after {MAX_RETRIES} retries; last log line: {tail(log)}")
    return False


def main():
    steps = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    TMP.mkdir(exist_ok=True)
    event(f"runner started with {len(steps)} steps from {sys.argv[1]}")
    for i, st in enumerate(steps, 1):
        if not run_step(st, i, len(steps)):
            event("runner stopped (BLOCKED)")
            return
    state.update(state="DONE", step="all steps finished", progress="")
    event("ALL STEPS FINISHED")


if __name__ == "__main__":
    main()
