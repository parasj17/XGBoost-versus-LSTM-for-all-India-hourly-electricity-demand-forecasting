"""Supervisor for the long-running pipeline (owner request 2026-10-16): keeps each step running, resumes it after a crash or stall, and chains the declared steps.

Steps (each resumable or single-use as noted; all rules are in docs/experiment_protocol.md):
  1 stage3        run_lstm_stage3.py       equal-budget LSTM tuning (resumable). An already-running instance is adopted, not duplicated.
  2 commit_stage3 git commit of the stage-3 outputs
  3 stage3_check  run_test_lstm_stage3.py --check   dry run on VALIDATION data (no test data)
  4 stage3_test   run_test_lstm_stage3.py           the single test evaluation of the equal-budget LSTM (lock file)
  5 commit_test   git commit
  6 walk_forward  run_walk_forward.py     walk-forward evaluation (resumable)
  7 commit_wf     git commit
A failed step is retried (with growing pauses) up to MAX_RETRIES times; a step whose log has not changed for STALL_MIN minutes is killed (only if this script started it) and retried.
After MAX_RETRIES the state becomes BLOCKED and the supervisor stops so that a person can look. Status: D:/tmp_pip/watchdog_status.json, events: D:/tmp_pip/watchdog.log.
Usage: py -I -u scripts/pipeline_watchdog.py
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
T = ROOT / "reports" / "tables"
MAX_RETRIES, POLL_S = 6, 30
state = {"state": "STARTING", "step": "", "attempt": 0, "progress": "", "last_log_line": "", "updated": "", "recent_events": []}


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def event(msg):
    line = f"{now()} {msg}"
    print(line, flush=True)
    with open(EVENTS, "a", encoding="utf-8") as f:
        f.write(line + chr(10))
    state["recent_events"] = (state["recent_events"] + [line])[-6:]
    save()


def save():
    state["updated"] = now()
    STATUS.write_text(json.dumps(state, indent=1), encoding="utf-8")


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []


def procs(script):
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match '" + script + "' } | ForEach-Object { $_.ProcessId }"],
                         capture_output=True, text=True).stdout.split()
    return [int(x) for x in out]


def tail(path, n=1):
    try:
        lines = [l for l in Path(path).read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        return lines[-1][:200] if lines else ""
    except Exception:
        return ""


def git_commit(msg):
    subprocess.run(["git", "add", "-A"], cwd=ROOT)
    r = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    if r.returncode != 0:
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT)
    return subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip() == ""


# ---------------------------------------------------------------- step definitions
def stage3_done():
    p = T / "lstm_stage3_config.json"
    return p.exists() and set(json.loads(p.read_text(encoding="utf-8"))) == {"1", "6", "24", "48"} and (ROOT / "reports" / "lstm_stage3.md").exists()


def stage3_progress():
    rows = [r for r in jl(T / "lstm_stage3.jsonl") if r.get("stage") == "tuning_trial"]
    fin = [r for r in jl(T / "lstm_stage3.jsonl") if r.get("stage") == "final_fit"]
    per = {h: sum(1 for r in rows if r["horizon"] == h) for h in (1, 6, 24, 48)}
    return "new trials " + ", ".join(f"{h} h: {n}/78" for h, n in per.items() if n or h == 1) + f"; final fits {len(fin)}/12"


def wf_done():
    return (ROOT / "reports" / "walk_forward.md").exists() and len({(r["window"], r["horizon"]) for r in jl(T / "walk_forward.jsonl")}) >= 24


def wf_progress():
    n = len({(r["window"], r["horizon"]) for r in jl(T / "walk_forward.jsonl")})
    return f"walk-forward (window, horizon) cells done {n}/24"


STEPS = [
    {"name": "stage3", "script": "run_lstm_stage3.py", "args": [], "log": TMP / "lstm_stage3.log", "done": stage3_done, "progress": stage3_progress, "stall_min": 60},
    {"name": "commit_stage3", "commit": "LSTM stage 3 (equal tuning budget): tuning results and selected configuration"},
    {"name": "stage3_check", "script": "run_test_lstm_stage3.py", "args": ["--check"], "log": TMP / "stage3_check.log",
     "done": lambda: (ROOT / "reports" / "lstm_stage3_test_check.md").exists(), "progress": lambda: "dry run on validation", "stall_min": 30},
    {"name": "stage3_test", "script": "run_test_lstm_stage3.py", "args": [], "log": TMP / "stage3_test.log", "done": lambda: (T / "LSTM_STAGE3_TEST_DONE.lock").exists(),
     "progress": lambda: "single test evaluation of the equal-budget LSTM", "stall_min": 60, "before": "remove_check"},
    {"name": "commit_test", "commit": "Equal-budget LSTM: single test evaluation results"},
    {"name": "walk_forward", "script": "run_walk_forward.py", "args": [], "log": TMP / "walk_forward.log", "done": wf_done, "progress": wf_progress, "stall_min": 60},
    {"name": "commit_wf", "commit": "Walk-forward evaluation results"},
]


def run_step(st):
    name = st["name"]
    state.update(step=name, attempt=0)
    if "commit" in st:
        state["progress"] = "committing"
        event(f"step {name}: committing")
        if not git_commit(st["commit"]):
            state.update(state="BLOCKED")
            event(f"step {name}: working tree still not clean after the commit")
            return False
        return True
    if st.get("before") == "remove_check":
        p = ROOT / "reports" / "lstm_stage3_test_check.md"
        if p.exists():
            p.unlink()
        if not stage3_done():
            state.update(state="BLOCKED")
            event("step stage3_test: the stage-3 configuration is incomplete; not running the single-use test evaluation")
            return False
        if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip():
            state.update(state="BLOCKED")
            event("step stage3_test: working tree is not clean; the test script would refuse to run")
            return False
    for attempt in range(1, MAX_RETRIES + 2):
        if st["done"]() and st["name"] != "stage3_check":
            event(f"step {name}: already complete")
            return True
        state.update(attempt=attempt, state="RUNNING")
        running = procs(st["script"])
        child = None
        if running:
            event(f"step {name}: adopting the running process {running}")
        else:
            event(f"step {name}: starting (attempt {attempt})")
            child = subprocess.Popen([PY, "-I", "-u", str(ROOT / "scripts" / st["script"])] + st["args"], cwd=ROOT,
                                     stdout=open(st["log"], "a", encoding="utf-8"), stderr=subprocess.STDOUT)
        last_change, last_size = time.time(), -1
        while True:
            time.sleep(POLL_S)
            alive = bool(procs(st["script"]))
            size = st["log"].stat().st_size if st["log"].exists() else 0
            if size != last_size:
                last_size, last_change = size, time.time()
            state["progress"] = st["progress"]()
            state["last_log_line"] = tail(st["log"])
            save()
            if not alive:
                break
            if time.time() - last_change > st["stall_min"] * 60:
                event(f"step {name}: log unchanged for {st['stall_min']} minutes")
                if child is not None:
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/F"], capture_output=True)
                    event(f"step {name}: killed the stalled process {child.pid}; will resume")
                    time.sleep(5)
                    break
                state.update(state="BLOCKED")
                event(f"step {name}: an adopted process looks stalled; not killing it, a person should look")
                return False
        if st["done"]():
            event(f"step {name}: finished")
            return True
        if attempt > MAX_RETRIES:
            break
        pause = 60 * attempt
        state.update(state="RETRY")
        event(f"step {name}: ended without finishing (attempt {attempt}); last log line: {tail(st['log'])}; resuming in {pause} s")
        time.sleep(pause)
    state.update(state="BLOCKED")
    event(f"step {name}: BLOCKED after {MAX_RETRIES} retries; last log line: {tail(st['log'])}")
    return False


def main():
    TMP.mkdir(exist_ok=True)
    event("watchdog started")
    for st in STEPS:
        if not run_step(st):
            event("watchdog stopped (BLOCKED)")
            return
    state.update(state="DONE", step="all steps finished", progress="")
    event("ALL STEPS FINISHED")


if __name__ == "__main__":
    main()
