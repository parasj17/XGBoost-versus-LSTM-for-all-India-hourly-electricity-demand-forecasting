"""Prints one status line every 15 minutes, and immediately when the supervisor changes step or reaches RETRY / BLOCKED / DONE (read by a Monitor).
Usage: py -I -u scripts/watchdog_report.py
"""
import json
import time
from pathlib import Path

STATUS = Path(r"D:\tmp_pip\watchdog_status.json")
PERIOD_S = 900


def line(s):
    p = s.get("progress", "")
    ev = s.get("recent_events", [])
    return f"[{s.get('updated', '?')}] state={s.get('state')} step={s.get('step')} attempt={s.get('attempt')} | {p} | last log: {s.get('last_log_line', '')[:110]}" + (f" | latest event: {ev[-1][20:130]}" if ev else "")


def main():
    last_print, last_key = 0.0, None
    while True:
        try:
            s = json.loads(STATUS.read_text(encoding="utf-8"))
        except Exception:
            s = None
        if s is None:
            if time.time() - last_print > PERIOD_S:
                print("watchdog status file not found or unreadable", flush=True)
                last_print = time.time()
        else:
            key = (s.get("state"), s.get("step"), s.get("attempt"))
            urgent = key != last_key and (s.get("state") in ("RETRY", "BLOCKED", "DONE") or last_key is not None and key[1] != last_key[1])
            if urgent or time.time() - last_print >= PERIOD_S:
                print(line(s), flush=True)
                last_print = time.time()
            last_key = key
        time.sleep(20)


if __name__ == "__main__":
    main()
