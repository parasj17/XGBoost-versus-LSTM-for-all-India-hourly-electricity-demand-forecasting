"""Numbers audit: compare the figures in the hand-written summary documents with the result files.

Ground truth comes from the logged results, never from the documents: reports/tables/*.jsonl (test metrics, validation fits), the generated reports
(baselines / xgboost / lstm validation markdown) and the saved predictions in data/interim/predictions (paired differences are recomputed).
Checked here: tables of README.md, reports/test_summary.md and reports/validation_summary.md. Text claims in the answers document are checked by hand and listed
in reports/numbers_audit.md (written by this script, which appends the hand-check list from reports/numbers_audit_manual.md if that file exists).
Usage: py scripts/audit_numbers.py
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "reports" / "tables"
HZ = [1, 6, 24, 48]
results, nfail = [], 0


def num(cell):
    m = re.search(r"[-+]?\d[\d,]*\.?\d*", cell.replace("**", ""))
    return float(m.group(0).replace(",", "")) if m else None


def rows(text):
    out = []
    for line in text.splitlines():
        if line.startswith("|") and not set(line.replace("|", "").strip()) <= set("-: "):
            out.append([c.strip() for c in line.strip().strip("|").split("|")])
    return out


def find(rowsl, prefix, nth=0):
    hits = [r for r in rowsl if r[0].replace("**", "").startswith(prefix)]
    return hits[nth] if len(hits) > nth else None


def check(doc, what, doc_val, truth, tol=0.51):
    global nfail
    ok = doc_val is not None and abs(doc_val - truth) <= tol
    if not ok:
        nfail += 1
    results.append((doc, what, doc_val, truth, ok))


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def jl(name):
    return [json.loads(l) for l in open(T / name, encoding="utf-8")]


# ------------------------------------------------------------------ ground truth
test = {r["horizon"]: r["metrics"] for r in jl("test_results.jsonl")}
T_MAE = {h: {m: v["MAE_MW"] for m, v in test[h].items()} for h in HZ}
T_MAE_naive = {h: min(T_MAE[h][k] for k in ("persistence", "snaive24", "snaive168")) for h in HZ}


def md_sections(path):
    text = read(path)
    secs = re.split(r"^#{2,3} Horizon (\d+) h", text, flags=re.M)
    return {int(secs[i]): secs[i + 1] for i in range(1, len(secs), 2)}


base = md_sections("reports/tables/baselines_validation.md")
V = {"ridge": {}, "naive": {}, "xgb1": {}, "xgb2": {}, "l4": {}, "l1_s1": {}, "l1_s2": {}}
for h in HZ:
    rr = rows(base[h])
    V["ridge"][h] = min(num(r[1]) for r in rr if r[0].startswith("ridge_") and len(r) >= 6)
    V["naive"][h] = min(num(r[1]) for r in rr if r[0] in ("persistence", "snaive24", "snaive168"))
for key, path in (("xgb1", "reports/tables/xgboost_validation.md"), ("xgb2", "reports/tables/xgboost_validation_wide.md")):
    for h, sec in md_sections(path).items():
        V[key][h] = float(re.search(r"Headline \(S4_full_A, ratio target, mean of 3 seeds\): MAE ([\d,]+) MW", sec).group(1).replace(",", ""))
for key, path, tag in (("l4", "reports/tables/lstm_validation.md", "L4_full_A"), ("l1_s1", "reports/tables/lstm_validation.md", "L1_history_calendar"),
                       ("l1_s2", "reports/tables/lstm_validation_stage2_L1.md", "L1_history_calendar")):
    for h, sec in md_sections(path).items():
        r = find(rows(sec), tag)
        V[key][h] = num(r[2])

# ------------------------------------------------------------------ README
readme = rows(read("README.md"))
for label, key in (("Best naive", None), ("Ridge", "ridge_best"), ("XGBoost (declared headline)", "xgb_S4_full_A"), ("LSTM full features (declared headline)", "lstm_L4_full_A")):
    r = find(readme, label)
    for i, h in enumerate(HZ):
        check("README results table", f"{label} {h} h", num(r[1 + i]), T_MAE_naive[h] if key is None else T_MAE[h][key])

# ------------------------------------------------------------------ test_summary
ts_text = read("reports/test_summary.md")
ts = rows(ts_text)
for label, key in (("Best naive", None), ("Ridge", "ridge_best"), ("XGBoost S4_full_A (declared headline)", "xgb_S4_full_A"),
                   ("LSTM L4_full_A (declared headline)", "lstm_L4_full_A"), ("LSTM L1 history+calendar (own tuning)", "lstm_L1_history_calendar")):
    r = find(ts, label)
    for i, h in enumerate(HZ):
        check("test_summary headline", f"{label} {h} h", num(r[1 + i]), T_MAE_naive[h] if key is None else T_MAE[h][key])
# validation -> test table
vt = {"Ridge": ("ridge", "ridge_best"), "XGBoost headline": ("xgb2", "xgb_S4_full_A"), "LSTM L4": ("l4", "lstm_L4_full_A"), "LSTM L1 (own tuning)": ("l1_s2", "lstm_L1_history_calendar")}
for label, (vk, tk) in vt.items():
    r = [x for x in ts if x[0] == label and "->" in x[1]][0]
    for i, h in enumerate(HZ):
        a, b = [float(x.replace(",", "")) for x in re.findall(r"[\d,]+", r[1 + i])]
        check("test_summary validation->test", f"{label} {h} h validation", a, V[vk][h])
        check("test_summary validation->test", f"{label} {h} h test", b, T_MAE[h][tk])
# paired differences: means recomputed from saved predictions
pairs = {"XGBoost - ridge": ("xgb_S4_full_A", "ridge_best"), "LSTM L4 - XGBoost": ("lstm_L4_full_A", "xgb_S4_full_A"),
         "LSTM L1 - XGBoost": ("lstm_L1_history_calendar", "xgb_S4_full_A"), "LSTM L1 - ridge": ("lstm_L1_history_calendar", "ridge_best")}
tr_md = {h: rows(sec) for h, sec in md_sections("reports/test_results.md").items()}
for label, (a, b) in pairs.items():
    r = find(ts, label)
    for i, h in enumerate(HZ):
        z = np.load(ROOT / "data" / "interim" / "predictions" / f"h{h}.npz")
        d = np.mean(np.abs(z[f"te__{a}"] - z["y_te"]) - np.abs(z[f"te__{b}"] - z["y_te"]))
        check("test_summary paired difference (recomputed from predictions)", f"{label} {h} h mean", num(r[1 + i]), d)
        row = [x for x in tr_md[h] if len(x) >= 4 and x[0] == a and x[1] == b][0]
        cell = r[1 + i]
        ci = re.search(r"\(([^)]*)\)", cell).group(1)
        ok = ci == row[3]
        results.append(("test_summary paired CI vs reports/test_results.md", f"{label} {h} h CI", ci, row[3], ok))
        if not ok:
            nfail += 1
# skill ranges quoted in prose: parse the sentence in the document and compare with values recomputed from the test metrics
skill = {h: {m: 100 * (1 - T_MAE[h][m] / T_MAE_naive[h]) for m in ("ridge_best", "xgb_S4_full_A", "lstm_L4_full_A", "lstm_L1_history_calendar")} for h in HZ}
sent = re.search(r"Skill against the best naive baseline[^\n]*", ts_text).group(0)
quoted = [(int(a), int(b)) for a, b in re.findall(r"(\d+)-(\d+)%", sent)]
rx_ = [round(skill[1]["ridge_best"]), round(skill[1]["xgb_S4_full_A"])]
lm_ = [round(skill[1]["lstm_L4_full_A"]), round(skill[1]["lstm_L1_history_calendar"])]
expect = [(min(rx_), max(rx_)), (min(lm_), max(lm_))]
for h in (6, 24, 48):
    expect.append((round(min(skill[h].values())), round(max(skill[h].values()))))
for q, e, lab in zip(quoted, expect, ("1 h ridge/XGBoost", "1 h LSTMs", "6 h", "24 h", "48 h")):
    ok = q == e
    results.append(("test_summary skill ranges", f"{lab}: quoted {q[0]}-{q[1]}% vs recomputed {e[0]}-{e[1]}%", None, None, ok))
    if not ok:
        nfail += 1
if len(quoted) != 5:
    nfail += 1
    results.append(("test_summary skill ranges", f"expected 5 ranges, parsed {len(quoted)}", None, None, False))


# ------------------------------------------------------------------ validation_summary
vs = rows(read("reports/validation_summary.md"))
first = lambda label, k: [r for r in vs if r[0].replace("**", "").startswith(label)][k]
for k, (label, key) in enumerate((("Best naive", "naive"), ("Best ridge", "ridge"), ("XGBoost, full features", "xgb1"), ("LSTM, full features", "l4"), ("LSTM, history + calendar", "l1_s1"))):
    r = first(label, 0)
    for i, h in enumerate(HZ):
        check("validation_summary table 1", f"{label} {h} h", num(r[1 + i]), V[key][h])
for label, key, nth in (("Best ridge", "ridge", 1), ("XGBoost stage 1", "xgb1", 0), ("XGBoost stage 2", "xgb2", 0), ("LSTM full features (stage-1", "l4", 0),
                        ("LSTM history+calendar, stage 1", "l1_s1", 0), ("LSTM history+calendar, own tuning", "l1_s2", 0)):
    r = first(label, nth)
    for i, h in enumerate(HZ):
        check("validation_summary stage-2 table", f"{label} {h} h", num(r[1 + i]), V[key][h])

# ------------------------------------------------------------------ equal-budget LSTM (stage 3)
s3 = {}
for line in read('reports/lstm_stage3_test.md').splitlines():
    cells = [c.strip() for c in line.strip().strip('|').split('|')]
    if len(cells) >= 7 and cells[0].isdigit():
        s3[int(cells[0])] = cells
def exact(rowsl, label):
    return [r for r in rowsl if r[0].replace('**', '') == label][0]
r = find(readme, 'LSTM full features, equal tuning budget')
for i, h in enumerate(HZ):
    check('README results table', f'LSTM equal budget {h} h', num(r[1 + i]), num(s3[h][1]))
for label, col in (('LSTM L4, equal tuning budget', 1), ('LSTM L4, original (12 trials)', 2), ('XGBoost S4', 3)):
    r = exact(ts, label)
    for i, h in enumerate(HZ):
        check('test_summary equal-budget table', f'{label} {h} h', num(r[1 + i]), num(s3[h][col]))
        if label == 'LSTM L4, original (12 trials)':
            check('equal-budget table vs frozen test run', f'original LSTM {h} h', num(s3[h][2]), T_MAE[h]['lstm_L4_full_A'])
        if label == 'XGBoost S4':
            check('equal-budget table vs frozen test run', f'XGBoost {h} h', num(s3[h][3]), T_MAE[h]['xgb_S4_full_A'])
for label, col in (('Equal-budget minus original (95% CI)', 5), ('Equal-budget minus XGBoost (95% CI)', 6)):
    r = exact(ts, label)
    for i, h in enumerate(HZ):
        ok = r[1 + i].replace(' ', '') == s3[h][col].replace(' ', '')
        results.append(('test_summary equal-budget paired cells', f'{label} {h} h: {r[1 + i]} vs {s3[h][col]}', None, None, ok))
        if not ok:
            nfail += 1

# ------------------------------------------------------------------ report
passed = sum(1 for r in results if r[4])
lines = ["# Numbers audit (generated by `scripts/audit_numbers.py`)", "",
         f"Tables in README.md, reports/test_summary.md and reports/validation_summary.md were compared with values recomputed from the result files "
         f"(test metrics in `reports/tables/test_results.jsonl`, validation numbers in the generated validation reports, paired differences recomputed from the saved predictions). "
         f"**{passed} of {len(results)} checks passed, {nfail} failed.**", ""]
if nfail:
    lines += ["## Failed checks", "", "| document | item | in document | recomputed |", "|---|---|---:|---:|"]
    for d, w, a, b, ok in results:
        if not ok:
            lines.append(f"| {d} | {w} | {a} | {b} |")
    lines.append("")
manual = ROOT / "reports" / "numbers_audit_manual.md"
if manual.exists():
    lines += [manual.read_text(encoding="utf-8")]
(ROOT / "reports" / "numbers_audit.md").write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
print(f"{passed}/{len(results)} passed; {nfail} failed")
for d, w, a, b, ok in results:
    if not ok:
        print("FAIL", d, "|", w, "| doc:", a, "| truth:", b)


if __name__ == "__main__":
    pass
