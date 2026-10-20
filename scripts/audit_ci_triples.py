"""Audit part 2: every 'difference (low to high)' triple quoted in the hand-written documents must exist in a generated report.

Hand-written documents checked: reports/research_question_answers.md, reports/test_summary.md, reports/validation_summary.md.
Generated sources: reports/test_results.md, ablation_paired.md, extensions_xgb.md, extensions_xgb_test.md, regional_daily_study_test.md and reports/tables/*.md.
A quoted triple such as '+62 (-14 to +202)' or '-486 MW, -615 to -354' is normalised (commas, signs, spaces removed) and looked up in the sources.
The result is appended to reports/numbers_audit_manual.md (read by audit_numbers.py). Usage: py scripts/audit_ci_triples.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["reports/research_question_answers.md", "reports/test_summary.md", "reports/validation_summary.md", "docs/paper_draft.md"]
SOURCES = ["reports/test_results.md", "reports/ablation_paired.md", "reports/extensions_xgb.md", "reports/extensions_xgb_test.md",
           "reports/regional_daily_study_test.md", "reports/regional_daily_study.md", "reports/lstm_stage3_test.md", "reports/statepop_weather_test.md", "reports/long_horizons_test.md", "reports/ridge_cddh_test.md", "reports/daily_multiyear.md", "reports/long_ablation_test.md", "reports/daywise_test.md", "reports/peak_events_test.md", "reports/dayahead_models_test.md", "reports/other_families_test.md", "reports/dayahead_analysis_test.md", "reports/dayahead_analysis_validation.md", "reports/long_cv_tuning_test.md", "reports/long_lstm_tuning_test.md", "reports/long_extensions_xgb_test.md", "reports/long_extensions_xgb.md"] + [str(p.relative_to(ROOT)) for p in (ROOT / "reports" / "tables").glob("*.md")]
N = r"([+-]?\d[\d,]*\.?\d*)"
PATTERNS = [
    re.compile(N + r"(?: MW| MU)?,? ?\((?:95% block-bootstrap )?(?:95% )?(?:CI )?" + N + r" to " + N),   # +62 (-14 to +202)  /  +11 MW (CI -3 to +22)
    re.compile(N + r"(?: MW| MU)?, (?:95% )?CI " + N + r" to " + N),                                      # -335 MW, CI -416 to -237
    re.compile(N + r"(?: MW| MU)?, " + N + r" to " + N),                                          # +14 MW, -110 to +105
]


def norm(x):
    x = x.replace(",", "").replace("+", "")
    return str(float(x)) if x else x


def triples(text):
    text = text.replace("−", "-")                      # typographic minus of the paper draft
    out = set()
    for pat in PATTERNS:
        for m in pat.finditer(text):
            out.add(tuple(norm(g) for g in m.groups()))
    # table rows of test_results.md: | a | b | diff | lo to hi | ...
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and re.fullmatch(N, cells[2] or "x") and re.fullmatch(N + " to " + N, cells[3] or "x"):
            lo, hi = cells[3].split(" to ")
            out.add((norm(cells[2]), norm(lo), norm(hi)))
    return out


def main():
    src = set()
    for s in SOURCES:
        p = ROOT / s
        if p.exists():
            src |= triples(p.read_text(encoding="utf-8"))
    lines = ["## Part 2: interval triples quoted in the hand-written documents versus the generated reports", ""]
    total = bad = 0
    for d in DOCS:
        t = triples((ROOT / d).read_text(encoding="utf-8"))
        miss = sorted(x for x in t if x not in src)
        total += len(t)
        bad += len(miss)
        lines.append(f"- `{d}`: {len(t)} distinct triples found, {len(t) - len(miss)} matched a generated report, {len(miss)} not matched" + (": " + "; ".join(f"{a} ({b} to {c})" for a, b, c in miss) if miss else "."))
    lines += ["", f"Total: {total - bad} of {total} matched. Unmatched triples are not necessarily wrong (they may come from a differently formatted source) and must be checked by hand.", ""]
    out = ROOT / "reports" / "numbers_audit_manual.md"
    prev = out.read_text(encoding="utf-8") if out.exists() else ""
    marker = "## Part 2:"
    if marker in prev:
        prev = prev[:prev.index(marker)]
    out.write_text(prev + chr(10).join(lines) + chr(10), encoding="utf-8")
    print(chr(10).join(lines))


if __name__ == "__main__":
    main()
