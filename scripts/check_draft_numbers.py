"""Checks that every number in docs/paper_draft.md also appears in a generated report, a result log or the audited answers document (reports/research_question_answers.md).
Presence check only (a number can match by coincidence); the unmatched numbers are listed for manual review. Digits inside DOIs, URLs, references and years are skipped.
Usage: py -X utf8 -I scripts/check_draft_numbers.py [path-to-draft]
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "paper_draft.md"
NUM = re.compile(r"(?<![\w.])[+-]?\d[\d,]*\.?\d*")
EXCLUDE = {"test_summary.md", "validation_summary.md", "numbers_audit.md", "numbers_audit_manual.md"}   # answers document stays in as an audited source


def val(tok):
    t = tok.replace(",", "").lstrip("+-").rstrip(".")
    try:
        return float(t)
    except ValueError:
        return None


src = set()
for p in (ROOT / "reports").rglob("*.md"):
    if p.name in EXCLUDE:
        continue
    for m in NUM.finditer(p.read_text(encoding="utf-8", errors="replace")):
        v = val(m.group())
        if v is not None:
            src.add(v)
            src.add(round(v, 0))


def walk(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield float(o)


for p in (ROOT / "reports" / "tables").glob("*.json*"):
    txt = p.read_text(encoding="utf-8", errors="replace")
    for line in txt.splitlines() if p.suffix == ".jsonl" else [txt]:
        try:
            for f in walk(json.loads(line)):
                for d in (0, 1, 2, 3):
                    src.add(round(f, d))
                    src.add(round(abs(f), d))
        except Exception:
            pass

lines = DRAFT.read_text(encoding="utf-8").splitlines()
stop = next((i for i, l in enumerate(lines) if l.startswith("## References")), len(lines))
bad = []
for ln, line in enumerate(lines[:stop], 1):
    clean = re.sub(r"`[^`]*`", " ", line)                 # file names in backticks
    clean = re.sub(r"\[verify[^\]]*\]|\bDOI\s*\S+|https?://\S+", " ", clean)
    for m in NUM.finditer(clean):
        tok = m.group()
        v = val(tok)
        digits = re.sub(r"\D", "", tok)
        if v is None or (len(digits) < 3 and "." not in tok) or re.fullmatch(r"(19|20)\d\d", tok.strip("+-")):
            continue
        if v in src or round(v, 0) in src or round(v, 1) in src:
            continue
        bad.append((ln, tok, line.strip()[max(0, m.start() - 60):m.end() + 40]))
print(f"{len(bad)} numbers of the draft not found in the generated reports, logs or the audited answers document")
for ln, tok, ctx in bad:
    print(f"  L{ln} [{tok}] ...{ctx}")
