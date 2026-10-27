r"""Syntax-level checks of the LaTeX sources in paper/ (no TeX engine needed): expands \input, parses the whole document with pylatexenc (strict mode), checks that begin/end pairs balance,
that every \cite key exists in paper/references.bib, that every \includegraphics file exists and that every \ref has a \label.
This does NOT replace compiling the documents (Overleaf or a local TeX installation); it catches the typical conversion errors.
Needs: pip install pylatexenc.   Usage: py -I -X utf8 scripts/check_paper_tex.py
"""
import re
import sys
from pathlib import Path

from pylatexenc.latexwalker import LatexWalker, LatexWalkerParseError

PAPER = Path(__file__).resolve().parents[1] / "paper"


def expand(path):
    text = path.read_text(encoding="utf-8")
    return re.sub(r"\\input\{([^}]+)\}", lambda m: expand((path.parent / (m.group(1) + ".tex")).resolve()), text)


problems = 0
bib = (PAPER / "references.bib").read_text(encoding="utf-8")
keys = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.M))
for variant in ("ieee", "springer"):
    full = expand(PAPER / variant / "main.tex")
    try:
        parse_text = "\n".join(ln for ln in full.splitlines() if not ln.lstrip().startswith("\\newenvironment"))   # float wrappers: valid LaTeX that the parser rejects
        nodes, _, _ = LatexWalker(parse_text, tolerant_parsing=False).get_latex_nodes()
        parsed = f"parsed {len(nodes)} top-level nodes"
    except LatexWalkerParseError as e:
        parsed = "PARSE ERROR: " + str(e)[:150]
        problems += 1
    envs, ends = sorted(re.findall(r"\\begin\{(\w+\*?)\}", full)), sorted(re.findall(r"\\end\{(\w+\*?)\}", full))
    cites = set(k.strip() for g in re.findall(r"\\cite\{([^}]*)\}", full) for k in g.split(","))
    graphics = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", full)
    labels, refs = set(re.findall(r"\\label\{([^}]*)\}", full)), set(re.findall(r"\\ref\{([^}]*)\}", full))
    checks = {"begin/end balanced": envs == ends, "all cite keys in the bib": cites <= keys, "all figure files exist": all((PAPER / "figures" / (g + ".png")).exists() for g in graphics),
              "all refs have labels": refs <= labels, "no leftover markdown or unicode": not re.search(r"[\u2212\u00d7]|(?<!`)`(?!`)|\*\*|\[@|\[#[A-Za-z]", full)}
    problems += sum(not v for v in checks.values())
    print(f"{variant}: {parsed}; {len(full.splitlines())} lines; {len(cites)} citations, {len(graphics)} figures, {len(labels)} labels")
    for k, v in checks.items():
        print(f"   {'ok ' if v else 'FAIL'} {k}")
print("PROBLEMS:", problems)
sys.exit(1 if problems else 0)
