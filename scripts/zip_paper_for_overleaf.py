"""Packs the contents of paper/ into paper_overleaf.zip (repository root, git-ignored) for upload to Overleaf: New Project > Upload Project.
The folder structure inside the zip is ieee/, ieee-conference/, springer/, figures/, body.tex, abstract.tex, keywords.tex, title.tex, references.bib (plus a copy in each template folder for BibTeX), so that the relative \\input and \\includegraphics
paths of the main.tex files keep working; after the upload choose Menu > Main document > ieee-conference/main.tex (or ieee/main.tex, springer/main.tex).
Usage: py -I scripts/zip_paper_for_overleaf.py   (run scripts/build_paper.py first if the draft changed)
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER, OUT = ROOT / "paper", ROOT / "paper_overleaf.zip"
skip_suffix = {".pdf", ".aux", ".log", ".bbl", ".blg", ".out", ".synctex.gz"}
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    n = 0
    for p in sorted(PAPER.rglob("*")):
        if p.is_file() and p.suffix not in skip_suffix:
            z.write(p, p.relative_to(PAPER).as_posix())
            n += 1
    # BibTeX on Overleaf cannot read a .bib file through a parent path, so every template folder gets its own copy.
    for d in ("ieee", "ieee-conference", "springer"):
        z.write(PAPER / "references.bib", f"{d}/references.bib")
        n += 1
print(f"wrote {OUT.name}: {n} files, {OUT.stat().st_size / 1e6:.1f} MB")
