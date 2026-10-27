r"""Builds the LaTeX sources of the paper from docs/paper_draft.md (single source of truth): paper/body.tex, paper/abstract.tex, paper/keywords.tex, paper/title.tex and paper/figures/*.png.
title.tex defines \papertitle and the body uses \startappendix (defined by each wrapper). The two class-specific wrappers paper/ieee/main.tex (IEEEtran, journal mode) and paper/springer/main.tex (Springer LNCS, llncs) input these files.
Markdown subset handled: headings, paragraphs, **bold**, *italic*, `code`, pipe tables followed by an italic caption line (*Table N. ...*), images ![caption](path), numbered and bullet lists,
citations [@key; @key] and figure references "Fig. [#name]". Bracketed report pointers such as [`reports/x.md`] are dropped (they belong to the data and code availability statement).
The script also checks its own output: citation keys against paper/references.bib, figure files, brace balance and begin/end balance.
Usage: py -I -X utf8 scripts/build_paper.py
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT, PAPER = ROOT / "docs" / "paper_draft.md", ROOT / "paper"
WIDE_FIG = {"fig06_walk_forward", "fig16_dayahead_models", "fig10_learning_curves", "fig14_dayahead", "fig13_daywise", "fig12_study_design", "fig17_other_families"}
SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
UNI = {"\u2212": "$-$", "\u00d7": r"$\times$", "\u00e8": r"\`e", "\u00fc": r"\"u"}


class Inline:
    """Markdown inline text -> LaTeX, with placeholders for code, citations and references so that they are not escaped."""
    def __init__(self):
        self.store = []

    def hold(self, latex):
        self.store.append(latex)
        return "\x00%d\x00" % (len(self.store) - 1)

    def code(self, m):
        s = m.group(1)
        out = "".join(SPECIAL.get(c, c) if c != "\\" else r"\textbackslash{}" for c in s).replace("/", r"/\allowbreak{}").replace(r"\_", r"\_\allowbreak{}")
        return self.hold(r"\texttt{" + out + "}")

    def cite(self, m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        return self.hold("~" + r"\cite{" + ",".join(keys) + "}")

    def conv(self, s):
        s = re.sub(r"\s*\[`[^`]*`(?:,\s*`[^`]*`)*\]", "", s)                     # report pointers
        s = re.sub(r"`([^`]+)`", self.code, s)
        s = re.sub(r"\s*\[(@[A-Za-z0-9_:\-]+(?:;\s*@[A-Za-z0-9_:\-]+)*)\]", self.cite, s)
        s = re.sub(r"Fig\.\s*\[#([A-Za-z0-9_]+)\]", lambda m: self.hold(r"Fig.~\ref{fig:" + m.group(1) + "}"), s)
        s = "".join(SPECIAL.get(c, c) if c != "\\" else r"\textbackslash{}" for c in s)
        s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
        s = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"\\emph{\1}", s)
        s = re.sub(r'"([^"]+)"', r"``\1''", s)
        for k, v in UNI.items():
            s = s.replace(k, v)
        s = re.sub(r"\x00(\d+)\x00", lambda m: self.store[int(m.group(1))], s)
        return s


def inline(s):
    return Inline().conv(s)


def parse_table(lines):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    header, sep, body = rows[0], rows[1], rows[2:]
    align = ["r" if c.endswith(":") and not c.startswith(":") else ("c" if c.startswith(":") and c.endswith(":") else "l") for c in sep]
    return header, align, body


def table_tex(lines, caption, n):
    header, align, body = parse_table(lines)
    width = sum(max(len(re.sub(r"[*`]", "", r[i])) for r in [header] + body) for i in range(len(header))) + 2 * len(header)
    env = "wtable" if width > 62 else "table"
    out = [r"\begin{" + env + "}[!t]", r"\caption{" + inline(caption) + r"}\label{tab:%d}" % n, r"\centering", r"{\tablesize", r"\begin{adjustbox}{max width=\linewidth}",
           r"\begin{tabular}{" + "".join(align) + "}", r"\toprule", " & ".join(inline(c) for c in header) + r" \\", r"\midrule"]
    out += [" & ".join(inline(c) for c in r) + r" \\" for r in body]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", "}", r"\end{" + env + "}", ""]
    return out


def figure_tex(path, caption):
    name = Path(path).stem
    env = "wfigure" if name in WIDE_FIG else "figure"
    return [r"\begin{" + env + "}[!t]", r"\centering", r"\includegraphics[width=\linewidth]{" + name + "}", r"\caption{" + inline(caption) + r"}\label{fig:" + name + "}", r"\end{" + env + "}", ""], name


def convert(md):
    lines = md.splitlines()
    out, para, i, ntab, appendix, figs = [], [], 0, 0, False, []

    def flush():
        if para:
            out.append(inline(" ".join(para)) + "\n")
            para.clear()

    while i < len(lines):
        ln = lines[i]
        if ln.strip() in ("", "---"):
            flush(); i += 1; continue
        m = re.match(r"^(#{2,3}) (.+)$", ln)
        if m:
            flush()
            title = m.group(2)
            if m.group(1) == "##":
                if re.match(r"Appendix [A-Z]\. ", title):
                    if not appendix:
                        out.append(r"\startappendix" + "\n"); appendix = True
                    title = re.sub(r"^Appendix [A-Z]\.\s*", "", title)
                else:
                    title = re.sub(r"^\d+\.\s*", "", title)
                out.append(r"\section{" + inline(title) + "}\n")
            else:
                out.append(r"\subsection{" + inline(re.sub(r"^\d+\.\d+\s*", "", title)) + "}\n")
            i += 1; continue
        m = re.match(r"^!\[(.*)\]\((.+)\)\s*$", ln)
        if m:
            flush()
            block, name = figure_tex(m.group(2), m.group(1)); out += block; figs.append(name); i += 1; continue
        if ln.lstrip().startswith("|"):
            flush()
            tl = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                tl.append(lines[i]); i += 1
            j = i
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            cap = ""
            if j < len(lines) and re.match(r"^\*Table \d+\.", lines[j]):
                cap = re.sub(r"^\*Table \d+\.\s*", "", lines[j].strip()).rstrip("*").strip()
                i = j + 1
            ntab += 1
            out += table_tex(tl, cap, ntab); continue
        if re.match(r"^\d+\. ", ln) or ln.startswith("- "):
            flush()
            ordered = bool(re.match(r"^\d+\. ", ln))
            env = "enumerate" if ordered else "itemize"
            items = []
            while i < len(lines) and (re.match(r"^\d+\. ", lines[i]) if ordered else lines[i].startswith("- ")):
                items.append(re.sub(r"^(\d+\.|-) ", "", lines[i])); i += 1
            out += [r"\begin{" + env + "}"] + [r"\item " + inline(it) for it in items] + [r"\end{" + env + "}", ""]
            continue
        para.append(ln.strip()); i += 1
    flush()
    return "\n".join(out), figs


def main():
    md = DRAFT.read_text(encoding="utf-8")
    title = re.match(r"# (.+)", md).group(1)
    ab = md.split("## Abstract", 1)[1].split("**Keywords:**", 1)
    abstract = "\n\n".join(inline(p.replace("\n", " ").strip()) for p in ab[0].strip().split("\n\n") if p.strip() and p.strip() != "---")
    keywords = ab[1].split("\n", 1)[0].strip()
    body_md = md.split("## 1. Introduction", 1)[1]
    body_md = "## 1. Introduction" + body_md
    body_md = body_md.split("## References", 1)[0]
    body, figs = convert(body_md)
    PAPER.mkdir(exist_ok=True)
    (PAPER / "figures").mkdir(exist_ok=True)
    for name in sorted(set(figs)):
        src = ROOT / "reports" / "figures" / (name + ".png")
        if not src.exists():
            sys.exit("missing figure " + str(src))
        shutil.copy2(src, PAPER / "figures" / src.name)
    (PAPER / "body.tex").write_text(body, encoding="utf-8")
    (PAPER / "abstract.tex").write_text(abstract + "\n", encoding="utf-8")
    (PAPER / "keywords.tex").write_text(inline(keywords) + "\n", encoding="utf-8")
    (PAPER / "title.tex").write_text(r"\newcommand{\papertitle}{" + inline(title) + "}\n", encoding="utf-8")
    # ---- checks of the generated sources
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.M))
    used = set(k for grp in re.findall(r"\\cite\{([^}]*)\}", body + abstract) for k in grp.split(","))
    problems = []
    if used - keys:
        problems.append("citation keys without a bib entry: " + str(sorted(used - keys)))
    labels = set(re.findall(r"\\label\{([^}]*)\}", body))
    refs = set(re.findall(r"\\ref\{([^}]*)\}", body))
    if refs - labels:
        problems.append("references without a label: " + str(sorted(refs - labels)))
    for name, text in (("body", body), ("abstract", abstract)):
        t = re.sub(r"\\[{}]", "", text)
        if t.count("{") != t.count("}"):
            problems.append(f"{name}: unbalanced braces ({t.count('{')} open, {t.count('}')} close)")
    envs = re.findall(r"\\(begin|end)\{(\w+\*?)\}", body)
    stack = []
    for kind, e in envs:
        if kind == "begin":
            stack.append(e)
        elif not stack or stack.pop() != e:
            problems.append("environment mismatch near " + e)
    if stack:
        problems.append("unclosed environments: " + str(stack))
    if re.search(r"[\u2212\u00d7]", body + abstract):
        problems.append("unconverted unicode characters remain")
    print(f"body.tex: {len(body.splitlines())} lines | figures: {len(set(figs))} | tables: {len(re.findall(r'begin.(?:w)?table.', body))} | citations used: {len(used)} of {len(keys)} bib entries | problems: {problems or 'none'}")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
