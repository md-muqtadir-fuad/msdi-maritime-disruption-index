"""List abbreviations in a LaTeX manuscript: first use (abstract and body separately),
the section it is in, and where it is first spelled out as "Full name (ABBR)".

    python audit_abbreviations.py ../../../latex/main.tex > abbreviations.tsv
"""
import re, sys
from collections import defaultdict

path = sys.argv[1]
raw = open(path, encoding="utf-8").read()


def blank(m):  # keep offsets and line numbers
    return re.sub(r"[^\n]", " ", m.group(0))


s = re.sub(r"(?<!\\)%[^\n]*", blank, raw)                                     # comments
s = re.sub(r"\\(?:cite[tp]?|ref|eqref|label|includegraphics|url|href|bibliography\w*|"
           r"begin|end|usepackage|documentclass|input|counterwithin)\*?(\[[^\]]*\])?\{[^}]*\}", blank, s)
s = re.sub(r"\\\[.*?\\\]|\$\$.*?\$\$|\\begin\{(equation|align)\*?\}.*?\\end\{\1\*?\}", blank, s, flags=re.S)
s = re.sub(r"(?<!\\)\$[^$]*\$", blank, s)                                     # inline math
s = re.sub(r"\\[A-Za-z]+", blank, s)                                          # remaining commands
s = re.sub(r"\\texttt\{[^}]*\}", blank, s)

abs_start = raw.find(r"\begin{abstract}")
abs_end = raw.find(r"\end{abstract}")
body_start = abs_end
appendix = raw.find(r"\appendix")
line_of = lambda pos: raw.count("\n", 0, pos) + 1

heads = [(m.start(), m.group(2)) for m in re.finditer(r"\\(sub)*section\*?\{([^}]*)\}", raw)]
def section(pos):
    name = "Abstract" if abs_start <= pos < abs_end else "Front matter"
    for p, h in heads:
        if p <= pos:
            name = h
    return name + (" (appendix)" if appendix != -1 and pos > appendix else "")


TOK = re.compile(r"(?<![A-Za-z0-9\-/])([A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*(?:[\-/][A-Z0-9][A-Za-z0-9]*)*\+?s?)(?![A-Za-z0-9])")
SKIP = {"SHA", "II", "III", "IV", "VI", "ID"}
uses = defaultdict(list)
for m in TOK.finditer(s):
    tok = m.group(1)
    base = tok[:-1] if tok.endswith("s") and tok[:-1].isupper() else tok
    if sum(c.isupper() for c in base) < 2 or base in SKIP:
        continue
    uses[base].append(m.start())


def defined_at(tok):
    """Position of the first 'Full name (TOK)' or 'TOK (full name)' in the text."""
    pat = re.compile(r"[A-Za-z][^()]{3,}\(\s*" + re.escape(tok) + r"s?\s*[,;)]|" +
                     re.escape(tok) + r"s?\s*\(\s*[A-Za-z][a-z]", re.S)
    m = pat.search(s)
    return m.start() if m else None


rows = []
for tok, pos in uses.items():
    in_abs = [p for p in pos if abs_start <= p < abs_end]
    in_body = [p for p in pos if p >= body_start]
    d = defined_at(tok)
    first_body = in_body[0] if in_body else None
    rows.append((pos[0], tok, len(pos),
                 line_of(in_abs[0]) if in_abs else "",
                 line_of(first_body) if first_body else "",
                 section(first_body) if first_body else "",
                 line_of(d) if d is not None else "never",
                 section(d) if d is not None else ""))

print("abbr\tuses\tabstract_line\tbody_first_line\tbody_first_section\tdefined_line\tdefined_section")
for _, tok, n, al, bl, bs, dl, ds in sorted(rows):
    print(f"{tok}\t{n}\t{al}\t{bl}\t{bs}\t{dl}\t{ds}")
