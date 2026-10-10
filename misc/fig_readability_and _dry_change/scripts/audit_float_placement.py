import re, sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()

# blank out comments (keep offsets)
def strip_comments(s):
    out = []
    for line in s.split("\n"):
        m = re.search(r"(?<!\\)%", line)
        out.append(line if not m else line[:m.start()] + " " * (len(line) - m.start()))
    return "\n".join(out)

s = strip_comments(src)
lineno = lambda pos: s.count("\n", 0, pos) + 1

ENVS = ["figure", "figure*", "table", "table*", "longtable", "sidewaystable", "sidewaysfigure"]
floats = []
for env in ENVS:
    for m in re.finditer(r"\\begin\{" + re.escape(env) + r"\}", s):
        end = s.find(r"\end{" + env + "}", m.end())
        body = s[m.start():end]
        labels = re.findall(r"\\label\{([^}]+)\}", body)
        floats.append(dict(env=env, start=m.start(), end=end + len(env) + 6, labels=labels))
floats.sort(key=lambda f: f["start"])
# drop nested (e.g. table inside landscape is fine; table inside figure unlikely)
spans = [(f["start"], f["end"]) for f in floats]
def in_float(pos):
    return any(a <= pos < b for a, b in spans)

# expand float start to include wrapping landscape
for f in floats:
    pre = s[:f["start"]].rstrip()
    if pre.endswith(r"\begin{landscape}"):
        f["outer_start"] = pre.rfind(r"\begin{landscape}")
    else:
        f["outer_start"] = f["start"]

heads = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\\(sub)*section\*?\{[^}]*\}|\\appendix|\\chapter\{[^}]*\}", s)]
refs = list(re.finditer(r"\\(?:ref|autoref|cref|Cref|eqref|pageref)\{([^}]+)\}", s))

def para_end_phrase(pos):
    # paragraph containing pos ends at next blank line or \begin{ / \subsection
    m = re.compile(r"\n\s*\n|\\begin\{(?:figure|table|longtable|landscape|equation)|\\(?:sub)*section").search(s, pos)
    e = m.start() if m else len(s)
    txt = s[pos:e].rstrip()
    tail = txt[-110:].replace("\n", " ")
    return tail, lineno(e)

def heading_before(pos):
    prev = s[:pos]
    # strip trailing whitespace and \label{...} lines
    t = re.sub(r"(\s|\\label\{[^}]*\}|\\begin\{landscape\})*$", "", prev)
    for a, b, h in heads:
        if b == len(t):
            return h
    return None

for f in floats:
    lbl = ",".join(f["labels"]) or "(no label)"
    first = None
    for r in refs:
        keys = r.group(1).split(",")
        if any(k.strip() in f["labels"] for k in keys) and not (f["start"] <= r.start() < f["end"]):
            first = r
            break
    under = heading_before(f["outer_start"])
    status = []
    if under:
        status.append(f"UNDER-HEADING[{under}]")
    if first is None:
        status.append("NEVER-CITED")
    elif first.start() > f["start"]:
        status.append("CITED-AFTER")
    cite_in_float = first is not None and in_float(first.start())
    print(f"L{lineno(f['start'])} {f['env']} {lbl} first_ref=L{lineno(first.start()) if first else '-'}"
          f"{' (inside another float)' if cite_in_float else ''} {' '.join(status) or 'OK'}")
    if first is not None and (first.start() > f["start"] or under):
        tail, el = para_end_phrase(first.start())
        print(f"    move after paragraph ending (L{el}): ...{tail}")
