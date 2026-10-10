import re, sys
from collections import Counter
from pathlib import Path
from pypdf import PdfReader
LATEX = Path(__file__).resolve().parents[3] / "latex"
OUT = Path(__file__).resolve().parents[1] / "output_figs"
def toks(p):
    t = PdfReader(str(p)).pages[0].extract_text() or ""
    return Counter(re.findall(r"[A-Za-z][A-Za-z0-9/\-\+\.]*|[\-\u2212]?\d+(?:[.,]\d+)?", t.replace("\u2212", "-")))
for name in sys.argv[1:]:
    old, new = toks(LATEX / f"{name}.pdf"), toks(OUT / f"{name}.pdf")
    only_old = sorted((old - new).elements()); only_new = sorted((new - old).elements())
    print(f"== {name}: old {sum(old.values())} tokens, new {sum(new.values())}")
    print("   only in old:", " ".join(only_old[:60]))
    print("   only in new:", " ".join(only_new[:60]))
