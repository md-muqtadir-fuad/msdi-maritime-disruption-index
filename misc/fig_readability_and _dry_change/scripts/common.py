"""Shared helpers: run original plotting code with all disk writes blocked,
capture the figures, restyle to print-size fonts, save to the scratch dir."""
import contextlib, importlib.util, math, os, sys, hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure as mfig
from matplotlib.text import Text
import pandas as pd

REPO = Path(__file__).resolve().parents[3]                  # scripts -> folder -> misc -> repo
OUT = Path(__file__).resolve().parents[1] / "output_figs"   # redrawn PDFs/PNGs
CACHE = OUT / "_cache"                                      # captured figures and tables
CACHE.mkdir(parents=True, exist_ok=True)
WIDTH_IN = 6.4         # canvas width; LaTeX text width is 6.5 in
TICK, LABEL, TITLE = 8.0, 9.0, 9.0

sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)  # scripts use repo-relative paths


def load(script: str, name: str | None = None):
    path = REPO / "scripts" / script
    spec = importlib.util.spec_from_file_location(name or path.stem.replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def capture():
    """Block figure saving, closing and table writing; collect saved figures."""
    caught: dict[str, mfig.Figure] = {}
    orig = (mfig.Figure.savefig, plt.close, pd.DataFrame.to_csv, pd.DataFrame.to_parquet)

    def fake_savefig(self, fname, *a, **k):
        caught[Path(str(fname)).stem] = self
    def fake_pltsave(fname, *a, **k):
        caught[Path(str(fname)).stem] = plt.gcf()
    mfig.Figure.savefig = fake_savefig
    orig_pltsave = plt.savefig
    plt.savefig = fake_pltsave
    plt.close = lambda *a, **k: None
    pd.DataFrame.to_csv = lambda self, *a, **k: None
    pd.DataFrame.to_parquet = lambda self, *a, **k: None
    try:
        yield caught
    finally:
        mfig.Figure.savefig, plt.close, pd.DataFrame.to_csv, pd.DataFrame.to_parquet = orig
        plt.savefig = orig_pltsave


def snapshot(*dirs):
    snap = {}
    for d in dirs:
        for p in (REPO / d).rglob("*"):
            if p.is_file():
                st = p.stat()
                snap[str(p)] = (st.st_size, st.st_mtime_ns)
    return snap


def titles(ax):
    return [ax.title, ax._left_title, ax._right_title]


def restyle(fig, tick=TICK, label=LABEL, title=TITLE, other=TICK):
    """Axis labels -> label, titles -> title, tick labels/legends/annotations -> tick/other."""
    special = set()
    for ax in fig.axes:
        ax.tick_params(axis="both", which="both", labelsize=tick)
        for t in (ax.xaxis.label, ax.yaxis.label):
            t.set_fontsize(label); special.add(id(t))
        for t in titles(ax):
            t.set_fontsize(title); special.add(id(t))
        ax.xaxis.get_offset_text().set_fontsize(tick)
        ax.yaxis.get_offset_text().set_fontsize(tick)
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_fontsize(title + 1); special.add(id(fig._suptitle))
    for t in fig.findobj(Text):
        if id(t) in special or not t.get_text():
            continue
        t.set_fontsize(other)


def slim(fig, marker=0.7, line=0.8):
    """Original markers/lines were sized for a ~12 in canvas; thin them for 6.5 in."""
    from matplotlib.lines import Line2D
    from matplotlib.collections import Collection
    from matplotlib.patches import Patch
    for ln in fig.findobj(Line2D):
        ln.set_markersize(ln.get_markersize() * marker)
        ln.set_linewidth(ln.get_linewidth() * line)
        ln.set_markeredgewidth(ln.get_markeredgewidth() * line)
    for c in fig.findobj(Collection):
        lws = c.get_linewidths()
        if len(lws):
            c.set_linewidths([w * line for w in lws])
        try:
            sz = c.get_sizes()
            if len(sz):
                c.set_sizes(sz * marker ** 2)
        except AttributeError:
            pass
    for p in fig.findobj(Patch):
        if p.get_linewidth():
            p.set_linewidth(p.get_linewidth() * line)


def shared_labels(fig, x=None, y=None):
    """Replace repeated per-panel axis labels with one figure-level label."""
    for ax in fig.axes:
        if x is not None and ax.get_xlabel():
            ax.set_xlabel("")
        if y is not None and ax.get_ylabel():
            ax.set_ylabel("")
    if x:
        fig.supxlabel(x, fontsize=LABEL)
    if y:
        fig.supylabel(y, fontsize=LABEL)


def legend_bottom(fig, ncol, source=None, loc="outside lower center"):
    """Move legends to one figure legend under the panels (needs constrained layout)."""
    if source is None:
        leg = (fig.legends[0] if fig.legends else
               next(a.get_legend() for a in fig.axes if a.get_legend()))
    else:
        leg = source
    handles = leg.legend_handles
    labels = [t.get_text() for t in leg.get_texts()]
    for a in fig.axes:
        if a.get_legend():
            a.get_legend().remove()
    for l in list(fig.legends):
        l.remove()
    fig.legend(handles, labels, loc=loc, ncol=ncol, fontsize=TICK, frameon=False)


def constrained(fig, w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02):
    fig.subplots_adjust()  # drop any manual adjust
    fig.set_layout_engine("constrained", w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace)


def resize(fig, height, width=WIDTH_IN):
    fig.set_size_inches(width, height, forward=True)


def save(fig, name, dpi_png=170):
    pdf = OUT / f"{name}.pdf"
    fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{name}.png", format="png", dpi=dpi_png, bbox_inches="tight", pad_inches=0.03)
    return pdf


def measure(pdf, textwidth_bp=6.5 * 72, show=8):
    from pypdf import PdfReader
    page = PdfReader(str(pdf)).pages[0]
    w = float(page.mediabox.width); h = float(page.mediabox.height)
    scale = textwidth_bp / w
    small = []
    sizes = []
    def mult(a, b):
        return [a[0]*b[0]+a[1]*b[2], a[0]*b[1]+a[1]*b[3], a[2]*b[0]+a[3]*b[2], a[2]*b[1]+a[3]*b[3],
                a[4]*b[0]+a[5]*b[2]+b[4], a[4]*b[1]+a[5]*b[3]+b[5]]
    def visit(text, cm, tm, fd, fs):
        if not text or not text.strip():
            return
        m = mult(tm, cm)
        p = fs * math.hypot(m[2], m[3]) * scale
        sizes.append(p)
        if p < 7.95:
            small.append((round(p, 2), text.strip()[:30]))
    page.extract_text(visitor_text=visit)
    lo = min(sizes) if sizes else float("nan")
    print(f"{pdf.name}: {w/72:.2f}x{h/72:.2f} in, scale {scale:.3f}, printed min {lo:.2f} pt, "
          f"max {max(sizes) if sizes else float('nan'):.2f} pt, below-8 spans: {len(small)}")
    for s in small[:show]:
        print("   ", s)
    return lo, small


def wrap(text, width):
    """Wrap at spaces outside $...$ math so mathtext is never split."""
    words, cur, inside = [], "", False
    for ch in text:
        if ch == "$":
            inside = not inside
        if ch == " " and not inside:
            words.append(cur); cur = ""
        else:
            cur += ch
    words.append(cur)
    lines, line = [], ""
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            lines.append(line); line = w
        else:
            line = f"{line} {w}" if line else w
    lines.append(line)
    return "\n".join(lines)


def wrap_titles(fig, width):
    for ax in fig.axes:
        t = ax.get_title()
        if t and "\n" not in t and len(t) > width:
            ax.title.set_text(wrap(t, width))  # keeps weight/size, unlike set_title


def repel_y(ax, texts, min_gap_pt=10.0, iters=200):
    """Push apart text labels that overlap, moving them vertically only."""
    fig = ax.figure
    fig.canvas.draw()
    to_pt = 72.0 / fig.dpi
    for _ in range(iters):
        moved = False
        boxes = [t.get_window_extent() for t in texts]
        for a in range(len(texts)):
            for b in range(a + 1, len(texts)):
                A, B = boxes[a], boxes[b]
                if A.x1 < B.x0 or B.x1 < A.x0:
                    continue
                gap = (abs((A.y0 + A.y1) - (B.y0 + B.y1)) / 2) * to_pt
                if gap >= min_gap_pt:
                    continue
                push = (min_gap_pt - gap) / 2 + 0.5
                up, dn = (texts[a], texts[b]) if (A.y0 + A.y1) >= (B.y0 + B.y1) else (texts[b], texts[a])
                for t, sgn in ((up, 1), (dn, -1)):
                    x, y = t.get_position()
                    px, py = ax.transData.transform((x, y))
                    t.set_position(ax.transData.inverted().transform((px, py + sgn * push / to_pt)))
                moved = True
        if not moved:
            break
        fig.canvas.draw()


def fit_save(fig, name, lo=6.45, hi=6.49, tries=8):
    """Save, then widen or narrow the canvas until the saved (tight) width is just under
    the 6.5 in text width, so LaTeX scales by ~1.0 and text prints at its set size."""
    from pypdf import PdfReader
    for _ in range(tries):
        pdf = save(fig, name)
        w = float(PdfReader(str(pdf)).pages[0].mediabox.width) / 72
        if lo <= w <= hi:
            break
        cw, ch = fig.get_size_inches()
        fig.set_size_inches(cw + ((lo + hi) / 2 - w), ch, forward=True)
        if fig.get_layout_engine() is None:
            fig.tight_layout()
    return pdf


def match_font(fig):
    """New artists (figure legends, sup-labels) should use the figure's own font."""
    name = fig.axes[0].xaxis.label.get_fontname()
    plt.rcParams["font.family"] = [name]
    return name


def clear_of(ax, texts, box_artist, step_pt=2.0, max_steps=80):
    """Shift texts right until they no longer overlap box_artist (e.g. a legend)."""
    fig = ax.figure
    fig.canvas.draw()
    bb = box_artist.get_window_extent()
    for t in texts:
        for _ in range(max_steps):
            tb = t.get_window_extent()
            if not tb.overlaps(bb):
                break
            x, y = t.get_position()
            px, py = ax.transData.transform((x, y))
            t.set_position(ax.transData.inverted().transform((px + step_pt * fig.dpi / 72, py)))
            fig.canvas.draw()


def keep_inside(ax, pad_pt=2.0):
    """Move rotated annotation texts down so they do not cross the top of the axes."""
    fig = ax.figure
    fig.canvas.draw()
    top = ax.get_window_extent().y1 - pad_pt * fig.dpi / 72
    for t in ax.texts:
        if not t.get_text() or abs(t.get_rotation() - 90) > 1:
            continue
        bb = t.get_window_extent()
        if bb.y1 > top:
            x, y = t.get_position()
            px, py = ax.transData.transform((x, y))
            t.set_position(ax.transData.inverted().transform((px, py - (bb.y1 - top))))
