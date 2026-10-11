"""Stage 24's draw_pcmci_causal_graph(), unchanged in data and geometry, with the
point-sized quantities (arrow shrink, head size, line widths) rescaled to the smaller
panel and the label offsets widened so 8 pt labels clear the arcs and loops."""
import math
import pandas as pd
from matplotlib.patches import Arc, Circle, FancyArrowPatch

ORIGINAL_UNIT_PT = 75.0  # data unit in points in the 16.5 in original


def draw(ax, subset, colors, title, s24, unit_pt, fs=8.0):
    k = unit_pt / ORIGINAL_UNIT_PT
    P, L, R = s24.PCMCI_GRAPH_POSITIONS, s24.PCMCI_GRAPH_NODE_LABELS, s24.PCMCI_GRAPH_PAIR_RADIUS
    node_r, loop_r = 0.32, 0.19
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-1.95, 1.95); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title)
    for name, (x, y) in P.items():
        ax.add_patch(Circle((x, y), node_r, facecolor="#eef2f7", edgecolor="#111827", zorder=4, lw=1.4 * k ** 0.5))
        ax.text(x, y, L[name], ha="center", va="center", fontsize=fs, zorder=5, linespacing=1.0)
    cross = subset.loc[subset["source"] != subset["target"]]
    loops = subset.loc[subset["source"] == subset["target"]]
    it = {key: iter([b, -b]) for key, b in R.items()}
    for _, row in cross.iterrows():
        x0, y0 = P[row["source"]]; x1, y1 = P[row["target"]]
        key = tuple(sorted([row["source"], row["target"]]))
        # opposite directions of a pair get opposite bulges (stage 24's sign rule drew
        # both on the same arc); each label then sits on its own arc's bulge side
        rad = next(it[key]) * (1 if row["source"] == key[0] else -1)
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
            mutation_scale=16 * k, shrinkA=24 * k, shrinkB=24 * k,
            linewidth=(1.0 + 3.6 * abs(row["partial_correlation"])) * k ** 0.5,
            color=colors["primary"], zorder=2))
        t = 0.5
        chord = math.hypot(x1 - x0, y1 - y0)
        offset = abs(rad) * chord / 2 + 0.26  # clear of the arc at 8 pt
        mx, my = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        # arc3 bulges along rad*(dy, -dx)
        bx, by = (y1 - y0), -(x1 - x0)
        n = math.hypot(bx, by); px, py = math.copysign(1, rad) * bx / n, math.copysign(1, rad) * by / n
        ax.text(mx + offset * px, my + offset * py, f"lag {int(row['source_lag'])}", fontsize=fs,
                ha="center", va="center", color=colors["primary"], zorder=6, fontweight="bold")
    for name, (ux, uy) in P.items():
        rows = loops.loc[loops["source"] == name]
        if rows.empty:
            continue
        d = 1.0 + node_r + loop_r + 0.05
        lx, ly = ux * d, uy * d
        w = (1.2 + 2.6 * pd.to_numeric(rows["partial_correlation"], errors="coerce").abs().max()) * k ** 0.5
        ax.add_patch(Arc((lx, ly), 2 * loop_r, 2 * loop_r, angle=0, theta1=25, theta2=335,
                         edgecolor=colors["robustness"], lw=w, zorder=3))
        ha, ta = math.radians(25), math.radians(43)
        ax.annotate("", xy=(lx + loop_r * math.cos(ha), ly + loop_r * math.sin(ha)),
                    xytext=(lx + loop_r * math.cos(ta), ly + loop_r * math.sin(ta)),
                    arrowprops=dict(arrowstyle="-|>", color=colors["robustness"], lw=w, mutation_scale=11 * k),
                    zorder=3)
        lags = ", ".join(str(int(v)) for v in sorted(rows["source_lag"]))
        if uy == 0:   # left/right nodes: label under the loop, not across it
            tx, ty, va = lx, ly - loop_r - 0.06, "top"
        else:
            tx, ty, va = 0.0, uy * (d + loop_r + 0.08), "bottom" if uy > 0 else "top"
        ax.text(tx, ty, f"lags {lags}", fontsize=fs, ha="center", va=va,
                color=colors["robustness"], zorder=6, fontweight="bold")
