"""Script 49's port map rebuilt at text width: same table, land, links, markers, colours,
legend and colorbar; ISO3 labels re-placed at 8 pt with leader lines where displaced."""
import math, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from common import OUT, CACHE, WIDTH_IN, load, capture, measure, fit_save

FS = 8.0


def get_table():
    cache = CACHE / "portmap_table.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    s49 = load("49_shanghai_vs_other_ports.py", "s49")
    grabbed = {}
    s49.build_figure = lambda table: grabbed.setdefault("table", table.copy())
    with capture():
        s49.main()
    pickle.dump(grabbed["table"], open(cache, "wb"))
    return grabbed["table"]


def place_labels(fig, ax, items, obstacles):
    """Greedy first-fit over rings of candidate offsets (points). items are dicts with
    x, y, text, size, weight, color, halo; they are placed in order, each avoiding the
    obstacles (legend, port markers) and every label already placed. A thin leader line
    is drawn when a label ends up far from its port."""
    fig.canvas.draw()
    scale = fig.dpi / 72.0
    cands = []
    for r in (10, 14, 19, 25, 32, 40, 49):
        for ang in (270, 90, 0, 180, 315, 225, 45, 135, 292, 248, 68, 112, 338, 202, 22, 158):
            cands.append((r * math.cos(math.radians(ang)) * 1.3, r * math.sin(math.radians(ang))))
    placed = list(obstacles)
    axbb = ax.get_window_extent()
    for it in items:
        px, py = ax.transData.transform((it["x"], it["y"]))
        w = (0.6 * it["size"] * len(it["text"]) + 4.0) * scale
        h = (it["size"] + 3.0) * scale
        chosen = None
        for dx, dy in cands:
            x0, y0 = px + dx * scale - w / 2, py + dy * scale - h / 2
            if x0 < axbb.x0 or x0 + w > axbb.x1 or y0 < axbb.y0 or y0 + h > axbb.y1:
                continue
            if not any(x0 < bx + bw and x0 + w > bx and y0 < by + bh and y0 + h > by
                       for bx, by, bw, bh in placed):
                chosen = (dx, dy)
                placed.append((x0, y0, w, h))
                break
        if chosen is None:
            chosen = cands[0]
            print("label could not be placed cleanly:", it["text"])
        far = math.hypot(*chosen) > 15
        ax.annotate(
            it["text"], xy=(it["x"], it["y"]), xytext=chosen, textcoords="offset points",
            ha="center", va="center", fontsize=it["size"], weight=it["weight"], color=it["color"],
            zorder=8, path_effects=[path_effects.withStroke(linewidth=it["halo"], foreground="white")],
            arrowprops=dict(arrowstyle="-", lw=0.4, color="#6b7280", shrinkA=4, shrinkB=3)
            if far else None,
        )


def build(table):
    s49 = load("49_shanghai_vs_other_ports.py", "s49")
    C = s49.COLORS
    fig = plt.figure(figsize=(WIDTH_IN, 2.85))
    ax = fig.add_axes([0.0, 0.2, 1.0, 0.8])
    ax.set_facecolor(C["ocean"])
    s49.draw_land(ax)

    mapped = table.dropna(subset=["lat", "lon"]).copy()
    mapped["x"] = mapped["lon"].map(s49.shift_lon)
    focus = mapped[mapped["portname"].eq(s49.FOCUS_PORT)]
    others = mapped[~mapped["portname"].eq(s49.FOCUS_PORT)]
    ox, oy = float(focus["x"].iloc[0]), float(focus["lat"].iloc[0])
    cut = float(others["connectivity"].median())
    for row in others.itertuples():
        if pd.isna(row.connectivity):
            continue
        strong = row.connectivity >= cut
        rad = -0.16 if row.lon < s49.LON_CUT else 0.16
        ax.add_patch(FancyArrowPatch(
            (ox, oy), (row.x, row.lat), connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
            linewidth=0.6, color=C["link_high"] if strong else C["link_low"],
            alpha=0.92 if strong else 0.78, zorder=3 if strong else 2))

    calls = others["portcalls_container"].astype(float)
    k = 0.55  # marker areas scaled with the map (map is ~0.55 of the original width)
    scatter = ax.scatter(others["x"], others["lat"], s=k * (26 + 150 * calls / calls.max()),
                         c=others["correlation"], cmap=s49.HOUSE_CMAP, vmin=0.2, vmax=0.85,
                         edgecolor="none", linewidth=0, alpha=0.95, zorder=4)
    ax.scatter(focus["x"], focus["lat"],
               s=k * (26 + 150 * float(focus["portcalls_container"].iloc[0]) / float(calls.max())),
               marker="o", color=C["focus"], edgecolor="none", linewidth=0, zorder=6)

    ax.set_xlim(s49.LON_CUT, s49.LON_CUT + 360)
    ax.set_ylim(-48, 66)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("N")
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_linewidth(0.5); side.set_color("#999999")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C["focus"],
               markeredgecolor="none", markersize=7, label="Shanghai"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9fb8cd",
               markeredgecolor="none", markersize=4, label=r"marker area $\propto$ container port calls"),
        Line2D([0], [0], color=C["link_high"], linewidth=1.2, alpha=0.92, label="stronger"),
        Line2D([0], [0], color=C["link_low"], linewidth=1.2, alpha=0.78, label="weaker"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.85,
                    edgecolor="none", fontsize=FS, borderpad=0.4, labelspacing=0.3)

    fig.canvas.draw()
    bb = leg.get_window_extent()
    obstacles = [(bb.x0, bb.y0, bb.width, bb.height)]
    # every port marker is an obstacle too, so labels do not sit on neighbouring ports
    sizes = list(k * (26 + 150 * calls / calls.max()))
    sizes.append(k * (26 + 150 * float(focus["portcalls_container"].iloc[0]) / float(calls.max())))
    for (mx, my), s_pt2 in zip(list(zip(others["x"], others["lat"])) + [(ox, oy)], sizes):
        cx, cy = ax.transData.transform((mx, my))
        r = (math.sqrt(s_pt2) / 2 + 1.0) * fig.dpi / 72.0
        obstacles.append((cx - r, cy - r, 2 * r, 2 * r))
    items = [dict(x=ox, y=oy, text="Shanghai", size=9, weight="bold", color=C["focus_label"], halo=2.4)]
    for row in others.sort_values("portcalls_container", ascending=False).itertuples():
        items.append(dict(x=row.x, y=row.lat, text=row.ISO3, size=FS, weight="normal",
                          color="#3a3a3a", halo=2.2))
    place_labels(fig, ax, items, obstacles)

    box = ax.get_position()
    cax = fig.add_axes([box.x0 + 0.28 * box.width, box.y0 - 0.085, 0.44 * box.width, 0.035])
    cbar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
    cbar.set_label("Correlation with Shanghai (monthly log change in port calls)", fontsize=9)
    cbar.ax.tick_params(labelsize=FS)
    return fig


if __name__ == "__main__":
    s49 = load("49_shanghai_vs_other_ports.py", "s49")  # applies the script's rcParams
    fig = build(get_table())
    measure(fit_save(fig, "shanghai_vs_other_ports"))
