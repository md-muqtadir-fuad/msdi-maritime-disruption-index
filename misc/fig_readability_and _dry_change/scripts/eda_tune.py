"""Layout pass for the captured EDA-family figures (loaded from pickles)."""
import sys, pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from common import (OUT, CACHE, fit_save, restyle, resize, save, measure, slim, shared_labels, legend_bottom,
                    constrained, wrap, wrap_titles, repel_y, LABEL, match_font, clear_of, keep_inside)

HEIGHT = {"eda_correlation": 5.6, "eda_distributions": 5.8, "empirical_relationships": 5.2,
          "persistent_event_shocks_timeline": 5.2, "msdi_vs_external_benchmarks": 6.4,
          "dimension_pca": 3.3, "sample_size_convergence": 3.2}


def years(ax, step=2, fmt="%Y"):
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    for l in ax.get_xticklabels():
        l.set_rotation(0); l.set_ha("center")


def t_correlation(fig):
    ax = fig.axes[0]
    for t in ax.texts:
        txt = t.get_text()
        num = txt.rstrip("*")
        stars = txt[len(num):]
        if stars:
            t.set_text(f"{num}\n{stars}")
            t.set_linespacing(0.85)
    labels = [l.get_text() for l in ax.get_yticklabels()]
    n = len(labels)
    # strict lower triangle: row 0 and column n-1 hold no cells, so crop them away
    ax.set_xlim(-0.5, n - 1.5)
    ax.set_ylim(n - 0.5, 0.5)
    ax.set_yticks(range(1, n))
    ax.set_yticklabels([wrap(l, 16) for l in labels[1:]])
    ax.set_xticks(range(n - 1))
    ax.set_xticklabels(labels[:-1], rotation=45, ha="right", rotation_mode="anchor")
    # no background grid over the empty upper triangle; keep the white cell separators
    ax.grid(False)
    ax.set_xticks([x - 0.5 for x in range(n)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(n)], minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    return "tight"


def t_distributions(fig):
    a, d = fig.axes[0], fig.axes[3]
    # mathtext puts a space after a comma ("6, 480"); {,} keeps the thousands separator tight
    a.title.set_text(a.get_title().replace("6,480", "6{,}480"))
    wrap_titles(fig, 40)
    years(d, step=2)
    leg = d.get_legend()  # was on top of the 2021-22 peak; move under the panel
    h, l = leg.legend_handles, [t.get_text() for t in leg.get_texts()]
    leg.remove()
    d.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False,
             handlelength=1.8, columnspacing=1.0)
    constrained(fig)


def t_empirical(fig):
    for ax in fig.axes:
        t = ax.get_title()
        if " ($" in t:
            name, rho = t.split(" ($", 1)
            # stars inside mathtext render as spaced operators ("0.82 * * *"); move them out
            body = rho.rstrip(")").rstrip("$")
            num = body.rstrip("*")
            stars = body[len(num):]
            ax.title.set_text(f"{name}\n(${num}${stars})")
    leg = fig.axes[0].get_legend()
    if leg:
        leg.set_loc("upper left")  # upper right sat on the end of the fit line
    shared_labels(fig, y=fig.axes[0].get_ylabel())
    constrained(fig)


def t_timeline(fig):
    lo, hi = fig.axes[0].get_ylim()
    fig.axes[0].set_ylim(lo, hi * 1.12)  # headroom so the legend clears the event labels
    for ax in fig.axes:
        if ax.get_ylabel():
            ax.set_ylabel(wrap(ax.get_ylabel(), 22))
    constrained(fig)


def t_benchmarks(fig):
    axes = [a for a in fig.axes]
    main = [a for a in axes if a.get_title()]
    wrap_titles(fig, 36)
    a, b, c, d = main[:4]
    la = a.get_legend()
    if la:
        h, l = la.legend_handles, [t.get_text() for t in la.get_texts()]
        la.remove()
        a.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False,
                 handlelength=1.8)
    years(a, step=2)
    for ax in (c, d):
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for l in ax.get_xticklabels():
            l.set_rotation(35); l.set_ha("right")
    b.set_xlabel(wrap(b.get_xlabel(), 38))
    for ax in axes:
        if ax.get_ylabel():
            ax.set_ylabel(wrap(ax.get_ylabel(), 26))
    leg = b.get_legend()
    if leg:
        h, l = leg.legend_handles, [t.get_text() for t in leg.get_texts()]
        leg.remove()
        b.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1, frameon=False,
                 handlelength=1.8)
    return "tight"


def t_pca(fig):
    constrained(fig)


def t_sample(fig):
    legend_bottom(fig, ncol=4)
    constrained(fig)


TUNE = {"eda_correlation": t_correlation, "eda_distributions": t_distributions,
        "empirical_relationships": t_empirical, "persistent_event_shocks_timeline": t_timeline,
        "msdi_vs_external_benchmarks": t_benchmarks, "dimension_pca": t_pca,
        "sample_size_convergence": t_sample}


def halo(texts, lw=2.4):
    """White outline so a label stays legible where a line or arrow crosses it."""
    import matplotlib.patheffects as pe
    for t in texts:
        t.set_path_effects([pe.withStroke(linewidth=lw, foreground="white")])


def place_loading_labels(ax, out_pt=7.0, gap_pt=10.5, iters=300):
    """Biplot labels: put each just beyond its arrow tip, along the arrow, push overlapping
    labels apart vertically, keep them clear of the legend, and draw a thin leader line
    from the tip whenever a label had to move."""
    import numpy as np
    fig = ax.figure
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + 0.12 * (x1 - x0))  # room for the right-most labels inside the axes
    texts = list(ax.texts)
    tips = [t.xy for t in texts if hasattr(t, "xy") and not t.get_text()]
    labels = [t for t in texts if t.get_text() and not hasattr(t, "xy")]
    new = []
    for tip, old in zip(tips, labels):
        px, py = ax.transData.transform(tip) - ax.transData.transform((0, 0))
        u = np.array([px, py]) / max(np.hypot(px, py), 1e-9)
        ha = "left" if u[0] > 0.35 else "right" if u[0] < -0.35 else "center"
        va = "bottom" if u[1] > 0.35 else "top" if u[1] < -0.35 else "center"
        new.append(ax.annotate(old.get_text(), xy=tip, xytext=tuple(out_pt * u), textcoords="offset points",
                               ha=ha, va=va, fontsize=old.get_fontsize(), color=old.get_color(), zorder=6))
        old.remove()
    legend = ax.get_legend()
    to_pt = 72.0 / fig.dpi
    for _ in range(iters):
        fig.canvas.draw()
        moved = False
        boxes = [a.get_window_extent().expanded(1.0, 1.0) for a in new]
        for i in range(len(new)):
            for j in range(i + 1, len(new)):
                A, B = boxes[i], boxes[j]
                if A.x1 < B.x0 or B.x1 < A.x0:
                    continue
                dy = abs((A.y0 + A.y1) - (B.y0 + B.y1)) / 2 * to_pt
                if dy >= gap_pt:
                    continue
                push = (gap_pt - dy) / 2 + 0.3
                up, dn = (new[i], new[j]) if A.y0 + A.y1 >= B.y0 + B.y1 else (new[j], new[i])
                for a, s in ((up, 1), (dn, -1)):
                    x, y = a.xyann
                    a.xyann = (x, y + s * push)
                moved = True
            if legend is not None and boxes[i].overlaps(legend.get_window_extent()):
                x, y = new[i].xyann
                new[i].xyann = (x + 2.0, y)
                moved = True
        if not moved:
            break
    for a in new:  # a leader on every label ties it to its own arrow tip
        x, y = a.xyann
        if True:
            ax.annotate("", xy=a.xy, xytext=(x, y), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="#6b7280", shrinkA=0, shrinkB=0),
                        zorder=5)
    halo(new)


def post(name, fig):
    if name == "persistent_event_shocks_timeline":
        for ax in fig.axes:
            keep_inside(ax)
            labels = [t for t in ax.texts if t.get_text()]
            halo(labels)
            for t in labels:
                t.set_zorder(6)  # above the MSDI line, which otherwise cuts "Shanghai Lockdown"
    if name == "sample_size_convergence":
        for ax in fig.axes:
            halo([t for t in ax.texts if t.get_text()])
    if name == "dimension_pca":
        place_loading_labels(fig.axes[1])


def run(name):
    fig = pickle.load(open(CACHE / f"{name}.raw.pkl", "rb"))
    match_font(fig)
    resize(fig, HEIGHT[name])
    slim(fig)
    mode = TUNE[name](fig)
    restyle(fig)
    if mode == "tight":
        fig.tight_layout()
    post(name, fig)
    return measure(fit_save(fig, name))


if __name__ == "__main__":
    for name in sys.argv[1:] or HEIGHT:
        try:
            run(name)
        except Exception as e:
            import traceback; traceback.print_exc()
