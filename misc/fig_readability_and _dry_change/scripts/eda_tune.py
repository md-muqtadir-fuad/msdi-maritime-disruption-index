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
    wrap_titles(fig, 40)
    years(fig.axes[3], step=2)
    constrained(fig)


def t_empirical(fig):
    for ax in fig.axes:
        t = ax.get_title()
        if " ($" in t:
            name, rho = t.split(" ($", 1)
            ax.title.set_text(f"{name}\n(${rho}")
        if ax.get_ylabel():
            pass
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


def post(name, fig):
    if name == "persistent_event_shocks_timeline":
        for ax in fig.axes:
            keep_inside(ax)
    if name == "dimension_pca":
        ax = fig.axes[1]
        labels = [t for t in ax.texts if t.get_text()]
        repel_y(ax, labels, min_gap_pt=10)
        if ax.get_legend():
            clear_of(ax, labels, ax.get_legend())


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
