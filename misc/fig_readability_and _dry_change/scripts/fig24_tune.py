"""Per-figure layout fixes for the script-24 figures."""
import sys
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, NullFormatter
from common import restyle, resize, save, fit_save, measure, slim, shared_labels, legend_bottom, constrained
from fig24 import build, HEIGHT

def t_primary(fig):
    constrained(fig)

def t_ablation(fig):
    shared_labels(fig, x=fig.axes[0].get_xlabel())
    constrained(fig)

def t_bottom_legend(fig):
    legend_bottom(fig, ncol=3)
    constrained(fig)

def t_granger(fig):
    for ax in fig.axes:
        if ax.get_xscale() == "log":
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
        leg = ax.get_legend()
        if leg:  # inside, it ran across the alpha line and the VARX rows; put it under the axis
            h, l = leg.legend_handles, [t.get_text() for t in leg.get_texts()]
            leg.remove()
            ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=4,
                      frameon=False, columnspacing=1.2, handlelength=1.8)
    return "tight"

def t_dml(fig):
    return "tight"

def t_errors(fig):
    shared_labels(fig, x=fig.axes[-1].get_xlabel(), y=fig.axes[0].get_ylabel())
    for ax in fig.axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(0); lbl.set_ha("center")
    legend_bottom(fig, ncol=2, loc="outside upper center")
    constrained(fig)

def rebuild_brackets(fig, gap_pt=6.0, label_gap_pt=3.0, edge_pt=4.0):
    """Script 24 places its row-group brackets at fixed axes fractions (-0.235, -0.315),
    which at text width land on the tick labels. Redraw them at fixed point distances
    left of the tick labels, one level per original x position, same labels and style."""
    from matplotlib.transforms import ScaledTranslation
    to_pt = 72.0 / fig.dpi
    for ax in fig.axes:
        fig.canvas.draw()
        old_lines = [l for l in ax.lines if len(l.get_xdata()) == 2 and max(l.get_xdata()) < 0]
        old_texts = [t for t in ax.texts if abs(t.get_rotation() - 90) < 1 and t.get_position()[0] < 0]
        verticals = [l for l in old_lines if l.get_xdata()[0] == l.get_xdata()[1]]
        if not verticals:
            continue
        color, lw = verticals[0].get_color(), verticals[0].get_linewidth()
        groups = []
        for l in verticals:
            x, (low, high) = l.get_xdata()[0], sorted(l.get_ydata())
            label = next(t for t in old_texts if abs(t.get_position()[0] - (x - 0.012)) < 1e-6
                         and low <= t.get_position()[1] <= high)
            groups.append((x, low, high, label))
        spine = ax.get_window_extent().x0
        labels_left = max((spine - t.get_window_extent().x0
                           for t in ax.get_yticklabels() if t.get_text()), default=0.0) * to_pt
        level_off, offset = {}, labels_left + gap_pt
        for x in sorted({g[0] for g in groups}, reverse=True):   # level nearest the axis first
            level_off[x] = offset
            size = max(g[3].get_fontsize() for g in groups if g[0] == x)
            offset += label_gap_pt + 1.25 * size + gap_pt
        for artist in old_lines + old_texts:
            artist.remove()
        # x = 0 is the left spine; each artist is shifted left by a fixed number of points
        shift = lambda pt: ax.get_yaxis_transform() + ScaledTranslation(-pt / 72, 0, fig.dpi_scale_trans)
        for x, low, high, label in groups:
            off = level_off[x]
            ax.plot([0, 0], [low, high], transform=shift(off), color=color, lw=lw, clip_on=False)
            # end ticks pointing toward the axis, drawn as horizontal markers edge_pt wide
            ax.plot([0, 0], [low, high], transform=shift(off - edge_pt / 2), linestyle="none",
                    marker="_", markersize=edge_pt, markeredgewidth=lw, color=color, clip_on=False)
            ax.text(0, (low + high) / 2, label.get_text(), transform=shift(off + label_gap_pt),
                    ha="right", va="center", rotation=90, fontsize=label.get_fontsize(),
                    color=label.get_color(), fontfamily=label.get_fontfamily(), clip_on=False)


POST = {"granger_multiple_testing": rebuild_brackets, "dml_predictive_effects": rebuild_brackets}


TUNE = {
    "primary_forecast_effects": t_primary,
    "ablation_sensitivity": t_ablation,
    "var_impulse_responses": t_bottom_legend,
    "local_projection_paths": t_bottom_legend,
    "granger_multiple_testing": t_granger,
    "dml_predictive_effects": t_dml,
    "monthly_forecast_errors": t_errors,
}

def run(name):
    fig = build(name)
    resize(fig, HEIGHT[name])
    slim(fig)
    mode = TUNE[name](fig)
    restyle(fig)
    if name in POST:
        POST[name](fig)
    if mode == "tight":
        fig.tight_layout()
    return measure(fit_save(fig, name))

if __name__ == "__main__":
    for name in sys.argv[1:] or TUNE:
        run(name)
