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
    if mode == "tight":
        fig.tight_layout()
    return measure(fit_save(fig, name))

if __name__ == "__main__":
    for name in sys.argv[1:] or TUNE:
        run(name)
