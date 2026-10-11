"""Script-24 figures, redrawn from the saved publication tables (no estimation)."""
import sys
import yaml
import pandas as pd
from common import REPO, load, restyle, resize, save, measure, capture

s24 = load("24_make_figures_tables.py", "stage24")
config = yaml.safe_load(open(REPO / "configs/publication_outputs.yaml", encoding="utf-8"))
s24.configure_plot_style(config)
st = config["figures"]
colors = {"primary": st["color_primary"], "secondary": st["color_secondary"],
          "robustness": st["color_robustness"], "warning": st["color_warning"]}
T = lambda label: pd.read_csv(REPO / f"outputs/tables/publication_v001_{label}.csv")
traj = config["forecast_trajectory_figures"]

BUILD = {
    "primary_forecast_effects": lambda: s24.plot_primary_effects(T("T04_primary_forecast_inference"), colors),
    "ablation_sensitivity": lambda: s24.plot_ablation(T("S01_full_ablation"), colors),
    "var_impulse_responses": lambda: s24.plot_var_irf(
        pd.read_csv(REPO / "outputs/time_series_causality/causality_v001_var_impulse_responses.csv"), colors),
    "local_projection_paths": lambda: s24.plot_local_projections(T("T06_local_projections"), colors),
    "granger_multiple_testing": lambda: s24.plot_granger_correction_shift(T("T07_granger_tests"), colors),
    "dml_predictive_effects": lambda: s24.plot_dml(T("T09_dml_effects_diagnostics"), colors),
    "monthly_forecast_errors": lambda: s24.plot_monthly_forecast_errors(T("S08_one_step_forecasts"), traj, colors),
}

HEIGHT = {
    "primary_forecast_effects": 3.2, "ablation_sensitivity": 3.6, "var_impulse_responses": 2.8,
    "local_projection_paths": 2.8, "granger_multiple_testing": 4.6, "dml_predictive_effects": 3.2,
    "monthly_forecast_errors": 7.0,
}

def build(name):
    s24.configure_plot_style(config)  # re-apply per figure: run_all resets rcParams between figures
    with capture():
        fig = BUILD[name]()
    return fig

def finish(name, fig, layout=True):
    resize(fig, HEIGHT[name])
    restyle(fig)
    if layout:
        fig.tight_layout()
    measure(save(fig, name))

if __name__ == "__main__":
    for name in sys.argv[1:] or BUILD:
        finish(name, build(name))
