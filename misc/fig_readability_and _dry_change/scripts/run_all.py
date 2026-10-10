"""Redraw every manuscript figure at text width (6.5 in) with >= 8 pt text, from saved
results only (no estimation), into ../output_figs/. With --install, back up the current
latex/ PDFs to ../original_latex_pdfs/<timestamp>/ and copy the redrawn PDFs over them.

    .venv/Scripts/python.exe run_all.py                 # redraw all, measure
    .venv/Scripts/python.exe run_all.py dml_predictive_effects   # redraw one
    .venv/Scripts/python.exe run_all.py --install       # redraw all, back up, install

Run with the repo's .venv interpreter. Nothing under outputs/, data/ or validation/ is
written: every original plotting call runs inside common.capture(), and the run ends
with a before/after file check of those folders.
"""
import shutil, sys, time
from pathlib import Path

import common
from common import OUT, REPO, measure, snapshot

FIGURES = {
    # script 24 (publication tables)
    "primary_forecast_effects": "fig24", "ablation_sensitivity": "fig24",
    "var_impulse_responses": "fig24", "local_projection_paths": "fig24",
    "granger_multiple_testing": "fig24", "dml_predictive_effects": "fig24",
    "monthly_forecast_errors": "fig24",
    # scripts 47, 48, 46
    "monthly_model_forecasts": "mid", "pcmci_links": "mid",
    "interaction_strength": "mid", "pairwise_pdp_grid": "mid",
    # scripts 31, 39, 40, 43, 44
    "eda_correlation": "eda", "eda_distributions": "eda", "empirical_relationships": "eda",
    "persistent_event_shocks_timeline": "eda", "msdi_vs_external_benchmarks": "eda",
    "dimension_pca": "eda", "sample_size_convergence": "eda",
    # script 49
    "shanghai_vs_other_ports": "map",
}
# frame-work.pdf is a drawn diagram with no source in the repo; it is not regenerated.


def redraw(name):
    # each figure gets only its own script's style; without this, rcParams set by an
    # earlier figure (e.g. script 24's whitegrid style) leak into later ones
    import matplotlib
    matplotlib.rcdefaults()
    kind = FIGURES[name]
    if kind == "fig24":
        import fig24_tune
        fig24_tune.run(name)
    elif kind == "mid":
        import fig_mid
        measure(common.fit_save(fig_mid.BUILD[name](), name))
    elif kind == "eda":
        import pickle, fig_eda, eda_tune
        with open(common.CACHE / f"{name}.raw.pkl", "wb") as fh:
            pickle.dump(fig_eda.BUILD[name](), fh)
        eda_tune.run(name)
    elif kind == "map":
        import portmap
        common.load("49_shanghai_vs_other_ports.py", "s49")  # script's rcParams
        measure(common.fit_save(portmap.build(portmap.get_table()), name))


def install(names):
    backup = OUT.parent / "original_latex_pdfs" / time.strftime("%Y%m%d-%H%M%S")
    backup.mkdir(parents=True)
    for name in names:
        src, dst = OUT / f"{name}.pdf", REPO / "latex" / f"{name}.pdf"
        if dst.exists():
            shutil.copy2(dst, backup / dst.name)
        shutil.copy2(src, dst)
    print(f"installed {len(names)} PDFs into latex/; originals backed up to {backup}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = args or list(FIGURES)
    before = snapshot("outputs", "data", "validation")
    for name in names:
        redraw(name)
    after = snapshot("outputs", "data", "validation")
    changed = [k for k in set(before) | set(after) if before.get(k) != after.get(k)]
    print(f"files changed under outputs/, data/, validation/: {len(changed)}")
    if changed:
        print("\n".join(changed[:20]))
        sys.exit("aborting: the redraw touched pipeline files")
    if "--install" in sys.argv:
        install(names)
