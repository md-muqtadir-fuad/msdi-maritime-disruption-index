"""Scripts 31 (EDA), 39 (event timeline), 40 (external benchmarks), 43 (PCA),
44 (validation sample size), 49 (port map): original code run with writes blocked."""
import sys
from common import (REPO, OUT, CACHE, load, restyle, resize, save, measure, slim, shared_labels,
                    legend_bottom, constrained, capture)


def _s31(fn_name, basename):
    s31 = load("31_generate_publication_eda_figures.py", "s31")
    with capture() as caught:
        getattr(s31, fn_name)()
    return caught[basename]


def eda_correlation():
    return _s31("make_spearman_correlation_figure", "eda_figure01_spearman_correlation_lower_triangle")


def eda_distributions():
    return _s31("make_data_distributions_figure", "eda_figure02_data_distributions")


def empirical_relationships():
    return _s31("make_empirical_relationships_figure", "eda_figure04_empirical_relationships_grid")


def persistent_event_shocks_timeline():
    s39 = load("39_persistent_event_shocks_and_ripple_effects.py", "s39")
    w, m = s39.load_data()
    summary = s39.analyze_persistent_shocks(w, m)
    with capture() as caught:
        s39.plot_persistent_shocks_figure(m, summary)
    return next(iter(caught.values()))


def msdi_vs_external_benchmarks():
    s40 = load("40_external_disruption_benchmark.py", "s40")
    w, m = s40.load_panels()
    lead_lag = s40.compute_lead_lag_correlations(m, target_col="scfi_monthly_mean_log_change_1m", max_lags=12)
    with capture() as caught:
        s40.plot_external_benchmark_comparison(m, lead_lag, REPO / "outputs/figures")
    return next(iter(caught.values()))


def dimension_pca():
    s43 = load("43_dimension_pca.py", "s43")
    with capture() as caught:
        s43.main()
    return caught["eda_figure05_dimension_pca"]


def sample_size_convergence():
    s44 = load("44_validation_sample_size_figure.py", "s44")
    with capture() as caught:
        s44.main()
    return next(iter(caught.values()))


def shanghai_vs_other_ports():
    s49 = load("49_shanghai_vs_other_ports.py", "s49")
    with capture() as caught:
        s49.main()
    return caught["figure24_shanghai_vs_other_ports"]


BUILD = {f.__name__: f for f in (eda_correlation, eda_distributions, empirical_relationships,
                                  persistent_event_shocks_timeline, msdi_vs_external_benchmarks,
                                  dimension_pca, sample_size_convergence, shanghai_vs_other_ports)}

if __name__ == "__main__":
    import pickle
    for name in sys.argv[1:] or BUILD:
        fig = BUILD[name]()
        print(name, "original size", fig.get_size_inches())
        with open(CACHE / f"{name}.raw.pkl", "wb") as fh:
            pickle.dump(fig, fh)
