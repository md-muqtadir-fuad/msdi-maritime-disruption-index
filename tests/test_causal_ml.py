from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/08b_causal_ml.py"
SPEC = importlib.util.spec_from_file_location("causal_ml", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def weekly_spec() -> MODULE.FrequencySpecification:
    return MODULE.FrequencySpecification(
        frequency="weekly",
        role="primary",
        panel_path=Path("unused.csv"),
        date_column="observation_date",
        scfi_column="scfi",
        brent_column="brent_price",
        article_count_column="article_count",
        source_count_column="source_count",
        msdi_columns={
            "peak": "peak_score_media_corrected",
            "breadth": "breadth_score_media_corrected",
        },
        horizons=(1, 2, 4),
        confounder_lags=(1, 2),
        minimum_training_rows=20,
        refit_every=5,
        volatility_window=4,
        minimum_complete_date=None,
    )


class CausalMLTests(unittest.TestCase):
    def test_config_freezes_claim_timing_primary_path_and_learner(self) -> None:
        config = MODULE.load_yaml(MODULE.DEFAULT_CONFIG)
        self.assertFalse(config["identification"]["causal_claim_allowed"])
        self.assertEqual(config["treatment"]["primary_path"], "peak")
        self.assertEqual(config["learners"]["primary"], "elastic_net")
        self.assertEqual(
            config["outcome"]["formula"],
            "log_scfi_t_plus_h_minus_log_scfi_t_minus_1",
        )
        self.assertFalse(config["heterogeneity"]["causal_forest_enabled"])

    def test_analysis_frame_has_exact_treatment_and_outcome_timing(self) -> None:
        n = 40
        dates = pd.date_range("2020-01-06", periods=n, freq="W-MON")
        log_scfi = np.arange(n, dtype=float) / 10
        panel = pd.DataFrame(
            {
                "observation_date": dates,
                "scfi": np.exp(log_scfi),
                "brent_price": np.exp(3 + np.arange(n) / 100),
                "article_count": np.arange(n) + 1,
                "source_count": np.arange(n) % 5 + 1,
                "peak_score_media_corrected": np.arange(n, dtype=float) ** 2,
                "breadth_score_media_corrected": np.arange(n, dtype=float),
            }
        )
        frame, features = MODULE.prepare_analysis_frame(panel, weekly_spec(), "peak")
        t = 10
        self.assertAlmostEqual(
            frame.at[t, "msdi_change"],
            panel.at[t, "peak_score_media_corrected"]
            - panel.at[t - 1, "peak_score_media_corrected"],
        )
        self.assertAlmostEqual(
            frame.at[t, "outcome_h2"], log_scfi[t + 2] - log_scfi[t - 1]
        )
        self.assertAlmostEqual(
            frame.at[t, "placebo_outcome_h2"], log_scfi[t - 1] - log_scfi[t - 3]
        )
        self.assertIn("msdi_level_lag_1", features)
        self.assertEqual(
            frame.at[t, "msdi_level_lag_1"],
            panel.at[t - 1, "peak_score_media_corrected"],
        )
        self.assertFalse(any(name.startswith("outcome_") for name in features))

    def test_common_expanding_folds_are_strictly_forward_and_embargoed(self) -> None:
        n = 100
        frame = pd.DataFrame(
            {
                "position": np.arange(n),
                "msdi_change": np.sin(np.arange(n)),
                "outcome_h4": np.cos(np.arange(n)),
                "placebo_outcome_h4": np.sin(np.arange(n) / 2),
                "x": np.arange(n, dtype=float),
            }
        )
        folds = MODULE.build_expanding_folds(
            frame,
            horizon=4,
            feature_columns=["x"],
            minimum_training_rows=20,
            evaluation_block_size=7,
            maximum_evaluation_rows=28,
        )
        self.assertEqual(len(folds), 4)
        prior_training = 0
        seen: set[int] = set()
        for fold in folds:
            self.assertGreaterEqual(len(fold.training_positions), prior_training)
            prior_training = len(fold.training_positions)
            self.assertLess(
                int(fold.training_positions[-1]) + 4,
                int(fold.evaluation_positions[0]),
            )
            self.assertFalse(set(fold.training_positions).intersection(fold.evaluation_positions))
            self.assertFalse(seen.intersection(fold.evaluation_positions))
            seen.update(int(item) for item in fold.evaluation_positions)
            np.testing.assert_array_equal(
                np.diff(fold.evaluation_positions),
                np.ones(len(fold.evaluation_positions) - 1, dtype=int),
            )

    def test_orthogonal_estimate_matches_closed_form_and_statsmodels_hac(self) -> None:
        rng = np.random.default_rng(44)
        d = rng.normal(size=240)
        y = 0.65 * d + rng.normal(scale=0.5, size=240)
        scale = 1.7
        result = MODULE.orthogonal_estimate(
            y,
            d,
            alpha=0.05,
            hac_lags=4,
            bootstrap_repetitions=0,
            block_length=4,
            seed=9,
            effect_scale=scale,
        )
        expected_theta = float(np.dot(d, y) / np.dot(d, d))
        reference = sm.OLS(y, d[:, None]).fit(
            cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True}
        )
        self.assertAlmostEqual(
            result["coefficient_per_raw_treatment_unit"], expected_theta, places=12
        )
        self.assertAlmostEqual(result["raw_standard_error_hac"], reference.bse[0], places=12)
        self.assertAlmostEqual(
            result["effect_log_points_per_one_sd_innovation"],
            expected_theta * scale,
            places=12,
        )
        self.assertAlmostEqual(result["standard_error_hac"], reference.bse[0] * scale, places=12)

    def test_calibration_scale_is_past_only_and_future_mutation_invariant(self) -> None:
        rng = np.random.default_rng(12)
        n = 120
        frame = pd.DataFrame(
            {
                "position": np.arange(n),
                "date": pd.date_range("2018-01-01", periods=n, freq="W-MON"),
                "msdi_change": rng.normal(size=n),
                "x": rng.normal(size=n),
            }
        )
        parameters = {"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 20000}
        first, audit = MODULE.calibrate_treatment_scale(
            frame=frame,
            feature_columns=["x"],
            cutoff_position_exclusive=80,
            spec=weekly_spec(),
            path_name="peak",
            learner="elastic_net",
            learner_parameters=parameters,
            seed=5,
        )
        mutated = frame.copy()
        mutated.loc[80:, ["msdi_change", "x"]] = 10_000
        second, _ = MODULE.calibrate_treatment_scale(
            frame=mutated,
            feature_columns=["x"],
            cutoff_position_exclusive=80,
            spec=weekly_spec(),
            path_name="peak",
            learner="elastic_net",
            learner_parameters=parameters,
            seed=5,
        )
        self.assertAlmostEqual(first, second, places=14)
        self.assertTrue(audit["validation_precedes_first_evaluation"])
        self.assertAlmostEqual(first, audit["frozen_residual_standard_deviation"])

    def test_level_innovation_reparameterization_identity(self) -> None:
        level = np.array([2.0, 4.5, 3.0])
        lagged = np.array([1.0, 2.0, 4.5])
        predicted_change = np.array([0.4, 1.2, -0.2])
        change = level - lagged
        level_innovation = level - (lagged + predicted_change)
        change_innovation = change - predicted_change
        np.testing.assert_allclose(level_innovation, change_innovation, atol=1e-15)

    def test_common_sample_and_horizon_invariant_treatment(self) -> None:
        rows = []
        dates = pd.date_range("2023-01-02", periods=15, freq="W-MON")
        for horizon in (1, 4):
            for path in ("peak", "breadth"):
                for learner in ("elastic_net", "random_forest"):
                    for i, date in enumerate(dates):
                        rows.append(
                            {
                                "frequency": "weekly",
                                "frequency_role": "primary",
                                "msdi_path": path,
                                "horizon": horizon,
                                "learner": learner,
                                "date": date,
                                "fold": 1 if i < 8 else 2,
                                "treatment_residual_oof": float(i) / 10,
                                "treatment_prediction": float(i) / 20,
                            }
                        )
        common, audit = MODULE.enforce_common_samples(
            pd.DataFrame(rows),
            ["peak", "breadth"],
            ["elastic_net", "random_forest"],
        )
        self.assertTrue(audit["balanced_common_sample_passed"].all())
        invariant = MODULE.treatment_horizon_invariance_audit(common)
        self.assertTrue(invariant["treatment_nuisance_horizon_invariance_passed"].all())
        broken = common.copy()
        mask = broken["horizon"].eq(4) & broken["date"].eq(dates[0])
        broken.loc[mask, "treatment_residual_oof"] += 1
        with self.assertRaises(AssertionError):
            MODULE.treatment_horizon_invariance_audit(broken)

    def test_primary_multiplicity_family_is_weekly_peak_only(self) -> None:
        frame = pd.DataFrame(
            {
                "frequency": ["weekly"] * 4 + ["monthly"],
                "msdi_path": ["peak", "peak", "breadth", "peak", "peak"],
                "learner": [
                    "elastic_net",
                    "elastic_net",
                    "elastic_net",
                    "random_forest",
                    "elastic_net",
                ],
                "p_value": [0.01, 0.04, 0.02, 0.03, 0.01],
            }
        )
        result = MODULE.add_effect_multiplicity(frame, "elastic_net")
        self.assertEqual(result["primary_family"].tolist(), [True, True, False, False, False])
        self.assertTrue(result.loc[result["primary_family"], "p_value_holm_primary_family"].notna().all())

    def test_monthly_completeness_contract_drops_partial_edges(self) -> None:
        config = MODULE.load_yaml(MODULE.DEFAULT_CONFIG)
        spec = MODULE.frequency_specifications(config)["monthly"]
        panel = MODULE.load_panel(spec.panel_path, spec)
        filtered, audit = MODULE.apply_panel_completeness_contract(panel, spec)
        self.assertEqual(filtered[spec.date_column].min(), pd.Timestamp("2016-03-31"))
        self.assertEqual(filtered[spec.date_column].max(), pd.Timestamp("2025-12-31"))
        self.assertEqual(len(filtered), 118)
        self.assertEqual(audit["rows_removed"], 3)

    def test_current_script08_hash_contract(self) -> None:
        audit = MODULE.validate_script08_contract(
            MODULE.DEFAULT_CAUSALITY_MANIFEST,
            MODULE.DEFAULT_INNOVATIONS,
            MODULE.DEFAULT_LOCAL_PROJECTIONS,
        )
        self.assertTrue(
            all(item["passed"] for item in audit["hash_checks"].values())
        )

    def test_partial_and_smoke_scopes_never_receive_publication_status(self) -> None:
        args = SimpleNamespace(smoke_test=False, maximum_evaluation_rows=None)
        self.assertTrue(
            MODULE.publication_scope_complete(
                args=args,
                requested_frequencies=["weekly", "monthly"],
                requested_paths=["peak", "breadth"],
                requested_learners=list(MODULE.LEARNERS),
                bootstrap_repetitions=999,
            )
        )
        args.smoke_test = True
        self.assertFalse(
            MODULE.publication_scope_complete(
                args=args,
                requested_frequencies=["weekly", "monthly"],
                requested_paths=["peak", "breadth"],
                requested_learners=list(MODULE.LEARNERS),
                bootstrap_repetitions=999,
            )
        )

    def test_cli_smoke_both_frequencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [
                str(PROJECT_ROOT / ".venv/Scripts/python.exe"),
                str(SCRIPT),
                "--smoke-test",
                "--paths",
                "peak",
                "--learners",
                "elastic_net",
                "--bootstrap-repetitions",
                "0",
                "--version",
                "causal_ml_test_smoke",
                "--output-dir",
                str(root / "outputs"),
                "--log-dir",
                str(root / "logs"),
            ]
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stdout + completed.stderr)
            manifest_path = root / "outputs/causal_ml_test_smoke_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "debug_or_partial_run_not_for_publication")
            self.assertFalse(manifest["causal_claim_allowed"])
            self.assertTrue(manifest["audit"]["all_horizon_purges_passed"])


if __name__ == "__main__":
    unittest.main()
