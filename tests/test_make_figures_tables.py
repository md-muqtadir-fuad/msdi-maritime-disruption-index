from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/24_make_figures_tables.py"
SPEC = importlib.util.spec_from_file_location("make_figures_tables", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PublicationOutputUnitTests(unittest.TestCase):
    def test_frozen_config_and_script_integrity(self) -> None:
        config = MODULE.load_yaml(MODULE.DEFAULT_CONFIG)
        self.assertEqual(
            MODULE.sha256_file(MODULE.DEFAULT_CONFIG),
            MODULE.CANONICAL_CONFIG_SHA256,
        )
        lock = MODULE.resolve_project_path(
            config["outputs"]["script_sha256_lock"]
        )
        self.assertEqual(MODULE.read_sha256_lock(lock), MODULE.sha256_file(SCRIPT))

    def test_guardrails_are_fail_closed(self) -> None:
        guardrails = MODULE.load_yaml(MODULE.DEFAULT_CONFIG)["guardrails"]
        self.assertEqual(guardrails["primary_frequency"], "weekly")
        self.assertEqual(guardrails["monthly_role"], "robustness")
        self.assertTrue(guardrails["human_validation_required"])
        self.assertFalse(guardrails["publication_eligible_without_human_validation"])
        self.assertFalse(guardrails["causal_claim_allowed"])
        self.assertFalse(guardrails["economic_importance_claim_allowed"])

    def test_safe_zscore_and_boolean_parser(self) -> None:
        observed = MODULE.safe_zscore(pd.Series([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(observed.mean()), 0.0)
        self.assertAlmostEqual(float(observed.std(ddof=0)), 1.0)
        self.assertTrue(MODULE.safe_zscore(pd.Series([2.0, 2.0])).isna().all())
        parsed = MODULE.strict_bool_series(
            pd.Series([True, "false", "YES", 0]), "unit boolean"
        )
        self.assertEqual(parsed.tolist(), [True, False, True, False])
        with self.assertRaisesRegex(ValueError, "Unknown boolean"):
            MODULE.strict_bool_series(pd.Series(["maybe"]), "unit boolean")

    def test_sensitivity_summary_retains_frequency_roles(self) -> None:
        frame = pd.DataFrame(
            {
                "frequency": ["weekly", "monthly"],
                "reject_adjusted_at_alpha": [True, False],
                "p_value_holm": [0.01, 0.20],
                "rmse_improvement_percent": [2.0, -1.0],
            }
        )
        result = MODULE.summarize_sensitivity({"ablation": frame})
        roles = dict(zip(result["frequency"], result["frequency_role"]))
        self.assertEqual(roles, {"monthly": "robustness", "weekly": "primary"})
        self.assertEqual(
            int(
                result.loc[
                    result["frequency"] == "weekly",
                    "adjusted_rejections_any_registered_stream",
                ].iloc[0]
            ),
            1,
        )

    def test_publication_labels_are_concise_and_human_readable(self) -> None:
        expected = {
            "qwen_3_8b": "Qwen3-8B",
            "forecast_nnls_stack": "NNLS forecast stack",
            "glm": "Penalized Gaussian GLM",
            "random_forest": "Random forest",
            "jarque_bera": "Jarque–Bera",
        }
        for raw, label in expected.items():
            self.assertEqual(MODULE.publication_label(raw), label)
            self.assertNotIn("_", label)
        for raw in MODULE.ABLATION_LABELS:
            label = MODULE.ablation_label(raw)
            self.assertNotIn("_", label)
            self.assertLessEqual(len(label), 38)
            self.assertTrue(label[0].isupper())
        dynamic = MODULE.granger_label(
            pd.Series(
                {
                    "frequency": "monthly",
                    "msdi_path": "breadth",
                    "causing": "scfi_return",
                    "caused": "msdi_change",
                    "model": "conditional_ols_hac_prespecified_max_lag",
                }
            )
        )
        self.assertEqual(dynamic, "Monthly · Breadth · SCFI→MSDI · OLS–HAC")
        self.assertLessEqual(len(dynamic), 48)

    def test_no_overall_figure_titles_are_embedded(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn(".suptitle(", source)
        self.assertNotIn("add_guardrail(", source)

    def test_five_algorithms_use_dynamic_three_by_two_weekly_grid(self) -> None:
        figure, axes, rows, columns = MODULE.make_weekly_panel_grid(
            5,
            width=10.0,
            height_per_row=3.0,
            sharex=True,
            sharey=True,
        )
        try:
            MODULE.hide_unused_panel_axes(axes, 5)
            self.assertEqual((rows, columns), (3, 2))
            self.assertEqual(axes.shape, (3, 2))
            self.assertTrue(axes.flat[4].get_visible())
            self.assertFalse(axes.flat[5].get_visible())
        finally:
            MODULE.plt.close(figure)

    def test_residual_model_family_preserves_and_fills(self) -> None:
        frame = pd.DataFrame(
            {
                "analysis_source": [
                    "stage07d_secondary_nonlinear",
                    "stage07d_secondary_nonlinear",
                    "stage09_common_cohort_ridge",
                ],
                "model_id": [
                    "lstm__weekly__signal",
                    "rf__weekly__signal",
                    "peak_only",
                ],
                "model_family": ["upstream_lstm", None, ""],
            }
        )
        result = MODULE.complete_residual_model_family(frame)
        self.assertEqual(
            result["model_family"].tolist(),
            ["upstream_lstm", "random_forest", "ridge"],
        )

    def test_manifest_hash_can_be_unpinned_but_output_is_still_locked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "output.csv"
            output.write_text("value\n1\n2\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "version": "unit_v001",
                        "status": "complete",
                        "outputs": {"sample": str(output)},
                        "output_sha256": {
                            "sample": MODULE.sha256_file(output)
                        },
                        "output_row_counts": {"sample": 2},
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "source_manifests": {
                    "unit": {
                        "path": str(manifest),
                        "version": "unit_v001",
                        "status": "complete",
                        "required_outputs": ["sample"],
                    }
                }
            }
            logger = logging.getLogger("publication_manifest_unit")
            logger.handlers.clear()
            logger.addHandler(logging.NullHandler())
            audited, source_audit = MODULE.audit_sources(config, logger)
            self.assertEqual(
                audited["unit"]["manifest_hash_contract_source"], "observed_only"
            )
            self.assertEqual(
                source_audit.iloc[0]["hash_contract_source"], "manifest"
            )
            self.assertEqual(
                source_audit.iloc[0]["row_count_contract_source"],
                "manifest_output_row_counts",
            )

    def test_trajectory_panel_requires_both_registered_paths(self) -> None:
        frame = pd.DataFrame(
            {
                "target_date": ["2024-01-01", "2024-01-01"],
                "forecast_path": ["peak", "breadth"],
                "actual_scfi_level": [1000.0, 1000.0],
                "predicted_scfi_level": [990.0, 1010.0],
            }
        )
        panel = MODULE.trajectory_panel_data(frame)
        self.assertEqual(panel.columns.tolist(), ["actual", "peak", "breadth"])
        self.assertEqual(float(panel.iloc[0]["actual"]), 1000.0)
        with self.assertRaisesRegex(ValueError, "lacks path"):
            MODULE.trajectory_panel_data(frame.loc[frame["forecast_path"] == "peak"])


class PublicationOutputIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = MODULE.load_yaml(MODULE.DEFAULT_CONFIG)
        logger = logging.getLogger("publication_output_test")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        cls.audited, cls.source_audit = MODULE.audit_sources(cls.config, logger)
        MODULE.validate_global_guardrails(cls.config, cls.audited)
        cls.main, cls.supplements = MODULE.build_tables(
            cls.config, cls.audited, cls.source_audit
        )

    def test_all_frozen_source_contracts_pass(self) -> None:
        self.assertEqual(len(self.audited), 8)
        self.assertEqual(len(self.source_audit), 37)
        self.assertTrue(self.source_audit["contract_passed"].all())

    def test_publication_tables_preserve_registered_counts(self) -> None:
        primary = self.main["T04_primary_forecast_inference"]
        self.assertEqual(len(primary), 100)
        self.assertEqual(
            int(
                MODULE.strict_bool_series(
                    primary["adjusted_significant_positive_improvement"],
                    "primary improvement",
                ).sum()
            ),
            0,
        )
        self.assertEqual(
            int(
                MODULE.strict_bool_series(
                    primary["adjusted_significant_worsening"],
                    "primary worsening",
                ).sum()
            ),
            21,
        )
        self.assertEqual(len(self.supplements["S01_full_ablation"]), 1806)
        trajectories = self.supplements["S08_one_step_forecasts"]
        self.assertEqual(len(trajectories), 4780)
        np.testing.assert_allclose(
            trajectories["level_error_observed_minus_forecast"],
            trajectories["actual_scfi_level"]
            - trajectories["predicted_scfi_level"],
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertTrue((trajectories["absolute_level_error"] >= 0.0).all())
        self.assertEqual(
            len(
                trajectories[
                    [
                        "frequency",
                        "index_construction",
                        "algorithm",
                        "forecast_path",
                    ]
                ].drop_duplicates()
            ),
            30,
        )
        self.assertGreaterEqual(len(self.main), 14)

    def test_human_and_causal_claim_gates_remain_closed(self) -> None:
        status = self.main["T01_analysis_status"]
        human = status.loc[status["stage"] == "human_validation"].iloc[0]
        self.assertFalse(bool(human["publication_scope_complete"]))
        self.assertTrue(status["human_validation_required"].all())
        self.assertFalse(status["publication_eligible_without_human_validation"].any())
        self.assertFalse(status["causal_claim_allowed"].any())
        heterogeneity = self.main["T10_guarded_heterogeneity"]
        self.assertFalse(heterogeneity["causal_forest_estimate"].any())


if __name__ == "__main__":
    unittest.main()
