from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/99_run_full_pipeline.py"
SPEC = importlib.util.spec_from_file_location("full_pipeline_runner", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FullPipelineRunnerTests(unittest.TestCase):
    def test_registry_is_strictly_sequential(self) -> None:
        self.assertEqual([stage.number for stage in MODULE.STAGES], list(range(25)))
        self.assertEqual(len({stage.script for stage in MODULE.STAGES}), 25)
        for stage in MODULE.STAGES:
            self.assertTrue(stage.script.startswith(f"{stage.number:02d}_"))
            self.assertTrue(
                all(
                    dependency < stage.number
                    for dependency in MODULE.STAGE_DEPENDENCIES[stage.number]
                )
            )

    def test_full_forecast_stage_enables_all_models_and_tuning(self) -> None:
        stage = next(item for item in MODULE.STAGES if item.number == 18)
        command = stage.command(Path("python"))
        self.assertIn("--tune", command)
        self.assertIn("--resume", command)
        models = command[command.index("--models") + 1].split(",")
        self.assertEqual(models, list(MODULE.EXPECTED_FORECAST_ALGORITHMS))

    def test_smoke_plan_is_check_only(self) -> None:
        report = MODULE.build_smoke_report(deep_hash=False)
        self.assertTrue(report["read_only"])
        for item in report["planned_full_commands"]:
            command = item["command"] or []
            self.assertNotIn("--smoke-test", command)

    def test_glm_contract_is_gaussian_identity(self) -> None:
        _, configs = MODULE.config_checks()
        checks = {
            row["check"]: row for row in MODULE.glm_contract_checks(configs)
        }
        self.assertTrue(checks["glm_gaussian_identity_contract"]["passed"])
        self.assertTrue(checks["glm_tuning_grid"]["passed"])

    def test_config_and_script_integrity_locks_match(self) -> None:
        self.assertTrue(
            all(row["passed"] for row in MODULE.canonical_config_hash_checks())
        )
        self.assertTrue(all(row["passed"] for row in MODULE.integrity_lock_checks()))


if __name__ == "__main__":
    unittest.main()
