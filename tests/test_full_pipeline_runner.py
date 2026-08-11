from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


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

    def test_event_embedding_stage_is_external_kaggle_only(self) -> None:
        stage = next(item for item in MODULE.STAGES if item.number == 8)
        self.assertEqual(stage.mode, "external_kaggle")
        command, role = MODULE.stage_command(stage, object(), Path(sys.executable))
        self.assertIsNone(command)
        self.assertEqual(role, "external_kaggle_artifacts_present")

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

    def test_atomic_manifest_write_retries_transient_windows_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "pipeline_run.json"
            destination.write_text('{"status": "old"}\n', encoding="utf-8")
            real_replace = MODULE.os.replace
            attempts = 0

            def flaky_replace(source: Path, target: Path) -> None:
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError(5, "transient file lock", str(target))
                real_replace(source, target)

            with patch.object(MODULE.os, "replace", side_effect=flaky_replace), patch.object(
                MODULE.time, "sleep"
            ):
                MODULE.atomic_write_json(destination, {"status": "new"})

            self.assertEqual(attempts, 3)
            self.assertEqual(
                MODULE.json.loads(destination.read_text(encoding="utf-8")),
                {"status": "new"},
            )
            self.assertEqual(list(destination.parent.glob(".*.tmp")), [])

    def test_manifest_audit_resolves_imported_kaggle_outputs_locally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "canonical_event_article_map.csv"
            output.write_text("article_id,event_id\na1,e1\n", encoding="utf-8")
            manifest = root / "canonical_event_manifest.json"
            manifest.write_text(
                MODULE.json.dumps(
                    {
                        "outputs": {
                            "article_event_map": (
                                "/kaggle/working/event_dedup/"
                                "canonical_event_article_map.csv"
                            )
                        }
                    }
                ),
                encoding="utf-8",
            )
            audit = MODULE.manifest_audit(manifest, deep_hash=False)
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["missing_outputs"], [])


if __name__ == "__main__":
    unittest.main()
