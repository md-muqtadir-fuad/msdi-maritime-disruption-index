"""Run the complete nine-point EDA protocol in a fixed, auditable order."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EDA_ROOT = ROOT / "eda"
DEFAULT_OUTPUT = EDA_ROOT / "outputs"

STAGES = [
    (1, "Raw-data integrity", "01_raw_data_integrity.py"),
    (2, "Variable construction", "02_variable_construction.py"),
    (3, "Temporal properties", "03_temporal_properties.py"),
    (4, "Availability and leakage", "04_availability_leakage.py"),
    (5, "Relationship with SCFI", "05_scfi_relationships.py"),
    (6, "Relationships among controls", "06_predictor_redundancy.py"),
    (7, "MSDI-specific validation", "07_msdi_validation.py"),
    (8, "Forecasting readiness", "08_forecasting_readiness.py"),
    (9, "Causal-readiness audit", "09_causal_readiness.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA protocol Points 1 through 9.")
    parser.add_argument("--mode", choices=("test", "full"), default="test")
    parser.add_argument("--from-stage", type=int, choices=range(1, 10), default=1)
    parser.add_argument("--to-stage", type=int, choices=range(1, 10), default=9)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Development only: run available stages and report missing stages without failing.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--list-stages", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expected_venv_python() -> Path:
    windows = ROOT / ".venv/Scripts/python.exe"
    posix = ROOT / ".venv/bin/python"
    return windows if windows.is_file() else posix


def validate_python() -> None:
    expected = expected_venv_python()
    if not expected.is_file():
        raise FileNotFoundError(f"Project virtual-environment Python is missing: {expected}")
    if Path(sys.executable).resolve() != expected.resolve():
        raise RuntimeError(
            "EDA must run with the project .venv. Use: "
            r".\.venv\Scripts\python.exe eda\99_run_eda.py --mode test"
        )


def selected_stages(first: int, last: int) -> list[tuple[int, str, str]]:
    if first > last:
        raise ValueError("--from-stage cannot be greater than --to-stage")
    return [stage for stage in STAGES if first <= stage[0] <= last]


def stage_command(number: int, filename: str, mode: str, output_root: Path) -> list[str]:
    script = EDA_ROOT / filename
    stage_output = output_root / f"point_{number:02d}"
    command = [sys.executable, str(script), "--output-dir", str(stage_output)]
    if mode == "test":
        command.append("--test-run")
    return command


def run_stage(command: list[str], log_path: Path) -> tuple[int, float]:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    return return_code, time.perf_counter() - started


def main() -> int:
    args = parse_args()
    validate_python()
    stages = selected_stages(args.from_stage, args.to_stage)

    if args.list_stages:
        for number, label, filename in STAGES:
            status = "available" if (EDA_ROOT / filename).is_file() else "missing"
            print(f"{number:02d}  {label:<32} {status:<9} {filename}")
        return 0

    missing = [filename for _, _, filename in stages if not (EDA_ROOT / filename).is_file()]
    if missing and not args.allow_incomplete:
        raise FileNotFoundError(
            "EDA protocol is incomplete; missing stage scripts: "
            + ", ".join(missing)
            + ". During development only, add --allow-incomplete."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "runner_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = args.output_dir / f"eda_run_{run_id}_manifest.json"
    run_started = utc_now()
    records = []
    overall_status = "dry_run" if args.dry_run else "complete"

    for number, label, filename in stages:
        script = EDA_ROOT / filename
        if not script.is_file():
            records.append(
                {
                    "stage": number,
                    "label": label,
                    "script": filename,
                    "status": "missing_skipped_development_only",
                }
            )
            overall_status = "incomplete_development_run"
            print(f"Stage {number:02d}: MISSING — {filename}")
            continue

        command = stage_command(number, filename, args.mode, args.output_dir)
        printable = subprocess.list2cmdline(command)
        print(f"Stage {number:02d}: {label}")
        print(printable)
        if args.dry_run:
            records.append(
                {
                    "stage": number,
                    "label": label,
                    "script": filename,
                    "status": "dry_run",
                    "command": printable,
                }
            )
            continue

        log_path = log_dir / f"{run_id}_{number:02d}_{script.stem}.log"
        return_code, runtime = run_stage(command, log_path)
        record = {
            "stage": number,
            "label": label,
            "script": filename,
            "status": "complete" if return_code == 0 else "failed",
            "return_code": return_code,
            "runtime_seconds": runtime,
            "command": printable,
            "log": str(log_path.relative_to(ROOT)),
        }
        records.append(record)
        if return_code != 0:
            overall_status = "failed"
            print(f"EDA stopped: Point {number} returned {return_code}.")
            break

    manifest = {
        "run_id": run_id,
        "mode": args.mode,
        "status": overall_status,
        "started_utc": run_started,
        "finished_utc": utc_now(),
        "python": sys.executable,
        "stage_range": [args.from_stage, args.to_stage],
        "allow_incomplete": args.allow_incomplete,
        "missing_stage_scripts": missing,
        "stages": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"EDA manifest: {manifest_path}")
    return 1 if overall_status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())

