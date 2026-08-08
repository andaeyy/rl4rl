from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openevolve.config import Config, load_config
from openevolve.database import Program, ProgramDatabase

from agents.openevolve_semantic.semantic_archive import install_semantic_archive
from common.evaluator import file_hash
from private_eval.regression import evaluate_pretrained_baseline_regression
from scripts.validate_engineering_canaries import validate_controller_surfaces


def _database_smoke(kind: str, result) -> dict:
    agent_dir = ROOT / "agents" / f"openevolve_{kind}"
    config: Config = load_config(agent_dir / "config.yaml")
    if kind == "semantic":
        install_semantic_archive()
    database = ProgramDatabase(config.database)
    candidate = Program(
        id=str(uuid.uuid4()),
        code=(ROOT / "common" / "initial_candidate.ir.json").read_text(),
        metrics=result.metrics(),
        iteration_found=0,
    )
    database.add(candidate, iteration=0, target_island=0)
    coords = database._calculate_feature_coords(candidate)
    return {
        "program_id": candidate.id,
        "feature_dimensions": config.database.feature_dimensions,
        "coordinates": coords,
        "stored": candidate.id in database.programs,
    }


def _pretrained_regression_fixture(source: Path, result, run_id: str) -> dict:
    """Describe a decoder regression without impersonating controller lineage."""

    return {
        "schema_name": "OfflinePretrainedDecoderRegressionFixture",
        "schema_version": "1.0",
        "run_id": run_id,
        "scientific": False,
        "controller_lineage": False,
        "retention_decision": "not_applicable",
        "candidate_source_sha256": file_hash(source),
        "parent_id": None,
        "evaluation": result.to_dict(),
    }


def main() -> None:
    os.environ["DISCOVERY_EVAL_NUM_TESTS"] = "16"
    os.environ["DISCOVERY_SHADOW_NUM_TESTS"] = "16"
    pretrained_source = ROOT / "common" / "initial_candidate.py"
    result = evaluate_pretrained_baseline_regression(
        official_count=16, shadow_count=16, device="cpu"
    )
    if not result.qualifies:
        raise SystemExit("shared baseline failed")
    controller_surfaces = validate_controller_surfaces(ROOT)

    smoke_run_id = f"offline-smoke-{uuid.uuid4().hex[:8]}"
    smoke_root = ROOT / "outputs" / "raw" / "offline_smoke" / smoke_run_id
    smoke_root.mkdir(parents=True, exist_ok=True)

    regression_fixture = _pretrained_regression_fixture(
        pretrained_source,
        result,
        smoke_run_id,
    )
    (smoke_root / "pretrained_regression_fixture.json").write_text(
        json.dumps(regression_fixture, indent=2, sort_keys=True) + "\n"
    )

    report = {
        "mode": "offline_pretrained_regression_and_static_surface_checks",
        "provider_calls": 0,
        "candidate_training_runs": 0,
        "scientific": False,
        "baseline": result.to_dict(),
        "pretrained_decoder_regression_fixture": regression_fixture,
        "openevolve_generic": _database_smoke("generic", result),
        "openevolve_semantic": _database_smoke("semantic", result),
        "semantic_autoresearch_static_surface": next(
            item
            for item in controller_surfaces["harnesses"]
            if item["harness_id"] == "semantic_autoresearch"
        ),
        "four_harness_static_surfaces": controller_surfaces,
        "provider_credentials_present": all(
            os.environ.get(name)
            for name in (
                "DISCOVERY_API_KEY",
                "DISCOVERY_API_BASE",
                "DISCOVERY_MODEL",
            )
        ),
    }
    destination = smoke_root / "smoke_report.json"
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    destination.write_text(report_text)
    (ROOT / "outputs" / "raw" / "latest_smoke_report.json").write_text(report_text)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not controller_surfaces["passed"]:
        raise SystemExit(
            "four-harness static surface metadata is incomplete; smoke failed"
        )


if __name__ == "__main__":
    main()
