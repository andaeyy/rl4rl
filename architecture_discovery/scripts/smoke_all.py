from __future__ import annotations

import json
import os
import shutil
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
from common.lineage_schema import CandidateRecord, append_record, utc_now
from private_eval.regression import evaluate_pretrained_baseline_regression


def _database_smoke(kind: str, result) -> dict:
    agent_dir = ROOT / "agents" / f"openevolve_{kind}"
    config: Config = load_config(agent_dir / "config.yaml")
    if kind == "semantic":
        install_semantic_archive()
    database = ProgramDatabase(config.database)
    candidate = Program(
        id=str(uuid.uuid4()),
        code=(ROOT / "common" / "initial_candidate.py").read_text(),
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


def main() -> None:
    os.environ["DISCOVERY_EVAL_NUM_TESTS"] = "16"
    os.environ["DISCOVERY_SHADOW_NUM_TESTS"] = "16"
    source = ROOT / "common" / "initial_candidate.py"
    result = evaluate_pretrained_baseline_regression(
        official_count=16, shadow_count=16, device="cpu"
    )
    if not result.qualifies:
        raise SystemExit("shared baseline failed")

    smoke_run_id = f"offline-smoke-{uuid.uuid4().hex[:8]}"
    smoke_root = ROOT / "outputs" / "raw" / "offline_smoke" / smoke_run_id
    smoke_root.mkdir(parents=True, exist_ok=True)

    greedy_child = smoke_root / "greedy_noop.py"
    shutil.copy2(source, greedy_child)
    greedy_result = result
    greedy_record = CandidateRecord(
        run_id=smoke_run_id,
        condition="greedy_autoresearch",
        seed=0,
        candidate_id=file_hash(greedy_child),
        parent_id=file_hash(source),
        proposal_text="Offline no-op candidate for controller plumbing.",
        mechanism_hypothesis="No architectural change.",
        code_hash=file_hash(greedy_child),
        proposal_timestamp=utc_now(),
        completion_timestamp=utc_now(),
        evaluation=greedy_result.to_dict(),
        retention_decision="accept" if greedy_result.qualifies else "reject",
    )
    append_record(smoke_root / "greedy_lineage.jsonl", greedy_record)

    report = {
        "baseline": result.to_dict(),
        "greedy_autoresearch": {
            "qualifies": greedy_result.qualifies,
            "lineage_recorded": True,
            "mode": "offline_pretrained_regression_noop",
        },
        "openevolve_generic": _database_smoke("generic", result),
        "openevolve_semantic": _database_smoke("semantic", result),
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


if __name__ == "__main__":
    main()
