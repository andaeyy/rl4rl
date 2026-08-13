#!/usr/bin/env python3
"""Trusted post-search Layer B/C orchestration entrypoint.

The production surface is :class:`SealedPostSearchOrchestrator`. This command
performs a deliberately provider-free import/preflight only; scientific inputs
must be supplied by a separate, reviewed operator integration. No checked-in
defaults can authorize sealed evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.study_sink import ImmutableStudyEventSink  # noqa: E402
from sealed_eval.orchestration import SealedPostSearchOrchestrator  # noqa: E402
from sealed_eval.qualification import LayerBQualificationRunner  # noqa: E402
from sealed_eval.snapshot import freeze_completed_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed preflight for trusted post-search orchestration."
    )
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight is not True:
        parser.error(
            "sealed execution requires an external reviewed integration and "
            "explicit Layer B/C authorization inputs"
        )
    components = (
        LayerBQualificationRunner,
        freeze_completed_run,
        ImmutableStudyEventSink,
        SealedPostSearchOrchestrator,
    )
    print(
        json.dumps(
            {
                "mode": "sealed_post_search_preflight",
                "provider_calls": 0,
                "training_runs": 0,
                "scientific_authorized": False,
                "components": [component.__name__ for component in components],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
