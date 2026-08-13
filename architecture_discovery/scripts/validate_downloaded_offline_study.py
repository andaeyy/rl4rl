"""Validate one downloaded Modal offline-study bundle without remote calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reconstruction.downloaded_offline import (  # noqa: E402
    DownloadedOfflineValidationError,
    validate_downloaded_offline_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only local validation, reconstruction, and reporting adaptation "
            "for one downloaded Modal offline-study run"
        )
    )
    parser.add_argument(
        "bundle",
        help="downloaded run directory containing artifact_manifest.json",
    )
    arguments = parser.parse_args()
    try:
        result = validate_downloaded_offline_bundle(arguments.bundle)
    except (DownloadedOfflineValidationError, OSError) as error:
        raise SystemExit(
            f"downloaded offline-study validation failed: {error}"
        ) from error
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
