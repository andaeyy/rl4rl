from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from private_eval.regression import evaluate_pretrained_baseline_regression


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=64)
    args = parser.parse_args()
    result = evaluate_pretrained_baseline_regression(
        official_count=args.cases,
        shadow_count=args.cases,
        device="cpu",
    )
    output = ROOT / "outputs" / "raw" / "baseline_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if not result.qualifies:
        raise SystemExit("pretrained decoder regression did not pass")


if __name__ == "__main__":
    main()
