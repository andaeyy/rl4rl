"""Command-line entry point for trajectory ingestion and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl4rl.adapters.autoresearch import parse_autoresearch_tsv
from rl4rl.io import load_events, write_events
from rl4rl.lineage import summarize_lineage
from rl4rl.metrics import summarize_metrics
from rl4rl.plots import write_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rl4rl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate canonical JSONL")
    validate.add_argument("path", type=Path)

    summarize = subparsers.add_parser("summarize", help="compute trajectory metrics")
    summarize.add_argument("path", type=Path)
    summarize.add_argument("--external-frontier", type=int)
    summarize.add_argument("--output", type=Path)

    parse = subparsers.add_parser(
        "parse-autoresearch", help="normalize an autoresearch TSV log"
    )
    parse.add_argument("path", type=Path)
    parse.add_argument("--run-id", required=True)
    parse.add_argument("--output", type=Path, required=True)

    plot = subparsers.add_parser("plot", help="write starter trajectory figures")
    plot.add_argument("path", type=Path)
    plot.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    plot.add_argument("--window", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        events = load_events(args.path)
        lineage = summarize_lineage(events)
        result = {
            "events": len(events),
            "runs": len({event.run_id for event in events}),
            "missing_parents": list(lineage.missing_parents),
            "max_depth": lineage.max_depth,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "summarize":
        events = load_events(args.path)
        result = summarize_metrics(
            events, external_frontier=args.external_frontier
        ).to_dict()
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
    elif args.command == "parse-autoresearch":
        events = parse_autoresearch_tsv(args.path, run_id=args.run_id)
        write_events(args.output, events)
        print(f"wrote {len(events)} events to {args.output}")
    elif args.command == "plot":
        events = load_events(args.path)
        paths = write_all(events, args.output_dir, window=args.window)
        for path in paths:
            print(path)
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
