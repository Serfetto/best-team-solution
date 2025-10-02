#!/usr/bin/env python3
"""Send batch JSON to the FastAPI service once and store predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import requests

DEFAULT_API = "http://127.0.0.1:8000"
DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "samples" / "batch_reviews.json"
DEFAULT_OUTPUT = Path("batch_predictions.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run batch analysis via /analyze/batch endpoint")
    parser.add_argument("--api", default=DEFAULT_API, help="Base URL of the FastAPI service")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to JSON with structure {\"data\": [...]}")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to write response JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    url = f"{args.api.rstrip('/')}/analyze/batch"
    response = requests.post(url, json=payload, timeout=600)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        print(f"Request failed: {exc} -> {response.text}", file=sys.stderr)
        return 1

    result = response.json()
    output_path = Path(args.output)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = result.get("metrics", {})
    print("Saved predictions to", output_path)
    if metrics:
        print(
            "Metrics: duration={duration_seconds:.2f}s processed={processed} llm_calls={llm_calls} cost_total={cost_total}".format(
                duration_seconds=metrics.get("duration_seconds", 0.0),
                processed=metrics.get("processed", 0),
                llm_calls=metrics.get("llm_calls", 0),
                cost_total=metrics.get("cost_total", 0.0),
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
