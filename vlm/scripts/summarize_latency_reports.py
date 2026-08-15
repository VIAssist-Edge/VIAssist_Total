from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_warm import save_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 latency report summary")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "reports/latency_optimization/summary.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "reports/latency_optimization/summary.md",
    )
    return parser.parse_args(argv)


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    samples = report["samples"]
    elevator = samples.get("elevator")
    escalator = samples.get("escalator")
    if elevator is None or escalator is None:
        raise ValueError("elevator와 escalator sample 결과가 모두 필요합니다.")
    all_records = elevator["records"] + escalator["records"]
    all_profiles = [record["profile"] for record in all_records]
    all_results = [record["result"] for record in all_records]
    pass_count = sum(not result["used_fallback"] for result in all_results)
    return {
        "variant": report["variant"],
        "elevator_mean_ms": elevator["timings_ms"]["total_ms"]["mean"],
        "escalator_mean_ms": escalator["timings_ms"]["total_ms"]["mean"],
        "p95_ms": max(
            elevator["timings_ms"]["total_ms"]["p95"],
            escalator["timings_ms"]["total_ms"]["p95"],
        ),
        "gpu_peak_mb": max(profile["gpu_peak_mb"] for profile in all_profiles),
        "input_tokens_mean": round(
            sum(profile["input_token_count"] for profile in all_profiles) / len(all_profiles), 2
        ),
        "visual_shapes": sorted({str(profile["pixel_values_shape"]) for profile in all_profiles}),
        "generated_tokens_mean": round(
            sum(profile["generated_token_count"] for profile in all_profiles) / len(all_profiles), 2
        ),
        "validator_pass_rate": round(pass_count / len(all_results), 4),
        "fallback_rate": round(1 - pass_count / len(all_results), 4),
        "errors": sum(len(sample["errors"]) for sample in samples.values()),
    }


def choose_best(rows: list[dict[str, Any]]) -> str | None:
    eligible = [
        row for row in rows
        if math.isclose(row["validator_pass_rate"], 1.0)
        and math.isclose(row["fallback_rate"], 0.0)
        and row["errors"] == 0
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (row["elevator_mean_ms"] + row["escalator_mean_ms"]) / 2,
    )["variant"]


def markdown_table(rows: list[dict[str, Any]], best: str | None) -> str:
    lines = [
        "# Phase 5 Latency Optimization Summary",
        "",
        "| Variant | Elevator mean | Escalator mean | p95 | GPU MB | Input tokens | Visual shape | Generated tokens | Validator pass | Fallback | Errors | Verdict |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        verdict = "BEST" if row["variant"] == best else ("eligible" if row["validator_pass_rate"] == 1 and row["fallback_rate"] == 0 and row["errors"] == 0 else "reject")
        lines.append(
            f"| {row['variant']} | {row['elevator_mean_ms']:.2f} ms | "
            f"{row['escalator_mean_ms']:.2f} ms | {row['p95_ms']:.2f} ms | "
            f"{row['gpu_peak_mb']:.2f} | {row['input_tokens_mean']:.2f} | "
            f"{' / '.join(row['visual_shapes'])} | {row['generated_tokens_mean']:.2f} | "
            f"{row['validator_pass_rate']:.1%} | {row['fallback_rate']:.1%} | "
            f"{row['errors']} | {verdict} |"
        )
    lines.extend(["", f"Best safety-eligible variant: `{best or 'none'}`", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = [
            summarize_report(json.loads(path.read_text(encoding="utf-8")))
            for path in args.reports
        ]
        best = choose_best(rows)
        summary = {"schema": "viassist.phase5.latency_summary.v1", "best": best, "variants": rows}
        save_report(args.json_output, summary)
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown_table(rows, best), encoding="utf-8")
    except Exception as error:
        print(f"summary 실패: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(f"JSON: {args.json_output.resolve()}")
    print(f"Markdown: {args.markdown_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
