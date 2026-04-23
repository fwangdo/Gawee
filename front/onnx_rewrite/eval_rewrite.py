from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from front.onnx_rewrite import (
    OptimizationResult,
    build_inputs_for_model,
    compare_models,
    measure_latency,
)
from front.onnx_rewrite.run_onnx_rewrite import default_output_path, resolve_input_path, run_rewrite
from front.onnx_rewrite.utils.io import write_json


@dataclass(frozen=True)
class ComparisonResult:
    inputs: dict[str, int]
    validation: dict[str, Any]
    latency: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": self.inputs,
            "validation": self.validation,
            "latency": self.latency,
        }


@dataclass(frozen=True)
class EvalResult:
    optimization: OptimizationResult
    comparison: ComparisonResult

    def to_dict(self) -> dict[str, Any]:
        data = self.optimization.to_dict()
        data["comparison"] = self.comparison.to_dict()
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ONNX rewrite and evaluate runtime accuracy/latency deltas."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input ONNX path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output ONNX path for rewrite result.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations for latency measurement.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Measured iterations for latency measurement.",
    )
    return parser.parse_args()


def build_comparison(
    input_path: Path,
    output_path: Path,
    warmup: int,
    repeat: int,
) -> ComparisonResult:
    validation = compare_models(str(input_path), str(output_path))
    ort_inputs = build_inputs_for_model(
        str(input_path),
        seed=42,
        dynamic_size=1,
    )
    before_latency = measure_latency(str(input_path), ort_inputs, warmup=warmup, repeat=repeat)
    after_latency = measure_latency(str(output_path), ort_inputs, warmup=warmup, repeat=repeat)

    return ComparisonResult(
        inputs={"seed": 42, "dynamic_size": 1},
        validation=validation.to_dict(),
        latency={
            "before": before_latency.to_dict(),
            "after": after_latency.to_dict(),
            "delta_median_ms": after_latency.median_ms - before_latency.median_ms,
            "speedup_ratio": (
                (before_latency.median_ms / after_latency.median_ms)
                if after_latency.median_ms > 0.0
                else None
            ),
        },
    )


def run_eval(args: argparse.Namespace) -> EvalResult:
    optimization = run_rewrite(args)
    input_path = resolve_input_path(args)
    output_path = args.output or default_output_path(input_path)
    comparison = build_comparison(input_path, output_path, args.warmup, args.repeat)
    return EvalResult(optimization=optimization, comparison=comparison)


def write_report_if_requested(result: EvalResult, report_path: Path | None) -> None:
    if report_path is not None:
        write_json(result.to_dict(), report_path)


def print_eval_summary(result: EvalResult, report_path: Path | None) -> None:
    print(f"[OK] {result.optimization.input_path} -> {result.optimization.output_path}")
    print(f"nodes={result.optimization.after['total_nodes']}")

    validation = result.comparison.validation
    latency = result.comparison.latency
    print(
        "value_diff:"
        f" max_abs={validation['max_abs_diff']:.6g}"
        f" tol={validation['max_abs_tolerance']:.6g}"
        f" pass={validation['success']}"
        f" worst_case={validation['worst_case']}"
    )
    print(
        "latency:"
        f" before={latency['before']['median_ms']:.3f}ms"
        f" after={latency['after']['median_ms']:.3f}ms"
        f" delta={latency['delta_median_ms']:.3f}ms"
        f" speedup={latency['speedup_ratio']:.3f}x"
    )

    if report_path is not None:
        print(f"report={report_path}")


def main() -> int:
    args = parse_args()
    result = run_eval(args)
    write_report_if_requested(result, args.report)
    print_eval_summary(result, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
