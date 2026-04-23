from __future__ import annotations

import argparse
import json
from pathlib import Path

from front.onnx_rewrite import (
    UnsupportedOpError,
    audit_path,
    optimize_model,
)
from front.onnx_rewrite.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal frontend ONNX rewrite scaffold with supported-op-only enforcement."
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
        "--audit-only",
        action="store_true",
        help="Only print audit result without running the pipeline.",
    )
    return parser.parse_args()


def resolve_input_path(args: argparse.Namespace) -> Path:
    return args.input


def default_output_path(input_path: Path) -> Path:
    return Path("artifacts/front_rewrite") / input_path.name


def run_rewrite(args: argparse.Namespace):
    # main.
    input_path = resolve_input_path(args)
    output_path = args.output or default_output_path(input_path)
    # here.  
    return optimize_model(input_path, output_path, report_path=None)


def write_report_if_requested(result, report_path: Path | None) -> None:
    if report_path is not None:
        write_json(result.to_dict(), report_path)


def print_run_summary(result, report_path: Path | None, success: bool) -> None:
    status = "OK" if success else "FAIL"
    before_nodes = result.before["total_nodes"]
    after_nodes = result.after["total_nodes"]
    before_unsupported = result.before["unsupported_histogram"]
    after_unsupported = result.after["unsupported_histogram"]

    print(f"{status}: {result.input_path}")
    print(f"output: {result.output_path}")
    print(f"nodes: {before_nodes} -> {after_nodes}")
    print(f"unsupported_before: {len(before_unsupported)} kinds")
    print(f"unsupported_after: {len(after_unsupported)} kinds")
    print("")
    print("rewrite_log:")
    for line in result.logs:
        print(line)
    if report_path is not None:
        print("")
        print(f"report={report_path}")


def main() -> int:
    args = parse_args()

    if args.audit_only:
        summary = audit_path(resolve_input_path(args))
        print(json.dumps(summary.to_dict(), indent=2))
        return 0 if summary.is_supported_only else 2

    try:
        result = run_rewrite(args)
    except UnsupportedOpError as exc:
        print(str(exc))
        print_run_summary(exc.result, args.report, success=False)
        return 2

    write_report_if_requested(result, args.report)
    print_run_summary(result, args.report, success=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
