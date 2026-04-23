from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx

from ..checker.op_checker import OpChecker
from ..passes.passer import Passer
from ..utils.io import load_model, save_model, write_json, write_text


@dataclass(frozen=True)
class OptimizationResult:
    input_path: str
    output_path: str
    before: dict[str, Any]
    after: dict[str, Any]
    logs: list[str]

    @property
    def is_supported_only(self) -> bool:
        return bool(self.after.get("is_supported_only", False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "before": self.before,
            "after": self.after,
            "logs": self.logs,
        }


class UnsupportedOpError(RuntimeError):
    """Raised when unsupported ops remain after optimization."""

    def __init__(self, message: str, result: OptimizationResult) -> None:
        super().__init__(message)
        self.result = result


def _format_histogram(histogram: dict[str, int]) -> str:
    if not histogram:
        return "none"
    return ", ".join(f"{op}={count}" for op, count in sorted(histogram.items()))


def _build_text_report(result: OptimizationResult) -> str:
    before_nodes = result.before["total_nodes"]
    after_nodes = result.after["total_nodes"]
    before_unsupported = _format_histogram(result.before["unsupported_histogram"])
    after_unsupported = _format_histogram(result.after["unsupported_histogram"])
    lines = [
        "ONNX Rewrite Report",
        f"input: {result.input_path}",
        f"output: {result.output_path}",
        "",
        "Summary",
        f"  nodes: {before_nodes} -> {after_nodes}",
        f"  unsupported: {before_unsupported} -> {after_unsupported}",
        f"  supported_only: {result.after['is_supported_only']}",
        "",
        "Pass Log",
    ]
    lines.extend(result.logs)
    return "\n".join(lines) + "\n"


def _write_reports(result: OptimizationResult, report_path: str | Path | None) -> None:
    if report_path is None:
        return
    write_json(result.to_dict(), report_path)
    write_text(_build_text_report(result), Path(report_path).with_suffix(".txt"))


def optimize_model(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
) -> OptimizationResult:
    src = Path(input_path)
    dst = Path(output_path)

    model = load_model(src)
    before = OpChecker.summarize(model, path=str(src))
    logs: list[str] = []

    if before["unsupported_histogram"]:
        logs.append(
            "Input unsupported ops: "
            f"{_format_histogram(before['unsupported_histogram'])}"
        )
    else:
        logs.append("Input graph is already supported-op-only")

    passer = Passer()
    model, pass_logs = passer.optimize(model)
    logs.extend(pass_logs)

    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        logs.append(f"ONNX checker failed: {exc}")
        failed_after = OpChecker.summarize(model, path=str(dst))
        result = OptimizationResult(
            input_path=str(src),
            output_path=str(dst),
            before=before,
            after=failed_after,
            logs=logs,
        )
        _write_reports(result, report_path)
        raise

    after = OpChecker.summarize(model, path=str(dst))
    if after["unsupported_histogram"]:
        detail = _format_histogram(after["unsupported_histogram"])
        logs.append(f"Output unsupported ops: {detail}")
        result = OptimizationResult(
            input_path=str(src),
            output_path=str(dst),
            before=before,
            after=after,
            logs=logs,
        )
        _write_reports(result, report_path)
        raise UnsupportedOpError(f"final graph is not supported-op-only: {detail}", result)

    logs.append("Output graph is supported-op-only")
    save_model(model, dst)

    result = OptimizationResult(
        input_path=str(src),
        output_path=str(dst),
        before=before,
        after=after,
        logs=logs,
    )
    _write_reports(result, report_path)
    return result
