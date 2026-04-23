from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import onnx

from ..analysis import AuditSummary, audit_model


class UnsupportedOpError(RuntimeError):
    """Raised when the final ONNX graph still contains unsupported ops."""


class RewritePass(Protocol):
    name: str

    def apply(self, model: onnx.ModelProto) -> bool:
        """Return True when the pass changed the graph."""


@dataclass(frozen=True)
class PipelineResult:
    input_path: str
    output_path: str
    changed: bool
    passes_run: list[str]
    before: AuditSummary
    after: AuditSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "changed": self.changed,
            "passes_run": self.passes_run,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }


class RewritePipeline:
    """
    Very basic scaffold for ONNX graph rewrite.

    Current contract:
    - run zero or more rewrite passes
    - validate the final graph
    - fail if any unsupported op remains
    """

    def __init__(self, passes: list[RewritePass] | None = None):
        self.passes = [] if passes is None else passes

    def run(self, input_path: str | Path, output_path: str | Path) -> PipelineResult:
        src = Path(input_path)
        dst = Path(output_path)
        dst.parent.mkdir(parents=True, exist_ok=True)

        model = onnx.load(str(src))
        before = audit_model(model)

        changed = False
        passes_run: list[str] = []
        for rewrite_pass in self.passes:
            passes_run.append(rewrite_pass.name)
            changed |= rewrite_pass.apply(model)

        onnx.checker.check_model(model)
        after = audit_model(model)
        if not after.is_supported_only:
            detail = ", ".join(f"{op}={count}" for op, count in after.unsupported_histogram.items())
            raise UnsupportedOpError(f"final graph is not supported-op-only: {detail}")

        if changed:
            onnx.save(model, str(dst))
        else:
            shutil.copyfile(src, dst)

        return PipelineResult(
            input_path=str(src),
            output_path=str(dst),
            changed=changed,
            passes_run=passes_run,
            before=AuditSummary(
                path=str(src),
                total_nodes=before.total_nodes,
                supported_ops=before.supported_ops,
                op_histogram=before.op_histogram,
                unsupported_histogram=before.unsupported_histogram,
            ),
            after=AuditSummary(
                path=str(dst),
                total_nodes=after.total_nodes,
                supported_ops=after.supported_ops,
                op_histogram=after.op_histogram,
                unsupported_histogram=after.unsupported_histogram,
            ),
        )

    @staticmethod
    def write_report(result: PipelineResult, report_path: str | Path) -> None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
