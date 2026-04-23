from __future__ import annotations

from collections import Counter
from typing import Any

import onnx

from ..specs import SUPPORTED_OPS


class OpChecker:
    """Validate that a graph contains only supported ops."""

    @classmethod
    def histogram(cls, model: onnx.ModelProto) -> dict[str, int]:
        return dict(sorted(Counter(node.op_type for node in model.graph.node).items()))

    @classmethod
    def get_unsupported_ops(cls, model: onnx.ModelProto) -> set[str]:
        return {node.op_type for node in model.graph.node if node.op_type not in SUPPORTED_OPS}

    @classmethod
    def get_violations(cls, model: onnx.ModelProto) -> list[str]:
        violations: list[str] = []
        for node in model.graph.node:
            if node.op_type not in SUPPORTED_OPS:
                name = node.name or (node.output[0] if node.output else node.op_type)
                violations.append(f"{name}: unsupported op '{node.op_type}'")
        return violations

    @classmethod
    def assert_supported_only(cls, model: onnx.ModelProto) -> None:
        violations = cls.get_violations(model)
        if violations:
            raise AssertionError(
                "graph is not supported-op-only:\n" + "\n".join(f"  - {item}" for item in violations)
            )

    @classmethod
    def summarize(cls, model: onnx.ModelProto, path: str = "") -> dict[str, Any]:
        histogram = cls.histogram(model)
        unsupported = {op: count for op, count in histogram.items() if op not in SUPPORTED_OPS}
        return {
            "path": path,
            "total_nodes": sum(histogram.values()),
            "supported_ops": sorted(SUPPORTED_OPS),
            "op_histogram": histogram,
            "unsupported_histogram": unsupported,
            "is_supported_only": not unsupported,
        }
