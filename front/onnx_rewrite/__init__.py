"""Minimal ONNX rewrite scaffold focused on supported-op-only graphs."""

from .analysis import AuditSummary, audit_model, audit_path
from .checker.op_checker import OpChecker
from .core import OptimizationResult, UnsupportedOpError, optimize_model
from .runtime import LatencyResult, ValidationResult, build_inputs_for_model, compare_models, measure_latency
from .specs import PRIORITY_MODELS, SUPPORTED_OPS

__all__ = [
    "AuditSummary",
    "LatencyResult",
    "OpChecker",
    "OptimizationResult",
    "PRIORITY_MODELS",
    "SUPPORTED_OPS",
    "UnsupportedOpError",
    "ValidationResult",
    "audit_model",
    "audit_path",
    "build_inputs_for_model",
    "compare_models",
    "measure_latency",
    "optimize_model",
]
