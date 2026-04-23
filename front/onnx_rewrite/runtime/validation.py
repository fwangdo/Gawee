from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

MAX_ABS_TOLERANCE = 1e-4


@dataclass(frozen=True)
class VerificationCase:
    name: str
    seed: int
    dynamic_size: int
    mask_mode: str
    int_mode: str


@dataclass(frozen=True)
class ModelProfile:
    vocab_upper: int
    node_count: int
    initializer_count: int


@dataclass(frozen=True)
class OutputDiff:
    index: int
    shape: list[int]
    max_abs_diff: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CaseResult:
    name: str
    seed: int
    dynamic_size: int
    mask_mode: str
    int_mode: str
    outputs: list[OutputDiff]
    max_abs_diff: float

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "seed": self.seed,
            "dynamic_size": self.dynamic_size,
            "mask_mode": self.mask_mode,
            "int_mode": self.int_mode,
            "outputs": [item.to_dict() for item in self.outputs],
            "max_abs_diff": self.max_abs_diff,
        }


@dataclass(frozen=True)
class ValidationResult:
    success: bool
    cases_run: int
    cases: list[CaseResult]
    worst_case: str
    max_abs_tolerance: float
    max_abs_diff: float

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "cases_run": self.cases_run,
            "cases": [case.to_dict() for case in self.cases],
            "worst_case": self.worst_case,
            "max_abs_tolerance": self.max_abs_tolerance,
            "max_abs_diff": self.max_abs_diff,
        }


def _build_model_profile(model_path: str) -> ModelProfile:
    model = onnx.load(model_path)
    vocab_upper = 100
    for init in model.graph.initializer:
        if len(init.dims) != 2:
            continue
        rows = int(init.dims[0])
        cols = int(init.dims[1])
        if rows > cols and rows > vocab_upper:
            vocab_upper = rows
    return ModelProfile(
        vocab_upper=vocab_upper,
        node_count=len(model.graph.node),
        initializer_count=len(model.graph.initializer),
    )


def _resolve_input_shape(raw_shape: list[Any], dynamic_size: int) -> list[int]:
    resolved: list[int] = []
    symbol_sizes: dict[str, int] = {}

    for dim in raw_shape:
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
            continue
        if isinstance(dim, str) and dim:
            if dim not in symbol_sizes:
                symbol_sizes[dim] = dynamic_size
            resolved.append(symbol_sizes[dim])
            continue
        resolved.append(dynamic_size)

    return resolved


def _build_mask_input(
    shape: list[int],
    rng: np.random.Generator,
    mode: str,
    dtype: np.dtype,
) -> np.ndarray:
    if mode == "ones":
        return np.ones(shape, dtype=dtype)

    if mode == "random_binary":
        mask = rng.integers(0, 2, size=shape, dtype=np.int32)
        if not mask.any():
            mask.flat[0] = 1
        return mask.astype(dtype)

    if mode == "prefix_drop":
        mask = np.ones(shape, dtype=np.int32)
        last_dim = shape[-1] if shape else 1
        cutoff = max(1, last_dim // 3)
        slicer = [slice(None)] * len(shape)
        slicer[-1] = slice(cutoff, None)
        mask[tuple(slicer)] = 0
        return mask.astype(dtype)

    if mode == "checkerboard":
        total = int(np.prod(shape))
        mask = (np.arange(total) % 2).reshape(shape).astype(np.int32)
        if not mask.any():
            mask.flat[0] = 1
        return mask.astype(dtype)

    raise ValueError(f"Unsupported mask mode: {mode}")


def _build_int_input(
    shape: list[int],
    upper: int,
    rng: np.random.Generator,
    mode: str,
    dtype: np.dtype,
) -> np.ndarray:
    if upper <= 0:
        upper = 1

    if mode == "random_full":
        values = rng.integers(0, upper, size=shape, dtype=np.int64)
    elif mode == "edge_bias":
        edge_values = np.unique(np.array([0, 1, max(0, upper - 2), upper - 1], dtype=np.int64))
        values = rng.choice(edge_values, size=shape)
    elif mode == "low_band":
        band = max(2, min(64, upper))
        values = rng.integers(0, band, size=shape, dtype=np.int64)
    elif mode == "repeated_token":
        token = int(rng.integers(0, upper, size=()))
        values = np.full(shape, token, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported int mode: {mode}")

    return values.astype(dtype)


def _build_float_input(
    shape: list[int],
    rng: np.random.Generator,
    seed: int,
) -> np.ndarray:
    scale = 1.0 + (seed % 3)
    shift = float((seed % 5) - 2)
    return (rng.standard_normal(shape).astype(np.float32) * scale + shift).astype(np.float32)


def _generate_inputs_for_case(
    session: ort.InferenceSession,
    vocab_upper: int,
    case: VerificationCase,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(case.seed)
    inputs: dict[str, np.ndarray] = {}

    for inp in session.get_inputs():
        shape = _resolve_input_shape(list(inp.shape), case.dynamic_size)
        input_name = inp.name.lower()
        input_type = inp.type

        if "mask" in input_name:
            if "int64" in input_type:
                inputs[inp.name] = _build_mask_input(shape, rng, case.mask_mode, np.int64)
            elif "int32" in input_type:
                inputs[inp.name] = _build_mask_input(shape, rng, case.mask_mode, np.int32)
            else:
                inputs[inp.name] = _build_mask_input(shape, rng, case.mask_mode, np.float32)
        elif "token_type" in input_name:
            if "int64" in input_type:
                inputs[inp.name] = _build_int_input(shape, 2, rng, case.int_mode, np.int64)
            elif "int32" in input_type:
                inputs[inp.name] = _build_int_input(shape, 2, rng, case.int_mode, np.int32)
            else:
                inputs[inp.name] = np.zeros(shape, dtype=np.float32)
        elif "int64" in input_type:
            inputs[inp.name] = _build_int_input(shape, vocab_upper, rng, case.int_mode, np.int64)
        elif "int32" in input_type:
            inputs[inp.name] = _build_int_input(shape, vocab_upper, rng, case.int_mode, np.int32)
        elif "bool" in input_type:
            inputs[inp.name] = rng.integers(0, 2, size=shape).astype(bool)
        else:
            inputs[inp.name] = _build_float_input(shape, rng, case.seed)

    return inputs


def _build_case_pool() -> list[VerificationCase]:
    return [
        VerificationCase("baseline", 42, 1, "ones", "random_full"),
        VerificationCase("dynamic_2", 7, 2, "ones", "random_full"),
        VerificationCase("dynamic_3", 13, 3, "ones", "random_full"),
        VerificationCase("mask_random", 11, 2, "random_binary", "random_full"),
        VerificationCase("mask_prefix", 23, 2, "prefix_drop", "random_full"),
        VerificationCase("mask_checker", 31, 3, "checkerboard", "random_full"),
        VerificationCase("edge_indices", 19, 1, "ones", "edge_bias"),
        VerificationCase("low_band_vocab", 29, 2, "random_binary", "low_band"),
        VerificationCase("repeat_token", 37, 3, "prefix_drop", "repeated_token"),
        VerificationCase("mixed_stress", 41, 2, "checkerboard", "edge_bias"),
    ]


def _select_cases(profile: ModelProfile) -> list[VerificationCase]:
    case_pool = _build_case_pool()
    if profile.node_count <= 32 and profile.initializer_count <= 32:
        return case_pool
    if profile.node_count <= 256:
        return case_pool[:9]
    return case_pool[:8]


def _run_session(
    session: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
) -> list[np.ndarray]:
    return [np.asarray(out) for out in session.run(None, inputs)]


def _strip_shared_nans(
    before: np.ndarray,
    after: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if before.shape != after.shape:
        raise ValueError(f"shape mismatch: {before.shape} vs {after.shape}")

    before = before.astype(np.float32)
    after = after.astype(np.float32)
    before_nan = np.isnan(before)
    after_nan = np.isnan(after)

    if np.logical_xor(before_nan, after_nan).any():
        raise ValueError("NaN mismatch detected between compared outputs")

    valid_mask = ~(before_nan & after_nan)
    return before[valid_mask], after[valid_mask]


def _compare_output_pair(index: int, before: np.ndarray, after: np.ndarray) -> OutputDiff:
    before_clean, after_clean = _strip_shared_nans(before, after)
    if before_clean.size == 0:
        return OutputDiff(
            index=index,
            shape=list(before.shape),
            max_abs_diff=0.0,
        )

    abs_diff = np.abs(before_clean - after_clean)
    return OutputDiff(
        index=index,
        shape=list(before.shape),
        max_abs_diff=float(np.max(abs_diff)),
    )


def build_inputs_for_model(
    model_path: str,
    seed: int = 42,
    dynamic_size: int = 1,
) -> dict[str, np.ndarray]:
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    profile = _build_model_profile(model_path)
    case = VerificationCase("manual", seed, dynamic_size, "ones", "random_full")
    return _generate_inputs_for_case(session, profile.vocab_upper, case)


def compare_models(
    before_model_path: str,
    after_model_path: str,
) -> ValidationResult:
    before_session = ort.InferenceSession(before_model_path, providers=["CPUExecutionProvider"])
    after_session = ort.InferenceSession(after_model_path, providers=["CPUExecutionProvider"])

    profile = _build_model_profile(before_model_path)
    cases = _select_cases(profile)
    case_results: list[CaseResult] = []

    for case in cases:
        inputs = _generate_inputs_for_case(before_session, profile.vocab_upper, case)
        before_outputs = _run_session(before_session, inputs)
        after_outputs = _run_session(after_session, inputs)

        if len(before_outputs) != len(after_outputs):
            raise ValueError(
                f"output count mismatch in case {case.name}: "
                f"before={len(before_outputs)} after={len(after_outputs)}"
            )

        diffs = [
            _compare_output_pair(index, before, after)
            for index, (before, after) in enumerate(zip(before_outputs, after_outputs))
        ]
        case_results.append(
            CaseResult(
                name=case.name,
                seed=case.seed,
                dynamic_size=case.dynamic_size,
                mask_mode=case.mask_mode,
                int_mode=case.int_mode,
                outputs=diffs,
                max_abs_diff=max(item.max_abs_diff for item in diffs) if diffs else 0.0,
            )
        )

    worst_case = max(case_results, key=lambda item: item.max_abs_diff)
    max_abs_diff = max(case.max_abs_diff for case in case_results) if case_results else 0.0
    return ValidationResult(
        success=max_abs_diff <= MAX_ABS_TOLERANCE,
        cases_run=len(case_results),
        cases=case_results,
        worst_case=worst_case.name,
        max_abs_tolerance=MAX_ABS_TOLERANCE,
        max_abs_diff=max_abs_diff,
    )
