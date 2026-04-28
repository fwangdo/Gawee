from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, shape_inference


ROOT = Path(__file__).resolve().parent.parent
MLIR_BUILD = ROOT / "middle/mlir/build"
BACK_BUILD = ROOT / "back/build"
ARTIFACT_DIR = ROOT / "artifacts/back_eval"

PRIORITY_MODELS = {
    "resnet18": ROOT / "artifacts/front_rewrite_bench/resnet18.onnx",
    "bert_tiny": ROOT / "artifacts/front_rewrite_bench/bert_tiny.onnx",
    "distilbert_base_uncased": ROOT / "artifacts/front_rewrite_bench/distilbert_base_uncased.onnx",
}

# Static shape overrides for models with dynamic dimensions.
# Maps model name → {input_name: [concrete_shape]}.
STATIC_SHAPE_OVERRIDES: dict[str, dict[str, list[int]]] = {
    "bert_tiny": {
        "input_ids": [1, 128],
        "attention_mask": [1, 128],
        "token_type_ids": [1, 128],
    },
    "distilbert_base_uncased": {
        "input_ids": [1, 128],
        "attention_mask": [1, 128],
    },
}

# Per-model atol for correctness comparison.
# NLP models accumulate more floating-point error than vision models.
MODEL_ATOL: dict[str, float] = {
    "distilbert_base_uncased": 5e-4,
    "bert_tiny": 5e-4,
}


@dataclass
class StageResult:
    ok: bool
    detail: str
    path: str | None = None


@dataclass
class ModelReport:
    model: str
    source_model: str
    inferred_model: StageResult
    translated_mlir: StageResult
    loops_mlir: StageResult
    llvm_mlir: StageResult
    aot_build: StageResult
    correctness: dict[str, Any] | None
    latency: dict[str, Any] | None
    notes: list[str]


def run(
    args: list[str],
    *,
    cwd: Path = ROOT,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def first_error_line(text: str) -> str:
    for line in text.splitlines():
      stripped = line.strip()
      if stripped:
          return stripped
    return "(no stderr)"


def bind_static_shapes(
    model: onnx.ModelProto,
    shape_overrides: dict[str, list[int]],
) -> onnx.ModelProto:
    """Replace dynamic dimensions in graph inputs with concrete values.

    Also resolves matching symbolic dims (dim_param) in graph outputs
    and value_info, since onnx.shape_inference may not propagate them.
    """
    # Collect dim_param → concrete value mapping from inputs.
    param_map: dict[str, int] = {}
    for inp in model.graph.input:
        if inp.name not in shape_overrides:
            continue
        target = shape_overrides[inp.name]
        dims = inp.type.tensor_type.shape.dim
        if len(dims) != len(target):
            raise ValueError(
                f"{inp.name}: expected rank {len(target)}, got {len(dims)}"
            )
        for dim, val in zip(dims, target):
            if dim.HasField("dim_param"):
                param_map[dim.dim_param] = val
            dim.ClearField("dim_param")
            dim.dim_value = val

    # Apply param_map to outputs and value_info.
    for vi in list(model.graph.output) + list(model.graph.value_info):
        if not vi.type.HasField("tensor_type"):
            continue
        for dim in vi.type.tensor_type.shape.dim:
            if dim.HasField("dim_param") and dim.dim_param in param_map:
                val = param_map[dim.dim_param]
                dim.ClearField("dim_param")
                dim.dim_value = val

    return shape_inference.infer_shapes(model)


def infer_model(src: Path, dst: Path) -> StageResult:
    try:
        model = onnx.load(src)
        inferred = shape_inference.infer_shapes(model)
        onnx.save(inferred, dst)
        return StageResult(True, f"value_info={len(inferred.graph.value_info)}", str(dst))
    except Exception as exc:  # pragma: no cover - diagnostic path
        return StageResult(False, str(exc), None)


def run_translate(src: Path, dst: Path) -> StageResult:
    proc = run([str(MLIR_BUILD / "gawee-onnx-translate"), str(src), "-o", str(dst)])
    if proc.returncode != 0:
        return StageResult(False, first_error_line(proc.stderr), None)
    return StageResult(True, "ok", str(dst))


def run_lower(pipeline: str, src: Path, dst: Path) -> StageResult:
    proc = run([str(MLIR_BUILD / "gawee-opt"), f"--{pipeline}", str(src)])
    if proc.returncode != 0:
        return StageResult(False, first_error_line(proc.stderr), None)
    dst.write_text(proc.stdout)
    return StageResult(True, "ok", str(dst))


def func_signature_line(path: Path) -> str | None:
    pattern = re.compile(r"func\.func @forward\((.*)")
    for line in path.read_text().splitlines():
        if "func.func @forward" in line:
            return line.strip()
    return None


def function_returns_memref(path: Path) -> bool:
    sig = func_signature_line(path)
    return sig is not None and "-> memref<" in sig


def has_dynamic_shape_signature(path: Path) -> bool:
    sig = func_signature_line(path)
    if sig is None:
        return False
    for match in re.finditer(r"memref<([0-9\?x]+)x[a-z0-9]+(?:,[^>]*)?>", sig):
        if "?" in match.group(1):
            return True
    return False


def graph_input_names(model: onnx.ModelProto) -> list[str]:
    initializer_names = {init.name for init in model.graph.initializer}
    return [value.name for value in model.graph.input if value.name not in initializer_names]


def tensor_proto_to_numpy(tensor: onnx.TensorProto) -> np.ndarray:
    return numpy_helper.to_array(tensor)


def _is_integer_type(data_type: int) -> bool:
    return data_type in {
        onnx.TensorProto.BOOL,
        onnx.TensorProto.INT8,
        onnx.TensorProto.INT16,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
        onnx.TensorProto.UINT8,
        onnx.TensorProto.UINT16,
        onnx.TensorProto.UINT32,
        onnx.TensorProto.UINT64,
    }


def _has_integer_data(tensor: onnx.TensorProto) -> bool:
    """Check if an integer initializer has non-empty data.

    Mirrors the collectI64TensorLiteral logic in ONNXMLIREmitter.cpp:
    only INT32/INT64 with non-empty data become arith.constant;
    empty tensors (shape contains 0) fall through to function args.
    """
    if tensor.data_type == onnx.TensorProto.INT64:
        return tensor.int64_data or tensor.raw_data
    if tensor.data_type == onnx.TensorProto.INT32:
        return tensor.int32_data or tensor.raw_data
    return False


def initializer_becomes_runtime_arg(tensor: onnx.TensorProto) -> bool:
    """Return True if this initializer becomes a function argument in MLIR."""
    if not _is_integer_type(tensor.data_type):
        return True
    # Integer initializers with data become arith.constant (not a func arg).
    # Empty integer tensors (e.g. tensor<0xi64>) become func args.
    return not _has_integer_data(tensor)


def static_shape_from_value_info(value_info: onnx.ValueInfoProto) -> list[int] | None:
    if not value_info.type.HasField("tensor_type"):
        return None
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            return None
        shape.append(dim.dim_value)
    return shape


def make_feed_dict(model: onnx.ModelProto) -> dict[str, np.ndarray] | None:
    feed: dict[str, np.ndarray] = {}
    initializer_names = {init.name for init in model.graph.initializer}
    rng = np.random.default_rng(42)
    for value in model.graph.input:
        if value.name in initializer_names:
            continue
        shape = static_shape_from_value_info(value)
        if shape is None:
            return None
        elem_type = value.type.tensor_type.elem_type
        if elem_type == onnx.TensorProto.INT64:
            feed[value.name] = rng.integers(0, 4, size=shape, dtype=np.int64)
        elif elem_type == onnx.TensorProto.INT32:
            feed[value.name] = rng.integers(0, 4, size=shape, dtype=np.int32)
        else:
            feed[value.name] = rng.standard_normal(size=shape).astype(np.float32)
    return feed


def save_runner_inputs(model: onnx.ModelProto, out_dir: Path) -> bool:
    feed = make_feed_dict(model)
    if feed is None:
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    initializer_names = {init.name for init in model.graph.initializer}
    ordered_inputs = [value.name for value in model.graph.input if value.name not in initializer_names]

    for idx, name in enumerate(ordered_inputs):
        np.save(out_dir / f"arg{idx}.npy", feed[name], allow_pickle=False)

    base = len(ordered_inputs)
    runtime_initializers = [
        init for init in model.graph.initializer if initializer_becomes_runtime_arg(init)
    ]
    for offset, init in enumerate(runtime_initializers):
        np.save(
            out_dir / f"arg{base + offset}.npy",
            tensor_proto_to_numpy(init),
            allow_pickle=False,
        )
    return True


def make_feed_dict_from_saved_inputs(
    model: onnx.ModelProto, inputs_dir: Path
) -> dict[str, np.ndarray] | None:
    initializer_names = {init.name for init in model.graph.initializer}
    ordered_inputs = [
        value.name for value in model.graph.input if value.name not in initializer_names
    ]
    feed: dict[str, np.ndarray] = {}
    for idx, name in enumerate(ordered_inputs):
        path = inputs_dir / f"arg{idx}.npy"
        if not path.exists():
            return None
        feed[name] = np.load(path)
    return feed


def measure_ort_latency(model_path: Path, warmup: int = 1, repeat: int = 5) -> dict[str, float] | None:
    model = onnx.load(model_path)
    feed = make_feed_dict(model)
    if feed is None:
        return None
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    for _ in range(warmup):
        session.run(None, feed)
    times_ms = []
    for _ in range(repeat):
        start = time.perf_counter()
        session.run(None, feed)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)
    return {
        "min_ms": float(min(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "max_ms": float(max(times_ms)),
    }


def build_aot_runner(abi_source: Path, llvm_mlir: Path, runner_path: Path) -> StageResult:
    num_output_args = 0 if function_returns_memref(abi_source) else 1
    proc = run(
        [
            str(BACK_BUILD / "gawee-aot"),
            "build",
            "--abi-source",
            str(abi_source),
            "--input",
            str(llvm_mlir),
            "--output",
            str(runner_path),
            "--entry",
            "forward",
            "--num-output-args",
            str(num_output_args),
        ]
    )
    if proc.returncode != 0:
        return StageResult(False, first_error_line(proc.stderr or proc.stdout), None)
    return StageResult(True, "ok", str(runner_path))


def compare_with_ort(model_path: Path, runner: Path, inputs_dir: Path, outputs_dir: Path, *, atol: float = 1e-4) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    model = onnx.load(model_path)
    feed = make_feed_dict_from_saved_inputs(model, inputs_dir)
    if feed is None:
      return None, None

    outputs_dir.mkdir(parents=True, exist_ok=True)
    proc = run([str(runner), str(inputs_dir), str(outputs_dir)])
    if proc.returncode != 0:
        return {"ok": False, "detail": first_error_line(proc.stderr or proc.stdout)}, None

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, feed)
    gawee_output_path = outputs_dir / "output0.npy"
    if not gawee_output_path.exists():
        return {"ok": False, "detail": "runner did not produce output0.npy"}, None
    gawee_output = np.load(gawee_output_path)
    ort_output = ort_outputs[0]
    if gawee_output.shape != ort_output.shape:
        return {
            "ok": False,
            "detail": f"shape mismatch: gawee={gawee_output.shape}, ort={ort_output.shape}",
        }, None

    max_abs = float(np.max(np.abs(gawee_output - ort_output)))
    close = bool(np.allclose(gawee_output, ort_output, atol=atol, rtol=1e-4))

    try:
        bench = run(
            [
                str(BACK_BUILD / "gawee-eval"),
                "benchmark",
                "--runner",
                str(runner),
                "--inputs",
                str(inputs_dir),
                "--outputs",
                str(outputs_dir / "bench"),
                "--warmup",
                "1",
                "--iters",
                "5",
            ]
        )
    except subprocess.TimeoutExpired:
        bench = None
    gawee_latency = None
    if bench is not None and bench.returncode == 0:
        text = bench.stdout
        match = re.search(r"p50:\s+([0-9.]+)", text)
        if match:
            gawee_latency = {"p50_ms": float(match.group(1))}

    ort_latency = measure_ort_latency(model_path)
    latency = None
    if gawee_latency is not None and ort_latency is not None:
        latency = {"gawee_end_to_end": gawee_latency, "onnxruntime": ort_latency}

    return {
        "ok": close,
        "max_abs_diff": max_abs,
        "atol": atol,
        "rtol": 1e-4,
    }, latency


def evaluate_model(name: str, model_path: Path) -> ModelReport:
    model_dir = ARTIFACT_DIR / name
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    notes: list[str] = []

    # Bind dynamic dims to concrete values before shape inference.
    src_path = model_path
    overrides = STATIC_SHAPE_OVERRIDES.get(name)
    if overrides is not None:
        try:
            raw = onnx.load(model_path)
            fixed = bind_static_shapes(raw, overrides)
            fixed_path = model_dir / "static.onnx"
            onnx.save(fixed, fixed_path)
            src_path = fixed_path
            notes.append(f"bound static shapes: {overrides}")
        except Exception as exc:
            return ModelReport(name, str(model_path), StageResult(False, f"shape bind: {exc}"), StageResult(False, "skipped"), StageResult(False, "skipped"), StageResult(False, "skipped"), StageResult(False, "skipped"), None, None, notes)

    inferred_path = model_dir / "inferred.onnx"
    inferred = infer_model(src_path, inferred_path)
    if not inferred.ok:
        return ModelReport(name, str(model_path), inferred, StageResult(False, "skipped"), StageResult(False, "skipped"), StageResult(False, "skipped"), StageResult(False, "skipped"), None, None, notes)

    translated_path = model_dir / "gawee.mlir"
    translated = run_translate(inferred_path, translated_path)
    if not translated.ok:
        return ModelReport(name, str(model_path), inferred, translated, StageResult(False, "skipped"), StageResult(False, "skipped"), StageResult(False, "skipped"), None, None, notes)

    loops_path = model_dir / "loops.mlir"
    loops = run_lower("gawee-to-loops", translated_path, loops_path)
    llvm_path = model_dir / "llvm.mlir"
    llvm = run_lower("gawee-to-llvm", translated_path, llvm_path)

    if loops.ok and has_dynamic_shape_signature(loops_path):
        notes.append("AOT runner currently supports only static-shape memref signatures.")

    aot = StageResult(False, "skipped")
    correctness = None
    latency = None

    if loops.ok and llvm.ok and not has_dynamic_shape_signature(loops_path):
        model = onnx.load(inferred_path)
        inputs_dir = model_dir / "inputs"
        if save_runner_inputs(model, inputs_dir):
            runner_path = model_dir / "forward_runner"
            aot = build_aot_runner(loops_path, llvm_path, runner_path)
            if aot.ok:
                atol = MODEL_ATOL.get(name, 1e-4)
                correctness, latency = compare_with_ort(
                    inferred_path, runner_path, inputs_dir, model_dir / "outputs",
                    atol=atol,
                )
        else:
            aot = StageResult(False, "could not build static runner inputs", None)

    return ModelReport(
        model=name,
        source_model=str(model_path),
        inferred_model=inferred,
        translated_mlir=translated,
        loops_mlir=loops,
        llvm_mlir=llvm,
        aot_build=aot,
        correctness=correctness,
        latency=latency,
        notes=notes,
    )


def render_markdown(reports: list[ModelReport]) -> str:
    lines = [
        "# Priority Model Backend Evaluation",
        "",
        "| model | infer | translate | loops | llvm | aot | correctness | latency |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for report in reports:
        correctness = "-"
        if report.correctness is not None:
            correctness = (
                f"{'pass' if report.correctness.get('ok') else 'fail'}"
                f" (max_abs={report.correctness.get('max_abs_diff', 'n/a')})"
            )
        latency = "-"
        if report.latency is not None:
            gawee = report.latency["gawee_end_to_end"]["p50_ms"]
            ort_med = report.latency["onnxruntime"]["median_ms"]
            latency = f"gawee={gawee:.3f}ms / ort={ort_med:.3f}ms"
        lines.append(
            f"| `{report.model}` | {'ok' if report.inferred_model.ok else 'fail'} | "
            f"{'ok' if report.translated_mlir.ok else 'fail'} | "
            f"{'ok' if report.loops_mlir.ok else 'fail'} | "
            f"{'ok' if report.llvm_mlir.ok else 'fail'} | "
            f"{'ok' if report.aot_build.ok else 'fail'} | {correctness} | {latency} |"
        )
        for stage_name, stage in [
            ("translate", report.translated_mlir),
            ("loops", report.loops_mlir),
            ("llvm", report.llvm_mlir),
            ("aot", report.aot_build),
        ]:
            if not stage.ok:
                lines.append(f"")
                lines.append(f"- `{report.model}` {stage_name} fail: {stage.detail}")
                break
        for note in report.notes:
            lines.append(f"- `{report.model}` note: {note}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate priority models through Gawee middle/backend.")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ARTIFACT_DIR,
        help="Directory for generated reports and intermediates.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global ARTIFACT_DIR
    ARTIFACT_DIR = args.report_dir
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    reports = [evaluate_model(name, path) for name, path in PRIORITY_MODELS.items()]
    json_path = ARTIFACT_DIR / "priority_models_report.json"
    md_path = ARTIFACT_DIR / "priority_models_report.md"
    json_path.write_text(json.dumps([asdict(report) for report in reports], indent=2))
    md_path.write_text(render_markdown(reports))
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
