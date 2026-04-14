from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


def resolve_dim(dim: int | str | None) -> int:
    if isinstance(dim, int) and dim > 0:
        return dim
    return 8


def make_input(array_info: ort.NodeArg) -> np.ndarray:
    shape = [resolve_dim(dim) for dim in array_info.shape]
    dtype_name = array_info.type.removeprefix("tensor(").removesuffix(")")
    if dtype_name in {"float", "float32"}:
        return np.random.randn(*shape).astype(np.float32)
    if dtype_name in {"double", "float64"}:
        return np.random.randn(*shape).astype(np.float64)
    if dtype_name in {"int64"}:
        return np.random.randint(0, 1000, size=shape, dtype=np.int64)
    if dtype_name in {"int32"}:
        return np.random.randint(0, 1000, size=shape, dtype=np.int32)
    if dtype_name in {"bool"}:
        return np.random.randint(0, 2, size=shape).astype(bool)
    return np.random.randn(*shape).astype(np.float32)


def verify_model(path: Path) -> tuple[bool, str]:
    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = {arg.name: make_input(arg) for arg in sess.get_inputs()}
        outputs = sess.run(None, inputs)
        output_shapes = [tuple(np.asarray(out).shape) for out in outputs]
        return True, f"inputs={len(inputs)} outputs={output_shapes}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    root = Path("benchmarks/onnx")
    model_paths = sorted(root.rglob("*.onnx"))
    if not model_paths:
        raise SystemExit(
            "No ONNX benchmark models found under benchmarks/onnx. "
            "Run `python scripts/fetch_benchmark_models.py` for NLP models and "
            "`python scripts/export_vision_benchmarks.py` for vision models first."
        )

    print(f"Found {len(model_paths)} benchmark models")
    failed = 0
    for path in model_paths:
        ok, message = verify_model(path)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {path}: {message}")
        if not ok:
            failed += 1

    if failed:
        raise SystemExit(f"{failed} benchmark models failed to load")


if __name__ == "__main__":
    main()
