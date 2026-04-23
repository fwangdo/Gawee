from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

from front.onnx_rewrite.runtime.validation import build_inputs_for_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open an ONNX Runtime session for a model and run one test inference."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input ONNX path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for dummy input generation.",
    )
    parser.add_argument(
        "--dynamic-size",
        type=int,
        default=1,
        help="Concrete size used for symbolic dimensions.",
    )
    return parser.parse_args()


def summarize_array(name: str, value: np.ndarray) -> str:
    return (
        f"{name}: shape={list(value.shape)} dtype={value.dtype} "
        f"min={float(np.min(value)):.6g} max={float(np.max(value)):.6g}"
    )


def main() -> int:
    args = parse_args()
    model_path = args.input

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    print(f"model={model_path}")
    print(f"providers={session.get_providers()}")

    print("inputs:")
    for item in session.get_inputs():
        print(f"  - {item.name}: type={item.type} shape={list(item.shape)}")

    print("outputs:")
    for item in session.get_outputs():
        print(f"  - {item.name}: type={item.type} shape={list(item.shape)}")

    inputs = build_inputs_for_model(
        str(model_path),
        seed=args.seed,
        dynamic_size=args.dynamic_size,
    )

    print("generated_inputs:")
    for name, value in inputs.items():
        print(f"  - {summarize_array(name, value)}")

    outputs = session.run(None, inputs)
    print("run_result:")
    for index, value in enumerate(outputs):
        array = np.asarray(value)
        print(f"  - output[{index}]: shape={list(array.shape)} dtype={array.dtype}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
