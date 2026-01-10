# scripts/verify_onnxruntime.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import onnxruntime as ort


def run_onnx(onnx_path: Path, input_shape=(1, 3, 224, 224)) -> np.ndarray:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # ONNX input name은 export에서 "input"으로 고정했음
    x = np.random.randn(*input_shape).astype(np.float32)
    outputs = sess.run(None, {"input": x})
    # output name은 "logits"였지만 sess.run(None, ...)이면 순서대로 나옴
    return outputs[0]


if __name__ == "__main__":
    onnx_path = Path("onnx/resnet18.onnx")
    y = run_onnx(onnx_path)
    print("[OK] ONNXRuntime output shape:", y.shape)
    print("     dtype:", y.dtype)
    print("     sample:", y.reshape(-1)[:5])
