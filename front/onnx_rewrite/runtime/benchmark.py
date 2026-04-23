from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class LatencyResult:
    warmup: int
    repeat: int
    median_ms: float
    p95_ms: float
    samples_ms: list[float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _make_session(model_path: str, intra_op_threads: int = 1) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])


def measure_latency(
    model_path: str,
    inputs: dict[str, np.ndarray],
    warmup: int = 5,
    repeat: int = 20,
    intra_op_threads: int = 1,
) -> LatencyResult:
    session = _make_session(model_path, intra_op_threads=intra_op_threads)

    for _ in range(warmup):
        session.run(None, inputs)

    samples_ms: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        session.run(None, inputs)
        end = time.perf_counter()
        samples_ms.append((end - start) * 1000.0)

    samples = np.asarray(samples_ms, dtype=np.float64)
    return LatencyResult(
        warmup=warmup,
        repeat=repeat,
        median_ms=float(np.median(samples)),
        p95_ms=float(np.percentile(samples, 95)),
        samples_ms=[float(x) for x in samples_ms],
    )
