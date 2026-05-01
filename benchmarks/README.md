# Gawee Benchmark Models

이 디렉토리는 `Gawee`에서 parser / rewrite / quantization 실험에 공통으로 쓸 benchmark 모델을 모아 두는 곳이다.
대용량 바이너리 자체는 저장소에 커밋하지 않고, 공개 소스에서 재현 시점에 다시 내려받는다.

## Priority Models (correctness 검증 완료)

MLIR lowering end-to-end correctness가 확인된 모델:

- `onnx/vision/resnet18.onnx` — vision baseline
- `onnx/nlp/bert_tiny/onnx/model.onnx` — transformer encoder
- `onnx/nlp/tinyllama_15m/onnx/model.onnx` — RoPE 포함 decoder LLM

## Extended Models (확장 대상)

- `onnx/vision/yolo26_nano/onnx/model.onnx` — object detection (397 nodes)
- `onnx/nlp/smollm_135m/onnx/model.onnx` — 30-layer decoder LLM (2844 nodes)
- `onnx/vision/yolo26_n/model.onnx` — YOLO variant
- `onnx/nlp/qwen3_0_6b/model.onnx` — 대형 decoder (semantic op 확인용)

현재 frontend benchmark scope는 `ai.onnx opset >= 13` 모델만 대상으로 한다.

## Vision

- `onnx/vision/resnet18.onnx`
  - `torchvision`에서 ���컬 export
- `onnx/vision/yolo26_nano/onnx/model.onnx`
  - 397 nodes, Conv/Sigmoid 위주의 detection 모델
  - 미지원 op: `TopK`, `ReduceMax`

## NLP

- `onnx/nlp/bert_tiny/onnx/model.onnx`
  - source: `onnx-community/bert-tiny-finetuned-sms-spam-detection-ONNX`
  - 186 nodes, encoder-only transformer
- `onnx/nlp/tinyllama_15m/onnx/model.onnx`
  - source: optimum export, 6-layer decoder with RoPE
- `onnx/nlp/smollm_135m/onnx/model.onnx`
  - source: `HuggingFaceTB/SmolLM-135M`
  - 2844 nodes, 30-layer decoder, 514MB
  - 미지원 op: `Trilu`, `ScatterND`, `Greater`

## Reproduction

NLP benchmark는 Hugging Face 공개 저장소에서 기본 ONNX 파일만 내려받는다.
현재는 `ai.onnx opset >= 13` 모델만 유지한다.

```bash
python scripts/fetch_benchmark_models.py
```

Vision benchmark는 로컬에서 다시 export한다.

```bash
python scripts/export_vision_benchmarks.py
```

## Verification

모델을 받은 뒤 아래 두 단계로 확인한다.

```bash
python scripts/fetch_benchmark_models.py
python scripts/export_vision_benchmarks.py
python scripts/verify_benchmark_models.py
```
