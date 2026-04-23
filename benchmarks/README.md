# Gawee Benchmark Models

이 디렉토리는 `Gawee`에서 parser / rewrite / quantization 실험에 공통으로 쓸 benchmark 모델을 모아 두는 곳이다.
대용량 바이너리 자체는 저장소에 커밋하지 않고, 공개 소스에서 재현 시점에 다시 내려받는다.

## Current Priority Trio

graph rewrite의 근시일 우선 모델은 아래 3개다.

- `onnx/vision/resnet18.onnx`
- `onnx/nlp/distilbert_base_uncased/onnx/model.onnx`
- `onnx/nlp/bert_tiny/onnx/model.onnx`

즉 현재 우선순위는 `vision 1개 + NLP 2개` 조합이다.

현재 frontend benchmark scope는 `ai.onnx opset >= 13` 모델만 대상으로 한다.

## Vision

- `onnx/vision/resnet18.onnx`

`torchvision`에서 로컬 export한다.
가중치는 benchmark graph 확보가 목적이므로 우선 `weights=None`으로 고정한다.

## NLP

- `onnx/nlp/bert_tiny/onnx/model.onnx`
  - source: `onnx-community/bert-tiny-finetuned-sms-spam-detection-ONNX`
  - link: `https://huggingface.co/onnx-community/bert-tiny-finetuned-sms-spam-detection-ONNX/tree/358e80a313103279be7292e32d112091c91de10b/onnx`
  - note: `prajjwal1/bert-tiny`의 직접 ONNX 파일이 공개되어 있지 않아, 우선 loadable한 `bert-tiny` 계열 ONNX 대체물로 확보
- `onnx/nlp/distilbert_base_uncased/onnx/model.onnx`
  - source: `onnx-community/distilbert-base-uncased-ONNX`
  - link: `https://huggingface.co/onnx-community/distilbert-base-uncased-ONNX/tree/a5d2f36/onnx`

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
