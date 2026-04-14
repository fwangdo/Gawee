# Gawee Benchmark Models

이 디렉토리는 `Gawee`에서 parser / rewrite / quantization 실험에 공통으로 쓸 benchmark 모델을 모아 두는 곳이다.
대용량 바이너리 자체는 저장소에 커밋하지 않고, 공개 소스에서 재현 시점에 다시 내려받는다.

## Vision

- `onnx/vision/resnet18.onnx`
- `onnx/vision/mobilenetv3_small.onnx`

둘 다 `torchvision`에서 로컬 export한다.
가중치는 benchmark graph 확보가 목적이므로 우선 `weights=None`으로 고정한다.

## NLP

- `onnx/nlp/bert_tiny/onnx/model.onnx`
  - source: `onnx-community/bert-tiny-finetuned-sms-spam-detection-ONNX`
  - link: `https://huggingface.co/onnx-community/bert-tiny-finetuned-sms-spam-detection-ONNX/tree/358e80a313103279be7292e32d112091c91de10b/onnx`
  - note: `prajjwal1/bert-tiny`의 직접 ONNX 파일이 공개되어 있지 않아, 우선 loadable한 `bert-tiny` 계열 ONNX 대체물로 확보
- `onnx/nlp/distilbert_base_uncased/onnx/model.onnx`
  - source: `onnx-community/distilbert-base-uncased-ONNX`
  - link: `https://huggingface.co/onnx-community/distilbert-base-uncased-ONNX/tree/a5d2f36/onnx`
- `onnx/nlp/minilm_l12_h384_uncased/onnx/model.onnx`
  - source: `microsoft/MiniLM-L12-H384-uncased`
  - link: `https://huggingface.co/microsoft/MiniLM-L12-H384-uncased/tree/86186eff27cda7c5bc520e45de4800c575d9d8b3/onnx`
- `onnx/nlp/mobilebert_uncased/onnx/model.onnx`
  - source: `onnx-community/mobilebert-uncased-ONNX`
  - link: `https://huggingface.co/onnx-community/mobilebert-uncased-ONNX/tree/942aa73/onnx`
- `onnx/nlp/distilroberta_base/onnx/model.onnx`
  - source: `Xenova/distilroberta-base`
  - link: `https://huggingface.co/Xenova/distilroberta-base/tree/main/onnx`

## Reproduction

NLP benchmark는 Hugging Face 공개 저장소에서 기본 ONNX 파일만 내려받는다.
revision이 명확히 확인된 모델은 해당 commit으로 고정한다.

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
