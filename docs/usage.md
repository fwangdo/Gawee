## 1. Python 가상환경 구성

본 프로젝트는 **Python 3.10 이상**을 요구합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

가상환경 활성화 후 Python 버전을 확인합니다.

```bash
python --version
# Python 3.10.x 이상
```

---

## 2. 필요 라이브러리 설치

필요한 모든 라이브러리는 스크립트로 한 번에 설치합니다.

```bash
python3 ./scripts/setup.sh
```

해당 스크립트는 다음과 같은 라이브러리를 설치합니다.

* torch / torchvision
* onnx / onnxruntime
* numpy / pytest
* onnxscript
* segmentation-models-pytorch / timm

---

## 3. 사전 학습된 모델 저장

실험에 사용되는 ResNet / UNet 모델 가중치는
PyTorch 기준으로 미리 저장합니다.

```bash
sh ./scripts/save_models.sh
```

실행 후, 모델 파일은 다음 디렉토리에 저장됩니다.

```
./torchdata/
  ├── resnet18.pt
  └── unet.pt
```

---

## 4. ResNet 실행 및 비용 평가

ResNet-18 모델에 대해 다음 절차를 수행합니다.

1. PyTorch 모델을 FX 그래프로 변환
2. Gawee IR로 파싱
3. 그래프 최적화 수행
4. 최적화 전/후 비용(FLOPs, 메모리 접근량) 비교
```bash
sh ./scripts/show_resnet.sh
```

```text
== Before ==
...
== After ==
...
```

---

## 5. UNet 실행 및 비용 평가

UNet 모델에 대해 동일한 절차로 평가를 수행합니다.
```bash
sh ./scripts/show_unet.sh
```

UNet의 경우:
* `getattr / getitem` 등 Python bookkeeping 연산 제거
* Identity elimination
* Conv–BatchNorm folding

```text
== Before ==
...
== After ==
...
```