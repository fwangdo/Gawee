# scripts/export_onnx.py
from __future__ import annotations

from pathlib import Path
import torch
import onnx

from models.resnet18 import build_resnet18


def freeze_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """
    'freeze'의 실무적 의미:
    - eval 모드로 전환 (Dropout/BatchNorm 등 추론 동작)
    - grad 비활성화(파라미터 업데이트 안 함)
    - 파라미터의 requires_grad False
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def export_resnet18_to_onnx(
    onnx_path: Path,
    opset: int = 18,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) 모델 준비 + freeze
    model = build_resnet18(num_classes=1000)
    model = freeze_for_inference(model)

    # 2) 고정 입력(재현성)
    x = torch.randn(*input_shape, dtype=torch.float32)

    # 3) export 옵션 고정 (재현 가능)
    # - do_constant_folding=True: export 시점에 가능한 상수 폴딩을 수행(안정적으로)
    # - input/output 이름을 고정하여 후속 파이프라인에서 참조하기 쉽게
    torch.onnx.export(
        model,
        x,
        f=str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,  # 첫 단계에서는 고정 shape로 단순화
    )

    # 4) ONNX 모델 로드 + 체크
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)

    print(f"[OK] Exported ONNX: {onnx_path}")
    print(f"      opset={opset}, input_shape={input_shape}")


if __name__ == "__main__":
    export_resnet18_to_onnx(Path("onnx/resnet18.onnx"))
