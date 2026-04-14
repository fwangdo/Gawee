from __future__ import annotations

from pathlib import Path

import onnx
import torch
import torchvision


VISION_BENCHMARKS = {
    "resnet18": lambda: torchvision.models.resnet18(weights=None),
    "mobilenetv3_small": lambda: torchvision.models.mobilenet_v3_small(weights=None),
}


def freeze_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def export_model(name: str, out_dir: Path, opset: int = 18) -> Path:
    model = freeze_for_inference(VISION_BENCHMARKS[name]())
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{name}.onnx"
    x = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    torch.onnx.export(
        model,
        x,
        f=str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
    )

    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)
    print(f"[OK] Exported {name}: {onnx_path}")
    return onnx_path


def main() -> None:
    out_dir = Path("benchmarks/onnx/vision")
    for name in VISION_BENCHMARKS:
        export_model(name, out_dir)


if __name__ == "__main__":
    main()
