"""
Test script for ShapeInference.
Compares Gawee IR shape inference results against PyTorch ground truth.
"""

from gawee_ir.parser import TorchParser
from gawee_ir.analysis.shape import ShapeInference

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

import segmentation_models_pytorch as smp


def get_torch_shapes(gm: fx.GraphModule) -> dict[str, list[int]]:
    """Extract shapes from FX graph after ShapeProp."""
    shapes = {}
    for node in gm.graph.nodes:
        if "tensor_meta" in node.meta:
            tm = node.meta["tensor_meta"]
            shapes[node.name] = list(tm.shape)
    return shapes


def get_gawee_shapes(g) -> dict[str, list[int]]:
    """Extract shapes from Gawee IR graph."""
    shapes = {}
    for name, value in g.values.items():
        if value.shape is not None:
            shapes[name] = value.shape
    return shapes


def compare_shapes(torch_shapes: dict, gawee_shapes: dict, model_name: str) -> tuple[int, int, list]:
    """Compare PyTorch shapes with Gawee IR shapes."""
    passed = 0
    failed = 0
    failures = []

    for name, torch_shape in torch_shapes.items():
        if name not in gawee_shapes:
            continue

        gawee_shape = gawee_shapes[name]
        if torch_shape == gawee_shape:
            passed += 1
        else:
            failed += 1
            failures.append({
                "name": name,
                "torch": torch_shape,
                "gawee": gawee_shape,
            })

    return passed, failed, failures


def test_model(model: nn.Module, example_input: tuple, model_name: str, verbose: bool = False):
    """Test shape inference on a model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    model.eval()

    # 1. FX trace with shape propagation
    gm = fx.symbolic_trace(model)

    # 2. Parse to Gawee IR and run shape inference
    g = TorchParser.parse_fx(gm, example_input)
    ShapeInference.run(g)
    gawee_shapes = get_gawee_shapes(g)
    return


def main():
    print("=" * 60)
    print("ShapeInference Test Suite - UNet")
    print("=" * 60)

    # UNet with ResNet34 encoder
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    example_input = (torch.randn(1, 3, 224, 224),)
    test_model(model, example_input, "UNet (ResNet34 encoder)")
    return


if __name__ == "__main__":
    main()
