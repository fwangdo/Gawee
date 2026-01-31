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

from torchvision.models import resnet18, resnet50, vgg16


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
    ShapeProp(gm).propagate(*example_input)
    torch_shapes = get_torch_shapes(gm)

    # 2. Parse to Gawee IR and run shape inference
    g = TorchParser.parse_fx(gm, example_input)
    ShapeInference.run(g)
    gawee_shapes = get_gawee_shapes(g)

    # 3. Compare
    passed, failed, failures = compare_shapes(torch_shapes, gawee_shapes, model_name)

    # 4. Report
    print(f"\nResults: {passed} passed, {failed} failed (out of {passed + failed} values)")

    if verbose or failed > 0:
        print(f"\n--- All Value Shapes ---")
        for name in sorted(torch_shapes.keys()):
            torch_shape = torch_shapes[name]
            gawee_shape = gawee_shapes.get(name, "N/A")
            match = "OK" if torch_shape == gawee_shape else "MISMATCH"
            print(f"  {name}: torch={torch_shape}, gawee={gawee_shape} [{match}]")

    if failures:
        print(f"\n--- Failures ---")
        for f in failures:
            print(f"  {f['name']}: expected {f['torch']}, got {f['gawee']}")

    # 5. Check input/output shapes
    print(f"\n--- Graph Input/Output ---")
    print(f"  Inputs:  {[v.shape for v in g.inputs]}")
    print(f"  Outputs: {[v.shape for v in g.outputs]}")

    # Verify output matches PyTorch
    with torch.no_grad():
        torch_output = model(*example_input)
    expected_output_shape = list(torch_output.shape)
    actual_output_shape = g.outputs[0].shape

    if expected_output_shape == actual_output_shape:
        print(f"  Output shape matches PyTorch: {expected_output_shape}")
    else:
        print(f"  OUTPUT MISMATCH: expected {expected_output_shape}, got {actual_output_shape}")
        failed += 1

    return passed, failed


def main():
    print("=" * 60)
    print("ShapeInference Test Suite")
    print("Comparing Gawee IR shapes against PyTorch ground truth")
    print("=" * 60)

    total_passed = 0
    total_failed = 0
    skipped = 0

    # Test 1: ResNet18
    model = resnet18(weights=None)
    example_input = (torch.randn(1, 3, 224, 224),)
    p, f = test_model(model, example_input, "ResNet18")
    total_passed += p
    total_failed += f

    # Test 2: ResNet18 with different batch size
    model = resnet18(weights=None)
    example_input = (torch.randn(4, 3, 224, 224),)
    p, f = test_model(model, example_input, "ResNet18 (batch=4)")
    total_passed += p
    total_failed += f

    # Test 3: ResNet18 with different input size
    model = resnet18(weights=None)
    example_input = (torch.randn(1, 3, 128, 128),)
    p, f = test_model(model, example_input, "ResNet18 (128x128)")
    total_passed += p
    total_failed += f

    # Test 4: ResNet50
    model = resnet50(weights=None)
    example_input = (torch.randn(1, 3, 224, 224),)
    p, f = test_model(model, example_input, "ResNet50")
    total_passed += p
    total_failed += f

    # Test 5: VGG16 (skip - contains Dropout which is not supported yet)
    print(f"\n{'='*60}")
    print(f"Skipping: VGG16 (Dropout not supported)")
    print(f"{'='*60}")
    skipped += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed, {skipped} skipped")
    print("=" * 60)

    return total_failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
