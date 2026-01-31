"""
Test script for verifying lowering to JSON + binary files.
Checks:
1. All weight files exist and can be loaded
2. Weight shapes match metadata
3. All node inputs/outputs reference valid values
4. Graph structure is valid
"""

from __future__ import annotations
import json
import os
import numpy as np

from gawee_ir.parser import TorchParser
from gawee_ir.analysis.shape import ShapeInference
from gawee_ir.passes.passer import Passer
from gawee_ir.translator import Translator

import torch
import torch.fx as fx
from torchvision.models import resnet18, resnet50


def verify_graph_json(out_dir: str) -> tuple[int, int, list]:
    """Verify the exported graph.json and binary files."""
    passed = 0
    failed = 0
    errors = []

    graph_path = os.path.join(out_dir, "graph.json")
    with open(graph_path, "r") as f:
        graph = json.load(f)

    # 1. Check basic structure
    required_keys = ["inputs", "outputs", "values", "nodes"]
    for key in required_keys:
        if key in graph:
            passed += 1
        else:
            failed += 1
            errors.append(f"Missing required key: {key}")

    # 2. Check all inputs are in values
    for inp in graph["inputs"]:
        if inp in graph["values"]:
            passed += 1
        else:
            failed += 1
            errors.append(f"Input '{inp}' not in values")

    # 3. Check all outputs are in values
    for out in graph["outputs"]:
        if out in graph["values"]:
            passed += 1
        else:
            failed += 1
            errors.append(f"Output '{out}' not in values")

    # 4. Check all node inputs/outputs reference valid values
    for node in graph["nodes"]:
        for inp in node["inputs"]:
            if inp in graph["values"]:
                passed += 1
            else:
                failed += 1
                errors.append(f"Node '{node['name']}' input '{inp}' not in values")

        for out in node["outputs"]:
            if out in graph["values"]:
                passed += 1
            else:
                failed += 1
                errors.append(f"Node '{node['name']}' output '{out}' not in values")

    # 5. Check weight files exist and can be loaded
    weight_attrs = ["weight", "bias", "running_mean", "running_var"]
    for node in graph["nodes"]:
        attrs = node.get("attrs", {})
        for attr_name in weight_attrs:
            if attr_name in attrs and isinstance(attrs[attr_name], dict):
                weight_meta = attrs[attr_name]
                if "path" in weight_meta:
                    weight_path = os.path.join(out_dir, weight_meta["path"])
                    if os.path.exists(weight_path):
                        # Try loading the binary
                        try:
                            dtype = np.dtype(weight_meta["dtype"])
                            data = np.fromfile(weight_path, dtype=dtype)
                            expected_size = np.prod(weight_meta["shape"])
                            if len(data) == expected_size:
                                passed += 1
                            else:
                                failed += 1
                                errors.append(
                                    f"Node '{node['name']}' {attr_name}: size mismatch "
                                    f"(expected {expected_size}, got {len(data)})"
                                )
                        except Exception as e:
                            failed += 1
                            errors.append(f"Node '{node['name']}' {attr_name}: load error - {e}")
                    else:
                        failed += 1
                        errors.append(f"Node '{node['name']}' {attr_name}: file not found - {weight_path}")

    # 6. Check values have required fields
    for name, value in graph["values"].items():
        if "id" in value and "shape" in value and "dtype" in value:
            passed += 1
        else:
            failed += 1
            errors.append(f"Value '{name}' missing required fields")

    return passed, failed, errors


def test_model(model_name: str, model, example_input: tuple, out_dir: str):
    """Test lowering for a model."""
    print(f"\n{'='*60}")
    print(f"Testing lowering: {model_name}")
    print(f"{'='*60}")

    model.eval()

    # 1. Trace and parse
    gm = fx.symbolic_trace(model)
    g = TorchParser.parse_fx(gm, example_input)
    ShapeInference.run(g)

    # 2. Run passes
    Passer.run(g)

    # 3. Export
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    translator = Translator(out_dir)
    translator.export(g)

    # 4. Verify
    passed, failed, errors = verify_graph_json(out_dir)

    print(f"\nVerification: {passed} passed, {failed} failed")

    if errors:
        print("\n--- Errors ---")
        for e in errors[:10]:  # Show first 10 errors
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    # 5. Summary stats
    with open(os.path.join(out_dir, "graph.json"), "r") as f:
        graph = json.load(f)

    print(f"\n--- Summary ---")
    print(f"  Nodes: {len(graph['nodes'])}")
    print(f"  Values: {len(graph['values'])}")
    print(f"  Inputs: {graph['inputs']}")
    print(f"  Outputs: {graph['outputs']}")

    # Count weight files
    weight_dir = os.path.join(out_dir, "weights")
    const_dir = os.path.join(out_dir, "constants")
    weight_count = len(os.listdir(weight_dir)) if os.path.exists(weight_dir) else 0
    const_count = len(os.listdir(const_dir)) if os.path.exists(const_dir) else 0
    print(f"  Weight files: {weight_count}")
    print(f"  Constant files: {const_count}")

    return passed, failed


def main():
    print("=" * 60)
    print("Lowering Test Suite")
    print("Verifying JSON + binary export")
    print("=" * 60)

    total_passed = 0
    total_failed = 0

    # Test 1: ResNet18
    model = resnet18(weights=None)
    example_input = (torch.randn(1, 3, 224, 224),)
    p, f = test_model("ResNet18", model, example_input, "test_output/resnet18")
    total_passed += p
    total_failed += f

    # Test 2: ResNet50
    model = resnet50(weights=None)
    example_input = (torch.randn(1, 3, 224, 224),)
    p, f = test_model("ResNet50", model, example_input, "test_output/resnet50")
    total_passed += p
    total_failed += f

    # Summary
    print("\n" + "=" * 60)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    print("=" * 60)

    # Cleanup
    import shutil
    if os.path.exists("test_output"):
        shutil.rmtree("test_output")

    return total_failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
