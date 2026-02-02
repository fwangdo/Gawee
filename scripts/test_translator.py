"""
Test Translator.export() - verifies JSON + binary export works correctly.
Cleans up generated files after testing.
"""
from __future__ import annotations

import json
import os
import shutil
import numpy as np

import torch
import torch.fx as fx
from torchvision.models import resnet18

from gawee_ir.parser         import TorchParser
from gawee_ir.analysis.shape import ShapeInference
from gawee_ir.passes.passer  import Passer
from gawee_ir.translator     import Translator

TEST_DIR = "/tmp/gawee_test_translator"


def setup() -> None:
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)


def teardown() -> None:
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


def test_export() -> bool:
    model = resnet18(weights=None)
    model.eval()
    example_input = (torch.randn(1, 3, 224, 224),)

    gm = fx.symbolic_trace(model)
    g = TorchParser.parse_fx(gm, example_input)
    ShapeInference.run(g)
    Passer.run(g)

    translator = Translator(TEST_DIR)
    translator.export(g)
    translator.show_nodes()

    # Verify graph.json exists and is valid JSON.
    graph_path = os.path.join(TEST_DIR, "graph.json")
    assert os.path.exists(graph_path), "graph.json not found"

    with open(graph_path, "r") as f:
        graph = json.load(f)

    # Check structure.
    assert "inputs" in graph, "missing 'inputs'"
    assert "outputs" in graph, "missing 'outputs'"
    assert "values" in graph, "missing 'values'"
    assert "nodes" in graph, "missing 'nodes'"
    assert len(graph["nodes"]) > 0, "no nodes exported"

    # Check weight files exist.
    weight_dir = os.path.join(TEST_DIR, "weights")
    assert os.path.exists(weight_dir), "weights/ not found"
    weight_files = os.listdir(weight_dir)
    assert len(weight_files) > 0, "no weight files exported"

    # Check one weight file can be loaded.
    sample_weight = os.path.join(weight_dir, weight_files[0])
    data = np.fromfile(sample_weight, dtype=np.float32)
    assert len(data) > 0, "weight file is empty"

    print(f"[PASS] Exported {len(graph['nodes'])} nodes, {len(weight_files)} weight files")
    return True


def main() -> int:
    setup()
    try:
        test_export()
        print("[OK] All tests passed")
        return 0
    except AssertionError as e:
        print(f"[FAIL] {e}")
        return 1
    finally:
        teardown()


if __name__ == "__main__":
    import sys
    sys.exit(main())
