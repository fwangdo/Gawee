# scripts/inspect_onnx.py
from __future__ import annotations

from pathlib import Path
import onnx
from onnx import numpy_helper


def fmt_shape(t):
    if not t.type.HasField("tensor_type"):
        return "N/A"
    shape = []
    for d in t.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            shape.append(str(d.dim_value))
        elif d.HasField("dim_param"):
            shape.append(d.dim_param)
        else:
            shape.append("?")
    return "(" + ", ".join(shape) + ")"


def inspect_onnx(onnx_path: Path) -> None:
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    graph = model.graph

    print("=" * 80)
    print(f"ONNX MODEL: {onnx_path}")
    print(f"IR VERSION : {model.ir_version}")
    print(f"OPSET     : {[opset.version for opset in model.opset_import]}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1) Graph inputs / outputs
    # ------------------------------------------------------------------
    print("\n[Graph Inputs]")
    for inp in graph.input:
        print(f"  - name={inp.name:30s} shape={fmt_shape(inp)}")

    print("\n[Graph Outputs]")
    for out in graph.output:
        print(f"  - name={out.name:30s} shape={fmt_shape(out)}")

    # ------------------------------------------------------------------
    # 2) Initializers (weights / constants)
    # ------------------------------------------------------------------
    print("\n[Initializers]")
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        print(
            f"  - name={init.name:30s} "
            f"dtype={arr.dtype} "
            f"shape={arr.shape}"
        )

    # ------------------------------------------------------------------
    # 3) Nodes (core compiler input)
    # ------------------------------------------------------------------
    print("\n[Nodes]")
    for i, node in enumerate(graph.node):
        print(f"\nNode #{i}")
        print(f"  op_type : {node.op_type}")
        print(f"  name    : {node.name or '(anonymous)'}")

        print("  inputs  :")
        for inp in node.input:
            print(f"    - {inp}")

        print("  outputs :")
        for out in node.output:
            print(f"    - {out}")

        if node.attribute:
            print("  attrs   :")
            for attr in node.attribute:
                # attribute type별 간단 출력
                if attr.type == onnx.AttributeProto.INT:
                    val = attr.i
                elif attr.type == onnx.AttributeProto.FLOAT:
                    val = attr.f
                elif attr.type == onnx.AttributeProto.INTS:
                    val = list(attr.ints)
                elif attr.type == onnx.AttributeProto.FLOATS:
                    val = list(attr.floats)
                elif attr.type == onnx.AttributeProto.STRING:
                    val = attr.s.decode("utf-8")
                else:
                    val = "(complex)"
                print(f"    - {attr.name}: {val}")

    print("\n[Summary]")
    print(f"  #inputs     : {len(graph.input)}")
    print(f"  #outputs    : {len(graph.output)}")
    print(f"  #nodes      : {len(graph.node)}")
    print(f"  #initializers: {len(graph.initializer)}")


if __name__ == "__main__":
    inspect_onnx(Path("onnx/resnet18.onnx"))
