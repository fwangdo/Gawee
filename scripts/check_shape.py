"""
Diagnostic script for Gawee IR pipeline.
Checks each stage: FX tracing, ShapeProp, parsing, attr extraction.
"""

import argparse
import torch
import torch.fx as fx
from torchvision.models import resnet18
import segmentation_models_pytorch as smp

from gawee_ir.parser import TorchParser
from gawee_ir.constant.ops import TENSOR_META


def load_model(model_name: str):
    if model_name == "resnet":
        model = resnet18(weights=None)
        example_input = (torch.randn(1, 3, 224, 224),)
    elif model_name == "unet":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        example_input = (torch.randn(1, 3, 224, 224),)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()
    return model, example_input


def check_fx_trace(model):
    """Stage 1: FX symbolic trace."""
    print("\n" + "=" * 60)
    print("Stage 1: FX Symbolic Trace")
    print("=" * 60)

    gm = fx.symbolic_trace(model)

    print(f"Graph nodes: {len(list(gm.graph.nodes))}")
    print("\nNode summary:")
    op_counts = {}
    for node in gm.graph.nodes:
        op_counts[node.op] = op_counts.get(node.op, 0) + 1
    for op, count in sorted(op_counts.items()):
        print(f"  {op}: {count}")

    return gm


def check_shape_prop(gm, example_input):
    """Stage 2: PyTorch ShapeProp."""
    print("\n" + "=" * 60)
    print("Stage 2: PyTorch ShapeProp")
    print("=" * 60)

    from torch.fx.passes.shape_prop import ShapeProp
    ShapeProp(gm).propagate(*example_input)

    has_shape = 0
    no_shape = 0
    for node in gm.graph.nodes:
        tm = node.meta.get(TENSOR_META)
        if tm is not None:
            has_shape += 1
        else:
            no_shape += 1

    print(f"Nodes with tensor_meta: {has_shape}")
    print(f"Nodes without tensor_meta: {no_shape}")

    return gm


def check_parsing(gm, example_input):
    """Stage 3: Gawee IR parsing."""
    print("\n" + "=" * 60)
    print("Stage 3: Gawee IR Parsing")
    print("=" * 60)

    g = TorchParser.parse_fx(gm, example_input)

    print(f"Parsed nodes: {len(g.nodes)}")
    print(f"Values: {len(g.values)}")
    print(f"Inputs: {len(g.inputs)}")
    print(f"Outputs: {len(g.outputs)}")

    # Op type distribution
    print("\nOp type distribution:")
    op_counts = {}
    for node in g.nodes:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op}: {count}")

    # Check for None shapes
    none_shapes = []
    for name, value in g.values.items():
        if value.shape is None:
            none_shapes.append(name)

    print(f"\nValues with None shape: {len(none_shapes)}")
    if none_shapes:
        for name in none_shapes[:20]:
            print(f"  {name}")
        if len(none_shapes) > 20:
            print(f"  ... and {len(none_shapes) - 20} more")

    return g


def check_attrs(g):
    """Stage 4: Attribute extraction check."""
    print("\n" + "=" * 60)
    print("Stage 4: Attribute Extraction")
    print("=" * 60)

    seen_ops = set()
    for node in g.nodes:
        if node.op_type in seen_ops:
            continue
        seen_ops.add(node.op_type)

        print(f"\n[{node.op_type}] {node.name}")
        print(f"  call_type: {node.call_type}")
        for key, val in node.attrs.items():
            if key in ("weight", "bias", "running_mean", "running_var", "mod"):
                print(f"  {key}: <tensor/module>")
            else:
                print(f"  {key}: {val}")


def check_shapes(g):
    """Stage 5: Shape summary."""
    print("\n" + "=" * 60)
    print("Stage 5: Shape Summary")
    print("=" * 60)

    print("\nNode shapes (input -> output):")
    for node in g.nodes[:20]:
        in_shapes = [str(v.shape) for v in node.inputs]
        out_shapes = [str(v.shape) for v in node.outputs]
        print(f"  [{node.op_type}] {node.name}")
        print(f"    in:  {', '.join(in_shapes) if in_shapes else '(none)'}")
        print(f"    out: {', '.join(out_shapes)}")

    if len(g.nodes) > 20:
        print(f"  ... and {len(g.nodes) - 20} more nodes")


def main():
    parser = argparse.ArgumentParser(description="Gawee IR pipeline diagnostic")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet",
        choices=["resnet", "unet"],
        help="Model to check (resnet | unet)",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=0,
        help="Run up to stage N (0=all, 1-5 for specific stages)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"Gawee IR Pipeline Diagnostic - {args.model.upper()}")
    print("=" * 60)

    model, example_input = load_model(args.model)

    # Stage 1: FX trace
    gm = check_fx_trace(model)
    if args.stage == 1:
        return

    # Stage 2: ShapeProp
    gm = check_shape_prop(gm, example_input)
    if args.stage == 2:
        return

    # Stage 3: Parsing (uses ShapeProp results)
    g = check_parsing(gm, example_input)
    if args.stage == 3:
        return

    # Stage 4: Attrs
    check_attrs(g)
    if args.stage == 4:
        return

    # Stage 5: Shapes
    check_shapes(g)

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)


if __name__ == "__main__":
    main()
