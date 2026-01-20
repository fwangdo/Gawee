# scripts/compare_cost_before_after.py

from gawee_ir.parser import Parser
from gawee_ir.analysis.shape import ShapeInference
from gawee_ir.analysis.cost import CostModel
from gawee_ir.passes.conv_bn_folding import ConvBNFolding
from gawee_ir.passes.constant_folding import ConstantFolding


def print_summary(title: str, report: dict):
    print(f"\n=== {title} ===")
    print(f"Nodes                : {report['num_nodes']}")
    print(f"Total FLOPs          : {report['total_flops']}")
    print(f"Total Read  (bytes)  : {report['total_bytes_read']}")
    print(f"Total Write (bytes)  : {report['total_bytes_write']}")
    print(
        f"Coverage (F/R/W)     : "
        f"{report['known_nodes_flops']}/"
        f"{report['known_nodes_read']}/"
        f"{report['known_nodes_write']}"
    )


def main():
    # 1. Parse + shape inference
    g = Parser.parse_onnx("./onnxdata/resnet18.onnx")
    ShapeInference.run(g)

    # 2. Cost before optimization
    before = CostModel.run(g).get_dict()
    print_summary("BEFORE Conv+BN Folding", before)

    # 3. Apply optimization
    changed = ConvBNFolding.run(g)
    # changed = ConstantFolding.run(g)
    print(f"\nOptimization applied: {changed}")

    # 4. Re-run shape inference (safe)
    ShapeInference.run(g)

    # 5. Cost after optimization
    after = CostModel.run(g).get_dict()
    print_summary("AFTER Conv+BN Folding", after)

    # 6. Diff
    print("\n=== DIFF (After - Before) ===")
    print(f"Nodes Δ        : {after['num_nodes'] - before['num_nodes']}")
    print(f"FLOPs Δ        : {after['total_flops'] - before['total_flops']}")
    print(
        f"Read  bytes Δ  : {after['total_bytes_read'] - before['total_bytes_read']}"
    )
    print(
        f"Write bytes Δ  : {after['total_bytes_write'] - before['total_bytes_write']}"
    )


if __name__ == "__main__":
    main()
