# gawee_ir/analysis/cost.py

from __future__ import annotations
from typing     import *  

from gawee_ir.graph import Graph, Node, Value
from gawee_ir.constant.ops import *

from dataclasses    import dataclass

# data structure
@dataclass
class NodeCost: 
    index: int  
    name: str | None 
    op_type: str 
    flops: int | None
    bytes_read: int | None
    bytes_write: int | None 


@dataclass
class AllCost:  
    total_flops: int 
    total_bytes_read: int
    total_bytes_write: int
    known_nodes_flops: int
    known_nodes_read: int
    known_nodes_write: int
    num_nodes: int
    per_node: List[NodeCost]


# --------- dtype handling ---------

_DTYPE_BYTES: Dict[str, int] = {
    # common numpy-style strings
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "bool": 1,

    # sometimes you may store elem_type as stringified enum/int (best-effort)
    "1": 4,   # TensorProto.FLOAT
    "10": 2,  # TensorProto.FLOAT16
    "11": 8,  # TensorProto.DOUBLE
    "6": 4,   # TensorProto.INT32
    "7": 8,   # TensorProto.INT64
    "3": 1,   # TensorProto.INT8
    "2": 1,   # TensorProto.UINT8
    "9": 1,   # TensorProto.BOOL
}

_DEFAULT_DTYPE_BYTES = 4  # assume float32


def _dtype_bytes(dtype: str | None) -> int:
    if dtype is None:
        return _DEFAULT_DTYPE_BYTES
    key = str(dtype).strip().lower()
    return _DTYPE_BYTES.get(key, _DEFAULT_DTYPE_BYTES)


def _numel(shape: List[int] | None) -> int | None:
    if shape is None:
        return None
    prod = 1
    for d in shape:
        if d is None:
            return None
        if d < 0:
            return None  # unknown/symbolic
        prod *= int(d)
    return prod


def _tensor_bytes(v: Value) -> int | None:
    n = _numel(v.shape)
    if n is None:
        return None
    return n * _dtype_bytes(v.dtype)


def _safe_mul(*xs: int | None) -> int | None:
    out = 1
    for x in xs:
        if x is None:
            return None
        out *= x
    return out


# --------- cost model ---------

class CostModel:
    """
    Reports:
      - flops: integer count of floating point ops (approx; mul+add counted as 2)
      - bytes_read / bytes_write: tensor traffic estimate
    Notes:
      - This is a graph-level estimate, not a hardware-accurate model.
      - Unknown shapes propagate as None and will be skipped in totals.
    """

    @classmethod
    def run(cls, g: Graph) -> AllCost:
        per_node: List[NodeCost] = []

        total_flops = 0
        total_r = 0
        total_w = 0

        known_flops = 0
        known_r = 0
        known_w = 0

        for idx, n in enumerate(g.nodes):
            flops = cls._node_flops(n)
            r, w = cls._node_bytes(n)

            rec = NodeCost(
                idx, n.name, n.op_type, flops, r, w
            )
            per_node.append(rec)

            if flops is not None:
                total_flops += flops
                known_flops += 1
            if r is not None:
                total_r += r
                known_r += 1
            if w is not None:
                total_w += w
                known_w += 1

        return AllCost(total_flops, 
                       total_r, 
                       total_w, 
                       known_flops, 
                       known_r, 
                       known_w, 
                       len(g.nodes), 
                       per_node)

    @classmethod
    def print_report(cls, g: Graph, topk: int = 20) -> None:
        report = cls.run(g)

        print("=== Cost Report ===")
        print(f"Nodes: {report.num_nodes}")
        print(f"Total FLOPs (known only): {report.total_flops}")
        print(f"Total Read  (known only): {report.total_bytes_read} bytes")
        print(f"Total Write (known only): {report.total_bytes_write} bytes")
        print(
            f"Coverage: flops={report.known_nodes_flops}/{report.num_nodes}, "
            f"read={report.known_nodes_read}/{report.num_nodes}, "
            f"write={report.known_nodes_write}/{report.num_nodes}"
        )

        # top-k by flops
        nodes = [r for r in report.per_node if r.flops is not None]
        nodes.sort(key=lambda r: r.flops, reverse=True) # type: ignore 

        print(f"\n[Top {min(topk, len(nodes))} nodes by FLOPs]")
        for r in nodes[:topk]:
            nm = r.name if r.name else "-"
            print(
                f"  ({r.index:4d}) {r.op_type:<20} name={nm:<20} "
                f"flops={r.flops:<12} read={r.bytes_read} write={r.bytes_write}"
            )

    # ---------------- internal: bytes ----------------

    @classmethod
    def _node_bytes(cls, n: Node) -> Tuple[int | None, int | None]:
        # Conservative estimate: sum input tensor sizes (read) + sum output tensor sizes (write)
        # We do NOT attempt to deduplicate shared inputs/outputs across nodes.
        reads: List[int] = []
        writes: List[int] = []

        for v in n.inputs:
            b = _tensor_bytes(v)
            if b is None:
                return (None, None)
            reads.append(b)

        for v in n.outputs:
            b = _tensor_bytes(v)
            if b is None:
                return (None, None)
            writes.append(b)

        return (sum(reads), sum(writes))

    # ---------------- internal: flops ----------------
    # count all flops for each op. 

    @classmethod
    def _node_flops(cls, n: Node) -> int | None:
        op = n.op_type

        if op == CONV:
            return cls._flops_conv(n)
        if op in { MATMUL, GEMM }:  
            return cls._flops_matmul(n)
        if op in { ADD, MUL, SUB, DIV }: 
            return cls._flops_elementwise(n)

        # shape-preserving unary ops: ~1 op per element (very rough)
        # if op in ("Relu", "Sigmoid", "Tanh", "Identity", "BatchNormalization"):
        if op in { RELU, SIGMOID, TANH, ID, BATCH_NORM }:
            return cls._flops_unary(n)

        # unknown: skip
        # raise Exception(f'[ERROR]: {op} is not supported yet')
        return 


    @staticmethod
    def _flops_unary(n: Node) -> int | None:
        if not n.outputs:
            return 
        out = n.outputs[0]
        elems = _numel(out.shape)
        if elems is None:
            return 

        # Rough: 1 op per element (Relu/Identity), but Sigmoid/Tanh are more expensive.
        # Keep it simple and consistent.
        return elems


    @staticmethod
    def _flops_elementwise(n: Node) -> int | None:
        if not n.outputs:
            return None
        out = n.outputs[0]
        elems = _numel(out.shape)
        if elems is None:
            return None

        # One op per output element (very rough; ignores broadcast overhead)
        return elems


    @staticmethod
    def _flops_matmul(n: Node) -> int | None:
        # note that, we do not consider addition part in GEMM op. 
        if len(n.inputs) < 2 or not n.outputs:
            return # error case. we cannot count.  

        a, b = n.inputs[0], n.inputs[1]
        out = n.outputs[0]
        if a.shape is None or b.shape is None or out.shape is None:
            return None

        # Use ONNX MatMul convention: A[..., M, K] x B[..., K, N] -> Out[..., M, N]
        if len(a.shape) < 2 or len(b.shape) < 2 or len(out.shape) < 2:
            return None

        M = a.shape[-2]
        K1 = a.shape[-1]
        K2 = b.shape[-2]
        N = b.shape[-1]

        if any(d < 0 for d in (M, K1, K2, N)):
            return None
        if K1 != K2:
            return None

        # batch multiplier = product of prefix dims of output (out[:-2])
        batch = _numel(out.shape[:-2])
        if batch is None:
            return None

        # mul+add = 2 flops per MAC
        flops = batch * int(M) * int(N) * int(K1) * 2
        return flops


    @staticmethod
    def _flops_conv(n: Node) -> int | None:
        # X[N, Cin, H, W], W[Cout, Cin/groups, Kh, Kw] -> Y[N, Cout, Hout, Wout]
        if len(n.inputs) < 2 or not n.outputs:
            return None

        x, w = n.inputs[0], n.inputs[1]
        y = n.outputs[0]

        if x.shape is None or w.shape is None or y.shape is None:
            return None
        if len(x.shape) != 4 or len(w.shape) != 4 or len(y.shape) != 4:
            return None

        N, Cin, _, _ = x.shape
        Cout, Cin_per_g, Kh, Kw = w.shape
        _, _, Hout, Wout = y.shape

        if any(d < 0 for d in (N, Cin, Cout, Cin_per_g, Kh, Kw, Hout, Wout)):
            return None

        groups = int(n.attrs.get("group", 1))
        if groups <= 0:
            groups = 1

        # Consistency: Cin_per_g should be Cin/groups, but some exports may omit/leave unknown.
        # We'll trust weight shape primarily.
        # MACs per output element = Cin_per_g * Kh * Kw
        mac_per_out = int(Cin_per_g) * int(Kh) * int(Kw)

        # number of output elements = N * Cout * Hout * Wout
        out_elems = int(N) * int(Cout) * int(Hout) * int(Wout)

        # mul+add counted as 2 flops per MAC
        flops = out_elems * mac_per_out * 2
        return flops
