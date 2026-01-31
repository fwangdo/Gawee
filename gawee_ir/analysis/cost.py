# gawee_ir/analysis/cost.py

from __future__ import annotations
from typing     import *  

from gawee_ir.graph import Graph, Node, Value
from gawee_ir.constant.ops import *

from dataclasses    import dataclass
import torch.nn     as nn 
from gawee_ir.analysis.errors import DimensionError, NoneCaseError 
from gawee_ir.mapper          import *
 

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


    def __repr__(self) -> str: 
        temp = dict()
        temp["total_flops"] = self.total_flops 
        temp["total_bytes_read"] = self.total_bytes_read
        temp["total_bytes_write"] = self.total_bytes_write
        temp["known_nodes_flops"] = self.known_nodes_flops 
        temp["known_nodes_read"] = self.known_nodes_read 
        temp["known_nodes_write"] =self.known_nodes_write 
        temp["num_nodes"] = self.num_nodes 
        return str(temp) 


    def get_dict(self) -> Dict[str, Any]:
        temp = dict()
        temp["total_flops"] = self.total_flops 
        temp["total_bytes_read"] = self.total_bytes_read
        temp["total_bytes_write"] = self.total_bytes_write
        temp["known_nodes_flops"] = self.known_nodes_flops 
        temp["known_nodes_read"] = self.known_nodes_read 
        temp["known_nodes_write"] =self.known_nodes_write 
        temp["num_nodes"] = self.num_nodes 
        return temp 

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
    def init(
        cls,
        gm: fx.GraphModule,
    ) -> None:
        cls.gm = gm 
        return 


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
            # print(f'op_type -> {n.op_type}')
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
    def print_report(cls, g: Graph, topk: int = 10) -> None:
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


    # helper functions. 
    @staticmethod
    def _get_module(n: Node) -> nn.Module:
        assert n.attrs['op'] == CALL_MODULE, f'[ERROR]: {n} is not module' 
        return n.attrs['mod']

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

    # parsing functions. 
    @classmethod 
    def _check_is_conv(cls, n: Node) -> bool:
        mod = n.attrs["mod"]
        return isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d))


    @classmethod 
    def _parse_primitive(cls, n: Node) -> str:
        mod = n.attrs["mod"].__name__
        # print(f'mod -> {mod}, type -> {type(mod)}')

        if mod == "add":
            return ADD  
        elif mod == "sub": 
            return SUB
        elif mod == "mul":
            return MUL
        elif mod == "flatten":
            return FLATTEN

        print(f'attrs -> {n.attrs["target"]}')
        raise Exception(f'[ERROR] {n} is not supported')


    @classmethod
    def _check_is_module(cls, n: Node) -> bool:
        mod = n.attrs["mod"]
        return isinstance(mod, nn.Module)

    
    @classmethod
    def _parse_module_name(cls, n: Node) -> str:  
        mod = n.attrs["mod"]
        if isinstance(mod, CONV_TYPE):
            return CONV
        elif isinstance(mod, LINEAR_TYPE):
            return MATMUL
        elif isinstance(mod, BN_TYPE):
            return BATCH_NORM
        elif isinstance(mod, RELU_TYPE):
            return RELU
        elif isinstance(mod, MXPOOL_TYPE):
            return MAXPOOL
        elif isinstance(mod, AVGPOOL_TYPE):
            return AVGPOOL
        elif isinstance(mod, AD_AVGPOOL_TYPE):
            return AD_AVGPOOL

        raise Exception(f'[ERROR]: {mod} is not defined yet')


    @classmethod 
    def _refine_op(cls, n: Node) -> str: 
        # print(f'n -> {n.attrs}')
        # mod = gm.get_submodule(node.target)
        op = n.attrs["op"]

        if op == CALL_MODULE:
            res = cls._parse_module_name(n) 
        elif op == CALL_FUNCTION: 
            res = cls._parse_primitive(n)
        elif op == CALL_METHOD:
            raise Exception(f'[ERROR]: {n} is called by call method')
        else:
            raise Exception(f'[ERROR]: {n.op_type} is not supported yet. ')
        
        # print(f'refined operator -> {res}')
        return res 

    # ---------------- internal: flops ----------------
    # count all flops for each op. 

    @classmethod
    def _node_flops(cls, n: Node) -> int | None:
        """
        we only consider conv / matmul / elemenetwise as flops. 
        """
        # op = cls._refine_op(n)
        op = Mapper.translate(n.raw, cls.gm)
        # print(f'operators -> {op} ')

        if op == CONV:
            return cls._flops_conv(n)
        if op in { MATMUL }:  
            # print(f'result of matmul -> {cls._flops_matmul(n)}')
            return cls._flops_matmul(n)
        if op in { ADD, MUL, SUB, DIV }: 
            return cls._flops_elementwise(n)

        return 


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
    def _flop_multi_matmul(n: Node) -> int | None:
        a, b = n.inputs[0], n.inputs[1]
        out = n.outputs[0]

        # print(f'matmul: a -> {a}, b -> {b}, out -> {out}')
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
            raise DimensionError(MATMUL)
        if K1 != K2:
            raise DimensionError(MATMUL)

        # batch multiplier = product of prefix dims of output (out[:-2])
        batch = _numel(out.shape[:-2])
        if batch is None:
            return None

        # mul+add = 2 flops per MAC
        flops = batch * int(M) * int(N) * int(K1) * 2
        return flops


    @staticmethod
    def _flops_linear_matmul(n: Node) -> int | None:
        # Linear: y = x @ W^T (+ b)
        # input  : [..., in_features]
        # output : [..., out_features]

        assert len(n.inputs) == 1

        x = n.inputs[0]
        out = n.outputs[0]

        if x.shape is None or out.shape is None:
            raise NoneCaseError(n.op_type, x.shape is None, out.shape is None)

        if len(x.shape) < 2 or len(out.shape) < 2:
            raise DimensionError(n.op_type)

        mod = CostModel._get_module(n)  # nn.Linear
        assert isinstance(mod, nn.Linear), f'[ERRRO]: {mod} is not linear.'
        in_features = mod.in_features
        out_features = mod.out_features

        # batch = product of prefix dims (everything except last dim)
        batch = _numel(x.shape[:-1])
        if batch is None:
            return None

        # MACs = batch * in_features * out_features
        # 1 MAC = 1 mul + 1 add = 2 FLOPs
        flops = batch * int(in_features) * int(out_features) * 2
        # print(f'flops -> {flops}')

        # print(f'node -> {n.name}, {n}, {n.attrs}')
        return flops


    @staticmethod
    def _flops_matmul(n: Node) -> int | None:
        assert len(n.inputs) > 0, f'[ERROR]'

        if len(n.inputs) >= 2: 
            return CostModel._flop_multi_matmul(n) 
        else:
            return CostModel._flops_linear_matmul(n) 


    @staticmethod
    def _flops_conv(n: Node) -> int | None:
        # input activation
        x = n.inputs[0]
        y = n.outputs[0]
        conv = n.attrs["mod"]  # nn.Conv2d
        assert isinstance(conv, nn.Conv2d), f'[ERROR] conv error. conv -> {conv} / {type(conv)} '
       
        # print()
        # print(f'x -> {x} / {type(x)}')
        # print(f'y -> {y} / {type(y)}')

        if x.shape is None or y.shape is None:
            return None
        if len(x.shape) != 4 or len(y.shape) != 4:
            return None

        N, Cin, _, _ = x.shape
        _, Cout, Hout, Wout = y.shape

        Kh, Kw = conv.kernel_size
        groups = conv.groups

        # MACs per output element
        mac_per_out = (Cin // groups) * Kh * Kw

        out_elems = N * Cout * Hout * Wout
        flops = out_elems * mac_per_out * 2
        return flops