from __future__ import annotations
from typing import List, Optional

import numpy as np
import onnx
from onnx import helper

from ..utils import cons
from .folder import Folder


class ConvertMatmul(Folder):
    """Convert matmul -> 1 by 1 conv"""

    # case for not static weight. 
    def _convert_dynamic(
        self, prefix: str, a_name: str, b_name: str, out_name: str,
        a_shape: Optional[List], b_shape: Optional[List],
        graph: onnx.GraphProto,
    ) -> List[onnx.NodeProto] | None:
        """Convert dynamic MatMul A @ B using Unsqueeze + Mul + ReduceMean * K.

        A[..., M, K] @ B[..., K, N] → [..., M, N]
          1. Unsqueeze A at last dim:      [..., M, K, 1]
          2. Unsqueeze B before K dim:     [..., 1, K, N]
          3. Mul (broadcast):              [..., M, K, N]
          4. ReduceMean(axis=K) * K:       [..., M, N]
        """
        if a_shape is None or b_shape is None:
            return

        ndim_a = len(a_shape)
        ndim_b = len(b_shape)

        K = a_shape[-1]
        if not isinstance(K, int) or K <= 0:
            self.log.append(
                f" - MatMul({prefix}) could not be converted "
                f"(reduction axis K must be statically known, got {K!r})"
            )
            return None

        nodes: List[onnx.NodeProto] = []

        # Unsqueeze A at last dim: [..., M, K] → [..., M, K, 1]
        a_axes_name = self.tensor_name(prefix, "a_unsqueeze_axes")
        self.add_init(graph, a_axes_name, np.array([ndim_a], dtype=np.int64))
        a_us = self.tensor_name(prefix, "a_unsqueezed")
        nodes.append(helper.make_node(cons.OP_UNSQUEEZE, [a_name, a_axes_name], [a_us],
                                      name=self.node_name(prefix, "a_unsqueeze")))

        # Unsqueeze B before K dim: [..., K, N] → [..., 1, K, N]
        b_axes_name = self.tensor_name(prefix, "b_unsqueeze_axes")
        self.add_init(graph, b_axes_name, np.array([ndim_b - 2], dtype=np.int64))
        b_us = self.tensor_name(prefix, "b_unsqueezed")
        nodes.append(helper.make_node(cons.OP_UNSQUEEZE, [b_name, b_axes_name], [b_us],
                                      name=self.node_name(prefix, "b_unsqueeze")))

        # Mul (broadcast): [..., M, K, N]
        mul_out = self.tensor_name(prefix, "mul")
        nodes.append(helper.make_node(cons.OP_MUL, [a_us, b_us], [mul_out],
                                      name=self.node_name(prefix, "mul")))

        # ReduceMean over K axis (second-to-last in result)
        result_ndim = max(ndim_a + 1, ndim_b + 1)
        k_axis = result_ndim - 2

        mean_out = self.tensor_name(prefix, "mean")
        nodes.append(helper.make_node(cons.OP_REDUCE_MEAN, [mul_out], [mean_out],
                                      name=self.node_name(prefix, "reduce_mean"), axes=[k_axis], keepdims=0))

        # Mul by K to get sum
        k_name = self.tensor_name(prefix, "k_const")
        self.add_init(graph, k_name, np.array(K, dtype=np.float32))
        nodes.append(helper.make_node(cons.OP_MUL, [mean_out, k_name], [out_name],
                                      name=self.node_name(prefix, "scale")))

        return nodes


    @staticmethod
    def _squeeze_to_2d(weight: np.ndarray) -> np.ndarray | None:
        """Convert into 2-dimension shape"""
        if weight.ndim == 2:
            return weight
        if all(d == 1 for d in weight.shape[:-2]):
            return weight.reshape(weight.shape[-2], weight.shape[-1])
        return 


    # build. 
    def _build_2d(
        self, prefix: str, a_name: str, conv_w_name: str, out_name: str, N: int,
        graph: onnx.GraphProto,
    ) -> List[onnx.NodeProto]:
        nodes: List[onnx.NodeProto] = []

        rs1_shape_name = self.tensor_name(prefix, "reshape1_shape")
        rs2_shape_name = self.tensor_name(prefix, "reshape2_shape")

        # -1 means auto-calculation for shape. 
        self.add_init(graph, rs1_shape_name, np.array([1, 0, -1, 1], dtype=np.int64))
        self.add_init(graph, rs2_shape_name, np.array([-1, N], dtype=np.int64))

        # reshape -> [1, K, M, 1]
        rs1_out = self.tensor_name(prefix, "reshape1")
        nodes.append(helper.make_node(cons.OP_RESHAPE, [a_name, rs1_shape_name], [rs1_out],
                                      name=self.node_name(prefix, "reshape1")))

        # shape of conv weight -> [N, K, 1, 1] 
        # done, it would be [1, N, M, 1]
        conv_out = self.tensor_name(prefix, "conv_out")
        nodes.append(helper.make_node(cons.OP_CONV, [rs1_out, conv_w_name], [conv_out],
                                      name=self.node_name(prefix, "conv"), kernel_shape=[1, 1]))

        # Transpose ,[1, M, N, 1]
        t2_out = self.tensor_name(prefix, "transpose2")
        nodes.append(helper.make_node(cons.OP_TRANSPOSE, [conv_out], [t2_out],
                                      name=self.node_name(prefix, "transpose2"), perm=[0, 2, 1, 3]))

        # [M, N], which is what we are looking for. 
        nodes.append(helper.make_node(cons.OP_RESHAPE, [t2_out, rs2_shape_name], [out_name],
                                      name=self.node_name(prefix, "reshape2")))
        return nodes


    def _build_3d(
        self, prefix: str, a_name: str, conv_w_name: str, out_name: str, N: int,
        graph: onnx.GraphProto,
    ) -> List[onnx.NodeProto]:
        # grouped conv. 
        nodes: List[onnx.NodeProto] = []

        # transpose -> [B, K, M]
        t1_out = self.tensor_name(prefix, "transpose1")
        nodes.append(helper.make_node(cons.OP_TRANSPOSE, [a_name], [t1_out],
                                      name=self.node_name(prefix, "transpose1"), perm=[0, 2, 1]))

        # unsqueeze -> [B, K, M, 1]
        axes_name = self.tensor_name(prefix, "unsqueeze_axes")
        self.add_init(graph, axes_name, np.array([3], dtype=np.int64))
        us_out = self.tensor_name(prefix, "unsqueezed")
        nodes.append(helper.make_node(cons.OP_UNSQUEEZE, [t1_out, axes_name], [us_out],
                                      name=self.node_name(prefix, "unsqueeze")))

        # result -> [B, N, M, 1]
        conv_out = self.tensor_name(prefix, "conv_out")
        nodes.append(helper.make_node(cons.OP_CONV, [us_out, conv_w_name], [conv_out],
                                      name=self.node_name(prefix, "conv"), kernel_shape=[1, 1]))

        # transpose -> [B, M, N, 1]
        t2_out = self.tensor_name(prefix, "transpose2")
        nodes.append(helper.make_node(cons.OP_TRANSPOSE, [conv_out], [t2_out],
                                      name=self.node_name(prefix, "transpose2"), perm=[0, 2, 1, 3]))

        # copy and auto calc -> [B, M, ]
        rs_shape_name = self.tensor_name(prefix, "reshape_shape")
        self.add_init(graph, rs_shape_name, np.array([0, 0, -1], dtype=np.int64))
        nodes.append(helper.make_node(cons.OP_RESHAPE, [t2_out, rs_shape_name], [out_name],
                                      name=self.node_name(prefix, "reshape")))
        return nodes


    def _build_4d(
        self, prefix: str, a_name: str, conv_w_name: str, out_name: str,
        N: int, K: int, a_shape: List, graph: onnx.GraphProto,
    ) -> Optional[List[onnx.NodeProto]]:
        nodes: List[onnx.NodeProto] = []
        H_val = a_shape[1]
        M_val = a_shape[2]

        rs1_shape_name = self.tensor_name(prefix, "reshape1_shape")
        m = M_val if isinstance(M_val, int) and M_val > 0 else 0
        self.add_init(graph, rs1_shape_name, np.array([-1, m or 0, K], dtype=np.int64))
        rs1_out = self.tensor_name(prefix, "reshape1")
        nodes.append(helper.make_node(cons.OP_RESHAPE, [a_name, rs1_shape_name], [rs1_out],
                                      name=self.node_name(prefix, "reshape1")))

        inner_out = self.tensor_name(prefix, "inner3d_out")
        inner = self._build_3d(f"{prefix}_3d", rs1_out, conv_w_name, inner_out, N, graph)
        nodes.extend(inner)

        rs2_shape_name = self.tensor_name(prefix, "reshape2_shape")
        if isinstance(H_val, int) and isinstance(M_val, int):
            self.add_init(graph, rs2_shape_name, np.array([-1, H_val, M_val, N], dtype=np.int64))
        else:
            self.add_init(graph, rs2_shape_name, np.array([-1, 0, 0, N], dtype=np.int64))
        nodes.append(helper.make_node(cons.OP_RESHAPE, [inner_out, rs2_shape_name], [out_name],
                                      name=self.node_name(prefix, "reshape2")))
        return nodes


    def _convert_right_static(
        self, prefix: str, a_name: str, weight: np.ndarray,
        out_name: str, a_shape: List, graph: onnx.GraphProto,
    ) -> Optional[List[onnx.NodeProto]]:
        """Right case. X @ W case. """
        K, N = weight.shape
        ndim = len(a_shape)

        conv_weight = weight.T.reshape(N, K, 1, 1).astype(np.float32)
        conv_weight_name = self.tensor_name(prefix, "conv_weight")
        self.add_init(graph, conv_weight_name, conv_weight)

        if ndim == 2:
            return self._build_2d(prefix, a_name, conv_weight_name, out_name, N, graph)
        elif ndim == 3:
            return self._build_3d(prefix, a_name, conv_weight_name, out_name, N, graph)
        elif ndim == 4:
            return self._build_4d(prefix, a_name, conv_weight_name, out_name, N, K, a_shape, graph)
        return None


    def _convert_left_static(
        self, prefix: str, b_name: str, weight: np.ndarray,
        out_name: str, b_shape: List, graph: onnx.GraphProto,
    ) -> Optional[List[onnx.NodeProto]]:
        M, K = weight.shape
        ndim = len(b_shape)
        if ndim not in (2, 3):
            return None

        wt = weight.T.astype(np.float32)
        wt_name = self.tensor_name(prefix, "weight_transposed")
        self.add_init(graph, wt_name, wt)

        perm = [1, 0] if ndim == 2 else [0, 2, 1]
        nodes: List[onnx.NodeProto] = []

        bt_out = self.tensor_name(prefix, "b_transposed")
        nodes.append(helper.make_node(cons.OP_TRANSPOSE, [b_name], [bt_out],
                                      name=self.node_name(prefix, "transpose_b"), perm=perm))

        K2, M2 = wt.shape
        conv_w = wt.T.reshape(M2, K2, 1, 1).astype(np.float32)
        conv_w_name = self.tensor_name(prefix, "inner_conv_weight")
        self.add_init(graph, conv_w_name, conv_w)

        temp_out = self.tensor_name(prefix, "inner_temp")
        if ndim == 2:
            inner = self._build_2d(f"{prefix}_inner", bt_out, conv_w_name, temp_out, M2, graph)
        else:
            inner = self._build_3d(f"{prefix}_inner", bt_out, conv_w_name, temp_out, M2, graph)
        if inner is None:
            self.log.append(f" - MatMul({prefix}) could not be converted (left-static inner build failed)")
            return None
        nodes.extend(inner)

        nodes.append(helper.make_node(cons.OP_TRANSPOSE, [temp_out], [out_name],
                                      name=self.node_name(prefix, "transpose_final"), perm=perm))
        return nodes


    def _gen_new_nodes(self, 
                        a_is_init: bool, 
                        b_is_init: bool,
                        a_name: str, 
                        b_name: str, 
                        prefix: str, 
                        out_name: str, 
                       ) -> List[onnx.NodeProto] | None:
        """Generate new nodes to replace matmul 
        Args:
            a_is_init: whether a is weight or not  
            b_is_init: whether b is weight or not  
            a_name: name of node a
            b_name: name of node b 
            prefix: the name of this process. It will be used in name. 
            out_name: 
        """
        if not a_is_init and not b_is_init:
            # dynamic MatMul: Unsqueeze + Mul + ReduceMean
            a_shape = self.shape_info.get(a_name)
            b_shape = self.shape_info.get(b_name)
            new_nodes = self._convert_dynamic(
                prefix, a_name, b_name, out_name, a_shape, b_shape, self.graph
            )

            if new_nodes is None:
                self.log.append(f" - MatMul({prefix}) could not be converted (dynamic, shape unknown)")
                return 
        else:
            # static case. 
            if b_is_init:
                weight = self.init_map[b_name]
                activation_name = a_name # runtime value. 
            else:
                weight = self.init_map[a_name]
                activation_name = b_name # runtime value. 
            # weight and activation name are ready.

            a_shape = self.shape_info.get(activation_name)
            if a_shape is None:
                self.log.append(f" - MatMul({prefix}) could not be converted (shape unknown)")
                return None

            weight_2d = self._squeeze_to_2d(weight)
            if weight_2d is None:
                self.log.append(f" - MatMul({prefix}) could not be converted (unsupported weight shape)")
                return None
            elif b_is_init:
                new_nodes = self._convert_right_static(prefix, activation_name, weight_2d, out_name, a_shape, self.graph)
            else:
                new_nodes = self._convert_left_static(prefix, activation_name, weight_2d, out_name, a_shape, self.graph)

            if new_nodes is None:
                self.log.append(f" - MatMul({prefix}) could not be converted (static path build failed)")
                return None
        
        return new_nodes 


    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, List[str]]:
        self.prepare(model)

        for node in list(self.graph.node):
            if node.op_type != cons.OP_MATMUL:
                continue

            a_name = node.input[0]
            b_name = node.input[1]
            out_name = node.output[0]
            prefix = self.get_prefix(node)

            # find out that it is static or dynamic. 
            a_is_init = a_name in self.init_map
            b_is_init = b_name in self.init_map

            new_nodes: List[onnx.NodeProto] | None = self._gen_new_nodes(a_is_init, b_is_init, a_name,
                                                                  b_name, prefix, out_name)

            if new_nodes is None:
                self.log.append(f" - MatMul({prefix}) could not be converted (unsupported rank)")
                continue

            self.nodes_to_remove.append(node)
            for n in new_nodes:
                self.graph.node.append(n)

            conv_names = [n.name for n in new_nodes if n.op_type == cons.OP_CONV]
            if conv_names:
                self.log.append(f" - MatMul({prefix}) is converted to Conv({conv_names[0]})")
            else:
                self.log.append(f" - MatMul({prefix}) is converted to Mul+ReduceMean")

        for node in self.nodes_to_remove:
            self.graph.node.remove(node)
            self.deleted_node += 1

        return self.model, self.log
