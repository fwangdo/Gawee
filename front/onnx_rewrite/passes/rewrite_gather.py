from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper
import math

from ..utils import cons
from .folder import Folder


@dataclass(frozen=True)
class GatherNodeContext:
    prefix: str
    axis: int
    data_name: str
    index_name: str
    output_name: str
    data_array: np.ndarray | None
    index_array: np.ndarray | None


class RewriteGather(Folder):
    """Rewrite Gather into a supported expression when the pattern is tractable."""

    def __init__(self) -> None:
        super().__init__()
        # Vocabulary size above which we stop emitting one branch per token
        # and switch to chunked lowering.
        self.chunk_threshold: int = 2000

        # Number of vocabulary rows grouped into one chunk in the chunked path.
        self.chunk_size: int = 256

    def _build_context(self, node: onnx.NodeProto) -> GatherNodeContext:
        """Collect cached metadata for one Gather node.

        Args:
            node: Gather node currently being inspected.

        Returns:
            Context object containing axis, input/output names, and any
            initializer-backed data or index arrays.
        """
        axis = 0
        for attr in node.attribute:
            if attr.name == "axis":
                axis = int(attr.i)

        data_name, index_name = node.input[:2]
        return GatherNodeContext(
            prefix=self.get_prefix(node),
            axis=axis,
            data_name=data_name,
            index_name=index_name,
            output_name=node.output[0],
            data_array=self.init_map.get(data_name),
            index_array=self.init_map.get(index_name),
        )

    def _rewrite_static_gather(
        self,
        node: onnx.NodeProto,
        context: GatherNodeContext,
        graph: onnx.GraphProto,
    ) -> bool:
        """Fold Gather into a constant initializer when both inputs are static.

        Args:
            node: Gather node to rewrite.
            context: Cached metadata for the node.
            graph: Graph being rewritten.

        Returns:
            True if the node was replaced by an initializer, otherwise False.
        """
        if context.data_array is None or context.index_array is None:
            return False

        folded = np.take(context.data_array, context.index_array.astype(np.int64), axis=context.axis)
        self.add_init(graph, context.output_name, folded)
        self.mark_for_removal(node)
        self.log.append(f" - Gather({context.prefix}) is rewritten as initializer fold")
        return True

    def _rewrite_vocab_gather(
        self,
        node: onnx.NodeProto,
        context: GatherNodeContext,
        graph: onnx.GraphProto,
    ) -> bool:
        """Rewrite embedding-style Gather into supported arithmetic.

        Args:
            node: Gather node to rewrite.
            context: Cached metadata for the node.
            graph: Graph being rewritten.

        Returns:
            True if one of the vocab-specific rewrites handled the node,
            otherwise False.
        """
        if context.data_array is None or context.axis != 0:
            return False

        vocab_size = context.data_array.shape[0]

        # To reduce compile time. 
        if vocab_size > self.chunk_threshold:
            new_nodes = self._emit_chunked_vocab_chain(context, graph)
            self.replace_node(node, new_nodes)
            chunk_count = math.ceil(vocab_size / self.chunk_size)
            self.log.append(
                f" - Gather({context.prefix}) is rewritten as chunked Equal+Mul+ReduceSum ({chunk_count} chunks)"
            )
            return True

        # default. 
        new_nodes = self._emit_small_vocab_chain(context, graph)
        self.replace_node(node, new_nodes)
        self.log.append(
            f" - Gather({context.prefix}) is rewritten as Equal+Where-style accumulation (V={vocab_size})"
        )
        return True

    def _emit_small_vocab_chain(
        self,
        context: GatherNodeContext,
        graph: onnx.GraphProto,
    ) -> list[onnx.NodeProto]:
        """Build one-branch-per-token lowering for a small embedding table.

        Args:
            context: Cached metadata for the source Gather node.
            graph: Graph that receives generated initializers.

        Returns:
            Replacement nodes implementing the small-vocab rewrite.
        """
        assert context.data_array is not None

        nodes: list[onnx.NodeProto] = []
        vocab_size = context.data_array.shape[0]
        if vocab_size == 0:
            return nodes

        unsqueeze_axes_name = self.tensor_name(context.prefix, "mask_unsqueeze_axes")
        self.add_init(graph, unsqueeze_axes_name, np.array([-1], dtype=np.int64))

        running_sum_name = self._emit_weighted_vocab_row(
            context=context,
            graph=graph,
            nodes=nodes,
            unsqueeze_axes_name=unsqueeze_axes_name,
            vocab_index=0,
        )

        for vocab_index in range(1, vocab_size):
            weighted_row_name = self._emit_weighted_vocab_row(
                context=context,
                graph=graph,
                nodes=nodes,
                unsqueeze_axes_name=unsqueeze_axes_name,
                vocab_index=vocab_index,
            )

            sum_output_name = (
                context.output_name
                if vocab_index == vocab_size - 1
                else self.tensor_name(context.prefix, f"running_sum_{vocab_index}")
            )
            nodes.append(
                helper.make_node(
                    cons.OP_ADD,
                    [running_sum_name, weighted_row_name],
                    [sum_output_name],
                    name=self.node_name(context.prefix, f"row_add_{vocab_index}"),
                )
            )
            running_sum_name = sum_output_name

        if vocab_size == 1 and running_sum_name != context.output_name:
            # With a single vocabulary row there is no final Add node to materialize
            # `context.output_name`, so emit a shape-preserving Reshape as a cheap
            # passthrough and bind the weighted row tensor to the original output.
            passthrough_shape_name = self.tensor_name(context.prefix, "single_vocab_shape")
            self.add_init(graph, passthrough_shape_name, np.array([0, 0], dtype=np.int64))
            nodes.append(
                helper.make_node(
                    cons.OP_RESHAPE,
                    [running_sum_name, passthrough_shape_name],
                    [context.output_name],
                    name=self.node_name(context.prefix, "single_vocab_reshape"),
                )
            )

        return nodes

    def _emit_chunked_vocab_chain(
        self,
        context: GatherNodeContext,
        graph: onnx.GraphProto,
    ) -> list[onnx.NodeProto]:
        """Build chunked lowering for larger embedding tables.

        Args:
            context: Cached metadata for the source Gather node.
            graph: Graph that receives generated initializers.

        Returns:
            Replacement nodes implementing the chunked-vocab rewrite.
        """
        assert context.data_array is not None

        vocab_size = context.data_array.shape[0]
        chunk_count = math.ceil(vocab_size / self.chunk_size)
        nodes: list[onnx.NodeProto] = []
        if chunk_count == 0:
            return nodes

        outer_unsqueeze_axes_name = self.tensor_name(context.prefix, "chunk_outer_unsqueeze_axes")
        self.add_init(graph, outer_unsqueeze_axes_name, np.array([-1], dtype=np.int64))
        index_vector_name = self.tensor_name(context.prefix, "chunk_index_vector")
        nodes.append(
            helper.make_node(
                cons.OP_UNSQUEEZE,
                [context.index_name, outer_unsqueeze_axes_name],
                [index_vector_name],
                name=self.node_name(context.prefix, "chunk_index_unsqueeze"),
            )
        )

        inner_unsqueeze_axes_name = self.tensor_name(context.prefix, "chunk_inner_unsqueeze_axes")
        self.add_init(graph, inner_unsqueeze_axes_name, np.array([-1], dtype=np.int64))

        running_sum_name = self._emit_reduced_chunk(
            context=context,
            graph=graph,
            nodes=nodes,
            index_vector_name=index_vector_name,
            inner_unsqueeze_axes_name=inner_unsqueeze_axes_name,
            chunk_index=0,
        )

        for chunk_index in range(1, chunk_count):
            reduced_chunk_name = self._emit_reduced_chunk(
                context=context,
                graph=graph,
                nodes=nodes,
                index_vector_name=index_vector_name,
                inner_unsqueeze_axes_name=inner_unsqueeze_axes_name,
                chunk_index=chunk_index,
            )

            sum_output_name = (
                context.output_name
                if chunk_index == chunk_count - 1
                else self.tensor_name(context.prefix, f"chunk_sum_{chunk_index}")
            )
            nodes.append(
                helper.make_node(
                    cons.OP_ADD,
                    [running_sum_name, reduced_chunk_name],
                    [sum_output_name],
                    name=self.node_name(context.prefix, f"chunk_add_{chunk_index}"),
                )
            )
            running_sum_name = sum_output_name

        if chunk_count == 1 and running_sum_name != context.output_name:
            passthrough_shape_name = self.tensor_name(context.prefix, "chunk_single_shape")
            self.add_init(graph, passthrough_shape_name, np.array([0], dtype=np.int64))
            nodes.append(
                helper.make_node(
                    cons.OP_RESHAPE,
                    [running_sum_name, passthrough_shape_name],
                    [context.output_name],
                    name=self.node_name(context.prefix, "chunk_single_reshape"),
                )
            )

        return nodes

    def _emit_weighted_vocab_row(
        self,
        context: GatherNodeContext,
        graph: onnx.GraphProto,
        nodes: list[onnx.NodeProto],
        unsqueeze_axes_name: str,
        vocab_index: int,
    ) -> str:
        """Emit the mask-and-weight subgraph for one vocabulary row.

        Args:
            context: Cached metadata for the source Gather node.
            graph: Graph that receives generated initializers.
            nodes: Node list being accumulated for the enclosing rewrite.
            unsqueeze_axes_name: Initializer name for the final mask unsqueeze.
            vocab_index: Vocabulary row emitted by this subgraph.

        Returns:
            Tensor name of the weighted row contribution.
        """
        assert context.data_array is not None

        equality_mask_name = self._emit_index_mask(
            prefix=context.prefix,
            index_name=context.index_name,
            vocab_index=vocab_index,
            graph=graph,
            nodes=nodes,
        )

        # for broadcast. 
        unsqueezed_mask_name = self.tensor_name(context.prefix, f"mask_{vocab_index}")
        nodes.append(
            helper.make_node(
                cons.OP_UNSQUEEZE,
                [equality_mask_name, unsqueeze_axes_name],
                [unsqueezed_mask_name],
                name=self.node_name(context.prefix, f"mask_unsqueeze_{vocab_index}"),
            )
        )

        row_name = self.tensor_name(context.prefix, f"row_{vocab_index}")
        self.add_init(graph, row_name, context.data_array[vocab_index].astype(np.float32))

        weighted_row_name = self.tensor_name(context.prefix, f"weighted_row_{vocab_index}")
        nodes.append(
            helper.make_node(
                cons.OP_MUL,
                [unsqueezed_mask_name, row_name],
                [weighted_row_name],
                name=self.node_name(context.prefix, f"row_mul_{vocab_index}"),
            )
        )
        return weighted_row_name

    def _emit_reduced_chunk(
        self,
        context: GatherNodeContext,
        graph: onnx.GraphProto,
        nodes: list[onnx.NodeProto],
        index_vector_name: str,
        inner_unsqueeze_axes_name: str,
        chunk_index: int,
    ) -> str:
        """Emit one chunked Gather contribution and reduce it to one tensor.

        Args:
            context: Cached metadata for the source Gather node.
            graph: Graph that receives generated initializers.
            nodes: Node list being accumulated for the enclosing rewrite.
            index_vector_name: Unsqueezed index tensor used for equality tests.
            inner_unsqueeze_axes_name: Initializer name for expanding chunk masks.
            chunk_index: Zero-based chunk number.

        Returns:
            Tensor name of the reduced contribution for `chunk_index`.
        """
        assert context.data_array is not None

        vocab_size = context.data_array.shape[0]
        start = chunk_index * self.chunk_size
        end = min(start + self.chunk_size, vocab_size)
        range_name = self.tensor_name(context.prefix, f"chunk_range_{chunk_index}")
        self.add_init(graph, range_name, np.arange(start, end, dtype=np.int64))

        # ranged eq node. 
        equality_name = self.tensor_name(context.prefix, f"chunk_equal_{chunk_index}")
        nodes.append(
            helper.make_node(
                cons.OP_EQUAL,
                [index_vector_name, range_name],
                [equality_name],
                name=self.node_name(context.prefix, f"chunk_equal_{chunk_index}"),
            )
        )

        float_mask_name = self.tensor_name(context.prefix, f"chunk_mask_{chunk_index}")
        nodes.append(
            helper.make_node(
                cons.OP_CAST,
                [equality_name],
                [float_mask_name],
                name=self.node_name(context.prefix, f"chunk_cast_{chunk_index}"),
                to=TensorProto.FLOAT,
            )
        )

        expanded_mask_name = self.tensor_name(context.prefix, f"chunk_mask_expanded_{chunk_index}")
        nodes.append(
            helper.make_node(
                cons.OP_UNSQUEEZE,
                [float_mask_name, inner_unsqueeze_axes_name],
                [expanded_mask_name],
                name=self.node_name(context.prefix, f"chunk_unsqueeze_{chunk_index}"),
            )
        )
        # the same to normal mask algorithm  

        chunk_weight_name = self.tensor_name(context.prefix, f"chunk_weight_{chunk_index}")
        self.add_init(graph, chunk_weight_name, context.data_array[start:end].astype(np.float32))

        weighted_chunk_name = self.tensor_name(context.prefix, f"chunk_weighted_{chunk_index}")
        nodes.append(
            helper.make_node(
                cons.OP_MUL,
                [expanded_mask_name, chunk_weight_name],
                [weighted_chunk_name],
                name=self.node_name(context.prefix, f"chunk_mul_{chunk_index}"),
            )
        )

        reduced_chunk_name = self.tensor_name(context.prefix, f"chunk_reduced_{chunk_index}")
        nodes.append(
            helper.make_node(
                cons.OP_REDUCE_SUM,
                [weighted_chunk_name],
                [reduced_chunk_name],
                name=self.node_name(context.prefix, f"chunk_reduce_sum_{chunk_index}"),
                axes=[-2],
                keepdims=0,
            )
        )
        return reduced_chunk_name

    def _emit_index_mask(
        self,
        prefix: str,
        index_name: str,
        vocab_index: int,
        graph: onnx.GraphProto,
        nodes: list[onnx.NodeProto],
    ) -> str:
        """Append nodes that build a float mask for one vocabulary row.

        Args:
            prefix: Stable prefix derived from the source Gather node.
            index_name: Tensor containing Gather indices.
            vocab_index: Vocabulary row tested by this mask.
            graph: Graph that receives generated initializers.
            nodes: Node list being accumulated for the enclosing rewrite.

        Returns:
            Tensor name of the float mask generated for `vocab_index`.
        """
        # note that, index shape is normally [B, S]. The input is a sentence. 
        scalar_index_name = self.tensor_name(prefix, f"index_{vocab_index}")
        self.add_init(graph, scalar_index_name, np.array(vocab_index, dtype=np.int64))

        equality_name = self.tensor_name(prefix, f"equal_{vocab_index}")
        nodes.append(
            helper.make_node(
                cons.OP_EQUAL,
                [index_name, scalar_index_name],
                [equality_name],
                name=self.node_name(prefix, f"equal_{vocab_index}"),
            )
        )

        # convert bool to 0 / 1
        float_mask_name = self.tensor_name(prefix, f"float_mask_{vocab_index}")
        nodes.append(
            helper.make_node(
                cons.OP_CAST,
                [equality_name],
                [float_mask_name],
                name=self.node_name(prefix, f"mask_cast_{vocab_index}"),
                to=TensorProto.FLOAT,
            )
        )
        return float_mask_name

    def _rewrite_node(self, node: onnx.NodeProto, graph: onnx.GraphProto) -> None:
        """Dispatch a Gather node to the first compatible rewrite strategy.

        Args:
            node: Gather node to inspect.
            graph: Graph being rewritten.

        Returns:
            None. Graph and log are updated in place.
        """
        context = self._build_context(node)

        if self._rewrite_static_gather(node, context, graph):
            return
        if self._rewrite_vocab_gather(node, context, graph):
            return

        self.log.append(f" - Gather({context.prefix}) kept as Gather (dynamic and unsupported)")

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        """Rewrite every Gather node matched by this pass.

        Args:
            model: ONNX model processed by the pass.

        Returns:
            Tuple of the rewritten model and pass log messages.
        """
        self.prepare(model)
        graph = self.require_graph()

        for node in list(graph.node):
            if node.op_type == cons.OP_GATHER:
                self._rewrite_node(node, graph)

        self.remove_marked_nodes()
        return model, self.log
