from __future__ import annotations
from typing import Dict, Iterable, List

import numpy as np
import onnx
from onnx import numpy_helper


class Folder:
    """Base class for rewrite passes.

    `prepare()` initializes the per-run execution state stored on `self`.
    Generated node/tensor names should use the helpers below so the two
    namespaces stay visually distinct in rewrite code.
    """

    model: onnx.ModelProto
    graph: onnx.GraphProto

    def __init__(self) -> None:
        self.deleted_node: int = 0
        self.eps: float = 1e-5

        # Per-run state populated by `prepare()`.
        self.log: List[str] = []
        self.init_map: Dict[str, np.ndarray] = {}
        self.shape_info: Dict[str, List] = {}
        self.nodes_to_remove: List[onnx.NodeProto] = []

        # Tensor-name based graph indexes populated by `prepare()`.
        self.producer_by_output: Dict[str, onnx.NodeProto] = {}
        self.consumers_by_input: Dict[str, List[onnx.NodeProto]] = {}

    def prepare(self, model: onnx.ModelProto) -> None:
        """Initialize model-specific state for one pass execution."""
        self.model = model
        self.graph = model.graph
        self.log = []
        self.init_map = self.get_init_map(model.graph)
        self.shape_info = self._get_shape_info(model)
        self.nodes_to_remove = []
        self._parse_relation()


    def require_graph(self) -> onnx.GraphProto:
        """Return the prepared graph or fail fast when `prepare()` was skipped."""
        return self.graph


    def require_model(self) -> onnx.ModelProto:
        """Return the prepared model or fail fast when `prepare()` was skipped."""
        return self.model


    def append_nodes(self, nodes: Iterable[onnx.NodeProto]) -> None:
        """Append generated nodes to the prepared graph in order."""
        graph = self.require_graph()
        for node in nodes:
            graph.node.append(node)


    def mark_for_removal(self, node: onnx.NodeProto) -> None:
        """Register a node for deletion after pass processing completes."""
        self.nodes_to_remove.append(node)


    def replace_node(self, node: onnx.NodeProto, new_nodes: Iterable[onnx.NodeProto]) -> None:
        """Replace one existing node with a sequence of generated nodes."""
        self.mark_for_removal(node)
        self.append_nodes(new_nodes)


    def remove_marked_nodes(self) -> None:
        """Remove nodes previously registered via `mark_for_removal()`."""
        graph = self.require_graph()
        for node in self.nodes_to_remove:
            graph.node.remove(node)
            self.deleted_node += 1


    def _parse_relation(self) -> None:
        """Build tensor-name indexes for producer and consumer lookup."""
        self.producer_by_output = {}
        self.consumers_by_input = {}

        for node in self.graph.node:
            for input_name in node.input:
                if not input_name:
                    continue
                self.consumers_by_input.setdefault(input_name, []).append(node)

            for output_name in node.output:
                if not output_name:
                    continue
                if output_name in self.producer_by_output:
                    raise ValueError(f"duplicate ONNX value producer for '{output_name}'")
                self.producer_by_output[output_name] = node
        
        return 

    @property
    def producer(self) -> Dict[str, onnx.NodeProto]:
        """Backward-compatible alias for producer-by-output lookup."""
        return self.producer_by_output

    @property
    def consumer(self) -> Dict[str, List[onnx.NodeProto]]:
        """Backward-compatible alias for consumers-by-input lookup."""
        return self.consumers_by_input

    def get_producer(self, value_name: str) -> onnx.NodeProto | None:
        """Return the node that produces `value_name`, if it exists."""
        if not value_name:
            return None
        return self.producer_by_output.get(value_name)

    def get_consumers(self, value_name: str) -> List[onnx.NodeProto]:
        """Return nodes that consume `value_name`."""
        if not value_name:
            return []
        return self.consumers_by_input.get(value_name, [])


    @staticmethod
    def get_init_map(graph: onnx.GraphProto) -> Dict[str, np.ndarray]:
        """Build a name → numpy array mapping from graph initializers.

        Args:
            graph: ONNX graph to extract initializers from.

        Returns:
            Dict mapping initializer name to its numpy array value.
        """
        return {init.name: numpy_helper.to_array(init) for init in graph.initializer}


    @staticmethod
    def add_init(graph: onnx.GraphProto, name: str, array: np.ndarray) -> None:
        """Add or replace an initializer in the graph.

        If an initializer with the same name exists, it is removed first.

        Args:
            graph: ONNX graph to add the initializer to.
            name: Name for the initializer tensor.
            array: Numpy array value to store.
        """
        for init in list(graph.initializer):
            if init.name == name:
                graph.initializer.remove(init)
                break
        graph.initializer.append(numpy_helper.from_array(array, name=name))


    @staticmethod
    def _get_shape_info(model: onnx.ModelProto) -> Dict[str, List]:
        """Run shape inference and collect known tensor shapes.

        Tries onnx.shape_inference.infer_shapes; falls back to the
        original model if inference fails (e.g., dynamic ops).

        Args:
            model: ONNX model to infer shapes from.

        Returns:
            Dict mapping tensor name to list of dimension values.
            Dynamic dims appear as strings (dim_param), static as ints.
        """
        try:
            inferred = onnx.shape_inference.infer_shapes(model)
        except Exception:
            inferred = model

        shape_info: Dict[str, List] = {}
        for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
            tt = vi.type.tensor_type
            if tt.HasField('shape'):
                shape: List = []
                for dim in tt.shape.dim:
                    if dim.dim_param:
                        shape.append(dim.dim_param)
                    else:
                        shape.append(dim.dim_value)
                shape_info[vi.name] = shape

        return shape_info


    @staticmethod
    def get_prefix(node: onnx.NodeProto) -> str:
        """Get a stable prefix for generated names derived from one source node.

        Uses node.name if available, otherwise falls back to output name.
        ONNX node names are optional and may be empty strings.

        Args:
            node: The node to derive a prefix from.

        Returns:
            Non-empty string usable as a name prefix.
        """
        if node.name:
            return node.name
        return node.output[0]


    @staticmethod
    def node_name(prefix: str, role: str) -> str:
        """Build a generated node name."""
        return f"{prefix}_node_{role}"


    @staticmethod
    def tensor_name(prefix: str, role: str) -> str:
        """Build a generated tensor name."""
        return f"{prefix}_tensor_{role}"


    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        raise NotImplementedError
