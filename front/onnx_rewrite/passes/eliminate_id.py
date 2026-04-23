from __future__ import annotations

import onnx

from ..utils import cons
from .folder import Folder


class EliminateId(Folder):
    """Eliminate Identity nodes by redirecting edges.

    Identity is a no-op (output = input). This pass builds a mapping
    of Identity outputs → inputs, resolves transitive chains, then
    rewires all downstream consumers to skip Identity nodes entirely.
    """

    def _build_id_map(self, graph: onnx.GraphProto) -> dict[str, str]:
        """Build a mapping from Identity output names to their input names.

        Args:
            graph: ONNX graph to scan.

        Returns:
            Dict mapping each Identity's output name to its input name.
        """
        id_map: dict[str, str] = {}
        for node in graph.node:
            if node.op_type == cons.OP_IDENTITY:
                id_map[node.output[0]] = node.input[0]

        return id_map


    @staticmethod
    def _resolve(name: str, id_map: dict[str, str]) -> str:
        """Follow Identity chains to find the original non-Identity source.

        e.g. A → Identity → B → Identity → C resolves C to A.

        Args:
            name: Tensor name to resolve.
            id_map: Identity output → input mapping.

        Returns:
            The resolved tensor name (unchanged if not in id_map).
        """
        visited: set[str] = set()
        while name in id_map and name not in visited:
            visited.add(name)
            name = id_map[name]

        return name


    @classmethod
    def _rewire_nodes(cls, graph: onnx.GraphProto, id_map: dict[str, str]) -> None:
        """Redirect all node inputs and graph outputs past Identity nodes.

        Args:
            graph: ONNX graph to modify.
            id_map: Identity output → input mapping.
        """
        for node in graph.node:
            if node.op_type == cons.OP_IDENTITY:
                continue
            for i in range(len(node.input)):
                resolved = cls._resolve(node.input[i], id_map)
                if resolved != node.input[i]:
                    node.input[i] = resolved

        for out in graph.output:
            resolved = cls._resolve(out.name, id_map)
            if resolved != out.name:
                out.name = resolved


    def _remove_identity_nodes(self, graph: onnx.GraphProto) -> None:
        """Remove all Identity nodes from the graph.

        Args:
            graph: ONNX graph to clean up.
            log: Log list to append removal messages to.
        """
        nodes_to_remove = [n for n in graph.node if n.op_type == cons.OP_IDENTITY]
        for node in nodes_to_remove:
            self.log.append(f" - Identity({node.name}) is removed (no-op)")
            graph.node.remove(node)
            self.deleted_node += 1


    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        """Remove all Identity nodes from the model.

        Args:
            model: ONNX model to optimize.

        Returns:
            (model, log): Modified model and list of log messages.
        """
        self.prepare(model)

        id_map = self._build_id_map(self.graph)
        if not id_map:
            return model, self.log

        self._rewire_nodes(self.graph, id_map)
        self._remove_identity_nodes(self.graph)

        return self.model, self.log
