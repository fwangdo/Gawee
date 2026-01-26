from __future__ import annotations
from typing     import *

from gawee_ir.graph import Graph, Node, Value
from gawee_ir.constant.ops import IDENTITY
from gawee_ir.passes.folder import Folder


class IdentityElimination(Folder):
    """
    Eliminate Identity nodes.

    Pattern:
        Identity(x) -> y
    Rewrite:
        y uses x directly

    Assumptions:
      - inference graph (no in-place mutation)
      - Identity has exactly 1 input and 1 output
      - Identity is semantics-preserving alias
    """

    @classmethod
    def _is_identity_node(cls, n: Node) -> bool:
        return (
            n.op_type == IDENTITY
            and len(n.inputs) == 1
            and len(n.outputs) == 1
        )

    @classmethod
    def run(cls, g: Graph) -> bool:
        changed = False

        # iterate over a snapshot since we mutate the graph
        for n in list(g.nodes):
            if not cls._is_identity_node(n):
                continue

            src: Value = n.inputs[0]
            dst: Value = n.outputs[0]

            # Replace all uses of Identity output with its input
            g.replace_all_uses(dst, src)

            # Detach from producer/consumer links (defensive)
            if n in src.consumers:
                src.consumers.remove(n)

            dst.producer = None

            # Remove Identity node
            g.remove_node(n)

            cls.deleted_node += 1
            changed = True

        return changed
