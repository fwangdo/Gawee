from __future__ import annotations
from typing     import List, Tuple  

from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import math

from ..utils import cons
from .folder import Folder


class ConstantFolding(Folder): 

    # preliminaries. 
    @staticmethod
    def _get_constant_value(node: onnx.NodeProto) -> np.ndarray | None:
        for attr in node.attribute:
            if attr.name == 'value':
                return numpy_helper.to_array(attr.t)
            if attr.name == 'value_float':
                return np.array(attr.f, dtype=np.float32)
            if attr.name == 'value_int':
                return np.array(attr.i, dtype=np.int64)
            if attr.name == 'value_floats':
                return np.array(list(attr.floats), dtype=np.float32)
            if attr.name == 'value_ints':
                return np.array(list(attr.ints), dtype=np.int64)
        return None

    def _rewrite_constant(self, node: onnx.NodeProto) -> None:
        output_name = node.output[0]
        value = self._get_constant_value(node)
        if value is not None:
            self.add_init(self.graph, output_name, value)
            self.init_map[output_name] = value
            self.nodes_to_remove.append(node)
            self.log.append(f" - Constant({node.name}) is folded into initializer")
        return 


    def run(self, model: onnx.ModelProto) -> Tuple[onnx.ModelProto, List[str]]:
        self.prepare(model)
        graph = self.require_graph()

        for node in list(graph.node):
            if node.op_type == cons.OP_CONSTANT:
                self._rewrite_constant(node)
            elif node.op_type == cons.OP_CONSTANT_OF_SHAPE:
                shape_input = node.input[0]
                if shape_input in init_map:
                    shape = init_map[shape_input].astype(np.int64)
                    fill_value = 0.0
                    fill_dtype = np.float32
                    for attr in node.attribute:
                        if attr.name == 'value':
                            t = numpy_helper.to_array(attr.t)
                            fill_value = t.item()
                            fill_dtype = t.dtype
                    output_array = np.full(shape, fill_value, dtype=fill_dtype)
                    output_name = node.output[0]
                    cls.add_init(graph, output_name, output_array)
                    init_map[output_name] = output_array
                    nodes_to_remove.append(node)
                    log.append(f" - ConstantOfShape({node.name}) is folded into initializer")

        self.remove_marked_nodes()

        return model, self.log 

