"""Audited opt-in replacement of one original fixed FP32 LayerNormalization."""
import copy
import hashlib

import numpy as np
import onnx
from onnx import numpy_helper

from .refine_layernorm_onnx import _initializer_digest


def refine_layernorm_node(model,template,*,node_name,expected_scale_sha256,expected_epsilon):
    unsupported={onnx.TensorProto.DOUBLE,onnx.TensorProto.FLOAT16,onnx.TensorProto.BFLOAT16}
    if (len(template.graph.input)!=2 or len(template.graph.output)!=1
            or any(value.type.tensor_type.elem_type!=onnx.TensorProto.FLOAT for value in (*template.graph.input,*template.graph.output))
            or any(node.domain not in ("","ai.onnx") for node in template.graph.node)
            or any(weight.data_type in unsupported for weight in template.graph.initializer)
            or any((node.op_type=="Cast" and attribute.name=="to" and attribute.i in unsupported)
                   or (attribute.type==onnx.AttributeProto.TENSOR and attribute.t.data_type in unsupported)
                   for node in template.graph.node for attribute in node.attribute)):
        raise ValueError("LayerNorm template requires two standard FP32 inputs and one output")
    matches=[node for node in model.graph.node if node.name==node_name]
    if len(matches)!=1 or matches[0].op_type!="LayerNormalization" or len(matches[0].input) not in (2,3) or len(matches[0].output)!=1:
        raise ValueError("Selected LayerNormalization is missing or ambiguous")
    target=matches[0];attributes={value.name:onnx.helper.get_attribute_value(value) for value in target.attribute}
    if (attributes.get("axis",-1)!=-1 or attributes.get("stash_type",1)!=1
            or np.float32(attributes.get("epsilon",1e-5))!=np.float32(expected_epsilon)):
        raise ValueError("LayerNorm axis, FP32 statistics or epsilon changed")
    weights={weight.name:weight for weight in model.graph.initializer}
    producers={name:node for node in model.graph.node for name in node.output}
    def constant(name):
        if name in weights:return numpy_helper.to_array(weights[name])
        node=producers.get(name)
        values=[attribute.t for attribute in node.attribute if attribute.name=="value"] if node is not None and node.op_type=="Constant" else []
        return numpy_helper.to_array(values[0]) if len(values)==1 else None
    weight=weights.get(target.input[1]);scale=constant(target.input[1])
    if (weight is None or weight.data_type!=onnx.TensorProto.FLOAT or scale.ndim!=1 or scale.size not in (32,128)
            or not np.isfinite(scale).all() or hashlib.sha256(scale.astype("<f4").tobytes()).hexdigest()!=expected_scale_sha256):
        raise ValueError("LayerNorm requires the audited original FP32 scale vector")
    if len(target.input)==3:
        bias=constant(target.input[2])
        if bias is None or bias.dtype!=np.float32 or bias.shape!=scale.shape or not np.all(bias==0):
            raise ValueError("LayerNorm candidate requires the original zero bias")
    infos={value.name:value for value in (*model.graph.input,*model.graph.value_info,*model.graph.output)}
    def shape(value):return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
    info=infos.get(target.input[0]);data_shape=shape(template.graph.input[0])
    if (info is None or info.type.tensor_type.elem_type!=onnx.TensorProto.FLOAT or shape(info)!=data_shape
            or not data_shape or any(value<=0 for value in data_shape) or data_shape[-1]!=scale.size
            or shape(template.graph.input[1])!=list(scale.shape) or shape(template.graph.output[0])!=data_shape):
        raise ValueError("LayerNorm template requires the exact original fixed input/output shapes")
    reserved={value.name for value in (*model.graph.input,*model.graph.output,*model.graph.initializer,*model.graph.value_info)}
    reserved.update(value for node in model.graph.node for value in (*node.input,*node.output))
    names={template.graph.input[0].name:target.input[0],template.graph.input[1].name:target.input[1],template.graph.output[0].name:target.output[0]}
    def fresh(value):
        while value in reserved:value+="_"
        reserved.add(value);return value
    def rename(value):
        if not value:return value
        if value not in names:names[value]=fresh("__eg_layernorm_"+value)
        return names[value]
    extra=[];replacement=[]
    for original in template.graph.initializer:
        value=copy.deepcopy(original);value.name=rename(value.name);extra.append(value)
    for original in template.graph.node:
        value=copy.deepcopy(original);value.name=fresh("__eg_layernorm_node_"+value.name)
        value.input[:]=[rename(name) for name in original.input];value.output[:]=[rename(name) for name in original.output];replacement.append(value)
    before=_initializer_digest(model.graph.initializer);count=len(model.graph.initializer);nodes=[]
    for node in model.graph.node:nodes.extend(replacement if node.name==node_name else [node])
    del model.graph.node[:];model.graph.node.extend(nodes);model.graph.initializer.extend(extra)
    if _initializer_digest(model.graph.initializer[:count])!=before:raise RuntimeError("Original LayerNorm coefficient bytes changed")
    return dict(replaced_layernorm_count=1,target_node=node_name,scale_sha256=expected_scale_sha256,
        existing_initializer_bytes_preserved=True,existing_initializer_sha256=before,full_encoder_validated=False)
