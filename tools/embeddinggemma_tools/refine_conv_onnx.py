"""Audited opt-in replacement of one fixed-shape FP32 convolution."""
import copy
import hashlib
import onnx
from onnx import numpy_helper
from .refine_layernorm_onnx import _initializer_digest


def refine_conv2d(model,template,*,node_name,expected_weight_sha256):
    """Inline a two-input template, using the exact existing 4D weight tensor."""
    unsupported_float={onnx.TensorProto.DOUBLE,onnx.TensorProto.FLOAT16,onnx.TensorProto.BFLOAT16}
    if (len(template.graph.input)!=2 or len(template.graph.output)!=1
            or any(v.type.tensor_type.elem_type!=onnx.TensorProto.FLOAT for v in (*template.graph.input,*template.graph.output))
            or any(n.domain not in ('','ai.onnx') for n in template.graph.node)
            or any(w.data_type in unsupported_float for w in template.graph.initializer)
            or any((n.op_type=='Cast' and a.name=='to' and a.i in unsupported_float)
                   or (a.type==onnx.AttributeProto.TENSOR and a.t.data_type in unsupported_float)
                   for n in template.graph.node for a in n.attribute)):
        raise ValueError('Convolution template requires two FP32 standard-op inputs and one output')
    matches=[n for n in model.graph.node if n.name==node_name]
    if len(matches)!=1 or matches[0].op_type!='Conv' or len(matches[0].input)!=2 or len(matches[0].output)!=1:
        raise ValueError('Selected unbiased convolution is missing or ambiguous')
    target=matches[0];attrs={a.name:onnx.helper.get_attribute_value(a) for a in target.attribute}
    if (attrs.get('strides')!=[2,2] or attrs.get('pads')!=[1,1,1,1]
            or attrs.get('dilations',[1,1])!=[1,1] or attrs.get('group',1)!=1
            or attrs.get('kernel_shape')!=[3,3] or attrs.get('auto_pad',b'NOTSET')!=b'NOTSET'):
        raise ValueError('Convolution geometry is outside the audited contract')
    weights={w.name:w for w in model.graph.initializer};weight=weights.get(target.input[1])
    if weight is None or weight.data_type!=onnx.TensorProto.FLOAT:
        raise ValueError('Convolution requires an existing FP32 initializer')
    digest=hashlib.sha256(numpy_helper.to_array(weight).astype('<f4').tobytes()).hexdigest()
    if digest!=expected_weight_sha256:raise ValueError('Convolution weight bytes differ')
    infos={v.name:v for v in (*model.graph.input,*model.graph.value_info,*model.graph.output)}
    def shape(value):return [d.dim_value for d in value.type.tensor_type.shape.dim]
    info=infos.get(target.input[0]);data_shape=shape(template.graph.input[0]);weight_shape=shape(template.graph.input[1])
    if (info is None or info.type.tensor_type.elem_type!=onnx.TensorProto.FLOAT
            or len(data_shape)!=4 or any(d<=0 for d in data_shape) or shape(info)!=data_shape
            or weight_shape!=list(weight.dims) or len(weight_shape)!=4
            or weight_shape[-2:]!=[3,3] or weight_shape[1]!=data_shape[1]
            or shape(template.graph.output[0])!=[data_shape[0],weight_shape[0],(data_shape[2]+1)//2,(data_shape[3]+1)//2]):
        raise ValueError('Convolution template needs the exact fixed data and weight shapes')
    reserved={v.name for v in (*model.graph.input,*model.graph.output,*model.graph.initializer,*model.graph.value_info)}
    reserved.update(v for n in model.graph.node for v in (*n.input,*n.output))
    names={template.graph.input[0].name:target.input[0],template.graph.input[1].name:target.input[1],template.graph.output[0].name:target.output[0]}
    def fresh(value):
        while value in reserved:
            value = value + "_"
        reserved.add(value);return value
    def rename(value):
        if not value:return value
        if value not in names:names[value]=fresh('__eg_conv_'+value)
        return names[value]
    extra=[];replacement=[]
    for original in template.graph.initializer:
        value=copy.deepcopy(original);value.name=rename(value.name);extra.append(value)
    for original in template.graph.node:
        node=copy.deepcopy(original);node.name=fresh('__eg_conv_node_'+node.name)
        node.input[:]=[rename(v) for v in node.input];node.output[:]=[rename(v) for v in node.output];replacement.append(node)
    count=len(model.graph.initializer);before=_initializer_digest(model.graph.initializer)
    nodes=[]
    for node in model.graph.node:nodes.extend(replacement if node.name==node_name else [node])
    del model.graph.node[:];model.graph.node.extend(nodes);model.graph.initializer.extend(extra)
    if _initializer_digest(model.graph.initializer[:count])!=before:raise RuntimeError('Existing convolution initializer bytes changed')
    return dict(replaced_conv_count=1,target_node=node_name,weight_sha256=digest,
        existing_initializer_bytes_preserved=True,existing_initializer_sha256=before,
        node_count=len(model.graph.node),full_encoder_validated=False)
