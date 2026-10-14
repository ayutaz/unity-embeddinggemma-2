"""Partition fixed ONNX projections while retaining every live residual edge."""
import hashlib
import onnx
from onnx import numpy_helper,helper
from onnx.utils import Extractor
from .refine_layernorm_onnx import _initializer_digest


def partition_static_projections(model,*,node_names,expected_weight_sha256):
    """Return unchanged-op segments and audited external projection boundaries.

    Coefficients stay in their original bytes. The caller supplies the GPU
    projections; this planner never executes or validates a full encoder.
    """
    if (not node_names or len(set(node_names))!=len(node_names)
            or len(node_names)!=len(expected_weight_sha256)):
        raise ValueError('Select unique ordered projection names and coefficient hashes')
    nodes=list(model.graph.node);weights={w.name:w for w in model.graph.initializer}
    infos={v.name:v for v in (*model.graph.input,*model.graph.output,*model.graph.value_info)}
    def shape(name):
        info=infos.get(name)
        dims=[d.dim_value for d in info.type.tensor_type.shape.dim] if info is not None else []
        if info is None or info.type.tensor_type.elem_type!=onnx.TensorProto.FLOAT or not dims or any(d<=0 for d in dims):
            raise ValueError('Projection boundaries require explicit fixed FP32 shapes')
        return dims
    selected=[];previous=-1
    for name,wanted in zip(node_names,expected_weight_sha256):
        matches=[(i,n) for i,n in enumerate(nodes) if n.name==name]
        if len(matches)!=1:raise ValueError('Selected projection is missing or ambiguous')
        index,node=matches[0]
        if index<=previous or node.op_type!='MatMul' or len(node.input)!=2 or len(node.output)!=1:
            raise ValueError('Projection order or operator differs')
        weight=weights.get(node.input[1])
        if weight is None or weight.data_type!=onnx.TensorProto.FLOAT or len(weight.dims)!=2:
            raise ValueError('Projection requires an existing FP32 matrix coefficient')
        digest=hashlib.sha256(numpy_helper.to_array(weight).astype('<f4').tobytes()).hexdigest()
        if digest!=wanted:raise ValueError('Projection coefficient bytes differ')
        data_shape=shape(node.input[0]);output_shape=shape(node.output[0])
        if len(data_shape) not in (2,3) or data_shape[-1]!=weight.dims[0] or output_shape!=[*data_shape[:-1],weight.dims[1]]:
            raise ValueError('Projection fixed matrix shapes differ')
        selected.append((index,node,dict(node=name,input=node.input[0],output=node.output[0],
            input_shape=data_shape,output_shape=output_shape,weight=node.input[1],
            weight_shape=list(weight.dims),weight_sha256=digest)))
        previous=index
    # Constant subgraphs are recomputed within each extracted segment, rather
    # than forcing shape/axis constants through intermediate tensor inputs.
    constants=set(weights)
    for node in nodes:
        known_shape=(node.op_type=='Shape' and node.input and node.input[0] in infos
                     and all(d.dim_value>0 for d in infos[node.input[0]].type.tensor_type.shape.dim))
        if node.op_type=='Constant' or known_shape or (node.input and all(v in constants for v in node.input)):
            constants.update(node.output)
    extractor=Extractor(model);frontier=[v.name for v in model.graph.input]
    seen=list(frontier);seen_set=set(seen);cursor=0;segments=[];metadata=[]
    def prepare(part,logical_outputs,number):
        used={v for n in part.graph.node for v in n.input}|{v.name for v in part.graph.output}
        inputs=[v for v in part.graph.input if v.name in used]
        del part.graph.input[:];part.graph.input.extend(inputs)
        input_names={v.name for v in inputs};aliases={}
        reserved={v for n in part.graph.node for v in (*n.input,*n.output)}|input_names
        for index,output in enumerate(part.graph.output):
            if output.name in input_names:
                original=output.name;name=f'__eg_segment_{number}_carry_{index}'
                while name in reserved:name+='_'
                reserved.add(name);output.name=name;aliases[name]=original
                part.graph.node.append(helper.make_node('Identity',[original],[name],name=name+'_identity'))
        return dict(inputs=[v.name for v in inputs],outputs=logical_outputs,output_aliases=aliases,node_count=len(part.graph.node))
    for index,target,_ in selected:
        for node in nodes[cursor:index]:
            for value in node.output:
                if value not in constants and value not in seen_set:seen.append(value);seen_set.add(value)
        future={v for node in nodes[index:] for v in node.input}
        future.update(v.name for v in model.graph.output)
        live=[v for v in seen if v in future and v not in constants]
        if target.input[0] not in live:raise ValueError('Projection input is not live at its boundary')
        part=extractor.extract_model(frontier,live)
        if any(n.name in node_names for n in part.graph.node):raise ValueError('A selected projection escaped its cut')
        for weight in part.graph.initializer:
            if weight.name not in weights or weight.SerializeToString()!=weights[weight.name].SerializeToString():
                raise ValueError('Extracted coefficient bytes differ')
        metadata.append(prepare(part,live,len(segments)));segments.append(part)
        frontier=[*live,target.output[0]];seen.append(target.output[0]);seen_set.add(target.output[0]);cursor=index+1
    outputs=[v.name for v in model.graph.output]
    tail=extractor.extract_model(frontier,outputs)
    if any(n.name in node_names for n in tail.graph.node):raise ValueError('A selected projection escaped the tail cut')
    metadata.append(prepare(tail,outputs,len(segments)));segments.append(tail)
    return segments,dict(projections=[p for _,_,p in selected],segments=metadata,
        source_initializer_bytes_preserved=True,source_initializer_sha256=_initializer_digest(model.graph.initializer),
        full_encoder_validated=False,end_to_end_validated=False)
