"""Partition fixed ONNX projections while retaining every live residual edge."""
import hashlib
import copy
import onnx
from onnx import numpy_helper,helper
from onnx.utils import Extractor
from .refine_layernorm_onnx import _initializer_digest


def complete_projection_shape_audit(model,audit,*,expected_graph_sha256):
    """Account for runtime matrices and original constant-input matrices.

    Importers fold constant-input projections, so their absence from a runtime
    layer list is expected. Only tensor dimensions are read here: no matrix
    multiplication or numerical acceptance is performed.
    """
    if (audit.get('scope')!='fixed_model_metadata_only_no_inference'
            or audit.get('source_graph_sha256')!=expected_graph_sha256
            or hashlib.sha256(model.SerializeToString()).hexdigest()!=expected_graph_sha256):
        raise ValueError('Fixed projection audit source differs')
    runtime={}
    for row in audit.get('projections',[]):
        sha=row.get('weight_sha256')
        if sha in runtime and runtime[sha]!=row:raise ValueError('Projection shape hash is ambiguous')
        runtime[sha]=row
    constants={n.output[0]:next((a.t for a in n.attribute if a.name=='value'),None)
               for n in model.graph.node if n.op_type=='Constant' and len(n.output)==1}
    weights={w.name:w for w in model.graph.initializer}
    constants.update(weights)
    completed=[];runtime_count=0;constant_count=0;used=set()
    for node in model.graph.node:
        if node.op_type!='MatMul' or node.input[1] not in weights or len(weights[node.input[1]].dims)!=2:continue
        weight=weights[node.input[1]]
        if weight.data_type!=onnx.TensorProto.FLOAT:raise ValueError('Projection coefficient is not FP32')
        sha=hashlib.sha256(numpy_helper.to_array(weight).astype('<f4').tobytes()).hexdigest()
        value=constants.get(node.input[0])
        if value is not None:
            if value.data_type!=onnx.TensorProto.FLOAT:raise ValueError('Projection constant input is not FP32')
            x=list(value.dims)
            row=dict(kind='OriginalConstantInput',weight_sha256=sha,weight_shape=list(weight.dims),
                     input_shape=x,output_shape=[*x[:-1],weight.dims[1]])
            constant_count+=1
        else:
            if sha not in runtime:raise ValueError('Runtime projection shape is missing')
            row=copy.deepcopy(runtime[sha]);used.add(sha);runtime_count+=1
        completed.append(row)
    if used!=set(runtime):raise ValueError('Runtime audit contains an unaccounted coefficient')
    result=dict(success=True,scope='fixed_model_metadata_only_no_inference',source_graph_sha256=expected_graph_sha256,
                projections=completed,projection_count=len(completed),runtime_projection_count=runtime_count,
                constant_projection_count=constant_count,original_runtime_audit_success=audit.get('success'),
                original_runtime_audit_error=audit.get('error'),runtime_source=audit.get('runtime_source'),
                gpu_validated=False,full_encoder_validated=False,end_to_end_validated=False)
    # Reuse the complete shape/type/conflict checks on a copy before returning.
    apply_projection_shape_audit(copy.deepcopy(model),result,expected_graph_sha256=expected_graph_sha256)
    return result


def apply_projection_shape_audit(model,audit,*,expected_graph_sha256):
    """Fill unknown metadata from a successful hash-bound fixed Sentis audit."""
    if (audit.get('success') is not True or audit.get('scope')!='fixed_model_metadata_only_no_inference'
            or audit.get('source_graph_sha256')!=expected_graph_sha256
            or hashlib.sha256(model.SerializeToString()).hexdigest()!=expected_graph_sha256):
        raise ValueError('Fixed projection audit source or completion differs')
    rows={}
    for row in audit.get('projections',[]):
        sha=row.get('weight_sha256')
        if sha in rows and row!=rows[sha]:raise ValueError('Projection shape hash is ambiguous')
        rows[sha]=row
    weights={w.name:w for w in model.graph.initializer}
    infos={v.name:v for v in (*model.graph.input,*model.graph.output,*model.graph.value_info)}
    assignments=[];count=0
    for node in model.graph.node:
        if node.op_type!='MatMul' or node.input[1] not in weights or len(weights[node.input[1]].dims)!=2:continue
        weight=weights[node.input[1]]
        sha=hashlib.sha256(numpy_helper.to_array(weight).astype('<f4').tobytes()).hexdigest();row=rows.get(sha)
        if row is None or row.get('weight_shape')!=list(weight.dims):raise ValueError('Projection audit coefficient is missing or differs')
        x,y=row.get('input_shape'),row.get('output_shape')
        if (not isinstance(x,list) or not isinstance(y,list) or len(x) not in (2,3)
                or any(type(d) is not int or d<=0 for d in (*x,*y))
                or x[-1]!=weight.dims[0] or y!=[*x[:-1],weight.dims[1]]):
            raise ValueError('Projection audit shape differs')
        for name,dimensions in [(node.input[0],x),(node.output[0],y)]:
            info=infos.get(name)
            if info is None or info.type.tensor_type.elem_type!=onnx.TensorProto.FLOAT or len(info.type.tensor_type.shape.dim)!=len(dimensions):
                raise ValueError('Projection metadata type or rank differs')
            for existing,actual in zip(info.type.tensor_type.shape.dim,dimensions):
                if existing.dim_value>0 and existing.dim_value!=actual:raise ValueError('Known projection dimension conflicts with audit')
            assignments.append((info,dimensions))
        count+=1
    if not count:raise ValueError('No audited static projections found')
    for info,dimensions in assignments:
        for dimension,actual in zip(info.type.tensor_type.shape.dim,dimensions):dimension.dim_value=actual
    return dict(projection_count=count,source_graph_sha256=expected_graph_sha256,
                coefficient_bytes_preserved=True,non_metadata_operations_preserved=True,full_encoder_validated=False)


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
        if node.name in node_names:continue
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
        if target.input[0] in constants:live.append(target.input[0])
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
