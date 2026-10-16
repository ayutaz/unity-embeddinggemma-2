"""Opt-in pinned-order RMS rewrite, preserving all existing coefficient bytes."""
import copy

import numpy as np
import onnx
from onnx import numpy_helper

from .refine_layernorm_onnx import _initializer_digest


def _validate_template(template):
    if (len(template.graph.input)!=1 or len(template.graph.output)!=1
            or any(v.type.tensor_type.elem_type!=onnx.TensorProto.FLOAT for v in (*template.graph.input,*template.graph.output))
            or any(n.domain not in ('','ai.onnx') for n in template.graph.node)
            or any(w.data_type==onnx.TensorProto.DOUBLE for w in template.graph.initializer)
            or any((a.type==onnx.AttributeProto.TENSOR and a.t.data_type==onnx.TensorProto.DOUBLE)
                   or (n.op_type=='Cast' and a.name=='to' and a.i==onnx.TensorProto.DOUBLE)
                   for n in template.graph.node for a in n.attribute)):
        raise ValueError('RMS templates require standard FP32 input/output and arithmetic')


def refine_rms_means(model, mean_templates, inverse_template, *, expected_count, excluded_names=(), mean_overrides=None):
    """Audit squared last-axis RMS paths, then inline explicit FP32 templates."""
    mean_overrides={} if mean_overrides is None else mean_overrides
    for template in [*mean_templates.values(),*mean_overrides.values(),inverse_template]:_validate_template(template)
    mean_names={n.name for n in model.graph.node if n.op_type=='ReduceMean'}
    if len(set(excluded_names))!=len(excluded_names) or not set(excluded_names).issubset(mean_names):
        raise ValueError('Explicit RMS layout exclusion is missing or duplicated')
    if not set(mean_overrides).issubset(mean_names) or set(mean_overrides).intersection(excluded_names):
        raise ValueError('Explicit RMS layout override is missing or excluded')
    weights={w.name:w for w in model.graph.initializer}
    producers={output:n for n in model.graph.node for output in n.output}
    infos={v.name:v for v in (*model.graph.input,*model.graph.output,*model.graph.value_info)}
    def constant(name):
        if name in weights:return numpy_helper.to_array(weights[name])
        node=producers.get(name)
        if node is None or node.op_type!='Constant':raise ValueError('RMS scalar/axis must be an audited constant')
        attributes=[a.t for a in node.attribute if a.name=='value']
        if len(attributes)!=1:raise ValueError('Constant tensor is missing')
        return numpy_helper.to_array(attributes[0])
    def scalar(name,value):
        actual=constant(name)
        return actual.size==1 and np.isfinite(actual).all() and float(actual.item())==float(np.float32(value))
    targets=[]
    for mean in [n for n in model.graph.node if n.op_type=='ReduceMean']:
        if mean.name in excluded_names:continue
        info=infos.get(mean.input[0]);shape=info.type.tensor_type.shape.dim if info is not None else []
        width=shape[-1].dim_value if shape else 0
        axes=constant(mean.input[1]).reshape(-1).tolist() if len(mean.input)==2 else next((list(a.ints) for a in mean.attribute if a.name=='axes'),[])
        keep=next((a.i for a in mean.attribute if a.name=='keepdims'),1)
        square=producers.get(mean.input[0])
        squared=(square is not None and ((square.op_type=='Pow' and len(square.input)==2 and scalar(square.input[1],2))
                 or (square.op_type=='Mul' and len(square.input)==2 and square.input[0]==square.input[1])))
        if (width not in mean_templates or axes not in ([-1],[len(shape)-1]) or keep!=1 or not squared):
            raise ValueError('RMS mean requires a supported squared last-axis fixed width')
        template=mean_overrides.get(mean.name,mean_templates[width])
        if mean.name in mean_overrides:
            template_shape=template.graph.input[0].type.tensor_type.shape.dim
            if (not shape or any(d.dim_value<=0 for d in (*shape,*template_shape))
                    or [d.dim_value for d in shape]!=[d.dim_value for d in template_shape]):
                raise ValueError('Explicit RMS layout override requires the exact fixed input shape')
        users=[n for n in model.graph.node if mean.output[0] in n.input]
        if len(users)!=1 or users[0].op_type!='Add':raise ValueError('RMS epsilon edge is ambiguous')
        addition=users[0];epsilon=[i for i in addition.input if i!=mean.output[0]]
        if len(epsilon)!=1 or not scalar(epsilon[0],1e-6):raise ValueError('RMS epsilon differs')
        users=[n for n in model.graph.node if addition.output[0] in n.input]
        if len(users)!=1 or users[0].op_type!='Pow' or not scalar(users[0].input[1],-.5):
            raise ValueError('RMS inverse edge differs')
        targets.extend([(mean,template),(users[0],inverse_template)])
    if len(targets)!=expected_count*2 or expected_count<=0:raise ValueError('RMS target count differs')
    reserved={v.name for v in (*model.graph.input,*model.graph.output,*model.graph.initializer,*model.graph.value_info)}
    reserved.update(v for n in model.graph.node for v in (*n.input,*n.output))
    def fresh(value):
        while value in reserved:value+='_'
        reserved.add(value);return value
    replacements,extra={},[]
    for number,(target,template) in enumerate(targets):
        names={template.graph.input[0].name:target.input[0],template.graph.output[0].name:target.output[0]}
        def rename(value):
            if not value:return value
            if value not in names:names[value]=fresh(f'__eg_rms_{number}_'+value)
            return names[value]
        for original in template.graph.initializer:
            weight=copy.deepcopy(original);weight.name=rename(original.name);extra.append(weight)
        nodes=[]
        for original in template.graph.node:
            node=copy.deepcopy(original);node.name=fresh(f'__eg_rms_{number}_node_'+original.name)
            node.input[:]=[rename(v) for v in original.input];node.output[:]=[rename(v) for v in original.output];nodes.append(node)
        replacements[target.name]=nodes
    count=len(model.graph.initializer);before=_initializer_digest(model.graph.initializer)
    rewritten=[]
    for node in model.graph.node:rewritten.extend(replacements.get(node.name,[node]))
    del model.graph.node[:];model.graph.node.extend(rewritten);model.graph.initializer.extend(extra)
    if _initializer_digest(model.graph.initializer[:count])!=before:raise RuntimeError('Existing RMS coefficient bytes changed')
    return dict(replaced_rms_count=expected_count,existing_initializer_bytes_preserved=True,
                excluded_mean_names=list(excluded_names),
                explicit_mean_override_names=list(mean_overrides),
                existing_initializer_sha256=before,node_count=len(model.graph.node),full_encoder_validated=False)
