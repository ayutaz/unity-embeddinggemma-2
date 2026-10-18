"""Canonicalize audited index-only export patterns before GPU partitioning."""
import onnx
from onnx import helper

from .projection_coefficients import _Views
from .refine_layernorm_onnx import _initializer_digest


def normalize_glu_slices(model):
    views=_Views(model);nodes=list(model.graph.node);producer=views.producers
    positions={id(n):i for i,n in enumerate(nodes)};uses={name:[] for name in producer}
    for i,node in enumerate(nodes):
        for value in node.input:
            if value in uses:uses[value].append(i)
    outputs={v.name for v in model.graph.output};selected=[];removed=set();markers={}
    reserved={n.name for n in nodes}
    def shape(value):
        dims=views.shape(value);info=views.infos.get(value)
        if info is None or info.type.tensor_type.elem_type!=onnx.TensorProto.FLOAT:raise ValueError('GLU requires fixed FP32 data')
        return list(dims)
    def half_slice(node,source,begin,end,dimensions):
        if node.domain or node.op_type!='Slice' or len(node.input) not in (4,5) or len(node.output)!=1 or node.input[0]!=source:
            raise ValueError('GLU slice source differs')
        parameters=[views.read(value,True).reshape(-1) for value in node.input[1:]]
        if any(value.size!=1 or value.dtype.kind not in 'iu' for value in parameters):raise ValueError('GLU slice bounds must be fixed integers')
        start,stop,axis=(int(value.item()) for value in parameters[:3]);step=int(parameters[3].item()) if len(parameters)==4 else 1
        if axis not in (-1,len(dimensions)-1) or step!=1 or start!=begin or min(stop,dimensions[-1])!=end:
            raise ValueError('GLU slices must select the two complete last-axis halves')
    for multiply in nodes:
        if multiply.op_type!='Mul' or len(multiply.input)!=2:continue
        sigmoid=producer.get(multiply.input[1]);left=producer.get(multiply.input[0])
        if sigmoid is None or sigmoid.op_type!='Sigmoid' or len(sigmoid.input)!=1:continue
        right=producer.get(sigmoid.input[0])
        if left is None or right is None or left.op_type!='Slice' and right.op_type!='Slice':continue
        source=left.input[0];dimensions=shape(source)
        if not dimensions or dimensions[-1]%2:raise ValueError('GLU requires an even final dimension')
        half=dimensions[-1]//2
        half_slice(left,source,0,half,dimensions);half_slice(right,source,half,dimensions[-1],dimensions)
        a,b,s,i=(positions[id(n)] for n in (left,right,sigmoid,multiply))
        if not a<i or not b<s<i or sigmoid.domain or multiply.domain:raise ValueError('GLU branch order differs')
        expected=[*dimensions[:-1],half]
        for value,consumer in [(left.output[0],i),(right.output[0],s),(sigmoid.output[0],i)]:
            if value in outputs or uses[value]!=[consumer] or shape(value)!=expected:raise ValueError('GLU slice output is shared or has different dimensions')
        if shape(multiply.output[0])!=expected or removed.intersection((a,b)):raise ValueError('GLU boundaries overlap or differ')
        if next(o.version for o in model.opset_import if o.domain=='')<18:raise ValueError('GLU normalization requires the pinned modern Split schema')
        name='__eg_glu_split_'+str(len(selected))
        while name in reserved:
            name=name+'_'
        reserved.add(name)
        markers[min(a,b)]=helper.make_node('Split',[source],[left.output[0],right.output[0]],name=name,axis=-1,num_outputs=2)
        removed.update((a,b));selected.append(multiply.name)
    before=_initializer_digest(model.graph.initializer)
    rewritten=[markers[i] if i in markers else n for i,n in enumerate(nodes) if i in markers or i not in removed]
    del model.graph.node[:];model.graph.node.extend(rewritten)
    if _initializer_digest(model.graph.initializer)!=before:raise RuntimeError('GLU normalization changed coefficient bytes')
    return dict(normalized_glu_count=len(selected),glu_node_names=selected,source_initializer_bytes_preserved=True,
        source_initializer_sha256=before,model_executed=False,gpu_validated=False)
