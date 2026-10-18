"""Resolve fixed FP32 projection coefficients through metadata-only views.

No model data arithmetic or inference is performed. Integer shape expressions
are bounded; float values may only originate in one embedded constant tensor.
"""
import hashlib
import math
import copy

import numpy as np
import onnx
from onnx import helper, numpy_helper


class _Views:
    def __init__(self, model):
        self.weights={w.name:w for w in model.graph.initializer}
        self.producers={}
        for node in model.graph.node:
            for output in node.output:
                if output in self.producers or output in self.weights:
                    raise ValueError('Ambiguous coefficient graph values')
                self.producers[output]=node
        self.infos={v.name:v for v in (*model.graph.input,*model.graph.value_info,*model.graph.output)}
        self.cache={};self.active=set();self.roots={}

    def attributes(self,node):
        if node.domain not in ('','ai.onnx'):
            raise ValueError('Coefficient views require standard ONNX')
        return {a.name:helper.get_attribute_value(a) for a in node.attribute}

    def tensor(self,tensor):
        if tensor.external_data or tensor.data_location==onnx.TensorProto.EXTERNAL:
            raise ValueError('Coefficient tensors must be fully embedded')
        return numpy_helper.to_array(tensor)

    def bounded(self,value,integer):
        value=np.asarray(value)
        if integer:
            if value.dtype.kind not in 'iub' or value.size>4096:
                raise ValueError('Shape expression must contain bounded integers only')
        elif value.dtype!=np.float32 or value.size>2**28:
            raise ValueError('Coefficient views require bounded original FP32 tensors')
        return value

    def shape(self,name):
        if name in self.weights:return tuple(self.weights[name].dims)
        info=self.infos.get(name)
        dims=tuple(d.dim_value for d in info.type.tensor_type.shape.dim) if info is not None else ()
        if not dims or len(dims)>8 or any(d<=0 for d in dims):
            raise ValueError('Shape metadata must be explicit and fixed')
        return dims

    def read(self,name,integer=False):
        key=(name,integer)
        if key in self.cache:return self.cache[key]
        if key in self.active:raise ValueError('Cyclic coefficient views are not supported')
        self.active.add(key)
        try:
            if name in self.weights:
                value=self.tensor(self.weights[name])
                if not integer:self.roots[name]=value
            else:
                node=self.producers.get(name)
                if node is None:raise ValueError('Projection coefficient depends on runtime tensor data')
                attrs=self.attributes(node);op=node.op_type
                if op=='Constant':
                    if 'value' in attrs:value=self.tensor(attrs['value'])
                    elif integer and 'value_int' in attrs:value=np.array(attrs['value_int'],dtype=np.int64)
                    elif integer and 'value_ints' in attrs:value=np.array(attrs['value_ints'],dtype=np.int64)
                    else:raise ValueError('Unqualified coefficient constant')
                    if not integer:self.roots[name]=value
                elif op=='Shape' and integer:
                    dims=self.shape(node.input[0])
                    value=np.array(dims[attrs.get('start',0):attrs.get('end',len(dims))],dtype=np.int64)
                elif op=='Identity':value=self.read(node.input[0],integer)
                elif op=='Cast':
                    value=self.read(node.input[0],integer)
                    dtype=helper.tensor_dtype_to_np_dtype(attrs['to'])
                    if (integer and np.dtype(dtype).kind not in 'iub') or (not integer and np.dtype(dtype)!=np.float32):
                        raise ValueError('Coefficient cast changes its qualified value domain')
                    value=value.astype(dtype)
                elif op=='Transpose':value=self.read(node.input[0],integer).transpose(attrs.get('perm'))
                elif op=='Reshape':
                    source=self.read(node.input[0],integer)
                    dims=self.read(node.input[1],True).reshape(-1).tolist()
                    if len(dims)>8:raise ValueError('Coefficient view rank is outside its bound')
                    if not attrs.get('allowzero',0):dims=[source.shape[i] if d==0 else d for i,d in enumerate(dims)]
                    value=source.reshape(dims)
                elif op in ('Unsqueeze','Squeeze'):
                    source=self.read(node.input[0],integer)
                    axes=tuple(self.read(node.input[1],True).reshape(-1).tolist()) if len(node.input)>1 else tuple(attrs.get('axes',()))
                    value=np.expand_dims(source,axes) if op=='Unsqueeze' else np.squeeze(source,axes or None)
                elif op=='Expand':
                    source=self.read(node.input[0],integer)
                    dims=tuple(self.read(node.input[1],True).reshape(-1).tolist())
                    limit=4096 if integer else 2**28
                    if len(dims)>8 or any(d<=0 for d in dims) or math.prod(np.broadcast_shapes(source.shape,dims))>limit:
                        raise ValueError('Coefficient expansion exceeds its bound')
                    value=np.broadcast_to(source,np.broadcast_shapes(source.shape,dims))
                elif integer and op=='Concat':value=np.concatenate([self.read(v,True) for v in node.input],axis=attrs['axis'])
                elif integer and op=='Gather':value=np.take(self.read(node.input[0],True),self.read(node.input[1],True),axis=attrs.get('axis',0))
                elif integer and op=='Equal':value=np.equal(self.read(node.input[0],True),self.read(node.input[1],True))
                elif integer and op=='Where':value=np.where(*(self.read(v,True) for v in node.input))
                elif integer and op=='Mul':
                    left,right=(self.read(v,True) for v in node.input)
                    if left.dtype.kind not in 'iu' or right.dtype.kind not in 'iu' or math.prod(np.broadcast_shapes(left.shape,right.shape))>4096:
                        raise ValueError('Shape multiplication exceeds its integer domain')
                    product=left.astype(object)*right.astype(object)
                    limit=np.iinfo(np.int64)
                    if any(x<limit.min or x>limit.max for x in np.asarray(product).reshape(-1)):
                        raise ValueError('Shape multiplication overflows int64')
                    value=np.asarray(product,dtype=np.int64)
                elif integer and op=='ConstantOfShape':
                    dims=tuple(self.read(node.input[0],True).reshape(-1).tolist())
                    fill=self.tensor(attrs['value']) if 'value' in attrs else np.array(0,np.float32)
                    if len(dims)>8 or any(d<0 for d in dims) or math.prod(dims)>4096 or fill.size!=1 or fill.dtype.kind not in 'iub':
                        raise ValueError('Shape allocation exceeds its integer domain')
                    value=np.full(dims,fill.item(),dtype=fill.dtype)
                elif integer and op=='Slice':
                    source=self.read(node.input[0],True)
                    starts=self.read(node.input[1],True).reshape(-1);ends=self.read(node.input[2],True).reshape(-1)
                    axes=self.read(node.input[3],True).reshape(-1) if len(node.input)>3 else np.arange(len(starts))
                    steps=self.read(node.input[4],True).reshape(-1) if len(node.input)>4 else np.ones(len(starts),np.int64)
                    if not len(starts)==len(ends)==len(axes)==len(steps) or len(set(map(int,axes)))!=len(axes) or any(v==0 for v in steps):
                        raise ValueError('Shape slice parameters differ')
                    selection=[slice(None)]*source.ndim
                    for a,b,axis,step in zip(starts,ends,axes,steps):selection[int(axis)]=slice(int(a),int(b),int(step))
                    value=source[tuple(selection)]
                else:raise ValueError('Coefficient path contains data arithmetic or an unsupported view')
            value=self.bounded(value,integer);self.cache[key]=value;return value
        finally:self.active.remove(key)


def resolve_projection(model,node_name):
    matches=[n for n in model.graph.node if n.name==node_name]
    if len(matches)!=1:raise ValueError('One unambiguous projection node is required')
    node=matches[0];views=_Views(model);attrs=views.attributes(node)
    if node.op_type not in ('MatMul','Gemm') or len(node.input)!=2 or len(node.output)!=1:
        raise ValueError('Projection requires an unbiased matrix operation')
    transpose=False
    if node.op_type=='Gemm':
        if (set(attrs)-{'alpha','beta','transA','transB'} or attrs.get('alpha',1.)!=1.
                or attrs.get('beta',1.)!=0. or attrs.get('transA',0)!=0 or attrs.get('transB',0) not in (0,1)):
            raise ValueError('Gemm is outside the audited unscaled, unbiased projection contract')
        transpose=bool(attrs.get('transB',0))
    value=views.read(node.input[1])
    if node.op_type=='Gemm' and value.ndim!=2:raise ValueError('Gemm requires a fixed matrix coefficient')
    if value.ndim<2 or any(size!=1 and stride!=0 for size,stride in zip(value.shape[:-2],value.strides[:-2])):
        raise ValueError('Batched coefficients are not one broadcast matrix')
    matrix=value[(0,)*(value.ndim-2)]
    if transpose:matrix=matrix.T
    if matrix.ndim!=2 or not matrix.size or len(views.roots)!=1 or not np.isfinite(matrix).all():
        raise ValueError('Projection coefficient root, shape or values differ')
    matrix=np.ascontiguousarray(matrix)
    root,source=next(iter(views.roots.items()))
    return matrix,dict(weight=root,weight_shape=list(matrix.shape),weight_sha256=hashlib.sha256(matrix.astype('<f4').tobytes()).hexdigest(),
        source_weight_shape=list(source.shape),source_weight_sha256=hashlib.sha256(source.astype('<f4').tobytes()).hexdigest(),
        source_projection_operator=node.op_type,coefficient_source_value=node.input[1],coefficient_only_metadata_views=True)


def split_gemm_bias(model):
    """Keep an audited original vector bias after an external GPU product.

    Mutates the caller-owned derived graph only after auditing all targets.
    Original initializer bytes are retained. Full numerical/GPU validation is
    still required for the resulting explicit product-then-add expression.
    """
    from .refine_layernorm_onnx import _initializer_digest
    views=_Views(model);targets=[]
    for node in model.graph.node:
        if node.op_type!='Gemm' or len(node.input)!=3:continue
        attrs=views.attributes(node)
        if (set(attrs)-{'alpha','beta','transA','transB'} or attrs.get('alpha',1.)!=1.
                or attrs.get('beta',1.)!=1. or attrs.get('transA',0)!=0 or attrs.get('transB',0) not in (0,1)
                or not node.name or sum(n.name==node.name for n in model.graph.node)!=1 or len(node.output)!=1):
            raise ValueError('Biased Gemm is outside its unscaled static contract')
        weight=views.weights.get(node.input[1]);bias=views.weights.get(node.input[2])
        if (weight is None or bias is None or weight.data_type!=onnx.TensorProto.FLOAT or bias.data_type!=onnx.TensorProto.FLOAT
                or len(weight.dims)!=2 or weight.external_data or bias.external_data
                or weight.data_location==onnx.TensorProto.EXTERNAL or bias.data_location==onnx.TensorProto.EXTERNAL):
            raise ValueError('Biased Gemm requires original embedded FP32 coefficients')
        k,n=(weight.dims[1],weight.dims[0]) if attrs.get('transB',0) else weight.dims
        a=views.shape(node.input[0]);out=views.shape(node.output[0])
        if len(a)!=2 or a[-1]!=k or out!=(a[0],n) or list(bias.dims)!=[n]:
            raise ValueError('Biased Gemm fixed data or vector-bias shape differs')
        if any(views.infos[name].type.tensor_type.elem_type!=onnx.TensorProto.FLOAT for name in (node.input[0],node.output[0])):
            raise ValueError('Biased Gemm data must remain FP32')
        targets.append(node)
    before=_initializer_digest(model.graph.initializer)
    reserved=set(views.weights)|set(views.infos)|set(views.producers)|{n.name for n in model.graph.node}
    replacements={};extra=[]
    for index,node in enumerate(targets):
        value='__eg_gemm_unbiased_'+str(index)
        while value in reserved:
            value=value+'_'
        reserved.add(value)
        product=copy.deepcopy(node);del product.input[2:];product.output[0]=value
        beta=next((a for a in product.attribute if a.name=='beta'),None)
        if beta is None:product.attribute.append(helper.make_attribute('beta',0.))
        else:beta.f=0.
        addition=helper.make_node('Add',[value,node.input[2]],list(node.output),name=value+'_add')
        info=copy.deepcopy(views.infos[node.output[0]]);info.name=value;extra.append(info)
        replacements[node.name]=[product,addition]
    rewritten=[]
    for node in model.graph.node:rewritten.extend(replacements.get(node.name,[node]))
    del model.graph.node[:];model.graph.node.extend(rewritten);model.graph.value_info.extend(extra)
    if _initializer_digest(model.graph.initializer)!=before:raise RuntimeError('Original coefficient bytes changed')
    return dict(split_biased_gemm_count=len(targets),source_initializer_bytes_preserved=True,source_initializer_sha256=before,
        explicit_original_bias_after_product=True,full_encoder_validated=False)
