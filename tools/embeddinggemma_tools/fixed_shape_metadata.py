"""Prove fixed Pad geometry from bounded integer metadata, without inference."""
import copy
import hashlib

import onnx
from onnx import helper

from .projection_coefficients import _Views


def bind_fixed_pad_shapes(model):
    views=_Views(model);assignments=[];unresolved=[];verified=[]
    for node in model.graph.node:
        if node.op_type!='Pad':continue
        attrs=views.attributes(node)
        try:source=views.shape(node.input[0])
        except ValueError:
            unresolved.append(node.name);continue
        pads=views.read(node.input[1],True)
        axes=views.read(node.input[3],True).reshape(-1).tolist() if len(node.input)>3 and node.input[3] else list(range(len(source)))
        axes=[axis+len(source) if axis<0 else axis for axis in axes]
        if pads.ndim!=1 or pads.dtype.kind not in 'iu' or len(pads)!=2*len(axes) or len(set(axes))!=len(axes) or any(a<0 or a>=len(source) for a in axes):
            raise ValueError('Fixed Pad axes or integer padding dimensions differ')
        dimensions=list(source)
        for i,axis in enumerate(axes):dimensions[axis]+=int(pads[i])+int(pads[i+len(axes)])
        if any(d<=0 for d in dimensions):raise ValueError('Fixed Pad output must remain nonempty')
        input_info=views.infos.get(node.input[0])
        element=input_info.type.tensor_type.elem_type if input_info is not None else views.weights[node.input[0]].data_type
        existing=views.infos.get(node.output[0])
        if existing is not None:
            tensor=existing.type.tensor_type;known=list(tensor.shape.dim)
            if tensor.elem_type not in (0,element) or known and (len(known)!=len(dimensions) or any(d.dim_value>0 and d.dim_value!=v for d,v in zip(known,dimensions))):
                raise ValueError('Proved Pad dimensions conflict with existing metadata')
        if existing is not None and existing.type.tensor_type.elem_type==element and [d.dim_value for d in existing.type.tensor_type.shape.dim]==dimensions:
            verified.append(node.name);continue
        replacement=copy.deepcopy(existing) if existing is not None else helper.make_tensor_value_info(node.output[0],element,dimensions)
        replacement.type.tensor_type.elem_type=element
        shape=replacement.type.tensor_type.shape
        if not shape.dim:shape.dim.extend(onnx.TensorShapeProto.Dimension(dim_value=d) for d in dimensions)
        else:
            for dim,value in zip(shape.dim,dimensions):dim.dim_value=value
        assignments.append((existing,replacement,node.name))
        views.infos[node.output[0]]=replacement
        verified.append(node.name)
    # All dynamic data-dependent pads and conflicts fail before source metadata
    # is touched. Unknown input shapes remain explicitly unresolved.
    for existing,replacement,_ in assignments:
        if existing is None:model.graph.value_info.append(replacement)
        else:existing.CopyFrom(replacement)
    return dict(bound_pad_count=len(assignments),verified_pad_count=len(verified),
        bound_pad_names=[name for _,_,name in assignments],unresolved_pad_names=unresolved,
        complete=not unresolved,model_executed=False,gpu_validated=False)


def _calculation_digest(model):
    digest=hashlib.sha256()
    for values in (model.graph.node,model.graph.initializer):
        for value in values:
            raw=value.SerializeToString(deterministic=True)
            digest.update(len(raw).to_bytes(8,'little'));digest.update(raw)
    return digest.hexdigest()


def infer_with_fixed_pads(model, *, max_passes=4):
    if type(max_passes) is not int or not 1<=max_passes<=8:
        raise ValueError('Fixed metadata inference requires a bounded pass count')
    before=_calculation_digest(model)
    derived=onnx.shape_inference.infer_shapes(model,strict_mode=True,data_prop=True)
    names=[];passes=[]
    for _ in range(max_passes):
        report=bind_fixed_pad_shapes(derived);passes.append(report);names.extend(report['bound_pad_names'])
        if report['bound_pad_count']==0:break
        # Retain proved fixed information, then let ONNX propagate it again.
        fixed=[v for v in derived.graph.value_info if v.type.tensor_type.HasField('shape') and all(d.dim_value>0 for d in v.type.tensor_type.shape.dim)]
        del derived.graph.value_info[:];derived.graph.value_info.extend(fixed)
        derived=onnx.shape_inference.infer_shapes(derived,strict_mode=True,data_prop=True)
    if _calculation_digest(derived)!=before:raise ValueError('Shape propagation changed operations or coefficient bytes')
    return derived,dict(bound_pad_count=len(set(names)),bound_pad_names=sorted(set(names)),passes=passes,
        node_initializer_bytes_preserved=True,source_calculation_sha256=before,
        model_executed=False,gpu_validated=False)
