"""Compare immutable original audio ONNX profiles without executing a model."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import onnx
from onnx import numpy_helper


def _sha(message):return hashlib.sha256(message.SerializeToString()).hexdigest()


def _tensor(value):
    array=numpy_helper.to_array(value)
    result=dict(dtype=onnx.TensorProto.DataType.Name(value.data_type),shape=list(value.dims),sha256=_sha(value))
    if array.size<=64:result['values']=array.reshape(-1).tolist()
    return result


def _value(value):
    tensor=value.type.tensor_type
    return dict(name=value.name,dtype=onnx.TensorProto.DataType.Name(tensor.elem_type),
                shape=[d.dim_value if d.HasField('dim_value') else d.dim_param for d in tensor.shape.dim])


def _node(value):
    return dict(name=value.name,op_type=value.op_type,attributes=[dict(name=a.name,tensor=_tensor(a.t)) for a in value.attribute])


def compare_profiles(source,target,*,source_sha256,target_sha256,max_changed_tensor_bytes=2*1024*1024):
    """Report bounded Constant/initializer changes; reject topology changes.

    This is structural evidence only. The resulting delta is not permission to
    replace arbitrary coefficients and never counts as numerical acceptance.
    """
    if _sha(source)!=source_sha256 or _sha(target)!=target_sha256:
        raise ValueError('Immutable original profile SHA differs')
    if (not isinstance(max_changed_tensor_bytes,int) or max_changed_tensor_bytes<1
            or list(source.opset_import)!=list(target.opset_import)
            or list(source.graph.input)!=list(target.graph.input)):
        raise ValueError('Profile opset or input contract differs')
    result=dict(success=True,scope='original_onnx_profile_structure_comparison_only',
                source_sha256=source_sha256,target_sha256=target_sha256,
                numeric_validated=False,gpu_validated=False,changed_initializers=[],changed_nodes=[],outputs=[],
                unchanged_initializer_count=0,unchanged_node_count=0)
    if len(source.graph.initializer)!=len(target.graph.initializer) or len(source.graph.node)!=len(target.graph.node):
        raise ValueError('Profile node or initializer count differs')
    def check_tensor(left,right):
        if left.data_type!=right.data_type or list(left.dims)!=list(right.dims):
            raise ValueError('Changed profile tensor type or shape differs')
        if max(left.ByteSize(),right.ByteSize())>max_changed_tensor_bytes:
            raise ValueError('Changed profile tensor exceeds bounded payload')
        if left.data_location or right.data_location:
            raise ValueError('External profile tensor cannot be audited in memory')
    for left,right in zip(source.graph.initializer,target.graph.initializer,strict=True):
        if left.name!=right.name:raise ValueError('Profile initializer order or name differs')
        if left==right:result['unchanged_initializer_count']+=1;continue
        check_tensor(left,right)
        result['changed_initializers'].append(dict(name=left.name,before=_tensor(left),after=_tensor(right),
                                                  after_proto_base64=base64.b64encode(right.SerializeToString()).decode('ascii')))
    for left,right in zip(source.graph.node,target.graph.node,strict=True):
        if left==right:result['unchanged_node_count']+=1;continue
        if (left.name!=right.name or left.op_type!=right.op_type or left.domain!=right.domain
                or list(left.input)!=list(right.input) or list(left.output)!=list(right.output)
                or left.op_type!='Constant' or len(left.attribute)!=1 or len(right.attribute)!=1
                or left.attribute[0].name!='value' or right.attribute[0].name!='value'
                or left.attribute[0].type!=onnx.AttributeProto.TENSOR or right.attribute[0].type!=onnx.AttributeProto.TENSOR):
            raise ValueError('Only bounded Constant tensor changes are supported; topology differs')
        check_tensor(left.attribute[0].t,right.attribute[0].t)
        result['changed_nodes'].append(dict(name=left.name,before_sha256=_sha(left),after_sha256=_sha(right),
                                          before=_node(left),after=_node(right),
                                          after_proto_base64=base64.b64encode(right.SerializeToString()).decode('ascii')))
    if len(source.graph.output)!=len(target.graph.output):raise ValueError('Profile output count differs')
    for left,right in zip(source.graph.output,target.graph.output,strict=True):
        if left.name!=right.name or left.type.tensor_type.elem_type!=right.type.tensor_type.elem_type:
            raise ValueError('Profile output name or type differs')
        result['outputs'].append(dict(before=_value(left),after=_value(right)))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','target','output'):parser.add_argument('--'+name,required=True,type=Path)
    for name in ('source-sha256','target-sha256'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv)
    if args.output.exists():raise ValueError('Profile delta report already exists')
    result=compare_profiles(onnx.load(args.source),onnx.load(args.target),source_sha256=args.source_sha256,target_sha256=args.target_sha256)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(changed_initializers=len(result['changed_initializers']),changed_nodes=len(result['changed_nodes']),numeric_validated=False)))


if __name__=='__main__':main()
