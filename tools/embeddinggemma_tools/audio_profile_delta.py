"""Compare immutable original audio ONNX profiles without executing a model."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
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
    def consumers(name):
        result=[];pending=[name];seen=set()
        while pending:
            value=pending.pop()
            if value in seen:continue
            seen.add(value)
            for node in source.graph.node:
                for index,operand in enumerate(node.input):
                    if operand!=value:continue
                    if node.op_type=='Identity' and not node.domain and len(node.input)==len(node.output)==1:
                        pending.extend(node.output)
                    else:result.append(dict(node=node.name,input_index=index))
        return sorted(result,key=lambda row:(row['node'],row['input_index']))
    for left,right in zip(source.graph.initializer,target.graph.initializer,strict=True):
        if left.name!=right.name:raise ValueError('Profile initializer order or name differs')
        if left==right:result['unchanged_initializer_count']+=1;continue
        check_tensor(left,right)
        result['changed_initializers'].append(dict(name=left.name,before=_tensor(left),after=_tensor(right),
                                                  consumers=consumers(left.name),before_proto_base64=base64.b64encode(left.SerializeToString()).decode('ascii'),
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
                                          consumers=consumers(left.output[0]),before_proto_base64=base64.b64encode(left.SerializeToString()).decode('ascii'),
                                          after_proto_base64=base64.b64encode(right.SerializeToString()).decode('ascii')))
    if len(source.graph.output)!=len(target.graph.output):raise ValueError('Profile output count differs')
    for left,right in zip(source.graph.output,target.graph.output,strict=True):
        if left.name!=right.name or left.type.tensor_type.elem_type!=right.type.tensor_type.elem_type:
            raise ValueError('Profile output name or type differs')
        result['outputs'].append(dict(before=_value(left),after=_value(right)))
    return result


def apply_profile_delta(models,delta,*,expected_delta_sha256,expected_source_sha256,expected_target_sha256):
    """Apply audited binary masks and final crop by stable consumer edges.

    Audit every partition before changing any one of them. Export-generated
    initializer and Identity names may differ; original tensor values may not.
    This prepares a candidate and never grants numerical/GPU acceptance.
    """
    digest=hashlib.sha256(json.dumps(delta,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (digest!=expected_delta_sha256 or delta.get('source_sha256')!=expected_source_sha256
            or delta.get('target_sha256')!=expected_target_sha256 or not delta.get('success')
            or delta.get('scope')!='original_onnx_profile_structure_comparison_only'):
        raise ValueError('Audited original profile delta identity differs')
    indexes=[]
    for model in models:
        indexes.append(({w.name:w for w in model.graph.initializer},
                        {v:n for n in model.graph.node for v in n.output}))
    pending={};outputs=[]
    def same(left,right):
        return (left.data_type==right.data_type and list(left.dims)==list(right.dims)
                and numpy_helper.to_array(left).tobytes()==numpy_helper.to_array(right).tobytes())
    def locate(part,node,index):
        weights,producers=indexes[part];value=node.input[index];seen=set()
        while value not in weights:
            if value in seen:raise ValueError('Cyclic profile alias')
            seen.add(value);producer=producers.get(value)
            if producer is None:raise ValueError('Profile operand has no fixed producer')
            if producer.op_type=='Identity' and not producer.domain and len(producer.input)==1:
                value=producer.input[0];continue
            if (producer.op_type=='Constant' and not producer.domain and len(producer.attribute)==1
                    and producer.attribute[0].name=='value' and producer.attribute[0].type==onnx.AttributeProto.TENSOR):
                return ('node',part,producer.name),producer.attribute[0].t
            raise ValueError('Profile operand is no longer a fixed tensor')
        return ('initializer',part,value),weights[value]
    def stage(row,before,after):
        if not row.get('consumers'):raise ValueError('Profile delta has no audited consumers')
        for role in row['consumers']:
            found=0
            for part,model in enumerate(models):
                for node in model.graph.node:
                    if node.name!=role['node']:continue
                    index=role['input_index']
                    if not isinstance(index,int) or not 0<=index<len(node.input):raise ValueError('Profile operand index differs')
                    key,tensor=locate(part,node,index)
                    if not same(tensor,before):raise ValueError('Derived profile operand differs from the original full profile')
                    replacement=onnx.TensorProto();replacement.CopyFrom(after);replacement.name=tensor.name
                    if key in pending and pending[key][1]!=replacement:raise ValueError('Conflicting profile replacements')
                    pending[key]=(tensor,replacement);found+=1
            if not found:raise ValueError('Audited profile consumer is missing: '+role['node'])
    for row in delta['changed_initializers']:
        before=onnx.TensorProto.FromString(base64.b64decode(row['before_proto_base64']))
        after=onnx.TensorProto.FromString(base64.b64decode(row['after_proto_base64']))
        if _sha(before)!=row['before']['sha256'] or _sha(after)!=row['after']['sha256']:
            raise ValueError('Original profile tensor receipt differs')
        if (before.data_type not in (onnx.TensorProto.BOOL,onnx.TensorProto.FLOAT)
                or after.data_type!=before.data_type or list(before.dims)!=list(after.dims)
                or max(before.ByteSize(),after.ByteSize())>2*1024*1024
                or any(not np.isin(numpy_helper.to_array(v),[0,1]).all() for v in (before,after))):
            raise ValueError('Only bounded binary profile masks may change')
        stage(row,before,after)
    for row in delta['changed_nodes']:
        before=onnx.NodeProto.FromString(base64.b64decode(row['before_proto_base64']))
        after=onnx.NodeProto.FromString(base64.b64decode(row['after_proto_base64']))
        if (_sha(before)!=row['before_sha256'] or _sha(after)!=row['after_sha256']
                or before.op_type!=after.op_type or before.op_type!='Constant'
                or len(before.attribute)!=len(after.attribute) or len(before.attribute)!=1):
            raise ValueError('Original profile Constant receipt differs')
        a,b=before.attribute[0].t,after.attribute[0].t
        if (a.data_type!=onnx.TensorProto.INT64 or b.data_type!=a.data_type or list(a.dims)!=[1] or list(b.dims)!=[1]
                or not 0<int(numpy_helper.to_array(b).item())<=int(numpy_helper.to_array(a).item())):
            raise ValueError('Only a shorter positive final crop is supported')
        stage(row,a,b)
    for row in delta['outputs']:
        before,after=row['before'],row['after'];found=0
        if (before['name']!=after['name'] or before['dtype']!=after['dtype']
                or not before['shape'] or len(before['shape'])!=len(after['shape'])
                or before['shape'][1:]!=after['shape'][1:] or not 0<after['shape'][0]<=before['shape'][0]):
            raise ValueError('Only the final output row count may change')
        for model in models:
            for value in (*model.graph.output,*model.graph.value_info):
                if value.name!=before['name']:continue
                if _value(value)!=before:raise ValueError('Derived final output contract differs')
                outputs.append((value,after['shape']))
                if value in model.graph.output:found+=1
        if not found:raise ValueError('Original final output is missing')
    if not pending:raise ValueError('No profile changes were audited')
    for tensor,replacement in pending.values():tensor.CopyFrom(replacement)
    for value,shape in outputs:
        for dimension,size in zip(value.type.tensor_type.shape.dim,shape,strict=True):dimension.dim_value=size
    return dict(profile_items_applied=len(delta['changed_initializers'])+len(delta['changed_nodes']),
                updated_tensor_count=len(pending),updated_output_contracts=len(outputs),
                changed_parts=sorted({key[1] for key in pending}),gpu_validated=False,numeric_validated=False,
                delta_sha256=digest,original_source_sha256=expected_source_sha256,original_target_sha256=expected_target_sha256)


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
