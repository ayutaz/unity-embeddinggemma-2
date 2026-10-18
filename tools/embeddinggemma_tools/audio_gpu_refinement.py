"""Audit and prepare fixed candidate audio GPU arithmetic; no GPU acceptance."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import onnx
from onnx import helper,numpy_helper
import torch

from .projection_coefficients import _Views
from .export import (export_audio_native_onnx,fp32_fma,fp32_refined_rstd,
    fp32_cascade_mean1024,fp32_cascade_mean1536,fp32_outer_cascade_mean1024)

SOURCE='0cee4c7b77c89410895d52e4420816a5e66a8845'
RUN=39117103059
SEEDS={
 'silence':dict(artifact_id=12132399063,archive_sha256='f1d458f90322972a09adf84da4f2326ae620790853fce8272ad8cb5072550864',
    report_sha256='4765d65c23ae380145eb25a13ec662d35b5ee3a0370b54ceaeda16dcd440871e',onnx_sha256='525871a8dbccbbd75b5c521ab959588323938e58fac361d69035eb80a61d2095',onnx_bytes=1231902161),
 'padded_noise':dict(artifact_id=12132404118,archive_sha256='8c7fdb42c84c01d5ec342defa2452391309819aed8fa65941834c64b91139805',
    report_sha256='95fb569a029ba2c2a3aa82c95b22e3440bd351f21b68622f7610fdde5217ba49',onnx_sha256='87a2a058da07b0ce4338769723d0dbb572a3048ae782a8db79a37e952319ee99',onnx_bytes=1231902858)
}
STEM_SHA='68309f01ea3e738fccb24a0db5db608841a3a8e59b0a4fa5d0fb521ed60bf43c'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def checked_seed(directory,*,case_id):
    from .audio_generation_candidate import checked_file
    if case_id not in SEEDS:raise ValueError('One fixed seed profile is required')
    root=Path(directory);spec=SEEDS[case_id];receipt=root/'audio-gpu-seed.json'
    if not receipt.is_file() or receipt.is_symlink() or digest(receipt)!=spec['report_sha256']:
        raise ValueError('Pinned seed receipt hash differs')
    report=json.loads(receipt.read_text(encoding='utf-8'))
    if (report.get('success') is not True or report.get('source')!=SOURCE
            or report.get('onnx_sha256')!=spec['onnx_sha256'] or report.get('onnx_bytes')!=spec['onnx_bytes']
            or report.get('source_tensor_hashes_preserved_after_export') is not True
            or any(report.get(k) is not False for k in ('numeric_acceptance','gpu_validated','generation_qualified','original_golden_changed'))):
        raise ValueError('Pinned unqualified seed provenance differs')
    path=checked_file(root,'audio.onnx',{'bytes':spec['onnx_bytes'],'sha256':spec['onnx_sha256']})
    return path,report


def fetch_seed(directory,*,case_id):
    from .artifact_zip import github_artifact_reader,extract_members,RangeReader
    if case_id not in SEEDS:raise ValueError('One fixed seed profile is required')
    directory=Path(directory)
    if directory.exists():raise ValueError('Seed download requires a new directory')
    spec=SEEDS[case_id]
    reader,metadata=github_artifact_reader('ayutaz/unity-embeddinggemma-2',spec['artifact_id'])
    try:
        provenance=metadata.get('workflow_run',{})
        if (metadata.get('id')!=spec['artifact_id'] or metadata.get('digest')!='sha256:'+spec['archive_sha256']
                or provenance.get('id')!=RUN or provenance.get('head_sha')!=SOURCE
                or provenance.get('repository_id')!=1408387878 or provenance.get('head_repository_id')!=1408387878):
            raise ValueError('Pinned seed artifact provenance differs')
        directory.mkdir(parents=True)
        archive=directory/'original-upload.zip'
        with archive.open('xb') as stream:
            for block in iter(lambda:reader.read(32*1024*1024),b''):stream.write(block)
        if digest(archive)!=spec['archive_sha256']:raise ValueError('Pinned seed archive bytes differ')
        with archive.open('rb') as stream:
            def fetch(start,stop):
                # This local reader is deliberately serial: the stream is shared.
                stream.seek(start);return stream.read(stop-start)
            with RangeReader(archive.stat().st_size,fetch,max_parallel=1) as local:
                selection=extract_members(local,directory,['audio.onnx','audio-gpu-seed.json'],max_entry_bytes=2**31)
        checked_seed(directory,case_id=case_id)
        return dict(artifact_id=spec['artifact_id'],run=RUN,source=SOURCE,archive_sha256=spec['archive_sha256'],
            archive_bytes=archive.stat().st_size,selected_files=selection['files'],member_hashes_verified=True)
    finally:reader.close()


def _destination(directory,output,source):
    if re.fullmatch('[0-9a-f]{40}',source or '') is None:raise ValueError('Immutable execution source is required')
    directory,output=Path(directory).resolve(),Path(output).resolve()
    if output.exists() or output.is_relative_to(directory) or directory.is_relative_to(output):
        raise ValueError('Refinement requires new output separate from source evidence')
    return directory,output


def prepare(directory,output,*,case_id,source):
    directory,output=_destination(directory,output,source)
    path,seed=checked_seed(directory,case_id=case_id)
    output.mkdir(parents=True)
    report=dict(schema=1,scope='unqualified_saved_candidate_gpu_refined_partition_preparation_only',
        success=False,source=source,case_id=case_id,seed_source=SOURCE,seed_run=RUN,
        seed_artifact_id=SEEDS[case_id]['artifact_id'],seed_onnx_sha256=SEEDS[case_id]['onnx_sha256'],
        seed_receipt_sha256=SEEDS[case_id]['report_sha256'],numeric_acceptance=False,
        gpu_validated=False,generation_qualified=False,original_golden_changed=False,model_executed=False,
        model_execution_scope='Small arithmetic template export tracing only; full encoder not executed',
        versions=dict(torch=str(torch.__version__),onnx=onnx.__version__),phase='load_verified_seed')
    receipt=output/'audio-gpu-refinement.json'
    def save():receipt.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    def phase(value):report['phase']=value;save()
    save()
    try:
        model=onnx.load_model(path,load_external_data=False)
        plan=refine_and_partition(model,output,phase=phase)
        plan.update(producer=source,source_graph_sha256=SEEDS[case_id]['onnx_sha256'],
            seed_source=SOURCE,seed_run=RUN,seed_artifact_id=SEEDS[case_id]['artifact_id'],
            seed_receipt_sha256=SEEDS[case_id]['report_sha256'],
            profile='frames-1119-valid-1119' if case_id=='silence' else 'frames-1119-valid-839',
            candidate_provenance=seed.get('provenance',{}))
        (output/'partition-plan.json').write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        report.update(success=True,phase='prepared_not_numerically_qualified',
            operation_count=len(plan['operations']),segment_count=len(plan['segments']),
            plan_sha256=digest(output/'partition-plan.json'),source_initializer_sha256=plan['original_initializer_sha256'])
    except Exception as error:
        report['error_type']=type(error).__name__;save();raise
    save();return report


def audit_refinement_targets(model,*,expected_rms=109,expected_outer=12,stem_weight_sha256=STEM_SHA):
    views=_Views(model);nodes=list(model.graph.node);weights=views.weights
    infos=views.infos;producers=views.producers
    def attrs(node):return views.attributes(node)
    def fixed(name):
        shape=views.shape(name);info=infos.get(name)
        if info is None or info.type.tensor_type.elem_type!=onnx.TensorProto.FLOAT:raise ValueError('Refinement data must be fixed FP32')
        return list(shape)
    def scalar(name,value):
        tensor=weights.get(name)
        node=producers.get(name)
        if tensor is None and node is not None and node.op_type=='Constant':tensor=attrs(node).get('value')
        if tensor is None or tensor.data_type!=onnx.TensorProto.FLOAT:return False
        array=views.tensor(tensor)
        return array.size==1 and array.item()==np.float32(value)
    normalizations=[]
    for inverse in nodes:
        if inverse.op_type!='Reciprocal':continue
        if not inverse.name.endswith('/Reciprocal') or len(inverse.input)!=1:raise ValueError('Unqualified inverse scope')
        root=producers.get(inverse.input[0]);prefix=inverse.name.rsplit('/',1)[0]
        if root is None or root.op_type!='Sqrt' or root.name!=prefix+'/Sqrt':raise ValueError('Normalization inverse pair differs')
        vectors={value for n in nodes if n.name.startswith(prefix+'/') and n.op_type=='Mul' for value in n.input
            if value in weights and weights[value].data_type==onnx.TensorProto.FLOAT and list(weights[value].dims) in ([32],[128])}
        if len(vectors)!=1:raise ValueError('Normalization scale binding is not unique')
        width=weights[next(iter(vectors))].dims[0]
        expected=[1,560,64,1] if width==128 else [1,280,32,1]
        if fixed(root.input[0])!=expected:raise ValueError('Normalization statistic geometry differs')
        normalizations.append(dict(prefix=prefix,width=width))
    if sorted(x['width'] for x in normalizations)!=[32,128]:raise ValueError('Both original normalization widths are required')
    means=[n for n in nodes if n.op_type=='ReduceMean'];outer=[]
    if len(means)!=expected_rms:raise ValueError('Original RMS target count differs')
    for mean in means:
        shape=fixed(mean.input[0]);square=producers.get(mean.input[0]);a=attrs(mean)
        axes=views.read(mean.input[1],True).reshape(-1).tolist() if len(mean.input)==2 else a.get('axes',[])
        if shape not in ([1,280,1024],[1,280,1536]) or axes not in ([-1],[2]) or a.get('keepdims',1)!=1:
            raise ValueError('Original fixed last-axis RMS geometry differs')
        if square is None or not ((square.op_type=='Pow' and scalar(square.input[1],2)) or (square.op_type=='Mul' and square.input[0]==square.input[1])):
            raise ValueError('RMS square edge differs')
        data=producers.get(square.input[0])
        if data is not None and data.op_type=='Clip':data=producers.get(data.input[0])
        if data is not None and data.op_type=='Transpose':
            convolution=producers.get(data.input[0]);c=attrs(convolution) if convolution is not None else {}
            weight=weights.get(convolution.input[1]) if convolution is not None and len(convolution.input)>1 else None
            if (convolution is None or convolution.op_type!='Conv' or attrs(data).get('perm')!=[0,2,1]
                    or weight is None or list(weight.dims)!=[1024,1,5] or weight.data_type!=onnx.TensorProto.FLOAT
                    or c.get('group',1)!=1024 or c.get('kernel_shape')!=[5] or c.get('strides',[1])!=[1]
                    or c.get('dilations',[1])!=[1] or c.get('pads',[0,0])!=[0,0]
                    or fixed(convolution.input[0])!=[1,1024,284] or fixed(convolution.output[0])!=[1,1024,280]):
                raise ValueError('Outer RMS native convolution layout is unproved')
            outer.append(mean.name)
    if len(outer)!=expected_outer:raise ValueError('Original outer RMS count differs')
    convolutions=[]
    for node in nodes:
        if node.op_type!='Conv' or len(node.input)!=2:continue
        weight=weights.get(node.input[1]);a=attrs(node)
        if weight is None or list(weight.dims)!=[32,128,3,3]:continue
        if (weight.data_type!=onnx.TensorProto.FLOAT or a.get('group',1)!=1 or a.get('strides')!=[2,2]
                or a.get('pads')!=[1,1,1,1] or a.get('dilations',[1,1])!=[1,1] or a.get('kernel_shape')!=[3,3]
                or fixed(node.input[0])!=[1,128,560,64] or fixed(node.output[0])!=[1,32,280,32]
                or hashlib.sha256(views.tensor(weight).astype('<f4').tobytes()).hexdigest()!=stem_weight_sha256):
            raise ValueError('Original fixed stem convolution differs')
        convolutions.append(node.name)
    if len(convolutions)!=1:raise ValueError('One original second stem convolution is required')
    return dict(normalizations=normalizations,rms_count=len(means),outer_rms_mean_names=outer,
        stem_conv_name=convolutions[0],stem_weight_sha256=stem_weight_sha256,model_executed=False,gpu_validated=False)


def partition_boundaries(model):
    from .projection_coefficients import resolve_projection
    weights={w.name:w for w in model.graph.initializer}
    producers={v:n for n in model.graph.node for v in n.output}
    selected=[];hashes=[];dynamic=[]
    for node in model.graph.node:
        if node.op_type not in ('Gemm','MatMul'):continue
        try:_,metadata=resolve_projection(model,node.name)
        except ValueError:
            # An unresolved Gemm is never a dynamic matrix product. Every
            # remaining MatMul is checked for explicit FP32 geometry by the
            # planner, and the exact expected counts are checked below.
            if node.op_type!='MatMul':raise
            dynamic.append(node.name);continue
        selected.append(node.name);hashes.append(metadata['weight_sha256'])
    silu=[n.name for n in model.graph.node if n.op_type=='Div' and len(n.input)==2
        and n.input[1].startswith('__embeddinggemma_silu_') and n.input[1].endswith('_denominator')]
    scales=[n.name for n in model.graph.node if n.op_type=='Mul' and len(n.input)==2
        and n.input[1] in weights and n.input[1].endswith('.per_dimension_query_scale')]
    tanh=[n.name for n in model.graph.node if n.op_type=='Tanh']
    divide=[]
    for node in model.graph.node:
        if node.op_type!='Tanh':continue
        parent=producers.get(node.input[0])
        if parent is None or parent.op_type!='Div':raise ValueError('Native Tanh scalar division is missing')
        divide.append(parent.name)
    softmax=[n.name for n in model.graph.node if n.op_type=='Softmax']
    glu=[n.name for n in model.graph.node if n.op_type=='Mul' and len(n.input)==2
        and n.input[1] in producers and producers[n.input[1]].op_type=='Sigmoid']
    selections=dict(node_names=selected,silu_node_names=silu,dynamic_node_names=dynamic,
        query_scale_node_names=scales,tanh_node_names=tanh,scalar_divide_node_names=divide,
        softmax_node_names=softmax,glu_node_names=glu)
    expected=dict(node_names=135,silu_node_names=36,dynamic_node_names=36,query_scale_node_names=12,
        tanh_node_names=12,scalar_divide_node_names=12,softmax_node_names=12,glu_node_names=12)
    for key,names in selections.items():
        if len(names)!=expected[key] or len(set(names))!=len(names):raise ValueError('Fixed candidate GPU operation selection differs: '+key)
    return dict(**selections,expected_weight_sha256=hashes)


def write_partition_files(model,parts,plan,directory):
    from .partition_projections import projection_weight_array
    from .audio_candidate_gpu_seed import audit_seed_metadata
    directory=Path(directory);weights={w.name:w for w in model.graph.initializer};files={}
    def record(path):files[path.name]=dict(bytes=path.stat().st_size,sha256=digest(path))
    for index,operation in enumerate(p for p in plan['operations'] if p['kind']=='matmul'):
        path=directory/f'weight-{index}.bin';matrix=projection_weight_array(model,operation)
        with path.open('xb') as stream:stream.write(matrix.astype('<f4').tobytes())
        if digest(path)!=operation['weight_sha256']:raise ValueError('Written projection coefficient hash differs')
        operation['file']=path.name;record(path)
    # plan.projections and plan.operations share objects in memory. JSON readers
    # get complete independent rows with the same exact coefficient file.
    by_name={p['node']:p for p in plan['operations'] if p['kind']=='matmul'}
    for operation in plan['projections']:operation.update(by_name[operation['node']])
    for index,operation in enumerate(p for p in plan['operations'] if p['kind']=='sequential_scale'):
        weight=weights.get(operation['scale_weight'])
        if weight is None or weight.data_type!=onnx.TensorProto.FLOAT:raise ValueError('Original query coefficient is missing')
        path=directory/f'query-scale-{index}.bin'
        with path.open('xb') as stream:stream.write(numpy_helper.to_array(weight).astype('<f4').tobytes())
        if digest(path)!=operation['weight_sha256']:raise ValueError('Written query coefficient hash differs')
        operation['file']=path.name;record(path)
    audits=[]
    if len(parts)!=len(plan['segments']):raise ValueError('Partition segment count differs')
    for index,part in enumerate(parts):
        if any(v.type.tensor_type.elem_type!=onnx.TensorProto.FLOAT or not v.type.tensor_type.HasField('shape')
                or not v.type.tensor_type.shape.dim or any(d.dim_value<=0 for d in v.type.tensor_type.shape.dim)
                for v in (*part.graph.input,*part.graph.output)):
            raise ValueError('Every partition boundary requires fixed positive FP32 geometry')
        sanitized,audit=audit_seed_metadata(part)
        onnx.checker.check_model(sanitized)
        path=directory/f'segment-{index}.onnx'
        if path.exists():raise ValueError('Refinement never overwrites prior segment evidence')
        onnx.save_model(sanitized,path);plan['segments'][index]['file']=path.name;record(path);audits.append(audit)
    plan['files']=files
    plan['metadata_audits']=dict(all_segments_textual_metadata_checked=True,
        segment_count=len(audits),optional_metadata_fields_removed=sum(a['optional_metadata_fields_removed'] for a in audits),
        segment_computation_unchanged=all(a['computation_unchanged'] for a in audits))


def refine_and_partition(model,output,*,phase):
    from .fixed_shape_metadata import infer_with_fixed_pads
    from .refine_layernorm_onnx import (_initializer_digest,refine_layernorm_inverses,
        refine_welford_scalar_means,refine_welford_scalar_moments)
    from .refine_rms_onnx import refine_rms_means
    from .refine_conv_onnx import refine_conv2d
    from .projection_coefficients import split_gemm_bias
    from .gpu_graph_normalization import normalize_glu_slices
    from .partition_projections import partition_static_projections
    count=len(model.graph.initializer);original_digest=_initializer_digest(model.graph.initializer)
    changes={}
    def preserved():
        if _initializer_digest(model.graph.initializer[:count])!=original_digest:raise ValueError('Original seed initializer bytes changed')
    phase('prove_fixed_source_geometry')
    model,changes['source_shape_metadata']=infer_with_fixed_pads(model)
    targets=audit_refinement_targets(model)
    phase('export_small_arithmetic_templates')
    templates=make_templates(output/'templates')
    phase('refine_both_layernorm_inverses')
    changes['layernorm_inverses']=refine_layernorm_inverses(model,templates['inverse'],prefixes=tuple(x['prefix'] for x in targets['normalizations']));preserved()
    changes['scalar_means']=[];changes['scalar_moments']=[]
    for target in targets['normalizations']:
        phase('refine_layernorm_scalar_mean_'+str(target['width']))
        changes['scalar_means'].append(refine_welford_scalar_means(model,templates['fma'],prefix=target['prefix']+'/',expected_count=8));preserved()
        phase('refine_layernorm_scalar_moment_'+str(target['width']))
        changes['scalar_moments'].append(refine_welford_scalar_moments(model,templates['fma'],prefix=target['prefix']+'/',width=target['width'],expected_count=8));preserved()
    phase('refine_all_109_rms_paths')
    changes['rms']=refine_rms_means(model,templates['means'],templates['inverse'],expected_count=109,
        mean_overrides={name:templates['outer'] for name in targets['outer_rms_mean_names']});preserved()
    phase('refine_second_stem_convolution')
    changes['convolution']=refine_conv2d(model,templates['conv'],node_name=targets['stem_conv_name'],expected_weight_sha256=targets['stem_weight_sha256']);preserved()
    phase('normalize_original_bias_and_glu_slices')
    changes['gemm_bias']=split_gemm_bias(model)
    changes['glu_slices']=normalize_glu_slices(model);preserved()
    if changes['gemm_bias']['split_biased_gemm_count']!=1 or changes['glu_slices']['normalized_glu_count']!=12:
        raise ValueError('Original biased Gemm or GLU normalization count differs')
    phase('prove_refined_partition_geometry')
    model,changes['refined_shape_metadata']=infer_with_fixed_pads(model);preserved()
    phase('select_audited_gpu_boundaries')
    selection=partition_boundaries(model)
    phase('partition_gpu_graph')
    parts,plan=partition_static_projections(model,**selection)
    if len(plan['operations'])!=267 or len(parts)!=268:raise ValueError('Full candidate GPU partition count differs')
    phase('audit_and_write_partition_payloads')
    write_partition_files(model,parts,plan,output)
    preserved()
    plan.update(schema=1,scope='unqualified_fixed_candidate_gpu_partition_plan',
        original_initializer_sha256=original_digest,original_initializer_count=count,
        original_initializer_bytes_preserved=True,refinement_targets=targets,refinements=changes,
        original_references_changed=False,original_golden_changed=False,gpu_validated=False,
        numeric_acceptance=False,generation_qualified=False,model_executed=False,
        operation_count=len(plan['operations']))
    (output/'partition-plan.json').write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return plan


def make_templates(directory):
    from .compensated_conv import fp32_block16_conv2d
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    class Inverse(torch.nn.Module):
        def forward(self,x):return fp32_refined_rstd(x)
    class Fma(torch.nn.Module):
        def forward(self,a,b,c):return fp32_fma(a,b,c)
    class Mean(torch.nn.Module):
        def __init__(self,width):super().__init__();self.width=width
        def forward(self,x):return fp32_cascade_mean1024(x) if self.width==1024 else fp32_cascade_mean1536(x)
    class Outer(torch.nn.Module):
        def forward(self,x):return fp32_outer_cascade_mean1024(x)
    class Conv(torch.nn.Module):
        def forward(self,x,w):return fp32_block16_conv2d(x,w)
    def export(module,value,name):
        path=directory/(name+'.onnx');export_audio_native_onnx(module,value,path);return onnx.load(path)
    result={'inverse':export(Inverse(),torch.ones(1,1),'inverse'),
        'means':{w:export(Mean(w),torch.ones(1,1,w),'mean'+str(w)) for w in (1024,1536)},
        'outer':export(Outer(),torch.ones(1,1024,280).transpose(1,2),'outer')}
    path=directory/'fma.onnx'
    torch.onnx.export(Fma(),(torch.ones(1,1),)*3,path,dynamo=False,opset_version=20,input_names=['a','b','c'],output_names=['output'])
    result['fma']=onnx.version_converter.convert_version(onnx.load(path),24);onnx.save(result['fma'],path)
    path=directory/'conv.onnx'
    torch.onnx.export(Conv(),(torch.ones(1,128,5,4),torch.ones(32,128,3,3)),path,dynamo=False,opset_version=20,
        input_names=['values','weight'],output_names=['output'],dynamic_axes={'values':{2:'height',3:'width'},'output':{2:'out_height',3:'out_width'}})
    conv=onnx.version_converter.convert_version(onnx.load(path),24)
    for dim,value in zip(conv.graph.input[0].type.tensor_type.shape.dim,[1,128,560,64]):dim.dim_value=value
    for dim,value in zip(conv.graph.output[0].type.tensor_type.shape.dim,[1,32,280,32]):dim.dim_value=value
    onnx.save(conv,path);result['conv']=conv
    return result


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',choices=tuple(SEEDS),required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--fetch',action='store_true')
    args=parser.parse_args(argv)
    try:
        _destination(args.input,args.output,args.source)
        if args.fetch:fetch_seed(args.input,case_id=args.case)
        report=prepare(args.input,args.output,case_id=args.case,source=args.source)
        print(json.dumps({k:report[k] for k in ('success','phase','operation_count','segment_count','gpu_validated','generation_qualified')}))
        return 0
    except Exception as error:
        print(json.dumps(dict(success=False,error_type=type(error).__name__,gpu_validated=False,generation_qualified=False)))
        return 1


if __name__=='__main__':raise SystemExit(main())
