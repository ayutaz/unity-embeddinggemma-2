"""Derive an unqualified GPU-preparation seed from the exact saved candidate.

Only the two audited native LayerNorm calls change. All other graph operations
and their original tensor bindings are borrowed, not reloaded from a checkpoint.
This is preparation, not original-reference or GPU numerical acceptance.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import onnx
import torch

from .audio_native_norm_replay import checked_normalizations
from .export import TorchWelfordLayerNorm, export_audio_native_onnx
from .privacy import scan_bytes


def _resolve(root, target):
    for part in target.split('.'):
        root = getattr(root, part)
    return root


def _tensor_bindings(program):
    bindings = {}
    for node in program.graph.nodes:
        if node.op == 'get_attr':
            value = _resolve(program, node.target)
            if not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
                raise ValueError('Saved seed requires original CPU tensor attributes')
            bindings[node.target] = value
    return bindings


def _binding_digest(bindings):
    rows = []
    for name, value in sorted(bindings.items()):
        array = value.detach().contiguous().numpy()
        rows.append((name, str(value.dtype), list(value.shape), hashlib.sha256(memoryview(array).cast('B')).hexdigest()))
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


class _WelfordArguments(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, values, weight):
        context = SimpleNamespace(width=self.width, weight=weight, eps=1e-6,
            refine_inverse=False, fuse_scalar_mean=False, fuse_scalar_moment=False)
        return TorchWelfordLayerNorm.forward(context, values)


def rewrite_welford_seed(program):
    selected = checked_normalizations(program)
    if sorted(node.args[1][0] for node in selected) != [32, 128]:
        raise ValueError('Seed requires exactly the original width-32 and width-128 normalizations')
    bindings = _tensor_bindings(program)
    for node in selected:
        weight = node.args[2]
        if not isinstance(weight, torch.fx.Node) or weight.op != 'get_attr':
            raise ValueError('Normalization scale must retain its original parameter binding')
        value = bindings[weight.target]
        if value.dtype != torch.float32 or tuple(value.shape) != tuple(node.args[1]) or not torch.isfinite(value).all():
            raise ValueError('Normalization scale must be the original finite FP32 vector')
    # graph_copy copies graph routing, not tensor storage or the source module.
    graph = torch.fx.Graph(); result = graph.graph_copy(program.graph, {})
    # ExportedProgram.module() unflattens its one-element graph tuple in an
    # output hook. The ordinary derived GraphModule returns that same Tensor.
    if not isinstance(result, (tuple, list)) or len(result) != 1 or not isinstance(result[0], torch.fx.Node):
        raise ValueError('Saved audio seed requires one Tensor output')
    graph.output(result[0])
    derived = torch.fx.GraphModule(program, graph)
    for index, node in enumerate(checked_normalizations(derived)):
        name = '_eg_welford_seed_' + str(index)
        while hasattr(derived, name): name += '_'
        derived.add_module(name, _WelfordArguments(node.args[1][0]))
        with graph.inserting_before(node):
            replacement = graph.call_module(name, (node.args[0], node.args[2]))
        for user in list(node.users):
            if user.args[1] == 0: user.replace_all_uses_with(replacement)
            graph.erase_node(user)
        graph.erase_node(node)
    graph.lint(); derived.recompile()
    actual = _tensor_bindings(derived)
    if set(actual) != set(bindings) or any(actual[name] is not value for name, value in bindings.items()):
        raise ValueError('Derived graph changed an original tensor binding')
    return derived, dict(replaced_normalizations=2, normalization_widths=[32, 128],
        source_parameter_objects_preserved=True, preserved_tensor_bindings=len(bindings),
        source_tensor_binding_sha256=_binding_digest(bindings), new_learned_parameters=0,
        gpu_validated=False, generation_qualified=False)


def audit_seed_metadata(model):
    sanitized = copy.deepcopy(model); removed = 0
    def visit(message):
        nonlocal removed
        if message.DESCRIPTOR.full_name == 'onnx.TensorProto':
            if message.external_data or message.data_location == onnx.TensorProto.EXTERNAL or message.string_data:
                raise ValueError('Seed metadata requires embedded numerical tensors')
        for field, value in list(message.ListFields()):
            if field.name in ('doc_string', 'metadata_props'):
                message.ClearField(field.name); removed += 1
            elif field.type == field.TYPE_MESSAGE:
                for child in value if field.is_repeated else [value]: visit(child)
            elif field.type in (field.TYPE_STRING, field.TYPE_BYTES) and field.name != 'raw_data':
                for text in value if field.is_repeated else [value]:
                    data = text.encode() if isinstance(text, str) else text
                    if scan_bytes(data, '<onnx-metadata>'):
                        raise ValueError('Seed metadata privacy validation failed')
    visit(sanitized)
    def equivalent(before, after):
        left = {f.name: (f, v) for f,v in before.ListFields() if f.name not in ('doc_string','metadata_props')}
        right = {f.name: (f, v) for f,v in after.ListFields() if f.name not in ('doc_string','metadata_props')}
        if left.keys() != right.keys(): return False
        for name,(field,value) in left.items():
            other = right[name][1]
            if field.type == field.TYPE_MESSAGE:
                a,b = (value,other) if field.is_repeated else ([value],[other])
                if len(a)!=len(b) or not all(equivalent(x,y) for x,y in zip(a,b)): return False
            elif value != other: return False
        return True
    if not equivalent(model, sanitized):
        raise ValueError('Seed computation changed during metadata audit')
    return sanitized, dict(optional_metadata_fields_removed=removed, computation_unchanged=True,
        textual_metadata_privacy_checked=True, opaque_numerical_tensor_payloads_preserved=True)


def _destination(directory, original, output):
    directory,original,output = (Path(p).resolve() for p in (directory,original,output))
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in (directory,original)):
        raise ValueError('New seed output must not overlap input or original evidence')
    if directory.is_relative_to(original) or original.is_relative_to(directory):
        raise ValueError('Candidate and original evidence directories must be separate')
    return directory,original,output


def prepare_seed(directory, original, output, *, case_id, source):
    from .audio_generation_candidate import load_candidate_input
    if re.fullmatch('[0-9a-f]{40}', source or '') is None:
        raise ValueError('Immutable seed execution source is required')
    directory,original,output = _destination(directory, original, output)
    program, values, _, provenance = load_candidate_input(directory, original, case_id=case_id)
    derived, changes = rewrite_welford_seed(program)
    output.mkdir(parents=True)
    receipt = output/'audio-gpu-seed.json'
    report = dict(schema=1, scope='unqualified_saved_candidate_welford_gpu_preparation_seed_only',
        success=False, source=source, provenance=provenance, changes=changes,
        versions=dict(torch=str(torch.__version__),onnx=onnx.__version__),
        original_golden_changed=False, gpu_validated=False, generation_qualified=False,
        numeric_acceptance=False, model_execution_scope='CPU export tracing only; no original-reference acceptance')
    def save(): receipt.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    save()
    try:
        path = output/'audio.onnx'
        export_audio_native_onnx(derived, values, path)
        model = onnx.load_model(path, load_external_data=False)
        sanitized, privacy = audit_seed_metadata(model)
        onnx.checker.check_model(sanitized)
        if any(n.domain not in ('','ai.onnx') or n.op_type=='LayerNormalization' for n in sanitized.graph.node):
            raise ValueError('Seed retained an unqualified runtime operation')
        onnx.save_model(sanitized, path)
        bindings = _binding_digest(_tensor_bindings(program))
        if bindings != changes['source_tensor_binding_sha256']:
            raise ValueError('Source tensors changed while tracing the seed')
        with path.open('rb') as stream: digest = hashlib.file_digest(stream,'sha256').hexdigest()
        report.update(success=True, metadata_audit=privacy, source_tensor_hashes_preserved_after_export=True,
            onnx_sha256=digest, onnx_bytes=path.stat().st_size, node_count=len(sanitized.graph.node),
            initializer_count=len(sanitized.graph.initializer))
    except Exception as error:
        report['error_type'] = type(error).__name__
        save(); raise
    save(); return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--case',choices=('silence','padded_noise'),required=True)
    parser.add_argument('--fetch',action='store_true')
    args=parser.parse_args(argv)
    try:
        if re.fullmatch('[0-9a-f]{40}',args.source or '') is None:
            raise ValueError('Immutable execution source required')
        _destination(args.input,args.original,args.output)
        directory=args.input
        if args.fetch:
            from .audio_generation_candidate import fetch_candidate
            fetch_candidate(directory,args.original,case_id=args.case)
            directory=directory/'audio-runtime-reference'
        report=prepare_seed(directory,args.original,args.output,case_id=args.case,source=args.source)
        print(json.dumps({k:report[k] for k in ('success','onnx_sha256','gpu_validated','generation_qualified')}))
        return 0
    except Exception as error:
        print(json.dumps(dict(success=False,error_type=type(error).__name__,gpu_validated=False,generation_qualified=False)))
        return 1


if __name__=='__main__': raise SystemExit(main())
