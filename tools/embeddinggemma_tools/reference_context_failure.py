"""Persist a rejected numeric context; never turn it into an accepted reference."""

import hashlib
import json
import os
import platform
from pathlib import Path
import re

import numpy as np
import torch


def record_feature_drift(destination, modality, actual, expected, *, role, profile, case,
                         source_commit, input_source_commit, saved_encoder=None):
    destination = Path(destination)
    actual = actual.detach().cpu()
    expected = expected.detach().cpu()
    report = dict(success=False, scope="official_encoder_feature_context_failure_only", role=role,
                  profile=profile, case=case, source_commit=source_commit, input_source_commit=input_source_commit,
                  actual_shape=list(actual.shape), expected_shape=list(expected.shape),
                  actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype), expected_unchanged=True,
                  unity_validated=False, end_to_end_validated=False,
                  runtime=dict(torch_version=torch.__version__, torch_threads=torch.get_num_threads(),
                               torch_interop_threads=torch.get_num_interop_threads(),
                               cpu_capability=torch.backends.cpu.get_cpu_capability(),
                               platform=platform.platform(), processor=platform.processor(), torch_config=torch.__config__.show(),
                               env={name: os.environ.get(name) for name in
                                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "MKL_CBWR", "DNNL_MAX_CPU_ISA", "ATEN_CPU_CAPABILITY")}))
    report.update(feature_error_metrics(actual, expected))
    if saved_encoder is not None:
        try:
            report["saved_encoder_vs_expected"] = feature_error_metrics(saved_encoder(), expected)
            report["saved_encoder_thread_contexts"] = probe_cpu_threads(saved_encoder, (), expected)
        except Exception as error:
            # Keep the original official mismatch even if saved replay cannot run.
            report["saved_encoder_vs_expected"] = dict(error=f"{type(error).__name__}: {error}", validated=False)
    path = destination / f"{modality}-role-context-actual-fp32.bin"
    actual.float().numpy().astype("<f4").tofile(path)
    report["actual_file"] = dict(file=path.name, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (destination / f"{modality}-role-context-failure.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def feature_error_metrics(actual, expected):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    report = dict(actual_finite=bool(torch.isfinite(actual).all()), same_shape=actual.shape == expected.shape)
    if report["same_shape"] and report["actual_finite"] and torch.isfinite(expected).all():
        difference = (actual.double() - expected.double()).abs()
        allowed = 1e-5 + 1e-4 * expected.double().abs()
        index = int((difference / allowed).argmax())
        report.update(mismatched_elements=int((difference > allowed).sum()),
                      maximum_absolute_error=float(difference.max()), maximum_tolerance_ratio=float((difference / allowed).max()),
                      worst_index=index, worst_actual=float(actual.flatten()[index]), worst_expected=float(expected.flatten()[index]))
    return report


def probe_cpu_threads(program, inputs, expected, *, thread_counts=(1, 2, 4)):
    """Failure-only controlled replay; restore the caller's intraop setting."""
    if (not thread_counts or len(set(thread_counts)) != len(thread_counts)
            or any(type(value) is not int or value <= 0 for value in thread_counts)):
        raise ValueError("thread counts must be unique positive integers")
    initial = torch.get_num_threads()
    conditions = []
    try:
        for requested in thread_counts:
            row = dict(requested_threads=requested, executed=False)
            conditions.append(row)
            try:
                torch.set_num_threads(requested)
                row["actual_threads"] = torch.get_num_threads()
                with torch.no_grad():
                    row.update(feature_error_metrics(program(*inputs), expected))
                row["executed"] = True
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
    finally:
        torch.set_num_threads(initial)
    return dict(scope="controlled_cpu_thread_replay_only", original_threads=initial, conditions=conditions,
                full_encoder_validated=False, end_to_end_validated=False)


def run_saved_graph_prefix(program, inputs, node_name):
    """Run through one saved FX node only; do not execute the remaining encoder.

    Returns an independent finite FP32 CPU observation without editing the graph.
    This diagnostic is not a full-program or original-golden acceptance result.
    """
    if not isinstance(program, torch.fx.GraphModule) or not any(node.name == node_name for node in program.graph.nodes):
        raise ValueError('Prefix node is absent from the saved graph')
    target=next(node for node in program.graph.nodes if node.name==node_name)
    needed=set();pending=[target]
    while pending:
        node=pending.pop()
        if node in needed:continue
        needed.add(node);pending.extend(node.all_input_nodes)
    graph=torch.fx.Graph();mapping={}
    for node in program.graph.nodes:
        if node.op=='placeholder' or node in needed:
            mapping[node]=graph.node_copy(node,lambda previous:mapping[previous])
    graph.output(mapping[target])
    prefix=torch.fx.GraphModule(program,graph)

    class Complete(Exception):
        pass

    class Prefix(torch.fx.Interpreter):
        result = None

        def run_node(self, node):
            value = super().run_node(node)
            if node.name == node_name:
                self.result = value
                raise Complete()
            return value

    runner = Prefix(prefix)
    with torch.no_grad():
        try:
            runner.run(*inputs)
        except Complete:
            pass
        else:
            raise ValueError('Prefix node was not executed')
        result = runner.result
        if not isinstance(result, torch.Tensor) or result.dtype != torch.float32 or result.device.type != 'cpu':
            raise TypeError('Prefix observation must be an FP32 CPU tensor')
        if not torch.isfinite(result).all():
            raise ValueError('Prefix observation must be finite')
        return result.detach().clone()


def probe_saved_static_matmuls(program, destination, node_names, *, rows=2520, prefix_rows=16, seed=41015):
    """Probe original weight/layout and full native GEMM shape, saving row prefixes.

    Uses deterministic synthetic operands, not encoder activations or a new golden.
    This does not call the saved program's forward or qualify full GPU inference.
    """
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Matmul evidence destination already exists')
    if (not isinstance(program,torch.fx.GraphModule) or not node_names or len(set(node_names))!=len(node_names)
            or type(rows) is not int or rows<1 or type(prefix_rows) is not int or not 1<=prefix_rows<=rows
            or type(seed) is not int or seed<0):
        raise ValueError('Unique static matmul nodes and positive full/prefix row counts are required')
    nodes={node.name:node for node in program.graph.nodes};selected=[]
    for name in node_names:
        if not isinstance(name,str) or re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',name) is None or name not in nodes:
            raise ValueError('Matmul node is missing or invalid')
        node=nodes[name]
        if node.op!='call_function' or node.target!=torch.ops.aten.mm.default or len(node.args)!=2 or not isinstance(node.args[1],torch.fx.Node):
            raise ValueError('Expected a saved two-dimensional static-weight mm node')
        pending=[node.args[1]];needed=set()
        while pending:
            ancestor=pending.pop()
            if ancestor in needed:continue
            needed.add(ancestor);pending.extend(ancestor.all_input_nodes)
        if any(ancestor.op=='placeholder' for ancestor in needed):raise ValueError('Matmul weight must not depend on model input')
        selected.append((name,node.args[1].name))
    inputs=(None,)*sum(node.op=='placeholder' for node in program.graph.nodes)
    weights=[run_saved_graph_prefix(program,inputs,name) for _,name in selected]
    if any(weight.ndim!=2 or not weight.numel() or min(weight.shape)<1 for weight in weights):raise ValueError('Expected finite FP32 matrix weights')
    destination.mkdir(parents=True)
    def save(name,tensor):
        raw=tensor.contiguous().numpy().tobytes();path=destination/name;path.write_bytes(raw)
        return dict(file=name,sha256=hashlib.sha256(raw).hexdigest(),shape=list(tensor.shape),bytes=len(raw))
    matrices=[]
    with torch.no_grad():
        for index,((name,weight_node),weight) in enumerate(zip(selected,weights,strict=True)):
            generator=torch.Generator().manual_seed(seed+index)
            values=torch.randn((rows,weight.shape[0]),generator=generator,dtype=torch.float32)
            actual=torch.mm(values,weight)
            if not torch.isfinite(actual).all():raise ValueError('Native matmul probe produced nonfinite values')
            matrices.append(dict(node=name,weight_node=weight_node,seed=seed+index,
                native_input_shape=list(values.shape),native_output_shape=list(actual.shape),saved_input_shape=[prefix_rows,weight.shape[0]],
                weight_stride=list(weight.stride()),input_stride=list(values.stride()),
                input=save(name+'-input-fp32.bin',values[:prefix_rows]),weights=save(name+'-weights-fp32.bin',weight),
                expected=save(name+'-expected-fp32.bin',actual[:prefix_rows])))
    report=dict(scope='saved_static_matmul_full_native_shape_and_prefix_only',matrices=matrices,
        runtime=dict(torch_version=torch.__version__,cpu_capability=torch.backends.cpu.get_cpu_capability(),threads=torch.get_num_threads()),
        full_encoder_executed=False,learned_weights_changed=False,full_gpu_encoder_validated=False,end_to_end_validated=False)
    (destination/'matmul-reference.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report


def capture_saved_program_nodes(program, inputs, node_names, destination, expected, *, row_prefixes=None):
    """Observe the existing saved graph without re-exporting or editing its ops."""
    available = {node.name for node in program.graph.nodes if node.op == 'call_function'}
    if (not node_names or len(set(node_names)) != len(node_names)
            or any(not isinstance(name, str) or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) is None
                   or name not in available for name in node_names)):
        raise ValueError('Capture requires unique existing operation node names')
    row_prefixes = {} if row_prefixes is None else row_prefixes
    if (not isinstance(row_prefixes, dict) or any(name not in node_names or type(count) is not int or count < 1
                                               for name, count in row_prefixes.items())):
        raise ValueError('Capture row prefixes need selected nodes and positive integer counts')
    observed, source_shapes = {}, {}

    class Observer(torch.fx.Interpreter):
        def run_node(self, node):
            value = super().run_node(node)
            if node.name in node_names:
                if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device.type != 'cpu'):
                    raise ValueError('Captured node must be a finite CPU FP32 tensor')
                capture = value.detach()
                if node.name in row_prefixes:
                    if value.ndim < 2 or row_prefixes[node.name] > value.shape[-2]:
                        raise ValueError('Capture row prefix exceeds the source tensor rows')
                    source_shapes[node.name] = list(value.shape)
                    capture = capture.narrow(-2, 0, row_prefixes[node.name])
                if not torch.isfinite(capture).all():
                    raise ValueError('Captured node must be a finite CPU FP32 tensor')
                observed[node.name] = capture.clone()
            return value

    with torch.no_grad():
        normal = program(*inputs)
        actual = Observer(program).run(*inputs)
    # Interpreter returns the FX graph's tuple, before ExportedProgram's output
    # pytree hook. This diagnostic accepts one final tensor only.
    if isinstance(actual, tuple) and len(actual) == 1:
        actual = actual[0]
    if (not isinstance(normal, torch.Tensor) or not isinstance(actual, torch.Tensor)
            or normal.dtype != torch.float32 or actual.dtype != torch.float32
            or normal.shape != actual.shape or not torch.isfinite(actual).all()
            or not torch.equal(normal.contiguous().view(torch.int32), actual.contiguous().view(torch.int32))):
        raise ValueError('Observed final output differs from normal saved-program bits')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in node_names:
        value = observed[name]
        path = destination / f'{name}-fp32.bin'
        raw = value.numpy().astype('<f4').tobytes()
        path.write_bytes(raw)
        rows.append(dict(node=name, file=path.name, shape=list(value.shape), bytes=len(raw),
                         sha256=hashlib.sha256(raw).hexdigest()))
        if name in row_prefixes:
            rows[-1].update(source_shape=source_shapes[name], prefix_rows=row_prefixes[name])
    return dict(scope='saved_program_node_observation_only', nodes=rows,
                observed_final_matches_normal_bits=True,
                original_golden_metrics=feature_error_metrics(actual, expected),
                full_encoder_validated=False, end_to_end_validated=False)


def replay_saved_reference_context(directory, modality, *, source_commit, revision, case_id,
                                   capture_nodes=None, capture_destination=None, capture_row_prefixes=None):
    """Audit the saved graph and selected fixture; this is not full acceptance."""
    if modality not in ("audio", "image") or any(re.fullmatch(r"[0-9a-f]{40}", value) is None for value in (source_commit, revision)):
        raise ValueError("Replay needs a fixed modality, source and revision")
    root = Path(directory).resolve()
    prefix = "audio-encoder" if modality == "audio" else "vision"
    audit = json.loads((root / (prefix + "-validation.json")).read_text(encoding="utf-8"))
    if (audit.get("stage") != ("audio_encoder_projector_only" if modality == "audio" else "vision_encoder_projector_only")
            or not (audit.get("native_reference_complete") if modality == "audio" else audit.get("success"))):
        raise ValueError("Replay requires a completed native reference")
    def audited(name):
        if not isinstance(name, str) or name not in audit["files"]:
            raise ValueError("Replay file is absent from the audit")
        path = (root / name).resolve()
        entry = audit["files"][name]
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError("Replay file path/size mismatch")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise ValueError("Replay file hash mismatch")
        return path
    reference = json.loads(audited(prefix + "-reference.json").read_text(encoding="utf-8"))
    metadata = reference["metadata"]
    if metadata != audit["metadata"] or metadata.get("source_commit") != source_commit or metadata.get("model_revision") != revision:
        raise ValueError("Replay reference provenance mismatch")
    matches = [row for row in reference["cases"] if row["id"] == case_id]
    if len(matches) != 1:
        raise ValueError("Replay case is missing or ambiguous")
    row = matches[0]; spec = reference["profiles"][row["profile"]]
    input_name = row["input_features"] if modality == "audio" else row["pixels"]
    input_path = audited(input_name)
    graph_path = audited(spec.get("native_graph", spec.get("graph")))
    shape = spec["feature_shape"] if modality == "audio" else spec["pixel_shape"]
    raw = np.fromfile(input_path, dtype="<f4")
    if not shape or any(type(value) is not int or value <= 0 for value in shape) or raw.size != int(np.prod(shape)) or not np.isfinite(raw).all():
        raise ValueError("Replay input shape/finite mismatch")
    values = torch.from_numpy(raw.reshape(shape).copy())
    expected = torch.tensor(row["features"], dtype=torch.float32)
    if not expected.numel() or not torch.isfinite(expected).all():
        raise ValueError("Replay expected features must be finite and nonempty")
    before_load_threads = torch.get_num_threads()
    program = torch.export.load(graph_path).module()
    report = dict(scope="audited_saved_reference_context_probe_only", modality=modality, case=case_id,
                reference_source=source_commit, model_revision=revision, input_sha256=audit["files"][input_name]["sha256"],
                graph_sha256=audit["files"][graph_path.relative_to(root).as_posix()]["sha256"],
                threads_before_program_load=before_load_threads, threads_after_program_load=torch.get_num_threads(),
                runtime=dict(torch_version=torch.__version__, cpu_capability=torch.backends.cpu.get_cpu_capability(),
                             torch_interop_threads=torch.get_num_interop_threads(),
                             env={name: os.environ.get(name) for name in ("ATEN_CPU_CAPABILITY", "MKL_CBWR", "MKL_ENABLE_INSTRUCTIONS", "DNNL_MAX_CPU_ISA")}),
                thread_contexts=probe_cpu_threads(program, (values,), expected),
                model_weights_downloaded=False, expected_unchanged=True, full_encoder_validated=False, end_to_end_validated=False)
    if capture_nodes is not None:
        if capture_destination is None:
            raise ValueError('Node capture destination is required')
        report['saved_node_capture'] = capture_saved_program_nodes(
            program, (values,), capture_nodes, capture_destination, expected, row_prefixes=capture_row_prefixes)
    return report
