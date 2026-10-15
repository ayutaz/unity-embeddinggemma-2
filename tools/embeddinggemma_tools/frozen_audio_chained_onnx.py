"""Bounded FX partitions with actual ONNX carry, never frozen stage resets."""
import argparse
import gc
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
from onnx.reference import ReferenceEvaluator

from .frozen_audio_prefix_onnx import (FrozenStageReference, boundary_for_stage, exact_difference, file_digest,
    load_frozen_input, load_frozen_stage_reference, tensor, weight_digests)
from .frozen_audio_segment_onnx import PRECISE, PRECISE_POSITIONS, VARIANTS, SegmentExport, extract_segment
from .export import export_audio_native_onnx
from .reference_context_failure import feature_error_metrics
from .streaming_reference import run_flat_reference


def _raw_tensor(value):
    # Preserve physical native layout across the control's cuts. tensor() is
    # intentionally contiguous for captures and is not used for this carry.
    if isinstance(value, tuple) and len(value) == 1:
        value = value[0]
    tensor(value)  # Validate dtype/device/finite without replacing the value.
    return value.detach()


def probe_chained_segments(program, frozen_input, frozen_output, output, *, boundaries,
                           variants=('native',), provenance=None):
    if not all(isinstance(x, FrozenStageReference) for x in (frozen_input, frozen_output)):
        raise ValueError('Chained diagnosis requires audited frozen endpoints')
    if (not isinstance(boundaries, (tuple, list)) or len(boundaries) < 3
            or len(set(boundaries)) != len(boundaries)):
        raise ValueError('Chained diagnosis needs distinct ordered boundaries')
    if (not isinstance(variants, (tuple, list)) or not variants or len(set(variants)) != len(variants)
            or any(x not in VARIANTS for x in variants)):
        raise ValueError('Chained variants must be unique supported choices')
    output = Path(output)
    receipt = output / 'frozen-audio-chained-onnx.json'
    if receipt.exists() or any(output.glob('*.onnx')) or any(output.glob('*.bin')):
        raise ValueError('Refusing to overwrite chained diagnostic evidence')
    pieces = [extract_segment(program, a, b) for a, b in zip(boundaries, boundaries[1:])]
    whole = extract_segment(program, boundaries[0], boundaries[-1])
    before = weight_digests(whole)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = dict(scope='connected_original_frozen_endpoint_onnx_partitions_only', complete=False,
        provenance=provenance, boundaries=list(boundaries), input_provenance=frozen_input.provenance,
        target_provenance=frozen_output.provenance, original_weights=before, cases=[],
        whole_encoder_validated=False, gpu_validated=False, end_to_end_validated=False)

    def save():
        receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    def phase(name, **extra):
        report['phase'] = dict(name=name, elapsed_seconds=time.perf_counter()-started, **extra)
        save()
        print(json.dumps(report['phase']), flush=True)

    def capture(name, values):
        path = output / name
        values.astype('<f4').tofile(path)
        return file_digest(path)

    def metrics(actual, expected):
        result = feature_error_metrics(actual, expected)
        result['exact_difference_count'] = exact_difference(actual, expected)
        return result

    save()
    try:
        with torch.no_grad():
            initial, expected = _raw_tensor(frozen_input.expected), tensor(frozen_output.expected)
            native = _raw_tensor(whole(initial))
            report['input_sha256'] = capture('frozen-input-fp32.bin', initial.numpy())
            report['reference_sha256'] = capture('reference-output-fp32.bin', expected.numpy())
            report['native_sha256'] = capture('native-output-fp32.bin', native.numpy())
            report['native_whole_metrics'] = metrics(native, expected)
            report['context_qualified'] = exact_difference(native, expected) == 0
            native_inputs, native_outputs, carried = [], [], initial
            for piece in pieces:
                native_inputs.append(carried)
                carried = _raw_tensor(piece(carried))
                native_outputs.append(carried)
            report['native_chain_control_bits_equal'] = exact_difference(carried, native) == 0
            if not report['native_chain_control_bits_equal']:
                raise ValueError('Native partition carry changed original control bits')
            save()
            for variant in variants:
                row = dict(variant=variant, parts=[], actual_carry='previous_onnx_output_without_frozen_stage_reset')
                report['cases'].append(row)
                actual = initial.numpy()
                for index, (piece, example, native_output) in enumerate(zip(pieces, native_inputs, native_outputs, strict=True)):
                    phase('exporting', variant=variant, part=index)
                    input_sha = capture(f'{variant}-part{index}-input-fp32.bin', actual)
                    wrapper = SegmentExport(piece, variant)
                    graph = output / f'{variant}-part{index}.onnx'
                    export_audio_native_onnx(wrapper, example, graph)
                    phase('evaluating', variant=variant, part=index)
                    evaluator = ReferenceEvaluator(str(graph))
                    values = run_flat_reference(evaluator, {'input': actual})[0]
                    if (values.dtype != np.float32 or tuple(values.shape) != tuple(native_output.shape)
                            or not np.isfinite(values).all()):
                        raise ValueError('Chained ONNX part output shape/dtype/finite contract differs')
                    row['parts'].append(dict(index=index, graph=graph.name, graph_sha256=file_digest(graph),
                        from_boundary=boundaries[index], to_boundary=boundaries[index + 1],
                        input_capture=f'{variant}-part{index}-input-fp32.bin', output_capture=f'{variant}-part{index}-output-fp32.bin',
                        input_sha256=input_sha, output_sha256=capture(f'{variant}-part{index}-output-fp32.bin', values),
                        native_input_strides=list(example.stride()), actual_output_strides=list(values.strides),
                        linear_replacements=len(wrapper.linear_nodes), attention_bmm_replacements=len(wrapper.bmm_nodes-wrapper.parameter_bmm_nodes),
                        relative_position_replacements=len(wrapper.position_nodes),
                        parameter_bmm_replacements=len(wrapper.parameter_bmm_nodes),
                        rms_mean_replacements=len(wrapper.rms_means), rms_inverse_replacements=len(wrapper.rms_inverses),
                        silu_replacements=len(wrapper.silu_nodes), softmax_replacements=len(wrapper.softmax_nodes),
                        sigmoid_replacements=len(wrapper.sigmoid_nodes), depthwise_replacements=len(wrapper.convolution_nodes)))
                    actual = values  # No file reload, native replacement, or frozen intermediate.
                    save()
                    del evaluator, wrapper
                    gc.collect()
                row['metrics'] = metrics(torch.from_numpy(actual), expected)
                if weight_digests(whole) != before:
                    raise ValueError('Original coefficients changed during chained diagnosis')
                row['coefficients_unchanged'] = True
                save()
        report['complete'] = True
        phase('complete')
    except Exception as error:
        report['diagnostic_error'] = dict(type=type(error).__name__, message=str(error))
        save()
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frozen-stage-source', type=Path, required=True)
    parser.add_argument('--variant', choices=('native', 'ordered_static_linear_all_bmm', PRECISE, PRECISE_POSITIONS), default='ordered_static_linear_all_bmm')
    args = parser.parse_args(argv)
    source = os.environ.get('GITHUB_SHA', '')
    if re.fullmatch('[0-9a-f]{40}', source) is None:
        raise ValueError('Chained diagnosis requires immutable job provenance')
    program, _, _, provenance = load_frozen_input(args.input)
    stages = ('subsample', 'layer_2', 'layer_5', 'layer_8', 'projected_features')
    boundaries = [boundary_for_stage(program, stage) for stage in stages]
    provenance.update(job_source=source, torch_version=str(torch.__version__), threads=torch.get_num_threads(),
                      cpu_capability=torch.backends.cpu.get_cpu_capability(), stages=stages)
    if args.variant != 'native':
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            wrapper = SegmentExport(extract_segment(program, start, end), args.variant)
            count = (29 if index == 3 else 27) + (3 if args.variant == PRECISE_POSITIONS else 0)
            if (len(wrapper.linear_nodes) != count
                    or len(wrapper.bmm_nodes-wrapper.parameter_bmm_nodes) != 9
                    or len(wrapper.parameter_bmm_nodes) != 3):
                raise ValueError('Pinned chained static-linear/BMM coverage differs')
            if args.variant in (PRECISE, PRECISE_POSITIONS):
                rms = 28 if index == 3 else 27
                if (len(wrapper.rms_means) != rms or len(wrapper.rms_inverses) != rms
                        or len(wrapper.silu_nodes) != 9
                        or any(len(nodes) != 3 for nodes in (wrapper.softmax_nodes, wrapper.sigmoid_nodes, wrapper.convolution_nodes))):
                    raise ValueError('Pinned chained RMS/SiLU/audio leaf coverage differs')
    report = probe_chained_segments(program,
        load_frozen_stage_reference(args.frozen_stage_source, stages[0]),
        load_frozen_stage_reference(args.frozen_stage_source, stages[-1]), args.output,
        boundaries=boundaries, variants=('native', args.variant) if args.variant != 'native' else ('native',),
        provenance=provenance)
    print(json.dumps(dict(complete=report['complete'], context_qualified=report['context_qualified'],
        cases=[dict(variant=row['variant'], metrics=row['metrics'], parts=len(row['parts'])) for row in report['cases']],
        whole_encoder_validated=False, gpu_validated=False)))
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
