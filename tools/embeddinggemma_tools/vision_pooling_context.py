"""Compare official/fixed pooling on an observed original encoder output only."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPooler

from .reference_context_failure import feature_error_metrics


def pooling_layouts(hidden, positions, pooling_weights):
    if (hidden.dtype != torch.float32 or hidden.device.type != 'cpu' or hidden.ndim != 3
            or hidden.shape[0] != 1 or not hidden.numel() or not torch.isfinite(hidden).all()
            or positions.dtype != torch.int64 or tuple(positions.shape) != (*hidden.shape[:2], 2)
            or pooling_weights.dtype != torch.float32 or pooling_weights.ndim != 3
            or pooling_weights.shape[0] != 1 or pooling_weights.shape[-1] != hidden.shape[1]
            or not torch.isfinite(pooling_weights).all() or hidden.shape[1] % 9):
        raise ValueError('Expected finite CPU FP32 original vision tensors and fixed kernel-three geometry')
    padding = (positions == -1).all(-1)
    masked = hidden.masked_fill(padding.unsqueeze(-1), 0.)
    length = hidden.shape[1] // 9
    clamped = positions.clamp(min=0)
    width = clamped[..., 0].max(-1, keepdim=True)[0] + 1
    cells = torch.div(clamped, 3, rounding_mode='floor')
    indices = cells[..., 0] + (width // 3) * cells[..., 1]
    official_weights = (torch.nn.functional.one_hot(indices, length).float() / 9).transpose(1, 2)
    official, mask = Gemma4VisionPooler(SimpleNamespace(hidden_size=hidden.shape[-1]))(
        hidden, positions, padding, output_length=length)
    fixed = (pooling_weights @ masked)[0] * hidden.shape[-1]**.5
    official = official[mask]
    if fixed.shape != official.shape:
        raise ValueError('Official mask and fixed pooling feature count differ')
    return dict(saved_contiguous=fixed, official_transposed=official), dict(
        saved_pool_shape=list(pooling_weights.shape), saved_pool_stride=list(pooling_weights.stride()),
        official_pool_shape=list(official_weights.shape), official_pool_stride=list(official_weights.stride()))


def replay_pooling_context(directory, captures):
    root, captures = Path(directory), Path(captures)
    for name, wanted in (
        ('vision-reference.json', '3dfccd1a487ce04df3f8d640483ca045f7a7a56231bc1096ad0f845c923a29c6'),
        ('vision-validation.json', 'ab1aa775b553f33a5e2892abed2cdcef7768e35fbad6fa5af9ec3286a7ba90ba'),
        ('square/vision.pt2', 'f9c5175a1d75289ce814a9651f496f9bdae5e873ac66199a5c3d03e0172c16cc')):
        with (root/name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != wanted:
                raise ValueError('Original pooling replay provenance mismatch')
    observed = json.loads((captures/'original-image-native.json').read_text())
    if (observed['graph_sha256'] != 'f9c5175a1d75289ce814a9651f496f9bdae5e873ac66199a5c3d03e0172c16cc'
            or not observed['saved_node_capture']['observed_final_matches_normal_bits']):
        raise ValueError('Original pooling replay needs verified saved-graph observation')

    def tensor(name):
        row, = [row for row in observed['saved_node_capture']['nodes'] if row['node'] == name]
        if row['file'] != name+'-fp32.bin' or row['bytes'] > 32*1024*1024:
            raise ValueError('Captured pooling node path or size mismatch')
        path = captures/'captures'/row['file']
        raw = path.read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Captured pooling node hash mismatch')
        result = torch.from_numpy(np.frombuffer(raw, '<f4').copy().reshape(row['shape']))
        if not torch.isfinite(result).all():
            raise ValueError('Captured pooling node is nonfinite')
        return result

    program = torch.export.load(root/'square/vision.pt2').module()
    hidden, captured_features, captured_normalized, captured_final = map(tensor, ('add_225', 'mul_372', 'mul_373', 'mm_113'))
    weights = program.get_parameter('projection.embedding_projection.weight')
    original = json.loads((root/'vision-reference.json').read_text())
    case, = [row for row in original['cases'] if row['id'] == 'square-0']
    golden = torch.tensor(case['features'], dtype=torch.float32)

    def metrics(actual, expected):
        return dict(**feature_error_metrics(actual, expected),
                    bit_differences=int(torch.count_nonzero(actual.contiguous().view(torch.int32) != expected.contiguous().view(torch.int32))))

    initial = torch.get_num_threads()
    try:
        torch.set_num_threads(observed['thread_contexts']['original_threads'])
        with torch.no_grad():
            layouts, metadata = pooling_layouts(hidden, program.get_buffer('positions'), program.get_buffer('pooling_weights'))
            rows = {}
            for name, features in layouts.items():
                normalized = features * torch.pow(features.square().mean(-1, keepdim=True)+1e-6, -.5)
                final = normalized @ weights.T
                rows[name] = dict(features_vs_saved=metrics(features, captured_features),
                    normalized_vs_saved=metrics(normalized, captured_normalized), final_vs_saved=metrics(final, captured_final),
                    final_vs_original_golden=metrics(final, golden))
            controls = all(rows['saved_contiguous'][key]['bit_differences'] == 0
                           for key in ('features_vs_saved', 'normalized_vs_saved', 'final_vs_saved'))
            return dict(scope='original_captured_encoder_output_pool_layout_comparison_only',
                controls_bit_exact=controls, layouts=metadata, comparisons=rows,
                captured_source=observed['job_source'], graph_sha256=observed['graph_sha256'],
                runtime=dict(torch_version=torch.__version__, threads=torch.get_num_threads(),
                             cpu_capability=torch.backends.cpu.get_cpu_capability()),
                original_golden_unchanged=True, encoder_inference_executed=False, gpu_validated=False,
                full_encoder_validated=False, end_to_end_validated=False)
    finally:
        torch.set_num_threads(initial)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--captures', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = replay_pooling_context(args.input, args.captures)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report, allow_nan=False))
    if not report['controls_bit_exact']:
        raise SystemExit('Saved pooling controls did not reproduce observed original bits')
