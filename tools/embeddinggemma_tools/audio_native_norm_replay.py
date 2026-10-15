"""Opt-in frozen graph normalization intervention; never a replacement golden."""

import argparse
import hashlib
import json
import operator
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import torch

from .export import TorchWelfordLayerNorm
from .reference_context_failure import feature_error_metrics

VARIANTS = ("native", "welford8", "refined_moment", "refined_mean_moment")
PRODUCER = "4edbe17a8105233c5aaa1514faa2862b08a4f7e0"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
REFERENCE_SHA = "ac23f39904b8149dd5ccb6485afcc93d96047f0977b5faf1d640c8bd018c13c9"
AUDIT_SHA = "68d29c96012ef0a44114f2d3edbbff8d14e116ac697abee1a36d0b212155cfa1"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_normalizations(program):
    selected = []
    for node in program.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.native_layer_norm.default:
            continue
        if (len(node.args) != 5 or node.kwargs or tuple(node.args[1]) not in ((32,), (128,))
                or node.args[3] is not None or node.args[4] != 1e-6):
            raise ValueError("Unqualified normalization contract")
        if not node.users or any(user.op != "call_function" or user.target != operator.getitem
                                 or len(user.args) != 2 or user.args[0] is not node
                                 or user.args[1] not in (0, 1, 2)
                                 or (user.args[1] != 0 and user.users) for user in node.users):
            raise ValueError("Normalization statistic outputs must be unused")
        selected.append(node)
    if not selected:
        raise ValueError("No audited native normalization nodes")
    return selected


def candidate_output(values, shape, weight, bias, epsilon, variant):
    if (weight.dtype != torch.float32 or weight.device.type != "cpu" or tuple(weight.shape) != tuple(shape)
            or not torch.isfinite(weight).all() or values.dtype != torch.float32 or values.device.type != "cpu"
            or values.shape[-1] != shape[0] or not torch.isfinite(values).all()):
        raise ValueError("Diagnostic normalization requires finite original CPU FP32 values and scale")
    source = SimpleNamespace(normalized_shape=tuple(shape), weight=weight, bias=bias, eps=epsilon)
    norm = TorchWelfordLayerNorm(source, refine_inverse=variant != "welford8",
                               fuse_scalar_moment=variant != "welford8",
                               fuse_scalar_mean=variant == "refined_mean_moment")
    # Only the first tuple member has users; rejecting live statistics is mandatory.
    return norm(values), None, None


def replay_norm_variants(program, values, expected, output, *, provenance=None):
    output = Path(output)
    receipt = output / "audio-native-norm-replay.json"
    if receipt.exists() or any((output / (variant + "-final-fp32.bin")).exists() for variant in VARIANTS):
        raise ValueError("Refusing to overwrite previous diagnostic evidence")
    selected = checked_normalizations(program)
    if any(value.dtype != torch.float32 or value.device.type != "cpu" or not value.numel()
           or not torch.isfinite(value).all() for value in (values, expected)):
        raise ValueError("Replay requires finite nonempty CPU FP32 input and expected features")
    output.mkdir(parents=True, exist_ok=True)
    report = dict(scope="frozen_silence_native_normalization_intervention_only", complete=False,
                  expected_unchanged=True, full_encoder_validated=False, gpu_validated=False,
                  end_to_end_validated=False, provenance=provenance,
                  normalizations=[dict(node=node.name, width=node.args[1][0]) for node in selected], cases=[])

    def save():
        receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def tensor(value):
        if isinstance(value, tuple) and len(value) == 1:
            value = value[0]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.shape != expected.shape
                or value.device.type != "cpu" or not torch.isfinite(value).all()):
            raise ValueError("Replay final tensor shape, dtype or finite contract differs")
        return value.detach().contiguous()

    save()
    with torch.no_grad():
        native = tensor(program(values))
        for variant in VARIANTS:
            observations = []

            class Replay(torch.fx.Interpreter):
                def run_node(self, node):
                    if node in selected:
                        args, kwargs = self.fetch_args_kwargs_from_env(node)
                        result = (super().run_node(node) if variant == "native"
                                  else candidate_output(*args, variant))
                        raw = result[0].detach().contiguous().numpy().astype("<f4").tobytes()
                        observations.append(dict(node=node.name, shape=list(result[0].shape),
                                                 output_sha256=hashlib.sha256(raw).hexdigest(),
                                                 scale_sha256=hashlib.sha256(args[2].detach().numpy().astype("<f4").tobytes()).hexdigest()))
                        return result
                    return super().run_node(node)

            actual = tensor(Replay(program).run(values))
            same_native = torch.equal(actual.view(torch.int32), native.view(torch.int32))
            if variant == "native" and not same_native:
                raise ValueError("Native Interpreter control differs from saved-program bits")
            metrics = feature_error_metrics(actual, expected)
            metrics["exact_difference_count"] = int((actual.view(torch.int32) != expected.contiguous().view(torch.int32)).sum())
            path = output / (variant + "-final-fp32.bin")
            actual.numpy().astype("<f4").tofile(path)
            report["cases"].append(dict(variant=variant, native_program_bits_equal=same_native,
                                        metrics=metrics, normalizations=observations,
                                        final_file=path.name, final_sha256=digest(path)))
            save()
    report["complete"] = True
    save()
    return report


def load_frozen_input(directory, *, case_id='silence'):
    root = Path(directory).resolve()
    for name, sha in (("audio-encoder-reference.json", REFERENCE_SHA), ("audio-encoder-validation.json", AUDIT_SHA)):
        if not (root / name).is_file() or digest(root / name) != sha:
            raise ValueError("Original frozen reference/audit hash mismatch")
    reference = json.loads((root / "audio-encoder-reference.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "audio-encoder-validation.json").read_text(encoding="utf-8"))
    metadata = reference["metadata"]
    if (metadata != audit["metadata"] or metadata["source_commit"] != PRODUCER or metadata["model_revision"] != REVISION
            or metadata["feature_atol"] != 1e-5 or metadata["feature_rtol"] != 1e-4 or not audit["native_reference_complete"]
            or str(torch.__version__).split("+")[0] != metadata["versions"]["torch"]):
        raise ValueError("Original frozen audio producer, tolerance or torch context differs")
    cases = [row for row in reference['cases'] if row['id'] == case_id]
    if len(cases) != 1:
        raise ValueError('Original frozen case must identify one pinned input without fallback')
    case = cases[0]
    spec = reference["profiles"][case["profile"]]
    paths = []
    for name in (case["input_features"], spec["native_graph"]):
        path = (root / name).resolve()
        entry = audit["files"][name]
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError("Original frozen input/graph path, size or hash mismatch")
        paths.append(path)
    values = torch.from_numpy(np.fromfile(paths[0], dtype="<f4").reshape(spec["feature_shape"]).copy())
    expected = torch.tensor(case["features"], dtype=torch.float32)
    program = torch.export.load(paths[1]).module()
    if sorted(node.args[1][0] for node in checked_normalizations(program)) != [32, 128]:
        raise ValueError("Original frozen graph requires exactly the two pinned normalization widths")
    return program, values, expected, dict(reference_source=PRODUCER, model_revision=REVISION,
        case_id=case_id, profile=case['profile'],
        reference_sha256=REFERENCE_SHA, audit_sha256=AUDIT_SHA,
        graph_sha256=audit["files"][spec["native_graph"]]["sha256"],
        input_sha256=audit["files"][case["input_features"]]["sha256"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = os.environ.get("GITHUB_SHA", "")
    if re.fullmatch("[0-9a-f]{40}", source) is None:
        raise ValueError("Replay job needs immutable source provenance")
    program, values, expected, provenance = load_frozen_input(args.input)
    provenance.update(job_source=source, torch_version=str(torch.__version__), threads=torch.get_num_threads(),
                      cpu_capability=torch.backends.cpu.get_cpu_capability(),
                      env={name: os.environ.get(name) for name in ("ATEN_CPU_CAPABILITY", "MKL_CBWR", "DNNL_MAX_CPU_ISA")})
    report = replay_norm_variants(program, values, expected, args.output, provenance=provenance)
    # Diagnostic completion only. Never expose a green full-encoder qualification.
    print(json.dumps(dict(complete=report["complete"], cases=report["cases"], full_encoder_validated=False)))


if __name__ == "__main__":
    main()
