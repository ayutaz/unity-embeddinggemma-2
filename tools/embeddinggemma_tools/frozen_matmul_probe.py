"""Audited single real MatMul ONNX lowering diagnosis, not encoder acceptance."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch

from .audio_primitive_replay import onnx_run
from .frozen_audio_prefix_onnx import exact_difference, file_digest
from .export import export_audio_native_onnx
from .reference_context_failure import feature_error_metrics
from .static_block192_onnx import static_block192_export
from .streaming_reference import run_flat_reference

CAPTURE_SOURCE = "11db7b17b640ea85ab6b04bef066a737d3e5ee41"
CAPTURE_REPORT_SHA = "f27073b3503e82a4b963a64b3ff2c1e2f782d891a8408eb10cb22413a3aa5fdf"


def load_capture(directory):
    root = Path(directory).resolve()
    report = root / "audio-primitive-replay.json"
    if not report.is_file() or file_digest(report) != CAPTURE_REPORT_SHA:
        raise ValueError("Pinned primitive capture report hash differs")
    data = json.loads(report.read_text(encoding="utf-8"))
    if (not all(data.get(key) is True for key in ("complete", "original_context_qualified",
            "native_interpreter_control_bits_equal", "coefficients_unchanged"))
            or data["provenance"]["job_source"] != CAPTURE_SOURCE
            or data["native_segment_original_metrics"]["exact_difference_count"] != 0):
        raise ValueError("Primitive source context is not original-output qualified")
    capture = data["worst_operator_captures"]["MatMul"]
    loaded, digests = [], {}
    for name in ("input0", "input1", "native"):
        matches = [spec for spec in capture["files"] if spec["file"].endswith(f"-{name}-fp32.bin")]
        if len(matches) != 1:
            raise ValueError("Frozen MatMul capture is ambiguous")
        spec = matches[0]
        path = (root / spec["file"]).resolve()
        shape, strides = spec["shape"], spec["native_strides"]
        if (not path.is_relative_to(root) or len(shape) != 2 or any(type(d) is not int or d < 1 for d in shape)
                or not path.is_file() or path.stat().st_size != int(np.prod(shape)) * 4
                or path.stat().st_size > 64 * 1024 * 1024 or file_digest(path) != spec["sha256"]):
            raise ValueError("Frozen MatMul tensor path, shape, size or hash differs")
        values = np.fromfile(path, dtype="<f4").reshape(shape)
        if strides == [shape[1], 1]:
            values = np.ascontiguousarray(values)
        elif strides == [1, shape[0]]:
            values = np.asfortranarray(values)
        else:
            raise ValueError("Frozen MatMul layout is not audited dense row/column major")
        if not np.isfinite(values).all():
            raise ValueError("Frozen MatMul values are not finite")
        loaded.append(torch.from_numpy(values))
        digests[name] = spec["sha256"]
    a, b, expected = loaded
    if a.shape[1] != b.shape[0] or tuple(expected.shape) != (a.shape[0], b.shape[1]):
        raise ValueError("Frozen matrix shapes differ")
    return a, b, expected, dict(capture_source=CAPTURE_SOURCE, report_sha256=CAPTURE_REPORT_SHA,
                                node=capture["node"], tensor_sha256=digests)


def metrics(actual, expected):
    result = feature_error_metrics(actual, expected)
    result["exact_difference_count"] = exact_difference(actual, expected)
    return result


def probe_capture(directory, output, *, provenance=None):
    output = Path(output)
    receipt = output / "frozen-matmul-probe.json"
    if receipt.exists() or (output / "static-block192.onnx").exists():
        raise ValueError("Refusing to overwrite frozen MatMul probe evidence")
    a, b, expected, source = load_capture(directory)
    before = hashlib.sha256(b.contiguous().numpy().tobytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    report = dict(scope="single_qualified_replay_derived_real_matmul_onnx_only", complete=False,
        provenance=provenance, capture_provenance=source,
        native_execution="original_MatMul_passthrough_not_fixed_order_execution",
        full_encoder_validated=False, gpu_validated=False, end_to_end_validated=False)

    def save():
        receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", b)
        def forward(self, values):
            return static_block192_export(values, self.weight)

    with torch.no_grad():
        report["native_passthrough_metrics"] = metrics(Module()(a), expected)
        baseline = onnx_run("MatMul", [a.numpy(), b.numpy()], {})
        report["standard_onnx_metrics"] = metrics(baseline, expected)
        save()
        path = output / "static-block192.onnx"
        start = time.monotonic()
        export_audio_native_onnx(Module(), a, path)
        report["export_seconds_observed"] = time.monotonic() - start
        report["graph_sha256"] = file_digest(path)
        report["graph_bytes"] = path.stat().st_size
        model = onnx.load(path)
        report["graph_nodes"] = len(model.graph.node)
        save()
        start = time.monotonic()
        actual = torch.from_numpy(run_flat_reference(ReferenceEvaluator(model), {"input": a.numpy()})[0])
        report["evaluation_seconds_observed"] = time.monotonic() - start
        report["onnx_metrics"] = metrics(actual, expected)
        capture = output / "static-block192-fp32.bin"
        actual.numpy().astype("<f4").tofile(capture)
        report.update(onnx_capture=capture.name, onnx_capture_sha256=file_digest(capture),
                      weights_unchanged=before == hashlib.sha256(b.contiguous().numpy().tobytes()).hexdigest())
        if not report["weights_unchanged"]:
            save()
            raise ValueError("Original frozen coefficient bytes changed")
    report["complete"] = True
    save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = os.environ.get("GITHUB_SHA", "")
    if re.fullmatch("[0-9a-f]{40}", source) is None:
        raise ValueError("MatMul probe needs immutable job provenance")
    torch.set_num_threads(2)
    report = probe_capture(args.input, args.output, provenance=dict(job_source=source,
        torch_version=str(torch.__version__), numpy_version=np.__version__,
        cpu_capability=torch.backends.cpu.get_cpu_capability(), threads=torch.get_num_threads()))
    print(json.dumps(report))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
