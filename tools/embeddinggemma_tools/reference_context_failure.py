"""Persist a rejected numeric context; never turn it into an accepted reference."""

import hashlib
import json
import os
import platform
from pathlib import Path

import torch


def record_feature_drift(destination, modality, actual, expected, *, role, profile, case,
                         source_commit, input_source_commit):
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
    report["actual_finite"] = bool(torch.isfinite(actual).all())
    if actual.shape == expected.shape and report["actual_finite"] and torch.isfinite(expected).all():
        difference = (actual.double() - expected.double()).abs()
        allowed = 1e-5 + 1e-4 * expected.double().abs()
        index = int((difference / allowed).argmax())
        report.update(mismatched_elements=int((difference > allowed).sum()),
                      maximum_absolute_error=float(difference.max()), maximum_tolerance_ratio=float((difference / allowed).max()),
                      worst_index=index, worst_actual=float(actual.flatten()[index]), worst_expected=float(expected.flatten()[index]))
    path = destination / f"{modality}-role-context-actual-fp32.bin"
    actual.float().numpy().astype("<f4").tofile(path)
    report["actual_file"] = dict(file=path.name, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (destination / f"{modality}-role-context-failure.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
