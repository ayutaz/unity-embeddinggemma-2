"""Independent native vision stage references and saved-graph localization."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import torch

from .export import VisionEncoderExport, export_vision_pt2
from .reference_environment import cpu_math_environment


class VisionEncoderStagesExport(VisionEncoderExport):
    """Diagnostic outputs only; preserves the ordinary final arithmetic."""

    def __init__(self, vision, projection, pixel_position_ids):
        super().__init__(vision, projection, pixel_position_ids)
        self.stage_names = ("patch_embeddings", "rotary_cos", "rotary_sin", *(
            f"layer_{index}" for index in range(vision.config.num_hidden_layers)),
            "vision_features", "pre_projection_norm", "projected_features")

    def forward(self, pixels):
        if not torch.compiler.is_compiling():
            self.validate_inputs(pixels)
        hidden = self.vision.patch_embedder(pixels, self.positions, self.padding)
        stages = [hidden]
        rotary = self.vision.encoder.rotary_emb(hidden, self.positions)
        stages.extend(rotary)
        for layer in self.vision.encoder.layers[:self.vision.config.num_hidden_layers]:
            hidden = layer(hidden, attention_mask=self.attention, position_embeddings=rotary,
                           position_ids=self.positions)
            stages.append(hidden)
        hidden = hidden.masked_fill(self.padding.unsqueeze(-1), 0.0)
        hidden = (self.pooling_weights @ hidden.float())[0] * self.vision.config.hidden_size**0.5
        if self.vision.config.standardize:
            hidden = (hidden - self.vision.std_bias.float()) * self.vision.std_scale.float()
        stages.append(hidden)
        hidden = self.projection.embedding_pre_projection_norm(hidden)
        stages.append(hidden)
        stages.append(self.projection.embedding_projection(hidden))
        return tuple(stages)


def capture_official_vision_stages(vision, projection, pixels, coords):
    """Hooks on the official forward; removes only handles owned by this call."""
    modules = [vision.patch_embedder, vision.encoder.rotary_emb, *vision.encoder.layers[:vision.config.num_hidden_layers], vision,
               projection.embedding_pre_projection_norm, projection]
    values, handles = [], []
    for module in modules:
        def capture(owner, args, output):
            if owner is vision.encoder.rotary_emb:
                values.extend(value.detach().clone() for value in output)
                return
            value = output.last_hidden_state if owner is vision else output
            values.append(value.detach().clone())
        handles.append(module.register_forward_hook(capture))
    try:
        with torch.no_grad():
            projection(vision(pixel_values=pixels, pixel_position_ids=coords).last_hidden_state)
    finally:
        for handle in handles:
            handle.remove()
    return tuple(values)


def compare_vision_stages(names, expected, actual):
    if not names or len(names) != len(expected) or len(names) != len(actual):
        raise ValueError("vision stage count mismatch")
    rows = []
    for name, native, saved in zip(names, expected, actual, strict=True):
        if (native.shape != saved.shape or native.dtype != torch.float32 or saved.dtype != torch.float32
                or not torch.isfinite(native).all() or not torch.isfinite(saved).all()):
            raise ValueError(f"vision stage shape, dtype or nonfinite mismatch: {name}")
        difference = (saved.double() - native.double()).abs()
        bound = 1e-5 + 1e-4 * native.double().abs()
        mismatches = int((difference > bound).sum())
        rows.append(dict(name=name, shape=list(native.shape), success=mismatches == 0,
                         maximum_absolute_error=difference.max().item(),
                         maximum_tolerance_ratio=(difference / bound).max().item(),
                         mismatched_elements=mismatches))
    return dict(success=all(row["success"] for row in rows), feature_atol=1e-5, feature_rtol=1e-4,
                first_nonzero_stage=next((row["name"] for row in rows if row["maximum_absolute_error"] > 0), None),
                first_divergent_stage=next((row["name"] for row in rows if not row["success"]), None), stages=rows)


def prepare_vision_diagnostics(vision, projection, pixels, coords, destination, original_graph, *, source_commit, revision,
                               additional_pixels=()):
    """One profile, shared graph, independent native references for every input."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    receipt = destination / "vision-diagnostics.json"
    receipt.unlink(missing_ok=True)
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None
           for value in (source_commit, revision)):
        raise ValueError("vision diagnostics require immutable source and model SHAs")
    wrapper = VisionEncoderStagesExport(vision, projection, coords)
    inputs = (pixels, *additional_pixels)
    for value in inputs:
        wrapper.validate_inputs(value)
    graph = destination / "vision-stages.pt2"
    export_vision_pt2(wrapper, pixels, graph)
    files, cases = {}, []
    def digest(path):
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    def record(path):
        files[path.name] = dict(bytes=path.stat().st_size, sha256=digest(path))
    def write(name, value, dtype):
        path = destination / name
        path.write_bytes(value.detach().cpu().numpy().astype(dtype).tobytes())
        record(path)
    record(graph)
    write("positions-int32.bin", coords, "<i4")
    exported = torch.export.load(graph).module()
    original_exported = torch.export.load(original_graph).module()
    for case_index, value in enumerate(inputs):
        native = capture_official_vision_stages(vision, projection, value, coords)
        with torch.no_grad():
            actual = exported(value)
            original = original_exported(value)
        case = compare_vision_stages(wrapper.stage_names, native, actual)
        same_shape = original.shape == actual[-1].shape
        context = same_shape and torch.equal(original, actual[-1])
        case.update(id=f"case_{case_index}", success=case["success"] and context,
                    trace_matches_original_saved=context,
                    trace_vs_original_maximum_absolute_error=(original - actual[-1]).abs().max().item() if same_shape else None,
                    pixels_file="pixels-fp32.bin" if case_index == 0 else f"pixels-case-{case_index}-fp32.bin")
        write(case["pixels_file"], value, "<f4")
        for stage_index, (row, expected) in enumerate(zip(case["stages"], native, strict=True)):
            row["reference_file"] = f"native-stage-{stage_index}.bin" if case_index == 0 else f"native-case-{case_index}-stage-{stage_index}.bin"
            write(row["reference_file"], expected, "<f4")
        cases.append(case)
        del actual, original, native
    report = dict(cases[0])
    report.update(success=all(case["success"] for case in cases), stage="vision_encoder_diagnostics_only",
                  unity_validated=False, end_to_end_validated=False, localization_validated=False,
                  trace_matches_original_saved=all(case["trace_matches_original_saved"] for case in cases),
                  metadata=dict(source_commit=source_commit, model_revision=revision,
                      cpu_math_environment=cpu_math_environment(),
                      versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "torchvision")},
                      pixel_shape=list(pixels.shape), position_shape=list(coords.shape), case_count=len(cases),
                      feature_count=wrapper.feature_count), cases=cases, files=files)
    report["original_graph"] = dict(bytes=Path(original_graph).stat().st_size, sha256=digest(original_graph))
    receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
