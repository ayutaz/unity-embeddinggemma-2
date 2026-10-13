"""Generate pinned reference data and validate the saved export before Unity import."""

import gc
import hashlib
import importlib.metadata
import json
import re
import shutil
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer

from .export import (AudioEncoderExport, AudioEncoderStagesExport, AudioFeedForwardStagesExport, AudioSubsampleStagesExport, MultimodalEmbeddingExport, VisionEncoderExport,
                     export_audio_pt2, export_audio_native_onnx, export_multimodal_pt2, export_pt2, export_vision_pt2)
from .model import load_audio_models, load_text_model, load_vision_models
from .reference import build_reference, load_cases, write_reference
from .search import generate_search_reference

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


def capture_official_audio_stages(audio, projection, features, frame_mask, *, detail_layer=None):
    """Capture native forward outputs, removing only this call's temporary hooks."""
    if detail_layer is not None and (type(detail_layer) is not int or not 0 <= detail_layer < audio.config.num_hidden_layers):
        raise ValueError("audio detail layer must be an existing integer layer index")
    values, handles = [], []
    modules = [(audio.subsample_conv_projection, True)]
    for index, layer in enumerate(audio.layers[:audio.config.num_hidden_layers]):
        if index == detail_layer:
            modules.extend((getattr(layer, part), part == "self_attn") for part in AudioEncoderStagesExport.detail_parts)
        modules.append((layer, False))
    modules.append((audio.output_proj, False))
    try:
        for module, tuple_output in modules:
            def capture(module, args, output, tuple_output=tuple_output):
                values.append((output[0] if tuple_output else output).detach().clone())
            handles.append(module.register_forward_hook(capture))
        with torch.no_grad():
            result = audio(input_features=features, attention_mask=frame_mask.bool())
            values.append(projection(result.last_hidden_state)[result.attention_mask].detach().clone())
    finally:
        for handle in handles:
            handle.remove()
    return tuple(values)


def compare_audio_stage_outputs(names, expected, actual):
    """Report each native-vs-saved stage under the unchanged encoder tolerance."""
    if not names or len(set(names)) != len(names) or len(names) != len(expected) or len(names) != len(actual):
        raise ValueError("audio stage names/count mismatch")
    rows, first = [], None
    for name, native, saved in zip(names, expected, actual, strict=True):
        if (native.dtype != torch.float32 or saved.dtype != torch.float32 or native.shape != saved.shape
                or native.numel() == 0 or not torch.isfinite(native).all() or not torch.isfinite(saved).all()):
            raise ValueError(f"audio stage shape/dtype/nonfinite mismatch: {name}")
        difference = (saved - native).abs()
        allowed = 1e-5 + 1e-4 * native.abs()
        mismatches = int((difference > allowed).sum())
        if mismatches and first is None:
            first = name
        rows.append(dict(name=name, shape=list(native.shape), success=mismatches == 0,
                         maximum_absolute_error=difference.max().item(),
                         maximum_tolerance_ratio=(difference / allowed).max().item(),
                         mismatched_elements=mismatches))
    return dict(first_divergent_stage=first, stages=rows)


def capture_official_audio_feed_forward(audio, features, frame_mask, *, detail_layer, part):
    """Use the native tower's actual selected input, not a synthetic replacement."""
    if (type(detail_layer) is not int or not 0 <= detail_layer < audio.config.num_hidden_layers
            or part not in ("feed_forward1", "feed_forward2")):
        raise ValueError("audio feed-forward selector must name an existing layer and feed-forward part")
    module = getattr(audio.layers[detail_layer], part)
    values, inputs, handles = {}, [], []
    def capture_input(mod, args):
        inputs.append(args[0].detach().clone())
    def observe(name):
        def capture(mod, args, output):
            values[name] = output.detach().clone()
            if name == "pre_layer_norm":
                values["clamp_input"] = args[0].detach().clone()
            elif name == "post_layer_norm":
                values["clamp_output"] = args[0].detach().clone()
        return capture
    try:
        handles.append(module.register_forward_pre_hook(capture_input))
        for name, child in [("pre_layer_norm", module.pre_layer_norm), ("ffw_layer_1", module.ffw_layer_1),
                            ("activation", module.act_fn), ("ffw_layer_2", module.ffw_layer_2),
                            ("post_layer_norm", module.post_layer_norm), ("residual_output", module)]:
            handles.append(child.register_forward_hook(observe(name)))
        with torch.no_grad():
            audio(input_features=features, attention_mask=frame_mask.bool())
            values["post_layer_scale"] = values["post_layer_norm"] * module.post_layer_scale
    finally:
        for handle in handles:
            handle.remove()
    return inputs[0], tuple(values[name] for name in AudioFeedForwardStagesExport.stage_names)


def diagnose_audio_feed_forward_stages(audio, features, frame_mask, destination, failed_stage, *, detail_layer, part):
    """Localize an operation only when this standalone final equals the full trace."""
    hidden, expected = capture_official_audio_feed_forward(audio, features, frame_mask,
                                                         detail_layer=detail_layer, part=part)
    wrapper = AudioFeedForwardStagesExport(getattr(audio.layers[detail_layer], part), hidden.shape)
    destination = Path(destination)
    path = destination / f"audio-layer-{detail_layer}-{part}-stages.pt2"
    export_audio_pt2(wrapper, hidden, path)
    with torch.no_grad():
        program = torch.export.load(path)
        actual = program.module()(hidden)
    result = compare_audio_stage_outputs(wrapper.stage_names, expected, actual)
    if (failed_stage.dtype != torch.float32 or failed_stage.shape != actual[-1].shape
            or not torch.isfinite(failed_stage).all()):
        raise ValueError("audio feed-forward trace cannot be compared to the full stage")
    matches = torch.equal(actual[-1], failed_stage)
    first_nonzero = next((row["name"] for row in result["stages"] if row["maximum_absolute_error"] > 0), None)
    input_path = path.with_suffix(".native-input-fp32.bin")
    hidden.numpy().astype("<f4").tofile(input_path)
    result.update(stage="audio_feed_forward_diagnostics_only", detail_layer=detail_layer, part=part,
                  feature_rtol=1e-4, feature_atol=1e-5, first_nonzero_stage=first_nonzero,
                  standalone_matches_full_stage=matches, localization_validated=matches,
                  standalone_vs_full_stage_maximum_absolute_error=(actual[-1] - failed_stage).abs().max().item(),
                  encoders_validated=False, end_to_end_validated=False, native_input_shape=list(hidden.shape),
                  operators=sorted({str(node.target) for node in program.graph.nodes if node.op == "call_function"}))
    for name, artifact in [("graph", path), ("native_input", input_path)]:
        with artifact.open("rb") as stream:
            result[f"{name}_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        result[f"{name}_bytes"] = artifact.stat().st_size
    (destination / f"audio-layer-{detail_layer}-{part}-diagnostics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def diagnose_audio_stages(audio, projection, features, frame_mask, destination, failed_saved):
    """Export a stage trace only after a numeric failure; never return encoder success."""
    destination = Path(destination)
    wrapper = AudioEncoderStagesExport(audio, projection, frame_mask)
    path = destination / "audio-stages.pt2"
    expected = capture_official_audio_stages(audio, projection, features, frame_mask)
    export_audio_pt2(wrapper, features, path)
    with torch.no_grad():
        actual = torch.export.load(path).module()(features)
    result = compare_audio_stage_outputs(wrapper.stage_names, expected, actual)
    if failed_saved.shape != actual[-1].shape or not torch.isfinite(failed_saved).all():
        raise ValueError("audio stage trace cannot be compared to the failed graph")
    result.update(stage="audio_encoder_stage_diagnostics_only", feature_rtol=1e-4, feature_atol=1e-5,
                  trace_matches_failed_graph=torch.equal(actual[-1], failed_saved),
                  trace_vs_failed_graph_maximum_absolute_error=(actual[-1] - failed_saved).abs().max().item())
    with path.open("rb") as stream:
        result["graph_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    result["graph_bytes"] = path.stat().st_size
    receipt = destination / "audio-stage-diagnostics.json"
    receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if result["trace_matches_failed_graph"] and result["first_divergent_stage"] is not None:
        first_difference = next((row for row in result["stages"] if row["maximum_absolute_error"] > 0), None)
        selected = re.fullmatch(r"layer_(\d+)", first_difference["name"]) if first_difference else None
        if selected:
            # Retain the completed coarse trace before another expensive export.
            # A final-path mismatch makes localization of the failed graph unsafe.
            del actual, expected, wrapper
            gc.collect()
            try:
                result["layer_diagnostics"] = diagnose_audio_layer_stages(
                    audio, projection, features, frame_mask, destination, failed_saved,
                    detail_layer=int(selected.group(1)))
            except Exception as diagnostic_error:
                result["layer_diagnostics_error"] = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
    receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def diagnose_audio_layer_stages(audio, projection, features, frame_mask, destination, failed_saved, *, detail_layer):
    """Inspect one native layer without changing the saved final numerical path."""
    wrapper = AudioEncoderStagesExport(audio, projection, frame_mask, detail_layer=detail_layer)
    path = Path(destination) / f"audio-layer-{detail_layer}-stages.pt2"
    expected = capture_official_audio_stages(audio, projection, features, frame_mask, detail_layer=detail_layer)
    export_audio_pt2(wrapper, features, path)
    with torch.no_grad():
        actual = torch.export.load(path).module()(features)
    result = compare_audio_stage_outputs(wrapper.stage_names, expected, actual)
    if failed_saved.shape != actual[-1].shape or not torch.isfinite(failed_saved).all():
        raise ValueError("audio detail stage trace cannot be compared to the failed graph")
    result.update(stage="audio_encoder_layer_diagnostics_only", detail_layer=detail_layer,
                  feature_rtol=1e-4, feature_atol=1e-5,
                  trace_matches_failed_graph=torch.equal(actual[-1], failed_saved),
                  trace_vs_failed_graph_maximum_absolute_error=(actual[-1] - failed_saved).abs().max().item())
    with path.open("rb") as stream:
        result["graph_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    result["graph_bytes"] = path.stat().st_size
    receipt = path.with_name(f"audio-layer-{detail_layer}-diagnostics.json")
    receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    first_difference = next((row for row in result["stages"] if row["maximum_absolute_error"] > 0), None)
    selected = re.fullmatch(rf"layer_{detail_layer}\.(feed_forward[12])", first_difference["name"]) if first_difference else None
    if result["trace_matches_failed_graph"] and result["first_divergent_stage"] is not None and selected:
        failed_stage = actual[wrapper.stage_names.index(first_difference["name"])].detach().clone()
        del actual, expected, wrapper
        gc.collect()
        try:
            result["feed_forward_diagnostics"] = diagnose_audio_feed_forward_stages(
                audio, features, frame_mask, destination, failed_stage, detail_layer=detail_layer, part=selected.group(1))
        except Exception as diagnostic_error:
            result["feed_forward_diagnostics_error"] = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
    receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def capture_official_audio_subsample_stages(audio, features, frame_mask):
    """Observe the native subsample module without replacing its forward path."""
    values, handles = {}, []
    module = audio.subsample_conv_projection
    def before(name):
        def capture(child, args):
            values[name] = args[0].detach().clone()
        return capture
    def after(name):
        def capture(child, args, output):
            values[name] = output.detach().clone()
        return capture
    try:
        for index, layer in enumerate((module.layer0, module.layer1)):
            prefix = f"layer{index}."
            handles.append(layer.conv.register_forward_pre_hook(before(prefix + "masked_input")))
            for name, child in (("conv", layer.conv), ("norm", layer.norm), ("relu", layer.act)):
                handles.append(child.register_forward_hook(after(prefix + name)))
        handles.append(module.input_proj_linear.register_forward_pre_hook(before("projection_input")))
        handles.append(module.input_proj_linear.register_forward_hook(after("projection")))
        with torch.no_grad():
            module(features, frame_mask.bool())
    finally:
        for handle in handles:
            handle.remove()
    return tuple(values[name] for name in AudioSubsampleStagesExport.stage_names)


def diagnose_audio_onnx_subsample(audio, features, frame_mask, destination, failed_subsample, official_subsample):
    """Inspect subsample operators only against the completed coarse trace."""
    from onnx.reference import ReferenceEvaluator

    destination = Path(destination)
    path = destination / "audio-onnx-subsample-stages.onnx"
    wrapper = AudioSubsampleStagesExport(audio.subsample_conv_projection, features.shape, frame_mask)
    expected = capture_official_audio_subsample_stages(audio, features, frame_mask)
    export_audio_native_onnx(wrapper, features, path)
    evaluator = ReferenceEvaluator(str(path))
    actual = tuple(torch.from_numpy(value) for value in evaluator.run(None, {"input": features.numpy()}))
    result = compare_audio_stage_outputs(wrapper.stage_names, expected, actual)
    for value in (failed_subsample, official_subsample):
        if (value.dtype != torch.float32 or value.shape != actual[-1].shape or not torch.isfinite(value).all()):
            raise ValueError("subsample diagnostic context has invalid shape/dtype/nonfinite data")
    trace_matches = torch.equal(actual[-1], failed_subsample)
    native_matches = torch.equal(expected[-1], official_subsample)
    result.update(stage="audio_onnx_subsample_diagnostics_only", trace_matches_full_subsample=trace_matches,
                  native_reference_matches_full_subsample=native_matches,
                  trace_vs_full_subsample_maximum_absolute_error=(actual[-1] - failed_subsample).abs().max().item(),
                  native_vs_full_subsample_maximum_absolute_error=(expected[-1] - official_subsample).abs().max().item(),
                  localization_validated=trace_matches and native_matches,
                  encoders_validated=False, end_to_end_validated=False, feature_rtol=1e-4, feature_atol=1e-5,
                  first_nonzero_stage=next((row["name"] for row in result["stages"]
                                            if row["maximum_absolute_error"] > 0), None))
    artifacts = [("graph", path)]
    for name, value, dtype in (("native_input", features, "<f4"), ("frame_mask", frame_mask, "<i4"),
                               ("native_subsample", official_subsample, "<f4")):
        binary = destination / f"audio-subsample-{name}.bin"
        value.numpy().astype(dtype).tofile(binary)
        artifacts.append((name, binary))
    result["files"] = {}
    for name, artifact in artifacts:
        with artifact.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        result["files"][name] = dict(path=artifact.name, bytes=artifact.stat().st_size, sha256=digest)
    (destination / "audio-onnx-subsample-diagnostics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def diagnose_audio_onnx_stages(audio, projection, features, frame_mask, destination, failed_onnx, *, detail_layer=None,
                              official_reference=None):
    """Trace standard ONNX only in the context of the original failed final path."""
    from onnx.reference import ReferenceEvaluator

    destination = Path(destination)
    wrapper = AudioEncoderStagesExport(audio, projection, frame_mask, detail_layer=detail_layer)
    suffix = "stages" if detail_layer is None else f"layer-{detail_layer}-stages"
    path = destination / f"audio-onnx-{suffix}.onnx"
    expected = capture_official_audio_stages(audio, projection, features, frame_mask, detail_layer=detail_layer)
    if official_reference is None:
        with torch.no_grad():
            native = audio(input_features=features, attention_mask=frame_mask.bool())
            official_reference = projection(native.last_hidden_state)[native.attention_mask]
    if (official_reference.dtype != torch.float32 or official_reference.shape != expected[-1].shape
            or not torch.isfinite(official_reference).all()):
        raise ValueError("audio ONNX diagnostic original reference is invalid")
    native_matches = torch.equal(expected[-1], official_reference)
    export_audio_native_onnx(wrapper, features, path)
    evaluator = ReferenceEvaluator(str(path))
    actual = tuple(torch.from_numpy(value) for value in evaluator.run(None, {"input": features.numpy()}))
    result = compare_audio_stage_outputs(wrapper.stage_names, expected, actual)
    if (failed_onnx.dtype != torch.float32 or failed_onnx.shape != actual[-1].shape
            or not torch.isfinite(failed_onnx).all()):
        raise ValueError("audio ONNX trace cannot be compared to the failed graph")
    matches = torch.equal(actual[-1], failed_onnx)
    with path.open("rb") as stream:
        graph_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    result.update(stage="audio_encoder_onnx_stage_diagnostics_only", detail_layer=detail_layer,
                  feature_rtol=1e-4, feature_atol=1e-5, trace_matches_failed_graph=matches,
                  native_reference_matches_original=native_matches,
                  native_reference_vs_original_maximum_absolute_error=(expected[-1] - official_reference).abs().max().item(),
                  localization_validated=matches and native_matches, encoders_validated=False, end_to_end_validated=False,
                  first_nonzero_stage=next((row["name"] for row in result["stages"]
                                            if row["maximum_absolute_error"] > 0), None),
                  trace_vs_failed_graph_maximum_absolute_error=(actual[-1] - failed_onnx).abs().max().item(),
                  graph_sha256=graph_hash, graph_bytes=path.stat().st_size)
    receipt = path.with_name(path.stem + "-diagnostics.json")
    receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    selected = re.fullmatch(r"layer_(\d+)", result["first_nonzero_stage"] or "")
    if (detail_layer is None and matches and native_matches and result["first_divergent_stage"] is not None
            and result["first_nonzero_stage"] == "subsample"):
        failed_subsample, official_subsample = actual[0], expected[0]
        del evaluator, actual, expected, wrapper
        gc.collect()
        try:
            result["subsample_diagnostics"] = diagnose_audio_onnx_subsample(
                audio, features, frame_mask, destination, failed_subsample, official_subsample)
        except Exception as diagnostic_error:
            result["subsample_diagnostics_error"] = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
    elif detail_layer is None and matches and native_matches and result["first_divergent_stage"] is not None and selected:
        # Release coarse ONNX weights before the selected-layer export.
        del evaluator, actual, expected, wrapper
        gc.collect()
        try:
            result["layer_diagnostics"] = diagnose_audio_onnx_stages(
                audio, projection, features, frame_mask, destination, failed_onnx,
                detail_layer=int(selected.group(1)), official_reference=official_reference)
        except Exception as diagnostic_error:
            result["layer_diagnostics_error"] = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
    receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def prepare_audio_encoder(snapshot, preprocessing, destination, *, revision=MODEL_REVISION, source_commit="local", native_silu=False):
    """Official audio tower/projector vs saved graph on audited PCM-derived features.

    Independent inputs are the official NumPy preprocessing references, not the
    output of our preprocessing graph. This is not final embeddings or Unity proof.
    """
    snapshot, preprocessing, destination = Path(snapshot), Path(preprocessing).resolve(), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "audio-encoder-validation.json").unlink(missing_ok=True)
    (destination / "audio-encoder-failure.json").unlink(missing_ok=True)
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("model revision must be an immutable lowercase commit SHA")

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    audit = json.loads((preprocessing / "audio-validation.json").read_text(encoding="utf-8"))
    if (audit.get("success") is not True or audit.get("stage") != "audio_preprocessing_only"
            or audit["metadata"].get("source_commit") != source_commit
            or audit["metadata"].get("model_revision") != revision):
        raise ValueError("preprocessing completion, stage, revision or source mismatch")
    audited = {}
    for name, item in audit["files"].items():
        path = (preprocessing / name).resolve()
        if (not path.is_relative_to(preprocessing) or not path.is_file()
                or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]):
            raise ValueError(f"preprocessing artifact hash/path mismatch: {name}")
        audited[name] = path
    required = {"audio-reference.json", "processor_config.json"}
    if not required.issubset(audited):
        raise ValueError("preprocessing manifest omits reference/config")
    reference = json.loads(audited["audio-reference.json"].read_text(encoding="utf-8"))
    metadata = reference["metadata"]
    if (metadata.get("source_commit") != source_commit or metadata.get("model_revision") != revision
            or metadata.get("stage") != "audio_preprocessing_only"
            or metadata.get("processor_sha256") != digest(snapshot / "processor_config.json")
            or digest(audited["processor_config.json"]) != metadata["processor_sha256"]):
        raise ValueError("preprocessing reference/config/source mismatch")
    shape = metadata["feature_shape"]
    if (not isinstance(shape, list) or len(shape) != 3 or shape[0] != 1 or shape[2] != 128
            or type(shape[1]) is not int or shape[1] <= 0 or metadata["frame_mask_shape"] != shape[:2]
            or not reference["cases"] or len(reference["cases"]) != audit["case_count"]):
        raise ValueError("preprocessing frame/feature/case profile mismatch")
    groups = {}
    for row in reference["cases"]:
        if re.fullmatch(r"[a-z][a-z0-9_-]*", row["id"]) is None:
            raise ValueError("preprocessing case ID must be safe")
        if not {row["features"], row["frame_mask"]}.issubset(audited):
            raise ValueError("preprocessing manifest omits feature/mask input")
        feature_array = np.fromfile(audited[row["features"]], dtype="<f4")
        mask_array = np.fromfile(audited[row["frame_mask"]], dtype="<i4")
        if feature_array.size != shape[1] * 128 or mask_array.size != shape[1]:
            raise ValueError("preprocessing binary input size mismatch")
        features = torch.from_numpy(feature_array.reshape(shape))
        frame_mask = torch.from_numpy(mask_array.reshape(1, shape[1])).to(torch.int64)
        # Validate before loading weights and before grouping by prefix length.
        if (not torch.isfinite(features).all() or not torch.all((frame_mask == 0) | (frame_mask == 1))
                or not frame_mask.any() or torch.any(frame_mask[:, 1:] > frame_mask[:, :-1])
                or torch.any(features[frame_mask == 0] != 0)):
            raise ValueError("preprocessing requires finite features and a nonempty zero-padded prefix mask")
        profile = f"frames-{shape[1]}-valid-{int(frame_mask.sum())}"
        groups.setdefault(profile, []).append((row, features, frame_mask))
    audio, projection = load_audio_models(snapshot)
    rows, profiles, produced, similarities = [], {}, [], []
    onnx_similarities = []
    for profile, inputs in groups.items():
        directory = destination / profile
        directory.mkdir(parents=True, exist_ok=True)
        wrapper = AudioEncoderExport(audio, projection, inputs[0][2])
        graph = directory / "audio.pt2"
        expected_rows = []
        with torch.no_grad():
            for row, features, frame_mask in inputs:
                output = audio(input_features=features, attention_mask=frame_mask.bool())
                expected = projection(output.last_hidden_state)[output.attention_mask]
                if not torch.isfinite(expected).all():
                    raise ValueError("official audio encoder reference must be finite")
                expected_rows.append((features, expected, row["id"]))
        export_audio_pt2(wrapper, inputs[0][1], graph)
        exported = torch.export.load(graph).module()
        with torch.no_grad():
            for features, expected, case_id in expected_rows:
                actual = exported(features)
                if actual.dtype != torch.float32 or actual.shape != expected.shape or not torch.isfinite(actual).all():
                    raise ValueError(f"audio encoder export mismatch: {case_id} shape/dtype/nonfinite")
                cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=-1).min().item()
                if cosine < 0.999999 or not torch.allclose(actual, expected, rtol=1e-4, atol=1e-5):
                    eager = wrapper(features)
                    difference = (actual - expected).abs()
                    allowed = 1e-5 + 1e-4 * expected.abs()
                    failure = dict(success=False, stage="audio_encoder_projector_only", source_commit=source_commit,
                                   profile=profile, case_id=case_id, minimum_cosine=cosine,
                                   saved_vs_official_maximum_absolute_error=difference.max().item(),
                                   eager_vs_official_maximum_absolute_error=(eager - expected).abs().max().item(),
                                   saved_vs_eager_maximum_absolute_error=(actual - eager).abs().max().item(),
                                   maximum_tolerance_ratio=(difference / allowed).max().item(),
                                   mismatched_elements=int((difference > allowed).sum()),
                                   graph_sha256=digest(graph), unity_validated=False, end_to_end_validated=False,
                                   operators=sorted({str(node.target) for node in exported.graph.nodes if node.op == "call_function"}))
                    failure_path = destination / "audio-encoder-failure.json"
                    failure_path.write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                    expected.numpy().astype("<f4").tofile(destination / "failed-official-features-fp32.bin")
                    actual.numpy().astype("<f4").tofile(destination / "failed-saved-features-fp32.bin")
                    # Release the saved module's duplicate weights before exporting
                    # another real graph. A trace changes the output contract, so
                    # record whether its final value still matches this failed graph.
                    del exported
                    gc.collect()
                    try:
                        failure["stage_diagnostics"] = diagnose_audio_stages(
                            audio, projection, features, inputs[0][2], directory, actual)
                    except Exception as diagnostic_error:
                        failure["stage_diagnostics_error"] = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                    failure_path.write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                    print(json.dumps(failure, allow_nan=False))
                    raise ValueError(f"audio encoder export mismatch: {case_id} cosine={cosine}")
                similarities.append(cosine)
                input_path = directory / f"{case_id}-features-fp32.bin"
                features.numpy().astype("<f4").tofile(input_path)
                produced.append(input_path)
                rows.append(dict(id=case_id, profile=profile, input_features=input_path.relative_to(destination).as_posix(),
                                 features=expected.tolist()))
        mask_path = directory / "frame-mask-int32.bin"
        inputs[0][2].numpy().astype("<i4").tofile(mask_path)
        produced.extend((graph, mask_path))
        profiles[profile] = dict(feature_shape=shape, feature_count=wrapper.feature_count,
                                 feature_width=projection.text_hidden_size, frame_mask_shape=shape[:2],
                                 frame_mask=mask_path.relative_to(destination).as_posix(),
                                 graph=graph.relative_to(destination).as_posix())
        if native_silu:
            from onnx.reference import ReferenceEvaluator

            # Release the duplicate native graph weights before loading ONNX.
            del exported
            gc.collect()
            imported_graph=directory/"audio.onnx"
            export_audio_native_onnx(wrapper,inputs[0][1],imported_graph)
            evaluator = ReferenceEvaluator(str(imported_graph))
            for features, expected, case_id in expected_rows:
                actual_array = evaluator.run(None, {"input": features.numpy()})[0]
                if (actual_array.dtype != np.float32 or actual_array.shape != tuple(expected.shape)
                        or not np.isfinite(actual_array).all()):
                    failure = dict(success=False, stage="audio_encoder_onnx", source_commit=source_commit,
                                   profile=profile, case_id=case_id, reason="shape/dtype/nonfinite",
                                   graph_sha256=digest(imported_graph), unity_validated=False,
                                   end_to_end_validated=False)
                else:
                    actual = torch.from_numpy(actual_array)
                    cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=-1).min().item()
                    difference = (actual - expected).abs()
                    allowed = 1e-5 + 1e-4 * expected.abs()
                    mismatched = int((difference > allowed).sum())
                    if cosine >= 0.999999 and mismatched == 0:
                        onnx_similarities.append(cosine)
                        row = next(item for item in rows if item["id"] == case_id)
                        row.update(onnx_minimum_cosine=cosine,
                                   onnx_maximum_absolute_error=difference.max().item(),
                                   onnx_maximum_tolerance_ratio=(difference / allowed).max().item())
                        continue
                    failure = dict(success=False, stage="audio_encoder_onnx", source_commit=source_commit,
                                   profile=profile, case_id=case_id, minimum_cosine=cosine,
                                   onnx_vs_official_maximum_absolute_error=difference.max().item(),
                                   maximum_tolerance_ratio=(difference / allowed).max().item(),
                                   mismatched_elements=mismatched, graph_sha256=digest(imported_graph),
                                   unity_validated=False, end_to_end_validated=False)
                    expected.numpy().astype("<f4").tofile(destination / "failed-official-features-fp32.bin")
                    actual_array.astype("<f4").tofile(destination / "failed-onnx-features-fp32.bin")
                (destination / "audio-encoder-failure.json").write_text(
                    json.dumps(failure, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                if failure.get("reason") is None:
                    del evaluator
                    gc.collect()
                    try:
                        failure["onnx_stage_diagnostics"] = diagnose_audio_onnx_stages(
                            audio, projection, features, inputs[0][2], directory, actual, official_reference=expected)
                    except Exception as diagnostic_error:
                        failure["onnx_stage_diagnostics_error"] = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                    (destination / "audio-encoder-failure.json").write_text(
                        json.dumps(failure, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                raise ValueError(f"audio ONNX export mismatch: {case_id}")
            del evaluator
            produced.append(imported_graph)
            profiles[profile].update(native_graph=graph.relative_to(destination).as_posix(),
                                     graph=imported_graph.relative_to(destination).as_posix(),import_format="onnx")
        else:
            profiles[profile]["import_format"]="native_torch_reference_only"
            del exported
        del wrapper
    encoder_metadata = dict(model_id=MODEL_ID, model_revision=revision, source_commit=source_commit,
                            stage="audio_encoder_projector_only", dtype="float32", batch_size=1,
                            attention_implementation="sdpa", minimum_cosine=0.999999,
                            feature_rtol=1e-4, feature_atol=1e-5,
                            preprocessing_reference_sha256=digest(audited["audio-reference.json"]),
                            fixture_provenance=metadata["fixture_provenance"],
                            input_sha256={name: digest(snapshot / name) for name in
                                          ("config.json", "processor_config.json", "model.safetensors")},
                            versions={name: importlib.metadata.version(name) for name in ("torch", "transformers")})
    reference_path = destination / "audio-encoder-reference.json"
    reference_path.write_text(json.dumps(dict(metadata=encoder_metadata, profiles=profiles, cases=rows),
                                         indent=2, allow_nan=False) + "\n", encoding="utf-8")
    produced.append(reference_path)
    summary = dict(success=True, stage=encoder_metadata["stage"], profile_count=len(profiles), case_count=len(rows),
                   minimum_cosine=min(similarities), unity_validated=False, end_to_end_validated=False,
                   metadata=encoder_metadata, files={path.relative_to(destination).as_posix():
                       dict(bytes=path.stat().st_size, sha256=digest(path)) for path in produced})
    summary["native_silu_preserved"]=True
    summary["onnx_structurally_validated"]=native_silu
    summary["onnx_numerically_validated"]=native_silu
    summary["onnx_case_count"] = len(onnx_similarities)
    summary["onnx_minimum_cosine"] = min(onnx_similarities) if onnx_similarities else None
    (destination / "audio-encoder-validation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                                              encoding="utf-8")
    return summary


def load_image_processor(snapshot):
    from transformers import Gemma4ImageProcessor
    configuration = json.loads((Path(snapshot) / "processor_config.json").read_text(encoding="utf-8"))
    settings = configuration["image_processor"]
    if settings.get("image_processor_type") != "Gemma4ImageProcessor":
        raise ValueError("fixed Gemma4 image processor is required")
    return Gemma4ImageProcessor(**{key: value for key, value in settings.items()
                                 if key not in ("image_processor_type", "image_seq_length")})


def prepare_vision(snapshot, destination, *, revision=MODEL_REVISION, source_commit="local", profiles=None):
    """Official processor and real tower/projector vs saved fixed-profile export.

    Fixtures are generated RGB images with known provenance. This stage does not
    prove Sentis compatibility, final embeddings or semantic retrieval quality.
    """
    from PIL import Image
    snapshot, destination = Path(snapshot), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "vision-validation.json").unlink(missing_ok=True)
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("model revision must be an immutable lowercase commit SHA")
    profiles = {"square": (96, 96), "landscape": (96, 160), "portrait": (160, 96)} if profiles is None else profiles
    if (not isinstance(profiles, dict) or not profiles or any(
            not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_-]*", name) is None
            or not isinstance(size, (tuple, list)) or len(size) != 2
            or any(type(dimension) is not int or dimension <= 0 for dimension in size)
            for name, size in profiles.items())):
        raise ValueError("vision profiles require safe names and positive integer image dimensions")
    vision, projection = load_vision_models(snapshot)
    processor = load_image_processor(snapshot)
    rows, profile_metadata, similarities, produced = [], {}, [], []
    for profile, (height, width) in profiles.items():
        directory = destination / profile
        directory.mkdir(parents=True, exist_ok=True)
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        images = [torch.stack((x * 7 % 256, y * 11 % 256, (x + y) * 5 % 256)).to(torch.uint8),
                  torch.stack(((x // 3 + y // 5) % 2 * 255, (x * 3 + y * 2) % 256,
                               (x * 13 - y * 7) % 256)).to(torch.uint8)]
        prepared = [processor(images=image, return_tensors="pt") for image in images]
        coords = prepared[0]["image_position_ids"]
        wrapper = VisionEncoderExport(vision, projection, coords)
        expected_rows = []
        for index, (image, inputs) in enumerate(zip(images, prepared, strict=True)):
            pixels = inputs["pixel_values"].float()
            wrapper.validate_inputs(pixels)
            if (not torch.equal(inputs["image_position_ids"], coords)
                    or torch.as_tensor(inputs["num_soft_tokens_per_image"]).tolist() != [wrapper.feature_count]):
                raise ValueError("image processor output does not match the fixed vision profile")
            with torch.no_grad():
                expected = projection(vision(pixel_values=pixels, pixel_position_ids=coords).last_hidden_state)
            if not torch.isfinite(expected).all():
                raise ValueError("official vision reference is not finite")
            case_id = f"{profile}-{index}"
            image_path = directory / f"{index}.png"
            Image.fromarray(image.permute(1, 2, 0).numpy()).save(image_path)
            pixel_path = directory / f"{index}-pixels-fp32.bin"
            pixels.numpy().astype("<f4").tofile(pixel_path)
            produced.extend((image_path, pixel_path))
            expected_rows.append((pixels, expected, case_id))
            rows.append(dict(id=case_id, profile=profile, image=image_path.relative_to(destination).as_posix(),
                             pixels=pixel_path.relative_to(destination).as_posix(), features=expected.tolist()))
        graph = directory / "vision.pt2"
        export_vision_pt2(wrapper, expected_rows[0][0], graph)
        exported = torch.export.load(graph).module()
        with torch.no_grad():
            for pixels, expected, case_id in expected_rows:
                actual = exported(pixels)
                if actual.shape != expected.shape or not torch.isfinite(actual).all():
                    raise ValueError(f"vision export mismatch: {case_id} shape or nonfinite output")
                cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=-1).min().item()
                if cosine < 0.999999 or not torch.allclose(actual, expected, rtol=1e-4, atol=1e-5):
                    raise ValueError(f"vision export mismatch: {case_id} cosine={cosine}")
                similarities.append(cosine)
        positions_path = directory / "positions-int32.bin"
        coords.numpy().astype("<i4").tofile(positions_path)
        produced.extend((graph, positions_path))
        profile_metadata[profile] = dict(pixel_shape=list(expected_rows[0][0].shape), feature_count=wrapper.feature_count,
                                         feature_width=projection.text_hidden_size, positions_shape=list(coords.shape),
                                         positions=positions_path.relative_to(destination).as_posix(),
                                         graph=graph.relative_to(destination).as_posix(), input_image_shape=[3, height, width])
        del exported, wrapper
    if not rows:
        raise ValueError("vision profiles must contain at least one input")
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = dict(model_id=MODEL_ID, model_revision=revision, source_commit=source_commit,
                    stage="vision_encoder_projector_only", dtype="float32", batch_size=1,
                    input_sha256={name: digest(snapshot / name)
                                  for name in ("config.json", "processor_config.json", "model.safetensors")},
                    fixture_provenance="procedurally_generated_rgb_patterns_no_user_data",
                    pixel_encoding="little_endian_float32_rgb_patches", position_encoding="little_endian_int32_xy",
                    minimum_cosine=0.999999, feature_rtol=1e-4, feature_atol=1e-5,
                    versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "torchvision")})
    reference_path = destination / "vision-reference.json"
    reference_path.write_text(json.dumps(dict(metadata=metadata, profiles=profile_metadata, cases=rows),
                                         indent=2, allow_nan=False) + "\n", encoding="utf-8")
    processor_path = destination / "processor_config.json"
    shutil.copyfile(snapshot / "processor_config.json", processor_path)
    produced.extend((reference_path, processor_path))
    summary = dict(success=True, stage=metadata["stage"], unity_validated=False, end_to_end_validated=False,
                   profile_count=len(profile_metadata), case_count=len(rows), minimum_cosine=min(similarities),
                   metadata=metadata, files={path.relative_to(destination).as_posix():
                       dict(bytes=path.stat().st_size, sha256=digest(path)) for path in produced})
    (destination / "vision-validation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                                       encoding="utf-8")
    return summary


def prepare_multimodal_backbone(snapshot, destination, sequence_length=1024, *, image_feature_count=280,
                                audio_feature_count=280, revision=MODEL_REVISION, source_commit="local"):
    """Validate the actual text backbone with deterministic projected features.

    This is an intermediate contract gate, not an image/audio encoder reference
    or a Sentis execution result. The independent oracle follows the official
    EmbeddingGemma2Model's safe-ID and masked-scatter input composition.
    """
    snapshot, destination = Path(snapshot), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "backbone-validation.json").unlink(missing_ok=True)
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("model revision must be an immutable lowercase commit SHA")
    model = load_text_model(snapshot)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    wrapper = MultimodalEmbeddingExport(model, sequence_length, image_token_id=config["image_token_id"],
                                       audio_token_id=config["audio_token_id"],
                                       image_feature_count=image_feature_count, audio_feature_count=audio_feature_count)
    if image_feature_count + audio_feature_count + 2 > sequence_length:
        raise ValueError("profile must also fit BOS and EOS")
    generator = torch.Generator().manual_seed(20261013)
    image = torch.randn(image_feature_count, model.config.hidden_size, generator=generator) * 0.2
    audio = torch.randn(audio_feature_count, model.config.hidden_size, generator=generator) * 0.2
    rows = []
    for reverse in (False, True):
        modality_ids = ([config["image_token_id"]] * image_feature_count +
                        [config["audio_token_id"]] * audio_feature_count)
        if reverse:
            modality_ids.reverse()
        values = [model.config.bos_token_id, *modality_ids, model.config.eos_token_id]
        ids = torch.tensor([values + [model.config.pad_token_id] * (sequence_length - len(values))])
        mask = torch.tensor([[1] * len(values) + [0] * (sequence_length - len(values))])
        feature_image, feature_audio = (image.flip(0), audio.flip(0)) if reverse else (image, audio)
        wrapper.validate_inputs(ids, mask, feature_image, feature_audio)
        with torch.no_grad():
            safe_ids = ids.clone()
            safe_ids[(ids == config["image_token_id"]) | (ids == config["audio_token_id"])] = model.config.pad_token_id
            embeddings = model.get_input_embeddings()(safe_ids)
            for token, feature in ((config["image_token_id"], feature_image), (config["audio_token_id"], feature_audio)):
                embeddings = embeddings.masked_scatter((ids == token).unsqueeze(-1).expand_as(embeddings), feature)
            hidden = model(inputs_embeds=embeddings, attention_mask=mask).last_hidden_state
            expected = torch.nn.functional.normalize(hidden[:, mask[0].bool()].mean(dim=1), dim=-1)
        rows.append(dict(id="reordered" if reverse else "ordered", input_ids=ids[0].tolist(),
                         attention_mask=mask[0].tolist(), image_features=feature_image.tolist(),
                         audio_features=feature_audio.tolist(), embedding=expected[0].tolist()))
    first = rows[0]
    export_multimodal_pt2(wrapper, torch.tensor([first["input_ids"]]), torch.tensor([first["attention_mask"]]),
                         image, audio, destination / "backbone.pt2")
    exported = torch.export.load(destination / "backbone.pt2").module()
    similarities = []
    with torch.no_grad():
        for row in rows:
            actual = exported(torch.tensor([row["input_ids"]]), torch.tensor([row["attention_mask"]]),
                              torch.tensor(row["image_features"]).reshape(image_feature_count, model.config.hidden_size),
                              torch.tensor(row["audio_features"]).reshape(audio_feature_count, model.config.hidden_size))[0]
            if not torch.isfinite(actual).all():
                raise ValueError("exported backbone embedding must be finite")
            cosine = torch.nn.functional.cosine_similarity(actual, torch.tensor(row["embedding"]), dim=0).item()
            if cosine < 0.999999:
                raise ValueError(f"backbone export mismatch: {row['id']} cosine={cosine}")
            similarities.append(cosine)
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = dict(model_id=MODEL_ID, model_revision=revision, source_commit=source_commit,
                    sequence_length=sequence_length, batch_size=1, dtype="float32",
                    image_token_id=config["image_token_id"], audio_token_id=config["audio_token_id"],
                    image_feature_count=image_feature_count, audio_feature_count=audio_feature_count,
                    feature_width=model.config.hidden_size, embedding_dimension=model.config.embedding_dim,
                    stage="projected_features_backbone_only", pooling="masked_mean_including_prompt", normalization="l2",
                    input_sha256={name: digest(snapshot / name)
                                  for name in ("config.json", "processor_config.json", "model.safetensors")},
                    versions={name: importlib.metadata.version(name) for name in ("torch", "transformers")})
    write_reference(destination / "backbone-reference.json", rows, metadata)
    shutil.copyfile(snapshot / "processor_config.json", destination / "processor_config.json")
    summary = dict(success=True, stage=metadata["stage"], unity_validated=False, encoders_validated=False,
                   case_count=len(rows), minimum_cosine=min(similarities), metadata=metadata,
                   files={name: dict(bytes=(destination / name).stat().st_size, sha256=digest(destination / name))
                          for name in ("backbone.pt2", "backbone-reference.json", "processor_config.json")})
    (destination / "backbone-validation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                                         encoding="utf-8")
    return summary


def prepare(
    snapshot: Path, cases_path: Path, destination: Path,
    sequence_length: int, revision: str, source_commit: str = "local", *, search_cases_path: Path | None = None,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    # Only the current completed generation may publish a success report.
    (destination / "export-validation.json").unlink(missing_ok=True)
    model = load_text_model(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    cases = build_reference(model, tokenizer, load_cases(cases_path), sequence_length)
    metadata = {
        "model_id": MODEL_ID, "model_revision": revision, "source_commit": source_commit,
        "sequence_length": sequence_length, "batch_size": 1, "dtype": "float32",
        "embedding_dimension": model.config.embedding_dim,
        "pooling": "masked_mean_including_prompt", "normalization": "l2",
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "sentence-transformers", "tokenizers")},
    }
    metadata["versions"]["torch"] = torch.__version__
    write_reference(destination / "reference.json", cases, metadata)
    search_cases = []
    if search_cases_path is not None:
        search = generate_search_reference(model, tokenizer, search_cases_path,
                                          destination / "search-reference.json", sequence_length, metadata)
        search_cases = search["documents"] + search["queries"]
    # The last reference call configures fixed-length truncation and padding in the backend.
    tokenizer.backend_tokenizer.save(str(destination / "tokenizer.json"))
    first = cases[0]
    export_pt2(model, torch.tensor([first["input_ids"]]), torch.tensor([first["attention_mask"]]),
               destination / "model.pt2")
    exported = torch.export.load(destination / "model.pt2").module()
    similarities = []
    with torch.no_grad():
        for case in cases + search_cases:
            actual = exported(torch.tensor([case["input_ids"]]), torch.tensor([case["attention_mask"]]))[0]
            expected = torch.tensor(case["embedding"])
            if not torch.isfinite(actual).all():
                raise ValueError(f"export embedding must be finite: {case['id']}")
            similarity = torch.nn.functional.cosine_similarity(actual, expected, dim=0).item()
            if similarity < 0.999999:
                raise ValueError(f"export differs from reference: {case['id']} cosine={similarity}")
            similarities.append({"id": case["id"], "cosine": similarity})
    digests = {}
    names = ["model.pt2", "reference.json", "tokenizer.json"]
    if search_cases_path is not None:
        names.append("search-reference.json")
    for name in names:
        with (destination / name).open("rb") as stream:
            digests[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    base_similarities = similarities[:len(cases)]
    summary = {"case_count": len(cases), "minimum_cosine": min(row["cosine"] for row in base_similarities),
               "cases": base_similarities, "sha256": digests, "metadata": metadata}
    if search_cases_path is not None:
        search_similarities = similarities[len(cases):]
        summary.update(search_case_count=len(search_similarities), search_cases=search_similarities,
                       search_minimum_cosine=min(row["cosine"] for row in search_similarities))
    (destination / "export-validation.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary
