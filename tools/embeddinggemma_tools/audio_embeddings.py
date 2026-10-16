"""Official audio-only final vectors; native saved-chain and ONNX gates stay separate."""

import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer, EmbeddingGemma2Processor

from . import export
from .model import load_audio_embedding_model
from .text import format_text
from .backbone_reuse import audit_backbone_source, validate_reuse_destination
from .reference_context_failure import record_feature_drift


def official_forward_from_frozen_features(core, *, ids, mask, input_features, frame_mask, frozen_features):
    """Reference-only boundary: original encoder bytes through official model forward.

    This deliberately does not execute or qualify the encoder. The caller must
    audit the frozen files and anchor this boundary to the original Document
    vectors before accepting any new role. No global/class method is modified.
    """
    if (frame_mask.ndim != 2 or frame_mask.shape[0] != 1 or frame_mask.shape != input_features.shape[:2]
            or not torch.all((frame_mask == 0) | (frame_mask == 1)) or not frame_mask.any()
            or torch.any(frame_mask[:, 1:].int() > frame_mask[:, :-1].int())
            or frozen_features.ndim != 2 or frozen_features.shape[0] != (int(frame_mask.sum()) + 3) // 4
            or frozen_features.shape[1] == 0 or frozen_features.dtype != torch.float32
            or not torch.isfinite(frozen_features).all()):
        raise ValueError("Frozen audio features require finite FP32 tokens and their matching prefix mask")
    original_features = frozen_features.clone()
    had_local = "get_audio_features" in core.__dict__
    original = core.__dict__.get("get_audio_features")
    def frozen_boundary(self, features, feature_mask, **kwargs):
        if features is not input_features or feature_mask is not frame_mask:
            raise ValueError("Frozen audio official forward changed the input boundary")
        tokens = original_features.clone().unsqueeze(0)
        return SimpleNamespace(pooler_output=tokens,
                               attention_mask=torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device))
    core.get_audio_features = MethodType(frozen_boundary, core)
    try:
        output = core(input_ids=ids, attention_mask=mask, input_features=input_features, input_features_mask=frame_mask)
        if not torch.equal(output.audio_hidden_states, original_features):
            raise ValueError("Frozen audio official forward changed the original encoder outputs")
        return output
    finally:
        if had_local:
            core.get_audio_features = original
        else:
            del core.get_audio_features


def prepare_audio_embeddings(snapshot, encoder_directory, destination, *, sequence_length=1024, revision, source_commit,
                             role="document", encoder_source_commit=None, backbone_source=None, frozen_encoder_features=False):
    snapshot, encoder_directory, destination = Path(snapshot), Path(encoder_directory).resolve(), Path(destination)
    validate_reuse_destination(backbone_source, destination)
    destination.mkdir(parents=True, exist_ok=True)
    receipt = destination / "audio-embedding-validation.json"
    receipt.unlink(missing_ok=True)
    if frozen_encoder_features and backbone_source is None:
        raise ValueError("Frozen audio reference requires an audited original Document anchor backbone")
    format_text("", role)
    encoder_source_commit = source_commit if encoder_source_commit is None else encoder_source_commit
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None for value in (revision, source_commit, encoder_source_commit)):
        raise ValueError("Audio reference requires immutable provenance")
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = json.loads((encoder_directory / "audio-encoder-validation.json").read_text(encoding="utf-8"))
    metadata = manifest.get("metadata", {})
    if (manifest.get("stage") != "audio_encoder_projector_only" or manifest.get("native_reference_complete") is not True
            or metadata.get("source_commit") != encoder_source_commit or metadata.get("model_revision") != revision):
        raise ValueError("Audio reference requires complete pinned native encoder provenance")
    def audited_path(name):
        if not isinstance(name, str):
            raise ValueError("Audio reference path is invalid")
        path = (encoder_directory / name).resolve()
        item = manifest.get("files", {}).get(name, {})
        if (not path.is_relative_to(encoder_directory) or not path.is_file()
                or path.stat().st_size != item.get("bytes") or digest(path) != item.get("sha256")):
            raise ValueError("Audio reference file hash/path mismatch")
        return path
    reference_path = audited_path("audio-encoder-reference.json")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (reference.get("metadata") != metadata
            or set(metadata.get("input_sha256", {})) != {"config.json", "processor_config.json", "model.safetensors"}
            or any(digest(snapshot / name) != value for name, value in metadata["input_sha256"].items())):
        raise ValueError("Audio reference snapshot/metadata mismatch")
    cases = reference.get("cases", [])
    signals = {"silence", "impulse", "sine", "nyquist", "noise", "padded_noise"}
    if len(cases) != 6 or {row.get("id") for row in cases} != signals or manifest.get("case_count") != 6:
        raise ValueError("Audio reference requires all six original signals")
    prepared = {}
    for profile, spec in reference["profiles"].items():
        if re.fullmatch(r"frames-\d+-valid-\d+", profile) is None:
            raise ValueError("Audio reference profile name is unsafe")
        shape = spec["feature_shape"]
        raw_mask = np.fromfile(audited_path(spec["frame_mask"]), dtype="<i4")
        if len(shape) != 3 or shape[0] != 1 or shape[2] != 128 or raw_mask.size != shape[1]:
            raise ValueError("Audio reference frame shape mismatch")
        mask = torch.from_numpy(raw_mask.copy()).reshape(shape[:2]).long()
        if (not torch.all((mask == 0) | (mask == 1)) or not mask.any() or torch.any(mask[:, 1:] > mask[:, :-1])
                or profile != f"frames-{shape[1]}-valid-{int(mask.sum())}" or spec["feature_count"] != (int(mask.sum()) + 3) // 4):
            raise ValueError("Audio reference requires the matching nonempty prefix mask")
        native_graph = audited_path(spec.get("native_graph", spec["graph"]))
        inputs = []
        for row in cases:
            if row["profile"] != profile: continue
            raw = np.fromfile(audited_path(row["input_features"]), dtype="<f4")
            if raw.size != int(np.prod(shape)):
                raise ValueError("Audio reference feature size mismatch")
            features = torch.from_numpy(raw.copy()).reshape(shape)
            expected = torch.tensor(row["features"], dtype=torch.float32)
            if (not torch.isfinite(features).all() or torch.any(features[mask == 0] != 0)
                    or expected.shape != (spec["feature_count"], spec["feature_width"]) or not torch.isfinite(expected).all()):
                raise ValueError("Audio reference finite feature/output mismatch")
            inputs.append((row, features, expected))
        prepared[profile] = (spec, mask, native_graph, inputs)
    if len(prepared) != 2 or sum(len(group[3]) for group in prepared.values()) != 6:
        raise ValueError("Audio reference must cover two profiles and every case")
    input_hashes = {name: digest(snapshot / name) for name in ("config.json", "processor_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "config_sentence_transformers.json")}
    reuse = None
    if backbone_source is not None:
        hidden_size = json.loads((snapshot / "config.json").read_text())["text_config"]["hidden_size"]
        specs = {name: dict(sequence_length=sequence_length, image_feature_count=0, audio_feature_count=value[0]["feature_count"], feature_width=hidden_size)
                 for name, value in prepared.items()}
        reuse = audit_backbone_source(backbone_source, "audio", revision=revision, input_sha256=input_hashes, profiles=specs)
    anchors = None
    if frozen_encoder_features:
        anchor_reference = reuse["reference"]
        context = anchor_reference["metadata"]
        anchor_rows = anchor_reference.get("cases", [])
        if (context.get("role", "document") != "document"
                or context.get("encoder_source_commit", context["source_commit"]) != encoder_source_commit
                or context.get("encoder_reference_sha256") != digest(reference_path)
                or len(anchor_rows) != 6 or {row.get("id") for row in anchor_rows} != signals
                or any(row.get("role", "document") != "document" for row in anchor_rows)):
            raise ValueError("Document anchor must cover all six original encoder cases and provenance")
        anchors = {row["id"]: row for row in anchor_rows}
    core = load_audio_embedding_model(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    token_configuration = json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))
    def token(name):
        value = token_configuration[name]
        return value["content"] if isinstance(value, dict) else value
    processor = SimpleNamespace(**{name: token(name) for name in ("audio_token", "boa_token", "eoa_token")})
    if tokenizer.convert_tokens_to_ids(processor.audio_token) != core.config.audio_token_id:
        raise ValueError("Audio reference token configuration disagrees with model")
    prefixes = json.loads((snapshot / "config_sentence_transformers.json").read_text(encoding="utf-8"))["prompts"]
    if prefixes.get(role, "") != format_text("", role):
        raise ValueError(f"Audio reference {role} prefix differs from the pinned processor")
    profiles, rows, produced, similarities = {}, [], [], []
    for profile, (spec, frame_mask, native_graph, case_inputs) in prepared.items():
        wrapper = export.MultimodalEmbeddingExport(core.language_model, sequence_length,
            image_token_id=core.config.image_token_id, audio_token_id=core.config.audio_token_id,
            image_feature_count=0, audio_feature_count=spec["feature_count"])
        saved_encoder = None if frozen_encoder_features else torch.export.load(native_graph).module()
        expanded = EmbeddingGemma2Processor.replace_audio_token(processor, {"input_features_mask": frame_mask.bool()}, 0)
        formatted = format_text(expanded, role)
        encoded = tokenizer(formatted, padding="max_length", max_length=sequence_length, truncation=False, return_tensors="pt")
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        document_encoded = tokenizer(format_text(expanded, "document"), padding="max_length", max_length=sequence_length,
                                     truncation=False, return_tensors="pt") if frozen_encoder_features else None
        if ids.shape != (1, sequence_length) or int((ids == core.config.audio_token_id).sum()) != spec["feature_count"]:
            raise ValueError("Audio reference prompt exceeds profile or changes placeholders")
        graph = destination / profile / "backbone.pt2"
        graph.parent.mkdir(parents=True, exist_ok=True)
        saved_backbone = None
        if reuse is not None:
            shutil.copyfile(reuse["programs"][profile], graph)
            saved_backbone = torch.export.load(graph).module(); produced.append(graph)
        for original, input_features, golden_features in case_inputs:
            with torch.no_grad():
                anchor_cosine = None
                if frozen_encoder_features:
                    anchor = anchors[original["id"]]
                    document_ids, document_mask = document_encoded["input_ids"], document_encoded["attention_mask"]
                    if (anchor.get("profile") != profile or anchor.get("input_ids") != document_ids[0].tolist()
                            or anchor.get("attention_mask") != document_mask[0].tolist()):
                        raise ValueError("Document anchor prompt or profile changed")
                    document_native = official_forward_from_frozen_features(core, ids=document_ids, mask=document_mask,
                        input_features=input_features, frame_mask=frame_mask.bool(), frozen_features=golden_features)
                    document_vector = torch.nn.functional.normalize(document_native.last_hidden_state[:, document_mask[0].bool()].mean(dim=1), dim=-1)
                    anchor_vector = torch.tensor(anchor["embedding"], dtype=torch.float32).unsqueeze(0)
                    if anchor_vector.shape != document_vector.shape or not torch.isfinite(anchor_vector).all():
                        raise ValueError("Document anchor vector shape or finiteness changed")
                    anchor_cosine = torch.nn.functional.cosine_similarity(document_vector, anchor_vector, dim=-1).min().item()
                    if (anchor_cosine < 0.999999 or not torch.isfinite(document_vector).all()
                            or not torch.allclose(document_vector, anchor_vector, rtol=1e-4, atol=1e-5)):
                        raise ValueError("Document anchor differs from the frozen original encoder / official backbone")
                    native = document_native if role == "document" else official_forward_from_frozen_features(core, ids=ids, mask=mask,
                        input_features=input_features, frame_mask=frame_mask.bool(), frozen_features=golden_features)
                else:
                    native = core(input_ids=ids, attention_mask=mask, input_features=input_features, input_features_mask=frame_mask.bool())
                features = native.audio_hidden_states
                if features.shape != golden_features.shape or not torch.allclose(features, golden_features, rtol=1e-4, atol=1e-5):
                    record_feature_drift(destination, "audio", features, golden_features, role=role, profile=profile,
                                         case=original["id"], source_commit=source_commit, input_source_commit=encoder_source_commit,
                                         saved_encoder=lambda: saved_encoder(input_features))
                    raise ValueError("Audio reference official encoder features changed")
                expected = torch.nn.functional.normalize(native.last_hidden_state[:, mask[0].bool()].mean(dim=1), dim=-1)
                image = torch.empty(0, core.config.text_config.hidden_size)
                wrapper.validate_inputs(ids, mask, image, features)
                if saved_backbone is None:
                    export.export_multimodal_pt2(wrapper, ids, mask, image, features, graph)
                    saved_backbone = torch.export.load(graph).module(); produced.append(graph)
                actual = saved_backbone(ids, mask, image, features)
                cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=-1).min().item()
                if not torch.isfinite(actual).all() or cosine < 0.999999 or not torch.allclose(actual, expected, rtol=1e-4, atol=1e-5):
                    raise ValueError("Saved audio embedding profile differs from the official final vector")
                chain_cosine = None
                if not frozen_encoder_features:
                    encoded_features = saved_encoder(input_features)
                    if encoded_features.shape != features.shape or not torch.allclose(encoded_features, features, rtol=1e-4, atol=1e-5):
                        raise ValueError("Audio reference saved native encoder differs from the official features")
                    chain = saved_backbone(ids, mask, image, encoded_features)
                    chain_cosine = torch.nn.functional.cosine_similarity(chain, expected, dim=-1).min().item()
                    if not torch.isfinite(chain).all() or chain_cosine < 0.999999 or not torch.allclose(chain, expected, rtol=1e-4, atol=1e-5):
                        raise ValueError("Saved audio embedding native chain differs from the official final vector")
                similarities.extend(value for value in (cosine, chain_cosine, anchor_cosine) if value is not None)
            path = destination / profile / (original["id"] + "-features-fp32.bin")
            features.detach().numpy().astype("<f4").tofile(path); produced.append(path)
            rows.append(dict(id=original["id"], profile=profile, input_features=original["input_features"],
                features_file=path.relative_to(destination).as_posix(), input_ids=ids[0].tolist(), attention_mask=mask[0].tolist(),
                formatted_text=formatted, role=role, backbone_cosine=cosine, native_saved_chain_cosine=chain_cosine,
                embedding=expected[0].tolist(), **(dict(document_anchor_cosine=anchor_cosine) if frozen_encoder_features else {})))
        del saved_encoder, saved_backbone, wrapper
        gc.collect()
        profiles[profile] = dict(sequence_length=sequence_length, image_feature_count=0, audio_feature_count=spec["feature_count"],
                                feature_width=core.config.text_config.hidden_size, graph=graph.relative_to(destination).as_posix())
    final_metadata = dict(source_commit=source_commit, encoder_source_commit=encoder_source_commit, role=role,
        model_revision=revision, embedding_dimension=core.config.text_config.embedding_dim,
        image_token_id=core.config.image_token_id, audio_token_id=core.config.audio_token_id,
        vocabulary_size=core.config.text_config.vocab_size, pad_token_id=core.config.text_config.pad_token_id,
        encoder_reference_sha256=digest(reference_path), feature_rtol=1e-4, feature_atol=1e-5,
        input_sha256=input_hashes,
        versions={name: importlib.metadata.version(name) for name in ("torch", "transformers")})
    path = destination / "audio-embedding-reference.json"
    if reuse is not None:
        final_metadata.update(backbone_source_commit=reuse["source_commit"], backbone_reference_sha256=reuse["reference_sha256"])
    if frozen_encoder_features:
        final_metadata["reference_boundary"] = "original_frozen_encoder_outputs_to_official_backbone"
    path.write_text(json.dumps(dict(metadata=final_metadata, profiles=profiles, cases=rows), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    produced.append(path)
    report = dict(success=True, stage="python_audio_embeddings_only", metadata=final_metadata, case_count=len(rows), profile_count=len(profiles),
        minimum_cosine=min(similarities), python_native_saved_chain_validated=not frozen_encoder_features,
        source_encoder_validation_success=manifest["success"],
        onnx_encoder_numerically_validated=manifest.get("onnx_numerically_validated", False),
        onnx_encoder_case_count=manifest.get("onnx_case_count", 0),
        onnx_encoder_failed_case_count=manifest.get("onnx_failed_case_count", 0),
        unity_validated=False, gpu_chain_validated=False, end_to_end_validated=False,
        files={path.relative_to(destination).as_posix(): dict(bytes=path.stat().st_size, sha256=digest(path)) for path in produced})
    if frozen_encoder_features:
        report.update(stage="python_audio_frozen_feature_role_reference", original_document_anchor_cases=len(rows),
                      frozen_encoder_features_validated=True, encoder_execution_validated=False)
    receipt.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report
