"""Generate pinned reference data and validate the saved export before Unity import."""

import hashlib
import importlib.metadata
import json
import re
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .export import MultimodalEmbeddingExport, VisionEncoderExport, export_multimodal_pt2, export_pt2, export_vision_pt2
from .model import load_text_model, load_vision_models
from .reference import build_reference, load_cases, write_reference
from .search import generate_search_reference

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


def load_image_processor(snapshot):
    from transformers import Gemma4ImageProcessor
    configuration = json.loads((Path(snapshot) / "processor_config.json").read_text(encoding="utf-8"))
    settings = configuration["image_processor"]
    if settings.get("image_processor_type") != "Gemma4ImageProcessor":
        raise ValueError("fixed Gemma4 image processor is required")
    return Gemma4ImageProcessor(**{key: value for key, value in settings.items()
                                 if key not in ("image_processor_type", "image_seq_length")})


def prepare_vision(snapshot, destination, *, revision=MODEL_REVISION, source_commit="local", profiles=None, diagnostics=False):
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
        if diagnostics and profile == next(iter(profiles)):
            from .vision_diagnostics import prepare_vision_diagnostics
            diagnostic_directory = destination / "diagnostics"
            diagnostic = prepare_vision_diagnostics(vision, projection, expected_rows[0][0], coords,
                diagnostic_directory, graph, source_commit=source_commit, revision=revision,
                additional_pixels=tuple(item[0] for item in expected_rows[1:]))
            if not diagnostic["success"]:
                raise ValueError("vision diagnostic numerical mismatch or original context mismatch")
            produced.extend(path for path in diagnostic_directory.rglob("*") if path.is_file())
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
