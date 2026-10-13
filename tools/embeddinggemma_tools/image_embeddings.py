"""Official real-image final vectors and fixed feature-count backbone exports."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer, EmbeddingGemma2Processor

from . import export
from .model import load_image_embedding_model
from .text import format_text


def prepare_image_embeddings(snapshot, vision_directory, destination, *, sequence_length=1024, revision, source_commit):
    snapshot, vision_directory, destination = Path(snapshot), Path(vision_directory).resolve(), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    receipt = destination/"image-embedding-validation.json"
    receipt.unlink(missing_ok=True)
    if any(not isinstance(value,str) or re.fullmatch(r"[0-9a-f]{40}",value) is None for value in (revision,source_commit)):
        raise ValueError("Image embedding reference requires immutable provenance")
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream,"sha256").hexdigest()
    manifest = json.loads((vision_directory/"vision-validation.json").read_text(encoding="utf-8"))
    reference = json.loads((vision_directory/"vision-reference.json").read_text(encoding="utf-8"))
    for metadata in (manifest["metadata"],reference["metadata"]):
        if metadata["source_commit"] != source_commit or metadata["model_revision"] != revision:
            raise ValueError("Image reference provenance mismatch")
        if set(metadata["input_sha256"]) != {"config.json","processor_config.json","model.safetensors"}:
            raise ValueError("Image reference snapshot hash names are invalid")
        if any(digest(snapshot/name)!=value for name,value in metadata["input_sha256"].items()):
            raise ValueError("Image reference snapshot hash mismatch")
    if not manifest["success"] or manifest["stage"] != "vision_encoder_projector_only":
        raise ValueError("Image reference encoder export did not pass")
    for name,item in manifest["files"].items():
        path=(vision_directory/name).resolve()
        if not path.is_relative_to(vision_directory) or path.stat().st_size!=item["bytes"] or digest(path)!=item["sha256"]:
            raise ValueError("Image reference file hash mismatch")
    def audited_path(name):
        if not isinstance(name,str) or name not in manifest["files"]:
            raise ValueError("Image reference path is not in the audited manifest")
        path=(vision_directory/name).resolve()
        if not path.is_relative_to(vision_directory):
            raise ValueError("Image reference path escaped its root")
        return path
    for profile,spec in reference["profiles"].items():
        if re.fullmatch(r"[a-z][a-z0-9_-]*",profile) is None:
            raise ValueError("Image reference profile name is unsafe")
        audited_path(spec["positions"])
        audited_path(spec["graph"])
    seen=set()
    for row in reference["cases"]:
        if row["profile"] not in reference["profiles"] or re.fullmatch(r"[a-z][a-z0-9_-]*",row["id"]) is None:
            raise ValueError("Image reference case name or profile is invalid")
        if row["id"] in seen:
            raise ValueError("Image reference case IDs must be unique")
        seen.add(row["id"])
        audited_path(row["pixels"])
    core = load_image_embedding_model(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    token_configuration=json.loads((snapshot/"tokenizer_config.json").read_text(encoding="utf-8"))
    def token(name):
        value=token_configuration[name]
        return value["content"] if isinstance(value,dict) else value
    processor=SimpleNamespace(**{name:token(name) for name in ("image_token","boi_token","eoi_token")})
    if tokenizer.convert_tokens_to_ids(processor.image_token) != core.config.image_token_id:
        raise ValueError("Image token configuration disagrees with model")
    prefixes=json.loads((snapshot/"config_sentence_transformers.json").read_text(encoding="utf-8"))["prompts"]
    if prefixes["document"]!=format_text("","document"):
        raise ValueError("Image document prompt differs from the fixed official prefix")
    profiles,rows,produced,similarities,graphs={},[],[],[],{}
    for profile,spec in reference["profiles"].items():
        if re.fullmatch(r"[a-z][a-z0-9_-]*",profile) is None:
            raise ValueError("Image reference profile name is unsafe")
        count=spec["feature_count"]
        wrapper=export.MultimodalEmbeddingExport(core.language_model,sequence_length,
            image_token_id=core.config.image_token_id,audio_token_id=core.config.audio_token_id,
            image_feature_count=count,audio_feature_count=0)
        positions=torch.from_numpy(np.fromfile(audited_path(spec["positions"]),dtype="<i4").copy()).long().reshape(spec["positions_shape"])
        saved_encoder=torch.export.load(audited_path(spec["graph"])).module()
        expanded=EmbeddingGemma2Processor.replace_image_token(processor,{"num_soft_tokens_per_image":[count]},0)
        formatted=format_text(expanded,"document")
        encoded=tokenizer(formatted,padding="max_length",max_length=sequence_length,truncation=False,return_tensors="pt")
        ids,mask=encoded["input_ids"],encoded["attention_mask"]
        if ids.shape!=(1,sequence_length) or int((ids==core.config.image_token_id).sum())!=count:
            raise ValueError("Image prompt exceeds profile or changes placeholder count")
        key=(sequence_length,count,0)
        graph=graphs.get(key,destination/profile/"backbone.pt2")
        exported=torch.export.load(graph).module() if key in graphs else None
        for original in (row for row in reference["cases"] if row["profile"]==profile):
            pixels=torch.from_numpy(np.fromfile(audited_path(original["pixels"]),dtype="<f4").copy()).reshape(spec["pixel_shape"])
            with torch.no_grad():
                native=core(input_ids=ids,attention_mask=mask,pixel_values=pixels,image_position_ids=positions)
                features=native.image_hidden_states
                expected=torch.nn.functional.normalize(native.last_hidden_state[:,mask[0].bool()].mean(dim=1),dim=-1)
                golden_features=torch.tensor(original["features"])
                if features.shape!=golden_features.shape or not torch.allclose(features,golden_features,rtol=1e-4,atol=1e-5):
                    raise ValueError("Official image embedding reference changed the encoder features")
                audio=torch.empty(0,core.config.text_config.hidden_size)
                wrapper.validate_inputs(ids,mask,features,audio)
                if exported is None:
                    export.export_multimodal_pt2(wrapper,ids,mask,features,audio,graph)
                    exported=torch.export.load(graph).module();produced.append(graph)
                    graphs[key]=graph
                actual=exported(ids,mask,features,audio)
                cosine=torch.nn.functional.cosine_similarity(actual,expected,dim=-1).min().item()
                if not torch.isfinite(actual).all() or cosine<0.999999 or not torch.allclose(actual,expected,rtol=1e-4,atol=1e-5):
                    raise ValueError("Saved image embedding profile differs from the official final vector")
                encoded_features=saved_encoder(pixels)
                if encoded_features.shape!=features.shape or not torch.allclose(encoded_features,features,rtol=1e-4,atol=1e-5):
                    raise ValueError("Saved image embedding encoder differs from the official features")
                chain=exported(ids,mask,encoded_features,audio)
                chain_cosine=torch.nn.functional.cosine_similarity(chain,expected,dim=-1).min().item()
                if not torch.isfinite(chain).all() or chain_cosine<0.999999 or not torch.allclose(chain,expected,rtol=1e-4,atol=1e-5):
                    raise ValueError("Saved image embedding chain differs from the official final vector")
                similarities.extend((cosine,chain_cosine))
            feature_file=destination/profile/(original["id"]+"-features-fp32.bin")
            feature_file.parent.mkdir(parents=True,exist_ok=True)
            features.detach().numpy().astype("<f4").tofile(feature_file);produced.append(feature_file)
            rows.append(dict(id=original["id"],profile=profile,pixels=original["pixels"],
                pixels_sha256=digest(audited_path(original["pixels"])),input_ids=ids[0].tolist(),attention_mask=mask[0].tolist(),
                features_file=feature_file.relative_to(destination).as_posix(),formatted_text=formatted,role="document",
                backbone_cosine=cosine,saved_chain_cosine=chain_cosine,
                embedding=expected[0].tolist()))
            del native,features,actual,encoded_features,chain
        profiles[profile]=dict(sequence_length=sequence_length,image_feature_count=count,audio_feature_count=0,
                              feature_width=core.config.text_config.hidden_size,graph=graph.relative_to(destination).as_posix())
        del exported,wrapper,saved_encoder
    if not rows or len(rows)!=manifest["case_count"]:
        raise ValueError("Image reference must cover every encoder case")
    metadata=dict(source_commit=source_commit,model_revision=revision,embedding_dimension=core.config.text_config.embedding_dim,
                  image_token_id=core.config.image_token_id,audio_token_id=core.config.audio_token_id,
                  vocabulary_size=core.config.text_config.vocab_size,pad_token_id=core.config.text_config.pad_token_id,
                  vision_reference_sha256=digest(vision_directory/"vision-reference.json"),
                  input_sha256={name:digest(snapshot/name) for name in ("config.json","processor_config.json","model.safetensors","tokenizer.json","tokenizer_config.json","config_sentence_transformers.json")},
                  versions={name:importlib.metadata.version(name) for name in ("torch","transformers")})
    path=destination/"image-embedding-reference.json"
    path.write_text(json.dumps(dict(metadata=metadata,profiles=profiles,cases=rows),indent=2,allow_nan=False)+"\n",encoding="utf-8");produced.append(path)
    report=dict(success=True,stage="python_image_embeddings_only",unity_validated=False,gpu_chain_validated=False,
                python_saved_chain_validated=True,end_to_end_validated=False,
                metadata=metadata,case_count=len(rows),profile_count=len(profiles),minimum_cosine=min(similarities),
                files={path.relative_to(destination).as_posix():dict(bytes=path.stat().st_size,sha256=digest(path)) for path in produced})
    receipt.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report
