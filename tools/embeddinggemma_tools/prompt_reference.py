"""Weights-free official processor/tokenizer references for multimodal prompts."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
from types import SimpleNamespace

import numpy as np
from transformers import AutoTokenizer, EmbeddingGemma2Processor

from .text import format_text


def prepare_prompt_reference(snapshot, destination, *, revision, source_commit):
    snapshot, destination = Path(snapshot), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    receipt = destination / "prompt-validation.json"
    receipt.unlink(missing_ok=True)
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None for value in (revision, source_commit)):
        raise ValueError("Prompt references require immutable source and model SHAs")
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "config_sentence_transformers.json")
    settings = json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    prefixes = json.loads((snapshot / "config_sentence_transformers.json").read_text(encoding="utf-8"))["prompts"]
    if prefixes["query"] != format_text("", "query") or prefixes["document"] != format_text("", "document"):
        raise ValueError("Official query/document prompts differ from the fixed text contract")
    def token(name):
        value = settings[name]
        return value["content"] if isinstance(value, dict) else value
    processor = SimpleNamespace(**{name: token(name) for name in ("image_token", "boi_token", "eoi_token", "audio_token", "boa_token", "eoa_token")})
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    image_id, audio_id = config["image_token_id"], config["audio_token_id"]
    if tokenizer.convert_tokens_to_ids(processor.image_token) != image_id or tokenizer.convert_tokens_to_ids(processor.audio_token) != audio_id:
        raise ValueError("Processor markers and model placeholder IDs disagree")
    cases = [
        dict(id="square", text=processor.image_token, image_feature_count=256, role="raw"),
        dict(id="landscape", text=processor.image_token, image_feature_count=252, role="document"),
        dict(id="portrait", text=processor.image_token, image_feature_count=252, role="document", title="猫の写真"),
        *(dict(id=f"audio-{frames}", text=processor.audio_token, audio_feature_mask=[1]*frames, role="raw") for frames in (1,4,5,8,9,17,2999)),
        dict(id="audio-hole", text=processor.audio_token, audio_feature_mask=[1,1,1,1,0,0,0,0,1], role="raw"),
        dict(id="image-audio", text=processor.image_token+" then "+processor.audio_token, image_feature_count=256, audio_feature_mask=[1]*17, role="raw"),
        dict(id="audio-image", text=processor.audio_token+" then "+processor.image_token, image_feature_count=252, audio_feature_mask=[1]*17, role="raw"),
        dict(id="text-query", text="音声と画像を探す", role="query"),
        dict(id="empty-raw", text="", role="raw"),
    ]
    for case in cases:
        text = case["text"]
        case.setdefault("image_feature_count", 0)
        case.setdefault("audio_feature_mask", None)
        case.setdefault("title", None)
        if case["image_feature_count"]:
            replacement = EmbeddingGemma2Processor.replace_image_token(processor, {"num_soft_tokens_per_image": [case["image_feature_count"]]}, 0)
            text = text.replace(processor.image_token, replacement)
        case["audio_feature_count"] = 0
        if case["audio_feature_mask"] is not None:
            replacement = EmbeddingGemma2Processor.replace_audio_token(processor, {"input_features_mask": [np.asarray(case["audio_feature_mask"],dtype=np.int32)]}, 0)
            case["audio_feature_count"] = replacement.count(processor.audio_token)
            text = text.replace(processor.audio_token, replacement)
        text = format_text(text, case["role"], case["title"])
        encoded = tokenizer(text, padding="max_length", max_length=1024, truncation=False)
        if len(encoded["input_ids"]) != 1024 or len(encoded["attention_mask"]) != 1024:
            raise ValueError("Official prompt exceeds the fixed sequence budget")
        if encoded["input_ids"].count(image_id) != case["image_feature_count"] or encoded["input_ids"].count(audio_id) != case["audio_feature_count"]:
            raise ValueError("Official tokenization changed the feature placeholder count")
        case.update(sequence_length=1024, formatted_text=text, input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream,"sha256").hexdigest()
    metadata = dict(source_commit=source_commit, model_revision=revision, stage="multimodal_prompt_tokens_only",
                    image_token_id=image_id, audio_token_id=audio_id, vocabulary_size=config["text_config"]["vocab_size"],
                    pad_token_id=config["text_config"]["pad_token_id"], input_sha256={name:digest(snapshot/name) for name in names},
                    versions={name:importlib.metadata.version(name) for name in ("transformers","tokenizers")})
    produced = []
    for name in names:
        target = destination/name; shutil.copyfile(snapshot/name,target); produced.append(target)
    reference = destination/"prompt-reference.json"
    reference.write_text(json.dumps(dict(metadata=metadata,cases=cases),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    produced.append(reference)
    report = dict(success=True,stage=metadata["stage"],metadata=metadata,case_count=len(cases),unity_validated=False,
                  end_to_end_validated=False,encoders_validated=False,backbone_validated=False,
                  files={path.name:dict(bytes=path.stat().st_size,sha256=digest(path)) for path in produced})
    receipt.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report
