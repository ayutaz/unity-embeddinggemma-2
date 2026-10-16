import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from embeddinggemma_tools import __main__ as cli


def require_preparation():
    function = getattr(cli, "prepare_prompt_reference", None)
    assert callable(function), "Official multimodal prompt reference preparation is missing."
    return function


def snapshot(path):
    path.mkdir()
    vocabulary = {"<pad>":0,"<bos>":1,"<eos>":2,"[UNK]":3,"hello":4,"then":5,
                  "task:":6,"search":7,"result":8,"query:":9,"<image>":10,"<audio>":11,
                  "<boi>":12,"<eoi>":13,"<boa>":14,"<eoa>":15}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.add_special_tokens(["<pad>","<bos>","<eos>","<image>","<audio>","<boi>","<eoi>","<boa>","<eoa>"])
    backend.post_processor = processors.TemplateProcessing(single="<bos> $A <eos>",special_tokens=[("<bos>",1),("<eos>",2)])
    backend.save(str(path/"tokenizer.json"))
    settings=dict(tokenizer_class="PreTrainedTokenizerFast",pad_token="<pad>",bos_token="<bos>",eos_token="<eos>",
                  image_token="<image>",boi_token="<boi>",eoi_token="<eoi>",audio_token="<audio>",boa_token="<boa>",eoa_token="<eoa>")
    (path/"tokenizer_config.json").write_text(json.dumps(settings))
    (path/"config.json").write_text(json.dumps(dict(image_token_id=10,audio_token_id=11,text_config=dict(vocab_size=32,pad_token_id=0))))
    (path/"config_sentence_transformers.json").write_text(json.dumps(dict(prompts=dict(query="task: search result | query: ",document="title: none | text: "))))
    return path


def test_official_prompt_fixtures_include_real_counts_masks_order_and_hashes(tmp_path):
    prepare = require_preparation()
    result = prepare(snapshot(tmp_path/"snapshot"),tmp_path/"out",revision="a"*40,source_commit="b"*40)
    assert result["success"] is True
    assert result["unity_validated"] is False and result["end_to_end_validated"] is False
    reference=json.loads((tmp_path/"out/prompt-reference.json").read_text(encoding="utf-8"))
    rows={row["id"]:row for row in reference["cases"]}
    assert rows["square"]["input_ids"].count(10)==256
    assert rows["landscape"]["input_ids"].count(10)==252
    assert rows["audio-5"]["input_ids"].count(11)==2
    assert rows["audio-hole"]["input_ids"].count(11)==2
    assert rows["image-audio"]["input_ids"].index(10)<rows["image-audio"]["input_ids"].index(11)
    assert rows["audio-image"]["input_ids"].index(11)<rows["audio-image"]["input_ids"].index(10)
    for row in rows.values():
        assert len(row["input_ids"])==len(row["attention_mask"])==1024
    assert set(reference["metadata"]["input_sha256"])=={"config.json","tokenizer.json","tokenizer_config.json","config_sentence_transformers.json"}
    for name,item in result["files"].items():
        assert (tmp_path/"out"/name).stat().st_size==item["bytes"]
        assert len(item["sha256"])==64


@pytest.mark.parametrize("field",["revision","source_commit"])
def test_mutable_provenance_cannot_leave_stale_success(tmp_path,field):
    prepare=require_preparation()
    output=tmp_path/"out";output.mkdir();receipt=output/"prompt-validation.json";receipt.write_text('{"success":true}')
    kwargs=dict(revision="a"*40,source_commit="b"*40);kwargs[field]="main"
    with pytest.raises(ValueError,match="immutable"):
        prepare(tmp_path/"missing",output,**kwargs)
    assert not receipt.exists()


def test_cli_downloads_only_prompt_assets_without_model_weights(tmp_path,monkeypatch,capsys):
    require_preparation()
    downloads=[];calls=[]
    def download(*args,**kwargs):
        downloads.append(kwargs);return str(tmp_path/"snapshot")
    def prepare(*args,**kwargs):
        calls.append(kwargs)
        return dict(success=True,stage="multimodal_prompt_tokens_only",case_count=14,unity_validated=False,end_to_end_validated=False)
    monkeypatch.setattr(cli,"snapshot_download",download)
    monkeypatch.setattr(cli,"prepare_prompt_reference",prepare)
    monkeypatch.setenv("GITHUB_SHA","c"*40)
    cli.main(["prepare-prompts","--output",str(tmp_path/"out")])
    assert set(downloads[0]["allow_patterns"])=={"config.json","tokenizer.json","tokenizer_config.json","config_sentence_transformers.json"}
    assert calls[0]["source_commit"]=="c"*40
    assert json.loads(capsys.readouterr().out)["unity_validated"] is False
