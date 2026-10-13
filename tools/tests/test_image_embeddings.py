import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import EmbeddingGemma2Model

from embeddinggemma_tools import model as loader, __main__ as cli, prepare as vision_preparation
from test_model import tiny_model
from test_vision import positions, vision_models
from test_vision_prepare import TinyProcessor
from test_prompt_reference import snapshot as tokenizer_snapshot


def required(owner, name):
    function=getattr(owner,name,None)
    assert callable(function), f"Image embedding reference contract is missing: {name}"
    return function


def snapshot(path, omit=None):
    tokenizer_snapshot(path)
    text=tiny_model(); vision,projection=vision_models()
    config=dict(text_config=text.config.to_dict(),vision_config=vision.config.to_dict(),image_token_id=10,audio_token_id=11)
    (path/"config.json").write_text(json.dumps(config))
    (path/"processor_config.json").write_text('{}')
    state={}
    for prefix,module in (("language_model.",text),("vision_tower.",vision),("embed_vision.",projection)):
        state.update({prefix+key:value.bfloat16().contiguous() for key,value in module.state_dict().items() if prefix+key!=omit})
    state["audio_tower.must_not_load"]=torch.ones(1)
    save_file(state,path/"model.safetensors")
    return path


def test_loader_uses_the_official_image_forward_without_constructing_audio(tmp_path):
    load=required(loader,"load_image_embedding_model")
    core=load(snapshot(tmp_path/"snapshot"))
    assert isinstance(core,EmbeddingGemma2Model)
    assert core.audio_tower is None and core.embed_audio is None
    assert not core.training
    assert all(p.dtype==torch.float32 and p.device.type=="cpu" for p in core.parameters())
    pixels=torch.linspace(0,1,216).reshape(1,18,12)
    ids=torch.tensor([[1,12,10,13,2,0,0,0]])
    mask=torch.tensor([[1,1,1,1,1,0,0,0]])
    result=core(input_ids=ids,attention_mask=mask,pixel_values=pixels,image_position_ids=positions(width=3))
    assert result.image_hidden_states.shape==(1,16)
    assert result.last_hidden_state.shape==(1,8,24)


def test_missing_image_projection_weight_is_not_hidden_by_partial_loading(tmp_path):
    load=required(loader,"load_image_embedding_model")
    with pytest.raises(RuntimeError,match="embedding_projection"):
        load(snapshot(tmp_path/"snapshot",omit="embed_vision.embedding_projection.weight"))


def prepare_source(tmp_path,monkeypatch):
    source=snapshot(tmp_path/"snapshot")
    monkeypatch.setattr(vision_preparation,"load_image_processor",lambda _:TinyProcessor())
    vision_preparation.prepare_vision(source,tmp_path/"vision",revision="a"*40,source_commit="b"*40,profiles={"square":(12,12)})
    return source,tmp_path/"vision"


def test_real_image_features_and_final_vectors_match_official_forward_and_saved_profile(tmp_path,monkeypatch):
    prepare=required(cli,"prepare_image_embeddings")
    source,vision=prepare_source(tmp_path,monkeypatch)
    report=prepare(source,vision,tmp_path/"out",sequence_length=16,revision="a"*40,source_commit="b"*40)
    assert report["success"] is True and report["stage"]=="python_image_embeddings_only"
    assert report["unity_validated"] is False and report["gpu_chain_validated"] is False
    reference=json.loads((tmp_path/"out/image-embedding-reference.json").read_text(encoding="utf-8"))
    assert reference["profiles"]["square"]["image_feature_count"]==1
    assert reference["profiles"]["square"]["audio_feature_count"]==0
    core=required(loader,"load_image_embedding_model")(source)
    graph=torch.export.load(tmp_path/"out/square/backbone.pt2").module()
    for row in reference["cases"]:
        ids=torch.tensor([row["input_ids"]]);mask=torch.tensor([row["attention_mask"]])
        pixels=torch.from_numpy(__import__('numpy').fromfile(vision/row["pixels"],dtype='<f4').copy()).reshape(1,18,12)
        with torch.no_grad():
            native=core(input_ids=ids,attention_mask=mask,pixel_values=pixels,image_position_ids=positions(width=3))
            expected=torch.nn.functional.normalize(native.last_hidden_state[:,mask[0].bool()].mean(dim=1),dim=-1)[0]
            actual=graph(ids,mask,native.image_hidden_states,torch.empty(0,16))[0]
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(torch.tensor(row["embedding"]),expected,rtol=1e-5,atol=1e-6)
    assert set(report["files"])>={"image-embedding-reference.json","square/backbone.pt2"}
    for name,item in report["files"].items():
        assert hashlib.sha256((tmp_path/"out"/name).read_bytes()).hexdigest()==item["sha256"]


@pytest.mark.parametrize("problem",["source","revision","hash"])
def test_reference_provenance_mismatch_fails_before_model_load_or_stale_pass(tmp_path,monkeypatch,problem):
    prepare=required(cli,"prepare_image_embeddings")
    source,vision=prepare_source(tmp_path,monkeypatch)
    manifest_path=vision/"vision-validation.json"
    manifest=json.loads(manifest_path.read_text())
    if problem=="hash":
        (vision/"square/0-pixels-fp32.bin").write_bytes(b"corrupt")
    else:
        key="source_commit" if problem=="source" else "model_revision"
        manifest["metadata"][key]="c"*40;manifest_path.write_text(json.dumps(manifest))
    output=tmp_path/"out";output.mkdir();receipt=output/"image-embedding-validation.json";receipt.write_text('{"success":true}')
    def fail(_): pytest.fail("The model must not load before reference provenance audit")
    import importlib
    monkeypatch.setattr(importlib.import_module(prepare.__module__),"load_image_embedding_model",fail)
    with pytest.raises(ValueError,match="reference"):
        prepare(source,vision,output,sequence_length=16,revision="a"*40,source_commit="b"*40)
    assert not receipt.exists()


def test_wrong_saved_profile_final_is_rejected(tmp_path,monkeypatch):
    prepare=required(cli,"prepare_image_embeddings")
    source,vision=prepare_source(tmp_path,monkeypatch)
    from embeddinggemma_tools import export
    original=export.export_multimodal_pt2
    def corrupt(wrapper,ids,mask,image,audio,destination):
        original(wrapper,ids,mask,image,audio,destination)
        program=torch.export.load(destination)
        with torch.no_grad(): program.state_dict["model.embedding_projection.weight"].zero_()
        torch.export.save(program,destination)
    monkeypatch.setattr(export,"export_multimodal_pt2",corrupt)
    with pytest.raises(ValueError,match="embedding"):
        prepare(source,vision,tmp_path/"out",sequence_length=16,revision="a"*40,source_commit="b"*40)
    assert not (tmp_path/"out/image-embedding-validation.json").exists()


def test_cli_uses_fixed_vision_source_and_model_revision(tmp_path,monkeypatch,capsys):
    required(cli,"prepare_image_embeddings")
    calls=[]
    def prepare(*args,**kwargs):
        calls.append((args,kwargs));return dict(success=True,stage="python_image_embeddings_only",case_count=6,profile_count=3,unity_validated=False,gpu_chain_validated=False)
    monkeypatch.setattr(cli,"prepare_image_embeddings",prepare)
    monkeypatch.setenv("GITHUB_SHA","b"*40)
    cli.main(["prepare-image-embeddings","--snapshot",str(tmp_path/"snapshot"),"--vision-source",str(tmp_path/"vision"),"--output",str(tmp_path/"out")])
    assert calls[0][0][1]==tmp_path/"vision"
    assert calls[0][1]["source_commit"]=="b"*40
    assert calls[0][1]["sequence_length"]==1024
    assert json.loads(capsys.readouterr().out)["unity_validated"] is False


def test_equal_profile_shapes_share_graph_without_dropping_image_cases(tmp_path,monkeypatch):
    prepare=required(cli,"prepare_image_embeddings")
    source=snapshot(tmp_path/"snapshot")
    monkeypatch.setattr(vision_preparation,"load_image_processor",lambda _:TinyProcessor())
    vision=tmp_path/"vision"
    vision_preparation.prepare_vision(source,vision,revision="a"*40,source_commit="b"*40,
                                     profiles={"square":(12,12),"landscape":(12,20),"portrait":(20,12)})
    report=prepare(source,vision,tmp_path/"out",sequence_length=16,revision="a"*40,source_commit="b"*40)
    reference=json.loads((tmp_path/"out/image-embedding-reference.json").read_text(encoding="utf-8"))
    assert report["case_count"]==6 and report["profile_count"]==3
    assert len({item["graph"] for item in reference["profiles"].values()})==1,"Equal static feature profiles must not duplicate the same text weights."
    assert len(list((tmp_path/"out").rglob('*.pt2')))==1


def test_saved_encoder_is_checked_in_the_final_embedding_chain(tmp_path,monkeypatch):
    prepare=required(cli,"prepare_image_embeddings")
    source,vision=prepare_source(tmp_path,monkeypatch)
    graph=vision/"square/vision.pt2"
    program=torch.export.load(graph)
    with torch.no_grad():program.state_dict["projection.embedding_projection.weight"].zero_()
    torch.export.save(program,graph)
    manifest_path=vision/"vision-validation.json"
    manifest=json.loads(manifest_path.read_text())
    manifest["files"]["square/vision.pt2"]=dict(bytes=graph.stat().st_size,sha256=hashlib.sha256(graph.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="embedding"):
        prepare(source,vision,tmp_path/"out",sequence_length=16,revision="a"*40,source_commit="b"*40)
    assert not (tmp_path/"out/image-embedding-validation.json").exists()


@pytest.mark.parametrize("problem",["case_pixel_path","snapshot_hash_key","profile_positions_path","case_id_path","duplicate_case_id"])
def test_unlisted_paths_are_rejected_before_loading_the_model(tmp_path,monkeypatch,problem):
    prepare=required(cli,"prepare_image_embeddings")
    source,vision=prepare_source(tmp_path,monkeypatch)
    outside=tmp_path/"outside.bin";outside.write_bytes(b"test fixture only")
    ref_path=vision/"vision-reference.json";manifest_path=vision/"vision-validation.json"
    reference=json.loads(ref_path.read_text());manifest=json.loads(manifest_path.read_text())
    if problem=="case_pixel_path":
        reference["cases"][0]["pixels"]="../outside.bin"
    elif problem=="profile_positions_path":
        reference["profiles"]["square"]["positions"]="../outside.bin"
    elif problem=="case_id_path":
        reference["cases"][0]["id"]="../outside"
    elif problem=="duplicate_case_id":
        reference["cases"][1]["id"]=reference["cases"][0]["id"]
    else:
        for metadata in (reference["metadata"],manifest["metadata"]):
            metadata["input_sha256"]["../outside.bin"]=hashlib.sha256(outside.read_bytes()).hexdigest()
    ref_path.write_text(json.dumps(reference))
    manifest["files"]["vision-reference.json"]=dict(bytes=ref_path.stat().st_size,sha256=hashlib.sha256(ref_path.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest))
    def fail(_):pytest.fail("Unlisted reference paths must be rejected before model loading")
    import importlib
    monkeypatch.setattr(importlib.import_module(prepare.__module__),"load_image_embedding_model",fail)
    with pytest.raises(ValueError,match="reference"):
        prepare(source,vision,tmp_path/"out",sequence_length=16,revision="a"*40,source_commit="b"*40)
