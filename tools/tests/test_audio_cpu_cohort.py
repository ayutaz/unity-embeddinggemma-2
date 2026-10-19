import hashlib
import importlib
import json

import numpy as np
import pytest


def implementation():
    return importlib.import_module('embeddinggemma_tools.audio_cpu_cohort')


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    module = implementation()
    pre_root, enc_root = tmp_path/'pre', tmp_path/'encoder'
    pre_root.mkdir()
    enc_root.mkdir()
    snapshot = {'config.json':'1'*64,'processor_config.json':'2'*64,'model.safetensors':'3'*64}
    pre_meta = dict(model_revision=module.MODEL_REVISION, source_commit='a'*40,
        fixture_provenance='procedurally_generated_pcm_no_user_audio', num_samples=179200,
        sampling_rate=16000, feature_shape=[1,1119,128], frame_mask_shape=[1,1119],
        processor_sha256=snapshot['processor_config.json'], feature_atol=.004, feature_rtol=1e-4)
    enc_meta = dict(model_revision=module.MODEL_REVISION, source_commit='b'*40,
        dtype='float32', input_sha256=snapshot, feature_atol=1e-5, feature_rtol=1e-4)
    final_meta = dict(model_revision=module.MODEL_REVISION, source_commit='c'*40,
        role='document', embedding_dimension=768, input_sha256={**snapshot,'tokenizer.json':'4'*64})
    pre = dict(metadata=pre_meta, cases=[])
    pre_audit = dict(success=True, stage='audio_preprocessing_only', metadata=pre_meta, files={})
    encoder = dict(metadata=enc_meta, cases=[], profiles={})
    enc_audit = dict(success=False, native_reference_complete=True, metadata=enc_meta, files={})
    final = dict(metadata=final_meta, cases=[], profiles={})
    def binary(root, audit, name, values):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = values.tobytes()
        path.write_bytes(raw)
        audit['files'][name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    for case_id in ('silence','impulse','sine','nyquist','noise','padded_noise'):
        valid_samples, frames, tokens = (134400,839,210) if case_id == 'padded_noise' else (179200,1119,280)
        profile = f'frames-1119-valid-{frames}'
        wave = np.zeros((1,179200), dtype='<f4')
        sample_mask = (np.arange(179200)[None] < valid_samples).astype('<i4')
        features = np.zeros((1,1119,128), dtype='<f4')
        frame_mask = (np.arange(1119)[None] < frames).astype('<i4')
        row = dict(id=case_id, valid_samples=valid_samples, waveform=case_id+'-pcm.bin',
            sample_mask=case_id+'-samples.bin', features=case_id+'-features.bin', frame_mask=case_id+'-frames.bin')
        for field, values in [('waveform',wave),('sample_mask',sample_mask),('features',features),('frame_mask',frame_mask)]:
            binary(pre_root,pre_audit,row[field],values)
        input_name = profile+'/'+case_id+'-features.bin'
        binary(enc_root,enc_audit,input_name,features)
        mask_name = profile+'/mask.bin'
        binary(enc_root,enc_audit,mask_name,frame_mask)
        pre['cases'].append(row)
        encoder['profiles'][profile] = dict(feature_count=tokens, feature_width=512, frame_mask=mask_name)
        encoder['cases'].append(dict(id=case_id,profile=profile,input_features=input_name,features=[[0.0]*512]*tokens))
        final['profiles'][profile] = dict(sequence_length=4,audio_feature_count=tokens,image_feature_count=0,feature_width=512)
        final['cases'].append(dict(id=case_id, profile=profile, role='document',
            input_ids=[1,2,3,0],attention_mask=[1,1,1,0],embedding=[1.0]+[0.0]*767))
    paths = [pre_root/'audio-reference.json', pre_root/'audio-validation.json',
             enc_root/'audio-encoder-reference.json', enc_root/'audio-encoder-validation.json', tmp_path/'document.json']
    def save():
        raw = json.dumps(pre).encode()
        paths[0].write_bytes(raw)
        pre_sha = hashlib.sha256(raw).hexdigest()
        pre_audit['files']['audio-reference.json'] = dict(bytes=len(raw),sha256=pre_sha)
        enc_meta['preprocessing_reference_sha256'] = pre_sha
        paths[2].write_bytes(json.dumps(encoder).encode())
        final_meta['encoder_reference_sha256'] = hashlib.sha256(paths[2].read_bytes()).hexdigest()
        for path,value in [(paths[1],pre_audit),(paths[3],enc_audit),(paths[4],final)]:
            path.write_bytes(json.dumps(value).encode())
        for constant,path in zip(('PREPROCESS_REFERENCE_SHA256','PREPROCESS_AUDIT_SHA256',
            'ENCODER_REFERENCE_SHA256','ENCODER_AUDIT_SHA256','DOCUMENT_REFERENCE_SHA256'), paths,strict=True):
            monkeypatch.setattr(module,constant,hashlib.sha256(path.read_bytes()).hexdigest())
        return pre_root,enc_root,paths[4]
    return module,pre,pre_audit,encoder,enc_audit,final,save


def test_load_actual_pcm_masks_and_both_fixed_cpu_cases_as_immutable_arrays(cohort):
    module,_,_,_,_,_,save = cohort
    result = module.load_audio_cpu_reference_cohort(*save())
    assert [v.id for v in result.cases] == ['silence','padded_noise']
    assert dict(result.snapshot_input_sha256)['processor_config.json'] == '2'*64
    for case_id,tokens,frames in [('silence',280,1119),('padded_noise',210,839)]:
        case = result.case(case_id)
        assert case.waveform.shape == case.sample_mask.shape == (1,179200)
        assert case.preprocessing_features.shape == (1,1119,128)
        assert case.frame_mask.shape == (1,1119) and int(case.frame_mask.sum()) == frames
        assert case.encoder_features.shape == (tokens,512) and case.embedding.shape == (768,)
        assert case.input_ids == (1,2,3,0) and case.attention_mask == (1,1,1,0)
        for array in (case.waveform,case.sample_mask,case.preprocessing_features,case.frame_mask,case.encoder_features,case.embedding):
            assert not array.flags.writeable
            with pytest.raises(ValueError): array.setflags(write=True)
    with pytest.raises(ValueError): result.case('unknown')


@pytest.mark.parametrize('which',range(5))
def test_any_changed_original_metadata_bytes_fail_closed(cohort,which):
    module,_,_,_,_,_,save = cohort
    pre,enc,document = save()
    paths = [pre/'audio-reference.json',pre/'audio-validation.json',enc/'audio-encoder-reference.json',enc/'audio-encoder-validation.json',document]
    with paths[which].open('ab') as stream: stream.write(b' ')
    with pytest.raises(ValueError): module.load_audio_cpu_reference_cohort(pre,enc,document)


@pytest.mark.parametrize('field',['waveform','sample_mask','features','frame_mask'])
def test_each_actual_input_buffer_is_checked_against_the_original_manifest(cohort,field):
    module,pre,_,_,_,_,save = cohort
    roots = save()
    path = roots[0]/pre['cases'][0][field]
    raw = bytearray(path.read_bytes())
    raw[0] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError): module.load_audio_cpu_reference_cohort(*roots)


@pytest.mark.parametrize('mutation',['revision','generation','processor','profile','tokens','mask','partial','feature_gate','pre_gate'])
def test_semantic_generation_and_case_contracts_are_not_replaced_by_hash_patching(cohort,mutation):
    module,pre,_,encoder,_,document,save = cohort
    if mutation == 'revision': document['metadata']['model_revision'] = 'f'*40
    if mutation == 'generation':
        roots = save()
        document['metadata']['encoder_reference_sha256'] = '0'*64
        roots[2].write_bytes(json.dumps(document).encode())
        module.DOCUMENT_REFERENCE_SHA256 = hashlib.sha256(roots[2].read_bytes()).hexdigest()
        with pytest.raises(ValueError): module.load_audio_cpu_reference_cohort(*roots)
        return
    if mutation == 'processor': pre['metadata']['processor_sha256'] = '0'*64
    if mutation == 'profile': document['cases'][0]['profile'] = 'frames-1119-valid-839'
    if mutation == 'tokens': document['cases'][0]['input_ids'][0] = True
    if mutation == 'mask': document['cases'][0]['attention_mask'] = [0]*4
    if mutation == 'partial': pre['cases'].pop()
    if mutation == 'feature_gate': encoder['metadata']['feature_atol'] = .01
    if mutation == 'pre_gate': pre['metadata']['feature_atol'] = .01
    roots = save()
    with pytest.raises(ValueError): module.load_audio_cpu_reference_cohort(*roots)


def test_original_document_snapshot_does_not_have_to_repeat_unused_processor_config(cohort):
    module,_,_,_,_,document,save = cohort
    document['metadata']['input_sha256'].pop('processor_config.json')
    result = module.load_audio_cpu_reference_cohort(*save())
    assert dict(result.snapshot_input_sha256)['processor_config.json'] == '2'*64


def test_original_document_role_is_explicit_in_each_case_when_not_repeated_in_metadata(cohort):
    module,_,_,_,_,document,save = cohort
    document['metadata'].pop('role')
    assert len(module.load_audio_cpu_reference_cohort(*save()).cases) == 2


def test_missing_metadata_role_cannot_hide_a_non_document_case(cohort):
    module,_,_,_,_,document,save = cohort
    document['metadata'].pop('role')
    document['cases'][2]['role'] = 'query'
    with pytest.raises(ValueError): module.load_audio_cpu_reference_cohort(*save())
