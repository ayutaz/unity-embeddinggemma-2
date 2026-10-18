import hashlib
import importlib
import json
import zipfile

import pytest


def module():
    return importlib.import_module('embeddinggemma_tools.exported_program_distribution')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def private_path():
    return '/' + 'home' + '/' + 'synthetic' + '/workspace/module.py'


def model():
    return {'graph_module': {'graph': {'nodes': [
        {'target': 'aten.mul.Tensor', 'metadata': {'stack_trace': private_path(), 'val': 'f32[2]'}},
        {'target': 'aten.add.Tensor', 'metadata': {'stack_trace': 'optional debug', 'val': 'f32[2]'}}
    ]}}, 'schema_version': {'major': 1}, 'range_constraints': {}}


def archive(path, document=None, *, weight=b'unchanged weight bytes'):
    with zipfile.ZipFile(path, 'w') as output:
        output.writestr('audio/models/model.json', json.dumps(document or model()))
        output.writestr('audio/data/weights/weight_0', weight)
        output.writestr('audio/data/weights/model_weights_config.json', '{"weight_0": "f32"}')
        output.writestr('audio/version', '1')
    return path


def test_only_optional_node_stack_traces_change_and_source_is_preserved(tmp_path):
    source = archive(tmp_path/'source.pt2'); before = source.read_bytes()
    destination = tmp_path/'distribution.pt2'
    report = module().sanitize(source, destination, expected_source_sha256=digest(source))
    assert source.read_bytes() == before
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(destination) as cleaned:
        expected = model()
        for node in expected['graph_module']['graph']['nodes']:
            del node['metadata']['stack_trace']
        assert json.loads(cleaned.read('audio/models/model.json')) == expected
        for name in original.namelist():
            if name != 'audio/models/model.json':
                assert original.read(name) == cleaned.read(name)
    assert report['source_sha256'] == digest(source)
    assert report['distribution_sha256'] == digest(destination) != digest(source)
    assert report['removed_stack_trace_fields'] == 2
    assert report['unchanged_members_verified'] == 3
    assert report['computational_structure_unchanged'] is True
    assert report['model_executed'] is False
    assert report['gpu_validated'] is False and report['generation_qualified'] is False
    assert private_path() not in json.dumps(report)
    assert report == module().verify(source, destination, expected_source_sha256=digest(source))


@pytest.mark.parametrize('kind', ['home', 'email', 'credential'])
def test_other_private_metadata_fails_closed_without_publishing(tmp_path, kind):
    document = model()
    document['graph_module']['graph']['nodes'][0]['metadata']['other'] = {
        'home': private_path(), 'email': 'person' + '@' + 'mail' + '.com',
        'credential': 'ghp_' + 'a'*30}[kind]
    source = archive(tmp_path/'source.pt2', document); destination = tmp_path/'out.pt2'
    with pytest.raises(ValueError, match='privacy') as error:
        module().sanitize(source, destination, expected_source_sha256=digest(source))
    assert private_path() not in str(error.value)
    assert not destination.exists()


def test_expected_source_hash_is_checked_before_rewriting(tmp_path):
    source = archive(tmp_path/'source.pt2'); destination = tmp_path/'out.pt2'
    with pytest.raises(ValueError, match='hash'):
        module().sanitize(source, destination, expected_source_sha256='0'*64)
    assert not destination.exists()


def test_never_overwrites_source_or_existing_output(tmp_path):
    source = archive(tmp_path/'source.pt2'); before = source.read_bytes()
    with pytest.raises(ValueError, match='overwrite'):
        module().sanitize(source, source, expected_source_sha256=digest(source))
    destination = tmp_path/'existing.pt2'; destination.write_bytes(b'preserve')
    with pytest.raises(ValueError, match='overwrite'):
        module().sanitize(source, destination, expected_source_sha256=digest(source))
    assert destination.read_bytes() == b'preserve' and source.read_bytes() == before


@pytest.mark.parametrize('tamper', ['weight', 'operator', 'missing', 'extra'])
def test_independent_verification_rejects_computation_or_member_changes(tmp_path, tamper):
    source = archive(tmp_path/'source.pt2'); destination = tmp_path/'out.pt2'
    module().sanitize(source, destination, expected_source_sha256=digest(source))
    with zipfile.ZipFile(destination) as content:
        members = {n: content.read(n) for n in content.namelist()}
    if tamper == 'weight': members['audio/data/weights/weight_0'] = b'different weight'
    if tamper == 'operator':
        document = json.loads(members['audio/models/model.json'])
        document['graph_module']['graph']['nodes'][0]['target'] = 'aten.div.Tensor'
        members['audio/models/model.json'] = json.dumps(document).encode()
    if tamper == 'missing': del members['audio/version']
    if tamper == 'extra': members['audio/extra'] = b'extra'
    with zipfile.ZipFile(destination, 'w') as output:
        for name, value in members.items(): output.writestr(name, value)
    with pytest.raises(ValueError):
        module().verify(source, destination, expected_source_sha256=digest(source))


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'audio/duplicate'])
def test_unsafe_or_ambiguous_member_names_are_rejected(tmp_path, name):
    source = archive(tmp_path/'source.pt2')
    with zipfile.ZipFile(source, 'a') as output:
        output.writestr(name, 'first')
        if name == 'audio/duplicate':
            with pytest.warns(UserWarning): output.writestr(name, 'second')
    destination = tmp_path/'out.pt2'
    with pytest.raises(ValueError):
        module().sanitize(source, destination, expected_source_sha256=digest(source))
    assert not destination.exists()


def test_real_tiny_export_load_and_outputs_are_bit_identical(tmp_path):
    import torch
    torch.set_num_threads(2)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.25, -2.5]))
        def forward(self, values):
            return values*self.weight+0.125
    inputs = torch.tensor([[2., -3.], [0., 4.]])
    program = torch.export.export(Tiny(), (inputs,))
    for node in program.graph.nodes:
        if node.op == 'call_function': node.meta['stack_trace'] = private_path()
    source = tmp_path/'audio.pt2'; destination = tmp_path/'clean.pt2'
    torch.export.save(program, source)
    report = module().sanitize(source, destination, expected_source_sha256=digest(source))
    assert report['removed_stack_trace_fields'] > 0
    original = torch.export.load(source).module()
    cleaned = torch.export.load(destination).module()
    with torch.inference_mode():
        assert torch.equal(original(inputs).view(torch.int32), cleaned(inputs).view(torch.int32))
