"""Retaining exports requires identical producer inputs and a clean executor."""
import subprocess
import pytest
from embeddinggemma_tools.audio_cpu_encoder import _qualify_retained_graph_producer

@pytest.fixture
def repository(tmp_path):
    def git(*args):return subprocess.check_output(['git',*args],cwd=tmp_path,stderr=subprocess.PIPE).decode().strip()
    git('init','-q');git('config','user.name','ayutaz');git('config','user.email','41669061+ayutaz@users.noreply.github.com')
    files={'tools/embeddinggemma_tools/frozen_audio_chained_onnx.py':'original producer\n',
        'tools/embeddinggemma_tools/audio_cpu_reference.py':'old observation\n',
        'tools/embeddinggemma_tools/audio_cpu_encoder.py':'old admission\n',
        'tools/references/fixed.json':'{"original":true}\n','tools/pyproject.toml':'[project]\n','tools/uv.lock':'version = 1\n'}
    for name,raw in files.items():
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(raw,encoding='utf-8')
    git('add','.');git('commit','-qm','original producer inputs');producer=git('rev-parse','HEAD')
    def changed(name,raw):
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(raw,encoding='utf-8')
        git('add','.');git('commit','-qm','qualified executor candidate');return git('rev-parse','HEAD')
    return tmp_path,git,producer,changed

def test_documentation_and_separate_observers_do_not_require_reexport(repository):
    root,git,producer,change=repository
    current=change('tools/embeddinggemma_tools/audio_cpu_reference.py','durable snapshots\n')
    current=change('tools/embeddinggemma_tools/native_gpu_memory.py','independent observation\n')
    current=change('docs/note.md','new documentation\n')
    result=_qualify_retained_graph_producer(producer,current,root=root)
    assert result['producer_source']==producer and result['execution_source']==current
    assert result['producer_inputs_identical'] is True
    assert result['graph_producer_reused'] is True and len(result['producer_inputs_sha256'])==64
    assert result['new_export_or_full_performance_accepted'] is False

@pytest.mark.parametrize('name',['cpu_fp32_reference.py','native_audio_cpu_encoder.py'])
def test_explicit_native_CPU_consumers_are_not_graph_generator_inputs(repository,name):
    root,git,producer,change=repository
    current=change('tools/embeddinggemma_tools/'+name,'separate explicit CPU consumer\n')
    assert _qualify_retained_graph_producer(producer,current,root=root)['producer_inputs_identical'] is True

@pytest.mark.parametrize('path',['tools/embeddinggemma_tools/frozen_audio_chained_onnx.py','tools/references/fixed.json','tools/uv.lock','tools/pyproject.toml','tools/embeddinggemma_tools/new_producer.py'])
def test_any_changed_or_added_producer_input_is_rejected(repository,path):
    root,git,producer,change=repository
    current=change(path,'different producer input\n')
    with pytest.raises(ValueError):_qualify_retained_graph_producer(producer,current,root=root)

@pytest.mark.parametrize('kind',['tracked','staged','untracked'])
def test_dirty_executor_cannot_claim_a_frozen_execution_source(repository,kind):
    root,git,producer,change=repository
    current=change('docs/note.md','candidate\n')
    path=root/('untracked.py' if kind=='untracked' else 'tools/embeddinggemma_tools/audio_cpu_reference.py')
    path.write_text('uncommitted changes\n',encoding='utf-8')
    if kind=='staged':git('add','.')
    with pytest.raises(ValueError):_qualify_retained_graph_producer(producer,current,root=root)

def test_unknown_producer_or_noncurrent_execution_commit_is_rejected(repository):
    root,git,producer,change=repository
    current=change('docs/note.md','candidate\n')
    with pytest.raises(ValueError):_qualify_retained_graph_producer('f'*40,current,root=root)
    with pytest.raises(ValueError):_qualify_retained_graph_producer(producer,'0'*40,root=root)

@pytest.mark.parametrize('bad',['HEAD','a'*39,'A'*40,None])
def test_only_full_immutable_sources_are_accepted(repository,bad):
    root,git,producer,change=repository
    with pytest.raises(ValueError):_qualify_retained_graph_producer(bad,producer,root=root)
