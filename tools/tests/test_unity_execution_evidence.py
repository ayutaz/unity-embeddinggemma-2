import pytest
from embeddinggemma_tools import bounded_process


def required():
    function = getattr(bounded_process, 'read_unity_execution_counts', None)
    assert callable(function), 'Unity exit zero with no executed tests must not pass'
    return function


@pytest.mark.parametrize('total,passed,failed', [(1,1,0), (1,0,1)])
def test_unity_counts_distinguish_real_execution_from_numeric_success(tmp_path,total,passed,failed):
    path=tmp_path/'result.xml'
    path.write_text(f'<test-run total="{total}" passed="{passed}" failed="{failed}" skipped="0"/>')
    assert required()(path) == dict(total=total,passed=passed,failed=failed,skipped=0)


@pytest.mark.parametrize('xml', [
    '<test-run total="0" passed="0" failed="0" skipped="0" result="Passed"/>',
    '<test-run total="1" passed="0" failed="0" skipped="1"/>',
    '<test-run total="2" passed="1" failed="0" skipped="0"/>',
])
def test_unity_counts_reject_missing_skipped_or_inconsistent_execution(tmp_path,xml):
    path=tmp_path/'result.xml';path.write_text(xml)
    with pytest.raises(ValueError): required()(path)


@pytest.mark.parametrize('total,passed,failed,success', [(0,0,0,False),(1,1,0,True),(1,0,1,False)])
def test_owned_runner_audits_unity_xml_even_when_process_exits_zero(tmp_path,monkeypatch,total,passed,failed,success):
    class Process:
        pid=123
        returncode=0
        def wait(self,timeout): return 0
        def poll(self): return self.returncode
    monkeypatch.setattr(bounded_process.subprocess,'Popen',lambda *args,**kwargs:Process())
    xml=tmp_path/'unity.xml'
    xml.write_text(f'<test-run total="{total}" passed="{passed}" failed="{failed}" skipped="0"/>')
    report=bounded_process.run_bounded(['owned-unity'],tmp_path/'process.json',unity_results=xml)
    assert report['terminal'] and report['exit_code']==0
    assert report['success'] is success
