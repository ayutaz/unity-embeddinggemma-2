from pathlib import Path
from xml.etree import ElementTree as ET

from embeddinggemma_tools.privacy import scan_bytes

pytest_plugins=['pytester']


def run_report(pytester, source):
    pytester.syspathinsert(Path(__file__).parents[1])
    pytester.makepyfile(test_cases=source)
    result=pytester.runpytest('--noconftest','-p','embeddinggemma_tools.pytest_privacy',
                             '-q','--junitxml=report.xml')
    return result,(pytester.path/'report.xml').read_bytes()


def test_skip_location_is_relative_without_changing_status_or_reason(pytester):
    result,raw=run_report(pytester,'''import pytest
def test_pass(): pass
def test_device(): pytest.skip('device unavailable; not a GPU pass')
''')
    result.assert_outcomes(passed=1,skipped=1)
    xml=ET.fromstring(raw);skips=xml.findall('.//skipped')
    assert len(skips)==1 and skips[0].get('message')=='device unavailable; not a GPU pass'
    assert 'test_cases.py:' in skips[0].text
    assert str(pytester.path) not in raw.decode()
    assert not scan_bytes(raw,'report.xml')


def test_private_skip_reason_is_preserved_and_still_blocked(pytester):
    reason='C:'+'\\'+'Users'+'\\'+'fixture-user'+'\\'+'private'
    result,raw=run_report(pytester,'import pytest\ndef test_private_reason(): pytest.skip('+repr(reason)+')\n')
    result.assert_outcomes(skipped=1)
    xml=ET.fromstring(raw)
    assert xml.find('.//skipped').get('message')==reason
    assert any(f['rule']=='home-path' for f in scan_bytes(raw,'report.xml'))
