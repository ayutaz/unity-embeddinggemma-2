from pathlib import Path

import pytest
import yaml

from embeddinggemma_tools.privacy_artifacts import audit


def test_generated_text_and_member_names_are_checked_without_echoing_values(tmp_path):
    private='C:'+'\\'+'Users'+'\\'+'fixture-user'+'\\report'
    (tmp_path/'report.json').write_text(private)
    findings=audit(tmp_path,['report.json'])
    assert findings and findings[0]['rule']=='home-path'
    assert 'fixture-user' not in str(findings)


def test_globs_and_exclusions_match_upload_selection(tmp_path):
    (tmp_path/'safe.txt').write_text('safe')
    (tmp_path/'private.txt').write_text('fixture'+'@'+'private-domain.dev')
    assert audit(tmp_path,['*.txt','!private.txt'])==[]
    assert audit(tmp_path,['*.txt'])


def test_missing_paths_match_optional_upload_but_escape_and_symlink_fail(tmp_path):
    assert audit(tmp_path,['missing/'])==[]
    with pytest.raises(ValueError):audit(tmp_path,['../outside'])


def test_numeric_model_bytes_are_separate_from_metadata_review(tmp_path):
    (tmp_path/'model.onnx').write_bytes(b'\x00\x01\x02')
    (tmp_path/'receipt.json').write_text('{"verified":true}')
    assert audit(tmp_path,['.'])==[]
    (tmp_path/'capture.png').write_bytes(b'not-reviewed')
    assert any(f['rule']=='unreviewed-image' for f in audit(tmp_path,['.']))


def test_every_workflow_upload_requires_matching_successful_privacy_guard():
    root=Path(__file__).resolve().parents[2]
    for path in (root/'.github/workflows').glob('*.yml'):
        workflow=yaml.load(path.read_text(encoding='utf-8'),Loader=yaml.BaseLoader)
        for job in workflow.get('jobs',{}).values():
            steps=job.get('steps',[])
            for index,step in enumerate(steps):
                if step.get('uses','').startswith('actions/upload-artifact@'):
                    guard=steps[index-1]
                    assert guard.get('uses')=='./.github/actions/privacy-artifact-guard',path.name
                    assert guard['with']['paths']==step['with']['path']
                    assert f"steps.{guard['id']}.outcome == 'success'" in step['if']
