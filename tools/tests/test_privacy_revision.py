from embeddinggemma_tools import privacy


def test_explicit_history_revision_checks_all_source_ancestors(monkeypatch,tmp_path):
    checked=[]
    monkeypatch.setattr(privacy,'reviews',lambda root:{})
    monkeypatch.setattr(privacy,'scan_objects',lambda root,revisions,reviewed:checked.extend(revisions) or [])
    assert privacy.main(['history','--root',str(tmp_path),'--revision','a'*40])==0
    assert checked==['a'*40]


def test_history_still_defaults_to_all_refs(monkeypatch,tmp_path):
    checked=[]
    monkeypatch.setattr(privacy,'reviews',lambda root:{})
    monkeypatch.setattr(privacy,'scan_objects',lambda root,revisions,reviewed:checked.extend(revisions) or [])
    assert privacy.main(['history','--root',str(tmp_path)])==0
    assert checked==['--all']
