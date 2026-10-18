from concurrent.futures import ThreadPoolExecutor
import threading
from urllib.error import HTTPError

import pytest

from embeddinggemma_tools import artifact_zip
from test_artifact_zip_http_cli import Response


def factory():
    return getattr(artifact_zip, 'renewable_http_range_fetcher')


def test_expired_redirect_renews_only_the_failed_range():
    calls=[]; renewals=[]
    def open_request(request, timeout):
        calls.append((request.full_url,request.get_header('Range')))
        if request.full_url.endswith('/old'):
            raise HTTPError(request.full_url,403,'expired',{},None)
        return Response(b'abc',206,'bytes 2-4/10')
    def renew(): renewals.append(True); return 'https://example.test/new'
    fetch=factory()('https://example.test/old',10,renew,opener=open_request)
    assert fetch(2,5)==b'abc'
    assert calls==[('https://example.test/old','bytes=2-4'),('https://example.test/new','bytes=2-4')]
    assert len(renewals)==1
    assert fetch(2,5)==b'abc' and len(renewals)==1


def test_parallel_expiry_has_one_renewal_for_the_same_generation():
    gate=threading.Barrier(4); renewals=[]
    def open_request(request, timeout):
        if request.full_url.endswith('/old'):
            gate.wait(timeout=5)
            raise HTTPError(request.full_url,403,'expired',{},None)
        return Response(b'abc',206,'bytes 2-4/10')
    def renew(): renewals.append(True); return 'https://example.test/new'
    fetch=factory()('https://example.test/old',10,renew,opener=open_request)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _:fetch(2,5),range(4)))==[b'abc']*4
    assert len(renewals)==1


@pytest.mark.parametrize('status',[401,404,500])
def test_non_expiry_http_failures_are_not_hidden(status):
    def open_request(request,timeout): raise HTTPError(request.full_url,status,'failure',{},None)
    fetch=factory()('https://example.test/old',10,lambda:pytest.fail('Unexpected renewal'),opener=open_request)
    with pytest.raises(HTTPError) as error: fetch(2,5)
    assert error.value.code==status


def test_repeated_forbidden_is_bounded_to_one_retry_per_range():
    calls=[];renewals=[]
    def open_request(request,timeout):
        calls.append(True);raise HTTPError(request.full_url,403,'forbidden',{},None)
    def renew(): renewals.append(True);return 'https://example.test/new'
    fetch=factory()('https://example.test/old',10,renew,opener=open_request)
    with pytest.raises(HTTPError): fetch(2,5)
    assert len(calls)==2 and len(renewals)==1


def test_renewed_response_still_requires_the_exact_range():
    def open_request(request,timeout):
        if request.full_url.endswith('/old'): raise HTTPError(request.full_url,403,'expired',{},None)
        return Response(b'abc',200,'bytes 2-4/10')
    fetch=factory()('https://example.test/old',10,lambda:'https://example.test/new',opener=open_request)
    with pytest.raises(ValueError,match='HTTP range'): fetch(2,5)


def test_insecure_renewal_is_rejected_before_network_access():
    calls=[]
    def open_request(request,timeout):
        calls.append(True);raise HTTPError(request.full_url,403,'expired',{},None)
    fetch=factory()('https://example.test/old',10,lambda:'http://example.test/insecure',opener=open_request)
    with pytest.raises(ValueError,match='HTTPS'): fetch(2,5)
    assert len(calls)==1


def test_github_renewal_keeps_the_artifact_identity_and_never_sends_auth_to_blob(monkeypatch):
    import json
    from types import SimpleNamespace
    calls=[];redirects=[];blobs=[]
    def run(arguments,**kwargs):
        calls.append(arguments)
        if arguments[1:]==['auth','token']: return SimpleNamespace(stdout='synthetic')
        assert arguments[1:]==['api','repos/owner/repo/actions/artifacts/17']
        return SimpleNamespace(stdout=json.dumps(dict(id=17,expired=False,size_in_bytes=10)))
    class Redirect:
        def open(self,request,timeout):
            redirects.append(request)
            assert request.full_url=='https://api.github.com/repos/owner/repo/actions/artifacts/17/zip'
            assert request.get_header('Authorization')=='Bearer synthetic'
            assert request.get_header('Cache-control')=='no-cache'
            location='https://example.test/'+('old' if len(redirects)==1 else 'new')
            raise HTTPError(request.full_url,302,'redirect',{'Location':location},None)
    def blob(request,timeout):
        blobs.append(request)
        assert request.get_header('Authorization') is None
        if request.full_url.endswith('/old'): raise HTTPError(request.full_url,403,'expired',{},None)
        return Response(b'abc',206,'bytes 2-4/10')
    original=artifact_zip.http_range_fetcher
    monkeypatch.setattr(artifact_zip.subprocess,'run',run)
    monkeypatch.setattr(artifact_zip,'build_opener',lambda *_:Redirect())
    monkeypatch.setattr(artifact_zip,'http_range_fetcher',lambda url,size,**_:original(url,size,opener=blob))
    reader,metadata=artifact_zip.github_artifact_reader('owner/repo',17)
    with reader:
        reader.seek(2)
        assert reader.read(3)==b'abc'
    assert metadata['id']==17 and len(redirects)==2 and len(blobs)==2 and len(calls)==2
