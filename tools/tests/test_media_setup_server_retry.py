"""Bounded same-origin recovery from a cached server error; keep TLS/hash gates."""
import io
import ssl
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest
from embeddinggemma_tools import media_setup

URL='https://example.invalid/models/'+'a'*64+'.sentis'

@pytest.mark.parametrize('status',[500,502,503,504])
def test_server_error_retries_once_with_fresh_query_and_identical_asset_path(monkeypatch,status):
    calls=[];payload=b'exact-source-bytes'
    error=HTTPError(URL,status,'synthetic server error',{},io.BytesIO(b'error'))
    def fetch(request,*,timeout):
        calls.append((request,timeout))
        if len(calls)==1:raise error
        return io.BytesIO(payload)
    monkeypatch.setattr(media_setup,'urlopen',fetch)
    response=media_setup._download(URL)
    assert response.read()==payload and len(calls)==2
    original=urlsplit(URL);retry=calls[1][0]
    assert urlsplit(retry.full_url)._replace(query='')==original
    assert urlsplit(retry.full_url).query and retry.get_header('Cache-control')=='no-cache'
    assert [timeout for _,timeout in calls]==[60,60]
    assert error.fp is None or error.fp.closed

@pytest.mark.parametrize('status',[400,401,403,404,429])
def test_client_auth_missing_and_rate_limit_failures_are_not_bypassed(monkeypatch,status):
    calls=[]
    def fetch(request,*,timeout):
        calls.append(request);raise HTTPError(URL,status,'synthetic client error',{},None)
    monkeypatch.setattr(media_setup,'urlopen',fetch)
    with pytest.raises(HTTPError) as error:media_setup._download(URL)
    assert error.value.code==status and len(calls)==1

def test_second_server_error_is_propagated_after_two_requests(monkeypatch):
    calls=[]
    def fetch(request,*,timeout):
        calls.append(request);raise HTTPError(URL,500,'synthetic server error',{},None)
    monkeypatch.setattr(media_setup,'urlopen',fetch)
    with pytest.raises(HTTPError):media_setup._download(URL)
    assert len(calls)==2

@pytest.mark.parametrize('error',[KeyboardInterrupt(),ssl.SSLCertVerificationError('synthetic rejected certificate')])
def test_cancellation_and_invalid_certificate_do_not_retry_or_disable_TLS(monkeypatch,error):
    calls=[]
    def fetch(request,*,timeout):calls.append(request);raise error
    monkeypatch.setattr(media_setup,'urlopen',fetch)
    with pytest.raises(type(error)):media_setup._download(URL)
    assert len(calls)==1
