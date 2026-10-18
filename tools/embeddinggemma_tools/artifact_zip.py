"""Read selected artifact ZIP members through bounded byte ranges."""

import hashlib
import io
from pathlib import Path, PurePosixPath
import zipfile
import argparse
import json
import re
import subprocess
import zlib
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen


class RangeReader(io.RawIOBase):
    def __init__(self, size, fetch, *, chunk_bytes=8 * 1024 * 1024, max_parallel=4):
        if type(size) is not int or size <= 0 or type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ValueError("range size/chunk must be positive integers")
        if type(max_parallel) is not int or not 1 <= max_parallel <= 4:
            raise ValueError("range parallelism must be between one and four")
        self.size, self.fetch, self.chunk_bytes, self.position = size, fetch, chunk_bytes, 0
        self.range_bytes, self.range_requests = 0, 0
        self.pool = ThreadPoolExecutor(max_workers=max_parallel)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        origins = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        if whence not in origins or not 0 <= origins[whence] + offset <= self.size:
            raise ValueError("range seek is outside archive")
        self.position = origins[whence] + offset
        return self.position

    def read(self, count=-1):
        if count < -1:
            raise ValueError("range read size is invalid")
        end = self.size if count == -1 else min(self.size, self.position + count)
        ranges = [(start, min(end, start+self.chunk_bytes)) for start in range(self.position, end, self.chunk_bytes)]
        chunks = list(self.pool.map(lambda bounds: self.fetch(*bounds), ranges))
        for (start, stop), raw in zip(ranges, chunks, strict=True):
            if len(raw) != stop - start:
                raise ValueError("range length mismatch")
            self.range_bytes += len(raw)
            self.range_requests += 1
        self.position = end
        return b"".join(chunks)

    def close(self):
        self.pool.shutdown(wait=True)
        super().close()


class _VerifiedLocalReader(io.RawIOBase):
    def __init__(self, stream, size):
        self.stream, self.size = stream, size
        self.range_bytes = self.range_requests = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.stream.tell()

    def seek(self, offset, whence=io.SEEK_SET):
        origins = {io.SEEK_SET: 0, io.SEEK_CUR: self.tell(), io.SEEK_END: self.size}
        if whence not in origins or not 0 <= origins[whence] + offset <= self.size:
            raise ValueError('local archive seek is outside the verified bytes')
        return self.stream.seek(origins[whence] + offset)

    def read(self, count=-1):
        raw = self.stream.read(count)
        self.range_bytes += len(raw); self.range_requests += 1
        return raw

    def close(self):
        self.stream.close()
        super().close()


def verified_local_artifact_reader(path, expected_sha256):
    """Verify the entire preserved archive; never invent fresh provider provenance."""
    if path is None:
        raise ValueError('An explicit verified local archive is required for retired raw sources')
    if not isinstance(expected_sha256, str) or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None:
        raise ValueError('The local archive requires an immutable complete SHA256 pin')
    try:
        stream = Path(path).open('rb')
    except (OSError, TypeError, ValueError):
        raise ValueError('The local archive could not be opened') from None
    try:
        size = stream.seek(0, io.SEEK_END); stream.seek(0)
        if size <= 0 or hashlib.file_digest(stream, 'sha256').hexdigest() != expected_sha256:
            raise ValueError('The complete local archive bytes differ from the immutable pin')
        stream.seek(0)
        return _VerifiedLocalReader(stream, size), dict(transport='verified_local_archive', archive_sha256=expected_sha256,
            archive_bytes=size, provider_metadata_revalidated=False)
    except BaseException:
        stream.close()
        raise


def extract_members(reader, destination, names, *, max_entry_bytes=64 * 1024 * 1024):
    destination = Path(destination).resolve()
    paths = {}
    for name in names:
        if (not isinstance(name, str) or not name or "\\" in name or ":" in name
                or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts):
            raise ValueError("unsafe ZIP member path")
        path = (destination / name).resolve()
        if not path.is_relative_to(destination) or name in paths:
            raise ValueError("unsafe/duplicate ZIP member path")
        paths[name] = path
    if not paths:
        raise ValueError("ZIP selection must contain members")
    files = {}
    with zipfile.ZipFile(reader) as archive:
        infos = []
        for name in paths:
            if archive.namelist().count(name) != 1:
                raise ValueError(f"ZIP member missing or ambiguous: {name}")
            info = archive.getinfo(name)
            if info.is_dir() or not 0 < info.file_size <= max_entry_bytes:
                raise ValueError(f"ZIP member size exceeds selection limit: {name}")
            infos.append(info)
        for info in infos:
            path = paths[info.filename]
            path.parent.mkdir(parents=True, exist_ok=True)
            def existing():
                if not path.is_file() or path.stat().st_size != info.file_size: return None
                crc, digest = 0, hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(32*1024*1024), b""):
                        crc = zlib.crc32(block, crc); digest.update(block)
                return digest.hexdigest() if crc == info.CRC else None
            cached = existing()
            if cached is not None:
                files[info.filename] = dict(bytes=info.file_size, crc32=info.CRC, sha256=cached, cache_reused=True)
                continue
            partial = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
            digest, count = hashlib.sha256(), 0
            try:
                with archive.open(info) as stream, partial.open("wb") as target:
                    for block in iter(lambda: stream.read(32*1024*1024), b""):
                        count += len(block); digest.update(block); target.write(block)
                if count != info.file_size:
                    raise ValueError("ZIP streamed member size mismatch")
                partial.replace(path)  # CRC completed before replacing an existing target.
            finally:
                partial.unlink(missing_ok=True)
            files[info.filename] = dict(bytes=count, crc32=info.CRC, sha256=digest.hexdigest(), cache_reused=False)
    return dict(selected_count=len(files), range_bytes=reader.range_bytes,
                range_requests=reader.range_requests, archive_bytes=reader.size, files=files)


def http_range_fetcher(url, size, *, opener=urlopen):
    def fetch(start, stop):
        if not 0 <= start < stop <= size:
            raise ValueError("HTTP range is outside archive")
        request = Request(url, headers={"Range": f"bytes={start}-{stop - 1}", "User-Agent": "embeddinggemma-tools"})
        with opener(request, timeout=120) as response:
            if (response.status != 206
                    or response.headers.get("Content-Range") != f"bytes {start}-{stop - 1}/{size}"):
                raise ValueError("HTTP range response requires exact 206 Content-Range")
            raw = response.read(stop - start + 1)
            if len(raw) != stop - start:
                raise ValueError("HTTP range body length mismatch")
            return raw
    return fetch


def renewable_http_range_fetcher(url, size, renew, *, opener=urlopen):
    """Renew an expired artifact redirect once per range, retaining exact ranges.

    Concurrent requests that used the same expired URL share one renewal. Other
    HTTP failures and repeated forbidden responses remain failures, not retries.
    GitHub credentials are used only by the fixed artifact redirect callback.
    """
    if not url.startswith('https://'):
        raise ValueError('Artifact redirects require HTTPS')
    lock = threading.Lock()
    current = url
    def fetch(start, stop):
        nonlocal current
        with lock:
            attempted = current
        try:
            return http_range_fetcher(attempted, size, opener=opener)(start, stop)
        except HTTPError as error:
            if error.code != 403:
                raise
            error.close()
        with lock:
            if current == attempted:
                replacement = renew()
                if not replacement.startswith('https://'):
                    raise ValueError('Renewed artifact redirect requires HTTPS')
                current = replacement
            replacement = current
        return http_range_fetcher(replacement, size, opener=opener)(start, stop)
    return fetch


def github_artifact_reader(repo, artifact_id):
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) is None or type(artifact_id) is not int or artifact_id <= 0:
        raise ValueError("GitHub artifact identity is invalid")
    endpoint = f"repos/{repo}/actions/artifacts/{artifact_id}"
    metadata = json.loads(subprocess.run(["gh", "api", endpoint], capture_output=True, text=True, check=True).stdout)
    if metadata.get("id") != artifact_id or metadata.get("expired") is not False:
        raise ValueError("GitHub artifact identity/expiry mismatch")
    token = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip()
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, message, headers, new_url):
            return None
    def redirect():
        request = Request(f"https://api.github.com/{endpoint}/zip", headers={
            "Authorization": f"Bearer {token}", "User-Agent": "embeddinggemma-tools",
            "Cache-Control": "no-cache"})
        try:
            with build_opener(NoRedirect).open(request, timeout=60):
                raise ValueError("GitHub artifact did not provide a signed redirect")
        except HTTPError as error:
            if error.code != 302:
                raise
            signed_url = error.headers.get("Location", "")
            error.close()
        if not signed_url.startswith("https://"):
            raise ValueError("GitHub artifact signed redirect is invalid")
        return signed_url
    size = metadata["size_in_bytes"]
    return RangeReader(size, renewable_http_range_fetcher(redirect(), size, redirect)), metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--artifact-id", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--member", required=True, action="append")
    parser.add_argument("--max-entry-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args(argv)
    reader, metadata = github_artifact_reader(args.repo, args.artifact_id)
    report = extract_members(reader, args.output, args.member, max_entry_bytes=args.max_entry_bytes)
    report.update(artifact_id=metadata["id"], artifact_name=metadata["name"])
    (args.output / "range-receipt.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(artifact_id=metadata["id"], archive_bytes=reader.size,
                         selected_count=report["selected_count"], range_bytes=reader.range_bytes,
                         range_requests=reader.range_requests)))


if __name__ == "__main__":
    main()
