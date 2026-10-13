"""Bounded HTTP range reads for selectively extracting large validation artifacts."""

from collections import OrderedDict
import io
from pathlib import Path, PurePosixPath
import shutil
import zipfile


class RangeReader(io.RawIOBase):
    def __init__(self, fetch, size, block_size=8 * 1024 * 1024):
        self.fetch, self.size, self.block_size = fetch, size, block_size
        self.position, self.fetched_bytes, self.cache = 0, 0, OrderedDict()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self.position = offset + (0 if whence == io.SEEK_SET else self.position if whence == io.SEEK_CUR else self.size)
        if self.position < 0:
            raise ValueError("negative range position")
        return self.position

    def read(self, size=-1):
        stop = self.size if size < 0 else min(self.size, self.position + size)
        result = bytearray()
        while self.position < stop:
            start = self.position // self.block_size * self.block_size
            end = min(self.size, start + self.block_size)
            if start not in self.cache:
                data = self.fetch(start, end)
                if len(data) != end - start:
                    raise OSError("incomplete HTTP range response")
                self.cache[start] = data
                self.fetched_bytes += len(data)
                if len(self.cache) > 2:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(start)
            count = min(stop - self.position, end - self.position)
            result.extend(self.cache[start][self.position - start:self.position - start + count])
            self.position += count
        return bytes(result)


def extract_subset(reader, destination, prefix):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(reader) as archive:
        for item in archive.infolist():
            if not item.filename.startswith(prefix) or item.is_dir():
                continue
            relative = PurePosixPath(item.filename[len(prefix):])
            target = (destination / relative).resolve()
            if relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(destination):
                raise ValueError("artifact path escaped destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            with archive.open(item) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            partial.replace(target)
            extracted.append(relative.as_posix())
    return extracted
