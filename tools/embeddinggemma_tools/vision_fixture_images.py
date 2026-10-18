"""Keep reviewed procedural input PNG bytes stable across image codec versions."""
import io
from pathlib import Path

from PIL import Image


def save_vision_fixture(image, destination):
    # These are input fixtures, not captured results. Only identical decoded
    # pixels may reuse their reviewed encoding; other inputs retain their pixels.
    for candidate in sorted((Path(__file__).parent/'fixtures/vision').glob('*/*.png')):
        raw=candidate.read_bytes()
        with Image.open(io.BytesIO(raw)) as pinned:
            if pinned.mode==image.mode and pinned.size==image.size and pinned.tobytes()==image.tobytes():
                Path(destination).write_bytes(raw)
                return
    image.save(destination)
