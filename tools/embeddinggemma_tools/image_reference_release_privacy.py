"""Exact original image-reference JSON publication; no generic large-file waiver."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

from .privacy import scan_bytes


REVIEW_PATH = 'tools/references/image-reference-release-review.json'
REVIEW_SHA256 = '6644eeabc7ac9721f241d381139148b412e73f4c145601560559d98cfec21864'
MAX_BYTES = 32 * 1024**2


def committed_review(root):
    """Never use untracked, working-tree or staged review authority."""
    raw = subprocess.check_output(['git', '-C', str(root), 'show', 'HEAD:' + REVIEW_PATH],
                                  stderr=subprocess.PIPE, timeout=15)
    if hashlib.sha256(raw).hexdigest() != REVIEW_SHA256:
        raise ValueError('Committed image reference review differs')
    value = json.loads(raw)
    if (value['schema'] != 1 or value['scope'] != 'three-original-image-reference-JSON-only'
            or not isinstance(value['assets'], dict)):
        raise ValueError('Image reference review scope differs')
    return value


def reviewed_image_reference_asset_safe(path, root):
    """Rehash and scan every byte of a committed, exact reviewed JSON asset.

    Hash-bound synthetic numerical values are not a general numerical-data
    privacy certification. This does not approve model files, images or notes.
    """
    try:
        path = Path(path)
        review = committed_review(root)
        if path.name not in review['assets'] or path.suffix != '.json' or path.is_symlink():
            return False
        entry = review['assets'][path.name]
        if (type(entry['bytes']) is not int or not 0 < entry['bytes'] <= MAX_BYTES
                or re.fullmatch('[a-f0-9]{64}', entry['sha256']) is None
                or path.stat().st_size != entry['bytes']):
            return False
        raw = path.read_bytes()
        if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
            return False
        if scan_bytes(raw, path.name):
            return False
        value = json.loads(raw)
        return (value['metadata']['model_revision'] == review['model_revision']
                and value['metadata']['source_commit'] == entry['producer_source'])
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
        return False
