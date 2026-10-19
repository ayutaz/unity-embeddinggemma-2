"""Anonymous exact-byte acquisition of original synthetic image reference JSON."""
import argparse
import hashlib
import json
from pathlib import Path

from .image_reference_cohort import load_image_reference_cohort
from .image_reference_release_privacy import REVIEW_SHA256
from .native_regression_distribution import _download, _empty
from .privacy import scan_bytes


CONFIG_PATH = Path(__file__).parents[1] / 'references/image-reference-distribution.json'
REVIEW_CONFIG_PATH = Path(__file__).parents[1] / 'references/image-reference-release-review.json'
CONFIG_SHA256 = '4d9acf6cf78d36e7667fb291829c12c4fa84140b82e443c58cd210cfa3143b37'
REFERENCE_NAMES = ('vision-reference.json', 'vision-validation.json', 'image-embedding-reference.json')
LEGAL_NAMES = ('LICENSE.txt', 'NOTICE.txt', 'MODIFICATIONS.txt')
BASE_URL = 'https://github.com/ayutaz/unity-embeddinggemma-2/releases/download/image-reference-preview-6644eea/'


def _configuration():
    raw = CONFIG_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError('Pinned image reference acquisition contract differs')
    config = json.loads(raw)
    raw = REVIEW_CONFIG_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REVIEW_SHA256:
        raise ValueError('Pinned original image reference review differs')
    review = json.loads(raw)
    if (config['schema'] != 1 or config['repository'] != 'ayutaz/unity-embeddinggemma-2'
            or config['tag'] != 'image-reference-preview-6644eea'
            or config['review_sha256'] != REVIEW_SHA256
            or set(config['assets']) != set(REFERENCE_NAMES + LEGAL_NAMES)
            or set(review['assets']) != set(REFERENCE_NAMES)):
        raise ValueError('Original reference acquisition scope differs')
    for name, row in config['assets'].items():
        if type(row['bytes']) is not int or not 0 < row['bytes'] <= 32 * 1024**2:
            raise ValueError('Original reference size differs')
        if name in REFERENCE_NAMES and any(row[key] != review['assets'][name][key] for key in ('bytes', 'sha256')):
            raise ValueError('Original reference and release review differ')
    return config


def fetch_image_reference_data(output):
    """Preserve partial evidence on failure; never replace or regenerate goldens."""
    try:
        config = _configuration()
        output = _empty(output)
        output.mkdir(parents=True, exist_ok=True)
        downloads = []
        for name, row in sorted(config['assets'].items()):
            path = output / name
            _download(BASE_URL + name, path, row)
            if scan_bytes(path.read_bytes(), name):
                raise ValueError('Downloaded original reference failed privacy inspection')
            downloads.append(dict(file=name, bytes=row['bytes'], sha256=row['sha256']))
        cohort = load_image_reference_cohort(*(output / name for name in REFERENCE_NAMES))
        if len(cohort.cases) != 6:
            raise ValueError('Incomplete original image reference cohort')
        result = dict(schema=1, transport='github-release-original-image-references-v1',
            configuration_sha256=CONFIG_SHA256, review_sha256=REVIEW_SHA256,
            downloads=downloads, original_cohort_cases=6,
            document_encoder_generation_sha256=cohort.document_encoder_source_sha256,
            model_executed=False, original_golden_changed=False,
            CPU_comparison_qualified=False, complete_performance_accepted=False)
        with (output / 'image-reference-download.json').open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return result
    except (OSError, ValueError, TypeError, KeyError):
        raise ValueError('Original image reference acquisition failed; preserve existing evidence') from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = fetch_image_reference_data(args.output)
    except ValueError:
        print(json.dumps(dict(success=False, error='Original image reference acquisition failed')))
        return 1
    print(json.dumps(dict(success=True, files=len(result['downloads']), model_executed=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
