"""Review only exact, committed native-regression distributions without Torch.

Opaque tensors are bound to reviewed source bytes, not text-privacy certified.
This does not permit models in Git or qualify CPU/GPU/release execution.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

from . import exported_program_distribution as exported

POLICY = '.privacy-reviewed-native-regressions.json'
INVENTORY = 'native-regression-assets.json'
CONTRACT = 'tools/references/audio-candidate-e9de158.json'
SELECTION = 'tools/references/selected-audio-generation.json'
CONTRACT_SHA = 'df73e109fc6dd62e8b1fff409930042b63e6c63548865b074b593dcc70c40deb'
PRODUCER = '6bfc2232412f88a78dc23227b5d57ee5a3c7c05c'
SIGNALS = ('silence', 'impulse', 'sine', 'nyquist', 'noise', 'padded_noise')
COHORT = {'candidate-1119.pt2', 'candidate-839.pt2', 'original-reference.json',
          'original-audit.json', 'candidate-reference.json', 'candidate-audit.json'} | {
              'input-'+signal+'.bin' for signal in SIGNALS}


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _json(raw):
    value = exported._json(raw)
    _canonical(value)  # Reject nonfinite JSON numbers, including unquoted NaN.
    return value


def _head_json(cwd, name):
    raw = subprocess.run(['git', '-C', str(cwd), 'show', 'HEAD:'+name],
                         check=True, capture_output=True).stdout
    if len(raw) > 8*1024**2:
        raise ValueError('Review metadata size limit exceeded')
    return _json(raw)


def _metadata(path):
    if path.is_symlink() or path.stat().st_size > 32*1024**2:
        raise ValueError('Metadata size or type differs')
    raw = path.read_bytes(); value = _json(raw); exported._strings(value)
    return raw, value


def _checked_review(root, cwd):
    raw, inventory = _metadata(root/INVENTORY)
    digest = hashlib.sha256(raw).hexdigest()
    policy = _head_json(cwd, POLICY)
    if policy['schema'] != 1:
        raise ValueError('Review policy schema differs')
    record = policy['inventories'][digest]; name = record['review_path']
    if re.fullmatch(r'docs/results/[a-z0-9-]+\.json', name) is None:
        raise ValueError('Review path differs')
    proof = _head_json(cwd, name)
    if (_canonical(proof) != record['review_canonical_sha256'] or proof['schema'] != 1
            or proof['inventory_sha256'] != digest or proof['asset_count'] != 12
            or proof['review'] != 'native-graph-metadata+exact-numeric-source'
            or not all(proof.get(k) is True for k in ('success', 'all_original_numeric_bytes_verified',
                'all_source_and_distribution_graph_hashes_verified', 'all_non_model_members_unchanged',
                'all_graph_semantics_unchanged', 'all_textual_metadata_checked'))
            or not all(proof.get(k) is False for k in ('contains_private_metadata', 'images_in_distribution',
                'opaque_tensor_privacy_certified', 'original_native_pt2_recovered', 'GPU_or_release_acceptance'))):
        raise ValueError('Review binding or scope differs')
    contract = _head_json(cwd, CONTRACT); selection = _head_json(cwd, SELECTION)
    if (inventory['schema'] != 1 or inventory['scope'] != 'selected-native-regression-exact-assets'
            or inventory['candidate_contract_sha256'] != CONTRACT_SHA
            or inventory['selected_generation_canonical_sha256'] != _canonical(selection)
            or selection['candidate_contract_sha256'] != CONTRACT_SHA
            or selection['model_generation'] != contract['generation']
            or inventory['trace_removal_producer'] != PRODUCER
            or set(inventory['files']) != COHORT):
        raise ValueError('Selected source or complete asset cohort differs')
    for name, row in inventory['files'].items():
        if (type(row['bytes']) is not int or not 0 < row['bytes'] <= 2*1024**3
                or re.fullmatch('[a-f0-9]{64}', row['sha256']) is None):
            raise ValueError('Asset hash or size descriptor differs')
        if name.endswith('.pt2'):
            profile = '1119' if '1119' in name else '839'
            source = contract['graphs']['frames-1119-valid-'+profile+'/audio.pt2']
            if (row['kind'] != 'pt2' or row['source_graph_sha256'] != source['sha256']
                    or type(row['member_count']) is not int or not 0 < row['member_count'] <= 2048
                    or any(re.fullmatch('[a-f0-9]{64}',row[k]) is None for k in
                           ('semantic_model_sha256','non_model_member_manifest_sha256'))):
                raise ValueError('Graph provenance differs')
        elif name.endswith('.bin'):
            signal = name.removeprefix('input-').removesuffix('.bin')
            profile = '839' if signal == 'padded_noise' else '1119'
            if (row['kind'] != 'original-input' or row['bytes'] % 4
                    or row['original_member'] != 'frames-1119-valid-'+profile+'/'+signal+'-features-fp32.bin'):
                raise ValueError('Original input mapping differs')
        elif row['kind'] != 'json' or row['bytes'] > 32*1024**2:
            raise ValueError('JSON asset descriptor differs')
    return inventory


def _graph_metadata(path, row):
    with zipfile.ZipFile(path) as archive:
        names, model = exported._members(archive)
        if len(names) != row['member_count'] or sum(v.file_size for v in archive.infolist()) > 2*1024**3:
            return False
        document = exported._model(archive, model)
        cleaned, removed = exported._without_traces(document)
        if removed or hashlib.sha256(exported._canonical(cleaned)).hexdigest() != row['semantic_model_sha256']:
            return False
        exported._text_members(archive, model)
        members = {}
        for info in archive.infolist():
            # Reading every member to EOF verifies each CRC as well as SHA256.
            with archive.open(info) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if info.filename != model:
                members[info.filename] = dict(bytes=info.file_size, sha256=digest)
        return _canonical(members) == row['non_model_member_manifest_sha256']


def reviewed_native_asset_safe(path, cwd):
    """Fail closed for unknown, changed, uncommitted, ambiguous or private bytes."""
    try:
        path = Path(path).absolute()
        if path.name not in COHORT or not path.is_file() or any(v.is_symlink() for v in (path,*path.parents)):
            return False
        inventory = _checked_review(path.parent, cwd); row = inventory['files'][path.name]
        if path.stat().st_size != row['bytes']:
            return False
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']:
                return False
        if row['kind'] == 'pt2':
            return _graph_metadata(path, row)
        if row['kind'] == 'json':
            _metadata(path)
        return True
    except Exception:
        return False
