"""Pinned public native-regression transport; numerical qualification is separate."""
from contextlib import closing
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlsplit
from urllib.request import urlopen
import uuid

from .native_regression_release_privacy import COHORT, _json, _metadata, _graph_metadata

CONFIG_PATH = Path(__file__).parents[1]/'references/native-regression-release.json'
INVENTORY_PATH = Path(__file__).parents[1]/'references/native-regression-distribution.json'
CONFIG_SHA = '947430cb584490077a900afed6d1fcc1415c08493ee7efea3e2727fa46fd6fb8'
TRANSPORT = 'github-release-native-regression-v1'
LEGAL = {'LICENSE.txt', 'NOTICE.txt', 'MODIFICATIONS.txt', 'native-regression-assets.json'}
SIGNALS = {'silence', 'impulse', 'sine', 'nyquist', 'noise', 'padded_noise'}


def _open(url):
    return urlopen(url, timeout=60)


def configuration():
    raw = CONFIG_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA:
        raise ValueError('Pinned release transport contract differs')
    config = _json(raw); raw = INVENTORY_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config['inventory_sha256']:
        raise ValueError('Pinned distribution inventory differs')
    inventory = _json(raw)
    if (config['schema'] != 1 or config['repository'] != 'ayutaz/unity-embeddinggemma-2'
            or config['tag'] != 'native-regression-preview-e9de158'
            or type(config['release_id']) is not int or config['release_id'] <= 0
            or re.fullmatch('[a-f0-9]{40}', config['publication_target']) is None
            or inventory['schema'] != 1 or inventory['scope'] != 'selected-native-regression-exact-assets'
            or set(inventory['files']) != COHORT or set(config['assets']) != COHORT | LEGAL):
        raise ValueError('Release source or exact asset cohort differs')
    for name, row in config['assets'].items():
        if (type(row['bytes']) is not int or not 0 < row['bytes'] <= 2*1024**3
                or type(row['asset_id']) is not int or row['asset_id'] <= 0
                or re.fullmatch('[a-f0-9]{64}', row['sha256']) is None):
            raise ValueError('Pinned asset descriptor differs')
        if name in inventory['files'] and any(row[k] != inventory['files'][name][k] for k in ('bytes','sha256')):
            raise ValueError('Asset and distribution identities differ')
    if (config['assets']['native-regression-assets.json']['sha256'] != config['inventory_sha256']
            or config['assets']['native-regression-assets.json']['bytes'] != len(raw)):
        raise ValueError('Public inventory asset differs')
    return config, inventory


def _empty(output):
    output = Path(output).absolute()
    if (any(p.is_symlink() for p in (output,*output.parents))
            or output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError('Download output must be an empty owned directory without symlinks')
    return output


def _relative(name):
    if (not name or '\\' in name or ':' in name or PurePosixPath(name).is_absolute()
            or any(p in ('','.','..') for p in name.split('/'))):
        raise ValueError('Canonical relative download mapping required')
    return name


def _download(url, destination, entry):
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name('.download-'+uuid.uuid4().hex+'.part')
    try:
        with closing(_open(url)) as response, pending.open('xb') as stream:
            final = urlsplit(response.geturl())
            if (final.scheme != 'https' or final.hostname not in
                    {'github.com','release-assets.githubusercontent.com','objects.githubusercontent.com'}):
                raise ValueError('Public asset redirect differs')
            length = response.headers.get('Content-Length')
            if length is not None and int(length) != entry['bytes']:
                raise ValueError('Public asset response length differs')
            count = 0; digest = hashlib.sha256()
            while raw := response.read(1024*1024):
                count += len(raw)
                if count > entry['bytes']:
                    raise ValueError('Public asset exceeds its exact size')
                stream.write(raw); digest.update(raw)
            if count != entry['bytes'] or digest.hexdigest() != entry['sha256']:
                raise ValueError('Public asset bytes differ')
        # An exclusive hard link publishes only fully verified bytes, never replaces evidence.
        destination.hardlink_to(pending)
    finally:
        pending.unlink(missing_ok=True)


def _fetch(output, mappings, config, inventory):
    output.mkdir(parents=True, exist_ok=True); rows = []
    base = 'https://github.com/'+config['repository']+'/releases/download/'+config['tag']+'/'
    for asset, relative in sorted(mappings.items()):
        path = output/_relative(relative); entry = config['assets'][asset]
        _download(base+asset, path, entry)
        if asset.endswith('.json'):
            _metadata(path)
        elif asset.endswith('.pt2'):
            if not _graph_metadata(path, inventory['files'][asset]):
                raise ValueError('Pinned distribution graph metadata differs')
        rows.append(dict(asset=asset, path=relative, bytes=entry['bytes'], sha256=entry['sha256']))
    return rows


def _receipt(output, name, rows, config, **extra):
    value = dict(schema=1, transport=TRANSPORT, release_id=config['release_id'],
                 release_contract_sha256=CONFIG_SHA, inventory_sha256=config['inventory_sha256'],
                 downloads=rows, graph_executed=False, original_golden_changed=False, **extra)
    with (output/name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    return value


def fetch_original(output):
    config, inventory = configuration(); output = _empty(output)
    mappings = {name:name for name in LEGAL}
    mappings.update({'original-reference.json':'audio-encoder-reference.json',
                     'original-audit.json':'audio-encoder-validation.json'})
    rows = _fetch(output, mappings, config, inventory)
    return _receipt(output, 'original-range-receipt.json', rows, config)


def fetch_candidate(output, original, *, case_id):
    if case_id not in SIGNALS:
        raise ValueError('Exactly one original case is required')
    config, inventory = configuration(); output = _empty(output); original = _empty(original)
    if output.resolve().is_relative_to(original.resolve()) or original.resolve().is_relative_to(output.resolve()):
        raise ValueError('Candidate and original evidence directories must be disjoint')
    fetch_original(original)
    profile = 'frames-1119-valid-'+('839' if case_id == 'padded_noise' else '1119')
    asset = 'candidate-'+('839' if case_id == 'padded_noise' else '1119')+'.pt2'
    graph = profile+'/audio.pt2'; input_name = profile+'/'+case_id+'-features-fp32.bin'
    if inventory['files']['input-'+case_id+'.bin']['original_member'] != input_name:
        raise ValueError('Original input mapping differs')
    mappings = {name:name for name in LEGAL}
    mappings.update({'candidate-reference.json':'audio-runtime-reference/audio-encoder-reference.json',
                     'candidate-audit.json':'audio-runtime-reference/audio-encoder-validation.json',
                     asset:'audio-runtime-reference/'+graph,
                     'input-'+case_id+'.bin':'audio-runtime-reference/'+input_name})
    rows = _fetch(output, mappings, config, inventory); entry = inventory['files'][asset]
    marker = dict(schema=1, transport=TRANSPORT, case_id=case_id, profile=profile, graph_path=graph,
                  source_graph_sha256=entry['source_graph_sha256'], executed_graph_sha256=entry['sha256'],
                  inventory_sha256=config['inventory_sha256'], release_contract_sha256=CONFIG_SHA,
                  distribution_producer=inventory['trace_removal_producer'])
    with (output/'audio-runtime-reference/native-distribution-source.json').open('x', encoding='utf-8') as stream:
        json.dump(marker, stream, indent=2); stream.write('\n')
    return _receipt(output, 'candidate-range-receipt.json', rows, config, selected_case=case_id)


def load_distribution_input(directory, original, *, case_id):
    """Load only an explicitly selected, hash/metadata-verified distribution graph."""
    import numpy as np
    import torch
    from . import audio_generation_candidate as candidate
    config, inventory = configuration()
    if inventory['candidate_contract_sha256'] != candidate.CONTRACT_SHA:
        raise ValueError('Distribution candidate generation differs')
    root, raw, spec, row, source_graph, provenance = candidate._candidate_metadata(
        directory, original, case_id=case_id)
    entries = [entry for name, entry in inventory['files'].items()
               if name.endswith('.pt2') and entry['source_graph_sha256'] == source_graph['sha256']]
    if len(entries) != 1:
        raise ValueError('Exactly one source-bound distribution graph is required')
    entry = entries[0]
    expected_marker = dict(schema=1, transport=TRANSPORT, case_id=case_id, profile=row['profile'],
        graph_path=spec['native_graph'], source_graph_sha256=source_graph['sha256'],
        executed_graph_sha256=entry['sha256'], inventory_sha256=config['inventory_sha256'],
        release_contract_sha256=CONFIG_SHA, distribution_producer=inventory['trace_removal_producer'])
    try:
        _, marker = _metadata(root/'native-distribution-source.json')
    except (OSError, ValueError):
        raise ValueError('Explicit distribution source marker is missing or invalid') from None
    if marker != expected_marker:
        raise ValueError('Explicit distribution source marker differs')
    program_path = candidate.checked_file(root, spec['native_graph'], entry)
    if not _graph_metadata(program_path, entry):
        raise ValueError('Executed distribution graph metadata differs')
    values = torch.from_numpy(np.fromfile(raw,dtype='<f4').reshape(spec['feature_shape']).copy())
    expected = torch.tensor(row['features'],dtype=torch.float32)
    program = torch.export.load(program_path).module()
    if sorted(node.args[1][0] for node in candidate.checked_normalizations(program)) != [32,128]:
        raise ValueError('Distribution graph normalization widths differ')
    provenance.update(source_graph_sha256=source_graph['sha256'], graph_sha256=entry['sha256'],
        executed_graph_sha256=entry['sha256'], graph_transport=TRANSPORT,
        graph_transport_contract_sha256=CONFIG_SHA, graph_distribution_inventory_sha256=config['inventory_sha256'],
        graph_distribution_producer=inventory['trace_removal_producer'])
    return program, values, expected, provenance


def audit_distribution_report(path, original):
    from . import audio_generation_candidate as candidate
    config, inventory = configuration()
    if inventory['candidate_contract_sha256'] != candidate.CONTRACT_SHA:
        raise ValueError('Distribution audit generation differs')
    _, report = _metadata(Path(path))
    source_sha = report.get('provenance',{}).get('source_graph_sha256')
    entries = [entry for name,entry in inventory['files'].items()
               if name.endswith('.pt2') and entry['source_graph_sha256'] == source_sha]
    if len(entries) != 1:
        raise ValueError('Exactly one pinned distribution graph audit identity is required')
    expected = dict(entries[0], transport=TRANSPORT, transport_contract_sha256=CONFIG_SHA,
                    inventory_sha256=config['inventory_sha256'],
                    distribution_producer=inventory['trace_removal_producer'])
    return candidate._audit_candidate_report(path,original,execution_graph=expected)


def audit_distribution_reports(paths, original, *, source):
    from . import audio_generation_candidate as candidate
    config, _ = configuration()
    result = candidate._aggregate_candidate_reports(paths,original,source=source,
                                                   audit_one=audit_distribution_report)
    if any(row.get('graph_transport') != TRANSPORT or row.get('graph_transport_contract_sha256') != CONFIG_SHA
           or row.get('graph_distribution_inventory_sha256') != config['inventory_sha256'] for row in result['cases']):
        raise ValueError('Every actual case must use the same pinned distribution transport')
    result.update(scope='distribution_candidate_all_six_original_raw_input_connected_onnx_numeric_capture_audit_only',
                  graph_transport=TRANSPORT,graph_transport_contract_sha256=CONFIG_SHA,
                  graph_distribution_inventory_sha256=config['inventory_sha256'])
    return result
