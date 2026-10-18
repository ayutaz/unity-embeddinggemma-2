"""Select a reviewed audio generation without replacing original numerical goldens.

The pinned evidence establishes Windows audio GPU compatibility only. Each CI
execution still needs all six actual raw-input ONNX captures at its own source.
Neither historical GPU evidence nor this CPU gate proves release acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from . import audio_generation_candidate as candidate

ROOT = Path(__file__).resolve().parents[2]
SELECTION_SHA = '6b1e84c4f9eb0bbbde77ff18f2b1532a4806a8bab42c1ab664beb2f891dc9067'
SELECTION = 'tools/references/selected-audio-generation.json'
PROOFS = {'numeric', 'seed', 'partitions', 'encoder', 'roles', 'saved_fp32', 'saved_float16'}
SIGNALS = {'silence', 'impulse', 'sine', 'nyquist', 'noise', 'padded_noise'}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def require(condition):
    if not condition:
        raise ValueError('Reviewed audio generation evidence is missing, changed or incomplete')


def validate_evidence(selection, evidence):
    contract = candidate.candidate_contract()
    require(selection.get('schema') == 1
            and selection.get('scope') == 'windows_gpu_audio_selection_for_required_original_numeric_regression_only'
            and selection.get('model_generation') == contract['generation']
            and selection.get('candidate_contract_sha256') == candidate.CONTRACT_SHA
            and selection.get('original_reference_sha256') == candidate.REFERENCE_SHA)
    require(all(selection.get(key) is False for key in (
        'original_golden_changed', 'original_native_pt2_recovered', 'full_multimodal_acceptance', 'release_ready')))
    require(set(selection.get('proofs', {})) == set(evidence) == PROOFS)
    require(all(row.get('success') is True and row.get('original_golden_changed') is False
                for row in evidence.values()))
    numeric = evidence['numeric']
    require(numeric['model_generation'] == contract['generation']
            and numeric['candidate_contract_sha256'] == candidate.CONTRACT_SHA
            and numeric['original_reference_sha256'] == candidate.REFERENCE_SHA
            and numeric['case_count'] == 6 and numeric['profile_count'] == 2
            and numeric['values_compared'] == 824320 and numeric['candidate_tolerance_failures'] == 0
            and len(numeric['cases']) == 6 and {row['case_id'] for row in numeric['cases']} == SIGNALS
            and all(row['original_tolerance_passed'] and row['all_capture_and_carry_hashes_verified']
                    and row['native_chain_control_independently_verified']
                    for row in numeric['cases']))
    # The existing capture protocol publishes no intermediate ONNX graphs and
    # explicitly reports graphs_independently_rehashed=False. Keep that limit
    # visible; graph acquisition is hash-checked independently by the producer.
    require(evidence['seed']['source_candidate_contract_sha256'] == candidate.CONTRACT_SHA)
    encoder = evidence['encoder']; roles = evidence['roles']
    require(encoder['preparation_source'] == evidence['partitions']['source']
            and encoder['seed_source'] == evidence['seed']['source']
            and encoder['original_reference_sha256'] == candidate.REFERENCE_SHA
            and encoder['original_audit_sha256'] == candidate.AUDIT_SHA
            and encoder['values'] == 824320 and encoder['signal_count'] == 6 and encoder['profile_count'] == 2
            and encoder['mismatched_elements'] == 0 and encoder['encoder_gpu_validated'] is True
            and encoder['encoder_atol'] == 1e-5 and encoder['encoder_rtol'] == 1e-4
            and all(encoder[key] == 0 for key in (
                'cpu_computation_or_input_count', 'intermediate_readback_count', 'readback_flush_count')))
    require(roles['encoder_gpu_runtime_source'] == encoder['runtime_source']
            and roles['preparation_source'] == encoder['preparation_source']
            and roles['signal_count'] == 6 and roles['profile_count'] == 2
            and roles['final_vector_count'] == 18 and set(roles['roles']) == {'query', 'document', 'raw'}
            and roles['minimum_cosine'] >= .999 and roles['original_minimum_cosine_gate'] == .999)
    fp32 = evidence['saved_fp32']; half = evidence['saved_float16']
    require(fp32['bundle']['descriptor_sha256'] == fp32['player']['descriptor_sha256']
            and fp32['bundle']['runtime_source'] == fp32['player']['bundle_producer_source']
            and fp32['bundle']['partition_preparation_source'] == encoder['preparation_source']
            and fp32['editor']['vectors'] == 6 and fp32['editor']['minimum_cosine'] >= .999)
    prepare = half['preparation']
    require(prepare['source_descriptor_sha256'] == fp32['bundle']['descriptor_sha256']
            and prepare['fp32_bundle_producer'] == fp32['bundle']['runtime_source']
            and prepare['producer_source'] == half['player']['bundle_producer_source']
            and prepare['descriptor_sha256'] == half['player']['descriptor_sha256']
            and prepare['models_bound'] == 285 and prepare['source_objects'] == prepare['derived_objects'] == 434
            and all(prepare[key] is True for key in ('all_original_and_derived_hashes_verified',
                'non_model_bytes_identical', 'module_rewrite_exact', 'source_unchanged',
                'model_description_and_json_privacy_checked', 'second_private_copy_verified'))
            and {row['module'] for row in prepare['module_coverage']} == {'audio-1119', 'audio-839'}
            and all(row['referenced_models'] == 270 and row['short_constants_added'] > 0
                    for row in prepare['module_coverage']))
    for receipt, gate in ((fp32, .999), (half, .99)):
        player = receipt['player']; audit = player if receipt is fp32 else receipt
        require(player['runtime_tree_sha'] == selection['sdk_runtime_tree']
                and player['base_vectors'] == 18 and player['warm_vectors'] == 18
                and player['negative_controls_rejected'] == 24
                and len(player['results']) == 2
                and {row['module'] for row in player['results']} == {'audio-1119', 'audio-839'}
                and all(row['success'] is True and row['minimum_recomputed_cosine'] >= gate
                        for row in player['results'])
                and all(audit[key] is True for key in ('gpu_audits_before_and_after',
                    'no_cpu_computation_or_intermediate_flush', 'session_released',
                    'pre_cancel_verified', 'pending_call_cancel_verified')))


def _runtime_tree(root):
    result = subprocess.run(['git', '-C', str(root), 'rev-parse',
                             'HEAD:Packages/com.ayutaz.embeddinggemma/Runtime'],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def checked_selection(root=None):
    root = Path(root) if root is not None else ROOT
    selection = json.loads((root/SELECTION).read_text(encoding='utf-8-sig'))
    require(canonical_sha(selection) == SELECTION_SHA)
    evidence = {}
    for key, entry in selection['proofs'].items():
        name = entry['path']
        require(re.fullmatch(r'docs/results/[a-z0-9-]+\.json', name) is not None)
        path = root/name
        require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink())
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        require(canonical_sha(data) == entry['canonical_json_sha256'])
        evidence[key] = data
    validate_evidence(selection, evidence)
    require(_runtime_tree(root) == selection['sdk_runtime_tree'])
    return selection


def validate_current_reports(paths, original, *, source):
    selected = checked_selection()
    result = candidate.audit_candidate_reports(paths, original, source=source)
    return _accept_current_result(result,selected,source=source)


def _accept_current_result(result,selected,*,source):
    require(result['success'] is True and result['source'] == source
            and result['model_generation'] == selected['model_generation']
            and result['candidate_contract_sha256'] == selected['candidate_contract_sha256']
            and result['case_count'] == 6 and result['profile_count'] == 2
            and result['values_compared'] == 824320 and result['candidate_tolerance_failures'] == 0
            and result['gpu_validated'] is False and result['generation_qualified'] is False)
    result.update(required_numeric_gate_passed=True, required_selection_canonical_sha256=SELECTION_SHA,
                  scope='selected_generation_current_source_all_six_original_raw_input_numeric_regression_only')
    return result


def validate_current_distribution_reports(paths, original, *, source):
    from . import native_regression_distribution as distribution
    selected = checked_selection()
    result = distribution.audit_distribution_reports(paths,original,source=source)
    config, _ = distribution.configuration()
    require(result.get('graph_transport') == distribution.TRANSPORT
            and result.get('graph_transport_contract_sha256') == distribution.CONFIG_SHA
            and result.get('graph_distribution_inventory_sha256') == config['inventory_sha256'])
    return _accept_current_result(result,selected,source=source)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('fetch','fetch-distribution'):
        fetch = commands.add_parser(name); fetch.add_argument('--output', type=Path, required=True)
        fetch.add_argument('--original', type=Path, required=True); fetch.add_argument('--case', choices=sorted(SIGNALS), required=True)
    for name in ('fetch-original','fetch-original-distribution'):
        golden = commands.add_parser(name); golden.add_argument('--output', type=Path, required=True)
    for name in ('audit','audit-distribution'):
        audit = commands.add_parser(name); audit.add_argument('--reports', type=Path, required=True)
        audit.add_argument('--original', type=Path, required=True); audit.add_argument('--source', required=True)
        audit.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    checked_selection()
    if args.command == 'fetch':
        candidate.fetch_candidate(args.output, args.original, case_id=args.case)
    elif args.command == 'fetch-distribution':
        from .native_regression_distribution import fetch_candidate
        fetch_candidate(args.output,args.original,case_id=args.case)
    elif args.command == 'fetch-original':
        candidate.fetch_original(args.output)
    elif args.command == 'fetch-original-distribution':
        from .native_regression_distribution import fetch_original
        fetch_original(args.output)
    else:
        if args.output.exists(): raise ValueError('Never overwrite an original-regression audit result')
        validate = validate_current_distribution_reports if args.command=='audit-distribution' else validate_current_reports
        result = validate(sorted(args.reports.rglob('frozen-audio-chained-onnx.json')), args.original, source=args.source)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(success=True, selection_canonical_sha256=SELECTION_SHA,
                         gpu_executed_in_this_ci_gate=False, release_ready=False)))
    return 0


if __name__ == '__main__': raise SystemExit(main())
