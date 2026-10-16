"""Required original-reference audio gate; fresh CPU drift remains diagnostic.

Execution is the real raw-input, five-part standard ONNX pipeline. The existing
independent auditor verifies captured arrays, original pins, and every carry.
This gate does not promote that CPU export to a Unity consumer or claim GPU proof.
"""
import argparse
import json
from pathlib import Path
import re

from .audit_audio_raw_chain import audit_reports


def validate(paths, original, *, source):
    if re.fullmatch('[0-9a-f]{40}', source) is None:
        raise ValueError('Immutable execution source required')
    result = audit_reports(paths, original, expected_variant='ordered_original_midpoint_bmm')
    if (result['success'] is not True or result['source'] != source
            or result['case_count'] != 6 or result['profile_count'] != 2
            or result['values_compared'] != 824320 or result['candidate_tolerance_failures'] != 0):
        raise ValueError('All six original-reference comparisons at this source must pass')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--reports',type=Path,required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    result=validate(sorted(args.reports.rglob('frozen-audio-chained-onnx.json')),
                    args.original,source=args.source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('success','source','values_compared',
        'candidate_tolerance_failures','native_all_six_context_qualified','gpu_validated')}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
