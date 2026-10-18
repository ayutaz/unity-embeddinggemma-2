from pathlib import Path
import yaml
from workflow_privacy_assertions import assert_privacy_gated_upload


def test_layer_four_capture_is_separate_opt_in_and_keeps_other_reference_modes_separate():
    root=Path(__file__).resolve().parents[2]/'.github/workflows'
    parent=yaml.safe_load((root/'reference-context.yml').read_text(encoding='utf-8'))
    inputs=parent.get('on',parent.get(True))['workflow_dispatch']['inputs']
    assert 'original_layer_four_detail' in inputs,'Registered layer-four detail opt-in is missing'
    assert inputs['original_layer_four_detail']['default'] is False
    assert parent['jobs']['original_layer_four_probe']['uses']=='./.github/workflows/vision-layer-four-context.yml'
    for name in ('original_vision_probe','original_rotary_probe','original_static_matmul_probe','original_ordered_context_probe'):
        assert '!inputs.original_layer_four_detail' in parent['jobs'][name]['if']


def test_layer_four_capture_uses_qualified_fixed_hidden_interval_and_never_reruns_full_encoder():
    path=Path(__file__).resolve().parents[2]/'.github/workflows/vision-layer-four-context.yml'
    assert path.is_file(),'Fixed-hidden fourth-layer capture workflow is missing'
    value=yaml.safe_load(path.read_text(encoding='utf-8'));commands='\n'.join(s.get('run','') for s in value['jobs']['probe']['steps'])
    for text in ('11975007068','d17722473eca4a82c92e45f1952043598eba8f964bfbfd3a4f67c74b17bdcece',
                 'extract_saved_graph_interval',"'add_43'","'add_57'",'VisionSoftmax16Diagnostic','row_index=173',
                 "'maximum_absolute_error'",'rms_operations','full_encoder_executed=False'):
        assert text in commands
    assert 'snapshot_download' not in commands and 'replay_saved_reference_context' not in commands
    upload=next(s for s in value['jobs']['probe']['steps'] if 'actions/upload-artifact@' in s.get('uses',''))
    assert_privacy_gated_upload(upload)
    assert 'row-captures/' in upload['with']['path']
    assert 'full-captures/' not in upload['with']['path'] and '.pt2' not in upload['with']['path']
