import importlib
import json

import pytest


def implementation():
    return importlib.import_module('embeddinggemma_tools.cpu_execution_context')


def test_records_cpu_model_and_selected_math_context_without_unrelated_machine_or_user_fields(tmp_path):
    module = implementation()
    cpu = tmp_path / 'cpuinfo'
    cpu.write_text('processor : 0\nvendor_id : GenuineIntel\ncpu family : 6\nmodel : 173\n'
                   'model name : Intel(R) Fixture Processor\nstepping : 1\n'
                   'hostname : fixture-private-machine\nserial : private-fixture-serial\n\n'
                   'processor : 1\nmodel name : Different Fixture\n', encoding='utf-8')
    result = module.cpu_execution_context(cpuinfo_path=cpu,
        environ={'OMP_NUM_THREADS': '4', 'MKL_CBWR': 'COMPATIBLE', 'USERNAME': 'fixture-private-user'})
    assert result['cpu_model']['model name'] == 'Intel(R) Fixture Processor'
    assert result['cpu_model']['vendor_id'] == 'GenuineIntel'
    assert result['cpu_model_available'] is True
    assert result['math_environment']['MKL_CBWR'] == 'COMPATIBLE'
    assert result['math_environment']['MKL_NUM_THREADS'] is None
    assert result['float32_matmul_precision'] in {'highest', 'high', 'medium'}
    assert type(result['torch_num_threads']) is int and result['torch_num_threads'] > 0
    assert type(result['torch_num_interop_threads']) is int and result['torch_num_interop_threads'] > 0
    assert isinstance(result['mkl_available'], bool) and isinstance(result['mkldnn_enabled'], bool)
    text = json.dumps(result)
    assert 'fixture-private' not in text and 'Different Fixture' not in text and str(tmp_path) not in text


def test_missing_hardware_information_remains_unknown(tmp_path):
    result = implementation().cpu_execution_context(cpuinfo_path=tmp_path / 'missing', environ={})
    assert result['cpu_model_available'] is False and result['cpu_model'] == {}
    assert result['flush_denormal_state'] is None
    assert all(value is None for value in result['math_environment'].values())


@pytest.mark.parametrize('location', ['selected_cpu_field', 'selected_environment'])
def test_private_selected_context_is_rejected_without_echoing_it(tmp_path, location):
    module = implementation()
    cpu = tmp_path / 'cpuinfo'
    private = 'C:' + '/' + 'Users' + '/' + 'fixture-private-user' + '/input'
    cpu.write_text('model name : ' + (private if location == 'selected_cpu_field' else 'Fixture CPU'), encoding='utf-8')
    env = {'MKL_CBWR': private} if location == 'selected_environment' else {}
    with pytest.raises(ValueError) as error:
        module.cpu_execution_context(cpuinfo_path=cpu, environ=env)
    assert private not in str(error.value) and 'fixture-private-user' not in str(error.value)
