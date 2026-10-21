"""Keep enough bounded Windows time for tests plus the mandatory privacy guard."""
import ast
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def timeout_for(os_name, python):
    job = yaml.load((ROOT / '.github/workflows/python-tests.yml').read_text(encoding='utf-8'),
                    Loader=yaml.BaseLoader)['jobs']['unit']
    expression = job['timeout-minutes'].removeprefix('${{').removesuffix('}}').strip()
    expression = expression.replace('matrix.os', repr(os_name)).replace('matrix.python', repr(python))
    expression = expression.replace('&&', ' and ').replace('||', ' or ')

    def value(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
            return value(node.left) == value(node.comparators[0])
        if isinstance(node, ast.BoolOp):
            for operand in node.values:
                result = value(operand)
                if isinstance(node.op, ast.And) and not result or isinstance(node.op, ast.Or) and result:
                    return result
            return result
        raise AssertionError('Unexpected timeout expression')

    return value(ast.parse(expression, mode='eval').body)


@pytest.mark.parametrize('os_name,python,minutes', [
    ('windows-latest', '3.13', 30),
    ('windows-latest', '3.14', 30),
    ('ubuntu-latest', '3.13', 20),
    ('ubuntu-latest', '3.14', 20),
])
def test_matrix_budget_covers_privacy_after_tests_without_becoming_unbounded(os_name, python, minutes):
    assert timeout_for(os_name, python) == minutes
