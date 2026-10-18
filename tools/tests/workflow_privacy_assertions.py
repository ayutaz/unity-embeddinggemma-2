"""Assert that failure diagnostics cannot bypass the artifact privacy gate."""
import re


def assert_privacy_gated_upload(step, *, base='always()'):
    pattern = (r'\$\{\{\s*\(' + re.escape(base)
               + r"\)\s*&&\s*steps\.[A-Za-z_][A-Za-z0-9_]*\.outcome\s*==\s*'success'\s*\}\}")
    assert re.fullmatch(pattern, step.get('if', '')), (
        'Retain the original execution condition and require a successful privacy guard'
    )
