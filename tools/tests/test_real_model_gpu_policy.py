"""Prevent the active real-model acceptance entry points from scheduling CPU inference."""
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "validation/com.ayutaz.embeddinggemma.validation/Tests/Editor"


@pytest.mark.parametrize("name,expected", [
    ("CompensatedVisionReferenceTests", 3),
    ("CompensatedNormAudioEncoderReferenceTests", 1),
])
def test_active_real_model_matrix_is_gpu_only_and_reports_its_scope(name, expected):
    source = (FIXTURES / f"{name}.cs").read_text(encoding="utf-8")
    backends = re.findall(r"\[TestCase\([^\]]*BackendType\.(\w+)\)\]", source)
    assert backends == ["GPUCompute"] * expected, "Real-model runs must not schedule CPU cases"
    assert '["execution_backend"] = "GPUCompute"' in source
    assert '["cpu_validated"] = false' in source
    assert f"conditions.Count == {expected}" in source
    assert 'Assert.That(backend, Is.EqualTo(BackendType.GPUCompute)' in source
    assert "Assert.That(fallback, Is.Zero)" in source or "Assert.That(fallbackCount, Is.Zero)" in source
