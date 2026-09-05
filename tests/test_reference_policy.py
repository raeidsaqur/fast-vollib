"""Reference checks permit rounding, not changed models or corrupt records."""

import json

import numpy as np
import pytest
from reference_fixtures import assert_reference_array, assert_reference_text, write_or_check


def test_reference_bounds_reject_material_changes(monkeypatch):
    monkeypatch.delenv("FV_STRICT_REFERENCE_FIXTURES", raising=False)
    assert_reference_array([100.0 + 1e-12], [100.0])
    assert_reference_array([1e-10], [0.0], pricing=True)
    with pytest.raises(AssertionError):
        assert_reference_array([100.0 + 1e-8], [100.0])
    with pytest.raises(AssertionError):
        assert_reference_array([1e-8], [0.0], pricing=True)


def test_strict_reference_mode_checks_the_last_bit(monkeypatch):
    monkeypatch.setenv("FV_STRICT_REFERENCE_FIXTURES", "1")
    with pytest.raises(AssertionError):
        assert_reference_array([np.nextafter(1.0, 2.0)], [1.0])


def test_metadata_and_digest_are_not_hidden_by_numeric_tolerance():
    with pytest.raises(AssertionError):
        assert_reference_text('{"scheme": "a"}', '{"scheme": "b"}')
    broken = json.dumps({"value": 1, "content_sha256": "bad"})
    with pytest.raises(AssertionError):
        assert_reference_text(broken, broken)


def test_check_mode_does_not_rewrite_a_stale_reference(tmp_path, monkeypatch):
    path = tmp_path / "fixture.json"
    path.write_text('{"scheme": "a"}')
    monkeypatch.setattr("sys.argv", ["generator", "--check"])
    with pytest.raises(AssertionError):
        write_or_check(path, '{"scheme": "b"}')
    assert path.read_text() == '{"scheme": "a"}'
