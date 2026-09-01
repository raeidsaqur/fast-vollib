"""The diagnostics-v1 wire contract: strict, deterministic, and self-checking."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import jsonschema
import numpy as np
import pytest

from fast_vollib.diagnostics import (
    SCHEMA_VERSION,
    Box,
    DiagnosticRecord,
    DiagnosticReport,
    DiagnosticsConfig,
    NamedRegion,
    SampleDiagnostics,
    SurfaceQuotes,
    diagnose_fit,
    fit_error_by_region,
    serialization,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "docs" / "schemas" / "diagnostics-v1.schema.json"
GOLDEN_PATH = ROOT / "tests" / "fixtures" / "diagnostics" / "golden-v1.json"
GENERATOR = ROOT / "scripts" / "generate_diagnostics_schema.py"


def _generator_module():
    spec = importlib.util.spec_from_file_location("generate_diagnostics_schema", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _small_report() -> DiagnosticReport:
    quotes = SurfaceQuotes(
        k=[-0.2, -0.1, 0.0, 0.1, 0.2],
        T=np.full(5, 0.5),
        iv=np.full(5, 0.2),
        surface_id="A",
        point_id=list(range(5)),
    )
    sample = diagnose_fit(np.array([0.21, 0.20, 0.20, 0.19, np.nan]), quotes)
    return DiagnosticReport.from_records(
        [
            DiagnosticRecord(model="prior", split="unif_5", sample_id=1, diagnostics=sample),
            DiagnosticRecord(model="prior", split="unif_5", sample_id=2, diagnostics=sample),
        ]
    )


# -- generated artifacts -----------------------------------------------------
def test_committed_schema_and_golden_fixture_are_byte_identical_to_the_generator():
    module = _generator_module()
    for path, content in module.render().items():
        assert path.read_text(encoding="utf-8") == content, f"{path.name} is stale"


def test_generator_check_mode_passes_on_the_committed_tree():
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr


def test_the_committed_schema_is_a_valid_closed_draft_2020_12_schema():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)

    def assert_closed(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                assert node.get("additionalProperties") is False
                assert sorted(node["required"]) == sorted(node["properties"])
            for value in node.values():
                assert_closed(value)
        elif isinstance(node, list):
            for item in node:
                assert_closed(item)

    assert_closed(schema)


def test_the_golden_fixture_validates_against_the_committed_schema():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    payload = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(payload)
    assert payload["schema"] == SCHEMA_VERSION


def test_the_golden_fixture_round_trips_byte_for_byte():
    text = GOLDEN_PATH.read_text(encoding="utf-8")
    assert serialization.dumps(serialization.loads(text), indent=2) == text


def test_the_golden_fixture_exercises_every_optional_block():
    payload = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    statuses = {summary["quality_status"] for summary in payload["summaries"]}
    assert statuses == {"complete", "partial"}
    assert any(summary["spread"] is not None for summary in payload["summaries"])
    assert any(summary["spread"] is None for summary in payload["summaries"])
    assert any(summary["grid"] is not None for summary in payload["summaries"])
    assert any(summary["grid"] is None for summary in payload["summaries"])
    assert any(summary["ragged"]["duplicate_group_count"] > 0 for summary in payload["summaries"])
    assert any(
        summary["ragged"]["nonpositive_total_variance_points"] > 0
        for summary in payload["summaries"]
    )


# -- determinism -------------------------------------------------------------
def test_rendering_is_deterministic_and_newline_terminated():
    report = _small_report()
    first = serialization.dumps(report)
    assert first == serialization.dumps(report)
    assert first.endswith("\n")


def test_record_order_is_canonical_regardless_of_input_order():
    quotes = SurfaceQuotes(k=[0.0], T=[1.0], iv=[0.2], surface_id="A")
    sample = diagnose_fit(np.array([0.2]), quotes)
    forward = [
        DiagnosticRecord(model="a", split="s", sample_id=1, diagnostics=sample),
        DiagnosticRecord(model="a", split="s", sample_id=2, diagnostics=sample),
    ]
    assert serialization.dumps(DiagnosticReport.from_records(forward)) == serialization.dumps(
        DiagnosticReport.from_records(list(reversed(forward)))
    )


def test_integer_and_string_sample_ids_sort_without_comparing_across_types():
    quotes = SurfaceQuotes(k=[0.0], T=[1.0], iv=[0.2], surface_id="A")
    sample = diagnose_fit(np.array([0.2]), quotes)
    report = DiagnosticReport.from_records(
        [
            DiagnosticRecord(model="a", split="s", sample_id="b", diagnostics=sample),
            DiagnosticRecord(model="a", split="s", sample_id=2, diagnostics=sample),
            DiagnosticRecord(model="a", split="s", sample_id="a", diagnostics=sample),
            DiagnosticRecord(model="a", split="s", sample_id=1, diagnostics=sample),
        ]
    )
    assert [record.sample_id for record in report.records] == [1, 2, "a", "b"]


# -- strictness --------------------------------------------------------------
def test_round_trip_preserves_the_report():
    report = _small_report()
    restored = serialization.loads(serialization.dumps(report))
    assert restored.records == report.records
    assert restored.summaries() == report.summaries()
    assert restored.config.regions == report.config.regions


def test_summary_only_projection_keeps_the_original_record_count():
    report = _small_report()
    projected = report.summary_only()
    text = serialization.dumps(projected)
    payload = json.loads(text)
    assert payload["records"] == []
    assert payload["records_included"] is False
    assert payload["record_count"] == 2
    assert len(payload["summaries"]) == 1
    restored = serialization.loads(text)
    assert restored.record_count == 2
    assert restored.records_included is False
    assert restored.summaries() == projected.summaries()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p: p.update(extra=1), "keys mismatch"),
        (lambda p: p.pop("summaries"), "keys mismatch"),
        (lambda p: p.update(schema="diagnostics-v2"), "Unsupported schema"),
        (lambda p: p["records"][0].update(extra=1), "keys mismatch"),
        (lambda p: p["records"][0].update(quality_status="excellent"), "quality_status"),
        (lambda p: p["records"][0].update(sample_id=True), "string or integer"),
        (lambda p: p["records"][0]["fit"]["overall"].update(target_count=1.5), "integer"),
        (lambda p: p.update(record_count="2"), "record_count must be an integer"),
        (lambda p: p.update(records_included="yes"), "records_included must be a boolean"),
        (
            lambda p: p["config"]["regions"][0].update(closed="sideways"),
            "closed must be one of",
        ),
        (
            lambda p: p["config"]["ragged"].update(duplicates="average"),
            "duplicates must be one of",
        ),
    ],
)
def test_the_decoder_fails_closed(mutate, message):
    payload = json.loads(serialization.dumps(_small_report()))
    mutate(payload)
    with pytest.raises(serialization.DiagnosticsDecodeError, match=message):
        serialization.decode(payload)


def test_a_tampered_derived_value_is_rejected():
    payload = json.loads(serialization.dumps(_small_report()))
    payload["summaries"][0]["fit"]["overall"]["rmse"] = 0.0
    with pytest.raises(serialization.DiagnosticsDecodeError, match="but its sums imply"):
        serialization.decode(payload)


def test_a_tampered_quality_status_is_rejected():
    payload = json.loads(serialization.dumps(_small_report()))
    payload["records"][0]["quality_status"] = "complete"
    with pytest.raises(serialization.DiagnosticsDecodeError, match="but its counts imply"):
        serialization.decode(payload)


def test_summaries_must_agree_with_the_records_they_claim_to_pool():
    payload = json.loads(serialization.dumps(_small_report()))
    payload["summaries"][0]["sample_count"] = 99
    with pytest.raises(serialization.DiagnosticsDecodeError, match="do not match the summaries"):
        serialization.decode(payload)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_standard_json_floats_are_refused_on_the_way_in(constant):
    text = serialization.dumps(_small_report()).replace(
        '"record_count":2', f'"record_count":{constant}'
    )
    with pytest.raises(serialization.DiagnosticsDecodeError, match="does not permit"):
        serialization.loads(text)


def test_non_standard_floats_cannot_be_written():
    from fast_vollib.diagnostics import ErrorSums, FitDiagnostics

    broken = SampleDiagnostics(
        fit=FitDiagnostics(overall=ErrorSums(squared_error_sum=float("nan")))
    )
    report = DiagnosticReport.from_records(
        [DiagnosticRecord(model="a", split="s", sample_id=1, diagnostics=broken)]
    )
    with pytest.raises(ValueError, match="cannot carry NaN"):
        serialization.dumps(report)


def test_a_region_without_a_descriptor_cannot_be_serialized():
    region = NamedRegion(name="odd", complement_name=None, predicate=lambda k, T: np.asarray(k) > 0)
    with pytest.raises(ValueError, match="no serializable descriptor"):
        DiagnosticsConfig(regions=(region,))
    sample = SampleDiagnostics(fit=fit_error_by_region([0.2], [0.2], [0.0], [1.0], (region,)))
    report = DiagnosticReport.from_records(
        [DiagnosticRecord(model="a", split="s", sample_id=1, diagnostics=sample)]
    )
    with pytest.raises(ValueError, match="refuses a region it cannot describe"):
        serialization.dumps(report)


def test_region_names_must_agree_with_their_descriptor():
    payload = json.loads(serialization.dumps(_small_report()))
    payload["records"][0]["fit"]["regions"][0]["name"] = "renamed"
    with pytest.raises(serialization.DiagnosticsDecodeError, match="names disagree"):
        serialization.decode(payload)


# -- metadata ----------------------------------------------------------------
def test_metadata_must_be_bounded_json():
    DiagnosticsConfig(metadata={"run": 1, "tags": ["a", "b"], "nested": {"ok": True}})
    report = DiagnosticReport.from_records(
        [], config=DiagnosticsConfig(metadata={"run": "x" * 100})
    )
    assert json.loads(serialization.dumps(report))["config"]["metadata"]["run"] == "x" * 100

    oversized = DiagnosticReport.from_records(
        [],
        config=DiagnosticsConfig(metadata={"blob": "x" * (serialization.MAX_METADATA_BYTES + 1)}),
    )
    with pytest.raises(ValueError, match="the limit is"):
        serialization.dumps(oversized)


def test_metadata_rejects_non_json_values_and_excessive_depth():
    with pytest.raises(TypeError, match="only contain JSON values"):
        serialization.dumps(
            DiagnosticReport.from_records([], config=DiagnosticsConfig(metadata={"k": {1, 2}}))
        )
    deep: dict = {"level": None}
    node = deep
    for _ in range(serialization.MAX_METADATA_DEPTH + 2):
        node["level"] = {"level": None}
        node = node["level"]
    with pytest.raises(ValueError, match="nests deeper"):
        serialization.dumps(
            DiagnosticReport.from_records([], config=DiagnosticsConfig(metadata=deep))
        )


def test_boxes_round_trip_through_their_descriptor():
    box = Box(name="wings", complement_name="core", k_min=0.1, T_min=0.2, closed="neither")
    assert Box.from_describe(box.describe()) == box
    with pytest.raises(ValueError, match="do not match"):
        Box.from_describe({"kind": "box", "name": "x"})
