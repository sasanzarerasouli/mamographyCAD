"""JSON/HTML report generation: provenance, disclaimer, base64 images."""
import json
import os
import re

import pytest


@pytest.fixture(scope="module")
def built_report(analysis, tmp_path_factory):
    from src.reporting.report_builder import ReportBuilder

    result, _m, _c = analysis
    out_dir = tmp_path_factory.mktemp("reports")
    builder = ReportBuilder(output_dir=str(out_dir))
    paths = builder.save(result, patient_id="TESTPAT", patient_name="Phantom^Test")
    return paths, builder, out_dir


def test_report_files_exist(built_report):
    paths, _builder, _dir = built_report
    for key in ("json", "html"):
        assert os.path.isfile(paths[key]), key
        assert os.path.getsize(paths[key]) > 200


def test_json_report_contains_provenance(built_report):
    paths, _b, _d = built_report
    with open(paths["json"], "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["report_meta"]["patient_id"] == "TESTPAT"
    assert "disclaimer" in payload and "radiolog" in payload["disclaimer"].lower()
    assert payload["birads"]["category"] >= 0
    assert "model_status" in payload


def test_json_report_never_invents_values(built_report):
    paths, _b, _d = built_report
    with open(paths["json"], "r", encoding="utf-8") as fh:
        text = fh.read()
    # Standalone NaN / Infinity tokens are invalid JSON for consumers and would
    # mean a placeholder leaked into the report.
    assert not re.search(r"[:\[,]\s*(NaN|Infinity|-Infinity)\b", text)
    assert "NaN" not in text or '"NaN"' not in text


def test_html_report_is_self_contained(built_report):
    paths, _b, _d = built_report
    with open(paths["html"], "r", encoding="utf-8") as fh:
        html = fh.read()
    assert "data:image/png;base64," in html, "overlay not embedded"
    assert "BI-RADS" in html
    assert "radiolog" in html.lower()
    # no external resources -> report renders offline
    assert "http://" not in html and "https://" not in html


def test_custom_output_directory(analysis, tmp_path):
    from src.reporting.report_builder import ReportBuilder

    result, _m, _c = analysis
    paths = ReportBuilder(output_dir=str(tmp_path)).save(result)
    assert os.path.isdir(tmp_path)
    assert os.path.isfile(paths["html"])
    assert os.path.isfile(paths["json"])
