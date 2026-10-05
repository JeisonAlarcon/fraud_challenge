from __future__ import annotations

from data.manifest import (
    ContractReportSummary,
    DatasetManifest,
    OutputParquetInfo,
    SourceCsvInfo,
    SplitInfo,
    compute_file_sha256,
)


def test_compute_file_sha256_is_deterministic(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("hola mundo")
    assert compute_file_sha256(path) == compute_file_sha256(path)


def test_compute_file_sha256_changes_with_content(tmp_path):
    p1, p2 = tmp_path / "a.txt", tmp_path / "b.txt"
    p1.write_text("a")
    p2.write_text("b")
    assert compute_file_sha256(p1) != compute_file_sha256(p2)


def test_manifest_json_roundtrip(tmp_path):
    manifest = DatasetManifest(
        created_at=DatasetManifest.now(),
        git_sha="abc123",
        source_csv=SourceCsvInfo(filename="x.csv", sha256="deadbeef", n_rows=10, n_cols=5),
        column_dtypes={"a": "int64"},
        cleaning_decisions=["dropped 1 duplicate"],
        contract_report=ContractReportSummary(is_valid=True, violations=[], warnings=["w1"]),
        split=SplitInfo(
            strategy="TemporalSplitter",
            validation_days=7,
            test_days=7,
            train_end="2020-03-20T00:00:00",
            valid_start="2020-03-20T00:00:00",
            valid_end="2020-03-27T00:00:00",
            test_start="2020-03-27T00:00:00",
            test_end="2020-04-03T00:00:00",
            row_counts={"TRAIN": 5, "VALID": 3, "TEST": 2},
            prevalence={"TRAIN": 0.1, "VALID": 0.1, "TEST": 0.1},
        ),
        output_parquet=OutputParquetInfo(path="out.parquet", sha256="feedface", n_rows=10),
    )
    out_path = tmp_path / "manifest.json"
    manifest.to_json(out_path)
    loaded = DatasetManifest.from_json(out_path)
    assert loaded.source_csv.sha256 == "deadbeef"
    assert loaded.split.row_counts == {"TRAIN": 5, "VALID": 3, "TEST": 2}
    assert loaded.contract_report.warnings == ["w1"]
