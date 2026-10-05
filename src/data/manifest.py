"""Manifest de trazabilidad para el Parquet limpio y su split temporal."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel


def compute_file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash determinista del contenido de un archivo (usado para parquet y CSV de origen)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_sha() -> str:
    """SHA del HEAD actual; "unknown" fuera de un repo git (p.ej. Colab sin clonar el repo)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


class SourceCsvInfo(BaseModel):
    filename: str
    sha256: str
    n_rows: int
    n_cols: int


class ContractReportSummary(BaseModel):
    is_valid: bool
    violations: list[str]
    warnings: list[str]


class SplitInfo(BaseModel):
    strategy: str
    validation_days: int
    test_days: int
    train_end: datetime
    valid_start: datetime
    valid_end: datetime
    test_start: datetime
    test_end: datetime
    row_counts: dict[str, int]
    prevalence: dict[str, float]


class OutputParquetInfo(BaseModel):
    path: str
    sha256: str
    n_rows: int


class DatasetManifest(BaseModel):
    created_at: datetime
    git_sha: str
    source_csv: SourceCsvInfo
    column_dtypes: dict[str, str]
    cleaning_decisions: list[str]
    contract_report: ContractReportSummary
    split: SplitInfo
    output_parquet: OutputParquetInfo

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, path: Path) -> DatasetManifest:
        return cls.model_validate_json(path.read_text())

    @classmethod
    def now(cls) -> datetime:
        return datetime.now(timezone.utc)
