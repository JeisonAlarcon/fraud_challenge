"""DatasetBuilder: limpieza/tipado + split temporal + escritura del Parquet+manifest congelados."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.contract import DataContract
from data.manifest import (
    ContractReportSummary,
    DatasetManifest,
    OutputParquetInfo,
    SourceCsvInfo,
    SplitInfo,
    compute_file_sha256,
    get_git_sha,
)
from data.split import TemporalSplitter


class DatasetBuilder:
    def __init__(self, contract: DataContract, splitter: TemporalSplitter, output_dir: Path):
        self.contract = contract
        self.splitter = splitter
        self.output_dir = Path(output_dir)

    def build(
        self,
        raw_csv_path: Path,
        output_name: str = "fraud_clean_v1.parquet",
        force: bool = False,
    ) -> DatasetManifest:
        raw_csv_path = Path(raw_csv_path)
        manifest_path = self.output_dir / "manifest.json"
        source_sha256 = compute_file_sha256(raw_csv_path)

        if not force and manifest_path.exists():
            existing = DatasetManifest.from_json(manifest_path)
            if existing.source_csv.sha256 == source_sha256:
                return existing

        df = pd.read_csv(raw_csv_path)
        contract_report = self.contract.validate_or_raise(df)

        df = self.contract.coerce_dtypes(df)

        cleaning_decisions: list[str] = []
        n_before = len(df)
        df = df.drop_duplicates()
        n_dropped = n_before - len(df)
        cleaning_decisions.append(f"Eliminadas {n_dropped} filas duplicadas de {n_before}")

        ts_col = self.contract.config.timestamp.column
        df = df.sort_values(ts_col, ascending=True).reset_index(drop=True)
        cleaning_decisions.append(f"Ordenado ascendentemente por '{ts_col}'")

        df = self.splitter.assign(df)
        self.splitter.validate_split(df)
        boundaries = self.splitter.compute_boundaries(df)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / output_name
        df.to_parquet(output_path, engine="pyarrow", index=False)

        manifest = DatasetManifest(
            created_at=DatasetManifest.now(),
            git_sha=get_git_sha(),
            source_csv=SourceCsvInfo(
                filename=raw_csv_path.name,
                sha256=source_sha256,
                n_rows=len(df),
                n_cols=df.shape[1],
            ),
            column_dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            cleaning_decisions=cleaning_decisions,
            contract_report=ContractReportSummary(
                is_valid=contract_report.is_valid,
                violations=contract_report.violations,
                warnings=contract_report.warnings,
            ),
            split=SplitInfo(
                strategy=self.splitter.__class__.__name__,
                validation_days=self.splitter.validation_days,
                test_days=self.splitter.test_days,
                train_end=boundaries.train_end,
                valid_start=boundaries.valid_start,
                valid_end=boundaries.valid_end,
                test_start=boundaries.test_start,
                test_end=boundaries.test_end,
                row_counts=boundaries.row_counts,
                prevalence=boundaries.prevalence,
            ),
            output_parquet=OutputParquetInfo(
                path=str(output_path),
                sha256=compute_file_sha256(output_path),
                n_rows=len(df),
            ),
        )
        manifest.to_json(manifest_path)
        return manifest
