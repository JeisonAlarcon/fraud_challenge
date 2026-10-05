"""Split temporal determinista: TEST (mas reciente) / VALID / TRAIN, sin solapamiento."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class EmptySplitError(Exception):
    """Alguna particion del split quedo sin filas."""


class SingleClassSplitError(Exception):
    """Alguna particion del split quedo con una sola clase del target."""


@dataclass
class SplitBoundaries:
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    row_counts: dict[str, int]
    prevalence: dict[str, float]


class TemporalSplitter:
    def __init__(
        self,
        timestamp_col: str,
        validation_days: int,
        test_days: int,
        target_col: str = "fraude",
        entity_col: str | None = None,
    ):
        self.timestamp_col = timestamp_col
        self.validation_days = validation_days
        self.test_days = test_days
        self.target_col = target_col
        self.entity_col = entity_col
        if entity_col is not None:
            # Upgrade: agrupar por entity_col y asignar el split a nivel de grupo, no de fila.
            raise NotImplementedError(
                "Split por grupo/entidad aun no implementado; confirmar entidad en el EDA primero."
            )

    def _boundaries_from_max_date(self, df: pd.DataFrame) -> tuple[pd.Timestamp, ...]:
        max_date = df[self.timestamp_col].max()
        test_start = max_date - pd.Timedelta(days=self.test_days)
        valid_start = test_start - pd.Timedelta(days=self.validation_days)
        return valid_start, test_start, max_date

    def assign(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega la columna `dataset_type` in ('TRAIN','VALID','TEST'). Metadato de particion,
        nunca se pasa como feature de entrada al pipeline."""
        df = df.copy()
        valid_start, test_start, _ = self._boundaries_from_max_date(df)
        ts = df[self.timestamp_col]
        dataset_type = pd.Series("TRAIN", index=df.index, dtype="object")
        dataset_type[(ts >= valid_start) & (ts < test_start)] = "VALID"
        dataset_type[ts >= test_start] = "TEST"
        df["dataset_type"] = dataset_type
        return df

    def compute_boundaries(self, df: pd.DataFrame) -> SplitBoundaries:
        if "dataset_type" not in df.columns:
            raise ValueError("Llamar `assign` antes de `compute_boundaries`.")
        valid_start, test_start, max_date = self._boundaries_from_max_date(df)
        train_end = valid_start
        valid_end = test_start
        row_counts = df["dataset_type"].value_counts().to_dict()
        prevalence = {
            split: float(df.loc[df["dataset_type"] == split, self.target_col].mean())
            for split in ("TRAIN", "VALID", "TEST")
            if row_counts.get(split, 0) > 0
        }
        return SplitBoundaries(
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            test_start=test_start,
            test_end=max_date,
            row_counts={s: int(row_counts.get(s, 0)) for s in ("TRAIN", "VALID", "TEST")},
            prevalence=prevalence,
        )

    def validate_split(self, df: pd.DataFrame) -> None:
        if "dataset_type" not in df.columns:
            raise ValueError("Llamar `assign` antes de `validate_split`.")
        for split in ("TRAIN", "VALID", "TEST"):
            subset = df.loc[df["dataset_type"] == split]
            if len(subset) == 0:
                raise EmptySplitError(
                    f"Particion '{split}' quedo vacia; ajustar validation_days/test_days"
                )
            classes = set(subset[self.target_col].dropna().unique().tolist())
            if len(classes) < 2:
                raise SingleClassSplitError(
                    f"Particion '{split}' tiene una sola clase de '{self.target_col}': {classes}"
                )
