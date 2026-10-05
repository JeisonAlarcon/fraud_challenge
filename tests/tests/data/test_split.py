from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.split import (
    EmptySplitError,
    SingleClassSplitError,
    TemporalSplitter,
)


def test_assign_produces_no_overlap_and_covers_all_rows(synthetic_df):
    splitter = TemporalSplitter("fecha", validation_days=7, test_days=7)
    df = splitter.assign(synthetic_df)
    assert set(df["dataset_type"].unique()) <= {"TRAIN", "VALID", "TEST"}
    assert len(df) == len(synthetic_df)

    valid_start, test_start, _ = splitter._boundaries_from_max_date(synthetic_df)
    train = df[df["dataset_type"] == "TRAIN"]
    valid = df[df["dataset_type"] == "VALID"]
    test = df[df["dataset_type"] == "TEST"]
    assert (train["fecha"] < valid_start).all()
    assert ((valid["fecha"] >= valid_start) & (valid["fecha"] < test_start)).all()
    assert (test["fecha"] >= test_start).all()


def test_both_classes_present_in_every_partition(synthetic_df):
    splitter = TemporalSplitter("fecha", validation_days=7, test_days=7)
    df = splitter.assign(synthetic_df)
    splitter.validate_split(df)  # no debe lanzar
    for split in ("TRAIN", "VALID", "TEST"):
        classes = set(df.loc[df["dataset_type"] == split, "fraude"].unique().tolist())
        assert classes == {0, 1}


def test_empty_partition_raises(synthetic_df):
    splitter = TemporalSplitter("fecha", validation_days=7, test_days=7)
    df = splitter.assign(synthetic_df)
    df = df[df["dataset_type"] != "VALID"].copy()  # fuerza particion vacia
    with pytest.raises(EmptySplitError):
        splitter.validate_split(df)


def test_single_class_partition_raises(synthetic_df):
    splitter = TemporalSplitter("fecha", validation_days=7, test_days=7)
    df = splitter.assign(synthetic_df)
    df = df.copy()
    df.loc[df["dataset_type"] == "TEST", "fraude"] = 0
    with pytest.raises(SingleClassSplitError):
        splitter.validate_split(df)


def test_compute_boundaries_row_counts_match_assign(synthetic_df):
    splitter = TemporalSplitter("fecha", validation_days=7, test_days=7)
    df = splitter.assign(synthetic_df)
    boundaries = splitter.compute_boundaries(df)
    counts = df["dataset_type"].value_counts().to_dict()
    for split in ("TRAIN", "VALID", "TEST"):
        assert boundaries.row_counts[split] == counts.get(split, 0)
