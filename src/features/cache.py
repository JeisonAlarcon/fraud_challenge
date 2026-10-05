"""Cache de features inmutable y content-addressed: `data/cache/features/<version>/<id>/`."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from config import FEATURE_PIPELINE_CODE_VERSION
from data.manifest import DatasetManifest, get_git_sha
from features.factory import FeaturePipelineFactory
from features.schema import FeatureVersionConfig

_DEPENDENCY_PACKAGES = ["scikit-learn", "category_encoders", "pandas", "numpy", "scipy"]


class FeatureCacheNotFoundError(Exception):
    pass


def _hash_obj(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _dependency_versions() -> dict[str, str]:
    return {pkg: pkg_version(pkg) for pkg in _DEPENDENCY_PACKAGES}


def compute_feature_cache_id(
    parquet_hash: str,
    split_manifest_hash: str,
    feature_yaml_hash: str,
    code_version: str,
    dependency_versions_hash: str,
) -> str:
    combined = (
        parquet_hash
        + split_manifest_hash
        + feature_yaml_hash
        + code_version
        + dependency_versions_hash
    )
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


@dataclass
class FeatureCachePaths:
    root: Path

    @property
    def X_train(self) -> Path:
        return self.root / "X_train.npz"

    @property
    def X_valid(self) -> Path:
        return self.root / "X_valid.npz"

    @property
    def X_test(self) -> Path:
        return self.root / "X_test.npz"

    @property
    def y_train(self) -> Path:
        return self.root / "y_train.npy"

    @property
    def y_valid(self) -> Path:
        return self.root / "y_valid.npy"

    @property
    def y_test(self) -> Path:
        return self.root / "y_test.npy"

    @property
    def preprocessor(self) -> Path:
        return self.root / "preprocessor.pkl"

    @property
    def feature_names(self) -> Path:
        return self.root / "feature_names.json"

    @property
    def input_schema(self) -> Path:
        return self.root / "input_schema.json"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    def load_metadata(self) -> dict:
        return json.loads(self.metadata.read_text())

    def load_split(self, split: str) -> tuple[np.ndarray | sp.spmatrix, np.ndarray]:
        X_path = {"train": self.X_train, "valid": self.X_valid, "test": self.X_test}[split]
        y_path = {"train": self.y_train, "valid": self.y_valid, "test": self.y_test}[split]
        metadata = self.load_metadata()
        if metadata["is_sparse"]:
            X = sp.load_npz(X_path)
        else:
            with np.load(X_path) as data:
                X = data["X"]
        y = np.load(y_path)
        return X, y


def _save_X(path: Path, X: Any) -> None:
    if sp.issparse(X):
        sp.save_npz(path, X.tocsr())
    else:
        np.savez_compressed(path, X=np.asarray(X))


class FeatureCacheBuilder:
    def build(
        self,
        parquet_path: Path,
        manifest: DatasetManifest,
        feature_version: str,
        feature_config: FeatureVersionConfig,
        cache_root: Path,
        force: bool = False,
    ) -> FeatureCachePaths:
        parquet_hash = manifest.output_parquet.sha256
        split_manifest_hash = _hash_obj(manifest.split.model_dump(mode="json"))
        feature_yaml_hash = _hash_obj(feature_config.model_dump(mode="json"))
        dependency_versions = _dependency_versions()
        dependency_versions_hash = _hash_obj(dependency_versions)

        feature_cache_id = compute_feature_cache_id(
            parquet_hash,
            split_manifest_hash,
            feature_yaml_hash,
            FEATURE_PIPELINE_CODE_VERSION,
            dependency_versions_hash,
        )
        cache_dir = Path(cache_root) / feature_version / feature_cache_id
        paths = FeatureCachePaths(root=cache_dir)

        if cache_dir.exists() and paths.metadata.exists() and not force:
            print(f"[FeatureCacheBuilder] cache hit: {cache_dir}")
            return paths

        print(f"[FeatureCacheBuilder] cache miss: {cache_dir} (fitting pipeline)")
        df = pd.read_parquet(parquet_path)
        target_col = feature_config.target
        splits = {s: df.loc[df["dataset_type"] == s] for s in ("TRAIN", "VALID", "TEST")}

        factory = FeaturePipelineFactory(feature_config)
        fitted = factory.fit(splits["TRAIN"], splits["TRAIN"][target_col].astype("int64"))

        X_by_split: dict[str, Any] = {}
        y_by_split: dict[str, np.ndarray] = {}
        for split_name, split_df in splits.items():
            X_by_split[split_name] = factory.transform(split_df, fitted)
            y_by_split[split_name] = split_df[target_col].astype("int64").to_numpy()

        feature_names = factory.get_feature_names(fitted, splits["TRAIN"].iloc[:1])
        is_sparse = sp.issparse(X_by_split["TRAIN"])

        cache_dir.mkdir(parents=True, exist_ok=True)
        _save_X(paths.X_train, X_by_split["TRAIN"])
        _save_X(paths.X_valid, X_by_split["VALID"])
        _save_X(paths.X_test, X_by_split["TEST"])
        np.save(paths.y_train, y_by_split["TRAIN"])
        np.save(paths.y_valid, y_by_split["VALID"])
        np.save(paths.y_test, y_by_split["TEST"])
        joblib.dump(fitted, paths.preprocessor)
        paths.feature_names.write_text(json.dumps(feature_names, indent=2))

        input_schema = {
            spec.name: {"role": spec.role.value, "dtype": str(splits["TRAIN"][spec.name].dtype)}
            for spec in feature_config.enabled_features()
        }
        paths.input_schema.write_text(json.dumps(input_schema, indent=2))

        metadata = {
            "feature_cache_id": feature_cache_id,
            "feature_version": feature_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parquet_hash": parquet_hash,
            "split_manifest_hash": split_manifest_hash,
            "feature_yaml_hash": feature_yaml_hash,
            "code_version": FEATURE_PIPELINE_CODE_VERSION,
            "dependency_versions": dependency_versions,
            "row_counts": {s: int(len(splits[s])) for s in splits},
            "prevalence": {s: float(y_by_split[s].mean()) for s in y_by_split},
            "is_sparse": bool(is_sparse),
            "n_features": len(feature_names),
            "git_sha": get_git_sha(),
        }
        paths.metadata.write_text(json.dumps(metadata, indent=2))
        return paths


class FeatureCacheResolver:
    def resolve(
        self,
        feature_version: str,
        manifest: DatasetManifest,
        feature_config: FeatureVersionConfig,
        cache_root: Path,
    ) -> FeatureCachePaths:
        parquet_hash = manifest.output_parquet.sha256
        split_manifest_hash = _hash_obj(manifest.split.model_dump(mode="json"))
        feature_yaml_hash = _hash_obj(feature_config.model_dump(mode="json"))
        dependency_versions_hash = _hash_obj(_dependency_versions())
        feature_cache_id = compute_feature_cache_id(
            parquet_hash,
            split_manifest_hash,
            feature_yaml_hash,
            FEATURE_PIPELINE_CODE_VERSION,
            dependency_versions_hash,
        )
        cache_dir = Path(cache_root) / feature_version / feature_cache_id
        paths = FeatureCachePaths(root=cache_dir)
        if not paths.metadata.exists():
            raise FeatureCacheNotFoundError(
                f"No existe cache para feature_version='{feature_version}' con id "
                f"'{feature_cache_id}' en {cache_dir}. Correr FeatureCacheBuilder primero."
            )
        return paths
