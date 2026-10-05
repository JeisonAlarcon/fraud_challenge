"""`OnnxScorer`: cadena de produccion `model_raw.onnx` -> `calibrator.onnx` + umbral.

`score()` recibe un diccionario de features crudas (aplica `preprocessor.pkl` en Python antes
de invocar los ONNX). `score_matrix()` opera directamente sobre una matriz ya preprocesada
(usado en las 3 paridades de promocion y en la evaluacion unica de TEST): mantiene esas
verificaciones estrictamente dentro del alcance ONNX que exige el documento.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from serving.contract import (
    FeatureContract,
    ThresholdContract,
    load_feature_contract,
    load_threshold_contract,
    validate_raw_features,
)
from serving.onnx_exporter import run_onnx_raw_score


class ScoringError(RuntimeError):
    pass


@dataclass
class ScoringResult:
    raw_score: float
    fraud_probability: float
    decision: bool
    threshold: float
    latency_ms: float


class OnnxScorer:
    def __init__(self, promoted_dir: Path):
        promoted_dir = Path(promoted_dir)
        self.feature_contract: FeatureContract = load_feature_contract(
            promoted_dir / "feature_contract.json"
        )
        self.threshold_contract: ThresholdContract = load_threshold_contract(
            promoted_dir / "threshold.json"
        )
        self.preprocessor = joblib.load(promoted_dir / "preprocessor.pkl")
        self.model_session = ort.InferenceSession(str(promoted_dir / "model_raw.onnx"))
        self.calibrator_session = ort.InferenceSession(str(promoted_dir / "calibrator.onnx"))

    def score(self, raw_features: dict) -> ScoringResult:
        start = time.perf_counter()
        validate_raw_features(self.feature_contract, raw_features)
        ordered = {col.name: raw_features[col.name] for col in self.feature_contract.input_columns}
        df = pd.DataFrame([ordered])
        X = self.preprocessor.transform(df)
        if hasattr(X, "toarray"):
            X = X.toarray()

        raw_score, calibrated = self._score_chain(np.asarray(X, dtype=np.float32))

        latency_ms = (time.perf_counter() - start) * 1000
        return ScoringResult(
            raw_score=float(raw_score[0]),
            fraud_probability=float(calibrated[0]),
            decision=bool(calibrated[0] >= self.threshold_contract.threshold),
            threshold=self.threshold_contract.threshold,
            latency_ms=latency_ms,
        )

    def score_matrix(self, X: np.ndarray) -> np.ndarray:
        return self._score_chain(X)[1]

    def _score_chain(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw_scores = run_onnx_raw_score(self.model_session, X)
        calibrated = run_onnx_raw_score(self.calibrator_session, raw_scores)
        if not np.all((calibrated >= 0.0) & (calibrated <= 1.0)):
            raise ScoringError("fraud_probability fuera de [0,1]")
        return raw_scores, calibrated
