"""Exportadores ONNX: `OnnxModelExporter` (modelo ganador -> `model_raw.onnx`) y
`OnnxCalibratorExporter` (calibrador -> `calibrator.onnx`). Comparten `assert_parity`.

El preprocesador (`preprocessor.pkl`) nunca se convierte a ONNX: `TargetEncoderOOF` y las
features ciclicas custom no tienen convertidor confiable en skl2onnx, y el documento solo exige
ONNX para modelo y calibrador. `OnnxScorer` (serving/onnx_scorer.py) aplica el pickle en Python
antes de invocar los dos ONNX cuando el input es un diccionario crudo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper
from skl2onnx import to_onnx

from models.versions.v1.base import BaseModel


# El onnxruntime disponible en el entorno de ejecución admite como máximo IR 13.
# `helper.make_model` toma el IR de la librería `onnx` instalada, que puede ser más nuevo.
ONNXRUNTIME_MAX_IR_VERSION = 13


def _runtime_compatible_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Fija el IR de grafos propios al máximo aceptado por onnxruntime."""
    model.ir_version = min(model.ir_version, ONNXRUNTIME_MAX_IR_VERSION)
    return model


class ParityReport(NamedTuple):
    max_abs_diff: float
    max_rel_diff: float
    n_samples: int
    passed: bool


class ParityError(RuntimeError):
    pass


def _extract_raw_score(output: np.ndarray) -> np.ndarray:
    arr = np.asarray(output)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 1]
    return arr.reshape(-1)


def run_onnx_raw_score(
    session: ort.InferenceSession, X: np.ndarray, input_name: str | None = None
) -> np.ndarray:
    input_name = input_name or session.get_inputs()[0].name
    outputs = session.run(None, {input_name: np.asarray(X, dtype=np.float32)})
    for out in outputs:
        arr = np.asarray(out)
        if np.issubdtype(arr.dtype, np.floating):
            return _extract_raw_score(arr)
    return _extract_raw_score(outputs[-1])


def assert_parity(
    reference_fn: Callable[[np.ndarray], np.ndarray],
    onnx_session: ort.InferenceSession,
    X: np.ndarray,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    strict: bool = True,
) -> ParityReport:
    reference = np.asarray(reference_fn(X)).reshape(-1)
    onnx_out = run_onnx_raw_score(onnx_session, X)
    diff = np.abs(reference - onnx_out)
    max_abs = float(diff.max()) if len(diff) else 0.0
    rel = diff / np.maximum(np.abs(reference), 1e-12)
    max_rel = float(rel.max()) if len(rel) else 0.0
    passed = bool(np.allclose(reference, onnx_out, atol=atol, rtol=rtol))
    report = ParityReport(
        max_abs_diff=max_abs, max_rel_diff=max_rel, n_samples=len(reference), passed=passed
    )
    if strict and not passed:
        raise ParityError(f"Paridad fallida: max_abs_diff={max_abs}, max_rel_diff={max_rel}")
    return report


class OnnxModelExporter:
    def export(self, model: BaseModel, X_sample: np.ndarray, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        X_sample = np.asarray(X_sample, dtype=np.float32)

        if model.framework == "sklearn":
            onnx_model = to_onnx(
                model.estimator, X_sample, options={id(model.estimator): {"zipmap": False}}
            )
            output_path.write_bytes(onnx_model.SerializeToString())
        elif model.framework == "catboost":
            model.estimator.save_model(str(output_path), format="onnx")
        elif model.framework == "torch":
            model.network.eval()
            dummy = torch.from_numpy(X_sample[:1])
            torch.onnx.export(
                model.network,
                dummy,
                str(output_path),
                input_names=["input"],
                output_names=["raw_score"],
                dynamic_axes={"input": {0: "batch"}, "raw_score": {0: "batch"}},
                opset_version=17,
            )
        else:
            raise ValueError(f"Framework desconocido: {model.framework}")
        return output_path


def _make_affine_sigmoid_graph(a: float, b: float) -> onnx.ModelProto:
    input_tensor = helper.make_tensor_value_info("raw_score", TensorProto.FLOAT, [None])
    output_tensor = helper.make_tensor_value_info("fraud_probability", TensorProto.FLOAT, [None])
    a_init = helper.make_tensor("a", TensorProto.FLOAT, [], [a])
    b_init = helper.make_tensor("b", TensorProto.FLOAT, [], [b])
    nodes = [
        helper.make_node("Mul", ["raw_score", "a"], ["scaled"]),
        helper.make_node("Add", ["scaled", "b"], ["shifted"]),
        helper.make_node("Sigmoid", ["shifted"], ["fraud_probability"]),
    ]
    graph = helper.make_graph(
        nodes, "calibrator_affine_sigmoid", [input_tensor], [output_tensor], [a_init, b_init]
    )
    return _runtime_compatible_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    )


def _make_lut_graph(lut_min: float, lut_max: float, lut_values: np.ndarray) -> onnx.ModelProto:
    """Isotonic via lookup table densa: ponytail, resolucion = (lut_max-lut_min)/(len(lut_values)-1)
    sobre el rango de raw_score observado en VALID. Upgrade: mas puntos si hace falta mas precision."""
    n_points = len(lut_values)
    input_tensor = helper.make_tensor_value_info("raw_score", TensorProto.FLOAT, [None])
    output_tensor = helper.make_tensor_value_info("fraud_probability", TensorProto.FLOAT, [None])
    lut_init = helper.make_tensor(
        "lut", TensorProto.FLOAT, [n_points], lut_values.astype(np.float32).tolist()
    )
    min_init = helper.make_tensor("lut_min", TensorProto.FLOAT, [], [lut_min])
    scale_init = helper.make_tensor(
        "lut_scale", TensorProto.FLOAT, [], [(n_points - 1) / max(lut_max - lut_min, 1e-12)]
    )
    zero_init = helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
    max_idx_init = helper.make_tensor("max_idx", TensorProto.FLOAT, [], [float(n_points - 1)])
    nodes = [
        helper.make_node("Sub", ["raw_score", "lut_min"], ["centered"]),
        helper.make_node("Mul", ["centered", "lut_scale"], ["scaled_idx"]),
        helper.make_node("Round", ["scaled_idx"], ["rounded_idx"]),
        helper.make_node("Max", ["rounded_idx", "zero"], ["clipped_low"]),
        helper.make_node("Min", ["clipped_low", "max_idx"], ["clipped_idx"]),
        helper.make_node("Cast", ["clipped_idx"], ["int_idx"], to=TensorProto.INT64),
        helper.make_node("Gather", ["lut", "int_idx"], ["fraud_probability"]),
    ]
    graph = helper.make_graph(
        nodes,
        "calibrator_isotonic_lut",
        [input_tensor],
        [output_tensor],
        [lut_init, min_init, scale_init, zero_init, max_idx_init],
    )
    return _runtime_compatible_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    )


class OnnxCalibratorExporter:
    def export(self, calibrator: object, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        method = calibrator.__class__.__name__
        if method == "PlattScaler":
            model = _make_affine_sigmoid_graph(calibrator.params["a"], calibrator.params["b"])  # type: ignore[attr-defined]
        elif method == "TemperatureScaler":
            temperature = calibrator.params["T"]  # type: ignore[attr-defined]
            model = _make_affine_sigmoid_graph(1.0 / temperature, 0.0)
        elif method == "IsotonicCalibrator":
            model = _make_lut_graph(
                calibrator.lut_min_, calibrator.lut_max_, calibrator.lut_values_
            )  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Calibrador desconocido: '{method}'")
        onnx.save(model, str(output_path))
        return output_path
