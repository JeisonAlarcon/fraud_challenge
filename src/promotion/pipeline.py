"""Pipeline de promoción reutilizable para notebook y línea de comandos.

``promote_winner`` contiene la operación de negocio. ``promotion.cli.main`` solo adapta los
argumentos del comando ``fraud-promote-winner`` a esta función.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import yaml

from data.manifest import DatasetManifest, get_git_sha
from evaluation.calibration import CalibrationRunner, ThresholdSelector
from evaluation.metrics import BinaryClassificationMetrics
from evaluation.model_card import write_model_card
from evaluation.plots import (
    plot_candidate_comparison,
    plot_pr_curve,
    plot_roc_curve,
    save_figure,
)
from evaluation.selection import (
    bootstrap_pr_auc_ci,
    check_segment_support,
    check_temporal_stability,
    measure_latency,
    select_winner,
)
from features.cache import FeatureCacheResolver
from features.schema import load_feature_version
from models.registry import MODEL_REGISTRY
from serving.contract import (
    FeatureColumnSpec,
    FeatureContract,
    ThresholdContract,
    ThresholdPolicy,
)
from serving.onnx_exporter import (
    OnnxCalibratorExporter,
    OnnxModelExporter,
    assert_parity,
    run_onnx_raw_score,
)
from serving.onnx_scorer import OnnxScorer


def _dtype_label(pandas_dtype: str) -> str:
    if "int" in pandas_dtype:
        return "int64"
    if "float" in pandas_dtype:
        return "float64"
    if "bool" in pandas_dtype:
        return "bool"
    return "string"


def load_model_for_run(run_dir: Path, model_name: str, model_version: str):
    model_class = MODEL_REGISTRY[model_version][model_name]
    metadata = json.loads((run_dir / "model" / "metadata.json").read_text())
    return model_class.load(run_dir / "model" / metadata["model_filename"])


def load_processed_parquet(manifest: DatasetManifest, data_root: Path) -> pd.DataFrame:
    """Carga el Parquet usando la ruta del manifest o su equivalente local."""
    manifest_path = Path(manifest.output_parquet.path)
    candidates = [
        manifest_path,
        data_root / "processed" / manifest_path.name,
        data_root / "processed" / "fraud_clean_v1.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return pd.read_parquet(candidate)
    raise FileNotFoundError(
        "No se encontró el Parquet procesado. Rutas revisadas: "
        + ", ".join(str(path) for path in candidates)
    )


def promote_winner(
    runs_dir: Path,
    config_path: Path,
    data_root: Path,
    configs_root: Path,
    promoted_root: Path,
    override_winner_run_id: str | None = None,
    force_test_rerun: bool = False,
) -> Path:
    """Promueve un run y devuelve el directorio de artefactos publicados.

    La función se puede llamar desde un notebook que ya tiene el entorno cargado.
    ``promotion.cli`` sigue siendo el punto de entrada para ejecuciones desde terminal.
    """
    promotion_cfg = yaml.safe_load(config_path.read_text())
    selection_cfg = promotion_cfg.get("selection", {})

    winner = select_winner(
        runs_dir,
        min_segment_support=selection_cfg.get("min_segment_support", 30),
        override_run_id=override_winner_run_id,
    )
    winner_run_id = winner.winner_run_id
    run_dir = runs_dir / winner_run_id
    metadata = json.loads((run_dir / "model" / "metadata.json").read_text())
    epoch_metrics_final = json.loads((run_dir / "metrics" / "epoch_metrics.json").read_text()).get(
        "final", {}
    )
    ensemble_weights = epoch_metrics_final.get("voting_weights") or epoch_metrics_final.get(
        "stacking_meta_coefficients"
    )
    print(f"[promote_winner] winner_run_id = {winner_run_id}")

    promoted_dir = promoted_root / winner_run_id
    promoted_dir.mkdir(parents=True, exist_ok=True)

    winner.comparison_table.to_csv(promoted_dir / "candidate_comparison.csv", index=False)
    fig, ax = plt.subplots()
    plot_candidate_comparison(winner.comparison_table, ax=ax)
    save_figure(fig, promoted_dir / "candidate_comparison.png")
    plt.close(fig)

    manifest = DatasetManifest.from_json(data_root / "processed" / "manifest.json")
    feature_config = load_feature_version(
        configs_root / "feature_versions" / f"{metadata['feature_version']}.yml"
    )
    cache_paths = FeatureCacheResolver().resolve(
        metadata["feature_version"], manifest, feature_config, data_root / "cache" / "features"
    )

    def _dense_f32(X: object) -> np.ndarray:
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    X_valid, y_valid = cache_paths.load_split("valid")
    X_test, y_test = cache_paths.load_split("test")
    X_valid, X_test = _dense_f32(X_valid), _dense_f32(X_test)

    raw_df = load_processed_parquet(manifest, data_root)

    def _aligned_amounts(dataset_type: str, y_split: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        split_df = raw_df.loc[raw_df["dataset_type"] == dataset_type].reset_index(drop=True)
        labels = split_df[feature_config.target].to_numpy(dtype=int)
        if len(split_df) != len(y_split) or not np.array_equal(
            labels, np.asarray(y_split).astype(int)
        ):
            raise RuntimeError(
                f"Filas de {dataset_type} desalineadas entre Parquet y cache de features"
            )
        amounts = split_df["monto"].to_numpy(dtype=float)
        if not np.all(np.isfinite(amounts)) or np.any(amounts < 0):
            raise ValueError(f"monto contiene valores inválidos en {dataset_type}")
        return split_df, amounts

    valid_df, amount_valid = _aligned_amounts("VALID", y_valid)
    _test_df, amount_test = _aligned_amounts("TEST", y_test)

    model = load_model_for_run(run_dir, metadata["model"], metadata["model_version"])

    # 1) Export modelo -> model_raw.onnx + Paridad 1
    model_raw_path = OnnxModelExporter().export(model, X_valid[:5], promoted_dir / "model_raw.onnx")
    model_session = ort.InferenceSession(str(model_raw_path))
    assert_parity(model.raw_score, model_session, X_valid)
    print("[promote_winner] Paridad 1 (modelo .pkl vs model_raw.onnx): OK")

    # 2) Calibracion sobre scores crudos de VALID (via ONNX)
    calibration_cfg = promotion_cfg.get("calibration", {})
    calib_result = CalibrationRunner(
        model_raw_path, X_valid, y_valid, method=calibration_cfg.get("method", "auto")
    ).run(promoted_dir, framework=model.framework)
    (promoted_dir / "calibration_metrics.json").write_text(
        json.dumps(
            {
                "method": calib_result.method_used,
                "pre_metrics": calib_result.pre_metrics,
                "post_metrics": calib_result.post_metrics,
            },
            indent=2,
            default=str,
        )
    )

    # 3) Export calibrador -> calibrator.onnx + Paridad 2
    calibrator_onnx_path = OnnxCalibratorExporter().export(
        calib_result.calibrator, promoted_dir / "calibrator.onnx"
    )
    calibrator_session = ort.InferenceSession(str(calibrator_onnx_path))
    raw_scores_valid = run_onnx_raw_score(model_session, X_valid)
    assert_parity(calib_result.calibrator.predict_proba, calibrator_session, raw_scores_valid)
    print("[promote_winner] Paridad 2 (calibrator.pkl vs calibrator.onnx): OK")

    # 4) Paridad 3: cadena completa .pkl+calibrador vs onnx+onnx
    chain_pkl_scores = calib_result.calibrator.predict_proba(model.raw_score(X_valid))
    model_raw_scores_valid = run_onnx_raw_score(model_session, X_valid)
    chain_onnx_scores = run_onnx_raw_score(calibrator_session, model_raw_scores_valid)
    if not np.allclose(chain_pkl_scores, chain_onnx_scores, atol=1e-3, rtol=1e-2):
        raise RuntimeError("Paridad 3 (cadena completa .pkl+calibrador vs ONNX+ONNX) fallida")
    print("[promote_winner] Paridad 3 (cadena completa): OK")

    # 5) Umbral sobre probabilidad calibrada (VALID). Si no hay politica de negocio (ver
    # configs/promotion/v1.yml), igual se publican las curvas y la tabla completa de umbrales.
    policy = ThresholdPolicy(**promotion_cfg["policy"])
    threshold_result = ThresholdSelector().select(
        y_valid,
        chain_onnx_scores,
        policy,
        amounts=amount_valid,
    )

    reference_curves = {}
    fig, ax = plt.subplots()
    plot_pr_curve(y_valid, chain_onnx_scores, ax=ax)
    reference_curves["pr_curve"] = str(save_figure(fig, promoted_dir / "pr_curve.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_roc_curve(y_valid, chain_onnx_scores, ax=ax)
    reference_curves["roc_curve"] = str(save_figure(fig, promoted_dir / "roc_curve.png"))
    plt.close(fig)

    threshold_table_path = promoted_dir / "threshold_table.csv"
    threshold_result.threshold_table.to_csv(threshold_table_path, index=False)
    reference_curves["threshold_table"] = str(threshold_table_path)

    if policy.type == "economic_profit":
        economic_table_path = promoted_dir / "economic_gain_curve.csv"
        threshold_result.threshold_table.to_csv(economic_table_path, index=False)
        reference_curves["economic_gain_curve"] = str(economic_table_path)

        fig, ax = plt.subplots()
        ax.plot(
            threshold_result.threshold_table["threshold"],
            threshold_result.threshold_table["profit_total"],
            color="#1f4e79",
        )
        ax.axvline(threshold_result.threshold, color="#c00000", linestyle="--")
        ax.set_xlabel("Threshold de rechazo")
        ax.set_ylabel("Ganancia total")
        ax.set_title("Ganancia económica en VALID")
        fig.tight_layout()
        reference_curves["economic_gain_curve_plot"] = str(
            save_figure(fig, promoted_dir / "economic_gain_curve.png")
        )
        plt.close(fig)
        (promoted_dir / "economic_metrics_valid.json").write_text(
            json.dumps(
                {
                    "policy": policy.model_dump(),
                    "selected_threshold": threshold_result.threshold,
                    "metrics": threshold_result.valid_metrics_at_threshold,
                },
                indent=2,
                default=str,
            )
        )

    # 6) feature_contract.json / threshold.json
    input_schema = json.loads(cache_paths.input_schema.read_text())
    feature_names = json.loads(cache_paths.feature_names.read_text())
    input_columns = [
        FeatureColumnSpec(
            name=name, dtype=_dtype_label(info["dtype"]), role=info["role"], nullable=True
        )
        for name, info in input_schema.items()
    ]
    excluded_columns = [
        {"name": f.name, "role": f.role.value, "reason": f.reason}
        for f in feature_config.features
        if not f.enabled
    ]
    preprocessor_hash = hashlib.sha256(cache_paths.preprocessor.read_bytes()).hexdigest()

    feature_contract = FeatureContract(
        feature_version=metadata["feature_version"],
        feature_cache_id=metadata["feature_cache_id"],
        target=feature_config.target,
        timestamp_column=feature_config.timestamp,
        input_columns=input_columns,
        excluded_columns=excluded_columns,
        preprocessor_output_dim=len(feature_names),
        preprocessor_hash=preprocessor_hash,
        source_manifest_hash=manifest.output_parquet.sha256,
        generated_at=datetime.now(UTC),
    )
    (promoted_dir / "feature_contract.json").write_text(feature_contract.model_dump_json(indent=2))

    threshold_contract = ThresholdContract(
        winner_run_id=winner_run_id,
        feature_version=metadata["feature_version"],
        model_version=metadata["model_version"],
        hyperparameters_version=metadata["hyperparameters_version"],
        model=metadata["model"],
        calibration_method=calib_result.method_used,
        threshold=threshold_result.threshold,
        policy=policy,
        valid_metrics_at_threshold=threshold_result.valid_metrics_at_threshold,
        reference_curves=reference_curves,
        calibration_applied=True,
        selected_on="calibrated_probability",
        generated_at=datetime.now(UTC),
        git_sha=get_git_sha(),
        seed=metadata["seed"],
    )
    (promoted_dir / "threshold.json").write_text(threshold_contract.model_dump_json(indent=2))
    shutil.copy(cache_paths.preprocessor, promoted_dir / "preprocessor.pkl")

    # 7) TEST se evalua una unica vez (guardia)
    final_test_metrics_path = promoted_dir / "final_test_metrics.json"
    reuse_test_metrics = final_test_metrics_path.exists() and not force_test_rerun
    if reuse_test_metrics and policy.type == "economic_profit":
        existing_test_metrics = json.loads(final_test_metrics_path.read_text())
        reuse_test_metrics = (
            existing_test_metrics.get("economic", {}).get("threshold") == threshold_result.threshold
        )

    if reuse_test_metrics:
        print(
            "[promote_winner] final_test_metrics.json ya existe; TEST no se recalcula "
            "(usar --force-test-rerun para forzar)."
        )
        final_test_metrics = json.loads(final_test_metrics_path.read_text())
    else:
        scorer = OnnxScorer(promoted_dir)
        test_probs = scorer.score_matrix(X_test)
        assert test_probs.dtype.kind == "f" and test_probs.shape[0] == X_test.shape[0]
        experiment_cfg_path = configs_root / "experiments" / "v1.yml"
        experiment_cfg = yaml.safe_load(experiment_cfg_path.read_text())["metrics"]
        final_test_metrics = BinaryClassificationMetrics.compute(
            y_test,
            test_probs,
            experiment_cfg["thresholds"],
            experiment_cfg["top_k_percent"],
            experiment_cfg.get("calibration_bins", 10),
            amounts=amount_test,
            economic_threshold=threshold_result.threshold,
            legitimate_gain_rate=policy.params.get("legitimate_gain_rate", 0.25),
            fraud_loss_rate=policy.params.get("fraud_loss_rate", 1.0),
        )
        final_test_metrics_path.write_text(json.dumps(final_test_metrics, indent=2, default=str))
        print("[promote_winner] TEST evaluado una unica vez sobre la cadena ONNX completa.")

    # 8) Verificaciones adicionales (bootstrap, latencia) + Model Card
    bootstrap_ci = bootstrap_pr_auc_ci(
        y_valid, chain_onnx_scores, n_boot=selection_cfg.get("bootstrap_n", 1000)
    )
    latency = measure_latency(lambda x: run_onnx_raw_score(model_session, x), X_valid, n=200)

    matching_winner_rows = winner.comparison_table[
        winner.comparison_table["run_id"] == winner_run_id
    ]
    if matching_winner_rows.empty:
        raise RuntimeError(f"run_id promovido no está en candidate_comparison.csv: {winner_run_id}")
    winner_summary = matching_winner_rows.iloc[0]
    context = {
        "model": metadata["model"],
        "model_version": metadata["model_version"],
        "hyperparameters_version": metadata["hyperparameters_version"],
        "feature_version": metadata["feature_version"],
        "winner_run_id": winner_run_id,
        "hyperparameters": metadata["hyperparameters"],
        "ensemble_weights": ensemble_weights,
        "summary": (
            f"Modelo '{metadata['model']}' seleccionado por mayor PR-AUC de validacion "
            f"({winner_summary['valid_pr_auc']:.4f}); "
            f"IC bootstrap 95%: [{bootstrap_ci[0]:.4f}, {bootstrap_ci[1]:.4f}]."
        ),
        "n_rows": manifest.output_parquet.n_rows,
        "date_start": str(manifest.split.train_end),
        "date_end": str(manifest.split.test_end),
        "split_row_counts": manifest.split.row_counts,
        "split_prevalence": manifest.split.prevalence,
        "metrics_summary_rows": [
            {
                "split": "valid_at_best_epoch",
                "average_precision": winner_summary["valid_pr_auc"],
                "roc_auc": winner_summary["valid_roc_auc"],
                "brier": winner_summary["valid_brier"],
            },
            {
                "split": "test (unica vez)",
                "average_precision": final_test_metrics["average_precision"],
                "roc_auc": final_test_metrics["roc_auc"],
                "brier": final_test_metrics["brier"],
            },
        ],
        "calibration_method": calib_result.method_used,
        "calibration_pre_metrics": calib_result.pre_metrics,
        "calibration_post_metrics": calib_result.post_metrics,
        "reliability_fig_path": str(calib_result.reliability_fig_path),
        "threshold": threshold_result.threshold,
        "policy_type": policy.type,
        "policy_defined_by": policy.defined_by,
        "valid_metrics_at_threshold": threshold_result.valid_metrics_at_threshold,
        "economic_metrics_valid": threshold_result.valid_metrics_at_threshold
        if policy.type == "economic_profit"
        else {},
        "economic_metrics_test": final_test_metrics.get("economic", {}),
        "latency": latency,
        "limitations": [
            "'k' excluida por cardinalidad unica (id_like); revisar disponibilidad en "
            "inferencia si se reconsidera.",
            "'score' requiere confirmar disponibilidad pre-decision antes de depender de "
            "ella en produccion.",
            "Ventana de datos ~45 dias; estabilidad temporal fuera de ese rango no esta validada.",
        ],
        "git_sha": get_git_sha(),
        "seed": metadata["seed"],
        "feature_cache_id": metadata["feature_cache_id"],
        "parquet_hash": manifest.output_parquet.sha256,
    }
    segment_support = check_segment_support(
        valid_df,
        selection_cfg.get("segment_columns", []),
        min_support=selection_cfg.get("min_segment_support", 30),
    )
    segment_support_path = promoted_dir / "segment_support.csv"
    segment_support.to_csv(segment_support_path, index=False)
    context["segment_support_path"] = str(segment_support_path)

    temporal_stability = check_temporal_stability(
        y_valid,
        chain_onnx_scores,
        valid_df["fecha"],
        window=selection_cfg.get("temporal_window", "7D"),
    )
    temporal_stability_path = promoted_dir / "temporal_stability.csv"
    temporal_stability.to_csv(temporal_stability_path)
    context["temporal_stability_path"] = str(temporal_stability_path)

    write_model_card(context, promoted_dir / "model_card.md")
    print(f"[promote_winner] Model Card escrito en {promoted_dir / 'model_card.md'}")
    return promoted_dir
