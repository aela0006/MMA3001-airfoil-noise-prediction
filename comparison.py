"""Compare the trained MLP against simple regression baselines.

The script reuses the model bundle and its saved 80/20 split from ``train.py``.
Both baseline estimators are fitted using only the training partition, then all
models are evaluated on the same untouched test partition. Results and plots
are saved under ``outputs/comparison/`` for reproducibility and reporting.

Run from the project root with::

    python comparison.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train import FEATURES, MODEL_FILE, TARGET, load_dataset, metric_summary

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "comparison"

MLP_LABEL = "MLP neural network"
LINEAR_LABEL = "Linear regression"
MEAN_LABEL = "Training-mean baseline"


def build_baseline_models() -> dict[str, Any]:
    """Create baseline estimators; callers fit them on training data only.

    Returns
    -------
    dict[str, estimator]
        A training-target-mean predictor and a standardised ordinary least
        squares regression model. StandardScaler is fitted within the training
        partition by the pipeline, preventing preprocessing leakage.
    """
    return {
        MEAN_LABEL: DummyRegressor(strategy="mean"),
        LINEAR_LABEL: Pipeline(
            steps=[
                ("x_scaler", StandardScaler()),
                ("linear_regression", LinearRegression()),
            ]
        ),
    }


def validate_split_indices(
    n_rows: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Validate that train/test indices form a complete, disjoint row split.

    Parameters
    ----------
    n_rows : int
        Number of observations in the loaded dataset.
    train_idx, test_idx : numpy.ndarray
        Integer row indices stored in the trained model bundle.

    Raises
    ------
    ValueError
        If indices are empty, duplicated, out of bounds, overlap, or do not
        cover every dataset row exactly once.
    """
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    if train_idx.ndim != 1 or test_idx.ndim != 1:
        raise ValueError("Training and test indices must be one-dimensional.")
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Training and test partitions must both be non-empty.")
    if not np.issubdtype(train_idx.dtype, np.integer) or not np.issubdtype(
        test_idx.dtype, np.integer
    ):
        raise ValueError("Training and test indices must contain integers.")
    if np.unique(train_idx).size != train_idx.size:
        raise ValueError("Training indices contain duplicates.")
    if np.unique(test_idx).size != test_idx.size:
        raise ValueError("Test indices contain duplicates.")
    if (train_idx < 0).any() or (test_idx < 0).any():
        raise ValueError("Row indices cannot be negative.")
    if (train_idx >= n_rows).any() or (test_idx >= n_rows).any():
        raise ValueError("A row index is outside the loaded dataset.")
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("Training and test indices overlap.")
    combined = np.concatenate((train_idx, test_idx))
    if combined.size != n_rows or not np.array_equal(
        np.sort(combined), np.arange(n_rows)
    ):
        raise ValueError("Training and test indices must cover every row exactly once.")


def load_project_data(
    model_file: Path = MODEL_FILE,
) -> tuple[dict[str, Any], pd.DataFrame, Path, np.ndarray, np.ndarray]:
    """Load the trained model and verify its dataset and split metadata.

    The dataset checksum must match the checksum stored by ``train.py``. This
    protects the comparison from silently using a replaced or reordered file.

    Raises
    ------
    FileNotFoundError
        If the trained model bundle or input dataset is missing.
    ValueError
        If the dataset checksum, schema, or stored split is inconsistent.
    """
    if not model_file.is_file():
        raise FileNotFoundError(
            "Trained model bundle not found. Run `python train.py` first. "
            f"Expected: {model_file}"
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = joblib.load(model_file)
    version_warnings = [
        warning
        for warning in caught
        if "InconsistentVersionWarning" in warning.category.__name__
    ]
    if version_warnings:
        raise RuntimeError(
            "The saved model was created with a different scikit-learn version. "
            "For a trustworthy comparison, install the version pinned in "
            "requirements.txt and rerun this script. Alternatively, retrain the "
            "MLP with the active environment by running train.py first, then rerun "
            "validation.py, test.py and comparison.py so all results use the same "
            "model and environment. The comparison was stopped to avoid reporting "
            "potentially incompatible predictions."
        )

    if not isinstance(bundle, dict):
        raise ValueError("The model file does not contain the expected model bundle.")
    for key in ("model", "train_idx", "test_idx", "dataset_sha256"):
        if key not in bundle:
            raise ValueError(f"Model bundle is missing required field: {key!r}.")

    df, data_path = load_dataset()
    actual_checksum = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if actual_checksum != bundle["dataset_sha256"]:
        raise ValueError(
            "The dataset differs from the file used during training, including "
            "possible row-order changes. Restore the original .dat file or rerun "
            "train.py with the intended data before comparing models."
        )
    if bundle.get("features") != FEATURES or bundle.get("target") != TARGET:
        raise ValueError("The saved model schema does not match the current project schema.")
    if len(df) != int(bundle.get("n_rows", -1)):
        raise ValueError("The loaded dataset row count differs from the model bundle.")

    train_idx = np.asarray(bundle["train_idx"])
    test_idx = np.asarray(bundle["test_idx"])
    validate_split_indices(len(df), train_idx, test_idx)
    return (
        bundle,
        df,
        data_path,
        train_idx.astype(int, copy=False),
        test_idx.astype(int, copy=False),
    )


def evaluate_models(
    fitted_mlp: Any,
    baseline_models: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit baselines and evaluate every estimator on the same held-out test set.

    The supplied MLP is already trained by ``train.py``. Baselines are fitted
    here using training rows only. No estimator is fitted on test targets.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        A metrics table and a prediction table, each with one row per test case
        in the latter and one row per model in the former.
    """
    if X_train.empty or X_test.empty:
        raise ValueError("Training and test feature tables must be non-empty.")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and target lengths do not match.")
    if list(X_train.columns) != FEATURES or list(X_test.columns) != FEATURES:
        raise ValueError("Input feature columns do not match the project feature order.")

    models: dict[str, Any] = {MLP_LABEL: fitted_mlp}
    for label, estimator in baseline_models.items():
        estimator.fit(X_train, y_train)
        models[label] = estimator

    y_actual = np.asarray(y_test, dtype=float).reshape(-1)
    predictions = pd.DataFrame({"actual_spl_db": y_actual})
    predictions.insert(0, "test_position", np.arange(len(y_actual)))
    metrics_rows: list[dict[str, float | str]] = []

    for label, estimator in models.items():
        y_pred = np.asarray(estimator.predict(X_test), dtype=float).reshape(-1)
        if y_pred.shape != y_actual.shape or not np.isfinite(y_pred).all():
            raise ValueError(f"{label} produced invalid or non-finite predictions.")
        summary = metric_summary(y_actual, y_pred)
        metrics_rows.append(
            {
                "model": label,
                "mae_db": summary["mae_db"],
                "rmse_db": summary["rmse_db"],
                "r2": summary["r2"],
            }
        )
        safe_label = label.lower().replace(" ", "_").replace("-", "_")
        predictions[f"{safe_label}_predicted_spl_db"] = y_pred

    metrics = pd.DataFrame(metrics_rows).sort_values("rmse_db").reset_index(drop=True)
    return metrics, predictions


def save_comparison_outputs(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, Path, Path, Path]:
    """Save metric/prediction CSVs and comparison plots for reporting.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]
        Paths to the metrics CSV, prediction CSV, metric PNG, and prediction PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "baseline_comparison_metrics.csv"
    predictions_path = output_dir / "baseline_comparison_predictions.csv"
    metrics_plot_path = output_dir / "baseline_comparison_metrics.png"
    predictions_plot_path = output_dir / "baseline_comparison_predictions.png"

    metrics.to_csv(metrics_path, index=False, float_format="%.8f")
    predictions.to_csv(predictions_path, index=False, float_format="%.8f")

    # Both error metrics share dB units; R² is plotted separately because it is unitless.
    labels = metrics["model"].tolist()
    x = np.arange(len(metrics))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    axes[0].bar(x - width / 2, metrics["mae_db"], width, label="MAE")
    axes[0].bar(x + width / 2, metrics["rmse_db"], width, label="RMSE")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylabel("Error (dB; lower is better)")
    axes[0].set_title("Prediction errors")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    r2_bars = axes[1].bar(x, metrics["r2"])
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].bar_label(
        r2_bars,
        labels=[f"{value:.3f}" for value in metrics["r2"]],
        padding=3,
        fontsize=9,
    )
    axes[1].set_ylim(min(-0.05, float(metrics["r2"].min()) - 0.05),
                     max(1.0, float(metrics["r2"].max()) + 0.05))
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylabel("R² (higher is better)")
    axes[1].set_title("Coefficient of determination")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Baseline comparison on the same untouched test set")
    fig.tight_layout()
    fig.savefig(metrics_plot_path, dpi=200, bbox_inches="tight")

    prediction_columns = [
        column
        for column in predictions.columns
        if column.endswith("_predicted_spl_db")
    ]
    y_actual = predictions["actual_spl_db"].to_numpy()
    all_values = [y_actual]
    for column in prediction_columns:
        all_values.append(predictions[column].to_numpy())
    low = float(min(np.min(values) for values in all_values))
    high = float(max(np.max(values) for values in all_values))

    fig2, axes2 = plt.subplots(1, len(prediction_columns), figsize=(5.1 * len(prediction_columns), 4.8), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes2)
    for ax, column in zip(axes_array, prediction_columns):
        model_label = column.removesuffix("_predicted_spl_db").replace("_", " ").title()
        ax.scatter(y_actual, predictions[column], alpha=0.58, s=18, edgecolors="none")
        ax.plot([low, high], [low, high], "k--", linewidth=1.2, label="Perfect prediction")
        ax.set_title(model_label)
        ax.set_xlabel("Measured SPL (dB)")
        ax.grid(alpha=0.2)
    axes_array[0].set_ylabel("Predicted SPL (dB)")
    axes_array[-1].legend(loc="best")
    fig2.suptitle("Predicted vs measured sound pressure level")
    fig2.tight_layout()
    fig2.savefig(predictions_plot_path, dpi=200, bbox_inches="tight")

    return metrics_path, predictions_path, metrics_plot_path, predictions_plot_path


def main(show_plots: bool = True) -> None:
    """Run the held-out comparison and save reproducible outputs."""
    bundle, df, data_path, train_idx, test_idx = load_project_data()
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    baselines = build_baseline_models()
    metrics, predictions = evaluate_models(
        bundle["model"], baselines, X_train, y_train, X_test, y_test
    )
    # Keep the original raw dataset row index visible in the saved predictions.
    predictions.insert(1, "dataset_row_index", test_idx)
    (
        metrics_path,
        predictions_path,
        metrics_plot_path,
        predictions_plot_path,
    ) = save_comparison_outputs(metrics, predictions)

    print("\n" + "=" * 84)
    print("AIRFOIL SELF-NOISE | BASELINE COMPARISON")
    print("=" * 84)
    print(f"Dataset: {data_path}")
    print(f"Observations: {len(df)} | Training: {len(train_idx)} | Untouched test: {len(test_idx)}")
    print("All models were evaluated on the same saved test rows.")
    print("The MLP was loaded from the existing model bundle; baseline models used training rows only.\n")
    print(
        metrics.to_string(
            index=False,
            formatters={
                "mae_db": lambda value: f"{value:.3f}",
                "rmse_db": lambda value: f"{value:.3f}",
                "r2": lambda value: f"{value:.4f}",
            },
        )
    )

    mlp_row = metrics.loc[metrics["model"] == MLP_LABEL].iloc[0]
    print("\nMLP RMSE change relative to each baseline (positive = lower MLP RMSE):")
    for baseline_label in (LINEAR_LABEL, MEAN_LABEL):
        baseline_row = metrics.loc[metrics["model"] == baseline_label].iloc[0]
        denominator = float(baseline_row["rmse_db"])
        if denominator > 0:
            change_pct = (denominator - float(mlp_row["rmse_db"])) / denominator * 100.0
            print(f"  vs {baseline_label}: {change_pct:+.2f}%")
        else:
            print(f"  vs {baseline_label}: percentage unavailable (baseline RMSE is zero).")

    print("\nSaved outputs:")
    print(f"  Metrics:       {metrics_path.relative_to(ROOT)}")
    print(f"  Predictions:   {predictions_path.relative_to(ROOT)}")
    print(f"  Metrics plot:  {metrics_plot_path.relative_to(ROOT)}")
    print(f"  Parity plots:  {predictions_plot_path.relative_to(ROOT)}")
    print("\nInterpretation reminder: lower MAE/RMSE and higher R² indicate better test-set performance.")
    print("A better score on this split supports model selection for this dataset; it does not prove generalisation to other airfoil designs or wind-tunnel conditions.")

    if show_plots:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
