"""Focused unit tests for the Airfoil Self-Noise prediction workflow."""

from importlib.metadata import version
from pathlib import Path
import sys

# Make imports work for both `python -m pytest` and the `pytest` console command.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

import comparison
import train


class ConstantPredictor:
    """Minimal fixed predictor used to test evaluation without model training."""

    def __init__(self, value: float = 110.0) -> None:
        self.value = value

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return one constant prediction for each input row."""
        return np.full(len(features), self.value, dtype=float)


def test_dataset_has_expected_rows_schema_and_finite_values() -> None:
    """The bundled raw dataset should load into the documented numeric schema."""
    data, path = train.load_dataset()

    assert path.name == "airfoil_self_noise.dat"
    assert data.shape == (1503, 6)
    assert list(data.columns) == train.COLUMNS
    assert not data.isna().any().any()
    assert np.isfinite(data.to_numpy(dtype=float)).all()


def test_metric_summary_matches_hand_calculated_example() -> None:
    """MAE, RMSE and R-squared should match a small known example."""
    metrics = train.metric_summary([1.0, 2.0, 3.0], [1.0, 2.0, 2.0])

    assert metrics["mae_db"] == pytest.approx(1.0 / 3.0)
    assert metrics["rmse_db"] == pytest.approx(np.sqrt(1.0 / 3.0))
    assert metrics["r2"] == pytest.approx(0.5)


def test_model_pipeline_uses_scaling_and_expected_mlp_settings() -> None:
    """The configured pipeline should scale features and use the intended MLP."""
    model = train.build_model()
    regressor = model.regressor
    mlp = regressor.named_steps["mlp"]

    assert list(regressor.named_steps) == ["x_scaler", "mlp"]
    assert mlp.hidden_layer_sizes == (64, 32)
    assert mlp.early_stopping is True
    assert mlp.random_state == 42
    assert regressor.named_steps["x_scaler"].__class__.__name__ == "StandardScaler"


def test_split_validator_accepts_complete_disjoint_partition() -> None:
    """A valid training/test split must cover every row exactly once."""
    comparison.validate_split_indices(
        6, np.array([0, 2, 4, 5]), np.array([1, 3])
    )


@pytest.mark.parametrize(
    ("n_rows", "train_idx", "test_idx"),
    [
        (4, np.array([0, 1, 2]), np.array([2, 3])),  # overlap
        (4, np.array([0, 1]), np.array([2])),  # missing row
        (4, np.array([0, 1, 2]), np.array([3, 4])),  # out of bounds
    ],
)
def test_split_validator_rejects_invalid_partitions(
    n_rows: int, train_idx: np.ndarray, test_idx: np.ndarray
) -> None:
    """Overlapping, incomplete and out-of-range splits must be rejected."""
    with pytest.raises(ValueError):
        comparison.validate_split_indices(n_rows, train_idx, test_idx)


def test_evaluation_compares_three_models_on_same_test_rows() -> None:
    """Baseline evaluation should return finite metrics and aligned predictions."""
    train_x = pd.DataFrame(
        np.arange(40, dtype=float).reshape(8, 5) + 1.0,
        columns=train.FEATURES,
    )
    test_x = pd.DataFrame(
        np.arange(15, dtype=float).reshape(3, 5) + 41.0,
        columns=train.FEATURES,
    )
    train_y = pd.Series([100, 101, 103, 105, 108, 110, 111, 115], dtype=float)
    test_y = pd.Series([102, 109, 113], dtype=float)

    metrics, predictions = comparison.evaluate_models(
        ConstantPredictor(), comparison.build_baseline_models(),
        train_x, train_y, test_x, test_y,
    )

    assert len(metrics) == 3
    assert len(predictions) == len(test_y)
    assert {"model", "mae_db", "rmse_db", "r2"}.issubset(metrics.columns)
    assert np.isfinite(metrics[["mae_db", "rmse_db", "r2"]].to_numpy()).all()
    assert np.allclose(predictions["actual_spl_db"], test_y.to_numpy())


def test_saved_model_bundle_when_required_sklearn_version_is_installed() -> None:
    """The trained bundle should load and predict when its pinned version is used."""
    if version("scikit-learn") != "1.9.1":
        pytest.skip(
            "Saved model uses scikit-learn 1.9.1; install requirements.txt to run this integration test."
        )

    bundle, data, _, train_idx, test_idx = comparison.load_project_data()
    predictions = bundle["model"].predict(data.iloc[test_idx][train.FEATURES])

    assert len(data) == bundle["n_rows"] == 1503
    assert len(train_idx) == 1202
    assert len(test_idx) == 301
    assert predictions.shape == (len(test_idx),)
    assert np.isfinite(predictions).all()
