"""Train an MLP neural network for the UCI Airfoil Self-Noise dataset.

Input: data/airfoil_self_noise.dat (raw, whitespace-separated, 6 columns, no header).
Split: 80% training / 20% untouched test. validation.py performs 5-fold CV within the 80%.
Plots are displayed interactively; no plots, CSVs, JSONs, or reports are saved.
Only the fitted model and split indices are saved in models/ so the other scripts can reuse them.
"""
from pathlib import Path
import hashlib
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "airfoil_nn.joblib"
DATA_FILE = DATA_DIR / "airfoil_self_noise.dat"
FEATURES = [
    "frequency_hz",
    "angle_of_attack_deg",
    "chord_length_m",
    "free_stream_velocity_m_s",
    "suction_side_displacement_thickness_m",
]
TARGET = "sound_pressure_level_db"
COLUMNS = FEATURES + [TARGET]


def load_dataset():
    """Read the original .dat file; allow it in project root for convenience too."""
    candidates = [DATA_FILE, ROOT / "airfoil_self_noise.dat"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None and DATA_DIR.is_dir():
        matches = sorted(DATA_DIR.glob("*.dat"))
        if len(matches) == 1:
            path = matches[0]
    if path is None:
        raise FileNotFoundError(
            "Dataset not found. Put airfoil_self_noise.dat in the project's data folder.\n"
            f"Expected path: {DATA_FILE}\n"
            "For convenience, the script also accepts the file in the project root."
        )
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#")
    if df.shape[1] != 6:
        raise ValueError(
            f"Expected 6 whitespace-separated columns in {path.name}; found {df.shape[1]}. "
            "Use the raw headerless .dat dataset, not a webpage or a formatted table."
        )
    df.columns = COLUMNS
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any() or not np.isfinite(df.to_numpy()).all():
        raise ValueError("The dataset contains missing/non-numeric values. Check the raw .dat file.")
    if len(df) < 100:
        warnings.warn("Fewer than 100 rows were loaded; results may be unreliable.")
    positive = ["frequency_hz", "chord_length_m", "free_stream_velocity_m_s",
                "suction_side_displacement_thickness_m"]
    if (df[positive] <= 0).any().any():
        raise ValueError("Frequency, chord, velocity and displacement thickness must be positive.")
    return df, path


def metric_summary(actual, predicted):
    """Calculate prediction errors in the target's decibel units.

    Parameters
    ----------
    actual, predicted : array-like
        Measured and predicted sound-pressure levels with matching lengths.

    Returns
    -------
    dict[str, float]
        Mean absolute error, root mean squared error, and coefficient of determination.
    """
    return {
        "mae_db": float(mean_absolute_error(actual, predicted)),
        "rmse_db": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def build_model():
    """Input and target scaling are fitted only on rows supplied to model.fit."""
    regressor = Pipeline([
        ("x_scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=2500,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=40,
            random_state=42,
        )),
    ])
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


def main():
    """Train the neural network, save the model bundle, and show diagnostics."""
    df, data_path = load_dataset()
    all_indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        all_indices, test_size=0.20, shuffle=True, random_state=42
    )
    X_train = df.iloc[train_idx][FEATURES]
    y_train = df.iloc[train_idx][TARGET]

    model = build_model()
    print("\n" + "=" * 72)
    print("AIRFOIL SELF-NOISE | NEURAL NETWORK TRAINING")
    print("=" * 72)
    print(f"Dataset: {data_path}")
    print(f"Observations: {len(df)}")
    print(f"Training rows: {len(train_idx)} ({len(train_idx)/len(df):.1%})")
    print(f"Untouched test rows: {len(test_idx)} ({len(test_idx)/len(df):.1%})")
    print("Validation plan: 5-fold cross-validation within the training 80%.")
    print("Network: 5 inputs -> 64 neurons -> 32 neurons -> 1 predicted SPL value")
    print("\nFitting model...\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)
    convergence = [w for w in caught if issubclass(w.category, ConvergenceWarning)]

    train_pred = model.predict(X_train)
    m = metric_summary(y_train, train_pred)
    fitted_mlp = model.regressor_.named_steps["mlp"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "columns": COLUMNS,
        "train_idx": np.asarray(train_idx, dtype=int),
        "test_idx": np.asarray(test_idx, dtype=int),
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "dataset_filename": data_path.name,
        "n_rows": len(df),
        "split_seed": 42,
        "train_metrics": m,
        "model_description": "StandardScaler -> MLPRegressor(64, 32), target StandardScaler",
    }
    joblib.dump(bundle, MODEL_FILE)

    print("TRAINING FIT (diagnostic only; optimistic because the model saw these rows)")
    print(f"MAE: {m['mae_db']:.3f} dB | RMSE: {m['rmse_db']:.3f} dB | R²: {m['r2']:.4f}")
    print(f"Optimizer iterations: {fitted_mlp.n_iter_}")
    if convergence:
        print("WARNING: MLP reached max_iter before full convergence. Interpret scores carefully.")
    print(f"\nSaved model and split indices: {MODEL_FILE}")
    print("No plot image, CSV, JSON, or report files are saved. Close the plot window to finish.")

    # Plot 1: optimisation learning curve.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(np.arange(1, len(fitted_mlp.loss_curve_) + 1), fitted_mlp.loss_curve_)
    axes[0].set_xlabel("Training iteration")
    axes[0].set_ylabel("Loss (standardised target scale)")
    axes[0].set_title("Neural-network training loss")
    axes[0].grid(alpha=0.25)
    val_scores = getattr(fitted_mlp, "validation_scores_", None)
    if val_scores is not None and len(val_scores):
        axes[1].plot(np.arange(1, len(val_scores) + 1), val_scores)
        axes[1].set_xlabel("Iteration with internal validation check")
        axes[1].set_ylabel("Internal validation R²")
        axes[1].set_title("Early-stopping validation score")
        axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    else:
        axes[1].text(0.5, 0.5, "Internal validation scores unavailable", ha="center", va="center")
        axes[1].set_axis_off()
    axes[1].grid(alpha=0.25)
    fig.suptitle("Training diagnostics (not the final generalisation score)")
    fig.tight_layout()
    print("\nPLOT EXPLANATION — TRAINING DIAGNOSTICS")
    print("Left: loss should generally fall as the network learns; a plateau suggests it has stopped improving. A noisy or still-falling curve can mean optimisation has not fully settled.")
    print("Right: the MLP's internal validation R² helps its early-stopping rule. A rise then plateau is common. It is not the held-out 20% test score; validation.py gives a fuller cross-validation estimate.")
    plt.show()

    print("\nNext run: python validation.py")


if __name__ == "__main__":
    main()
