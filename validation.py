"""Validation via 5-fold cross-validation within the 80% training partition.
Plots are shown on screen and not written to disk.
"""
from pathlib import Path
import hashlib
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_FILE = ROOT / "models" / "airfoil_nn.joblib"
FEATURES = ["frequency_hz", "angle_of_attack_deg", "chord_length_m",
            "free_stream_velocity_m_s", "suction_side_displacement_thickness_m"]
TARGET = "sound_pressure_level_db"
COLUMNS = FEATURES + [TARGET]


def load_dataset(bundle):
    """Load the raw data and verify it matches the saved training checksum.

    Parameters
    ----------
    bundle : dict
        Model metadata produced by ``train.py``; must contain ``dataset_sha256``.

    Returns
    -------
    pandas.DataFrame
        Validated six-column dataset in the project's fixed column order.

    Raises
    ------
    FileNotFoundError
        If the source dataset is missing.
    ValueError
        If the dataset checksum, shape, or values are invalid.
    """
    candidates = [DATA_DIR / "airfoil_self_noise.dat", ROOT / "airfoil_self_noise.dat"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None and DATA_DIR.is_dir():
        matches = sorted(DATA_DIR.glob("*.dat"))
        if len(matches) == 1:
            path = matches[0]
    if path is None:
        raise FileNotFoundError(f"Place airfoil_self_noise.dat inside {DATA_DIR}.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != bundle["dataset_sha256"]:
        raise ValueError("Dataset file differs from the one used for training (including row order). Rerun train.py with the intended .dat file.")
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#")
    if df.shape[1] != 6:
        raise ValueError(f"Expected six columns in the raw .dat file, found {df.shape[1]}.")
    df.columns = COLUMNS
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any() or not np.isfinite(df.to_numpy()).all():
        raise ValueError("Dataset has missing or non-numeric values.")
    return df


def main():
    """Run five-fold cross-validation using only the saved training partition."""
    if not MODEL_FILE.is_file():
        raise FileNotFoundError("Run train.py first so the trained model and 80/20 split are saved.")
    bundle = joblib.load(MODEL_FILE)
    df = load_dataset(bundle)
    train_idx = np.asarray(bundle["train_idx"], dtype=int)
    X = df.iloc[train_idx][FEATURES].reset_index(drop=True)
    y = df.iloc[train_idx][TARGET].reset_index(drop=True)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.full(len(X), np.nan)
    fold_results = []
    convergence_count = 0
    print("\n" + "=" * 72)
    print("VALIDATION | 5-FOLD CROSS-VALIDATION ON THE TRAINING 80%")
    print("=" * 72)
    print("Each fold trains on 4/5 of the training partition and predicts the remaining 1/5.")
    print("The final 20% test partition is not used here.")

    for fold, (fit_idx, val_idx) in enumerate(cv.split(X), start=1):
        fold_model = clone(bundle["model"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fold_model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        convergence_count += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        pred = fold_model.predict(X.iloc[val_idx])
        oof_pred[val_idx] = pred
        rmse = float(np.sqrt(mean_squared_error(y.iloc[val_idx], pred)))
        mae = float(mean_absolute_error(y.iloc[val_idx], pred))
        r2 = float(r2_score(y.iloc[val_idx], pred))
        fold_results.append({"fold": fold, "mae": mae, "rmse": rmse, "r2": r2})
        print(f"Fold {fold}: MAE={mae:.3f} dB | RMSE={rmse:.3f} dB | R²={r2:.4f}")

    overall_mae = mean_absolute_error(y, oof_pred)
    overall_rmse = np.sqrt(mean_squared_error(y, oof_pred))
    overall_r2 = r2_score(y, oof_pred)
    print("\nOUT-OF-FOLD SUMMARY (each prediction was made by a model that did not train on that row)")
    print(f"MAE : {overall_mae:.3f} dB")
    print(f"RMSE: {overall_rmse:.3f} dB")
    print(f"R²  : {overall_r2:.4f}")
    print(f"Mean fold RMSE: {np.mean([x['rmse'] for x in fold_results]):.3f} dB; std: {np.std([x['rmse'] for x in fold_results], ddof=1):.3f} dB")
    if convergence_count:
        print(f"Note: {convergence_count} fold fit(s) emitted an MLP convergence warning.")

    # Plot 1: cross-validated parity plot.
    fig, ax = plt.subplots(figsize=(7.3, 6.1))
    ax.scatter(y, oof_pred, alpha=0.70, edgecolors="none")
    lo = float(min(y.min(), oof_pred.min()))
    hi = float(max(y.max(), oof_pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Perfect predictions")
    ax.set_xlabel("Measured sound pressure level (dB)")
    ax.set_ylabel("Out-of-fold prediction (dB)")
    ax.set_title("Validation: out-of-fold predicted vs measured")
    ax.text(0.04, 0.96, f"MAE = {overall_mae:.2f} dB\nRMSE = {overall_rmse:.2f} dB\nR² = {overall_r2:.3f}", transform=ax.transAxes, va="top", bbox={"boxstyle": "round", "alpha": 0.85})
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    print("\nPLOT EXPLANATION — OUT-OF-FOLD PREDICTED VS MEASURED")
    print("Each dot is predicted by a fold model that did not train on that observation. Dots close to the dashed line are accurate. Systematic points above/below the line show under-/over-prediction; broad scatter means larger errors.")

    # Plot 2: residual pattern and distribution.
    residuals = oof_pred - y.to_numpy()
    fig2, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    axes[0].scatter(oof_pred, residuals, alpha=0.65, edgecolors="none")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Out-of-fold predicted SPL (dB)")
    axes[0].set_ylabel("Prediction error: predicted − measured (dB)")
    axes[0].set_title("Residuals vs prediction")
    axes[0].grid(alpha=0.2)
    axes[1].hist(residuals, bins=24, edgecolor="white")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Prediction error (dB)")
    axes[1].set_ylabel("Number of training-partition rows")
    axes[1].set_title("Residual distribution")
    fig2.suptitle("Validation error diagnostics")
    fig2.tight_layout()
    print("\nPLOT EXPLANATION — VALIDATION RESIDUALS")
    print("Left: errors should form a roughly patternless cloud around zero. A curve, funnel, or groups can indicate bias, changing error variance, or operating regions the model handles differently. Right: errors centred near zero with limited spread are desirable; long tails indicate occasional large misses.")

    # Plot 3: per-fold RMSE to show stability.
    fig3, ax3 = plt.subplots(figsize=(7.5, 4.8))
    fold_rmse = [row["rmse"] for row in fold_results]
    ax3.bar([f"Fold {i}" for i in range(1, 6)], fold_rmse)
    ax3.axhline(np.mean(fold_rmse), color="black", linestyle="--", label=f"Mean = {np.mean(fold_rmse):.2f} dB")
    ax3.set_ylabel("RMSE (dB; lower is better)")
    ax3.set_title("Validation stability across the five folds")
    ax3.grid(axis="y", alpha=0.25)
    ax3.legend()
    fig3.tight_layout()
    print("\nPLOT EXPLANATION — FIVE-FOLD RMSE")
    print("Each bar is error on a different held-out fold. Similar bar heights suggest the performance is reasonably stable across training/validation subsets. One much taller bar suggests the model is sensitive to which observations are held out.")

    print("\nNo plots or metric files were saved. Close the plot windows when you have reviewed them.")
    plt.show()
    print("\nNext run: python test.py for the final untouched 20% test set and physics investigations.")


if __name__ == "__main__":
    main()
