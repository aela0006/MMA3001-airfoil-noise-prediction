"""Final 20% test evaluation and interactive physics investigations.
No PNGs, CSVs, JSONs, or reports are written. Plots are shown by matplotlib.
"""
from pathlib import Path
import hashlib
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_FILE = ROOT / "models" / "airfoil_nn.joblib"
FEATURES = ["frequency_hz", "angle_of_attack_deg", "chord_length_m",
            "free_stream_velocity_m_s", "suction_side_displacement_thickness_m"]
TARGET = "sound_pressure_level_db"
COLUMNS = FEATURES + [TARGET]
DISPLAY = {
    "frequency_hz": "Frequency (Hz)",
    "angle_of_attack_deg": "Angle of attack (degrees)",
    "chord_length_m": "Chord length (m)",
    "free_stream_velocity_m_s": "Free-stream velocity (m/s)",
    "suction_side_displacement_thickness_m": "Suction-side displacement thickness (m)",
}


def load_dataset(bundle):
    """Load the raw dataset and confirm its checksum matches the saved bundle.

    Parameters
    ----------
    bundle : dict
        Trained model metadata containing the original dataset checksum.

    Returns
    -------
    pandas.DataFrame
        Validated dataset with five input features and the target column.

    Raises
    ------
    FileNotFoundError
        If the source dataset is missing.
    ValueError
        If the data differs from training data or contains invalid values.
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
        raise ValueError("Dataset differs from the one used for training (including row order). Rerun train.py with the intended .dat file.")
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#")
    if df.shape[1] != 6:
        raise ValueError(f"Expected six columns in .dat, found {df.shape[1]}.")
    df.columns = COLUMNS
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any() or not np.isfinite(df.to_numpy()).all():
        raise ValueError("Dataset contains missing or non-numeric values.")
    return df


def report_plot(number, title, what, interpretation):
    """Print a short plain-language guide to an interactive diagnostic plot."""
    print(f"\nPLOT {number} — {title.upper()}")
    print(f"What it is: {what}")
    print(f"What it explains / how to read it: {interpretation}")


def model_grid(model, baseline, variable, values, second_variable=None, second_values=None):
    """Predict across one input; optionally hold a second input at a chosen value."""
    grid = pd.DataFrame([baseline.copy() for _ in range(len(values))])
    grid[variable] = np.asarray(values)
    if second_variable is not None:
        grid[second_variable] = second_values
    return model.predict(grid[FEATURES])


def main():
    """Evaluate once on the untouched test partition and show diagnostics."""
    if not MODEL_FILE.is_file():
        raise FileNotFoundError("Run train.py first.")
    bundle = joblib.load(MODEL_FILE)
    df = load_dataset(bundle)
    train_idx = np.asarray(bundle["train_idx"], dtype=int)
    test_idx = np.asarray(bundle["test_idx"], dtype=int)
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    model = bundle["model"]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].to_numpy()
    predicted = model.predict(X_test)
    mae = mean_absolute_error(y_test, predicted)
    rmse = np.sqrt(mean_squared_error(y_test, predicted))
    r2 = r2_score(y_test, predicted)

    print("\n" + "=" * 72)
    print("FINAL TEST | UNTOUCHED 20% OF THE DATA")
    print("=" * 72)
    print(f"Test observations: {len(test_df)} / {len(df)} ({len(test_df)/len(df):.1%})")
    print(f"MAE : {mae:.3f} dB — average absolute prediction error")
    print(f"RMSE: {rmse:.3f} dB — penalises large errors more heavily")
    print(f"R²  : {r2:.4f} — 1 is perfect; 0 is mean-baseline performance; below 0 is worse than that baseline")
    print("Do not repeatedly tune the model against these test results; keep them as the final check.")

    # 1. Final test: observed vs predicted.
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.scatter(y_test, predicted, alpha=0.72, edgecolors="none")
    low = float(min(y_test.min(), predicted.min()))
    high = float(max(y_test.max(), predicted.max()))
    ax.plot([low, high], [low, high], "k--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Measured sound pressure level (dB)")
    ax.set_ylabel("Predicted sound pressure level (dB)")
    ax.set_title("Final test set: predicted vs measured")
    ax.text(0.04, 0.96, f"MAE = {mae:.2f} dB\nRMSE = {rmse:.2f} dB\nR² = {r2:.3f}", transform=ax.transAxes, va="top", bbox={"boxstyle": "round", "alpha": 0.85})
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    report_plot(1, "Test predicted vs measured", "Each dot is an unseen test observation; the dashed diagonal is perfect prediction.", "Dots close to the line indicate good predictions. Curved/systematic displacement suggests bias at some sound levels. This is the honest generalisation check because these rows were not used to fit the final model.")

    # 2. Test residual diagnostics.
    residuals = predicted - y_test
    fig2, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    axes[0].scatter(predicted, residuals, alpha=0.68, edgecolors="none")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted SPL (dB)")
    axes[0].set_ylabel("Error: predicted − measured (dB)")
    axes[0].set_title("Residuals vs prediction")
    axes[0].grid(alpha=0.2)
    axes[1].hist(residuals, bins=22, edgecolor="white")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Prediction error (dB)")
    axes[1].set_ylabel("Number of test observations")
    axes[1].set_title("Residual distribution")
    fig2.suptitle("Test-set error diagnostics")
    fig2.tight_layout()
    report_plot(2, "Residual diagnostics", "Left: prediction error against predicted SPL. Right: the distribution of prediction errors.", "A random cloud around zero and a centred histogram are desirable. A curve can indicate systematic bias; a funnel indicates error size changes with SPL; long tails indicate occasional large misses. Error is defined as prediction minus measurement, so positive means over-prediction.")

    # 3. Permutation importance on the held-out test set.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imp = permutation_importance(model, X_test, y_test, scoring="neg_mean_squared_error", n_repeats=10, random_state=42)
    importance_order = np.argsort(imp.importances_mean)
    fig3, ax3 = plt.subplots(figsize=(8.7, 5.4))
    means = imp.importances_mean[importance_order]
    stds = imp.importances_std[importance_order]
    names = [DISPLAY[FEATURES[i]] for i in importance_order]
    ax3.barh(names, means, xerr=stds, capsize=3)
    ax3.axvline(0, color="black", linewidth=1)
    ax3.set_xlabel("Increase in test MSE when shuffled (dB²)")
    ax3.set_title("Permutation importance on held-out test data")
    ax3.grid(axis="x", alpha=0.25)
    fig3.tight_layout()
    report_plot(3, "Permutation feature importance", "Each input column is shuffled in turn; the chart measures how much this damages prediction performance.", "A larger positive value means the trained model relies more on that input for predictions. Values near zero suggest little unique predictive contribution; negative values can arise from noise or correlated inputs. This is predictive usefulness, not physical causation or a universal ranking of physical mechanisms.")

    # Quantities used for controlled-response plots come only from training data.
    q05 = train_df[FEATURES].quantile(0.05)
    q95 = train_df[FEATURES].quantile(0.95)
    baseline = train_df[FEATURES].median().to_dict()

    # 4. Correlation heatmap within training observations.
    corr = train_df[FEATURES + [TARGET]].corr(numeric_only=True)
    labels = [DISPLAY.get(c, "Sound pressure level (dB)") for c in corr.columns]
    fig4, ax4 = plt.subplots(figsize=(10.5, 8.1))
    im = ax4.imshow(corr.to_numpy(), vmin=-1, vmax=1, aspect="auto")
    ax4.set_xticks(range(len(labels)))
    ax4.set_yticks(range(len(labels)))
    ax4.set_xticklabels(labels, rotation=45, ha="right")
    ax4.set_yticklabels(labels)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            value = corr.iloc[i, j]
            ax4.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    ax4.set_title("Linear correlations in the training data")
    fig4.colorbar(im, ax=ax4, label="Pearson correlation (−1 to +1)")
    fig4.tight_layout()
    report_plot(4, "Correlation heatmap", "Pearson correlation coefficients between each measured input and the sound pressure level target.", "Values near +1 indicate positive linear association; values near −1 indicate negative linear association; values near 0 mean weak linear association only. A near-zero value does not rule out a nonlinear relationship. Correlation alone cannot identify cause and effect, and correlated inputs can make importance harder to separate.")

    # Choose observed low/mid/high frequency levels to avoid arbitrary frequency labels.
    unique_freq = np.sort(train_df["frequency_hz"].unique())
    quantile_freqs = np.quantile(unique_freq, [0.15, 0.50, 0.85])
    chosen_freqs = []
    for q in quantile_freqs:
        f = float(unique_freq[np.argmin(np.abs(unique_freq - q))])
        if f not in chosen_freqs:
            chosen_freqs.append(f)
    if len(chosen_freqs) < 3:
        chosen_freqs = [float(v) for v in unique_freq[[0, len(unique_freq)//2, -1]]]

    # 5. Velocity response at several frequencies.
    velocity_values = np.linspace(q05["free_stream_velocity_m_s"], q95["free_stream_velocity_m_s"], 80)
    fig5, ax5 = plt.subplots(figsize=(8.6, 5.5))
    velocity_effects = {}
    for freq in chosen_freqs:
        preds = model_grid(model, baseline, "free_stream_velocity_m_s", velocity_values,
                           "frequency_hz", np.full(len(velocity_values), freq))
        ax5.plot(velocity_values, preds, linewidth=2.2, label=f"Frequency {freq:,.0f} Hz")
        velocity_effects[freq] = float(preds[-1] - preds[0])
    ax5.set_xlabel("Free-stream velocity (m/s)")
    ax5.set_ylabel("Predicted sound pressure level (dB)")
    ax5.set_title("Modelled velocity effect at selected frequencies")
    ax5.grid(alpha=0.25)
    ax5.legend()
    fig5.tight_layout()
    report_plot(5, "Airflow velocity and noise", "The network changes velocity from the 5th to 95th percentile of training values at low, middle, and high observed frequencies; angle, chord, and displacement thickness are held at training medians.", "An upward curve means the model predicts higher SPL as velocity increases under those controlled conditions; a downward curve means lower SPL. Different slopes or shapes across frequencies suggest a possible velocity–frequency interaction learned by the model. These are conditional model predictions, not proof that a real experiment changing only velocity would cause that exact change.")
    for f, delta in velocity_effects.items():
        print(f"  At {f:,.0f} Hz, modelled change from P05 to P95 velocity: {delta:+.3f} dB")

    # 6. Angle of attack response at selected frequencies.
    angle_values = np.linspace(q05["angle_of_attack_deg"], q95["angle_of_attack_deg"], 80)
    fig6, ax6 = plt.subplots(figsize=(8.6, 5.5))
    angle_effects = {}
    for freq in chosen_freqs:
        preds = model_grid(model, baseline, "angle_of_attack_deg", angle_values,
                           "frequency_hz", np.full(len(angle_values), freq))
        ax6.plot(angle_values, preds, linewidth=2.2, label=f"Frequency {freq:,.0f} Hz")
        angle_effects[freq] = float(preds[-1] - preds[0])
    ax6.set_xlabel("Angle of attack (degrees)")
    ax6.set_ylabel("Predicted sound pressure level (dB)")
    ax6.set_title("Modelled angle-of-attack effect at selected frequencies")
    ax6.grid(alpha=0.25)
    ax6.legend()
    fig6.tight_layout()
    report_plot(6, "Angle of attack at different frequencies", "Predicted SPL as angle of attack changes, shown separately at three observed frequency levels; other inputs are held at their medians.", "Compare line slopes and curvature. If curves change differently with angle, the fitted network has learned a frequency-dependent angle response. The sign of the slope describes the model's prediction within the observed angle range, not a general law for every airfoil or flow condition.")
    for f, delta in angle_effects.items():
        print(f"  At {f:,.0f} Hz, modelled change from P05 to P95 angle: {delta:+.3f} dB")

    # 7. Displacement thickness: raw association and controlled model response.
    thickness = "suction_side_displacement_thickness_m"
    thick_values = np.linspace(q05[thickness], q95[thickness], 80)
    thick_pred = model_grid(model, baseline, thickness, thick_values)
    fig7, axes7 = plt.subplots(1, 2, figsize=(12.4, 4.9))
    axes7[0].scatter(train_df[thickness], train_df[TARGET], alpha=0.30, s=17, edgecolors="none")
    axes7[0].set_xlabel(DISPLAY[thickness])
    axes7[0].set_ylabel("Measured sound pressure level (dB)")
    axes7[0].set_title("Raw data: other conditions also vary")
    axes7[0].grid(alpha=0.2)
    axes7[1].plot(thick_values, thick_pred, linewidth=2.5)
    axes7[1].set_xlabel(DISPLAY[thickness])
    axes7[1].set_ylabel("Predicted sound pressure level (dB)")
    axes7[1].set_title("Controlled model response (other inputs fixed)")
    axes7[1].grid(alpha=0.2)
    fig7.suptitle("Boundary-layer displacement thickness and self-noise")
    fig7.tight_layout()
    thick_delta = float(thick_pred[-1] - thick_pred[0])
    report_plot(7, "Displacement thickness and noise", "Left: measured thickness versus measured SPL across training rows. Right: the model's predicted response when thickness alone is varied from P05 to P95.", "The raw scatter is a mixed association because frequency, velocity, and angle differ between rows. The controlled curve isolates the network's learned sensitivity while holding other inputs at medians. A rise/fall/curvature describes the learned pattern over this range; it does not prove a causal mechanism. Displacement thickness is a boundary-layer descriptor, so a relationship may be connected to boundary-layer state and surface-pressure fluctuations, but this dataset alone cannot prove that pathway.")
    print(f"  Modelled change from P05 to P95 displacement thickness: {thick_delta:+.3f} dB")

    # 8. Velocity-frequency response surface.
    freq_grid = np.geomspace(max(float(q05["frequency_hz"]), 1e-9), float(q95["frequency_hz"]), 70)
    vel_grid = np.linspace(q05["free_stream_velocity_m_s"], q95["free_stream_velocity_m_s"], 65)
    F, V = np.meshgrid(freq_grid, vel_grid)
    grid = pd.DataFrame({
        "frequency_hz": F.ravel(),
        "angle_of_attack_deg": np.full(F.size, baseline["angle_of_attack_deg"]),
        "chord_length_m": np.full(F.size, baseline["chord_length_m"]),
        "free_stream_velocity_m_s": V.ravel(),
        "suction_side_displacement_thickness_m": np.full(F.size, baseline[thickness]),
    })
    Z = model.predict(grid[FEATURES]).reshape(F.shape)
    fig8, ax8 = plt.subplots(figsize=(9.2, 6.2))
    contour = ax8.contourf(F, V, Z, levels=20)
    ax8.set_xscale("log")
    ax8.set_xlabel("Frequency (Hz; logarithmic axis)")
    ax8.set_ylabel("Free-stream velocity (m/s)")
    ax8.set_title("Predicted SPL across frequency and velocity\n(other inputs fixed at training medians)")
    fig8.colorbar(contour, ax=ax8, label="Predicted SPL (dB)")
    fig8.tight_layout()
    report_plot(8, "Velocity–frequency interaction heatmap", "A contour map of network predictions over frequency and velocity together; angle, chord and displacement thickness are fixed at training medians.", "Colour shows predicted SPL. If the colour gradient across velocity changes substantially with frequency, that suggests the modelled velocity effect depends on frequency (an interaction). Smooth contour changes show how the fitted model interpolates. The model may create combinations that are sparse in the data, so read the map only over observed ranges and alongside its test error.")

    print("\n" + "=" * 72)
    print("NUMERICAL SUMMARY FOR THE PHYSICS CURVES")
    print("=" * 72)
    print("Predicted endpoint changes are from training P05 to P95; all other inputs are fixed at training medians.")
    print("Positive means predicted SPL rises across the plotted range; negative means it falls.")
    for f, delta in velocity_effects.items():
        print(f"Velocity, at {f:,.0f} Hz: {delta:+.3f} dB")
    for f, delta in angle_effects.items():
        print(f"Angle of attack, at {f:,.0f} Hz: {delta:+.3f} dB")
    print(f"Displacement thickness: {thick_delta:+.3f} dB")
    print("\nNo plot images, spreadsheets, CSVs, JSONs, or reports were saved. Close plot windows after reviewing them.")
    print("Physical caution: learned patterns and correlations are not causal proof or universal aerodynamic laws.")
    plt.show()


if __name__ == "__main__":
    main()
