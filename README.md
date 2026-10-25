# MMA3001-airfoil-noise-prediction
# Airfoil Self-Noise Prediction — Neural Network

This project predicts the measured airfoil sound-pressure level from five wind-tunnel input variables using a multilayer perceptron (MLP). It uses the original, headerless UCI `.dat` dataset and reports prediction errors in decibels (dB).

## Project structure

```text
airfoil_noise/
├── data/airfoil_self_noise.dat  # raw 1,503-row dataset
├── models/airfoil_nn.joblib     # trained model and fixed split metadata
├── tests/test_project.py        # automated unit/integration tests
├── docs/                        # generated HTML source-code documentation
├── train.py                     # train and save the MLP
├── validation.py                # 5-fold CV within the training partition
├── test.py                      # final held-out test and diagnostics
├── comparison.py                # compare MLP with two baselines
├── requirements.txt             # runtime dependencies
├── requirements-dev.txt         # testing dependencies
├── TEST_REPORT.md               # automated test summary
├── LICENSE                      # source-code licence and data-scope note
└── README.md
```

## Setup and run

Use Python 3.11 or newer. From the project folder, install the pinned runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the scripts in this order. Plots from the first three scripts open interactively; close each plot window to continue.

```bash
python train.py
python validation.py
python test.py
python comparison.py
```

`train.py` saves `models/airfoil_nn.joblib`. The later scripts reuse this model and the exact split stored in that file. Run `train.py` again whenever the dataset or training configuration changes.

## Data split and validation

- **Training:** 80% of the 1,503 observations (1,202 rows).
- **Final test:** the remaining untouched 20% (301 rows), used for final performance reporting and baseline comparison.
- **Cross-validation:** `validation.py` performs five-fold cross-validation using only the 80% training partition. The final test rows are not used in cross-validation.
- The shuffled split uses `random_state=42`. The model bundle stores row indices and a SHA-256 checksum so the scripts can detect a missing, replaced, or reordered dataset.

## What each script does

- **`train.py`** — fits an MLP with hidden layers of 64 and 32 neurons, with input and target scaling. Saves the model bundle and displays training-loss/early-stopping diagnostics.
- **`validation.py`** — performs five-fold cross-validation inside the training partition and displays out-of-fold predictions, residual diagnostics, and fold RMSE values.
- **`test.py`** — evaluates the final untouched 20% test partition and displays prediction diagnostics plus exploratory feature-response plots.
- **`comparison.py`** — compares the MLP with linear regression and a training-target-mean predictor on the same saved test rows. It saves metric/prediction CSVs and comparison plots under `outputs/comparison/`.

## Tests and HTML documentation

Install development dependencies and run the automated tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The test report is in `TEST_REPORT.md`. Rebuild the HTML API documentation from source docstrings with the Python standard library:

```bash
python docs/build_docs.py
```

Open `docs/index.html` to navigate the documentation pages.

## Metrics

- **MAE (dB):** mean absolute prediction error; lower is better.
- **RMSE (dB):** root mean squared error; penalises larger errors more strongly; lower is better.
- **R²:** coefficient of determination; 1 is ideal, 0 is similar to predicting the test-target mean, and a negative value means the model performed worse than that reference.

## Interpretation limits

The feature-response plots vary one input from the 5th to 95th percentile of training values while holding other inputs at their training medians. These plots describe the fitted model's response, not proof that changing one physical variable alone causes the observed effect. Some combinations may be sparsely represented because input variables can be correlated. Stay within the observed data range. The target is the recorded airfoil sound-pressure level; it is not total aircraft noise or an aircraft-noise certification result.

## Dataset attribution and licensing

The bundled dataset is from the UCI Machine Learning Repository and is licensed separately under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Cite it as:

Brooks, T., Pope, D., & Marcolini, M. (1989). *Airfoil Self-Noise* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VW2C.

The `LICENSE` file applies to the original project source code only; it does not replace the dataset's separate licence. Dataset source: https://archive.ics.uci.edu/dataset/291/airfoil%2Bself-noise.
