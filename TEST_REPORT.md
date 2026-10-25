# Automated Test Report

**Review date:** 25 October 2026  
**Project:** Airfoil Self-Noise Prediction — Neural Network

## Commands and results

- `python -m py_compile train.py validation.py test.py comparison.py tests/test_project.py docs/build_docs.py` — **passed**.
- `python -m pytest -q` — **8 passed, 1 skipped**.
- `pytest -q` — **8 passed, 1 skipped**.

The automated tests check dataset row count/schema/numeric integrity, hand-calculated regression metrics, MLP configuration, valid and invalid train/test partitions, and consistent baseline-evaluation output shapes/metrics.

## Environment used for this report

- Python: 3.13.5
- NumPy: 2.3.5
- pandas: 2.2.3
- scikit-learn: 1.8.0
- Matplotlib: 3.10.8
- joblib: 1.5.3
- pytest: 9.0.2

## Skipped integration check

The saved-model integration test was skipped because this review environment has scikit-learn **1.8.0**, while `requirements.txt` pins **scikit-learn==1.9.1**, the version used to create `models/airfoil_nn.joblib`. The test automatically runs when the pinned version is installed. For the definitive submission check, install `requirements-dev.txt` in a clean environment and rerun `python -m pytest -q`.

This report records the checks actually completed; the skipped integration test is not claimed as passed.
