"""Build static HTML API documentation using Python's standard-library pydoc."""

from __future__ import annotations

import os
from pathlib import Path
import pydoc
import sys

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MODULES = ("train", "validation", "test", "comparison")


def main() -> None:
    """Generate one HTML API page per project module and refresh the index."""
    sys.path.insert(0, str(ROOT))
    DOCS.mkdir(exist_ok=True)
    previous_directory = Path.cwd()
    try:
        os.chdir(DOCS)
        for module in MODULES:
            pydoc.writedoc(module)
    finally:
        os.chdir(previous_directory)

    links = "\n".join(
        f'<li><a href="{module}.html">{module}.py</a></li>'
        for module in MODULES
    )
    index = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Airfoil Self-Noise Prediction — API Documentation</title>
  <style>
    body {{ max-width: 820px; margin: 3rem auto; padding: 0 1.25rem; font: 16px/1.6 Arial, sans-serif; color: #1f2937; }}
    h1 {{ line-height: 1.2; }}
    li {{ margin: .5rem 0; }}
    code {{ background: #f3f4f6; padding: .1rem .3rem; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>Airfoil Self-Noise Prediction</h1>
  <p>Generated API documentation for the project's Python source files. Function descriptions are taken from source docstrings.</p>
  <h2>Modules</h2>
  <ul>{links}</ul>
  <h2>Workflow</h2>
  <ol>
    <li><code>train.py</code> fits and saves the MLP and data split.</li>
    <li><code>validation.py</code> performs five-fold validation on training rows.</li>
    <li><code>test.py</code> evaluates the untouched test partition.</li>
    <li><code>comparison.py</code> compares the MLP with linear and mean baselines.</li>
  </ol>
  <p>Return to <a href="../README.md">README.md</a> for setup, commands, data citation, and limitations.</p>
</body>
</html>
"""
    (DOCS / "index.html").write_text(index, encoding="utf-8")
    print(f"HTML documentation generated in: {DOCS}")


if __name__ == "__main__":
    main()
