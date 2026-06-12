"""Gaussian mixture model fitting for pair angle differences (pure NumPy)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _gaussian_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    std = max(std, 1e-6)
    z = (x - mean) / std
    return np.exp(-0.5 * z * z) / (std * np.sqrt(2.0 * np.pi))


def fit_angle_gmm(angles_deg: list[float], n_eval: int = 120) -> dict[str, Any] | None:
    """Fit a 1D GMM via EM and return PDF samples for plotting."""
    if not angles_deg:
        return None

    X = np.array(angles_deg, dtype=float)
    n_samples = len(X)
    n_components = min(3, n_samples)

    if n_components == 1:
        mean = float(np.mean(X))
        std = float(np.std(X)) if n_samples > 1 else 0.1
        weights = np.array([1.0])
        means = np.array([mean])
        stds = np.array([max(std, 0.05)])
    else:
        weights, means, stds = _em_gmm_1d(X, n_components)

    margin = max(0.5, 3.0 * float(np.std(X)))
    x_min = float(X.min()) - margin
    x_max = float(X.max()) + margin
    xs = np.linspace(x_min, x_max, n_eval)

    pdf = np.zeros_like(xs)
    for w, m, s in zip(weights, means, stds):
        pdf += w * _gaussian_pdf(xs, m, s)

    peak_idx = int(np.argmax(pdf))
    peak_x = float(xs[peak_idx])
    peak_y = float(pdf[peak_idx])

    components = [
        {
            "weight": round(float(w), 4),
            "mean": round(float(m), 4),
            "std": round(float(s), 4),
        }
        for w, m, s in zip(weights, means, stds)
    ]

    return {
        "x": [round(float(v), 4) for v in xs],
        "y": [round(float(v), 6) for v in pdf],
        "peak": {"x": round(peak_x, 4), "y": round(peak_y, 6)},
        "components": components,
        "samples": [round(float(v), 4) for v in angles_deg],
    }


def _em_gmm_1d(
    X: np.ndarray, n_components: int, max_iter: int = 100, tol: float = 1e-4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple EM for univariate GMM."""
    n = len(X)
    rng = np.random.default_rng(0)
    indices = rng.choice(n, size=n_components, replace=False)
    means = X[indices].astype(float)
    stds = np.full(n_components, max(float(np.std(X)), 0.1))
    weights = np.full(n_components, 1.0 / n_components)

    for _ in range(max_iter):
        resp = np.zeros((n, n_components))
        for k in range(n_components):
            resp[:, k] = weights[k] * _gaussian_pdf(X, means[k], stds[k])
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp_sum[resp_sum < 1e-12] = 1e-12
        resp /= resp_sum

        new_weights = resp.mean(axis=0)
        new_means = (resp * X[:, None]).sum(axis=0) / (resp.sum(axis=0) + 1e-12)
        new_stds = np.sqrt(
            (resp * (X[:, None] - new_means) ** 2).sum(axis=0)
            / (resp.sum(axis=0) + 1e-12)
        )
        new_stds = np.maximum(new_stds, 0.05)

        if (
            np.max(np.abs(new_means - means)) < tol
            and np.max(np.abs(new_stds - stds)) < tol
        ):
            weights, means, stds = new_weights, new_means, new_stds
            break
        weights, means, stds = new_weights, new_means, new_stds

    return weights, means, stds
