import numpy as np


def simulate_gbm(S0: float, mu: float, sigma: float, T_days: int, n_paths: int = 1000) -> np.ndarray:
    """
    Geometric Brownian Motion: dS = mu*S*dt + sigma*S*dW
    Returns array of shape (T_days+1, n_paths)
    """
    dt = 1 / 252
    T = T_days
    paths = np.zeros((T + 1, n_paths))
    paths[0] = S0
    np.random.seed(42)
    Z = np.random.standard_normal((T, n_paths))
    for t in range(1, T + 1):
        paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t - 1])
    return paths


def get_confidence_intervals(paths: np.ndarray) -> dict:
    return {
        "p5": np.percentile(paths, 5, axis=1),
        "p50": np.percentile(paths, 50, axis=1),
        "p95": np.percentile(paths, 95, axis=1),
    }
