import numpy as np
from scipy.stats import norm


def compute_risk_metrics(paths: np.ndarray, S0: float, K: float) -> dict:
    final_prices = paths[-1]
    returns = (final_prices - S0) / S0

    var_95 = np.percentile(returns, 5)
    es_95 = returns[returns <= var_95].mean()

    prob_above = (final_prices > K).mean()
    prob_below = (final_prices <= K).mean()

    return {
        "VaR_95": var_95,
        "ES_95": es_95,
        "prob_above_strike": prob_above,
        "prob_below_strike": prob_below,
        "mean_final": final_prices.mean(),
        "std_final": final_prices.std(),
        "min_final": final_prices.min(),
        "max_final": final_prices.max(),
    }
