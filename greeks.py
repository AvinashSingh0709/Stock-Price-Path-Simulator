import numpy as np
from scipy.stats import norm
from black_scholes import d1_d2


def delta(S, K, T, r, sigma, option="call"):
    if T <= 0:
        return 1.0 if (option == "call" and S > K) else 0.0
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) if option == "call" else norm.cdf(d1) - 1


def gamma(S, K, T, r, sigma):
    if T <= 0:
        return 0.0
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    if T <= 0:
        return 0.0
    d1, _ = d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% change in vol


def theta(S, K, T, r, sigma, option="call"):
    if T <= 0:
        return 0.0
    d1, d2 = d1_d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option == "call":
        return (term1 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        return (term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365


def rho(S, K, T, r, sigma, option="call"):
    if T <= 0:
        return 0.0
    _, d2 = d1_d2(S, K, T, r, sigma)
    if option == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100


def all_greeks(S, K, T, r, sigma, option="call"):
    return {
        "Delta": delta(S, K, T, r, sigma, option),
        "Gamma": gamma(S, K, T, r, sigma),
        "Vega": vega(S, K, T, r, sigma),
        "Theta": theta(S, K, T, r, sigma, option),
        "Rho": rho(S, K, T, r, sigma, option),
    }
