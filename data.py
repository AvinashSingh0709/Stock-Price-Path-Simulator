import yfinance as yf
import numpy as np
import pandas as pd


def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")
    df = df[["Close"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def estimate_parameters(df: pd.DataFrame) -> dict:
    log_returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    daily_vol = log_returns.std()
    daily_drift = log_returns.mean()
    annual_vol = daily_vol * np.sqrt(252)
    annual_drift = daily_drift * 252 + 0.5 * annual_vol**2  # mu = mean + 0.5*sigma^2
    current_price = float(df["Close"].iloc[-1])
    return {
        "current_price": current_price,
        "annual_vol": annual_vol,
        "annual_drift": annual_drift,
        "daily_vol": daily_vol,
        "daily_drift": daily_drift,
        "log_returns": log_returns,
    }
