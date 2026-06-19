#  Stock Price Path Simulator & Greeks Dashboard

A beginner-friendly Quantitative Finance dashboard built with Streamlit.

## Features
- **Historical Analysis** — Price chart + log return distribution + statistics
- **Monte Carlo Simulation** — 1,000 GBM paths with confidence intervals
- **Option Pricing** — Black-Scholes call/put with price surfaces
- **Greeks Dashboard** — Delta, Gamma, Vega, Theta, Rho with formulas and charts
- **Risk Metrics** — VaR 95%, Expected Shortfall, CDF, probability analysis

## Quick Start

```bash
# 1. Install dependencies
pip install streamlit yfinance scipy numpy pandas plotly

# 2. Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Project Structure
```
quant_app/
├── app.py           # Streamlit dashboard (main entry point)
├── data.py          # Yahoo Finance data fetching + parameter estimation
├── simulation.py    # Geometric Brownian Motion Monte Carlo
├── black_scholes.py # BS call/put pricing
├── greeks.py        # Delta, Gamma, Vega, Theta, Rho
└── risk.py          # VaR, Expected Shortfall, probabilities
```

## Usage
1. Enter any stock ticker (e.g. AAPL, TSLA, MSFT, SPY)
2. Set option parameters in the sidebar (strike, maturity, rate)
3. Click **Run Analysis**
4. Explore the 5 tabs

## Key Concepts
| Concept | Description |
|---|---|
| GBM | `dS = μS dt + σS dW` — models random stock price evolution |
| Monte Carlo | 1,000 random future price paths simulated |
| Black-Scholes | Closed-form formula for European option pricing |
| Delta | Option price sensitivity to stock price |
| VaR 95% | Maximum loss with 95% confidence |
