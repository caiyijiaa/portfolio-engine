"""Portfolio Optimization & Risk Engine
Run with:  streamlit run app.py
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, expected_returns, risk_models

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Portfolio Optimization & Risk Engine")

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("Inputs")
    tickers_raw = st.text_input("Tickers (comma separated)", "AAPL, MSFT, SPY, GLD, TLT")
    start = st.date_input("Start date", date.today() - timedelta(days=5 * 365))
    end = st.date_input("End date", date.today())
    investment = st.number_input("Investment amount ($)", 1000, 10_000_000, 10_000, step=1000)
    rf = st.slider("Risk-free rate (%)", 0.0, 8.0, 4.0, 0.25) / 100
    max_weight = st.slider("Max weight per asset (%)", 10, 100, 50, 5) / 100
    years = st.slider("Simulation horizon (years)", 1, 10, 5)
    n_sims = st.select_slider("Number of simulations", [500, 1000, 2500, 5000], value=1000)

tickers = tuple(t.strip().upper() for t in tickers_raw.split(",") if t.strip())


# ---------------------------------------------------------------- Step 1: data
@st.cache_data(ttl=3600, show_spinner="Downloading market data...")
def load_prices(tickers, start, end):
    data = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):  # single ticker
        data = data.to_frame(name=tickers[0])
    return data.dropna(how="all").dropna()


if len(tickers) < 2:
    st.warning("Enter at least two tickers.")
    st.stop()
if max_weight * len(tickers) < 1:
    st.warning("Max weight is too low for this number of assets. Raise it or add tickers.")
    st.stop()

prices = load_prices(tickers, start, end)
if prices.empty or prices.shape[1] < 2:
    st.error("Could not get enough price data. Check your tickers and dates.")
    st.stop()

# ---------------------------------------------------------------- Step 2: returns & risk
returns = prices.pct_change().dropna()
mu = expected_returns.mean_historical_return(prices)  # annualised
S = risk_models.sample_cov(prices)                    # annualised covariance


# ---------------------------------------------------------------- Step 3: optimise
def optimise(kind):
    ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
    if kind == "sharpe":
        ef.max_sharpe(risk_free_rate=rf)
    else:
        ef.min_volatility()
    w = ef.clean_weights()
    perf = ef.portfolio_performance(risk_free_rate=rf)
    return w, perf


try:
    w_sharpe, perf_sharpe = optimise("sharpe")
except Exception as e:
    st.error(f"Max-Sharpe optimisation failed: {e}. Try a lower risk-free rate or different tickers.")
    st.stop()
w_minvol, perf_minvol = optimise("minvol")


def frontier_points(n=40):
    lo, hi = perf_minvol[0], float(mu.max())
    pts = []
    for target in np.linspace(lo, hi * 0.999, n):
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
            ef.efficient_return(target)
            r, v, _ = ef.portfolio_performance(risk_free_rate=rf)
            pts.append((v, r))
        except Exception:
            continue
    return pts


def random_portfolios(n=3000):
    rng = np.random.default_rng(1)
    out = []
    for _ in range(n):
        w = rng.dirichlet(np.ones(len(mu)))
        r = float(w @ mu.values)
        v = float(np.sqrt(w @ S.values @ w))
        out.append((v, r, (r - rf) / v))
    return pd.DataFrame(out, columns=["vol", "ret", "sharpe"])


# ---------------------------------------------------------------- Step 4: Monte Carlo
def monte_carlo(weights, years, n_sims, investment):
    w = np.array([weights[c] for c in returns.columns])
    port = returns.values @ w
    days = int(252 * years)
    rng = np.random.default_rng(42)
    draws = rng.normal(port.mean(), port.std(), size=(days, n_sims))
    paths = investment * np.cumprod(1 + draws, axis=0)
    return np.vstack([np.full(n_sims, investment), paths])


paths = monte_carlo(w_sharpe, years, n_sims, investment)
final = paths[-1]

# ---------------------------------------------------------------- Step 5: UI
tab1, tab2, tab3 = st.tabs(["Optimal portfolio", "Efficient frontier", "Monte Carlo"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected return", f"{perf_sharpe[0]:.1%}")
    c2.metric("Volatility", f"{perf_sharpe[1]:.1%}")
    c3.metric("Sharpe ratio", f"{perf_sharpe[2]:.2f}")

    wdf = pd.DataFrame({
        "Max Sharpe": pd.Series(w_sharpe),
        "Min Volatility": pd.Series(w_minvol),
    })
    wdf = wdf[(wdf > 0).any(axis=1)]
    left, right = st.columns(2)
    with left:
        nz = wdf["Max Sharpe"][wdf["Max Sharpe"] > 0]
        fig = go.Figure(go.Pie(labels=nz.index, values=nz.values, hole=0.45))
        fig.update_layout(title="Max-Sharpe allocation", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        show = wdf.copy()
        show["Max Sharpe ($)"] = show["Max Sharpe"] * investment
        st.dataframe(
            show.style.format({"Max Sharpe": "{:.1%}", "Min Volatility": "{:.1%}", "Max Sharpe ($)": "${:,.0f}"}),
            use_container_width=True,
        )
        st.caption("Inputs use historical averages, which are noisy estimates of the future.")

with tab2:
    rp = random_portfolios()
    fp = frontier_points()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rp["vol"], y=rp["ret"], mode="markers", name="Random portfolios",
        marker=dict(color=rp["sharpe"], colorscale="Viridis", size=4, opacity=0.5,
                    colorbar=dict(title="Sharpe")),
    ))
    if fp:
        fig.add_trace(go.Scatter(x=[p[0] for p in fp], y=[p[1] for p in fp],
                                 mode="lines", name="Efficient frontier", line=dict(color="red", width=3)))
    fig.add_trace(go.Scatter(x=[perf_sharpe[1]], y=[perf_sharpe[0]], mode="markers", name="Max Sharpe",
                             marker=dict(symbol="star", size=18, color="gold", line=dict(width=1, color="black"))))
    fig.add_trace(go.Scatter(x=[perf_minvol[1]], y=[perf_minvol[0]], mode="markers", name="Min volatility",
                             marker=dict(symbol="diamond", size=14, color="cyan", line=dict(width=1, color="black"))))
    fig.add_trace(go.Scatter(x=np.sqrt(np.diag(S)), y=mu.values, mode="markers+text", text=list(mu.index),
                             textposition="top center", name="Individual assets",
                             marker=dict(size=9, color="white", line=dict(width=1, color="black"))))
    fig.update_layout(xaxis_title="Volatility (annual)", yaxis_title="Expected return (annual)",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    p5, p50, p95 = np.percentile(paths, [5, 50, 95], axis=1)
    x = np.arange(paths.shape[0]) / 252

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Worst case (5th pct)", f"${np.percentile(final, 5):,.0f}")
    k2.metric("Median", f"${np.percentile(final, 50):,.0f}")
    k3.metric("Best case (95th pct)", f"${np.percentile(final, 95):,.0f}")
    k4.metric("Probability of loss", f"{(final < investment).mean():.1%}")

    fig = go.Figure()
    for i in range(min(40, n_sims)):
        fig.add_trace(go.Scatter(x=x, y=paths[:, i], mode="lines", showlegend=False,
                                 line=dict(width=0.5, color="rgba(120,120,120,0.25)"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=p95, name="95th percentile", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=x, y=p50, name="Median", line=dict(color="blue", width=3)))
    fig.add_trace(go.Scatter(x=x, y=p5, name="5th percentile", line=dict(color="red")))
    fig.update_layout(title=f"{n_sims:,} simulated paths over {years} years",
                      xaxis_title="Years", yaxis_title="Portfolio value ($)", height=500)
    st.plotly_chart(fig, use_container_width=True)

    h = go.Figure(go.Histogram(x=final, nbinsx=60))
    h.add_vline(x=investment, line_dash="dash", annotation_text="Starting value")
    h.update_layout(title="Distribution of final portfolio value", xaxis_title="Final value ($)", height=350)
    st.plotly_chart(h, use_container_width=True)
    st.caption("Simulation assumes normally distributed daily returns, so it likely understates extreme crashes.")
