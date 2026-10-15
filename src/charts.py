# ============================================================
# src/charts.py
# [RECONSTRUCTED] — original matplotlib cells not provided.
# Chart set built from fields already produced by training.py /
# daily_pipeline.py (sim_df, last_backtest_report, latest_prediction).
# Runs as a SEPARATE step after the core pipeline, per design
# decision: chart generation is not part of the daily prediction
# critical path (matplotlib failures must never block state save).
# ============================================================
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — required for GitHub Actions runners
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.config import RESULTS_DIR

CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

REGIME_COLORS = {"bull": "#2ca02c", "bear": "#d62728", "neutral": "#7f7f7f"}


def _load_sim_df() -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, "sim_history.csv")
    if not os.path.exists(path):
        print(f"  ⚠️  {path} 不存在，無法生成圖表")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def _load_prediction() -> dict:
    path = os.path.join(RESULTS_DIR, "latest_prediction.pkl")
    if not os.path.exists(path):
        return {}
    return joblib.load(path)


# ============================================================
# 1. Equity Curve — strategy vs. market (buy & hold)
# ============================================================
def chart_equity_curve(sim_df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(sim_df["date"], sim_df["cum_strat"],  label="Strategy",  color="#1f77b4", linewidth=1.6)
    ax.plot(sim_df["date"], sim_df["cum_market"], label="HSI (Buy & Hold)", color="#999999", linewidth=1.2, linestyle="--")
    ax.set_title("Equity Curve — Strategy vs. Market")
    ax.set_ylabel("Cumulative Return (×)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# 2. Drawdown Chart
# ============================================================
def chart_drawdown(sim_df: pd.DataFrame, out_path: str) -> None:
    cum = sim_df["cum_strat"]
    roll_max = cum.cummax()
    dd = (cum - roll_max) / (roll_max + 1e-9)

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.fill_between(sim_df["date"], dd * 100, 0, color="#d62728", alpha=0.4)
    ax.plot(sim_df["date"], dd * 100, color="#d62728", linewidth=0.8)
    ax.set_title(f"Drawdown (Max: {dd.min():.1%})")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# 3. Regime Timeline — HSI price colored by HMM regime
# ============================================================
def chart_regime_timeline(sim_df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    for regime, color in REGIME_COLORS.items():
        mask = sim_df["regime"] == regime
        ax.scatter(sim_df.loc[mask, "date"], sim_df.loc[mask, "close"],
                   s=6, color=color, label=regime)
    ax.plot(sim_df["date"], sim_df["close"], color="black", linewidth=0.4, alpha=0.4, zorder=0)
    ax.set_title("HSI Close Price by HMM Regime")
    ax.set_ylabel("Close")
    ax.legend(loc="upper left", markerscale=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# 4. Rolling Prediction Confidence (ensemble_prob) + signal markers
# ============================================================
def chart_prediction_confidence(sim_df: pd.DataFrame, out_path: str, window: int = 120) -> None:
    recent = sim_df.tail(window).copy()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(recent["date"], recent["ensemble_prob"], color="#1f77b4", linewidth=1.2, label="Ensemble Prob")
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")

    long_pts  = recent[recent["signal"] == 1]
    short_pts = recent[recent["signal"] == -1]
    ax.scatter(long_pts["date"],  long_pts["ensemble_prob"],  color="#2ca02c", marker="^", s=40, label="Long", zorder=5)
    ax.scatter(short_pts["date"], short_pts["ensemble_prob"], color="#d62728", marker="v", s=40, label="Short", zorder=5)

    ax.set_title(f"Prediction Confidence — Last {window} Trading Days")
    ax.set_ylabel("Ensemble Probability")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# 5. Monthly Returns Bar Chart
# ============================================================
def chart_monthly_returns(sim_df: pd.DataFrame, out_path: str) -> None:
    sim_df = sim_df.copy()
    sim_df["month"] = pd.to_datetime(sim_df["date"]).dt.strftime("%Y-%m")
    monthly = sim_df.groupby("month")["strat_ret"].sum()

    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in monthly.values]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(monthly.index, monthly.values * 100, color=colors)
    ax.set_title("Monthly Strategy Return")
    ax.set_ylabel("Return (%)")
    ax.axhline(0, color="black", linewidth=0.6)
    plt.xticks(rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# 6. Hit Rate by Regime (bar chart)
# ============================================================
def chart_hit_rate_by_regime(report: dict, out_path: str) -> None:
    breakdown = report.get("regime_breakdown", {})
    if not breakdown:
        print("  ⚠️  regime_breakdown 為空，跳過 hit-rate 圖")
        return

    regimes = list(breakdown.keys())
    hit_rates = [breakdown[r]["hit_rate"] * 100 for r in regimes]
    colors = [REGIME_COLORS.get(r, "#7f7f7f") for r in regimes]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(regimes, hit_rates, color=colors)
    ax.axhline(50, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title("Hit Rate by Regime")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_ylim(0, 100)
    for b, v in zip(bars, hit_rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# 7. Ensemble Weight Trend (global vs regime model contribution)
# ============================================================
def chart_ensemble_weights(sim_df: pd.DataFrame, out_path: str) -> None:
    if "ens_weight_summary" not in sim_df.columns:
        print("  ⚠️  ens_weight_summary 欄位不存在，跳過")
        return

    def _extract(w, key):
        if isinstance(w, dict):
            return w.get(key, np.nan)
        return np.nan

    g_w = sim_df["ens_weight_summary"].apply(lambda w: _extract(w, "global"))
    r_w = sim_df["ens_weight_summary"].apply(lambda w: _extract(w, "regime"))

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.stackplot(sim_df["date"], g_w, r_w,
                 labels=["Global Model", "Regime Model"],
                 colors=["#1f77b4", "#ff7f0e"], alpha=0.75)
    ax.set_title("Ensemble Weight Allocation Over Time")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# Entry point — generates all charts, called after daily_pipeline.py
# ============================================================
def generate_all_charts() -> None:
    os.makedirs(CHARTS_DIR, exist_ok=True)

    sim_df = _load_sim_df()
    if sim_df.empty:
        print("  ⚠️  無模擬數據，跳過圖表生成")
        return

    prediction = _load_prediction()
    report = {}
    report_path = os.path.join(RESULTS_DIR, "pipeline_state_report.pkl")
    # Prefer report embedded alongside state if available
    import joblib as _jl
    try:
        state = _jl.load(os.path.join("state", "pipeline_state.pkl"))
        report = state.get("last_backtest_report", {})
    except Exception:
        pass

    jobs = [
        ("equity_curve.png",          lambda p: chart_equity_curve(sim_df, p)),
        ("drawdown.png",              lambda p: chart_drawdown(sim_df, p)),
        ("regime_timeline.png",       lambda p: chart_regime_timeline(sim_df, p)),
        ("prediction_confidence.png", lambda p: chart_prediction_confidence(sim_df, p)),
        ("monthly_returns.png",       lambda p: chart_monthly_returns(sim_df, p)),
        ("hit_rate_by_regime.png",    lambda p: chart_hit_rate_by_regime(report, p)),
        ("ensemble_weights.png",      lambda p: chart_ensemble_weights(sim_df, p)),
    ]

    for filename, fn in jobs:
        out_path = os.path.join(CHARTS_DIR, filename)
        try:
            fn(out_path)
            print(f"  ✅ {filename} 已生成")
        except Exception as e:
            print(f"  ⚠️  {filename} 生成失敗: {e}")

    print(f"\n✅ charts.py: 所有圖表已輸出至 {CHARTS_DIR}/")


if __name__ == "__main__":
    generate_all_charts()
