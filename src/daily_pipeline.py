# ============================================================
# src/daily_pipeline.py
# [RECONSTRUCTED ORCHESTRATOR] — ties together all modules for
# the GitHub Actions daily run. The original Colab "daily update"
# cells (live prediction for the next trading day) were not pasted
# into this conversation, so this is built from the established
# state-persistence design, not ported verbatim. Please diff
# against your actual Cell 9+ logic if you still have it.
# ============================================================
import os
import sys
import traceback
from datetime import datetime

import joblib
import pandas as pd
import pytz

from src.config import STATE_PATH, MODELS_DIR, RESULTS_DIR, STATE_KEYS
from src.data_loader import load_all_data
from src.features import build_features, get_feat_cols
from src.models import safe_predict, get_market_trend
from src.hl_model import build_hl_targets, build_hl_features, get_hl_feat_cols
from src.training import (
    run_initial_training, run_daily_simulation,
    save_models, load_models, compute_backtest_report,
)

HKT = pytz.timezone("Asia/Hong_Kong")


def load_pipeline_state() -> dict:
    """
    Load persisted state dict (models + adaptive components) from
    state/pipeline_state.pkl. Returns {} if not found, signalling
    the caller to run full initial training instead.
    """
    if not os.path.exists(STATE_PATH):
        print(f"  ⚠️  {STATE_PATH} 不存在 → 需執行完整初始訓練")
        return {}
    try:
        state = joblib.load(STATE_PATH)
        missing = [k for k in STATE_KEYS if k not in state]
        if missing:
            print(f"  ⚠️  state 缺少欄位 {missing} → 需執行完整初始訓練")
            return {}
        print(f"  ✅ state 已從 {STATE_PATH} 載入 "
              f"({len(STATE_KEYS)} 個元件)")
        return state
    except Exception as e:
        print(f"  ⚠️  state 載入失敗: {e} → 需執行完整初始訓練")
        return {}


def save_pipeline_state(state: dict) -> None:
    """Persist state dict (only STATE_KEYS fields) to disk."""
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    slim_state = {k: state[k] for k in STATE_KEYS if k in state}
    joblib.dump(slim_state, STATE_PATH)
    print(f"  ✅ state 已儲存至 {STATE_PATH}")


def run_full_bootstrap() -> dict:
    """
    First-ever run (no state file exists yet): download full
    history, run initial training + simulation, persist everything.
    """
    print("=" * 60)
    print("  🔄 FULL BOOTSTRAP (無既有 state，從零訓練)")
    print("=" * 60)

    df_all = load_all_data()
    state  = run_initial_training(df_all)
    sim_df = run_daily_simulation(state)

    report = compute_backtest_report(sim_df)
    state["last_backtest_report"] = report
    state["sim_df"] = sim_df

    save_models(state, out_dir=MODELS_DIR)
    save_pipeline_state(state)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    sim_df.to_csv(os.path.join(RESULTS_DIR, "sim_history.csv"), index=False)

    return state


def run_incremental_update(state: dict) -> dict:
    """
    Normal daily run: state already exists. Pull latest data,
    append to history, let run_daily_simulation() process any new
    rows (triggers retrain only if drift/perf-degradation fires),
    then persist.
    """
    print("=" * 60)
    print("  🔁 INCREMENTAL DAILY UPDATE")
    print("=" * 60)

    df_all = load_all_data()
    state["df_all"] = df_all

    try:
        df_all = state["hmm"].add_features(df_all)
        state["df_all"] = df_all
    except Exception as e:
        print(f"  ⚠️  HMM add_features 失敗: {e}")

    sim_df = run_daily_simulation(state)

    report = compute_backtest_report(sim_df)
    state["last_backtest_report"] = report
    state["sim_df"] = sim_df

    save_models(state, out_dir=MODELS_DIR)
    save_pipeline_state(state)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    sim_df.to_csv(os.path.join(RESULTS_DIR, "sim_history.csv"), index=False)

    return state


def predict_next_day(state: dict) -> dict:
    """
    Generate the live, forward-looking prediction for the next
    trading day using the latest row of known data and the current
    (possibly just-retrained) global_model + regime_models.
    """
    from src.config import threshold_calibrator, position_sizer, ensemble_weighter

    df_all    = state["df_all"]
    feat_cols = state["feat_cols"]
    hmm       = state["hmm"]

    last_known = df_all.iloc[[-1]]
    last_date  = pd.Timestamp(last_known["date"].values[0])

    g_prob = safe_predict(state["global_model"], last_known, feat_cols)

    regime = (str(last_known["hmm_regime"].values[0])
              if "hmm_regime" in last_known.columns else "neutral")

    r_prob = g_prob
    if regime in state["regime_models"]:
        r_prob = safe_predict(state["regime_models"][regime], last_known, feat_cols)

    ens_prob = ensemble_weighter.ensemble(g_prob, r_prob)
    signal   = threshold_calibrator.get_signal(ens_prob, regime)

    market_trend = get_market_trend(df_all, window=20)
    if market_trend == "bear" and signal == 1 and ens_prob < 0.65:
        signal = 0

    roll_acc = state["monitor"].rolling_accuracy()
    position_size = position_sizer.compute_size(
        prob=ens_prob, signal=signal, regime=regime, roll_acc=roll_acc,
    )

    result = {
        "as_of_date"   : last_date.strftime("%Y-%m-%d"),
        "predict_for"  : "next trading day",
        "regime"       : regime,
        "global_prob"  : round(g_prob,   4),
        "regime_prob"  : round(r_prob,   4),
        "ensemble_prob": round(ens_prob, 4),
        "signal"       : {1: "LONG", 0: "NEUTRAL", -1: "SHORT"}[signal],
        "position_size": round(position_size, 4),
        "market_trend" : market_trend,
        "roll_acc"     : round(roll_acc, 4),
    }

    print("\n" + "=" * 60)
    print("  📈 下一交易日預測")
    print("=" * 60)
    for k, v in result.items():
        print(f"    {k:14s}: {v}")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    joblib.dump(result, os.path.join(RESULTS_DIR, "latest_prediction.pkl"))

    return result


def main():
    """Entry point invoked by GitHub Actions (.github/workflows/daily.yml)."""
    now_hkt = datetime.now(HKT)
    print(f"🚀 HSI Daily Pipeline — {now_hkt.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    try:
        state = load_pipeline_state()

        if not state:
            state = run_full_bootstrap()
        else:
            state = run_incremental_update(state)

        prediction = predict_next_day(state)

        try:
            from src.email_report import send_email
            send_email(prediction, state.get("last_backtest_report", {}))
        except Exception as e:
            print(f"  ⚠️  email 發送失敗（pipeline 仍視為成功）: {e}")

        print("\n✅ daily_pipeline.py 執行完成")
        return 0

    except Exception as e:
        print(f"\n❌ Pipeline 執行失敗: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
