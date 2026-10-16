# ============================================================
# src/daily_pipeline.py
# GitHub Actions 入口：每個 weekday 16:45 HKT 執行一次。
# 無狀態 runner 設計：
#   - 若 state/models 已存在 → 增量更新 + 預測（不重訓全部）
#   - 若首次執行（無 state）→ 完整 bootstrap（下載 + 初訓 + 模擬）
#
# 本版本整合：
#   - 個股預測（stock_model.run_stock_pipeline），獨立 try/except 容錯
#   - compute_backtest_report() 呼叫修正（接收 sim_df，非 state）
#   - 盤中數據回填 + 技術指標全量重算（方式A）
# ============================================================
import os
import sys
import traceback

import joblib
import pandas as pd

from src import config
from src.config import STATE_PATH, STATE_KEYS, MODELS_DIR, RESULTS_DIR
from src.data_loader import (
    download_all_data, fetch_intraday_hsi, update_last_row_with_intraday,
)
from src.features import build_features
from src.training import (
    run_initial_training, run_daily_simulation,
    save_models, load_models, compute_backtest_report,
)
from src.models import safe_predict, align_features
from src.stock_model import run_stock_pipeline


# ============================================================
# State persistence
# ============================================================
def load_pipeline_state() -> dict:
    if not os.path.exists(STATE_PATH):
        print(f"  ℹ️  {STATE_PATH} 不存在 → 視為首次執行")
        return {}

    try:
        state = joblib.load(STATE_PATH)
    except Exception as e:
        print(f"  ⚠️  state 載入失敗: {e} → 視為首次執行")
        return {}

    if "threshold_calibrator" in state:
        config.threshold_calibrator = state["threshold_calibrator"]
    if "position_sizer" in state:
        config.position_sizer = state["position_sizer"]
    if "ensemble_weighter" in state:
        config.ensemble_weighter = state["ensemble_weighter"]

    model_state = load_models(MODELS_DIR)
    if model_state:
        state.update(model_state)
    else:
        print("  ⚠️  models/ 缺失但 state 存在 → 視為不完整，需 bootstrap")
        return {}

    missing = [k for k in ("global_model", "hmm", "feat_cols") if k not in state]
    if missing:
        print(f"  ⚠️  state 缺少關鍵欄位 {missing} → 視為不完整")
        return {}

    print(f"  ✅ state 已載入，包含 {len(state)} 個欄位")
    return state


def save_pipeline_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)

    save_models(state, MODELS_DIR)

    state["threshold_calibrator"] = config.threshold_calibrator
    state["position_sizer"]       = config.position_sizer
    state["ensemble_weighter"]    = config.ensemble_weighter

    persist_state = {k: state[k] for k in STATE_KEYS if k in state}
    joblib.dump(persist_state, STATE_PATH)

    print(f"  ✅ state 已儲存至 {STATE_PATH}（{len(persist_state)} 個欄位）")


# ============================================================
# 盤中數據回填（方式A：更新OHLC後，全量重跑 build_features()）
# ============================================================
def _apply_intraday_update(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    情境3（盤中執行）時，用即時報價更新 df_all 最後一行的 OHLC，
    再對整個 df_all 重新計算技術指標，確保 RSI/MACD/布林通道等
    與最新價格一致（方式A，已與使用者確認採用）。
    """
    ctx = config.get_prediction_context()
    scenario = ctx["scenario"]

    if scenario != 3:
        return df_all

    print("  📡 情境3（盤中）：取得即時報價並回填最後一行...")
    intraday = fetch_intraday_hsi(fallback_df=df_all)
    if intraday is None:
        print("  ⚠️  盤中報價全部來源失敗，沿用原始收盤數據")
        return df_all

    last_row_updated = update_last_row_with_intraday(df_all.iloc[[-1]], intraday)
    df_all = df_all.copy()
    df_all.iloc[-1] = last_row_updated.iloc[0]

    print("  🔄 重新計算全部技術指標（方式A：全量重跑 build_features）...")
    df_all = build_features(df_all)

    return df_all


# ============================================================
# Bootstrap / 增量更新 / 預測
# ============================================================
def run_full_bootstrap() -> dict:
    print("\n" + "=" * 60)
    print("  🚀 首次執行：完整 Bootstrap")
    print("=" * 60)

    df_all = download_all_data()
    df_all = build_features(df_all)
    df_all = _apply_intraday_update(df_all)

    state = run_initial_training(df_all)
    sim_df = run_daily_simulation(state)
    state["last_backtest_report"] = compute_backtest_report(sim_df)

    return state


def run_incremental_update(state: dict) -> dict:
    print("\n" + "=" * 60)
    print("  🔄 增量更新")
    print("=" * 60)

    df_all = download_all_data()
    df_all = build_features(df_all)
    df_all = _apply_intraday_update(df_all)
    state["df_all"] = df_all

    sim_df = run_daily_simulation(state)
    state["last_backtest_report"] = compute_backtest_report(sim_df)

    return state


def predict_next_day(state: dict) -> dict:
    """
    對最新一行數據產生明日預測。
    對齊 training.py 的 run_daily_simulation() 實際使用的介面：
      - HMMRegime 沒有單行 predict 方法，regime 需透過 add_features()
        對整個 df 計算後，取最後一行的 hmm_regime 欄位。
      - ensemble_weighter.ensemble(g_prob, r_prob) 不吃 regime 參數。
      - threshold_calibrator.get_signal(prob, regime) 直接回傳 -1/0/1，
        不需要自己比較 threshold。
      - position_sizer.compute_size(prob, signal, regime, roll_acc) 需要
        signal 與 roll_acc；首次預測無歷史 roll_acc，用 0.5 中性值。
    """
    df_all     = state["df_all"]
    feat_cols  = state["feat_cols"]
    hmm        = state["hmm"]
    global_m   = state["global_model"]
    regime_ms  = state["regime_models"]

    if "hmm_regime" not in df_all.columns:
        df_all = hmm.add_features(df_all)

    last_row = df_all.iloc[[-1]].copy()
    regime   = str(last_row["hmm_regime"].values[0]) if "hmm_regime" in last_row.columns else "neutral"

    global_prob = safe_predict(global_m, last_row, feat_cols)
    regime_prob = (
        safe_predict(regime_ms[regime], last_row, feat_cols)
        if regime in regime_ms else global_prob
    )

    ensemble_prob = config.ensemble_weighter.ensemble(global_prob, regime_prob)

    signal_code = config.threshold_calibrator.get_signal(ensemble_prob, regime)
    signal = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}.get(signal_code, "NEUTRAL")

    # 首次執行無歷史 roll_acc 可用，沿用 run_daily_simulation() 的初始化慣例（0.5）
    roll_acc = state.get("last_roll_acc", 0.5)
    position_size = config.position_sizer.compute_size(
        prob=ensemble_prob, signal=signal_code, regime=regime, roll_acc=roll_acc,
    )

    prediction = {
        "as_of_date": pd.Timestamp(last_row["date"].values[0]).strftime("%Y-%m-%d"),
        "regime": regime,
        "global_prob": float(global_prob),
        "regime_prob": float(regime_prob),
        "ensemble_prob": float(ensemble_prob),
        "signal": signal,
        "position_size": float(position_size),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    joblib.dump(prediction, os.path.join(RESULTS_DIR, "latest_prediction.pkl"))
    print(f"  ✅ 預測已儲存: {prediction}")

    return prediction



# ============================================================
# Main
# ============================================================
def main() -> int:
    try:
        state = load_pipeline_state()

        if not state:
            state = run_full_bootstrap()
        else:
            state = run_incremental_update(state)

        save_pipeline_state(state)

        prediction = predict_next_day(state)

        # ── 個股預測（獨立流程，失敗不影響恆指核心結果）──────
        try:
            board = run_stock_pipeline(df_all=state["df_all"], hsi_prediction=prediction)
            os.makedirs(RESULTS_DIR, exist_ok=True)
            board.to_csv(os.path.join(RESULTS_DIR, "stock_board.csv"), index=False)
            print(f"  ✅ 個股排行榜已儲存（{len(board)} 檔）")
        except Exception as e:
            print(f"  ⚠️  個股預測流程失敗，郵件將不含個股排行榜: {e}")
            traceback.print_exc()

        print("  ℹ️  Email 將在 charts 生成後，由獨立 workflow step 發送")
        return 0

    except Exception as e:
        print(f"  ❌ Pipeline 執行失敗: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
