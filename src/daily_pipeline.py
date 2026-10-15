# ============================================================
# src/daily_pipeline.py
# GitHub Actions 入口：每個 weekday 09:00 HKT 執行一次。
# 無狀態 runner 設計：
#   - 若 state/models 已存在 → 增量更新 + 預測（不重訓全部）
#   - 若首次執行（無 state）→ 完整 bootstrap（下載 + 初訓 + 模擬）
# ============================================================
import os
import joblib
import pandas as pd

from src import config
from src.config import STATE_PATH, STATE_KEYS, MODELS_DIR, RESULTS_DIR
from src.data_loader import download_all_data
from src.features import build_features
from src.training import (
    run_initial_training, run_daily_simulation,
    save_models, load_models, compute_backtest_report,
)
from src.models import safe_predict, align_features


# ============================================================
# State persistence
# ============================================================
def load_pipeline_state() -> dict:
    """
    從 state/pipeline_state.pkl 載入上次執行狀態。
    關鍵修正：threshold_calibrator / position_sizer / ensemble_weighter
    是 config.py 的 module-level singleton，必須在此用 state 裡已學習到
    的物件「覆蓋」config 模組變數，否則每次 process 重啟都會退回
    default 0.5/0.5 權重，等於白學。
    """
    if not os.path.exists(STATE_PATH):
        print(f"  ℹ️  {STATE_PATH} 不存在 → 視為首次執行")
        return {}

    try:
        state = joblib.load(STATE_PATH)
    except Exception as e:
        print(f"  ⚠️  state 載入失敗: {e} → 視為首次執行")
        return {}

    # ── 關鍵修正：還原自學 singleton 到 config 模組 ──────────
    if "threshold_calibrator" in state:
        config.threshold_calibrator = state["threshold_calibrator"]
    if "position_sizer" in state:
        config.position_sizer = state["position_sizer"]
    if "ensemble_weighter" in state:
        config.ensemble_weighter = state["ensemble_weighter"]

    # ── 模型檔案獨立存於 models/，而非塞進 pickle 內 ─────────
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
    """
    拆分儲存：
      - 大型模型物件 (global_model/regime_models/hmm/feat_cols)
        → save_models() 寫入 models/（Cell 7B 邏輯）
      - 其餘自學狀態 + 最新 df_all/backtest report
        → joblib.dump 整包寫入 state/pipeline_state.pkl
    config 模組的 singleton 在儲存前先同步進 state dict，
    確保下次 load 時能正確還原。
    """
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)

    save_models(state, MODELS_DIR)

    state["threshold_calibrator"] = config.threshold_calibrator
    state["position_sizer"]       = config.position_sizer
    state["ensemble_weighter"]    = config.ensemble_weighter

    persist_state = {k: state[k] for k in STATE_KEYS if k in state}
    joblib.dump(persist_state, STATE_PATH)

    print(f"  ✅ state 已儲存至 {STATE_PATH}（{len(persist_state)} 個欄位）")


# ============================================================
# Bootstrap：首次執行，無既有 state/models
# ============================================================
def run_full_bootstrap() -> dict:
    print("=" * 60)
    print("  Full Bootstrap（首次執行 / state 不完整）")
    print("=" * 60)

    df_raw  = download_all_data()
    df_feat = build_features(df_raw)

    state = run_initial_training(df_feat)

    sim_df = run_daily_simulation(state)
    report = compute_backtest_report(sim_df)
    state["last_backtest_report"] = report

    sim_df.to_csv(os.path.join(RESULTS_DIR, "sim_records.csv"), index=False)

    save_pipeline_state(state)
    print("✅ run_full_bootstrap() 完成")
    return state


# ============================================================
# 增量更新：已有 state/models，只需追加新一天資料
# ============================================================
def run_incremental_update(state: dict) -> dict:
    print("=" * 60)
    print("  Incremental Update（增量更新）")
    print("=" * 60)

    df_raw  = download_all_data()
    df_feat = build_features(df_raw)

    state["df_all"]    = df_feat
    hmm                = state["hmm"]
    try:
        state["df_all"] = hmm.add_features(df_feat)
    except Exception as e:
        print(f"  ⚠️  HMM add_features 失敗: {e}")

    sim_df = run_daily_simulation(state)
    report = compute_backtest_report(sim_df)
    state["last_backtest_report"] = report

    sim_df.to_csv(os.path.join(RESULTS_DIR, "sim_records.csv"), index=False)

    save_pipeline_state(state)
    print("✅ run_incremental_update() 完成")
    return state


# ============================================================
# 隔日預測（Cell 9 邏輯對應）
# ============================================================
def predict_next_day(state: dict) -> dict:
    df_all    = state["df_all"]
    feat_cols = state["feat_cols"]
    global_model  = state["global_model"]
    regime_models = state["regime_models"]

    last_row = df_all.iloc[[-1]]
    regime = (str(last_row["hmm_regime"].values[0])
              if "hmm_regime" in last_row.columns else "neutral")

    g_prob = safe_predict(global_model, last_row, feat_cols)
    r_prob = g_prob
    if regime in regime_models:
        r_prob = safe_predict(regime_models[regime], last_row, feat_cols)

    ens_prob = config.ensemble_weighter.ensemble(g_prob, r_prob)
    signal_code = config.threshold_calibrator.get_signal(ens_prob, regime)
    signal = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}[signal_code]

    position_size = config.position_sizer.compute_size(
        prob=ens_prob, signal=signal_code, regime=regime, roll_acc=0.5,
    )

    prediction = {
        "as_of_date"   : pd.Timestamp(df_all["date"].iloc[-1]).strftime("%Y-%m-%d"),
        "regime"       : regime,
        "global_prob"  : round(g_prob,   4),
        "regime_prob"  : round(r_prob,   4),
        "ensemble_prob": round(ens_prob, 4),
        "signal"       : signal,
        "position_size": round(position_size, 4),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    joblib.dump(prediction, os.path.join(RESULTS_DIR, "latest_prediction.pkl"))

    print("\n" + "=" * 60)
    print("  📈 隔日預測結果")
    print("=" * 60)
    for k, v in prediction.items():
        print(f"  {k:14s}: {v}")
    print("=" * 60)

    return prediction


# ============================================================
# Entry point
# ============================================================
def main():
    try:
        state = load_pipeline_state()

        if not state:
            state = run_full_bootstrap()
        else:
            state = run_incremental_update(state)

        predict_next_day(state)

        print("  ℹ️  Email 將在 charts 生成後，由獨立 workflow step 發送")
        print("\n✅ daily_pipeline.py 執行完成")
        return 0

    except Exception as e:
        import traceback
        print(f"\n❌ daily_pipeline.py 執行失敗: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
