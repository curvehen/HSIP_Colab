# ============================================================
# src/training.py
# Merged from: Cell 6 (初始訓練), Cell 7 (逐日模擬 + do_retrain_fixed),
#              Cell 7B (模型儲存), Cell 8 (回測績效計算)
# ============================================================
import os
import joblib
import traceback
import numpy as np
import pandas as pd

from src.config import (
    TRAIN_START, TRAIN_END, SIM_START, TRAIN_WINDOW,
    PROB_LONG, PROB_SHORT, TRANSACTION_COST, RETRAIN_INTERVAL,
    threshold_calibrator, position_sizer, ensemble_weighter,
)
from src.features import get_feat_cols
from src.models import (
    HMMRegime, PerfMonitor, ModelMemory, DriftDetector,
    train_lgbm, eval_model, train_model_fixed,
    align_features, safe_predict, safe_predict_batch,
    evaluate_short_confirmation, get_market_trend,
)


# ============================================================
# Cell 6: 初始訓練
# ============================================================
def run_initial_training(df_all: pd.DataFrame) -> dict:
    """
    對應 Cell 6。返回包含所有初始化物件的 state dict，
    供 training 與 daily pipeline 之後使用。
    """
    print("=" * 60)
    print("  Initial Training")
    print("=" * 60)

    train_mask = (
        (df_all["date"] >= pd.Timestamp(TRAIN_START)) &
        (df_all["date"] <= pd.Timestamp(TRAIN_END))
    )
    df_train = df_all[train_mask].copy().reset_index(drop=True)
    print(f"  訓練集行數 : {len(df_train)} rows ({TRAIN_START} → {TRAIN_END})")

    sim_mask = df_all["date"] >= SIM_START
    df_sim   = df_all[sim_mask].copy().reset_index(drop=True)

    print(f"  訓練集: {len(df_train)} 行 | "
          f"{df_train['date'].min().date()} → {df_train['date'].max().date()}")
    print(f"  模擬集: {len(df_sim)} 行 | "
          f"{df_sim['date'].min().date()} → {df_sim['date'].max().date()}")

    # ── HMM Regime ────────────────────────────────────────────
    print("\n  訓練 HMM Regime 分類器...")
    hmm = HMMRegime(n_states=3)
    hmm.fit(df_train)
    df_all   = hmm.add_features(df_all)
    df_train = df_all[train_mask].copy().reset_index(drop=True)
    df_sim   = df_all[sim_mask].copy().reset_index(drop=True)

    print(f"  Regime 分佈（訓練集）:")
    print(df_train["hmm_regime"].value_counts().to_string())

    feat_cols = get_feat_cols(df_all)

    # ── 初始模型訓練 ─────────────────────────────────────────
    print("\n  訓練初始 LightGBM 模型...")
    val_cut  = int(len(df_train) * 0.80)
    X_tr     = df_train[feat_cols].iloc[:val_cut]
    y_tr     = df_train["target"].iloc[:val_cut]
    X_val    = df_train[feat_cols].iloc[val_cut:]
    y_val    = df_train["target"].iloc[val_cut:]

    init_model   = train_lgbm(X_tr, y_tr, X_val, y_val)
    init_metrics = eval_model(init_model, X_val, y_val)

    print(f"  初始模型表現（Validation Set）:")
    print(f"    AUC      : {init_metrics['auc']:.4f}")
    print(f"    命中率   : {init_metrics['accuracy']:.2%}")
    print(f"    Brier    : {init_metrics['brier']:.4f}")

    # ── 按 Regime 分拆模型 ───────────────────────────────────
    print("\n  訓練 Regime-specific 模型...")
    regime_models = {}
    for regime in ["bull", "bear", "neutral"]:
        rdf = df_train[df_train["hmm_regime"] == regime]
        if len(rdf) < 60:
            print(f"    {regime}: 樣本不足（{len(rdf)}），跳過")
            continue
        vc     = int(len(rdf) * 0.80)
        Xr_tr  = rdf[feat_cols].iloc[:vc]
        yr_tr  = rdf["target"].iloc[:vc]
        Xr_val = rdf[feat_cols].iloc[vc:]
        yr_val = rdf["target"].iloc[vc:]
        if len(yr_tr.unique()) < 2 or len(yr_val.unique()) < 2:
            continue
        try:
            rm = train_lgbm(Xr_tr, yr_tr, Xr_val, yr_val)
            m  = eval_model(rm, Xr_val, yr_val)
            regime_models[regime] = rm
            print(f"    {regime:8s}: AUC={m['auc']:.4f} | "
                  f"命中率={m['accuracy']:.2%} | 樣本={len(rdf)}")
        except Exception as e:
            print(f"    {regime}: 訓練失敗 {e}")

    # ── 初始化自學組件 ───────────────────────────────────────
    monitor = PerfMonitor()
    memory  = ModelMemory()
    memory.save(init_model, init_metrics["auc"], {"phase": "init"})

    key_feats = [
        f for f in ["ret_1d", "vol_20", "rsi_14", "bb_pos",
                    "nb_zscore_lag1", "sp500_lag1", "vix"]
        if f in feat_cols
    ]
    drift = DriftDetector(key_feats)
    drift.fit_ref(df_train[feat_cols])

    global_model = init_model

    os.makedirs("models", exist_ok=True)
    base_m = (global_model.estimator
              if hasattr(global_model, "estimator")
              else global_model)
    feat_cols_fixed = list(base_m.feature_name_)
    joblib.dump(feat_cols_fixed, "models/feat_cols_fixed.pkl")
    print(f"  feat_cols_fixed.pkl saved: {len(feat_cols_fixed)} features")

    print(f"\n  Initial Training Complete")
    print(f"  Init Model AUC : {init_metrics['auc']:.4f}")
    print(f"  Regime Models  : {list(regime_models.keys())}")
    print("✅ training.py: run_initial_training() 完成")

    return {
        "df_all"       : df_all,
        "df_train"     : df_train,
        "df_sim"       : df_sim,
        "feat_cols"    : feat_cols,
        "hmm"          : hmm,
        "global_model" : global_model,
        "regime_models": regime_models,
        "monitor"      : monitor,
        "memory"       : memory,
        "drift"        : drift,
    }


# ============================================================
# Cell 7: do_retrain_fixed
# ============================================================
def do_retrain_fixed(train_df, reasons, fold_idx, feat_cols, state: dict) -> str:
    """
    對應 Cell 7 的 do_retrain_fixed()。
    原版用 global 讀寫 global_model/regime_models/hmm，
    這裡改為讀寫傳入的 state dict（in-place 更新）。
    """
    global_model  = state["global_model"]
    regime_models = state["regime_models"]
    hmm           = state["hmm"]
    memory        = state["memory"]

    half_window = TRAIN_WINDOW // 2

    if "性能退化" in reasons:
        n      = len(train_df)
        use_df = (train_df.iloc[-TRAIN_WINDOW:] if n >= TRAIN_WINDOW else
                  train_df.iloc[-half_window:]  if n >= half_window  else
                  train_df)
        label  = "近期窗口重訓"

    elif "特徵漂移" in reasons:
        use_df = train_df
        try:
            hmm.fit(use_df)
            use_df = hmm.add_features(use_df)
        except Exception:
            pass
        label  = "完整重訓"

    else:
        use_df = (train_df.iloc[-TRAIN_WINDOW:]
                  if len(train_df) >= TRAIN_WINDOW else train_df)
        label  = "定期重訓"

    if len(use_df) < 120:
        return "數據不足，跳過"

    new_model, new_auc = train_model_fixed(use_df, feat_cols)
    if new_model is None:
        return "訓練條件不足，跳過"

    if memory.should_rollback(new_auc):
        global_model = memory.get_best()[0]
        action = f"{label}→回滾(新={new_auc:.3f}<最佳={memory.best_auc:.3f})"
    else:
        global_model = new_model
        memory.save(new_model, new_auc, {
            "fold": fold_idx, "reasons": reasons, "label": label,
        })
        action = f"{label}→採用(AUC={new_auc:.3f})"

    if "hmm_regime" in use_df.columns:
        for regime in ["bull", "bear", "neutral"]:
            rdf = use_df[use_df["hmm_regime"] == regime]
            if len(rdf) < 80:
                continue
            rm, r_auc = train_model_fixed(rdf, feat_cols)
            if rm is not None:
                regime_models[regime] = rm

    # ── 寫回 state ────────────────────────────────────────────
    state["global_model"]  = global_model
    state["regime_models"] = regime_models
    state["hmm"]            = hmm

    return action


# ============================================================
# Cell 7: 逐日模擬
# ============================================================
def run_daily_simulation(state: dict) -> pd.DataFrame:
    """
    對應 Cell 7 主迴圈。state dict 需含:
    df_all, feat_cols, global_model, regime_models, hmm,
    monitor, memory, drift
    就地更新 state（retrain 會改變 global_model/regime_models），
    返回 sim_df。
    """
    df_all        = state["df_all"]
    feat_cols     = state["feat_cols"]
    global_model  = state["global_model"]
    regime_models = state["regime_models"]
    hmm           = state["hmm"]
    monitor       = state["monitor"]
    memory        = state["memory"]
    drift         = state["drift"]

    assert global_model is not None, "global_model 未定義"
    assert isinstance(memory,  ModelMemory),  "memory 未定義"
    assert isinstance(monitor, PerfMonitor),  "monitor 未定義"
    assert isinstance(drift,   DriftDetector),"drift 未定義"
    assert isinstance(hmm,     HMMRegime),    "hmm 未定義"

    print("✅ do_retrain_fixed 已定義，開始模擬...")
    print(f"  TRAIN_WINDOW = {TRAIN_WINDOW} days")

    sim_mask = df_all["date"] >= SIM_START
    df_sim   = df_all[sim_mask].copy().reset_index(drop=True)
    sim_records: list = []

    roll_acc           = 0.5
    roll_auc           = 0.5
    days_since_retrain = 0
    wf_monthly: dict   = {}

    print(f"\n  模擬日數: {len(df_sim)}")

    for sim_idx in range(len(df_sim)):

        sim_row  = df_sim.iloc[sim_idx]
        sim_date = pd.Timestamp(sim_row["date"])

        known_mask = df_all["date"] < sim_date
        known_df   = df_all[known_mask].copy()

        if len(known_df) == 0:
            continue

        if "hmm_regime" not in known_df.columns:
            try:
                known_df = hmm.add_features(known_df)
            except Exception:
                known_df["hmm_regime"] = "neutral"

        last_known = known_df.iloc[[-1]]

        g_prob = safe_predict(global_model, last_known, feat_cols)

        regime = (str(last_known["hmm_regime"].values[0])
                  if "hmm_regime" in last_known.columns else "neutral")

        r_prob = g_prob
        if regime in regime_models:
            r_prob = safe_predict(regime_models[regime], last_known, feat_cols)

        if sim_idx > 0 and len(sim_records) > 0:
            prev = sim_records[-1]
            if "actual_target" in prev:
                ensemble_weighter.update("global", prev["global_prob"], prev["actual_target"])
                ensemble_weighter.update("regime", prev["regime_prob"], prev["actual_target"])

        ens_prob = ensemble_weighter.ensemble(g_prob, r_prob)

        if sim_idx == 60 and len(sim_records) >= 30:
            cal_df = pd.DataFrame(sim_records)
            if all(c in cal_df.columns for c in ["regime", "ensemble_prob", "actual_target"]):
                threshold_calibrator.calibrate(
                    cal_df, prob_col="ensemble_prob",
                    target_col="actual_target", regime_col="regime")

        if sim_idx > 60 and sim_idx % 42 == 0 and len(sim_records) >= 60:
            cal_df = pd.DataFrame(sim_records[-120:])
            if all(c in cal_df.columns for c in ["regime", "ensemble_prob", "actual_target"]):
                threshold_calibrator.calibrate(
                    cal_df, prob_col="ensemble_prob",
                    target_col="actual_target", regime_col="regime")

        signal = threshold_calibrator.get_signal(ens_prob, regime)

        market_trend = get_market_trend(known_df, window=20)

        if market_trend == "bear" and signal == 1 and ens_prob < 0.65:
            signal = 0

        short_confirm = {"confirmed": False, "score": 0, "reasons": []}
        if signal == -1:
            short_confirm = evaluate_short_confirmation(last_known, regime, ens_prob)
            if not short_confirm["confirmed"]:
                signal = 0
            elif market_trend == "bull" and ens_prob > 0.35:
                signal = 0

        if sim_idx > 20 and roll_acc < 0.40:
            signal = 0

        position_size = position_sizer.compute_size(
            prob=ens_prob, signal=signal, regime=regime,
            roll_acc=roll_acc if sim_idx > 5 else 0.5,
        )

        actual_ret    = float(sim_row.get("next_ret",   0))
        actual_target = int(sim_row.get("target",       0))

        correct = None
        if signal == 1:
            correct = bool(actual_target == 1)
        elif signal == -1:
            correct = bool(actual_target == 0)

        if signal != 0:
            strat_ret  = position_size * signal * actual_ret
            strat_ret -= TRANSACTION_COST * position_size
        else:
            strat_ret  = 0.0

        monitor.update(ens_prob, actual_target)
        roll_acc = monitor.rolling_accuracy()
        roll_auc = monitor.rolling_auc()

        perf_trigger,  perf_reason = monitor.should_retrain()
        drift_trigger, drift_feats = drift.detect(align_features(last_known, feat_cols))
        days_since_retrain += 1

        periodic = False
        if days_since_retrain >= RETRAIN_INTERVAL:
            if roll_acc >= 0.58:
                days_since_retrain = 0
            else:
                periodic = True

        reasons = []
        if perf_trigger:  reasons.append("性能退化")
        if drift_trigger: reasons.append("特徵漂移")
        if periodic:      reasons.append("定期更新")

        do_rt  = bool(reasons)
        action = "—"

        if do_rt:
            retrain_all = df_all[df_all["date"] <= sim_date].copy()
            retrain_df  = (retrain_all.iloc[-TRAIN_WINDOW:]
                           if len(retrain_all) >= TRAIN_WINDOW else retrain_all)

            if "hmm_regime" not in retrain_df.columns:
                try:
                    retrain_df = hmm.add_features(retrain_df)
                except Exception:
                    retrain_df["hmm_regime"] = "neutral"

            action = do_retrain_fixed(retrain_df, reasons, sim_idx, feat_cols, state)
            # state 已被 do_retrain_fixed 就地更新，重新取出最新引用
            global_model  = state["global_model"]
            regime_models = state["regime_models"]
            hmm           = state["hmm"]

            try:
                drift.fit_ref(align_features(
                    retrain_df[feat_cols].iloc[-TRAIN_WINDOW:], feat_cols))
            except Exception:
                pass

            days_since_retrain = 0

        month_key = sim_date.strftime("%Y-%m")
        if month_key not in wf_monthly:
            wf_monthly[month_key] = {"trades": 0, "correct": 0, "ret": 0.0, "long": 0, "short": 0}

        if signal != 0:
            wf_monthly[month_key]["trades"] += 1
            wf_monthly[month_key]["ret"]    += strat_ret
            if signal == 1:
                wf_monthly[month_key]["long"]  += 1
            else:
                wf_monthly[month_key]["short"] += 1
            if correct:
                wf_monthly[month_key]["correct"] += 1

        sim_records.append({
            "date"               : sim_date,
            "close"              : float(sim_row.get("close", 0)),
            "regime"             : regime,
            "global_prob"        : round(g_prob,          4),
            "regime_prob"        : round(r_prob,          4),
            "ensemble_prob"      : round(ens_prob,        4),
            "signal"             : signal,
            "position_size"      : round(position_size,   4),
            "actual_ret"         : actual_ret,
            "strat_ret"          : round(strat_ret,       6),
            "actual_target"      : actual_target,
            "correct"            : correct,
            "retrain"            : do_rt,
            "retrain_reason"     : ",".join(reasons),
            "action"             : action,
            "roll_acc"           : round(roll_acc,        4),
            "roll_auc"           : round(roll_auc,        4),
            "best_model_auc"     : round(memory.best_auc, 4),
            "short_confirm_score": short_confirm["score"],
            "ens_weight_summary" : ensemble_weighter.weight_summary(),
            "market_trend"       : market_trend,
        })

    sim_df = pd.DataFrame(sim_records)

    if len(sim_df) == 0:
        print("\n  ⚠️  無模擬記錄")
        return sim_df

    sim_df["cum_strat"]  = (1 + sim_df["strat_ret"]).cumprod()
    sim_df["cum_market"] = (1 + sim_df["actual_ret"]).cumprod()

    traded        = sim_df[sim_df["signal"] != 0]
    total_trades  = len(traded)
    n_correct     = traded["correct"].sum() if total_trades > 0 else 0
    hit_rate      = n_correct / total_trades if total_trades > 0 else 0

    daily_strat   = sim_df["strat_ret"]
    sharpe        = (daily_strat.mean() / (daily_strat.std() + 1e-9) * np.sqrt(252))

    cum_ret       = sim_df["cum_strat"]
    rolling_max   = cum_ret.cummax()
    drawdown      = (cum_ret - rolling_max) / (rolling_max + 1e-9)
    max_drawdown  = drawdown.min()

    total_return  = cum_ret.iloc[-1] - 1
    market_return = sim_df["cum_market"].iloc[-1] - 1

    n_retrains = int(sim_df["retrain"].sum())

    print("\n" + "=" * 60)
    print("  逐日模擬完成")
    print("=" * 60)
    print(f"  模擬天數       : {len(sim_df)}")
    print(f"  交易次數       : {total_trades}")
    print(f"  命中率         : {hit_rate:.2%}")
    print(f"  策略總回報     : {total_return:.2%}")
    print(f"  市場總回報     : {market_return:.2%}")
    print(f"  Sharpe 比率    : {sharpe:.3f}")
    print(f"  最大回撤       : {max_drawdown:.2%}")
    print(f"  重訓練次數     : {n_retrains}")
    print(f"  Rollback 次數  : {memory.rollbacks}")
    print("=" * 60)

    # ── 寫回最終 state ──────────────────────────────────────
    state["global_model"]  = global_model
    state["regime_models"] = regime_models
    state["hmm"]           = hmm
    state["monitor"]       = monitor
    state["memory"]        = memory
    state["drift"]         = drift
    state["wf_monthly"]    = wf_monthly

    print("✅ training.py: run_daily_simulation() 完成")

    return sim_df


# ============================================================
# Cell 7B: 模型持久化
# ============================================================
def save_models(state: dict, out_dir: str = "models") -> None:
    """
    對應 Cell 7B。將訓練/模擬完成後的模型與自學狀態物件
    寫入磁碟，供 GitHub Actions 下次 run 時載入，避免每日重訓。
    """
    os.makedirs(out_dir, exist_ok=True)

    global_model  = state["global_model"]
    regime_models = state["regime_models"]
    hmm           = state["hmm"]
    feat_cols     = state["feat_cols"]

    joblib.dump(global_model,  os.path.join(out_dir, "global_model.pkl"))
    joblib.dump(regime_models, os.path.join(out_dir, "regime_models.pkl"))
    joblib.dump(hmm,           os.path.join(out_dir, "hmm_model.pkl"))
    joblib.dump(feat_cols,     os.path.join(out_dir, "feat_cols.pkl"))

    base_m = (global_model.estimator
              if hasattr(global_model, "estimator")
              else global_model)
    if hasattr(base_m, "feature_name_"):
        joblib.dump(list(base_m.feature_name_),
                    os.path.join(out_dir, "feat_cols_fixed.pkl"))

    print(f"  ✅ 模型已儲存至 {out_dir}/")
    print(f"     global_model.pkl, regime_models.pkl, "
          f"hmm_model.pkl, feat_cols.pkl")


def load_models(in_dir: str = "models") -> dict:
    """
    對應 Cell 7B 的反向操作（原版未顯式包含 load，
    但 daily_pipeline.py 需要此函數才能在 GitHub Actions
    的無狀態 runner 上避免每日重訓）。
    """
    required = ["global_model.pkl", "regime_models.pkl",
                "hmm_model.pkl", "feat_cols.pkl"]
    missing  = [f for f in required
                if not os.path.exists(os.path.join(in_dir, f))]
    if missing:
        print(f"  ⚠️  模型檔案缺失: {missing} → 需執行初始訓練")
        return {}

    state = {
        "global_model" : joblib.load(os.path.join(in_dir, "global_model.pkl")),
        "regime_models": joblib.load(os.path.join(in_dir, "regime_models.pkl")),
        "hmm"          : joblib.load(os.path.join(in_dir, "hmm_model.pkl")),
        "feat_cols"    : joblib.load(os.path.join(in_dir, "feat_cols.pkl")),
    }
    print(f"  ✅ 模型已從 {in_dir}/ 載入")
    return state


# ============================================================
# Cell 8: 回測績效報告
# ============================================================
def compute_backtest_report(sim_df: pd.DataFrame) -> dict:
    """
    對應 Cell 8。根據 run_daily_simulation() 產出的 sim_df
    計算完整回測績效指標（年化報酬、月度統計、regime 分解等）。
    """
    if len(sim_df) == 0:
        print("  ⚠️  sim_df 為空，無法計算回測報告")
        return {}

    traded = sim_df[sim_df["signal"] != 0].copy()

    total_days    = len(sim_df)
    total_trades  = len(traded)
    n_correct     = int(traded["correct"].sum()) if total_trades > 0 else 0
    hit_rate      = n_correct / total_trades if total_trades > 0 else 0.0

    long_trades   = traded[traded["signal"] == 1]
    short_trades  = traded[traded["signal"] == -1]
    long_hit      = (long_trades["correct"].mean()
                     if len(long_trades) > 0 else 0.0)
    short_hit     = (short_trades["correct"].mean()
                     if len(short_trades) > 0 else 0.0)

    total_return  = float(sim_df["cum_strat"].iloc[-1]  - 1)
    market_return = float(sim_df["cum_market"].iloc[-1] - 1)

    n_years       = max(total_days / 252, 1e-6)
    cagr          = (1 + total_return) ** (1 / n_years) - 1

    daily_strat   = sim_df["strat_ret"]
    sharpe        = float(daily_strat.mean() / (daily_strat.std() + 1e-9) * np.sqrt(252))
    downside      = daily_strat[daily_strat < 0]
    sortino       = float(
        daily_strat.mean() / (downside.std() + 1e-9) * np.sqrt(252)
        if len(downside) > 0 else 0.0
    )

    cum_ret       = sim_df["cum_strat"]
    rolling_max   = cum_ret.cummax()
    drawdown      = (cum_ret - rolling_max) / (rolling_max + 1e-9)
    max_drawdown  = float(drawdown.min())
    calmar        = float(cagr / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    win_trades    = traded[traded["strat_ret"] > 0]
    loss_trades   = traded[traded["strat_ret"] < 0]
    avg_win       = float(win_trades["strat_ret"].mean())  if len(win_trades)  > 0 else 0.0
    avg_loss      = float(loss_trades["strat_ret"].mean()) if len(loss_trades) > 0 else 0.0
    profit_factor = (
        float(win_trades["strat_ret"].sum() / abs(loss_trades["strat_ret"].sum()))
        if len(loss_trades) > 0 and loss_trades["strat_ret"].sum() != 0 else 0.0
    )

    # ── Regime 分解 ──────────────────────────────────────────
    regime_breakdown = {}
    for regime in ["bull", "bear", "neutral"]:
        rdf = traded[traded["regime"] == regime]
        if len(rdf) == 0:
            continue
        regime_breakdown[regime] = {
            "trades"  : int(len(rdf)),
            "hit_rate": round(float(rdf["correct"].mean()), 4),
            "avg_ret" : round(float(rdf["strat_ret"].mean()), 6),
        }

    # ── 月度統計 ─────────────────────────────────────────────
    sim_df["month"] = sim_df["date"].dt.strftime("%Y-%m")
    monthly = (
        sim_df.groupby("month")
        .agg(
            trades  = ("signal",    lambda s: int((s != 0).sum())),
            ret     = ("strat_ret", "sum"),
        )
        .reset_index()
    )
    monthly_records = monthly.to_dict(orient="records")

    report = {
        "total_days"       : total_days,
        "total_trades"     : total_trades,
        "hit_rate"         : round(hit_rate, 4),
        "long_trades"      : int(len(long_trades)),
        "long_hit_rate"    : round(float(long_hit), 4),
        "short_trades"     : int(len(short_trades)),
        "short_hit_rate"   : round(float(short_hit), 4),
        "total_return"     : round(total_return,  4),
        "market_return"    : round(market_return, 4),
        "cagr"             : round(cagr,     4),
        "sharpe"           : round(sharpe,   4),
        "sortino"          : round(sortino,  4),
        "max_drawdown"     : round(max_drawdown, 4),
        "calmar"           : round(calmar,   4),
        "avg_win"          : round(avg_win,  6),
        "avg_loss"         : round(avg_loss, 6),
        "profit_factor"    : round(profit_factor, 4),
        "regime_breakdown" : regime_breakdown,
        "monthly"          : monthly_records,
    }

    print("\n" + "=" * 60)
    print("  📊 回測績效報告 (Cell 8)")
    print("=" * 60)
    print(f"  總天數          : {report['total_days']}")
    print(f"  交易次數        : {report['total_trades']} "
          f"(多: {report['long_trades']} / 空: {report['short_trades']})")
    print(f"  整體命中率      : {report['hit_rate']:.2%}")
    print(f"  多單命中率      : {report['long_hit_rate']:.2%}")
    print(f"  空單命中率      : {report['short_hit_rate']:.2%}")
    print(f"  策略總回報      : {report['total_return']:.2%}")
    print(f"  市場總回報      : {report['market_return']:.2%}")
    print(f"  年化報酬(CAGR)  : {report['cagr']:.2%}")
    print(f"  Sharpe          : {report['sharpe']:.3f}")
    print(f"  Sortino         : {report['sortino']:.3f}")
    print(f"  最大回撤        : {report['max_drawdown']:.2%}")
    print(f"  Calmar          : {report['calmar']:.3f}")
    print(f"  Profit Factor   : {report['profit_factor']:.3f}")
    print(f"\n  Regime 分解:")
    for regime, stats in regime_breakdown.items():
        print(f"    {regime:8s}: 交易={stats['trades']:3d} | "
              f"命中率={stats['hit_rate']:.2%} | "
              f"均報酬={stats['avg_ret']:.4%}")
    print("=" * 60)
    print("✅ training.py: compute_backtest_report() 完成")

    return report
