# ============================================================
# src/training.py
# Merged from: Cell 6 (初始訓練), Cell 7 (逐日模擬 + do_retrain_fixed),
#              Cell 7B (模型儲存), Cell 8 (回測績效計算)
# HL (高低位) 模型整合：依共識接入初訓/重訓/持久化流程
# ============================================================
import os
import joblib
import traceback
import numpy as np
import pandas as pd

from src import config as cfg
from src.config import (
    TRAIN_START, TRAIN_END, SIM_START, TRAIN_WINDOW,
    PROB_LONG, PROB_SHORT, TRANSACTION_COST, RETRAIN_INTERVAL,
)
from src.features import get_feat_cols
from src.models import (
    HMMRegime, PerfMonitor, ModelMemory, DriftDetector,
    train_lgbm, eval_model, train_model_fixed,
    align_features, safe_predict, safe_predict_batch,
    evaluate_short_confirmation, get_market_trend,
)
from src.hl_model import (
    build_hl_targets, build_hl_features, get_hl_feat_cols, train_all_hl_models,
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
    df_train = df_all[train_mask].copy()
    # 防呆：若 TRAIN_END 恰好設為接近今天的日期，最後一行 target 可能為
    # NaN，訓練時需明確排除（正常情況下 TRAIN_END 是歷史日期，此行通常無影響）
    df_train = df_train.dropna(subset=["target"]).reset_index(drop=True)
    print(f"  訓練集行數 : {len(df_train)} rows ({TRAIN_START} → {TRAIN_END})")

    # 模擬迴圈需要已知的 target 才能驗證對錯，排除最新尚無結果的一行
    # （features.py 已不再自動過濾，此處明確處理，避免模擬loop收到NaN）
    sim_mask = df_all["date"] >= SIM_START
    df_sim   = df_all[sim_mask].copy()
    df_sim   = df_sim.dropna(subset=["target", "next_ret"]).reset_index(drop=True)



    print(f"  訓練集: {len(df_train)} 行 | "
          f"{df_train['date'].min().date()} → {df_train['date'].max().date()}")
    print(f"  模擬集: {len(df_sim)} 行 | "
          f"{df_sim['date'].min().date()} → {df_sim['date'].max().date()}")

    # ── HMM Regime ──────────────────────────────────────────
    print("\n  訓練 HMM Regime 分類器...")
    hmm = HMMRegime(n_states=3)
    hmm.fit(df_train)
    df_all   = hmm.add_features(df_all)
    df_train = df_all[train_mask].copy().dropna(subset=["target"]).reset_index(drop=True)
    df_sim   = df_all[sim_mask].copy().dropna(subset=["target"]).reset_index(drop=True)

    print(f"  Regime 分布（訓練集）:")
    print(df_train["hmm_regime"].value_counts().to_string())

    feat_cols = get_feat_cols(df_all)

    # ── 初始模型訓練 ──────────────────────────────────────────
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

    # ── 按 Regime 分拆模型 ──────────────────────────────────
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

    # ── 初始化自學組件 ──────────────────────────────────────
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

    # ── 訓練 HL (高低位) 模型 ──────────────────────────────
    # 依共識接入，失敗不阻擋主流程（HL 屬附加功能）
    print("\n  訓練 HL (高低位) 模型...")
    try:
        df_hl = build_hl_targets(df_all)
        df_hl = build_hl_features(df_hl)
        hl_feat_cols, hl_extra_cols = get_hl_feat_cols(df_hl, feat_cols)
        df_train_hl = df_hl[train_mask].copy().reset_index(drop=True)
        hl_models, hl_metrics = train_all_hl_models(df_train_hl, hl_feat_cols)
    except Exception as e:
        print(f"  ⚠️  HL模型訓練整體失敗，跳過: {e}")
        hl_models, hl_feat_cols, hl_metrics = {}, [], {}

    print(f"\n  Initial Training Complete")
    print(f"  Init Model AUC : {init_metrics['auc']:.4f}")
    print(f"  Regime Models  : {list(regime_models.keys())}")
    print(f"  HL Models      : {list(hl_models.keys())}")
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
        "hl_models"    : hl_models,
        "hl_feat_cols" : hl_feat_cols,
        "hl_metrics"   : hl_metrics,
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

    # ── HL (高低位) 模型同步重訓 ──────────────────────────
    # 依共識：沿用分類模型的觸發條件，不設獨立判斷邏輯
    hl_feat_cols = state.get("hl_feat_cols")
    if hl_feat_cols:
        try:
            df_hl_use = build_hl_targets(use_df)
            df_hl_use = build_hl_features(df_hl_use)
            new_hl_models, new_hl_metrics = train_all_hl_models(df_hl_use, hl_feat_cols)
            if new_hl_models:
                state["hl_models"]  = new_hl_models
                state["hl_metrics"] = new_hl_metrics
        except Exception as e:
            print(f"    ⚠️  HL模型重訓失敗，沿用舊模型: {e}")

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
    df_sim   = df_all[sim_mask].copy()
    df_sim   = df_sim.dropna(subset=["target"]).reset_index(drop=True)
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
                cfg.ensemble_weighter.update("global", prev["global_prob"], prev["actual_target"])
                cfg.ensemble_weighter.update("regime", prev["regime_prob"], prev["actual_target"])

        ens_prob = cfg.ensemble_weighter.ensemble(g_prob, r_prob)

        if sim_idx == 60 and len(sim_records) >= 30:
            cal_df = pd.DataFrame(sim_records)
            if all(c in cal_df.columns for c in ["regime", "ensemble_prob", "actual_target"]):
                cfg.threshold_calibrator.calibrate(
                    cal_df, prob_col="ensemble_prob",
                    target_col="actual_target", regime_col="regime")

        if sim_idx > 60 and sim_idx % 42 == 0 and len(sim_records) >= 60:
            cal_df = pd.DataFrame(sim_records[-120:])
            if all(c in cal_df.columns for c in ["regime", "ensemble_prob", "actual_target"]):
                cfg.threshold_calibrator.calibrate(
                    cal_df, prob_col="ensemble_prob",
                    target_col="actual_target", regime_col="regime")

        signal = cfg.threshold_calibrator.get_signal(ens_prob, regime)

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

        position_size = cfg.position_sizer.compute_size(
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
            wf_monthly[month_key] = {"ret": 0.0, "n_trades": 0, "n_correct": 0}
        wf_monthly[month_key]["ret"]       += strat_ret
        if signal != 0:
            wf_monthly[month_key]["n_trades"] += 1
            if correct:
                wf_monthly[month_key]["n_correct"] += 1

        sim_records.append({
            "date"          : sim_date,
            "regime"        : regime,
            "global_prob"   : g_prob,
            "regime_prob"   : r_prob,
            "ensemble_prob" : ens_prob,
            "signal"        : signal,
            "position_size" : position_size,
            "actual_target" : actual_target,
            "actual_ret"    : actual_ret,
            "strat_ret"     : strat_ret,
            "correct"       : correct,
            "roll_acc"      : roll_acc,
            "roll_auc"      : roll_auc,
            "retrain"       : do_rt,
            "retrain_action": action,
            "market_trend"  : market_trend,
            "short_confirmed": short_confirm.get("confirmed", False),
        })

    sim_df = pd.DataFrame(sim_records)
    print(f"\n  ✅ 模擬完成，共 {len(sim_df)} 筆記錄")

    # ── 更新 state（就地寫回最新模型引用）──────────────────
    state["global_model"]   = global_model
    state["regime_models"]  = regime_models
    state["hmm"]            = hmm
    state["df_sim_result"]  = sim_df
    state["last_roll_acc"]  = roll_acc   # 供 predict_next_day() 計算倉位時使用

    return sim_df


# ============================================================
# Cell 8: 回測績效計算
# ============================================================
def compute_backtest_report(sim_df: pd.DataFrame) -> dict:
    """
    對應 Cell 8。接收 run_daily_simulation() 回傳的 sim_df，
    計算 Sharpe / 最大回撤 / 命中率 / regime 拆解等指標。
    """
    if sim_df is None or len(sim_df) == 0:
        return {}

    rets = sim_df["strat_ret"]
    equity = (1 + rets).cumprod()

    sharpe = (
        np.sqrt(252) * rets.mean() / rets.std()
        if rets.std() > 0 else 0.0
    )

    running_max = equity.cummax()
    drawdown    = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    traded = sim_df[sim_df["signal"] != 0]
    hit_rate = (
        traded["correct"].mean()
        if len(traded) > 0 and traded["correct"].notna().any() else 0.0
    )

    regime_breakdown = {}
    for regime in sim_df["regime"].unique():
        rdf = sim_df[sim_df["regime"] == regime]
        rtraded = rdf[rdf["signal"] != 0]
        regime_breakdown[regime] = {
            "n_days"   : len(rdf),
            "n_trades" : len(rtraded),
            "hit_rate" : (rtraded["correct"].mean()
                          if len(rtraded) > 0 and rtraded["correct"].notna().any() else 0.0),
            "total_ret": rdf["strat_ret"].sum(),
        }

    report = {
        "sharpe"           : float(sharpe),
        "max_drawdown"     : float(max_drawdown),
        "hit_rate"         : float(hit_rate),
        "total_return"     : float(equity.iloc[-1] - 1),
        "n_trades"         : int(len(traded)),
        "n_days"           : int(len(sim_df)),
        "regime_breakdown" : regime_breakdown,
    }

    print("\n  📊 回測報告")
    print(f"    Sharpe       : {report['sharpe']:.3f}")
    print(f"    最大回撤     : {report['max_drawdown']:.2%}")
    print(f"    命中率       : {report['hit_rate']:.2%}")
    print(f"    總報酬       : {report['total_return']:.2%}")
    print(f"    交易次數     : {report['n_trades']}")

    return report


# ============================================================
# Cell 7B: 模型儲存 / 載入（含 HL 持久化）
# ============================================================
def save_models(state: dict, out_dir: str = "models") -> None:
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

    # ── HL (高低位) 模型持久化，缺失不影響主流程 ──────────
    hl_models    = state.get("hl_models")
    hl_feat_cols = state.get("hl_feat_cols")
    if hl_models:
        joblib.dump(hl_models,    os.path.join(out_dir, "hl_models.pkl"))
        joblib.dump(hl_feat_cols, os.path.join(out_dir, "hl_feat_cols.pkl"))
        print(f"     hl_models.pkl, hl_feat_cols.pkl")

    print(f"  ✅ 模型已儲存至 {out_dir}/")
    print(f"     global_model.pkl, regime_models.pkl, "
          f"hmm_model.pkl, feat_cols.pkl")


def load_models(in_dir: str = "models") -> dict:
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

    # ── HL (高低位) 模型為附加功能，缺失不阻擋主流程 ──────────
    hl_path           = os.path.join(in_dir, "hl_models.pkl")
    hl_feat_cols_path = os.path.join(in_dir, "hl_feat_cols.pkl")
    if os.path.exists(hl_path) and os.path.exists(hl_feat_cols_path):
        state["hl_models"]    = joblib.load(hl_path)
        state["hl_feat_cols"] = joblib.load(hl_feat_cols_path)
        print(f"  ✅ HL模型已從 {in_dir}/ 載入")
    else:
        state["hl_models"]    = {}
        state["hl_feat_cols"] = []
        print(f"  ⚠️  HL模型檔案缺失，HL功能本次停用")

    print(f"  ✅ 模型已從 {in_dir}/ 載入")
    return state
