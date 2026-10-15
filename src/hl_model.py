# ============================================================
# src/hl_model.py
# Merged from: Cell 17 (高低位預測模型, ENH#2: Open-Based Target)
#
# ⚠️ IMPORTANT: train_hl_model()'s body beyond the train/val split,
# and the loop that builds `hl_models` / `hl_metrics`, were NOT
# present in the source provided — that section was cut off before
# reaching this chat (confirmed: the next cell in your source was
# entirely commented-out debug code, not live logic).
#
# The functions marked [RECONSTRUCTED] below follow the exact same
# pattern as train_model_fixed() in Cell 5 and use LGBM_REG_PARAMS
# as already defined in your Cell 17. Please diff against your
# actual Colab notebook's real Cell 17/17B to confirm these match
# before relying on this module for production training.
# ============================================================
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.models import align_features

LGBM_REG_PARAMS = {
    "objective"        : "regression",
    "metric"           : "mae",
    "learning_rate"    : 0.02,
    "num_leaves"       : 48,
    "min_child_samples": 30,
    "feature_fraction" : 0.7,
    "bagging_fraction" : 0.8,
    "bagging_freq"     : 5,
    "lambda_l1"        : 0.3,
    "lambda_l2"        : 0.8,
    "verbose"          : -1,
    "n_estimators"     : 600,
    "random_state"     : 42,
}

HL_TARGET_COLS = [
    "next_high_pct", "next_low_pct", "next_range_pct",
    "next_high_abs", "next_low_abs",  "next_range_abs",
    "next_open_abs", "next_close_pct","next_gap_pct",
    "next_gap_up",   "next_gap_down",
    "next_high_pct_close_base",
    "next_low_pct_close_base",
]


def build_hl_targets(df: pd.DataFrame) -> pd.DataFrame:
    """ENH#2: next_open-based high/low targets. Verbatim from source."""
    df    = df.copy().sort_values("date").reset_index(drop=True)
    close = df["close"]

    has_ohlc = {"open", "high", "low", "close"}.issubset(df.columns)

    if has_ohlc:
        next_open  = df["open"].shift(-1)
        next_high  = df["high"].shift(-1)
        next_low   = df["low"].shift(-1)
        next_close = close.shift(-1)

        safe_open  = next_open.replace(0, np.nan)
        safe_close = close.replace(0, np.nan)

        df["next_open_abs"]  = next_open
        df["next_high_abs"]  = next_high
        df["next_low_abs"]   = next_low
        df["next_high_pct"]  = (next_high  - safe_open) / safe_open
        df["next_low_pct"]   = (next_low   - safe_open) / safe_open
        df["next_close_pct"] = (next_close - safe_open) / safe_open
        df["next_range_pct"] = (df["next_high_pct"] - df["next_low_pct"])
        df["next_range_abs"] = next_high - next_low

        df["next_gap_pct"]   = (next_open  - safe_close) / safe_close
        df["next_gap_up"]    = (df["next_gap_pct"] >  0.002).astype(int)
        df["next_gap_down"]  = (df["next_gap_pct"] < -0.002).astype(int)

        df["next_high_pct_close_base"] = (next_high - safe_close) / safe_close
        df["next_low_pct_close_base"]  = (next_low  - safe_close) / safe_close

        print("  ✅ ENH#2: 使用 next_open 做 HL target base")

    else:
        if "atr_14" in df.columns:
            atr = df["atr_14"]
        else:
            ret = close.pct_change()
            atr = ret.rolling(14).std() * close

        next_close = close.shift(-1)

        df["next_open_abs"]  = next_close
        df["next_high_abs"]  = next_close + atr * 0.5
        df["next_low_abs"]   = next_close - atr * 0.5
        df["next_high_pct"]  = (df["next_high_abs"] - next_close) / (next_close + 1e-9)
        df["next_low_pct"]   = (df["next_low_abs"]  - next_close) / (next_close + 1e-9)
        df["next_close_pct"] = next_close.pct_change().shift(-1)
        df["next_range_pct"] = (df["next_high_pct"] - df["next_low_pct"])
        df["next_range_abs"] = (df["next_high_abs"] - df["next_low_abs"])
        df["next_gap_pct"]   = pd.Series(0.0, index=df.index)
        df["next_gap_up"]    = 0
        df["next_gap_down"]  = 0
        df["next_high_pct_close_base"] = df["next_high_pct"]
        df["next_low_pct_close_base"]  = df["next_low_pct"]

        print("  ⚠️  ENH#2: 無 Open 欄位，ATR 估算 (精度有限)")

    return df


def build_hl_features(df: pd.DataFrame) -> pd.DataFrame:
    """Verbatim from source."""
    df    = df.copy()
    close = df["close"]

    if "high" in df.columns and "low" in df.columns:
        for w in [1, 3, 5, 10, 20]:
            df[f"range_pct_{w}d"] = (
                (df["high"].rolling(w).max() - df["low"].rolling(w).min())
                / (close + 1e-9)
            )
        df["today_high_pct"] = (df["high"] - close) / (close + 1e-9)
        df["today_low_pct"]  = (close - df["low"])  / (close + 1e-9)
        df["today_range"]    = ((df["high"] - df["low"]) / (close + 1e-9))

        for w in [5, 20]:
            ma = close.rolling(w).mean()
            df[f"high_vs_ma{w}"] = ((df["high"] - ma) / (ma + 1e-9))
            df[f"low_vs_ma{w}"]  = ((df["low"]  - ma) / (ma + 1e-9))

        df["high_ret_1d"] = df["high"].pct_change(1)
        df["high_ret_5d"] = df["high"].pct_change(5)
        df["low_ret_1d"]  = df["low"].pct_change(1)
        df["low_ret_5d"]  = df["low"].pct_change(5)

    if "next_gap_pct" in df.columns:
        df["gap_ma5"]     = df["next_gap_pct"].rolling(5).mean()
        df["gap_std5"]    = df["next_gap_pct"].rolling(5).std()
        df["gap_abs_ma5"] = df["next_gap_pct"].abs().rolling(5).mean()

    ret = close.pct_change()
    for w in [3, 5, 10, 20, 60]:
        df[f"vol_{w}d"]     = ret.rolling(w).std()
        df[f"vol_abs_{w}d"] = ret.abs().rolling(w).mean()

    df["vol_accel"] = (ret.rolling(5).std() / (ret.rolling(20).std() + 1e-9))

    if "vix" in df.columns:
        df["vix_level"]   = df["vix"]
        df["vix_chg"]     = df["vix"].diff()
        df["vix_chg_pct"] = df["vix"].pct_change()
        df["vix_ma5"]     = df["vix"].rolling(5).mean()
        df["vix_ratio"]   = (df["vix"] / (df["vix"].rolling(20).mean() + 1e-9))

    if "atr_14" in df.columns:
        df["atr_pct"]   = df["atr_14"] / (close + 1e-9)
        df["atr_ratio"] = (df["atr_14"] / (df["atr_14"].rolling(20).mean() + 1e-9))

    for col in ["sp500_ret", "nasdaq_ret", "csi300_ret"]:
        if col in df.columns:
            df[f"{col}_abs"]     = df[col].abs()
            df[f"{col}_abs_ma5"] = df[col].abs().rolling(5).mean()

    return df


def get_hl_feat_cols(df_hl: pd.DataFrame, classifier_feat_cols: list) -> tuple:
    """
    Verbatim logic from source (feature isolation between
    classifier feat_cols and HL-specific feat_cols).
    Returns (hl_feat_cols, hl_extra_cols).
    """
    from src.config import FEATURE_BLACKLIST

    hl_extra_cols = [
        c for c in df_hl.columns
        if c not in FEATURE_BLACKLIST
        and c not in HL_TARGET_COLS
        and c not in classifier_feat_cols
        and df_hl[c].dtype in [np.float64, np.int64, float, int]
        and df_hl[c].notna().mean() > 0.3
    ]
    hl_feat_cols = classifier_feat_cols + hl_extra_cols
    return hl_feat_cols, hl_extra_cols


# ============================================================
# [RECONSTRUCTED — see module docstring above]
# ============================================================
def train_hl_model(train_df: pd.DataFrame, target_col: str, hl_feat_cols: list):
    """
    Confirmed verbatim up to the train/val split. The fit/evaluate
    logic below is reconstructed from the LGBM_REG_PARAMS already
    defined in your source and the train_model_fixed() pattern in
    Cell 5 — NOT copied from a verbatim source line, since that
    section was missing from what was provided.

    Returns: (fitted_model, mae, r2) to match the 3-value return
    signature implied by `return None, 999.0, -999.0` in your
    original early-exit line.
    """
    from sklearn.metrics import mean_absolute_error, r2_score

    if len(train_df) < 120:
        return None, 999.0, -999.0

    vc    = int(len(train_df) * 0.85)
    X_tr  = align_features(train_df[hl_feat_cols].iloc[:vc], hl_feat_cols)
    y_tr  = train_df[target_col].iloc[:vc]
    X_val = align_features(train_df[hl_feat_cols].iloc[vc:], hl_feat_cols)
    y_val = train_df[target_col].iloc[vc:]

    try:
        model = lgb.LGBMRegressor(**LGBM_REG_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set  = [(X_val, y_val)],
            callbacks = [lgb.early_stopping(40, verbose=False),
                         lgb.log_evaluation(period=-1)]
        )

        pred = model.predict(X_val)
        mae  = mean_absolute_error(y_val, pred)
        r2   = r2_score(y_val, pred)
        return model, mae, r2
    except Exception as e:
        print(f"    ⚠️  HL訓練失敗 ({target_col}): {e}")
        return None, 999.0, -999.0



def train_all_hl_models(df_train_hl: pd.DataFrame, hl_feat_cols: list) -> tuple:
    """
    [RECONSTRUCTED] Loops over the 3 HL targets used by Cell 18
    (confirmed by its usage: hl_models["next_high_pct"], etc.)
    and builds the hl_models + hl_metrics dicts Cell 18 expects.
    """
    hl_models  = {}
    hl_metrics = {}

    for target_col in ["next_high_pct", "next_low_pct", "next_range_pct"]:
        model, mae, r2 = train_hl_model(df_train_hl, target_col, hl_feat_cols)
        if model is not None:
            hl_models[target_col]  = model
            hl_metrics[target_col] = {"mae": round(mae, 4), "r2": round(r2, 4)}
            print(f"  ✅ HL模型[{target_col}]: MAE={mae:.4f}  R²={r2:.4f}")
        else:
            print(f"  ⚠️  HL模型[{target_col}]: 訓練失敗（樣本不足）")

    return hl_models, hl_metrics
