# ============================================================
# src/features.py
# Merged from: Cell 4 (特徵工程)
# ============================================================
import pandas as pd
import numpy as np
from ta.momentum   import RSIIndicator
from ta.trend      import MACD
from ta.volatility import BollingerBands, AverageTrueRange

from src.config import FEATURE_BLACKLIST


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy().sort_values("date").reset_index(drop=True)
    close = df["close"]

    # ── 目標變數 ────────────────────────────────────────────
    df["next_ret"] = close.pct_change().shift(-1)
    df["target"]   = (df["next_ret"] > 0).astype(int)

    # ── 價格動能 ─────────────────────────────────────────────
    for w in [1,2,3,5,10,20,60]:
        df[f"ret_{w}d"] = close.pct_change(w)

    if "open" in df.columns:
        df["gap_ret"] = (df["open"] - close.shift(1)) / (close.shift(1) + 1e-9)

    if {"open","high","low"}.issubset(df.columns):
        body  = (df["open"] - close).abs()
        upper = df["high"] - df[["open","close"]].max(axis=1)
        lower = df[["open","close"]].min(axis=1) - df["low"]
        df["upper_wick"] = upper / (body + 1e-9)
        df["lower_wick"] = lower / (body + 1e-9)
        df["body_size"]  = body  / (close + 1e-9)

    for w in [5,10,20,60,120]:
        ma = close.rolling(w).mean()
        df[f"ma_dev_{w}"] = (close - ma) / (ma + 1e-9)

    df["ma5_20"]  = (close.rolling(5).mean()  > close.rolling(20).mean()).astype(int)
    df["ma20_60"] = (close.rolling(20).mean() > close.rolling(60).mean()).astype(int)

    # ── 成交量 ───────────────────────────────────────────────
    if "volume" in df.columns:
        vol = df["volume"]
        for w in [5,10,20]:
            df[f"vol_surge_{w}"] = vol / (vol.rolling(w).mean() + 1e-9)

    # ── 技術指標 ─────────────────────────────────────────────
    df["rsi_6"]       = RSIIndicator(close, window=6).rsi()
    df["rsi_14"]      = RSIIndicator(close, window=14).rsi()
    macd_obj          = MACD(close)
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_diff"]   = macd_obj.macd_diff()
    bb                = BollingerBands(close, window=20)
    df["bb_pos"]      = ((close - bb.bollinger_lband()) /
                         (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9))
    df["bb_width"]    = (bb.bollinger_hband() - bb.bollinger_lband()) / (close + 1e-9)
    if {"high","low"}.issubset(df.columns):
        atr            = AverageTrueRange(df["high"], df["low"], close, window=14)
        df["atr_14"]   = atr.average_true_range()
        df["atr_ratio"]= df["atr_14"] / (close + 1e-9)

    # ── 波動率 Regime ────────────────────────────────────────
    ret             = close.pct_change()
    df["vol_20"]    = ret.rolling(20).std()
    df["vol_60"]    = ret.rolling(60).std()
    df["vol_ratio"] = df["vol_20"] / (df["vol_60"] + 1e-9)
    df["high_vol"]  = (df["vol_20"] > df["vol_60"]).astype(int)
    df["trend_up"]  = (close.rolling(20).mean() > close.rolling(60).mean()).astype(int)

    # ── 外圍特徵 ─────────────────────────────────────────────
    for col in ["sp500","nasdaq","csi300","sse","gold","oil",
                "vix","us10y","dxy","usdcny","tencent"]:
        rc = f"{col}_ret"
        if rc in df.columns:
            df[f"{col}_lag1"] = df[rc].shift(1)
            df[f"{col}_lag2"] = df[rc].shift(2)
            df[f"{col}_ma5"]  = df[rc].rolling(5).mean()

    if "vix" in df.columns:
        df["vix_chg"]   = df["vix"].diff().shift(1)
        df["vix_high"]  = (df["vix"] > 25).astype(int)
        df["vix_spike"] = (df["vix"].pct_change() > 0.15).astype(int)

    if "us10y" in df.columns:
        df["us10y_chg"]  = df["us10y"].diff().shift(1)
        df["us10y_chg5"] = df["us10y"].diff(5).shift(1)

    if "dxy" in df.columns:
        df["dxy_chg"] = df["dxy"].pct_change().shift(1)

    # ── 北向資金特徵 ─────────────────────────────────────────
    for col in ["northbound_flow","nb_ma5","nb_ma20","nb_zscore",
                "nb_inflow","nb_cum5"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)

    if "northbound_flow" in df.columns:
        nb = df["northbound_flow"]
        df["nb_diverge"] = ((nb.shift(1) > 0) & (ret.shift(1) < 0)).astype(int)

    # ── 跨市場特徵 ───────────────────────────────────────────
    if "sp500_ret"  in df.columns:
        df["hsi_sp500_spread"] = ret.shift(1) - df["sp500_ret"].shift(1)
    if "csi300_ret" in df.columns:
        df["hsi_csi_spread"]   = ret.shift(1) - df["csi300_ret"].shift(1)

    # ── 日曆特徵 ─────────────────────────────────────────────
    d = df["date"]
    df["dow"]            = d.dt.dayofweek
    df["month"]          = d.dt.month
    df["is_month_end"]   = d.dt.is_month_end.astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_qtr_end"]     = d.dt.is_quarter_end.astype(int)
    df["month_sin"]      = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]      = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]        = np.sin(2 * np.pi * df["dow"] / 5)
    df["dow_cos"]        = np.cos(2 * np.pi * df["dow"] / 5)

    # 僅過濾技術指標暖機期的 NaN（開頭 ~26 行），
    # 不過濾 target/next_ret —— 這兩者對「最新一行」(尚無明日數據)
    # 必然是 NaN，但該行正是預測所需的最新資料，不應在此被丟棄。
    # target 為 NaN 的過濾責任交給 training.py 的訓練/模擬流程自行處理。
    df = df.dropna(subset=["rsi_14","macd","bb_pos"])
    df = df.reset_index(drop=True)
    return df


def get_feat_cols(df, top_n=45):
    candidates = [
        c for c in df.columns
        if c not in FEATURE_BLACKLIST
        and df[c].dtype in [np.float64, np.int64, float, int]
        and df[c].notna().sum() > len(df) * 0.5
    ]

    if len(candidates) <= top_n:
        return candidates

    import lightgbm as lgb

    _target = "target"
    if _target not in df.columns:
        return candidates[:top_n]

    _df = df[candidates + [_target]].dropna()
    if len(_df) < 100:
        return candidates[:top_n]

    _X = _df[candidates]
    _y = _df[_target]

    _selector = lgb.LGBMClassifier(
        n_estimators      = 100,
        num_leaves        = 20,
        min_child_samples = 50,
        feature_fraction  = 0.8,
        verbose           = -1,
        random_state      = 42,
    )
    _selector.fit(_X, _y)

    _imp = pd.Series(
        _selector.feature_importances_,
        index=candidates
    ).sort_values(ascending=False)

    top_feats = _imp.head(top_n).index.tolist()
    print(f"  [get_feat_cols] {len(candidates)} → {len(top_feats)} 個特徵")
    return top_feats
