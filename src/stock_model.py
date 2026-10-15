"""
src/stock_model.py
-------------------
個股預測模組（對應 Colab HSI_Colab.py Cell 20b/21/22/23/26/28）。

設計原則（與 hl_model.py 一致）：
  - 所有函式只接收明確參數，不碰觸 state dict。
  - state 的組裝/拆解交由 daily_pipeline.py 負責（本檔案目前尚未接入）。
  - 僅實作「個股獨立模型」路徑（每股一個 LightGBM），
    Colab Cell 25 的 Panel 共用模型路徑依共識捨棄不迁移。

已知待確認事項：
  - _update_last_row_with_intraday_stock() 標註 [RECONSTRUCTED]，
    因原始 update_last_row_with_intraday() 定義在 Colab 中找不到，
    此處依呼叫端用法反推實作，請核對後移除此警語。
"""

import os
import traceback

import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from ta.momentum   import RSIIndicator
from ta.trend      import MACD
from ta.volatility import BollingerBands, AverageTrueRange

from sklearn.metrics import accuracy_score, roc_auc_score

from src import config as cfg
from src.models import train_lgbm, align_features, safe_predict
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb


  
# ============================================================
# 常數：直接複用 config.py 既有定義，不重複定義
# ============================================================
STOCK_LIST        = cfg.STOCK_LIST
SECTOR_MAP        = cfg.SECTOR_MAP
STOCK_TRAIN_START = cfg.STOCK_TRAIN_START
STOCK_TRAIN_END   = cfg.TRAIN_END          # 同恆指模型一致，沿用 cfg.TRAIN_END
STOCK_SIM_START   = cfg.STOCK_SIM_START
STOCK_MIN_HISTORY = cfg.STOCK_MIN_HISTORY
STOCK_TOP_N       = cfg.STOCK_TOP_N

# 原 Colab 的 main_net/xlarge_net/large_net/retail_net 等10個欄位
# 依賴 AkShare 的 stock_individual_fund_flow()，該介面主要支援A股，
# 對港股支援度不穩定，故改用南向資金持股變化作為替代資金流特徵
# （評估結論：選項1，資料真實可取得，但欄位結構與原版不同）。
_FLOW_COLS = [
    "southbound_hold_chg_1d",
    "southbound_hold_chg_5d",
    "southbound_hold_pct",
]


_EXCLUDE_COLS = {
    "date", "target", "next_ret",
    "open", "high", "low", "close", "volume",
    "ticker", "name",
}


# ============================================================
# 1. 個股下載（含情境3盤中補數據）
# ============================================================
def download_stock(
    ticker: str,
    name: str,
    download_end: str,
    scenario: int,
    today,
) -> pd.DataFrame | None:
    """
    下載單隻股票日線資料，情境3（盤中）時自動補入今日盤中數據。

    Args:
        ticker       : 股票代碼，如 "0700.HK"
        name         : 股票名稱（中文）
        download_end : 下載截止日字串 (today+2，確保包含今日)
        scenario     : 1~4，對應 config.get_prediction_context() 的 scenario
        today        : date 物件，今天的日期（HKT）
    """
    try:
        tkr = yf.Ticker(ticker)
        df = tkr.history(
            start=STOCK_TRAIN_START,
            end=download_end,
            auto_adjust=True,
            timeout=30,
        )
    except Exception as e:
        print(f"  ❌ {name} ({ticker}) 下載失敗: {e}")
        return None

    if df is None or len(df) < STOCK_MIN_HISTORY:
        print(f"  ⚠️  {name} ({ticker}): 數據不足（{0 if df is None else len(df)} 行）")
        return None

    df = df.reset_index()

    date_col = next((c for c in df.columns if "date" in str(c).lower()), None)
    if date_col is None:
        print(f"  ⚠️  {name} 無日期欄位")
        return None

    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    df = df.rename(columns={date_col: "date"})

    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "open" in cl: col_map[c] = "open"
        elif "high" in cl: col_map[c] = "high"
        elif "low" in cl: col_map[c] = "low"
        elif "close" in cl: col_map[c] = "close"
        elif "volume" in cl: col_map[c] = "volume"
    df = df.rename(columns=col_map)

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)

    # ── 情境3：補入今日盤中數據 ─────────────────────────────
    if scenario == 3:
        latest_in_df = df["date"].max().date()
        if latest_in_df < today:
            try:
                intra = yf.download(
                    ticker, period="1d", interval="5m",
                    progress=False, auto_adjust=True,
                )
                if len(intra) > 0:
                    intra.index = pd.to_datetime(intra.index).tz_convert(
                        "Asia/Hong_Kong"
                    ).tz_localize(None)
                    today_rows = intra[intra.index.date == today]
                    if len(today_rows) > 0:
                        cols = today_rows.columns.str.lower()
                        today_row = pd.DataFrame([{
                            "date": pd.Timestamp(today),
                            "open": float(today_rows.iloc[0, cols.get_loc("open")]) if "open" in cols else np.nan,
                            "high": float(today_rows.iloc[:, cols.get_loc("high")].max()) if "high" in cols else np.nan,
                            "low": float(today_rows.iloc[:, cols.get_loc("low")].min()) if "low" in cols else np.nan,
                            "close": float(today_rows.iloc[-1, cols.get_loc("close")]) if "close" in cols else np.nan,
                            "volume": float(today_rows.iloc[:, cols.get_loc("volume")].sum()) if "volume" in cols else 0,
                        }])
                        df = pd.concat([df, today_row], ignore_index=True)
                        df = df.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)
                        print(f"  📡 {name}: 補入今日盤中數據（close={today_row['close'].iloc[0]:,.2f}）")
            except Exception as ex:
                print(f"  ⚠️  {name} 盤中數據補充失敗: {ex}")

    df["ticker"] = ticker
    df["name"] = name
    df["sector"] = SECTOR_MAP.get(ticker, "其他")
    return df

# ============================================================
# 1b. 南向資金持股數據（取代原 _FLOW_COLS 的主力/大單拆分，
#     因 AkShare 無法穩定提供港股個股的主力資金流拆分）
# ============================================================
def fetch_southbound_holding(ticker: str) -> pd.DataFrame | None:
    """
    取得單一港股的南向資金（港股通）每日持股變化。
    ticker 格式如 "0700.HK" -> 轉換為 AkShare 要求的 "00700"
    """
    import akshare as ak

    code = ticker.replace(".HK", "").zfill(5)
    try:
        df = ak.stock_hsgt_individual_em(symbol=code)
    except Exception as e:
        print(f"  ⚠️  {ticker} 南向資金數據取得失敗: {e}")
        return None

    if df is None or len(df) == 0:
        return None

    df = df.rename(columns={
        "持股日期": "date",
        "持股数量": "hold_shares",
        "持股市值": "hold_value",
        "持股数量占流通股比": "southbound_hold_pct",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["southbound_hold_chg_1d"] = df["hold_shares"].diff()
    df["southbound_hold_chg_5d"] = df["hold_shares"].diff(5)

    return df[["date", "southbound_hold_chg_1d", "southbound_hold_chg_5d", "southbound_hold_pct"]]

# ============================================================
# 2. [RECONSTRUCTED] 情境1盤中補數據（單行 DataFrame 版）
#    原始 update_last_row_with_intraday() 定義缺失，依呼叫端
#    用法反推：只寫入 intraday 衍生欄位，不動 close/volume/date/ret_1d
#    （這些由呼叫端自行覆寫）。請核對此假設是否正確。
# ============================================================
def _update_last_row_with_intraday_stock(
    last_row: pd.DataFrame, intra_dict: dict
) -> pd.DataFrame:
    last_row = last_row.copy()
    for key in ["open", "high", "low"]:
        if key in intra_dict:
            last_row.at[0, key] = intra_dict[key]
    for key in [
        "intraday_ret", "intraday_range",
        "high_vs_open", "low_vs_open", "ret_vs_prev",
    ]:
        if key in intra_dict:
            last_row[key] = intra_dict[key]
    return last_row


# ============================================================
# 3. 個股特徵工程
# ============================================================
def build_stock_features(
    df: pd.DataFrame, hsi_df: pd.DataFrame, ticker: str = ""
) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df["close"]

    df["next_ret"] = close.pct_change().shift(-1)
    df["target"] = (df["next_ret"] > 0).astype(int)

    for w in [1, 2, 3, 5, 10, 20, 60]:
        df[f"ret_{w}d"] = close.pct_change(w)

    if "open" in df.columns:
        df["gap_ret"] = (df["open"] - close.shift(1)) / (close.shift(1) + 1e-9)

    for w in [5, 10, 20, 60]:
        ma = close.rolling(w, min_periods=w // 2).mean()
        df[f"ma_dev_{w}"] = (close - ma) / (ma + 1e-9)

    df["ma5_20"] = (
        close.rolling(5, min_periods=3).mean() > close.rolling(20, min_periods=10).mean()
    ).astype(int)
    df["ma20_60"] = (
        close.rolling(20, min_periods=10).mean() > close.rolling(60, min_periods=30).mean()
    ).astype(int)

    if {"open", "high", "low"}.issubset(df.columns):
        body = (df["open"] - close).abs()
        upper = df["high"] - df[["open", "close"]].max(axis=1)
        lower = df[["open", "close"]].min(axis=1) - df["low"]
        df["upper_wick"] = upper / (body + 1e-9)
        df["lower_wick"] = lower / (body + 1e-9)
        df["body_size"] = body / (close + 1e-9)

    if "volume" in df.columns:
        vol = df["volume"]
        for w in [5, 10, 20]:
            df[f"vol_surge_{w}"] = vol / (vol.rolling(w, min_periods=w // 2).mean() + 1e-9)
        df["vol_trend"] = (
            vol.rolling(5, min_periods=3).mean() / (vol.rolling(20, min_periods=10).mean() + 1e-9)
        )
        df["vol_momentum"] = vol.pct_change(5)
        df["vol_price_corr"] = vol.rolling(10, min_periods=5).corr(close.pct_change().abs())

    df["rsi_6"] = RSIIndicator(close, window=6).rsi()
    df["rsi_14"] = RSIIndicator(close, window=14).rsi()
    df["rsi_slope"] = df["rsi_14"].diff(3)

    macd_obj = MACD(close)
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_diff"] = macd_obj.macd_diff()

    bb = BollingerBands(close, window=20, window_dev=2)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (close + 1e-9)
    df["bb_pct"] = bb.bollinger_pband()

    if {"high", "low"}.issubset(df.columns):
        atr = AverageTrueRange(df["high"], df["low"], close, window=14)
        df["atr_14"] = atr.average_true_range()
        df["atr_pct"] = df["atr_14"] / (close + 1e-9)

    ret1 = close.pct_change()
    df["vol_5"] = ret1.rolling(5, min_periods=3).std()
    df["vol_20"] = ret1.rolling(20, min_periods=10).std()
    df["vol_60"] = ret1.rolling(60, min_periods=30).std()
    df["vol_ratio"] = df["vol_5"] / (df["vol_20"] + 1e-9)
    df["vol_regime"] = pd.qcut(
        df["vol_20"].rank(method="first"), 3, labels=["low", "mid", "high"]
    ).astype(str)

    # ── 與恆指的相對強弱 / beta（個股特有，HSI本身無此特徵）──
    if hsi_df is not None and len(hsi_df) > 0:
        hsi_ret = (
            hsi_df.set_index("date")["close"].pct_change()
            if "date" in hsi_df.columns
            else hsi_df["close"].pct_change()
        )
        merged = df[["date"]].merge(
            hsi_ret.rename("hsi_ret").reset_index(), on="date", how="left"
        )
        df["hsi_ret"] = merged["hsi_ret"].values
        df["rel_strength_5"] = close.pct_change(5) - df["hsi_ret"].rolling(5, min_periods=3).sum()
        df["rel_strength_20"] = close.pct_change(20) - df["hsi_ret"].rolling(20, min_periods=10).sum()
        roll_cov = ret1.rolling(60, min_periods=30).cov(df["hsi_ret"])
        roll_var = df["hsi_ret"].rolling(60, min_periods=30).var()
        df["beta_60"] = roll_cov / (roll_var + 1e-9)
    else:
        df["rel_strength_5"] = np.nan
        df["rel_strength_20"] = np.nan
        df["beta_60"] = np.nan

    # ── 資金流欄位（若上游已合併 Stock Connect flow 資料則保留，否則補 NaN）──
    # ── 資金流欄位：南向資金持股變化（若呼叫端已合併則保留，否則補 NaN）──
    # 呼叫端應先呼叫 fetch_southbound_holding() 並 merge 到 df 再傳入本函式，
    # 本函式僅做欄位存在性保護，不主動下載（避免重複網路請求）。
    for c in _FLOW_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # ── 日曆特徵 ──
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    return df


# ============================================================
# 4. 特徵欄位篩選
# ============================================================
def get_stock_feat_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in _EXCLUDE_COLS:
            continue
        if c == "sector" or c == "vol_regime":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


# ============================================================
# 5. 多種子集成預測（輔助函式）
# ============================================================
def _ensemble_predict_proba(models: list, X: pd.DataFrame) -> np.ndarray:
    """對同一檔股票的多個 seed 模型取平均機率。"""
    if not models:
        return np.full(len(X), 0.5)
    probs = []
    for m in models:
        p = safe_predict(m, X)
        if p is not None:
            probs.append(p)
    if not probs:
        return np.full(len(X), 0.5)
    return np.mean(probs, axis=0)

# ============================================================
# 5b. 自訂 seed 版 LightGBM 訓練（複製 models.train_lgbm() 邏輯，
#     不修改共用檔案，僅本模組使用）
# ============================================================
def _train_lgbm_seeded(X_tr, y_tr, X_val, y_val, seed: int):
    """
    與 models.train_lgbm() 邏輯一致，額外支援自訂 random_state/bagging_seed，
    用於個股多 seed 集成訓練。
    """
    try:
        params = {**cfg.LGBM_PARAMS, "random_state": seed, "bagging_seed": seed}
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set  = [(X_val, y_val)],
            callbacks = [lgb.early_stopping(40, verbose=False),
                         lgb.log_evaluation(period=-1)]
        )
        cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        cal.fit(X_val, y_val)
        return cal
    except Exception as e:
        print(f"    ⚠️  seed={seed} 訓練失敗: {e}")
        return None

# ============================================================
# 6. 個股獨立模型訓練（Cell 22/23 對應，每股一個模型）
# ============================================================
def train_stock_models(
    stock_feat_data: dict[str, pd.DataFrame],
    n_seeds: int = 3,
) -> tuple[dict, dict, dict]:
    """
    對 stock_feat_data 中每檔股票各自訓練一個（或多 seed 集成）LightGBM 分類模型。

    Returns:
        stock_models        : {ticker: [model_seed1, model_seed2, ...]}
        stock_feat_cols_per  : {ticker: [feat_col, ...]}
        stock_train_scores   : {ticker: {"auc": float, "acc": float, "n_train": int}}
    """
    stock_models: dict[str, list] = {}
    stock_feat_cols_per: dict[str, list] = {}
    stock_train_scores: dict[str, dict] = {}

    for ticker, df in stock_feat_data.items():
        df_clean = df.dropna(subset=["target"]).copy()
        if len(df_clean) < STOCK_MIN_HISTORY:
            print(f"  ⚠️  {ticker}: 樣本不足（{len(df_clean)}），跳過訓練")
            continue

        feat_cols = get_stock_feat_cols(df_clean)
        train_df = df_clean[df_clean["date"] < pd.Timestamp(STOCK_TRAIN_END)]
        if len(train_df) < 120:
            print(f"  ⚠️  {ticker}: 訓練集不足（{len(train_df)}），跳過訓練")
            continue

        vc = int(len(train_df) * 0.85)
        X_tr  = align_features(train_df.iloc[:vc][feat_cols], feat_cols)
        y_tr  = train_df.iloc[:vc]["target"]
        X_val = align_features(train_df.iloc[vc:][feat_cols], feat_cols)
        y_val = train_df.iloc[vc:]["target"]

        seed_models = []
        try:
            for seed in range(n_seeds):
                model = _train_lgbm_seeded(X_tr, y_tr, X_val, y_val, seed)
                if model is not None:
                    seed_models.append(model)

            if not seed_models:
                print(f"  ❌ {ticker}: 全部 seed 訓練失敗")
                continue

            val_X = X_val
            val_y = y_val
            val_prob = _ensemble_predict_proba(seed_models, val_X)

            auc = roc_auc_score(val_y, val_prob) if val_y.nunique() > 1 else float("nan")
            acc = accuracy_score(val_y, (val_prob > 0.5).astype(int))

            stock_models[ticker] = seed_models
            stock_feat_cols_per[ticker] = feat_cols
            stock_train_scores[ticker] = {
                "auc": auc, "acc": acc, "n_train": len(train_df),
            }
            print(f"  ✅ {ticker}: AUC={auc:.4f}  ACC={acc:.4f}  (n={len(train_df)})")

        except Exception as e:
            print(f"  ❌ {ticker} 訓練失敗: {e}")
            traceback.print_exc()
            continue

    return stock_models, stock_feat_cols_per, stock_train_scores


# ============================================================
# 7. 模型持久化（每股一檔，依共識採用）
# ============================================================
def save_stock_models(
    stock_models: dict, stock_feat_cols_per: dict,
    out_dir: str = os.path.join("models", "stocks"),
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for ticker, models in stock_models.items():
        safe_name = ticker.replace(".", "_")
        payload = {
            "models": models,
            "feat_cols": stock_feat_cols_per.get(ticker, []),
        }
        joblib.dump(payload, os.path.join(out_dir, f"{safe_name}.pkl"))
        saved += 1
    print(f"  ✅ {saved} 檔個股模型已儲存至 {out_dir}/")


def load_stock_models(
    tickers: list[str], in_dir: str = os.path.join("models", "stocks"),
) -> tuple[dict, dict]:
    stock_models: dict[str, list] = {}
    stock_feat_cols_per: dict[str, list] = {}
    loaded = 0
    for ticker in tickers:
        safe_name = ticker.replace(".", "_")
        path = os.path.join(in_dir, f"{safe_name}.pkl")
        if os.path.exists(path):
            payload = joblib.load(path)
            stock_models[ticker] = payload["models"]
            stock_feat_cols_per[ticker] = payload["feat_cols"]
            loaded += 1
    print(f"  ✅ {loaded}/{len(tickers)} 檔個股模型已從 {in_dir}/ 載入")
    return stock_models, stock_feat_cols_per


# ============================================================
# 8. 每日個股更新 — 只保留排行榜部分
#    恆指大方向（hsi_direction / hsi_confidence）由呼叫端
#    （daily_pipeline.py）算好傳入，此函式不重複推論
#    global_model / regime_models（依共識，問題3）。
# ============================================================
def daily_stock_update(
    stock_feat_data: dict[str, pd.DataFrame],
    stock_models: dict[str, list],
    stock_feat_cols_per: dict[str, list],
    hsi_direction: str,
    hsi_confidence: float,
    top_n: int = STOCK_TOP_N,
) -> pd.DataFrame:
    """
    產出個股排行榜（多/空排名），不涉及恆指大方向推論。

    Args:
        stock_feat_data      : {ticker: df_with_features}，每檔股票已含最新一行特徵
        stock_models         : {ticker: [model_seed1, ...]}
        stock_feat_cols_per   : {ticker: [feat_col, ...]}
        hsi_direction        : 由 daily_pipeline.py 算好的恆指方向 ("bull"/"bear"/"neutral")
        hsi_confidence       : 由 daily_pipeline.py 算好的恆指信心值
        top_n                : 排行榜取前 N 檔

    Returns:
        DataFrame，欄位: ticker, name, sector, prob, signal, hsi_direction, hsi_confidence
    """
    rows = []
    for ticker, df in stock_feat_data.items():
        if ticker not in stock_models or df is None or len(df) == 0:
            continue

        feat_cols = stock_feat_cols_per.get(ticker, [])
        if not feat_cols:
            continue

        last_row = df.iloc[[-1]].copy()
        X = align_features(last_row[feat_cols], feat_cols)
        prob = _ensemble_predict_proba(stock_models[ticker], X)[0]

        rows.append({
            "ticker": ticker,
            "name": last_row["name"].iloc[0] if "name" in last_row.columns else ticker,
            "sector": last_row["sector"].iloc[0] if "sector" in last_row.columns else SECTOR_MAP.get(ticker, "其他"),
            "prob": prob,
            "signal": "LONG" if prob > 0.55 else ("SHORT" if prob < 0.45 else "NEUTRAL"),
            "hsi_direction": hsi_direction,
            "hsi_confidence": hsi_confidence,
        })

    if not rows:
        print("  ⚠️  沒有可用的個股預測結果")
        return pd.DataFrame(columns=[
            "ticker", "name", "sector", "prob", "signal",
            "hsi_direction", "hsi_confidence",
        ])

    board = pd.DataFrame(rows).sort_values("prob", ascending=False).reset_index(drop=True)

    print(f"\n  📊 個股排行榜（恆指方向: {hsi_direction}, 信心: {hsi_confidence:.3f}）")
    print(f"  多頭 Top {min(top_n, len(board))}:")
    print(board.head(top_n)[["ticker", "name", "prob", "signal"]].to_string(index=False))
    print(f"\n  空頭 Top {min(top_n, len(board))}:")
    print(board.tail(top_n).sort_values("prob")[["ticker", "name", "prob", "signal"]].to_string(index=False))

    return board
