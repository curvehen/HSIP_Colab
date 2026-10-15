# ============================================================
# src/data_loader.py
# Merged from: 市場數據下載 cell (yfinance/akshare/FRED/北向南向資金)
#              + Intraday HSI fetch function (fetch_intraday_hsi)
#
# STRUCTURAL FIXES applied vs. original notebook:
#   - [Fix A] dl_fred() 原本用 `FRED_API_KEY == "<洩漏的硬編碼key>"`
#     判斷是否跳過。移除硬編碼 fallback 後該 key 已不存在，此判斷式
#     永遠為 False，導致空 key 仍會真實發送 request 並拿到
#     不清晰的 400 錯誤。改為 `if not FRED_API_KEY:` 做正確的
#     「未設定時優雅跳過」判斷。
#   - [Fix B] fetch_intraday_hsi() 原本依賴 notebook 全域變數
#     `df_all` 做 Source 3 fallback。Module 化後無全域可用，
#     改為顯式參數 `fallback_df`。
#   - [Fix C] yf_map / fred_map 不在此檔重複定義，直接從
#     src.config 匯入，避免兩處定義漂移。
# ============================================================
import time

import akshare as ak
import numpy as np
import pandas as pd
import requests
import yfinance as yf

from src.config import (
    TRAIN_START,
    DOWNLOAD_END,
    FRED_API_KEY,
    yf_map,
    fred_map,
)


# ============================================================
# 單一數據源下載工具函式
# ============================================================
def dl_yf(ticker: str, name: str) -> pd.Series:
    """下載 yfinance 數據，返回 Close 價格 Series"""
    try:
        tkr = yf.Ticker(ticker)
        df = tkr.history(start=TRAIN_START, end=DOWNLOAD_END, auto_adjust=True)
        if df is None or len(df) == 0:
            print(f"  ❌ {name}: 空數據")
            return pd.Series(dtype=float, name=name)
        df = df.reset_index()
        dc = [c for c in df.columns if "date" in str(c).lower()][0]
        df[dc] = pd.to_datetime(df[dc]).dt.tz_localize(None)
        cc = [c for c in df.columns if "close" in str(c).lower()][0]
        s = df.set_index(dc)[cc].rename(name)
        print(f"  ✅ {name:15s}: {len(s)} 行 | {s.index.min().date()} → {s.index.max().date()}")
        return s
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return pd.Series(dtype=float, name=name)


def dl_yf_ohlcv(ticker: str) -> pd.DataFrame:
    """下載恒指完整 OHLCV"""
    try:
        tkr = yf.Ticker(ticker)
        df = tkr.history(start=TRAIN_START, end=DOWNLOAD_END, auto_adjust=True)
        df = df.reset_index()
        dc = [c for c in df.columns if "date" in str(c).lower()][0]
        df[dc] = pd.to_datetime(df[dc]).dt.tz_localize(None)
        df = df.rename(columns={dc: "date"})
        df.columns = [c.lower() for c in df.columns]
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = np.nan
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  ❌ OHLCV: {e}")
        return pd.DataFrame()


def dl_akshare_index(symbol: str, name: str) -> pd.DataFrame:
    """下載 AkShare 港股指數日數據"""
    try:
        df = ak.stock_hk_index_daily_sina(symbol=symbol)
        df.columns = [c.lower().strip() for c in df.columns]
        date_col = None
        for c in df.columns:
            try:
                test = pd.to_datetime(df[c], errors="coerce")
                if test.notna().sum() > len(df) * 0.5 and test.dropna().max().year > 2000:
                    date_col = c
                    break
            except Exception:
                continue
        if not date_col:
            return pd.DataFrame()
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df[df["date"] >= TRAIN_START]

        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if "open" in cl or "開" in cl: col_map[c] = "open"
            if "high" in cl or "高" in cl: col_map[c] = "high"
            if "low" in cl or "低" in cl: col_map[c] = "low"
            if "close" in cl or "收" in cl: col_map[c] = "close"
            if "vol" in cl or "量" in cl: col_map[c] = "volume"
        df = df.rename(columns=col_map)
        for c in ["open", "high", "low", "close"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        print(f"  ✅ {name:15s}: {len(df)} 行 | {df['date'].min().date()} → {df['date'].max().date()}")
        return df
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return pd.DataFrame()


def dl_hsgt(symbol: str, col_name: str) -> pd.DataFrame:
    """下載北向/南向資金"""
    try:
        df = ak.stock_hsgt_hist_em(symbol=symbol)
        df.columns = [c.strip() for c in df.columns]
        date_col = None
        for c in df.columns:
            try:
                test = pd.to_datetime(df[c], errors="coerce")
                if test.notna().sum() > len(df) * 0.5 and test.dropna().max().year > 2000:
                    date_col = c
                    break
            except Exception:
                continue
        if not date_col:
            return pd.DataFrame()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.rename(columns={date_col: "date"})
        df = df.sort_values("date").reset_index(drop=True)
        df = df[df["date"] >= TRAIN_START]

        flow_col = None
        for c in df.columns:
            if c == "date":
                continue
            try:
                vals = pd.to_numeric(df[c], errors="coerce")
                if vals.notna().sum() > len(df) * 0.5:
                    flow_col = c
                    break
            except Exception:
                continue
        if flow_col:
            df[col_name] = pd.to_numeric(df[flow_col], errors="coerce")
        df = df[["date", col_name]].dropna()
        print(f"  ✅ {col_name:15s}: {len(df)} 行 | {df['date'].min().date()} → {df['date'].max().date()}")
        return df
    except Exception as e:
        print(f"  ❌ {col_name}: {e}")
        return pd.DataFrame()


def dl_fred(series_id: str, col_name: str) -> pd.DataFrame:
    """
    下載 FRED 宏觀數據。
    [Fix A] 原始判斷式比對硬編碼洩漏 key 字串，移除該 fallback 後已失效；
    改為正確判斷 key 是否為空。
    """
    try:
        if not FRED_API_KEY:
            return pd.DataFrame()
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": TRAIN_START,
            "sort_order": "asc",
        }
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        obs = data.get("observations", [])
        df = pd.DataFrame(obs)
        df["date"] = pd.to_datetime(df["date"])
        df[col_name] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["date", col_name]].dropna()
        print(f"  ✅ {col_name:15s}: {len(df)} 行 | {df['date'].min().date()} → {df['date'].max().date()}")
        return df
    except Exception as e:
        print(f"  ❌ {col_name}: {e}")
        return pd.DataFrame()


# ============================================================
# 主下載流程：對應原始 notebook 1~6 步驟，整合為單一可呼叫函式
# ============================================================
def download_all_data() -> pd.DataFrame:
    """
    下載恒指 OHLCV、外圍市場、北向/南向資金、FRED 宏觀數據，
    合併並計算衍生欄位，返回 master dataframe。
    對應 daily_pipeline.py 中的 `df_all = download_all_data()`。
    """
    print("\n[1/6] 恒指 OHLCV...")
    hsi_df = dl_akshare_index("HSI", "恒指")
    if len(hsi_df) == 0:
        print("  → 備用：使用 yfinance")
        hsi_df = dl_yf_ohlcv("^HSI")

    print("\n[2/6] 外圍市場...")
    ext_series = {}
    for name, ticker in yf_map.items():
        ext_series[name] = dl_yf(ticker, name)
        time.sleep(0.3)

    ext_df = pd.DataFrame(ext_series)
    ext_df.index.name = "date"
    ext_df = ext_df.reset_index()
    ext_df["date"] = pd.to_datetime(ext_df["date"]).dt.tz_localize(None)

    print("\n[3/6] 北向/南向資金...")
    nb_df = dl_hsgt("北向资金", "northbound_flow")
    time.sleep(1)
    sb_df = dl_hsgt("南向资金", "southbound_flow")

    print("\n[4/6] FRED 宏觀數據...")
    fred_dfs = {}
    for col, sid in fred_map.items():
        fred_dfs[col] = dl_fred(sid, col)
        time.sleep(0.3)

    print("\n[5/6] 合併數據...")
    master = hsi_df.copy()
    master = master.merge(ext_df, on="date", how="left")

    if len(nb_df) > 0:
        master = master.merge(nb_df, on="date", how="left")
    else:
        master["northbound_flow"] = np.nan

    if len(sb_df) > 0:
        master = master.merge(sb_df, on="date", how="left")
    else:
        master["southbound_flow"] = np.nan

    for col, df in fred_dfs.items():
        if len(df) > 0:
            master = master.merge(df, on="date", how="left")
            master[col] = master[col].fillna(method="ffill")

    print("\n[6/6] 清洗...")
    for col in ["sp500", "nasdaq", "csi300", "sse", "gold", "oil",
                "vix", "us10y", "dxy", "usdcny", "tencent"]:
        if col in master.columns:
            master[f"{col}_ret"] = master[col].pct_change()

    if "northbound_flow" in master.columns:
        nb = master["northbound_flow"]
        master["nb_ma5"] = nb.rolling(5).mean()
        master["nb_ma20"] = nb.rolling(20).mean()
        master["nb_zscore"] = (nb - nb.rolling(20).mean()) / (nb.rolling(20).std() + 1e-9)
        master["nb_inflow"] = (nb > 0).astype(int)
        master["nb_cum5"] = nb.rolling(5).sum()

    master = master.dropna(subset=["close"])
    num_cols = master.select_dtypes(include=[np.number]).columns
    master[num_cols] = master[num_cols].fillna(method="ffill", limit=5)
    master = master.reset_index(drop=True)

    return master


# ============================================================
# 日內數據抓取：Sina → yfinance → 昨收 fallback 三層
# [Fix B] Source 3 fallback 原本讀取全域 df_all，改為顯式參數。
# ============================================================
def fetch_intraday_hsi(fallback_df: pd.DataFrame = None):
    """
    下載恒指今日日內數據（開高低現價）
    來源: Sina Finance → yfinance → 昨日收市 (三層 fallback)

    Args:
        fallback_df: 若 Sina 及 yfinance 均失敗，從此 dataframe
            的最後一行取 close/open/high/low/volume 作為 fallback。
            對應原始腳本的全域 df_all，呼叫時應傳入
            daily_pipeline.py 中已下載好的 df_all。
    """
    SINA_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://finance.sina.com.cn/",
    }

    def _safe_float(val: str):
        try:
            v = val.strip().replace(",", "")
            f = float(v)
            return f if f != 0 else None
        except (ValueError, AttributeError):
            return None

    def _is_valid_hsi(val) -> bool:
        return val is not None and 5000 < val < 60000


    # ── Source 1: Sina Finance ──────────────────────────────
    try:
        url = "https://hq.sinajs.cn/list=rt_hkHSI"
        resp = requests.get(url, headers=SINA_HEADERS, timeout=10)
        resp.encoding = "gbk"
        raw = resp.text.strip()

        content = raw.split('"')[1] if '"' in raw else ""
        if not content:
            raise ValueError("Sina 返回空內容")

        fields = content.split(",")
        if len(fields) < 7:
            raise ValueError(f"Sina 欄位不足: {len(fields)}")

        name       = fields[0]
        open_p     = _safe_float(fields[2])
        prev_close = _safe_float(fields[3])
        current_p  = _safe_float(fields[6])
        high_p     = _safe_float(fields[4])
        low_p      = _safe_float(fields[5])

        if _is_valid_hsi(current_p) and _is_valid_hsi(open_p):
            print(f"  ✅ Sina Finance: 現價={current_p:.2f} 開={open_p:.2f} "
                  f"高={high_p:.2f} 低={low_p:.2f}")
            return {
                "source": "sina",
                "name": name,
                "current": current_p,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "prev_close": prev_close,
            }
        else:
            raise ValueError(f"Sina 數據不合理: current={current_p}, open={open_p}")

    except Exception as e:
        print(f"  ⚠️ Sina Finance 失敗: {e}")

    # ── Source 2: yfinance ───────────────────────────────────
    try:
        tkr = yf.Ticker("^HSI")
        intraday = tkr.history(period="1d", interval="1m")
        if intraday is not None and len(intraday) > 0:
            current_p = float(intraday["Close"].iloc[-1])
            open_p    = float(intraday["Open"].iloc[0])
            high_p    = float(intraday["High"].max())
            low_p     = float(intraday["Low"].min())

            info = tkr.fast_info if hasattr(tkr, "fast_info") else {}
            prev_close = info.get("previous_close") if info else None

            if _is_valid_hsi(current_p) and _is_valid_hsi(open_p):
                print(f"  ✅ yfinance (備用): 現價={current_p:.2f} 開={open_p:.2f} "
                      f"高={high_p:.2f} 低={low_p:.2f}")
                return {
                    "source": "yfinance",
                    "name": "HSI",
                    "current": current_p,
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "prev_close": prev_close,
                }
        raise ValueError("yfinance 日內數據為空或不合理")

    except Exception as e:
        print(f"  ⚠️ yfinance 日內數據失敗: {e}")

    # ── Source 3: fallback_df 最後一行（昨收數據）────────────
    # [Fix B] 原本讀取全域 df_all，現改為顯式傳入的 fallback_df。
    if fallback_df is not None and len(fallback_df) > 0:
        last_row = fallback_df.iloc[-1]
        print("  ⚠️ 兩個即時數據源均失敗，使用最後已知收市數據作為 fallback")
        return {
            "source": "fallback_last_close",
            "name": "HSI",
            "current": float(last_row.get("close", np.nan)),
            "open": float(last_row.get("open", np.nan)),
            "high": float(last_row.get("high", np.nan)),
            "low": float(last_row.get("low", np.nan)),
            "prev_close": float(last_row.get("close", np.nan)),
        }

    print("  ❌ 所有日內數據源均失敗，且無 fallback_df 可用")
    return None


# ============================================================
# [RECONSTRUCTED] HSI大盤版：盤中數據回填最後一行
#     原始 Colab update_last_row_with_intraday() 定義缺失，
#     依 fetch_intraday_hsi() 的回傳結構反推重建。
#     僅更新 OHLC/close/ret_1d，不重算技術指標（RSI/MACD等）；
#     技術指標重算由呼叫端在更新後重跑 build_features() 處理（方式A）。
# ============================================================
def update_last_row_with_intraday(last_row: pd.DataFrame, intraday: dict) -> pd.DataFrame:
    """
    用盤中數據更新 df_all 最後一行的 OHLC/close/ret_1d。
    與 stock_model.py 的同名函式邏輯一致（通用合併規則）。

    Args:
        last_row : df_all.iloc[[-1]].copy()，單行 DataFrame
        intraday : fetch_intraday_hsi() 的回傳 dict，可為 None

    Returns:
        更新後的單行 DataFrame（不重算技術指標，呼叫端需自行決定是否
        重跑 build_features()）
    """
    row = last_row.copy().reset_index(drop=True)
    if intraday is None:
        return row

    if intraday.get("open") is not None:
        row.at[0, "open"] = intraday["open"]
    if intraday.get("high") is not None:
        row.at[0, "high"] = intraday["high"]
    if intraday.get("low") is not None:
        row.at[0, "low"] = intraday["low"]
    if intraday.get("current") is not None:
        row.at[0, "close"] = intraday["current"]

    prev_close = intraday.get("prev_close")
    current    = intraday.get("current")
    if prev_close and current and prev_close > 0:
        row.at[0, "ret_1d"] = (current - prev_close) / prev_close

    return row

