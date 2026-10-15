# ============================================================
# src/data_loader.py
# Merged from: Cell 3 (data download), Cell 3B (intraday fetch)
# ============================================================
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import akshare as ak

from src.config import (
    TRAIN_START, DOWNLOAD_END, FRED_API_KEY, CTX
)

print("=" * 60)
print("  📥 下載全部數據")
print("=" * 60)


# ── 工具函數 ─────────────────────────────────────────────────
def dl_yf(ticker, name):
    """下載 yfinance 數據，返回 Close 價格 Series"""
    try:
        tkr = yf.Ticker(ticker)
        df  = tkr.history(start=TRAIN_START, end=DOWNLOAD_END, auto_adjust=True)
        if df is None or len(df) == 0:
            print(f"  ❌ {name}: 空數據")
            return pd.Series(dtype=float, name=name)
        df  = df.reset_index()
        dc  = [c for c in df.columns if "date" in str(c).lower()][0]
        df[dc] = pd.to_datetime(df[dc]).dt.tz_localize(None)
        cc  = [c for c in df.columns if "close" in str(c).lower()][0]
        s   = df.set_index(dc)[cc].rename(name)
        print(f"  ✅ {name:15s}: {len(s)} 行 | "
              f"{s.index.min().date()} → {s.index.max().date()}")
        return s
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return pd.Series(dtype=float, name=name)


def dl_yf_ohlcv(ticker):
    """下載恆指完整 OHLCV"""
    try:
        tkr = yf.Ticker(ticker)
        df  = tkr.history(start=TRAIN_START, end=DOWNLOAD_END, auto_adjust=True)
        df  = df.reset_index()
        dc  = [c for c in df.columns if "date" in str(c).lower()][0]
        df[dc] = pd.to_datetime(df[dc]).dt.tz_localize(None)
        df  = df.rename(columns={dc: "date"})
        df.columns = [c.lower() for c in df.columns]
        for col in ["open","high","low","close","volume"]:
            if col not in df.columns:
                df[col] = np.nan
        df = df[["date","open","high","low","close","volume"]]
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  ❌ OHLCV: {e}")
        return pd.DataFrame()


def dl_akshare_index(symbol, name):
    """下載 AkShare 港股指數日數據"""
    try:
        df = ak.stock_hk_index_daily_sina(symbol=symbol)
        df.columns = [c.lower().strip() for c in df.columns]
        date_col = None
        for c in df.columns:
            try:
                test = pd.to_datetime(df[c], errors="coerce")
                if (test.notna().sum() > len(df)*0.5 and
                        test.dropna().max().year > 2000):
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
            if "open"  in cl or "開" in cl: col_map[c] = "open"
            if "high"  in cl or "高" in cl: col_map[c] = "high"
            if "low"   in cl or "低" in cl: col_map[c] = "low"
            if "close" in cl or "收" in cl: col_map[c] = "close"
            if "vol"   in cl or "量" in cl: col_map[c] = "volume"
        df = df.rename(columns=col_map)
        for c in ["open","high","low","close"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        print(f"  ✅ {name:15s}: {len(df)} 行 | "
              f"{df['date'].min().date()} → {df['date'].max().date()}")
        return df
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return pd.DataFrame()


def dl_hsgt(symbol, col_name):
    """下載北向/南向資金"""
    try:
        df = ak.stock_hsgt_hist_em(symbol=symbol)
        df.columns = [c.strip() for c in df.columns]
        date_col = None
        for c in df.columns:
            try:
                test = pd.to_datetime(df[c], errors="coerce")
                if (test.notna().sum() > len(df)*0.5 and
                        test.dropna().max().year > 2000):
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
                if vals.notna().sum() > len(df)*0.5:
                    flow_col = c
                    break
            except Exception:
                continue
        if flow_col:
            df[col_name] = pd.to_numeric(df[flow_col], errors="coerce")
        df = df[["date", col_name]].dropna()
        print(f"  ✅ {col_name:15s}: {len(df)} 行 | "
              f"{df['date'].min().date()} → {df['date'].max().date()}")
        return df
    except Exception as e:
        print(f"  ❌ {col_name}: {e}")
        return pd.DataFrame()


def dl_fred(series_id, col_name):
    """下載 FRED 宏觀數據"""
    try:
        # [FIX] 原版比對硬編碼 fallback key 字串來判斷是否跳過；
        # 現在 FRED_API_KEY 無效時直接是 None，改為判斷 None/空值
        if not FRED_API_KEY:
            return pd.DataFrame()
        url    = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id"        : series_id,
            "api_key"          : FRED_API_KEY,
            "file_type"        : "json",
            "observation_start": TRAIN_START,
            "sort_order"       : "asc",
        }
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        obs  = data.get("observations", [])
        df   = pd.DataFrame(obs)
        df["date"]     = pd.to_datetime(df["date"])
        df[col_name]   = pd.to_numeric(df["value"], errors="coerce")
        df = df[["date", col_name]].dropna()
        print(f"  ✅ {col_name:15s}: {len(df)} 行 | "
              f"{df['date'].min().date()} → {df['date'].max().date()}")
        return df
    except Exception as e:
        print(f"  ❌ {col_name}: {e}")
        return pd.DataFrame()


def download_all_data():
    """
    執行完整數據下載流程，返回 master DataFrame。
    對應原始 Cell 3 的主執行邏輯。
    """
    print("\n[1/6] 恆指 OHLCV...")
    hsi_df = dl_akshare_index("HSI", "恆指")
    if len(hsi_df) == 0:
        print("  → 備用：使用 yfinance")
        hsi_df = dl_yf_ohlcv("^HSI")

    print("\n[2/6] 外圍市場...")
    yf_map = {
        "sp500"  : "^GSPC",
        "nasdaq" : "^IXIC",
        "vix"    : "^VIX",
        "us10y"  : "^TNX",
        "us2y"   : "^IRX",
        "dxy"    : "DX-Y.NYB",
        "csi300" : "000300.SS",
        "sse"    : "000001.SS",
        "gold"   : "GC=F",
        "oil"    : "CL=F",
        "copper" : "HG=F",
        "usdcny" : "CNY=X",
        "tencent": "0700.HK",
        "hsbc"   : "0005.HK",
        "alibaba": "9988.HK",
        "meituan": "3690.HK",
    }
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
    fred_map = {
        "fed_rate"   : "FEDFUNDS",
        "us_cpi"     : "CPIAUCSL",
        "us_core_cpi": "CPILFESL",
        "nonfarm"    : "PAYEMS",
        "yield_curve": "T10Y2Y",
        "jobless"    : "ICSA",
    }
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
    for col in ["sp500","nasdaq","csi300","sse","gold","oil",
                "vix","us10y","dxy","usdcny","tencent"]:
        if col in master.columns:
            master[f"{col}_ret"] = master[col].pct_change()

    if "northbound_flow" in master.columns:
        nb = master["northbound_flow"]
        master["nb_ma5"]    = nb.rolling(5).mean()
        master["nb_ma20"]   = nb.rolling(20).mean()
        master["nb_zscore"] = (nb - nb.rolling(20).mean()) / (nb.rolling(20).std() + 1e-9)
        master["nb_inflow"] = (nb > 0).astype(int)
        master["nb_cum5"]   = nb.rolling(5).sum()

    master = master.dropna(subset=["close"])
    num_cols = master.select_dtypes(include=[np.number]).columns
    master[num_cols] = master[num_cols].fillna(method="ffill", limit=5)
    master = master.reset_index(drop=True)

    print(f"\n✅ 數據整合完成")
    print(f"   總行數: {len(master)}")
    print(f"   日期:   {master['date'].min().date()} → {master['date'].max().date()}")
    print(f"   欄位數: {len(master.columns)}")
    print("✅ data_loader.py: download_all_data() 完成")

    return master


# ============================================================
# Cell 3B: 日內實時數據下載（情境3專用）
# ============================================================
def fetch_intraday_hsi():
    """
    下載恒指今日日內數據（開高低現價）
    來源: Sina Finance → yfinance → 昨日收市 (三層 fallback)
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
        url  = "https://hq.sinajs.cn/list=rt_hkHSI"
        resp = requests.get(url, headers=SINA_HEADERS, timeout=10)
        resp.encoding = "gbk"
        raw  = resp.text.strip()

        content = raw.split('"')[1] if '"' in raw else ""
        if not content:
            raise ValueError("Sina 返回空內容")

        fields = content.split(",")
        if len(fields) < 6:
            raise ValueError(f"欄位不足: {len(fields)}")

        name   = fields[0].strip()
        offset = None

        for i in range(1, min(5, len(fields))):
            v = _safe_float(fields[i])
            if _is_valid_hsi(v):
                offset = i
                break

        if offset is None:
            raise ValueError(f"找不到合理 HSI 數值，fields={fields[:8]}")

        open_p    = _safe_float(fields[offset])
        prev_c    = _safe_float(fields[offset + 1]) if offset + 1 < len(fields) else None
        current_p = _safe_float(fields[offset + 2]) if offset + 2 < len(fields) else None
        high_p    = _safe_float(fields[offset + 3]) if offset + 3 < len(fields) else None
        low_p     = _safe_float(fields[offset + 4]) if offset + 4 < len(fields) else None
        volume_p  = _safe_float(fields[offset + 9]) if offset + 9 < len(fields) else None

        if not _is_valid_hsi(current_p): current_p = open_p
        if not _is_valid_hsi(high_p):   high_p    = current_p
        if not _is_valid_hsi(low_p):    low_p     = current_p

        for label, val in [("open", open_p), ("current", current_p),
                            ("high", high_p), ("low", low_p)]:
            if not _is_valid_hsi(val):
                raise ValueError(f"{label}={val} 無效，offset={offset}")

        result = {
            "name"      : name,
            "open"      : open_p,
            "prev_close": prev_c,
            "current"   : current_p,
            "high"      : high_p,
            "low"       : low_p,
            "volume"    : volume_p,
            "timestamp" : (fields[-2].strip() if len(fields) > 2 else ""),
        }

        if result["open"] and result["current"]:
            result["intraday_ret"] = (result["current"] - result["open"]) / result["open"]
        if result["high"] and result["low"] and result["open"]:
            result["intraday_range"] = (result["high"] - result["low"]) / result["open"]
            result["high_vs_open"]   = (result["high"] - result["open"]) / result["open"]
            result["low_vs_open"]    = (result["low"] - result["open"]) / result["open"]
        if result["current"] and result["prev_close"]:
            result["ret_vs_prev"] = (result["current"] - result["prev_close"]) / result["prev_close"]

        print(f"  ✅ Intraday HSI (Sina): {name} "
              f"現價={current_p:,.0f} 開={open_p:,.0f} "
              f"高={high_p:,.0f} 低={low_p:,.0f}")
        return result

    except Exception as e:
        print(f"  ⚠️  Sina intraday 失敗: {str(e)[:80]}")

    # ── Source 2: yfinance 1min fallback ────────────────────
    try:
        ticker = yf.Ticker("^HSI")
        hist   = ticker.history(period="1d", interval="1m")

        if len(hist) > 0:
            open_p    = float(hist.iloc[0]["Open"])
            current_p = float(hist.iloc[-1]["Close"])
            high_p    = float(hist["High"].max())
            low_p     = float(hist["Low"].min())
            volume_p  = float(hist["Volume"].sum())

            result = {
                "name"          : "^HSI",
                "open"          : open_p,
                "prev_close"    : None,
                "current"       : current_p,
                "high"          : high_p,
                "low"           : low_p,
                "volume"        : volume_p,
                "timestamp"     : "",
                "intraday_ret"  : (current_p - open_p) / open_p,
                "intraday_range": (high_p - low_p)     / open_p,
                "high_vs_open"  : (high_p - open_p)    / open_p,
                "low_vs_open"   : (low_p  - open_p)    / open_p,
            }
            print(f"  ✅ Intraday HSI (yfinance): "
                  f"現價={current_p:,.0f} 高={high_p:,.0f} 低={low_p:,.0f}")
            return result

    except Exception as e:
        print(f"  ⚠️  yfinance intraday 失敗: {str(e)[:60]}")

    # ── Source 3: 昨日收市 fallback ─────────────────────────
    print("  ⚠️  Intraday 全部失敗，使用昨日收市數據")
    try:
        # [NOTE] 原版讀取全局 df_all；此處改為接受參數傳入，
        # 由呼叫端 (daily_pipeline.py) 傳入 df_all.iloc[-1]
        return None
    except Exception as e:
        print(f"  ❌ 全部 fallback 失敗: {e}")
        return None


def fetch_intraday_hsi_with_fallback(df_all: pd.DataFrame):
    """
    包裝版本：Source 3 fallback 需要 df_all，
    原版用 global 讀取，這裡改為顯式參數傳入。
    """
    result = fetch_intraday_hsi()
    if result is not None:
        return result

    try:
        last_row = df_all.iloc[-1]
        c        = float(last_row["close"])
        o        = float(last_row.get("open",  c))
        h        = float(last_row.get("high",  c))
        lo       = float(last_row.get("low",   c))
        result   = {
            "name"          : "HSI_fallback",
            "open"          : o,
            "prev_close"    : c,
            "current"       : c,
            "high"          : h,
            "low"           : lo,
            "volume"        : float(last_row.get("volume", 0)),
            "timestamp"     : "",
            "intraday_ret"  : 0.0,
            "intraday_range": (h - lo) / max(o, 1),
            "high_vs_open"  : (h  - o) / max(o, 1),
            "low_vs_open"   : (lo - o) / max(o, 1),
        }
        print(f"  ✅ Fallback 昨收={c:,.0f}")
        return result
    except Exception as e:
        print(f"  ❌ 全部 fallback 失敗: {e}")
        return None


def update_last_row_with_intraday(last_row: pd.DataFrame, intraday: dict) -> pd.DataFrame:
    """
    用日內數據更新 last_row 的特徵:
    - 更新 open/high/low（今日已知）
    - 重新計算相關特徵
    """
    row = last_row.copy()

    if intraday is None:
        return row

    if intraday.get("open"):
        row["open"] = intraday["open"]
    if intraday.get("high"):
        row["high"] = intraday["high"]
    if intraday.get("low"):
        row["low"]  = intraday["low"]

    if "intraday_ret" in intraday:
        row["intraday_ret"]      = intraday["intraday_ret"]
    if "intraday_range" in intraday:
        row["intraday_range"]    = intraday["intraday_range"]
    if "high_vs_open" in intraday:
        row["high_vs_open"]      = intraday["high_vs_open"]
    if "low_vs_open" in intraday:
        row["low_vs_open"]       = intraday["low_vs_open"]
    if "ret_vs_prev" in intraday:
        row["ret_vs_prev_close"] = intraday["ret_vs_prev"]

    if intraday.get("open") and intraday.get("prev_close"):
        row["gap_ret"] = (intraday["open"] - intraday["prev_close"]) / intraday["prev_close"]

    if (intraday.get("open") and intraday.get("high") and
        intraday.get("low") and intraday.get("current")):
        body  = abs(intraday["open"] - intraday["current"])
        upper = intraday["high"] - max(intraday["open"], intraday["current"])
        lower = min(intraday["open"], intraday["current"]) - intraday["low"]
        if body > 0:
            row["upper_wick"] = upper / body
            row["lower_wick"] = lower / body
        row["body_size"] = body / (intraday["open"] + 1e-9)

    return row
