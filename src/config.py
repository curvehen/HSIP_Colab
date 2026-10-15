# ============================================================
# src/config.py
# Merged from: Cell 1 (imports + 全域常數), Cell 2 (交易日曆),
#              Cell 5 (ENH #1/#3/#5 自學組件: Threshold Calibrator,
#              Position Sizer, Ensemble Weighter)
#
# SECURITY FIXES applied vs. original notebook:
#   - FRED_API_KEY: removed hardcoded fallback value
#     ("9affb17a4df8b012b124b6921df4b010"), now required via env var.
#   - DASHSCOPE_API_KEY: removed hardcoded literal key, now required
#     via env var.
#   - Removed `from google.colab import userdata` (Colab-only import).
# ============================================================
import os
from datetime import datetime, timedelta
from collections import deque

import numpy as np
import pandas as pd
import pytz
from sklearn.metrics import f1_score

try:
    import pandas_market_calendars as mcal
    _HAS_MCAL = True
except ImportError:
    _HAS_MCAL = False


# ============================================================
# Cell 1: 輸出目錄
# ============================================================
os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("state",   exist_ok=True)


# ============================================================
# Cell 2: 交易日曆輔助函式（原樣保留，HK 時區手動計算，不依賴 pytz）
# ============================================================
def get_hkt_now():
    """返回當前香港時間 (UTC+8)，不依賴 pytz"""
    return datetime.utcnow() + timedelta(hours=8)


if _HAS_MCAL:
    _XHKG = mcal.get_calendar("XHKG")

    def is_hk_trading_day(chk_date) -> bool:
        d = pd.Timestamp(chk_date).normalize()
        sched = _XHKG.schedule(
            start_date=d.strftime("%Y-%m-%d"),
            end_date=d.strftime("%Y-%m-%d"),
        )
        return len(sched) > 0
else:
    # Fallback: 若 pandas_market_calendars 未安裝，退化為「非週末即交易日」
    def is_hk_trading_day(chk_date) -> bool:
        return pd.Timestamp(chk_date).weekday() < 5


def get_last_trading_day(reference_date=None):
    if reference_date is None:
        reference_date = get_hkt_now()
    if reference_date.weekday() < 5 and reference_date.hour >= 17:
        d = reference_date.replace(hour=0, minute=0, second=0, microsecond=0)
        if is_hk_trading_day(d):
            return d
    candidate = reference_date - timedelta(days=1)
    candidate = candidate.replace(hour=0, minute=0, second=0, microsecond=0)
    while not is_hk_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def get_next_trading_day(reference_date=None):
    if reference_date is None:
        reference_date = get_hkt_now()
    candidate = reference_date + timedelta(days=1)
    candidate = candidate.replace(hour=0, minute=0, second=0, microsecond=0)
    while not is_hk_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def is_market_open_now():
    now_hkt = get_hkt_now()
    if now_hkt.weekday() >= 5:
        return False
    market_open  = now_hkt.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now_hkt.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now_hkt <= market_close


# ============================================================
# Cell 1: 日期 / 時間常數
# ============================================================
NOW_HKT          = get_hkt_now()
TODAY            = NOW_HKT
TODAY_STR        = TODAY.strftime("%Y-%m-%d")
LAST_TRADING_DAY = get_last_trading_day(NOW_HKT)
LAST_TRADING_STR = LAST_TRADING_DAY.strftime("%Y-%m-%d")
NEXT_TRADING_DAY = get_next_trading_day(NOW_HKT)
NEXT_TRADING_STR = NEXT_TRADING_DAY.strftime("%Y-%m-%d")
MARKET_OPEN      = is_market_open_now()
AFTER_CLOSE      = (NOW_HKT.weekday() < 5 and NOW_HKT.hour >= 17)

ROLLING_YEARS     = 5
TRAIN_END_YEAR    = TODAY.year - 1
TRAIN_END         = f"{TRAIN_END_YEAR}-12-31"
TRAIN_START_YEAR  = TODAY.year - ROLLING_YEARS
TRAIN_START       = f"{TRAIN_START_YEAR}-01-01"
SIM_START         = f"{TODAY.year}-01-01"
SIM_END           = LAST_TRADING_STR
DOWNLOAD_START    = f"{TODAY.year - ROLLING_YEARS - 1}-01-01"
DOWNLOAD_END      = (datetime.strptime(TODAY_STR, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

TRAIN_WINDOW        = 504
PROB_LONG           = 0.60
PROB_SHORT          = 0.42
TRANSACTION_COST    = 0.002
SLIPPAGE_POINTS     = 3
PERF_WINDOW         = 42
PERF_THRESHOLD      = 0.46
RETRAIN_INTERVAL    = 42
KS_PVALUE           = 0.05
ROLLBACK_TOLERANCE  = 0.2


# ============================================================
# LightGBM 參數
# ============================================================
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.02,
    "num_leaves": 48,
    "min_child_samples": 30,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.3,
    "lambda_l2": 0.8,
    "verbose": -1,
    "n_estimators": 600,
    "random_state": 42,
}

LGBM_REG_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.02,
    "num_leaves": 48,
    "min_child_samples": 30,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.3,
    "lambda_l2": 0.8,
    "verbose": -1,
    "n_estimators": 600,
    "random_state": 42,
}


# ============================================================
# SECURITY FIX: API Keys — 嚴格從環境變數讀取，無硬編碼 fallback
# ============================================================
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
if not FRED_API_KEY:
    print("  ⚠️  FRED_API_KEY 未設定 — FRED 總經數據將被跳過")

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
if not DASHSCOPE_API_KEY:
    print("  ⚠️  DASHSCOPE_API_KEY 未設定 — AI 評論功能將被跳過")

AI_MODEL = "qwen-plus"


# ============================================================
# 特徵黑名單
# ============================================================
FEATURE_BLACKLIST = {
    "date", "next_ret", "target",
    "open", "high", "low", "close", "volume",
}


# ============================================================
# 時區 / 盤中時間常數
# ============================================================
HKT    = pytz.timezone("Asia/Hong_Kong")
HK_TZ  = pytz.timezone("Asia/Hong_Kong")

MARKET_OPEN_H    = 9
MARKET_OPEN_M    = 30
MARKET_CLOSE_H   = 16
MARKET_CLOSE_M   = 0
INTRADAY_START_H = 9
INTRADAY_START_M = 30


# ============================================================
# 資產 / 經濟數據代碼對照表
# ============================================================
yf_map = {
    "sp500":   "^GSPC",
    "nasdaq":  "^IXIC",
    "vix":     "^VIX",
    "us10y":   "^TNX",
    "us2y":    "^IRX",
    "dxy":     "DX-Y.NYB",
    "csi300":  "000300.SS",
    "sse":     "000001.SS",
    "gold":    "GC=F",
    "oil":     "CL=F",
    "copper":  "HG=F",
    "usdcny":  "CNY=X",
    "tencent": "0700.HK",
    "hsbc":    "0005.HK",
    "alibaba": "9988.HK",
    "meituan": "3690.HK",
}

fred_map = {
    "fed_rate":    "FEDFUNDS",
    "us_cpi":      "CPIAUCSL",
    "us_core_cpi": "CPILFESL",
    "nonfarm":     "PAYEMS",
    "yield_curve": "T10Y2Y",
    "jobless":     "ICSA",
}


# ============================================================
# 個股清單
# ============================================================
MODEL_DIR = "models"

STOCK_LIST = {
    "0700.HK": "騰訊",     "9988.HK": "阿里巴巴", "3690.HK": "美團",
    "9618.HK": "京東",     "0992.HK": "聯想",     "0005.HK": "匯豐",
    "0939.HK": "建設銀行", "1398.HK": "工商銀行", "0388.HK": "港交所",
    "2318.HK": "中國平安", "3988.HK": "中國銀行", "0016.HK": "新鴻基地產",
    "0001.HK": "長和",     "1109.HK": "華潤置地", "9999.HK": "網易",
    "0291.HK": "華潤啤酒", "0762.HK": "中國聯通", "0857.HK": "中國石油",
    "0386.HK": "中國石化", "1088.HK": "中國神華", "1177.HK": "中國生物製藥",
    "2269.HK": "藥明生物", "1211.HK": "比亞迪股份", "0968.HK": "信義光能",
}

SECTOR_MAP = {
    "0700.HK": "科技", "9988.HK": "科技", "3690.HK": "科技", "9618.HK": "科技",
    "0992.HK": "科技", "9999.HK": "科技",
    "0005.HK": "金融", "0939.HK": "金融", "1398.HK": "金融", "0388.HK": "金融",
    "2318.HK": "金融", "3988.HK": "金融",
    "0016.HK": "地產", "0001.HK": "地產", "1109.HK": "地產",
    "0291.HK": "消費",
    "0762.HK": "電訊",
    "0857.HK": "能源", "0386.HK": "能源", "1088.HK": "能源",
    "1177.HK": "醫療", "2269.HK": "醫療",
    "1211.HK": "車",
    "0968.HK": "光伏",
}

STOCK_TRAIN_START  = "2020-01-01"
STOCK_SIM_START    = SIM_START
STOCK_MIN_HISTORY  = 252
STOCK_TOP_N        = 5


# ============================================================
# 高/低點與 AI 評論相關常數
# ============================================================
SUDDEN_CHANGE_THRESHOLD = 0.3
SUDDEN_RANGE_THRESHOLD  = 0.4
SCHEDULED_AI_TIMES      = ["11:30", "12:30", "13:30"]


# ============================================================
# 隨機種子 / HMM 參數
# ============================================================
SEEDS        = [42, 0, 123]
RANDOM_STATE = 42
HMM_N_ITER   = 300
HMM_TOL      = 1e-5


# ============================================================
# ENH #1: Regime-Based Dynamic Threshold Calibrator
# ============================================================
class RegimeThresholdCalibrator:
    """
    為每個 HMM regime 獨立 calibrate PROB_LONG / PROB_SHORT。
    用 validation set F1 score 做優化目標。
    """

    def __init__(self, regimes=("bull", "neutral", "bear")):
        self.regimes = regimes
        self.thresholds = {
            r: {"long": PROB_LONG, "short": PROB_SHORT, "f1": 0.0, "n": 0}
            for r in regimes
        }
        self.fitted = False

    def _score_threshold(self, probs, actuals, long_thresh, short_thresh):
        signals = np.where(
            probs >= long_thresh, 1,
            np.where(probs <= short_thresh, -1, 0),
        )
        mask = signals != 0
        if mask.sum() < 10:
            return 0.0
        y_pred = (signals[mask] == 1).astype(int)
        y_true = np.array(actuals)[mask]
        if len(np.unique(y_true)) < 2:
            return 0.0
        return f1_score(y_true, y_pred, zero_division=0)

    def calibrate(self, val_df: pd.DataFrame,
                  prob_col: str = "ensemble_prob",
                  target_col: str = "actual_target",
                  regime_col: str = "regime"):
        if len(val_df) < 30:
            print("  [Calibrator] 樣本不足，使用 default threshold")
            return self

        for regime in self.regimes:
            rdf = val_df[val_df[regime_col] == regime]
            if len(rdf) < 20:
                continue

            probs   = rdf[prob_col].values
            actuals = rdf[target_col].values
            best_long  = PROB_LONG
            best_short = PROB_SHORT
            best_score = -1.0

            for lt in np.arange(0.52, 0.76, 0.02):
                for st in np.arange(0.28, 0.49, 0.02):
                    if st >= lt:
                        continue
                    score = self._score_threshold(probs, actuals, lt, st)
                    if score > best_score:
                        best_score = score
                        best_long  = lt
                        best_short = st

            

            self.thresholds[regime] = {
                "long":  round(best_long,  2),
                "short": round(best_short, 2),
                "f1":    round(best_score, 4),
                "n":     len(rdf),
            }
            print(f"  [Calibrator] {regime:8s}: "
                  f"long={best_long:.2f} short={best_short:.2f} "
                  f"F1={best_score:.4f} (n={len(rdf)})")

        self.fitted = True
        return self

    def get_thresholds(self, regime: str):
        if regime not in self.thresholds:
            return PROB_LONG, PROB_SHORT
        t = self.thresholds[regime]
        return t["long"], t["short"]

    def get_signal(self, prob: float, regime: str) -> int:
        long_t, short_t = self.get_thresholds(regime)
        if prob >= long_t:
            return 1
        elif prob <= short_t:
            return -1
        return 0


# ============================================================
# ENH #3: Kelly-based Dynamic Position Sizer
# ============================================================
class PositionSizer:
    """
    依據 Kelly Criterion 估計部位大小，並以信心度/regime/
    滾動命中率做調整，限制在 [MIN_POSITION, MAX_POSITION] 之間。
    """

    def __init__(self, max_position: float = 1.0, min_position: float = 0.2,
                 kelly_fraction: float = 0.5):
        self.max_position   = max_position
        self.min_position   = min_position
        self.kelly_fraction = kelly_fraction
        self.regime_risk = {"bull": 1.0, "neutral": 0.8, "bear": 0.6}

    def _kelly(self, prob: float) -> float:
        p = max(min(prob, 0.999), 0.001)
        q = 1 - p
        b = 1.0
        kelly = (b * p - q) / b
        return max(kelly, 0.0)

    def compute_size(self, prob: float, signal: int, regime: str,
                      roll_acc: float = 0.5) -> float:
        if signal == 0:
            return 0.0

        edge_prob = prob if signal == 1 else (1 - prob)
        kelly_raw = self._kelly(edge_prob)
        kelly_adj = kelly_raw * self.kelly_fraction

        regime_mult = self.regime_risk.get(regime, 0.8)

        conf_mult = 1.0
        if roll_acc < 0.45:
            conf_mult = 0.5
        elif roll_acc < 0.50:
            conf_mult = 0.75
        elif roll_acc > 0.58:
            conf_mult = 1.15

        size = kelly_adj * regime_mult * conf_mult
        size = max(min(size, self.max_position), self.min_position)
        return round(size, 4)


# ============================================================
# ENH #5: Dynamic Ensemble Weighter
# ============================================================
class DynamicEnsembleWeighter:
    """
    追蹤 global model 與 regime model 近期表現，
    動態調整兩者在 ensemble 中的權重（滑動視窗命中率反比）。
    """

    def __init__(self, window: int = 60, min_weight: float = 0.2):
        self.window     = window
        self.min_weight = min_weight
        self.history = {
            "global": deque(maxlen=window),
            "regime": deque(maxlen=window),
        }
        self.weights = {"global": 0.5, "regime": 0.5}

    def update(self, model_key: str, prob: float, actual_target: int):
        if model_key not in self.history:
            return
        pred_correct = int((prob >= 0.5) == bool(actual_target))
        self.history[model_key].append(pred_correct)
        self._recompute_weights()

    def _recompute_weights(self):
        g_hist = self.history["global"]
        r_hist = self.history["regime"]

        if len(g_hist) < 10 or len(r_hist) < 10:
            self.weights = {"global": 0.5, "regime": 0.5}
            return

        g_acc = sum(g_hist) / len(g_hist)
        r_acc = sum(r_hist) / len(r_hist)
        total = g_acc + r_acc

        if total <= 0:
            self.weights = {"global": 0.5, "regime": 0.5}
            return

        g_w = g_acc / total
        r_w = r_acc / total

        g_w = max(min(g_w, 1 - self.min_weight), self.min_weight)
        r_w = 1 - g_w

        self.weights = {"global": round(g_w, 4), "regime": round(r_w, 4)}

    def ensemble(self, g_prob: float, r_prob: float) -> float:
        w = self.weights
        return w["global"] * g_prob + w["regime"] * r_prob

    def weight_summary(self) -> dict:
        return dict(self.weights)


# ============================================================
# Singleton 實例 — 由 training.py / daily_pipeline.py 共用，
# 並透過 state['threshold_calibrator'] 等 key 持久化/還原
# ============================================================
threshold_calibrator = RegimeThresholdCalibrator()
position_sizer       = PositionSizer(
    max_position=1.0, min_position=0.2, kelly_fraction=0.5,
)
ensemble_weighter    = DynamicEnsembleWeighter(window=60, min_weight=0.2)


# ============================================================
# 持久化相關路徑與 state keys
# ============================================================
MODELS_DIR = "models"
RESULTS_DIR = "results"
STATE_DIR  = "state"
STATE_PATH = os.path.join(STATE_DIR, "pipeline_state.pkl")

STATE_KEYS = [
    "df_all", "feat_cols",
    "global_model", "regime_models", "hmm",
    "monitor", "memory", "drift",
    "threshold_calibrator", "position_sizer", "ensemble_weighter",
    "wf_monthly", "last_backtest_report",
]

CHART_STATE_KEYS = [
    "last_backtest_report",
]

CALIBRATION_MIN_SAMPLES = 30

