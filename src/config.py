# ============================================================
# src/config.py
# Merged from: Cell 1 (deps note), Cell 2 (global config +
# enhancement classes), Cell 2B (dynamic prediction context)
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import time
import requests
import pytz
from datetime import datetime, timedelta
from collections import deque
from scipy import stats
from sklearn.metrics import f1_score

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)

# ── ENH #9: pandas_market_calendars 假期日曆 ────────────────
try:
    import pandas_market_calendars as mcal
    _XHKG = mcal.get_calendar("XHKG")

    def is_hk_trading_day(chk_date) -> bool:
        d = pd.Timestamp(chk_date).normalize()
        sched = _XHKG.schedule(
            start_date=d.strftime("%Y-%m-%d"),
            end_date=d.strftime("%Y-%m-%d")
        )
        return len(sched) > 0

    def get_next_hk_trading_day(from_date):
        nxt = pd.Timestamp(from_date) + timedelta(days=1)
        while not is_hk_trading_day(nxt):
            nxt += timedelta(days=1)
        return nxt.to_pydatetime()

    def get_prev_hk_trading_day(from_date):
        prv = pd.Timestamp(from_date) - timedelta(days=1)
        while not is_hk_trading_day(prv):
            prv -= timedelta(days=1)
        return prv.to_pydatetime()

    print("  ✅ ENH#9: pandas_market_calendars XHKG loaded")

except ImportError:
    print("  ⚠️  pandas_market_calendars 未安裝，使用 fallback 假期表")

    HK_HOLIDAYS = {
        "2025-01-01","2025-01-29","2025-01-30","2025-01-31",
        "2025-04-04","2025-04-18","2025-04-19","2025-04-21",
        "2025-05-01","2025-05-05","2025-06-02","2025-07-01",
        "2025-09-29","2025-10-01","2025-10-06","2025-12-25","2025-12-26",
        "2026-01-01","2026-02-17","2026-02-18","2026-02-19","2026-02-20",
        "2026-04-03","2026-04-06","2026-05-01","2026-05-25",
        "2026-06-19","2026-07-01","2026-10-01","2026-10-07",
        "2026-12-25",
    }

    def is_hk_trading_day(chk_date) -> bool:
        if chk_date.weekday() >= 5:
            return False
        if str(chk_date.date()) in HK_HOLIDAYS:
            return False
        return True

    def get_next_hk_trading_day(from_date):
        nxt = from_date + timedelta(days=1)
        while not is_hk_trading_day(nxt):
            nxt += timedelta(days=1)
        return nxt

    def get_prev_hk_trading_day(from_date):
        prv = from_date - timedelta(days=1)
        while not is_hk_trading_day(prv):
            prv -= timedelta(days=1)
        return prv


# ── Trading Calendar Helper Functions ──────────────────────────
def get_hkt_now():
    return datetime.utcnow() + timedelta(hours=8)

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


# ── 關鍵日期 ──────────────────────────────────────────────────
NOW_HKT           = get_hkt_now()
TODAY             = NOW_HKT
TODAY_STR         = TODAY.strftime("%Y-%m-%d")

LAST_TRADING_DAY  = get_last_trading_day(NOW_HKT)
LAST_TRADING_STR  = LAST_TRADING_DAY.strftime("%Y-%m-%d")

NEXT_TRADING_DAY  = get_next_trading_day(NOW_HKT)
NEXT_TRADING_STR  = NEXT_TRADING_DAY.strftime("%Y-%m-%d")

MARKET_OPEN       = is_market_open_now()
AFTER_CLOSE       = (NOW_HKT.weekday() < 5 and NOW_HKT.hour >= 17)

ROLLING_YEARS    = 5
TRAIN_END_YEAR   = TODAY.year - 1
TRAIN_END        = f"{TRAIN_END_YEAR}-12-31"
TRAIN_START_YEAR = TODAY.year - ROLLING_YEARS
TRAIN_START      = f"{TRAIN_START_YEAR}-01-01"

SIM_START         = f"{TODAY.year}-01-01"
SIM_END           = LAST_TRADING_STR

DOWNLOAD_START   = f"{TODAY.year - ROLLING_YEARS - 1}-01-01"
DOWNLOAD_END      = (datetime.strptime(TODAY_STR, "%Y-%m-%d")
                     + timedelta(days=1)).strftime("%Y-%m-%d")

print(f"  訓練窗口 : {TRAIN_START} → {TRAIN_END}  ({ROLLING_YEARS} 年)")
print(f"  數據下載 : {DOWNLOAD_START} → 今日")

# ── 模型配置 ──────────────────────────────────────────────────
TRAIN_WINDOW         = 504
PROB_LONG            = 0.60
PROB_SHORT           = 0.42
TRANSACTION_COST     = 0.002
SLIPPAGE_POINTS      = 3

PERF_WINDOW          = 42
PERF_THRESHOLD       = 0.46
RETRAIN_INTERVAL     = 42
KS_PVALUE            = 0.05
ROLLBACK_TOLERANCE   = 0.2

LGBM_PARAMS = {
    "objective"        : "binary",
    "metric"           : "auc",
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

# ── ENH #11: Credentials 用環境變量 ─────────────────────────────
from dotenv import load_dotenv
load_dotenv(override=False)

EMAIL_SENDER   = os.environ.get("EMAIL_SENDER",   "")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", "")

if not EMAIL_SENDER:
    # [FIX] 原版在此處印出真實 Gmail 地址/應用密碼/收件人作為「示例」，
    # 已改為通用佔位提示，不洩露任何真實憑證
    print("  ⚠️  ENH#11: EMAIL_SENDER 未設定，請在 .env / GitHub Secrets 加入:")
    print("       EMAIL_SENDER=<your_email@gmail.com>")
    print("       EMAIL_PASSWORD=<your_app_password>")
    print("       EMAIL_RECEIVER=<comma,separated,recipients>")

# [FIX] 原版: FRED_API_KEY = os.environ.get("FRED_API_KEY", "9affb...literal key...")
# 已移除硬編碼 fallback，改為純環境變量讀取（無效則為 None）
FRED_API_KEY = os.environ.get("FRED_API_KEY")

# [FIX] DASHSCOPE_API_KEY 原本在 Intraday Monitoring 段落硬編碼明文，
# 該段落本身不納入排程版本（常駐 polling loop，與 GitHub Actions cron 不相容）。
# 若日後需要，改用:
# DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")

FEATURE_BLACKLIST = {
    "date","next_ret","target",
    "open","high","low","close","volume"
}

stock_models = {}


# ============================================================
# ENH #1: Regime-Based Dynamic Threshold Calibrator
# ============================================================
class RegimeThresholdCalibrator:
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
            np.where(probs <= short_thresh, -1, 0)
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
                "long" : round(best_long,  2),
                "short": round(best_short, 2),
                "f1"   : round(best_score, 4),
                "n"    : len(rdf),
            }
            print(f"  [Calibrator] {regime:8s}: "
                  f"long={best_long:.2f} short={best_short:.2f} "
                  f"F1={best_score:.4f} n={len(rdf)}")

        self.fitted = True
        return self

    def get_signal(self, prob: float, regime: str) -> int:
        t = self.thresholds.get(regime, self.thresholds.get("neutral"))
        if prob >= t["long"]:
            return 1
        elif prob <= t["short"]:
            return -1
        return 0

    def summary(self):
        print("\n  === Regime Threshold Summary ===")
        for r, t in self.thresholds.items():
            print(f"  {r:8s}: long={t['long']:.2f}  "
                  f"short={t['short']:.2f}  F1={t['f1']:.4f}")


# ============================================================
# ENH #3: Position Sizer (Fractional Kelly)
# ============================================================
class PositionSizer:
    def __init__(self, method: str = "fractional_kelly",
                 max_position: float = 1.0,
                 kelly_fraction: float = 0.25,
                 min_edge: float = 0.02):
        self.method         = method
        self.max_position   = max_position
        self.kelly_fraction = kelly_fraction
        self.min_edge       = min_edge

    def compute_size(self, prob: float, signal: int,
                     regime: str = "neutral", roll_acc: float = 0.5) -> float:
        if signal == 0:
            return 0.0

        win_prob = prob if signal == 1 else (1.0 - prob)
        edge     = win_prob - 0.5

        if edge < self.min_edge:
            return 0.0

        if self.method == "fractional_kelly":
            size = edge * self.kelly_fraction
        elif self.method == "prob_linear":
            size = (edge * 2) * self.max_position
        else:
            size = self.max_position

        regime_mult = {"bull": 1.10, "neutral": 1.00, "bear": 0.70}.get(regime, 1.0)
        acc_mult    = max(0.3, min(1.5, roll_acc / 0.5))

        size = size * regime_mult * acc_mult
        return float(np.clip(size, 0.0, self.max_position))


# ============================================================
# ENH #5: Dynamic Ensemble Weighter
# ============================================================
class DynamicEnsembleWeighter:
    def __init__(self, window: int = 20, min_weight: float = 0.1):
        self.window     = window
        self.min_weight = min_weight
        self.histories  = {
            "global": deque(maxlen=window),
            "regime": deque(maxlen=window),
        }

    def update(self, model_name: str, prob: float, actual: int):
        if model_name in self.histories:
            self.histories[model_name].append((float(prob), int(actual)))

    def _rolling_brier(self, history) -> float:
        if len(history) < 5:
            return 0.25
        probs   = np.array([h[0] for h in history])
        actuals = np.array([h[1] for h in history])
        return float(np.mean((probs - actuals) ** 2))

    def get_weights(self) -> dict:
        scores = {}
        for name, hist in self.histories.items():
            bs    = self._rolling_brier(hist)
            skill = max(0.0, 1.0 - bs / 0.25)
            scores[name] = skill

        total = sum(scores.values())
        if total < 1e-9:
            return {k: 1.0 / len(scores) for k in scores}

        weights = {k: max(v / total, self.min_weight) for k, v in scores.items()}
        total_w = sum(weights.values())
        return {k: v / total_w for k, v in weights.items()}

    def ensemble(self, g_prob: float, r_prob: float) -> float:
        w = self.get_weights()
        result = w.get("global", 0.5) * g_prob + w.get("regime", 0.5) * r_prob
        return float(np.clip(result, 0.0, 1.0))

    def weight_summary(self) -> str:
        w = self.get_weights()
        return f"global={w.get('global',0):.2f} regime={w.get('regime',0):.2f}"


threshold_calibrator = RegimeThresholdCalibrator()
position_sizer       = PositionSizer(
    method="fractional_kelly", max_position=1.0,
    kelly_fraction=0.25, min_edge=0.02,
)
ensemble_weighter = DynamicEnsembleWeighter(window=20)

print("✅ config.py 完成: 所有配置 + Enhancement 物件初始化完畢")
print(f"   TODAY={TODAY_STR}  TRAIN={TRAIN_START}~{TRAIN_END}")
print(f"   SIM  ={SIM_START}~{SIM_END}")


# ============================================================
# Cell 2B: 動態預測目標邏輯
# ============================================================
HKT = pytz.timezone("Asia/Hong_Kong")

MARKET_OPEN_H    = 9
MARKET_OPEN_M    = 30
MARKET_CLOSE_H   = 16
MARKET_CLOSE_M   = 0
INTRADAY_START_H = 9
INTRADAY_START_M = 30


def get_hkt_now_tz():
    return datetime.now(HKT)


def is_weekday(dt):
    return dt.weekday() < 5


def is_market_hours(dt):
    if not is_weekday(dt):
        return False
    t = dt.time()
    open_t  = dt.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M,
                         second=0, microsecond=0).time()
    close_t = dt.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M,
                         second=0, microsecond=0).time()
    return open_t <= t < close_t


def is_after_close(dt):
    if not is_weekday(dt):
        return True
    t      = dt.time()
    close_t = dt.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M,
                         second=0, microsecond=0).time()
    return t >= close_t


def is_before_open(dt):
    if not is_weekday(dt):
        return False
    t      = dt.time()
    open_t = dt.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M,
                        second=0, microsecond=0).time()
    return t < open_t


def get_next_trading_day_from(dt):
    candidate = dt + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def get_last_trading_day_from(dt):
    candidate = dt - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def get_prediction_context():
    now = get_hkt_now_tz()
    today = now.date()

    result = {
        "now_hkt"         : now,
        "today_str"       : today.strftime("%Y-%m-%d"),
        "now_time_str"    : now.strftime("%H:%M:%S"),
        "now_hkt_str"     : now.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if not is_weekday(now):
        last_trading = get_last_trading_day_from(
            datetime.combine(today, datetime.min.time()))
        next_trading = get_next_trading_day_from(
            datetime.combine(today, datetime.min.time()))
        result.update({
            "scenario"       : 4,
            "scenario_desc"  : "週末 → 預測下一交易日",
            "predict_date_str": next_trading.strftime("%Y-%m-%d"),
            "base_date_str"  : last_trading.strftime("%Y-%m-%d"),
            "use_intraday"   : False,
            "intraday_avail" : "無（週末）",
            "data_note"      : (
                f"使用 {last_trading.strftime('%Y-%m-%d')} "
                f"收市數據預測 {next_trading.strftime('%Y-%m-%d')}"
            ),
        })

    elif is_after_close(now):
        next_trading = get_next_trading_day_from(
            datetime.combine(today, datetime.min.time()))
        result.update({
            "scenario"        : 1,
            "scenario_desc"   : "平日收市後 → 預測明日",
            "predict_date_str": next_trading.strftime("%Y-%m-%d"),
            "base_date_str"   : today.strftime("%Y-%m-%d"),
            "use_intraday"    : False,
            "intraday_avail"  : "今日收市數據已可用",
            "data_note"       : (
                f"使用今日（{today}）完整收市數據，"
                f"預測 {next_trading.strftime('%Y-%m-%d')}
                ),
          })

    elif is_before_open(now):
        last_trading = get_last_trading_day_from(
            datetime.combine(today, datetime.min.time()))
        result.update({
            "scenario"        : 2,
            "scenario_desc"   : "開市前 → 預測今日（昨收數據）",
            "predict_date_str": today.strftime("%Y-%m-%d"),
            "base_date_str"   : last_trading.strftime("%Y-%m-%d"),
            "use_intraday"    : False,
            "intraday_avail"  : "未開市，無日內數據",
            "data_note"       : (
                f"使用昨日（{last_trading.strftime('%Y-%m-%d')}）"
                f"收市數據，預測今日 {today}"
            ),
        })

    elif is_market_hours(now):
        t = now.time()

        if t.hour < 10:
            intraday_qual = "早盤（數據有限）"
            intraday_pct  = 0.1
        elif t.hour < 12:
            intraday_qual = "上午盤（數據中等）"
            intraday_pct  = 0.4
        elif t.hour < 14:
            intraday_qual = "午後（數據較好）"
            intraday_pct  = 0.6
        else:
            intraday_qual = "尾市（數據充足）"
            intraday_pct  = 0.85

        last_trading = get_last_trading_day_from(
            datetime.combine(today, datetime.min.time()))

        result.update({
            "scenario"        : 3,
            "scenario_desc"   : f"開市中 → 預測今日（{intraday_qual}）",
            "predict_date_str": today.strftime("%Y-%m-%d"),
            "base_date_str"   : last_trading.strftime("%Y-%m-%d"),
            "use_intraday"    : True,
            "intraday_avail"  : intraday_qual,
            "intraday_pct"    : intraday_pct,
            "data_note"       : (
                f"使用昨收 + 今日實時數據（{intraday_qual}），"
                f"預測今日 {today} 收市方向"
            ),
        })

    else:
        last_trading = get_last_trading_day_from(
            datetime.combine(today, datetime.min.time()))
        next_trading = get_next_trading_day_from(
            datetime.combine(today, datetime.min.time()))
        result.update({
            "scenario"        : 1,
            "scenario_desc"   : "其他情況 → 預測明日",
            "predict_date_str": next_trading.strftime("%Y-%m-%d"),
            "base_date_str"   : last_trading.strftime("%Y-%m-%d"),
            "use_intraday"    : False,
            "intraday_avail"  : "未知",
            "data_note"       : "使用最後已知收市數據",
        })

    return result


# ── 立即執行，取得當前情境 ───────────────────────────────────
CTX = get_prediction_context()

print("=" * 60)
print("  🕐 動態預測情境判斷")
print("=" * 60)
print(f"  當前香港時間 : {CTX['now_hkt_str']}")
print(f"  今日日期     : {CTX['today_str']}")
print()
print(f"  情境         : {CTX['scenario']} — {CTX['scenario_desc']}")
print(f"  基準日期     : {CTX['base_date_str']}（最後已知數據）")
print(f"  預測目標     : {CTX['predict_date_str']}")
print(f"  日內數據     : {CTX['intraday_avail']}")
print(f"  數據說明     : {CTX['data_note']}")
print()

scenario_icons = {
    1: "🌙 收市後",
    2: "🌅 開市前",
    3: "📈 開市中",
    4: "📅 週末",
}
print(f"  {scenario_icons.get(CTX['scenario'], '❓')} "
      f"情境 {CTX['scenario']}：{CTX['scenario_desc']}")
print("=" * 60)
print("✅ config.py (Cell 1+2+2B) 載入完成")
