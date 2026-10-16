# ============================================================
# src/models.py
# Merged from: Cell 5 (HMM Regime + 工具函數), Cell 5B (align_features
# / safe_predict 重複定義 — 已去重，保留單一正確版本)
# ============================================================
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
from scipy import stats
from hmmlearn.hmm import GaussianHMM
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
import lightgbm as lgb

from src.config import (
    LGBM_PARAMS, PERF_WINDOW, PERF_THRESHOLD,
    KS_PVALUE, ROLLBACK_TOLERANCE
)

from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN_ESTIMATOR = True
except ImportError:
    _HAS_FROZEN_ESTIMATOR = False

# ============================================================
# ENH #6: Enhanced HMMRegime (6維輸入特徵)
# ============================================================
class HMMRegime:
    def __init__(self, n_states=3):
        self.n_states  = n_states
        self.model     = None
        self.state_map = {}
        self.fitted    = False

    def _obs(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"]
        ret   = close.pct_change().fillna(0)

        vol20 = ret.rolling(20).std().fillna(ret.std())

        nb_raw = df.get("northbound_flow", pd.Series(0, index=df.index))
        nb     = pd.to_numeric(nb_raw, errors="coerce").fillna(0)
        nb_mu  = nb.rolling(20).mean().fillna(0)
        nb_sd  = nb.rolling(20).std().fillna(1).replace(0, 1)
        nb_z   = ((nb - nb_mu) / (nb_sd + 1e-9)).clip(-3, 3)

        if "vix" in df.columns:
            vix     = pd.to_numeric(df["vix"], errors="coerce").ffill().fillna(20)
            vix_chg = vix.diff().fillna(0)
            vix_sd  = vix_chg.rolling(20).std().fillna(1).replace(0, 1)
            vix_z   = (vix_chg / vix_sd).clip(-3, 3)
        else:
            vix_z   = pd.Series(0.0, index=df.index)

        if "us10y" in df.columns and "us2y" in df.columns:
            us10y  = pd.to_numeric(df["us10y"], errors="coerce").ffill().fillna(4.0)
            us2y   = pd.to_numeric(df["us2y"],  errors="coerce").ffill().fillna(4.0)
            spread = (us10y - us2y).fillna(0)
        elif "us10y" in df.columns:
            us10y  = pd.to_numeric(df["us10y"], errors="coerce").ffill().fillna(4.0)
            spread = us10y.diff().fillna(0)
        else:
            spread = pd.Series(0.0, index=df.index)

        sp_mu    = spread.rolling(20).mean().fillna(0)
        sp_sd    = spread.rolling(20).std().fillna(1).replace(0, 1)
        spread_z = ((spread - sp_mu) / (sp_sd + 1e-9)).clip(-3, 3)

        up_days   = (ret > 0).astype(float)
        breadth   = up_days.rolling(10).mean().fillna(0.5)
        breadth_z = ((breadth - 0.5) / 0.15).clip(-3, 3)

        obs = np.column_stack([
            ret.values, vol20.values, nb_z.values,
            vix_z.values, spread_z.values, breadth_z.values,
        ])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _simple_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "ret_1d" in df.columns:
            roll = df["ret_1d"].rolling(20).mean()
        else:
            roll = df["close"].pct_change().rolling(20).mean()
        df["hmm_regime"] = "neutral"
        df.loc[roll >  0.002, "hmm_regime"] = "bull"
        df.loc[roll < -0.002, "hmm_regime"] = "bear"
        df["hmm_state"] = df["hmm_regime"].map(
            {"bull": 0, "neutral": 1, "bear": 2}
        ).fillna(1).astype(int)
        df["hmm_prob_bull"]    = (df["hmm_regime"] == "bull").astype(float)
        df["hmm_prob_neutral"] = (df["hmm_regime"] == "neutral").astype(float)
        df["hmm_prob_bear"]    = (df["hmm_regime"] == "bear").astype(float)
        return df

    def fit(self, df: pd.DataFrame):
        obs = self._obs(df)
        for n_st in ([self.n_states, 2] if self.n_states > 2 else [self.n_states]):
            for cov in ["diag", "full", "tied"]:
                try:
                    m = GaussianHMM(
                        n_components=n_st, covariance_type=cov,
                        n_iter=200, random_state=42, min_covar=1e-3,
                    )
                    m.fit(obs)
                    _             = m.predict(obs)
                    self.model    = m
                    self.n_states = n_st
                    means         = m.means_[:, 0]
                    rank          = np.argsort(means)
                    if n_st == 3:
                        self.state_map = {rank[0]: "bear", rank[1]: "neutral", rank[2]: "bull"}
                    else:
                        self.state_map = {rank[0]: "bear", rank[1]: "bull"}
                    self.fitted = True
                    print(f"  HMM fit OK (ENH#6 6-dim): n_states={n_st}, cov={cov}")
                    return self
                except Exception as e:
                    print(f"  HMM WARN: n_states={n_st}, cov={cov} → {str(e)[:40]}")
                    continue
        print("  HMM 完全失敗 → 標記 fitted=False")
        self.fitted = False
        return self

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        try:
            if not self.fitted or self.model is None:
                raise AttributeError("HMM 未 fit")
            obs      = self._obs(df)
            states   = self.model.predict(obs)
            _, post  = self.model.score_samples(obs)
            df["hmm_state"]  = states
            df["hmm_regime"] = (
                pd.Series(states, index=df.index).map(self.state_map)
                .fillna("neutral").values
            )
            for label in ["bear", "neutral", "bull"]:
                state_id = [k for k, v in self.state_map.items() if v == label]
                if state_id and post.shape[1] > state_id[0]:
                    df[f"hmm_prob_{label}"] = post[:, state_id[0]]
                else:
                    df[f"hmm_prob_{label}"] = 1 / self.n_states
        except Exception as e:
            print(f"  WARN HMM add_features 失敗: {e}")
            print(f"  → 使用簡單規則代替")
            df = self._simple_regime(df)
        return df


# ============================================================
# LightGBM 訓練
# ============================================================
def train_lgbm(X_tr, y_tr, X_val, y_val):
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set  = [(X_val, y_val)],
        callbacks = [lgb.early_stopping(40, verbose=False),
                     lgb.log_evaluation(period=-1)]
    )
    # cv="prefit" 已在新版 scikit-learn（1.6+）移除，改用 FrozenEstimator
    # 包裝已訓練模型，效果與原本的 "prefit" 模式等價（不重新訓練，只做校準）。
    if _HAS_FROZEN_ESTIMATOR:
        cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    else:
        # 相容舊版 scikit-learn（<1.6）環境的備援寫法
        cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    cal.fit(X_val, y_val)
    return cal



def eval_model(model, X, y):
    p    = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    auc  = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
    acc  = accuracy_score(y, pred)
    return {"auc": auc, "accuracy": acc, "brier": brier_score_loss(y, p)}


# ============================================================
# PerfMonitor
# ============================================================
class PerfMonitor:
    def __init__(self, window=PERF_WINDOW, threshold=PERF_THRESHOLD):
        self.history      = deque(maxlen=window)
        self.threshold    = threshold
        self.log          = []
        self.consec_fail  = 0
        self.CONSEC_LIMIT = 3

    def update(self, prob, actual):
        self.history.append((1 if prob >= 0.5 else 0, int(actual)))

    def update_batch(self, probas, actuals):
        for p, a in zip(probas, actuals):
            self.update(float(p), int(a))

    def rolling_accuracy(self):
        if len(self.history) < 5:
            return 1.0
        preds   = [h[0] for h in self.history]
        actuals = [h[1] for h in self.history]
        return accuracy_score(actuals, preds)

    def rolling_auc(self):
        if len(self.history) < 10:
            return 0.5
        try:
            preds   = [h[0] for h in self.history]
            actuals = [h[1] for h in self.history]
            if len(set(actuals)) < 2:
                return 0.5
            return roc_auc_score(actuals, preds)
        except Exception:
            return 0.5

    def should_retrain(self):
        acc      = self.rolling_accuracy()
        auc      = self.rolling_auc()
        degraded = False
        reason   = ""

        if acc < self.threshold:
            degraded = True
            reason   = f"命中率退化({acc:.3f})"
        elif auc < 0.48:
            degraded = True
            reason   = f"AUC退化({auc:.3f})"

        if degraded:
            self.consec_fail += 1
            if self.consec_fail >= self.CONSEC_LIMIT:
                self.consec_fail = 0
                self.log.append(reason)
                return True, reason
            else:
                return False, "正常"
        else:
            self.consec_fail = 0
            return False, "正常"


# ============================================================
# DriftDetector
# ============================================================
class DriftDetector:
    def __init__(self, key_feats, pval=KS_PVALUE):
        self.key_feats = key_feats
        self.pval      = pval
        self.ref       = {}
        self.drift_log = []

    def fit_ref(self, X):
        for f in self.key_feats:
            if f in X.columns:
                self.ref[f] = X[f].dropna().values

    def detect(self, X):
        drifted = []
        for f in self.key_feats:
            if f not in X.columns or f not in self.ref:
                continue
            new = X[f].dropna().values
            if len(new) < 10 or len(self.ref[f]) < 10:
                continue
            _, pv = stats.ks_2samp(self.ref[f], new)
            if pv < self.pval:
                drifted.append(f)
        if drifted:
            self.drift_log.append({
                "time" : datetime.now().strftime("%Y-%m-%d"),
                "feats": drifted
            })
        return len(drifted) > 0, drifted


# ============================================================
# ModelMemory
# ============================================================
class ModelMemory:
    def __init__(self, max_snap=5):
        self.snaps      = []
        self.max_snap   = max_snap
        self.best_auc   = 0.0
        self.best_model = None
        self.rollbacks  = 0
        self.no_improve = 0

    def save(self, model, auc, meta=None):
        self.snaps.append({
            "auc"  : auc, "model": model, "meta": meta or {},
            "time" : datetime.now().strftime("%Y-%m-%d")
        })
        self.snaps.sort(key=lambda x: x["auc"], reverse=True)
        self.snaps = self.snaps[:self.max_snap]
        if auc > self.best_auc:
            self.best_auc   = auc
            self.best_model = model
            self.no_improve = 0
        else:
            self.no_improve += 1

    def should_rollback(self, new_auc):
        if self.no_improve >= 30:
            print(f"    [ModelMemory] 強制接受新模型（舊模型已 {self.no_improve} 次無改善）")
            self.no_improve = 0
            return False
        if self.best_model is None:
            return False
        if new_auc < (self.best_auc - ROLLBACK_TOLERANCE):
            self.rollbacks += 1
            return True
        return False

    def get_best(self):
        return self.best_model, self.best_auc


# ============================================================
# Feature 對齊工具
# （原版 Cell 5 與 Cell 5B 各定義一次 align_features/safe_predict，
#   內容幾乎相同；此處去重保留 Cell 5 的完整版本，
#   包含 Cell 5B 新增的 n_features_in_ 驗證邏輯）
# ============================================================
def align_features(X: pd.DataFrame, feat_cols: list, verbose: bool = False) -> pd.DataFrame:
    X_out = pd.DataFrame(index=X.index)
    for f in feat_cols:
        if f in X.columns:
            X_out[f] = pd.to_numeric(X[f], errors="coerce").values
        else:
            X_out[f] = 0.0
    X_out = (X_out.astype(np.float64).fillna(0.0)
             .replace([np.inf, -np.inf], 0.0))
    return X_out


def safe_predict(model, X: pd.DataFrame, feat_cols: list) -> float:
    try:
        X_aligned = align_features(X, feat_cols)

        # 驗證特徵數量
        if hasattr(model, "n_features_in_"):
            expected = model.n_features_in_
            actual   = X_aligned.shape[1]
            if expected != actual:
                print(f"  ❌ 特徵數不符: 模型需要{expected}, "
                      f"輸入{actual} → 截斷/補零修正")
                if actual > expected:
                    X_aligned = X_aligned.iloc[:, :expected]
                else:
                    for i in range(expected - actual):
                        X_aligned[f"_pad_{i}"] = 0.0

        prob = model.predict_proba(X_aligned)[:, 1]
        return float(np.clip(prob[0], 0.0, 1.0))

    except Exception as e:
        print(f"  ⚠️  safe_predict 失敗: {str(e)[:80]}")
        return 0.5


def safe_predict_batch(model, X: pd.DataFrame, feat_cols: list) -> np.ndarray:
    try:
        X_aligned = align_features(X, feat_cols)
        probs = model.predict_proba(X_aligned)[:, 1]
        return np.clip(probs, 0.0, 1.0)
    except Exception as e:
        print(f"  ⚠️  safe_predict_batch 失敗: {str(e)[:60]}")
        return np.full(len(X), 0.5)


def safe_eval(model, X: pd.DataFrame, y: pd.Series, feat_cols: list) -> dict:
    try:
        X_aligned = align_features(X, feat_cols)
        p    = model.predict_proba(X_aligned)[:, 1]
        pred = (p >= 0.5).astype(int)
        auc  = (roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5)
        acc  = accuracy_score(y, pred)
        return {"auc": auc, "accuracy": acc, "brier": brier_score_loss(y, p)}
    except Exception as e:
        print(f"  ⚠️  safe_eval 失敗: {str(e)[:50]}")
        return {"auc": 0.5, "accuracy": 0.5, "brier": 0.25}


def train_model_fixed(train_df: pd.DataFrame, feat_cols: list, val_pct: float = 0.85):
    if len(train_df) < 120:
        return None, 0.0
    vc    = int(len(train_df) * val_pct)
    X_tr  = align_features(train_df[feat_cols].iloc[:vc],  feat_cols)
    y_tr  = train_df["target"].iloc[:vc]
    X_val = align_features(train_df[feat_cols].iloc[vc:],  feat_cols)
    y_val = train_df["target"].iloc[vc:]
    if (len(y_tr.unique()) < 2 or len(y_val.unique()) < 2 or len(y_tr) < 60):
        return None, 0.0
    try:
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set  = [(X_val, y_val)],
            callbacks = [lgb.early_stopping(40, verbose=False),
                         lgb.log_evaluation(period=-1)]
        )
        cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        cal.fit(X_val, y_val)
        m = safe_eval(cal, X_val, y_val, feat_cols)
        return cal, m["auc"]
    except Exception as e:
        print(f"    ⚠️  訓練失敗: {str(e)[:50]}")
        return None, 0.0


# ============================================================
# ENH #4: Short Signal Multi-Confirmation
# ============================================================
def evaluate_short_confirmation(last_known: pd.DataFrame, regime: str, ens_prob: float) -> dict:
    score   = 0
    reasons = []

    if regime == "bear":
        score += 2
        reasons.append("HMM=bear")

    if "vix_chg" in last_known.columns:
        vix_chg = float(last_known["vix_chg"].iloc[0])
        if vix_chg > 0.5:
            score += 1
            reasons.append(f"VIX↑{vix_chg:.1f}")

    if "vol_surge_5" in last_known.columns:
        vol_surge = float(last_known["vol_surge_5"].iloc[0])
        if vol_surge < 0.85:
            score += 1
            reasons.append(f"量縮{vol_surge:.2f}x")
    elif "vol_5d" in last_known.columns and "vol_20d" in last_known.columns:
        v5  = float(last_known["vol_5d"].iloc[0])
        v20 = float(last_known["vol_20d"].iloc[0])
        if v20 > 0 and (v5 / v20) < 0.85:
            score += 1
            reasons.append(f"量縮{v5/v20:.2f}x")

    if "ret_5d" in last_known.columns:
        ret5 = float(last_known["ret_5d"].iloc[0])
        if ret5 < -0.01:
            score += 1
            reasons.append(f"5d動能{ret5:.2%}")

    if "ma5_20" in last_known.columns:
        if int(last_known["ma5_20"].iloc[0]) == 0:
            score += 1
            reasons.append("MA5<MA20")

    down_prob = 1.0 - ens_prob
    if down_prob >= 0.65:
        score += 2
        reasons.append(f"P(down)={down_prob:.2f}")

    return {"confirmed": score >= 3, "score": score, "reasons": reasons}


def get_market_trend(df: pd.DataFrame, window: int = 20) -> str:
    if len(df) < window:
        return "neutral"
    recent_ret = df["close"].pct_change().tail(window).mean()
    if recent_ret > 0.003:
        return "bull"
    elif recent_ret < -0.003:
        return "bear"
    return "neutral"


print("✅ models.py (Cell 5+5B) 載入完成")

