# ============================================================
# src/adaptive.py
# [RECONSTRUCTED] — ThresholdCalibrator / PositionSizer /
# EnsembleWeighter were referenced throughout Cell 7 but their
# class definitions were not present in the source provided.
#
# Rebuilt strictly from observed call signatures:
#   threshold_calibrator.calibrate(df, prob_col, target_col, regime_col)
#   threshold_calibrator.get_signal(prob, regime) -> {1, 0, -1}
#   position_sizer.compute_size(prob, signal, regime, roll_acc) -> float
#   ensemble_weighter.update(source, prob, actual)
#   ensemble_weighter.ensemble(g_prob, r_prob) -> float
#   ensemble_weighter.weight_summary() -> dict
#
# Confirm against your original notebook cell if available.
# ============================================================
import numpy as np

from src.config import (
    PROB_LONG, PROB_SHORT, KELLY_FRACTION,
    MAX_POSITION, MIN_POSITION, CALIBRATION_MIN_SAMPLES,
)


# ============================================================
# ThresholdCalibrator — per-regime dynamic probability thresholds
# ============================================================
class ThresholdCalibrator:
    """
    Searches a grid of candidate long/short probability thresholds
    per HMM regime, picking the pair that maximizes realized hit-rate
    on recent history while keeping a minimum trade frequency (to
    avoid collapsing to "never trade" as a degenerate optimum).

    Falls back to static PROB_LONG / PROB_SHORT (from config) for any
    regime that hasn't been calibrated yet or has too few samples.
    """

    def __init__(self, grid_step: float = 0.02, min_trades: int = 10):
        self.grid_step = grid_step
        self.min_trades = min_trades
        self.regime_thresholds: dict = {}   # {regime: {"long": x, "short": y}}
        self.calibration_log: list = []

    def calibrate(self, df, prob_col: str, target_col: str, regime_col: str):
        if len(df) < CALIBRATION_MIN_SAMPLES:
            return

        for regime in df[regime_col].dropna().unique():
            rdf = df[df[regime_col] == regime]
            if len(rdf) < self.min_trades:
                continue

            probs   = rdf[prob_col].values
            targets = rdf[target_col].values

            best_long  = PROB_LONG
            best_short = PROB_SHORT
            best_score = -np.inf

            long_grid  = np.arange(0.50, 0.85 + 1e-9, self.grid_step)
            short_grid = np.arange(0.15, 0.50 + 1e-9, self.grid_step)

            for lt in long_grid:
                long_sig = probs >= lt
                if long_sig.sum() < self.min_trades:
                    continue
                long_hit = targets[long_sig].mean()

                for st in short_grid:
                    short_sig = probs <= st
                    if short_sig.sum() < self.min_trades:
                        continue
                    short_hit = 1 - targets[short_sig].mean()

                    n_total = long_sig.sum() + short_sig.sum()
                    score   = (
                        (long_hit * long_sig.sum() + short_hit * short_sig.sum())
                        / n_total
                    )

                    if score > best_score:
                        best_score = score
                        best_long  = float(lt)
                        best_short = float(st)

            self.regime_thresholds[regime] = {
                "long": best_long, "short": best_short, "score": best_score,
            }
            self.calibration_log.append({
                "regime": regime, "long": best_long,
                "short": best_short, "score": round(best_score, 4),
            })

    def get_signal(self, prob: float, regime: str) -> int:
        th = self.regime_thresholds.get(
            regime, {"long": PROB_LONG, "short": PROB_SHORT}
        )
        if prob >= th["long"]:
            return 1
        elif prob <= th["short"]:
            return -1
        return 0


# ============================================================
# PositionSizer — half-Kelly sizing scaled by rolling confidence
# ============================================================
class PositionSizer:
    """
    Classic Kelly fraction f* = p - (1-p)/b, with b=1 (even-money
    approximation since payoff is the raw return, not fixed odds).
    Scaled by KELLY_FRACTION (half-Kelly, for variance damping) and
    further scaled by recent rolling accuracy as a confidence factor.
    """

    def __init__(self, kelly_fraction: float = KELLY_FRACTION,
                 max_position: float = MAX_POSITION,
                 min_position: float = MIN_POSITION):
        self.kelly_fraction = kelly_fraction
        self.max_position   = max_position
        self.min_position   = min_position

    def compute_size(self, prob: float, signal: int,
                      regime: str, roll_acc: float = 0.5) -> float:
        if signal == 0:
            return 0.0

        edge = abs(prob - 0.5) * 2          # map [0.5,1.0] -> [0,1.0] edge strength
        kelly = max(0.0, edge) * self.kelly_fraction

        confidence = np.clip((roll_acc - 0.45) / 0.25, 0.0, 1.5)

        regime_mult = {"bull": 1.0, "neutral": 0.8, "bear": 0.6}.get(regime, 0.8)

        size = kelly * confidence * regime_mult
        return float(np.clip(size, self.min_position, self.max_position))


# ============================================================
# EnsembleWeighter — adaptive accuracy-weighted model blending
# ============================================================
class EnsembleWeighter:
    """
    Maintains an exponential moving average (EMA) of hit-rate for
    each sub-model source ("global", "regime"). Ensemble weight for
    each source is proportional to its recent EMA accuracy, so the
    better-performing model gets more influence over time.
    """

    def __init__(self, ema_alpha: float = 0.05):
        self.ema_alpha = ema_alpha
        self.acc = {"global": 0.55, "regime": 0.55}   # seeded slightly > 0.5

    def update(self, source: str, prob: float, actual: int):
        if source not in self.acc:
            self.acc[source] = 0.55
        correct = 1.0 if (prob >= 0.5) == bool(actual) else 0.0
        self.acc[source] = (
            (1 - self.ema_alpha) * self.acc[source] + self.ema_alpha * correct
        )

    def ensemble(self, g_prob: float, r_prob: float) -> float:
        w_g = max(self.acc["global"] - 0.45, 0.01)
        w_r = max(self.acc["regime"] - 0.45, 0.01)
        total = w_g + w_r
        return float((g_prob * w_g + r_prob * w_r) / total)

    def weight_summary(self) -> dict:
        w_g = max(self.acc["global"] - 0.45, 0.01)
        w_r = max(self.acc["regime"] - 0.45, 0.01)
        total = w_g + w_r
        return {
            "global": round(w_g / total, 3),
            "regime": round(w_r / total, 3),
        }
