"""
src/send_report.py
-------------------
daily.yml 的「Send email report」step 進入點（python -m src.send_report）。

職責:
  1. 讀取 daily_pipeline.py 產出的最新預測   -> results/latest_prediction.pkl
  2. 讀取 pipeline_state.pkl 內的最新回測報告 -> state["last_backtest_report"]
  3. 讀取個股排行榜（可選）                  -> results/stock_board.csv
  4. 呼叫 email_report.send_email(prediction, report, stock_board) 寄出郵件

設計原則:
  - 任一檔案缺失/讀取失敗時，不應讓整個 GitHub Actions job 無聲失敗，
    而是印出清楚的錯誤訊息並以非 0 exit code 結束。
  - 個股排行榜屬於可選內容，缺失時郵件仍正常寄出（不回傳錯誤）。
"""

import os
import sys

import joblib
import pandas as pd

from src.config import RESULTS_DIR, STATE_PATH
from src.email_report import send_email


PREDICTION_PATH   = os.path.join(RESULTS_DIR, "latest_prediction.pkl")
STOCK_BOARD_PATH  = os.path.join(RESULTS_DIR, "stock_board.csv")


def load_prediction() -> dict:
    if not os.path.exists(PREDICTION_PATH):
        raise FileNotFoundError(
            f"找不到預測檔案:{PREDICTION_PATH}\n"
            f"  -> 請確認 daily_pipeline.py 是否已成功執行並呼叫 predict_next_day()。"
        )
    prediction = joblib.load(PREDICTION_PATH)
    if not isinstance(prediction, dict):
        raise TypeError(f"{PREDICTION_PATH} 內容格式不正確，預期 dict，實際為 {type(prediction)}")
    return prediction


def load_backtest_report() -> dict:
    if not os.path.exists(STATE_PATH):
        raise FileNotFoundError(
            f"找不到狀態檔案:{STATE_PATH}\n"
            f"  -> 請確認 daily_pipeline.py 是否已成功執行並呼叫 save_pipeline_state()。"
        )
    state = joblib.load(STATE_PATH)
    report = state.get("last_backtest_report")
    if report is None:
        print("  ⚠️  state 中找不到 last_backtest_report，將以空白報告寄送郵件。")
        report = {}
    return report


def load_stock_board():
    """個股排行榜為可選內容，缺失時回傳 None，不影響郵件主體發送。"""
    if not os.path.exists(STOCK_BOARD_PATH):
        print("  ℹ️  找不到個股排行榜檔案，郵件將不含該區塊。")
        return None
    try:
        return pd.read_csv(STOCK_BOARD_PATH)
    except Exception as e:
        print(f"  ⚠️  個股排行榜讀取失敗，郵件將不含該區塊: {e}")
        return None


def main() -> int:
    try:
        prediction = load_prediction()
        report     = load_backtest_report()
    except Exception as e:
        print(f"  ❌ 讀取預測/報告失敗:{e}")
        return 1

    stock_board = load_stock_board()

    try:
        send_email(prediction, report, stock_board)
    except Exception as e:
        print(f"  ❌ 郵件發送失敗:{e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
