# ============================================================
# daily_run.py
# HSI Prediction - 核心流程：数据下载/训练/预测/发邮件
# ============================================================
import os
import sys
import joblib
import traceback
from datetime import datetime

# ---------------- 环境变量（从 GitHub Secrets 注入，不要硬编码） ----------------
FRED_API_KEY       = os.environ.get("FRED_API_KEY")
EMAIL_SENDER       = os.environ.get("EMAIL_SENDER", "")
EMAIL_PASSWORD     = os.environ.get("EMAIL_PASSWORD", "")
EMAIL_RECEIVER     = os.environ.get("EMAIL_RECEIVER", "")
DASHSCOPE_API_KEY  = os.environ.get("DASHSCOPE_API_KEY")

if not FRED_API_KEY:
    print("警告: 未设置 FRED_API_KEY，相关数据源可能无法使用。")

STATE_PATH = "state/pipeline_state.pkl"
os.makedirs("state", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ---------------- 模型/预测用的状态 ----------------
STATE_KEYS = [
    "df_all", "master", "global_model", "regime_models", "feat_cols",
    "hmm", "monitor", "memory", "drift",
    "threshold_calibrator", "position_sizer", "ensemble_weighter",
    "df_bt", "direction_acc",
    "hl_models", "hl_feat_cols",
    "STOCK_LIST", "SECTOR_MAP",
    "stock_data", "stock_feat_data", "stock_models",
    "stock_feat_cols_per", "stock_train_scores",
    "panel_cal_models", "stock_feat_cols",
]

# ---------------- charts.py 专用的中间变量，一并持久化 ----------------
CHART_STATE_KEYS = [
    "sim_df", "retrain_df", "reason_counts", "perf_df",
    "hl_sim_df", "perf_log",
    "long_df_all", "short_df_all", "watch_df_all", "pred_df",
    "monthly",
]

ALL_STATE_KEYS = STATE_KEYS + CHART_STATE_KEYS

def save_state():
    state = {k: globals()[k] for k in ALL_STATE_KEYS if k in globals()}
    joblib.dump(state, STATE_PATH)
    missing = [k for k in ALL_STATE_KEYS if k not in globals()]
    if missing:
        print(f"[{datetime.now()}] 警告，以下变量未找到，未被保存: {missing}")
    print(f"[{datetime.now()}] State 已保存 -> {STATE_PATH} ({len(state)} 项)")

def load_state():
    state = joblib.load(STATE_PATH)
    globals().update(state)
    print(f"[{datetime.now()}] State 已载入 <- {STATE_PATH} ({len(state)} 项)")


# ============================================================
# 阶段一：首次运行 = 全量训练；之后运行 = 直接载入已训练好的状态
# ============================================================
if os.path.exists(STATE_PATH):
    load_state()
else:
    print(f"[{datetime.now()}] 未发现已保存状态，开始首次全量训练...")

    # >>>>>>>>>>>>>>>> 在此处粘贴原脚本以下 Cell 的真实代码 <<<<<<<<<<<<<<<<
    #   Cell 1   : 安装/导入依赖、全局配置
    #   Cell 2   : FRED_API_KEY 配置、threshold_calibrator /
    #              position_sizer / ensemble_weighter 初始化
    #   Cell 2B  : 补充配置
    #   Cell 3/3B: 数据下载 (df_all 雏形)
    #   Cell 4   : 特征工程 (df_all, master)
    #   Cell 5/5B: 补充特征
    #   Cell 6   : HMM / LightGBM 训练
    #              -> global_model, regime_models, feat_cols,
    #                 hmm, monitor, memory, drift
    #   Cell 7   : 回测主逻辑 -> df_bt, sim_df, retrain_df, reason_counts
    #   Cell 8/8B: 回测补充 -> direction_acc, perf_df
    #   Cell 17  : 高低位模型训练 -> hl_models, hl_feat_cols
    #   Cell 16B/18: 高低位回测 -> hl_sim_df, perf_log
    #   Cell 21  : STOCK_LIST, SECTOR_MAP
    #   Cell 22~26: 个股数据/特征/模型训练
    #              -> stock_data, stock_feat_data, stock_models,
    #                 stock_feat_cols_per, stock_train_scores,
    #                 panel_cal_models, stock_feat_cols
    #   Cell 27  : 股票面板补充训练
    #   Cell 28  : 生成 long_df_all, short_df_all, watch_df_all, pred_df
    #
    # 注意：删除原文件中以下行（Colab 专属 / 密钥泄露）：
    #   from google.colab import userdata
    #   DASHSCOPE_API_KEY = "sk-ws-H.DMPMLMP..."（硬编码明文，务必删除并重新生成新 key）
    # >>>>>>>>>>>>>>>> 粘贴结束 <<<<<<<<<<<<<<<<

    save_state()


# ============================================================
# 阶段二：每日更新 + 预测 + 发邮件（对应 Cell 12/13/14/15/29B/30）
# ============================================================
def run_daily():
    try:
        # >>> 粘贴 Cell 12 内容（预测前置准备，如有） <<<

        # >>> 粘贴 Cell 13 的 daily_update() 定义与调用 <<<
        daily_update()

        # >>> 粘贴 Cell 14/15 内容（如涉及 monthly 等变量的更新） <<<

        # >>> 粘贴 Cell 29B 的 daily_stock_update() 定义与调用 <<<
        stock_today = daily_stock_update()

        # >>> 粘贴 Cell 30 的 build_email_html() + smtplib 发送逻辑 <<<

        print(f"[{datetime.now()}] 每日预测与邮件发送完成。")

    except Exception as e:
        print(f"[{datetime.now()}] 每日运行出错: {e}")
        traceback.print_exc()
        sys.exit(1)

    finally:
        # 无论成功失败，更新后的状态（含滚动校准器、图表用中间变量）都要存回去
        save_state()


if __name__ == "__main__":
    run_daily()
