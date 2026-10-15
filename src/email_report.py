"""
src/email_report.py
--------------------
產生郵件 HTML 內容並透過 SMTP 發送。
憑證一律從環境變數讀取，不寫死於程式碼。

新增（決策3）：
  - _build_stock_board_html()：個股排行榜表格區塊
  - send_email()/_build_html_body() 新增可選參數 stock_board=None，
    預設值確保向後相容（缺失時郵件仍正常寄出，只是少排行榜區塊）。
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


CHART_FILES = [
    "equity_curve.png",
    "drawdown.png",
    "regime_timeline.png",
    "prediction_confidence.png",
    "monthly_returns.png",
    "hit_rate_by_regime.png",
    "ensemble_weights.png",
]


def _build_stock_board_html(stock_board) -> str:
    """個股排行榜 HTML 表格，stock_board 為 None 或空時回傳空字串（不影響原郵件版面）。"""
    if stock_board is None or len(stock_board) == 0:
        return ""

    top_n = 5
    longs  = stock_board.head(top_n)
    shorts = stock_board.tail(top_n).sort_values("prob")

    def _rows(df):
        out = ""
        for _, r in df.iterrows():
            out += (
                f"<tr><td>{r['ticker']}</td><td>{r['name']}</td>"
                f"<td>{r['sector']}</td><td>{r['prob']:.3f}</td>"
                f"<td>{r['signal']}</td></tr>"
            )
        return out

    return f"""
    <h3>個股排行榜</h3>
    <p>恆指方向參考：{stock_board['hsi_direction'].iloc[0]}
       （信心 {stock_board['hsi_confidence'].iloc[0]:.3f}）</p>
    <h4>多頭 Top {top_n}</h4>
    <table border="1" cellpadding="4" cellspacing="0">
      <tr><th>代碼</th><th>名稱</th><th>板塊</th><th>機率</th><th>訊號</th></tr>
      {_rows(longs)}
    </table>
    <h4>空頭 Top {top_n}</h4>
    <table border="1" cellpadding="4" cellspacing="0">
      <tr><th>代碼</th><th>名稱</th><th>板塊</th><th>機率</th><th>訊號</th></tr>
      {_rows(shorts)}
    </table>
    """


def _build_html_body(prediction: dict, report: dict,
                      stock_board=None) -> str:
    """生成郵件 HTML 內容，涵蓋預測結果、回測摘要、個股排行榜（可選）。"""
    signal = prediction.get("signal", "N/A")
    regime = prediction.get("regime", "N/A")
    prob   = prediction.get("ensemble_prob", 0.0)
    pos    = prediction.get("position_size", 0.0)
    as_of  = prediction.get("as_of_date", "N/A")

    sharpe   = report.get("sharpe", "N/A")
    max_dd   = report.get("max_drawdown", "N/A")
    hit_rate = report.get("hit_rate", "N/A")

    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <h2>HSI 每日預測報告 — {as_of}</h2>
        <h3>明日預測</h3>
        <table border="1" cellpadding="6" cellspacing="0">
            <tr><td>訊號</td><td><b>{signal}</b></td></tr>
            <tr><td>市場狀態 (Regime)</td><td>{regime}</td></tr>
            <tr><td>集成機率</td><td>{prob:.4f}</td></tr>
            <tr><td>建議倉位</td><td>{pos:.2%}</td></tr>
        </table>

        <h3>回測摘要</h3>
        <table border="1" cellpadding="6" cellspacing="0">
            <tr><td>Sharpe</td><td>{sharpe}</td></tr>
            <tr><td>最大回撤</td><td>{max_dd}</td></tr>
            <tr><td>命中率</td><td>{hit_rate}</td></tr>
        </table>
    """

    html += _build_stock_board_html(stock_board)

    html += """
        <h3>圖表附件</h3>
        <p>equity_curve / drawdown / regime_timeline / prediction_confidence /
           monthly_returns / hit_rate_by_regime / ensemble_weights</p>
    </body>
    </html>
    """
    return html


def send_email(prediction: dict, report: dict, stock_board=None) -> None:
    """
    從環境變數讀取 SMTP 憑證並寄出郵件。
    stock_board=None 時郵件仍正常寄出（向後相容），只是少排行榜區塊。
    """
    smtp_server = os.environ.get("SMTP_SERVER")
    smtp_port   = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user   = os.environ.get("SMTP_USER")
    smtp_pass   = os.environ.get("SMTP_PASS")
    mail_to     = os.environ.get("MAIL_TO")

    if not all([smtp_server, smtp_user, smtp_pass, mail_to]):
        raise EnvironmentError(
            "缺少必要的 SMTP 環境變數 (SMTP_SERVER/SMTP_USER/SMTP_PASS/MAIL_TO)"
        )

    msg = MIMEMultipart("related")
    msg["Subject"] = f"HSI 每日預測報告 — {prediction.get('as_of_date', 'N/A')}"
    msg["From"] = smtp_user
    msg["To"] = mail_to

    html_body = _build_html_body(prediction, report, stock_board)
    msg.attach(MIMEText(html_body, "html"))

    for fname in CHART_FILES:
        path = os.path.join("results", "charts", fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                img = MIMEImage(f.read())
                img.add_header("Content-Disposition", "attachment", filename=fname)
                msg.attach(img)
        else:
            print(f"  ⚠️  圖表檔案缺失，跳過附件: {path}")

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)

    print(f"  ✅ 郵件已發送至 {mail_to}")
