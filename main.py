import time
from strategy import check_rsi_signal
from notifier import send_console_alert, send_email_alert
import yfinance as yf
import json

with open("config.json") as f:
    config = json.load(f)

PAIRS = config["pairs"]
RSI_BUY = config["rsi_buy_threshold"]
RSI_SELL = config["rsi_sell_threshold"]
INTERVAL = config["interval_minutes"] * 60
EMAIL = config["recipient_email"]

def run():
    while True:
        for pair in PAIRS:
            print(f"\n[INFO] Checking RSI signal for {pair}...")

            try:
                df = yf.download(pair, period="7d", interval="1h", progress=False)
            except Exception as e:
                print(f"[ERROR] Failed to download data for {pair}: {e}")
                continue

            if df is not None and not df.empty:
                try:
                    signal, rsi = check_rsi_signal(df, RSI_BUY, RSI_SELL)

                    if signal != "HOLD":
                        msg = f"{pair} | RSI: {rsi:.2f} → SIGNAL: {signal}"
                        send_console_alert(msg)
                        send_email_alert(EMAIL, f"Forex Alert for {pair}", msg)
                    else:
                        print(f"[INFO] {pair} RSI: {rsi:.2f} – No signal.")
                except Exception as e:
                    print(f"[ERROR] Failed to process data for {pair}: {e}")
            else:
                print(f"[ERROR] No data returned for {pair}, skipping.")

        print(f"[INFO] Waiting {INTERVAL / 60} minutes...\n")
        time.sleep(INTERVAL)

if __name__ == "__main__":
    run()
