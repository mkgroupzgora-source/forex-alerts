import pandas as pd

def calculate_rsi(data, period: int = 14):
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def check_rsi_signal(df, rsi_buy_threshold=30, rsi_sell_threshold=70):
    df = df.copy()
    df["RSI"] = calculate_rsi(df)
    last_rsi = df["RSI"].dropna().iloc[-1]

    if last_rsi < rsi_buy_threshold:
        return "BUY", last_rsi
    elif last_rsi > rsi_sell_threshold:
        return "SELL", last_rsi
    else:
        return "HOLD", last_rsi
