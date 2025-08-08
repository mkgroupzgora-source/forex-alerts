import yfinance as yf
import pandas as pd

def fetch_data(symbol: str) -> pd.DataFrame:
    pair = symbol.replace("USD", "=X") if symbol.endswith("USD") and len(symbol) == 6 else symbol
    df = yf.download(pair, period="5d", interval="15m")
    df.dropna(inplace=True)
    df = df.rename(columns={
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume"
    })
    return df
