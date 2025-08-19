# data_fetcher.py
from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_df(symbol: str, period: str = "5d", interval: str = "60m") -> pd.DataFrame:
    """
    Pobiera dane OHLC z Yahoo Finance jako DataFrame (index = Datetime).
    Zwraca zawsze DataFrame (może być pusty).
    """
    # yfinance od 0.2.4 ma auto_adjust domyślnie True – ustawiamy jawnie:
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
    except Exception:
        df = pd.DataFrame()

    # Uporządkuj indeks
    if not df.empty:
        if not isinstance(df.index, pd.DatetimeIndex):
            # sporadycznie przy błędach zwraca RangeIndex
            df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index()
    return df


def fetch_series(symbol: str, period: str = "5d", interval: str = "60m") -> pd.Series:
    """
    Zwraca Series z kolumny Close dla danego symbolu.
    """
    df = fetch_df(symbol, period=period, interval=interval)
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype="float64")
    s = df["Close"].astype("float64")
    s.name = "Close"
    return s
