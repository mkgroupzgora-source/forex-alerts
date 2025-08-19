# fx_symbols.py
from __future__ import annotations
import yfinance as yf
import pandas as pd

# Preferowane tickery (1. próbujemy SPOT =X)
PRIMARY = {
    "XAUUSD": "XAUUSD=X",
    "XAGUSD": "XAGUSD=X",
    "XPTUSD": "XPTUSD=X",
    "XPDUSD": "XPDUSD=X",
}

# Fallbacki (kontrakty futures na Yahoo)
FALLBACK = {
    "XAUUSD": "GC=F",   # Gold futures
    "XAGUSD": "SI=F",   # Silver futures
    "XPTUSD": "PL=F",   # Platinum futures
    "XPDUSD": "PA=F",   # Palladium futures
}

def to_yf_candidates(pair: str) -> list[str]:
    """
    Zwraca listę kandydatów tickerów Yahoo dla pary/metalu.
    Dla zwykłych par FX: tylko 'XXXYYY=X'.
    Dla metali: najpierw SPOT (=X), potem futures (np. GC=F).
    """
    p = pair.upper().replace("-", "").replace("_", "")
    if p in PRIMARY:
        return [PRIMARY[p], FALLBACK[p]]
    # zwykłe FX
    return [f"{p}=X"]

def fetch_series(pair: str, period: str = "7d", interval: str = "60m") -> pd.Series:
    """
    Pobiera serię Close (lub Adj Close) z Yahoo, z fallbackiem na alternatywne tickery.
    """
    for sym in to_yf_candidates(pair):
        df = yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            close_col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
            if close_col is None or df[close_col].dropna().empty:
                continue
            return df[close_col].dropna()
    raise ValueError(f"No data from Yahoo for {pair} (tried: {to_yf_candidates(pair)})")
