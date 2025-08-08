# candle_patterns.py
"""
Wykrywanie formacji świecowych.
Domyślnie używa TA‑Lib. Jeśli nie jest dostępna, spada do prostych heurystyk.
Zwraca: (pattern_name, pattern_signal) np. ("Engulfing", "BUY") lub ("None", "HOLD")
"""

from typing import Tuple
import pandas as pd

try:
    import talib # wymaga ta-lib
    _HAS_TALIB = True
except Exception:
    _HAS_TALIB = False


def _heuristic_patterns(df: pd.DataFrame) -> Tuple[str, str]:
    # Proste heurystyki dla ostatniej świecy
    o = df["Open"].iloc[-2]
    h = df["High"].iloc[-2]
    l = df["Low"].iloc[-2]
    c = df["Close"].iloc[-2]
    body = abs(c - o)
    rng = max(h - l, 1e-9)
    upper = h - max(c, o)
    lower = min(c, o) - l

    # Doji
    if body <= 0.1 * rng:
        return "Doji", "HOLD"

    # Hammer / Shooting Star
    if body < 0.35 * rng:
        if lower > 2 * body and upper < body:
            return "Hammer", "BUY"
        if upper > 2 * body and lower < body:
            return "Shooting Star", "SELL"

    # Engulfing
    o1, c1 = df["Open"].iloc[-3], df["Close"].iloc[-3]
    if c > o and c1 < o1 and c >= o1 and o <= c1:
        return "Engulfing", "BUY"
    if c < o and c1 > o1 and c <= o1 and o >= c1:
        return "Engulfing", "SELL"

    return "None", "HOLD"


def detect_candle_patterns(df: pd.DataFrame) -> Tuple[str, str]:
    if len(df) < 5:
        return "None", "HOLD"

    if not _HAS_TALIB:
        return _heuristic_patterns(df)

    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    patterns = [
        ("Hammer", talib.CDLHAMMER),
        ("Inverted Hammer", talib.CDLINVERTEDHAMMER),
        ("Engulfing", talib.CDLENGULFING),
        ("Doji", talib.CDLDOJI),
        ("Morning Star", talib.CDLMORNINGSTAR),
        ("Evening Star", talib.CDLEVENINGSTAR),
        ("Shooting Star", talib.CDLSHOOTINGSTAR),
        ("Harami", talib.CDLHARAMI),
    ]

    for name, func in patterns:
        arr = func(o, h, l, c)
        val = int(arr.iloc[-1])
        if val != 0:
            return name, ("BUY" if val > 0 else "SELL")

    return "None", "HOLD"

