"""
candle_patterns.py
Wykrywanie formacji świecowych bez TA‑Lib (zgodność ze Streamlit Cloud).

Obsługiwane formacje:
- Hammer (Młotek)
- Shooting Star (Spadająca Gwiazda)
- Bullish Engulfing (Objęcie Hossy)
- Bearish Engulfing (Objęcie Bessy)
- Piercing Pattern (Przenikanie)
- Dark Cloud Cover (Zasłona Ciemnej Chmury)
- Morning Star (Gwiazda Poranna)
- Evening Star (Gwiazda Wieczorna)

Wejście: DataFrame z kolumnami: ["Open", "High", "Low", "Close"]
Zwracane:
- funkcje bool dla pojedynczej świecy/formacji
- funkcja scan_patterns(df) → Series z nazwą formacji lub "" dla każdej świecy
- funkcja last_pattern(df) → nazwa ostatniej rozpoznanej formacji lub None

Uwaga:
Reguły są pragmaticzne (literatura różni się progami). Dla stabilności używamy
procentowych progów i relacji korpus/cienie.
"""

from __future__ import annotations
import pandas as pd
from typing import Optional

# ---------- Pomocnicze obliczenia na świecach ----------

def _body(o: pd.Series, c: pd.Series) -> pd.Series:
    return (c - o).abs()

def _is_bull(o: pd.Series, c: pd.Series) -> pd.Series:
    return c > o

def _is_bear(o: pd.Series, c: pd.Series) -> pd.Series:
    return c < o

def _upper_shadow(o: pd.Series, h: pd.Series, c: pd.Series) -> pd.Series:
    # dystans od wyższej z (O,C) do High
    return h - pd.concat([o, c], axis=1).max(axis=1)

def _lower_shadow(o: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    # dystans od Low do niższej z (O,C)
    return pd.concat([o, c], axis=1).min(axis=1) - l

def _total_range(h: pd.Series, l: pd.Series) -> pd.Series:
    return (h - l).replace(0, 1e-9)  # unikamy zero-division


# ---------- Jednoświecowe ----------

def is_hammer(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """Młotek: mały korpus u góry zakresu, długi dolny cień."""
    body = _body(o, c)
    rng = _total_range(h, l)
    up = _upper_shadow(o, h, c)
    low = _lower_shadow(o, l, c)

    # Kryteria:
    # 1) Korpus <= 30% całego zakresu
    # 2) Dolny cień >= 2 * korpus
    # 3) Górny cień niewielki (<= 20% zakresu)
    cond = (body <= 0.3 * rng) & (low >= 2.0 * body) & (up <= 0.2 * rng)
    return cond

def is_shooting_star(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """Spadająca Gwiazda: mały korpus u dołu zakresu, długi górny cień."""
    body = _body(o, c)
    rng = _total_range(h, l)
    up = _upper_shadow(o, h, c)
    low = _lower_shadow(o, l, c)

    cond = (body <= 0.3 * rng) & (up >= 2.0 * body) & (low <= 0.2 * rng)
    return cond


# ---------- Dwóch świec ----------

def is_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Objęcie Hossy: pierwsza świeca spadkowa, druga wzrostowa,
    korpus drugiej obejmuje korpus pierwszej."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    prev_o, prev_c = o.shift(1), c.shift(1)

    cond = (
        _is_bear(prev_o, prev_c) &
        _is_bull(o, c) &
        (o <= prev_c) &
        (c >= prev_o)
    )
    return cond.fillna(False)

def is_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Objęcie Bessy: pierwsza świeca wzrostowa, druga spadkowa,
    korpus drugiej obejmuje korpus pierwszej."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    prev_o, prev_c = o.shift(1), c.shift(1)

    cond = (
        _is_bull(prev_o, prev_c) &
        _is_bear(o, c) &
        (o >= prev_c) &
        (c <= prev_o)
    )
    return cond.fillna(False)

def is_piercing(df: pd.DataFrame) -> pd.Series:
    """Przenikanie: świeca 1 spadkowa, świeca 2 wzrostowa.
    Otwarcie 2 poniżej minimum korpusu 1, zamknięcie 2 > 50% korpusu 1."""
    o, c = df["Open"], df["Close"]
    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_body = (prev_o - prev_c).abs()

    cond = (
        _is_bear(prev_o, prev_c) &
        _is_bull(o, c) &
        (o < prev_c) &
        (c >= prev_c + 0.5 * prev_body)
    )
    return cond.fillna(False)

def is_dark_cloud(df: pd.DataFrame) -> pd.Series:
    """Zasłona Ciemnej Chmury: świeca 1 wzrostowa, świeca 2 spadkowa.
    Otwarcie 2 powyżej maks. korpusu 1, zamknięcie 2 < 50% korpusu 1."""
    o, c = df["Open"], df["Close"]
    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_body = (prev_c - prev_o).abs()

    cond = (
        _is_bull(prev_o, prev_c) &
        _is_bear(o, c) &
        (o > prev_c) &
        (c <= prev_o + 0.5 * prev_body)
    )
    return cond.fillna(False)


# ---------- Trzech świec ----------

def is_morning_star(df: pd.DataFrame) -> pd.Series:
    """Gwiazda Poranna: spadkowa świeca, mały korpus/doji, potem silna wzrostowa,
    która zamyka powyżej połowy korpusu świecy 1."""
    o, c = df["Open"], df["Close"]
    b0 = _body(o.shift(2), c.shift(2))
    b1 = _body(o.shift(1), c.shift(1))
    b2 = _body(o, c)

    cond = (
        _is_bear(o.shift(2), c.shift(2)) &                      # świeca 1 spadkowa
        (b1 <= b0 * 0.6) &                                      # świeca 2 relatywnie mała
        _is_bull(o, c) &                                        # świeca 3 wzrostowa
        (c >= (o.shift(2) + c.shift(2)) / 2)                    # zamknięcie 3 > 50% korpusu 1
    )
    return cond.fillna(False)

def is_evening_star(df: pd.DataFrame) -> pd.Series:
    """Gwiazda Wieczorna: wzrostowa świeca, mały korpus/doji, potem silna spadkowa,
    która zamyka poniżej połowy korpusu świecy 1."""
    o, c = df["Open"], df["Close"]
    b0 = _body(o.shift(2), c.shift(2))
    b1 = _body(o.shift(1), c.shift(1))
    b2 = _body(o, c)

    cond = (
        _is_bull(o.shift(2), c.shift(2)) &                      # świeca 1 wzrostowa
        (b1 <= b0 * 0.6) &                                      # świeca 2 relatywnie mała
        _is_bear(o, c) &                                        # świeca 3 spadkowa
        (c <= (o.shift(2) + c.shift(2)) / 2)                    # zamknięcie 3 < 50% korpusu 1
    )
    return cond.fillna(False)


# ---------- Skanowanie i wynik ----------

PATTERN_LABELS = [
    ("Hammer", is_hammer),
    ("Shooting Star", is_shooting_star),
]

# Funkcje zależne od >=2 świec operują na całym df:
def _two_three_candle_wrappers(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "Bullish Engulfing": is_bullish_engulfing(df),
        "Bearish Engulfing": is_bearish_engulfing(df),
        "Piercing": is_piercing(df),
        "Dark Cloud Cover": is_dark_cloud(df),
        "Morning Star": is_morning_star(df),
        "Evening Star": is_evening_star(df),
    }

def scan_patterns(df: pd.DataFrame) -> pd.Series:
    """
    Zwraca Series (index = df.index) z nazwą formacji lub "".
    Jeśli w danej świecy spełnia się kilka – łączymy nazwy przecinkiem.
    """
    if df is None or df.empty:
        return pd.Series([], dtype=str)

    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    out = pd.Series([""] * len(df), index=df.index, dtype=object)

    # 1-świecowe
    for label, fn in PATTERN_LABELS:
        mask = fn(o, h, l, c)
        out[mask] = out[mask].apply(lambda s: f"{s}, {label}" if s else label)

    # 2–3 świecowe
    multi = _two_three_candle_wrappers(df)
    for label, mask in multi.items():
        out[mask] = out[mask].apply(lambda s: f"{s}, {label}" if s else label)

    # oczyszczanie ewentualnych spacji/komów na początku
    out = out.str.strip(", ").fillna("")
    return out

def last_pattern(df: pd.DataFrame) -> Optional[str]:
    """
    Zwraca nazwę formacji dla ostatniej świecy (lub None).
    """
    if df is None or df.empty:
        return None
    labels = scan_patterns(df)
    val = labels.iloc[-1]
    return val if isinstance(val, str) and len(val) > 0 else None
