# interface.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from ta.momentum import RSIIndicator

from fx_symbols import DEFAULT_FX_PAIRS
from data_fetcher import fetch_series, fetch_df


# ============ USTAWIENIA UI / CONFIG ============

st.set_page_config(page_title="SEP Forex Signals", layout="wide")

# spróbuj doczytać config.json (opcjonalny)
def _load_pairs_from_config() -> List[str]:
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        pairs = cfg.get("pairs") or cfg.get("PAIRS")
        if isinstance(pairs, list) and pairs:
            return pairs
    except Exception:
        pass
    return DEFAULT_FX_PAIRS


PAIRS: List[str] = _load_pairs_from_config()


# ============ POMOCNICZE ============

def compute_rsi(series: pd.Series, window: int) -> float | None:
    """Liczy RSI; zwraca float albo None przy braku danych."""
    try:
        if series is None or series.empty:
            return None
        ser = series.astype("float64")
        rsi = RSIIndicator(close=ser, window=window).rsi()
        rsi = rsi.dropna()
        if rsi.empty:
            return None
        return float(rsi.iloc[-1])
    except Exception:
        return None


def build_signal(rsi_val: float | None, rsi_buy: float, rsi_sell: float) -> str:
    if rsi_val is None:
        return "—"
    if rsi_val <= rsi_buy:
        return "BUY"
    if rsi_val >= rsi_sell:
        return "SELL"
    return "—"


# ============ SIDEBAR ============

st.sidebar.header("Ustawienia strategii")
rsi_buy = st.sidebar.number_input("Próg RSI dla KUP (≤)", value=30, step=1)
rsi_sell = st.sidebar.number_input("Próg RSI dla SPRZEDAJ (≥)", value=70, step=1)
rsi_window = st.sidebar.number_input("Okres RSI", value=14, step=1, min_value=2)

# interwały wybieramy z listy – takie, które yfinance wspiera intraday
intervals = st.sidebar.multiselect(
    "Interwały",
    options=["15m", "30m", "60m"],
    default=["15m", "30m", "60m"],
)
only_active = st.sidebar.checkbox("Pokaż tylko wiersze z aktywnym sygnałem", value=False)

st.title("📈 SEP Forex Signals")
st.caption("Analiza RSI + wykresy (dane: Yahoo Finance)")

# ============ TABELA SYGNAŁÓW ============

def build_table(pairs: Iterable[str], intervals: Iterable[str]) -> pd.DataFrame:
    rows: List[Tuple[str, str, float | None, str, float | None, pd.Timestamp | None]] = []
    for sym in pairs:
        for itv in intervals:
            ser = fetch_series(sym, period="5d", interval=itv)
            last_px = float(ser.dropna().iloc[-1]) if not ser.empty else None
            last_ts = ser.dropna().index[-1] if not ser.empty else None
            rsi_val = compute_rsi(ser, window=int(rsi_window))
            sig = build_signal(rsi_val, rsi_buy, rsi_sell)
            rows.append((sym, itv, rsi_val, sig, last_px, last_ts))

    df = pd.DataFrame(
        rows,
        columns=["Symbol", "Interwał", "RSI", "Sygnał", "Cena", "Czas"],
    )
    return df


with st.spinner("Odświeżam dane..."):
    table_df = build_table(PAIRS, intervals)

if only_active:
    table_df = table_df[table_df["Sygnał"].isin(["BUY", "SELL"])]

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# ============ SZCZEGÓŁY (WYKRES) ============

st.subheader("Szczegóły")
sel = st.selectbox(
    "Wybierz wiersz do podglądu",
    options=[f"{r.Symbol} × {r.Interwał}" for r in table_df.itertuples()],
    index=0 if not table_df.empty else None,
)

if sel and not table_df.empty:
    # parsuj wybór
    sym, itv = sel.split(" × ", 1)

    # pobierz pełny DF (nie Series), żeby mieć OHLC na wykres/analizy
    df_full = fetch_df(sym, period="5d", interval=itv)

    if df_full.empty or "Close" not in df_full.columns:
        st.info("Brak danych do wykresu.")
    else:
        # przygotuj dane pod wykres liniowy
        ch_df = df_full[["Close"]].rename(columns={"Close": sym})
        ch_df.index.name = "time"

        st.line_chart(ch_df, use_container_width=True)

        # dodatkowo – aktualny RSI na tym interwale
        last_rsi = compute_rsi(df_full["Close"], window=int(rsi_window))
        met1, met2, met3 = st.columns(3)
        met1.metric("RSI", "—" if last_rsi is None else f"{last_rsi:.2f}")
        met2.metric("Cena", f"{float(df_full['Close'].iloc[-1]):.5f}")
        met3.metric("Ostatnia świeca", str(df_full.index[-1]))
else:
    st.info("Wybierz symbol z listy powyżej, aby zobaczyć szczegóły.")
