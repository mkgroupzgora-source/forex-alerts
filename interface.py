# interface.py
# SEP Forex Signals — Streamlit
# (pełna wersja z poprawnym pobieraniem kolumny Close i liczeniem RSI)

from __future__ import annotations

import math
from datetime import timedelta

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st


# ------------------------- USTAWIENIA APLIKACJI -------------------------

st.set_page_config(page_title="SEP Forex Signals", layout="wide")
st.markdown(
    """
    <style>
        .css-1v0mbdj, .block-container { padding-top: 1.2rem; }
        .stMetric { text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Domyślna lista instrumentów (możesz dopasować do swoich)
SYMBOLS = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "USDCHF=X",
    "USDCAD=X",
    # "EURPLN=X",
    # "USDPLN=X",
]

INTERVAL_OPTIONS = ["15m", "30m", "60m"]


# ------------------------- NARZĘDZIA / FUNKCJE -------------------------

@st.cache_data(ttl=120, show_spinner=False)
def fetch_df(symbol: str, period: str = "7d", interval: str = "60m") -> pd.DataFrame:
    """
    Pobierz świeczki z Yahoo Finance. Zwraca DataFrame (czas jako indeks).
    Zabezpiecza przed błędami i zwraca pusty DF w razie problemu.
    """
    try:
        df = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            auto_adjust=True,   # w nowych wersjach to domyślnie True (ostrzeżenia FutureWarning są OK)
            progress=False,
            prepost=False,
            threads=True,
        )
        # Czasem przychodzi wielopoziomowy indeks w kolumnach – zostawiamy jak jest,
        # bo extract_close to obsłuży. Normalizujemy typ indeksu.
        if not df.empty:
            df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()


def extract_close(df: pd.DataFrame, symbol: str) -> pd.Series:
    """
    Zwróć serię Close dla danego symbolu.
    Obsługuje zarówno zwykłe kolumny, jak i MultiIndex (np. ('Close', 'EURUSD=X')).
    Zwraca pustą serię float, jeśli nie ma danych.
    """
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    s = None

    # 1) Najprościej: pojedyncza kolumna "Close" lub "Adj Close" jako Series
    if "Close" in df.columns and not isinstance(df["Close"], pd.DataFrame):
        s = df["Close"]
    elif "Adj Close" in df.columns and not isinstance(df["Adj Close"], pd.DataFrame):
        s = df["Adj Close"]

    # 2) MultiIndex w kolumnach lub "Close" jako DataFrame
    if s is None:
        if isinstance(df.columns, pd.MultiIndex):
            for key in [
                ("Close", symbol),
                (symbol, "Close"),
                ("Adj Close", symbol),
                (symbol, "Adj Close"),
            ]:
                if key in df.columns:
                    s = df[key]
                    break

        if s is None and "Close" in df.columns:
            tmp = df["Close"]
            if isinstance(tmp, pd.DataFrame):
                s = tmp[symbol] if symbol in tmp.columns else tmp.iloc[:, 0]

        # 3) Ostatni fallback: pierwsza numeryczna kolumna
        if s is None:
            num = df.select_dtypes(include="number")
            s = num.iloc[:, 0] if not num.empty else pd.Series(dtype="float64", index=df.index)

    s = pd.to_numeric(s, errors="coerce").dropna()
    s.name = symbol
    return s


def compute_rsi(close: pd.Series, window: int = 14) -> float | None:
    """
    RSI liczony metodą Wildera na bazie EMA.
    Zwraca ostatnią wartość RSI jako float lub None, gdy danych jest za mało.
    """
    if close is None or close.empty or len(close) < max(3, window + 1):
        return None

    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Wilder EMA
    alpha = 1.0 / window
    roll_up = up.ewm(alpha=alpha, adjust=False).mean()
    roll_down = down.ewm(alpha=alpha, adjust=False).mean()

    # Unikamy dzielenia przez zero
    roll_down = roll_down.replace(0, np.nan)
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Ostatnia wartość (po dropna, aby uniknąć NaN)
    rsi = rsi.dropna()
    return float(rsi.iloc[-1]) if not rsi.empty else None


def safe_float(x) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


# ------------------------- INTERFEJS UŻYTKOWNIKA -------------------------

with st.sidebar:
    st.header("Ustawienia strategii")

    buy_th = st.number_input("Próg RSI dla KUP (≤)", min_value=1, max_value=99, value=30, step=1)
    sell_th = st.number_input("Próg RSI dla SPRZEDAJ (≥)", min_value=1, max_value=99, value=70, step=1)
    rsi_window = st.number_input("Okres RSI", min_value=2, max_value=100, value=14, step=1)

    intervals = st.multiselect(
        "Interwały",
        options=INTERVAL_OPTIONS,
        default=["15m", "30m", "60m"],
    )

    only_active = st.checkbox("Pokaż tylko wiersze z aktywnym sygnałem", value=False)

st.title("✅ SEP Forex Signals")
st.caption("Analiza RSI + wykresy (dane: Yahoo Finance)")

if not intervals:
    st.info("Wybierz przynajmniej jeden interwał po lewej.")
    st.stop()

# ------------------------- POBRANIE I PRZYGOTOWANIE TABELI -------------------------

rows = []
with st.spinner("Pobieranie danych..."):
    for sym in SYMBOLS:
        for itv in intervals:
            df = fetch_df(sym, period="7d", interval=itv)
            close = extract_close(df, sym)

            if close.empty:
                rsi_val = None
                price = None
                ts = None
            else:
                rsi_val = compute_rsi(close, window=int(rsi_window))
                price = safe_float(close.iloc[-1])
                ts = close.index[-1] if not close.index.empty else None

            # Ustal sygnał
            signal = "—"
            if rsi_val is not None:
                if rsi_val <= buy_th:
                    signal = "BUY"
                elif rsi_val >= sell_th:
                    signal = "SELL"

            rows.append(
                {
                    "Symbol": sym,
                    "Interwał": itv,
                    "RSI": None if rsi_val is None else round(rsi_val, 2),
                    "Sygnał": signal,
                    "Cena": price,
                    "Czas": ts,
                }
            )

table_df = pd.DataFrame(rows)

# Filtrowanie na życzenie
if only_active and not table_df.empty:
    table_df = table_df[table_df["Sygnał"].isin(["BUY", "SELL"])]

# Formatowanie kolumn
if not table_df.empty:
    # sortowanie: po symbolu i interwale (kolejność ręczna dla interwałów)
    itv_order = {v: i for i, v in enumerate(INTERVAL_OPTIONS)}
    table_df["__itv_order"] = table_df["Interwał"].map(itv_order).fillna(999).astype(int)
    table_df = table_df.sort_values(["Symbol", "__itv_order"]).drop(columns="__itv_order")

    # formatery
    fmt = {
        "RSI": lambda x: "—" if pd.isna(x) else f"{x:.2f}",
        "Sygnał": lambda x: x if isinstance(x, str) else "—",
        "Cena": lambda x: "—" if pd.isna(x) else f"{x:.6f}",
        "Czas": lambda x: "—" if pd.isna(x) else str(x),
    }

    st.dataframe(
        table_df,
        use_container_width=True,
        column_config={
            "Symbol": "Symbol",
            "Interwał": "Interwał",
            "RSI": "RSI",
            "Sygnał": "Sygnał",
            "Cena": "Cena",
            "Czas": "Czas",
        },
        hide_index=True,
    )
else:
    st.info("Brak danych do wyświetlenia (sprawdź połączenie/wybór interwałów).")

# ------------------------- SZCZEGÓŁY / WYKRES -------------------------

st.subheader("Szczegóły")

if table_df.empty:
    st.info("Brak wierszy do podglądu.")
    st.stop()

options = [f"{r.Symbol} × {r.Interwał}" for r in table_df.itertuples()]
sel = st.selectbox("Wybierz wiersz do podglądu", options=options, index=0)

if sel:
    sym, itv = sel.split(" × ", 1)
    df_full = fetch_df(sym, period="7d", interval=itv)
    ser = extract_close(df_full, sym)

    if ser.empty:
        st.info("Brak danych do wykresu.")
    else:
        st.line_chart(ser, use_container_width=True)
        last_rsi = compute_rsi(ser, window=int(rsi_window))

        c1, c2, c3 = st.columns(3)
        c1.metric("RSI", "—" if last_rsi is None else f"{last_rsi:.2f}")
        c2.metric("Cena", f"{float(ser.iloc[-1]):.6f}")
        c3.metric("Ostatnia świeca", str(ser.index[-1]))
