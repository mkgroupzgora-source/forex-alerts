# interface.py  — v1.4
# SEP Forex Signals: RSI + formacje świecowe + wykres świecowy (mplfinance)
# Źródło danych: Yahoo Finance (opóźnione). MT5 — placeholder (lokalnie, w przyszłości).

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import mplfinance as mpf

# ========= KONFIG =========

APP_TITLE = "SEP Forex Signals"
DEFAULT_RSI_PERIOD = 14
DEFAULT_INTERVALS = ["15m", "30m", "60m"]

# Główne pary + metale na Yahoo (=X)
SYMBOLS_YF = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "USDCAD=X",
    "AUDUSD=X", "NZDUSD=X",
    "EURGBP=X", "EURJPY=X", "GBPJPY=X",
    "XAUUSD=X", "XAGUSD=X",   # Złoto, Srebro (opóźnione)
]

# mapka do ładnych nazw
NICE = {s: s for s in SYMBOLS_YF}

# ========= NARZĘDZIA =========

@st.cache_data(ttl=300, show_spinner=False)
def fetch_series_yahoo(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Pobierz notowania z Yahoo dla jednego symbolu."""
    df = yf.download(
        symbol, period=period, interval=interval,
        progress=False, auto_adjust=True, threads=True,
    )
    # Uporządkowanie kolumn jak lubi mplfinance
    ren = {
        "Open": "Open", "High": "High", "Low": "Low", "Close": "Close"
    }
    if isinstance(df.columns, pd.MultiIndex):
        # czasem yfinance zwraca multiindex dla wielu ticków — tu zawsze 1
        df = df.xs(symbol, axis=1, drop_level=False) if symbol in df.columns.get_level_values(0) else df
        # spróbujmy wydobyć standardowe OHLC
        cols = {}
        for c in ["Open", "High", "Low", "Close"]:
            if (c in df.columns) or ((symbol, c) in df.columns):
                cols[(symbol, c)] = c
        if cols:
            df = df[list(cols.keys())]
            df.columns = list(cols.values())
    else:
        # standard single-ticker
        keep = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
        df = df[keep]

    # Odfiltruj NaN
    df = df.dropna()
    return df


def rsi(series: pd.Series, window: int) -> pd.Series:
    """RSI klasyczny (bez zależności zewn.)."""
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out


@dataclass
class PatternResult:
    name: Optional[str] = None
    strength: Optional[str] = None  # np. "weak"/"strong"
    direction: Optional[str] = None  # "bull"/"bear"


def _is_bull_engulf(o1, c1, o2, c2) -> bool:
    # świeca 1 (poprzednia) spadkowa, świeca 2 (ostatnia) wzrostowa
    return (c1 < o1) and (c2 > o2) and (o2 <= c1) and (c2 >= o1)

def _is_bear_engulf(o1, c1, o2, c2) -> bool:
    return (c1 > o1) and (c2 < o2) and (o2 >= c1) and (c2 <= o1)

def _is_hammer(o, h, l, c) -> bool:
    body = abs(c - o)
    lower = o if c >= o else c
    tail = lower - l
    upper = h - max(o, c)
    return (tail > 2 * body) and (upper < body) and (c > o)  # zielony młot

def _is_shooting_star(o, h, l, c) -> bool:
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    return (upper > 2 * body) and (lower < body) and (c < o)  # czerwona spadająca gwiazda

def _is_doji(o, c, tol=0.00005) -> bool:
    return abs(c - o) <= tol * max(abs(c), abs(o), 1.0)

def detect_pattern(df: pd.DataFrame) -> PatternResult:
    """Proste rozpoznanie formacji na ostatniej świecy (+ poprzednia dla engulfing)."""
    try:
        if df is None or len(df) < 2:
            return PatternResult()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        o2, h2, l2, c2 = last["Open"], last["High"], last["Low"], last["Close"]
        o1, h1, l1, c1 = prev["Open"], prev["High"], prev["Low"], prev["Close"]

        # Engulfingi
        if _is_bull_engulf(o1, c1, o2, c2):
            return PatternResult("Bullish Engulfing", "strong", "bull")
        if _is_bear_engulf(o1, c1, o2, c2):
            return PatternResult("Bearish Engulfing", "strong", "bear")

        # Młot / Spadająca gwiazda
        if _is_hammer(o2, h2, l2, c2):
            return PatternResult("Hammer", "medium", "bull")
        if _is_shooting_star(o2, h2, l2, c2):
            return PatternResult("Shooting Star", "medium", "bear")

        # Doji (neutral / słaby sygnał)
        if _is_doji(o2, c2):
            # kierunek doji niejednoznaczny
            return PatternResult("Doji", "weak", None)

        return PatternResult()
    except Exception:
        return PatternResult()


def pretty_signal(sig: str) -> str:
    if sig == "BUY":
        return "🟢 KUP"
    if sig == "SELL":
        return "🔴 SPRZEDAJ"
    return "—"


def build_row(symbol: str, interval: str, rsi_buy: int, rsi_sell: int, rsi_period: int) -> Optional[Dict]:
    """Zbuduj pojedynczy wiersz tabeli."""
    try:
        df = fetch_series_yahoo(symbol, period="5d", interval=interval)
        if df is None or df.empty or "Close" not in df.columns:
            return None

        close = df["Close"]
        rsi_series = rsi(close, rsi_period)
        rsi_val = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else np.nan

        pattern = detect_pattern(df)
        patt_name = pattern.name if pattern.name else "None"

        signal = "NONE"
        if not math.isnan(rsi_val):
            if rsi_val <= rsi_buy and pattern.direction == "bull":
                signal = "BUY"
            elif rsi_val >= rsi_sell and pattern.direction == "bear":
                signal = "SELL"

        last_price = float(close.dropna().iloc[-1])
        last_time = close.dropna().index[-1].to_pydatetime()

        return {
            "Symbol": symbol,
            "Interval": interval,
            "RSI": round(rsi_val, 2) if not math.isnan(rsi_val) else None,
            "Formacja": patt_name,
            "Sygnał": pretty_signal(signal),
            "Cena": last_price,
            "Czas": last_time,
            "_raw_df": df,               # przyda się do wykresu
            "_pattern": pattern,         # do etykiety
            "_raw_signal": signal,       # do filtra „tylko aktywne”
        }
    except Exception:
        return None


def plot_candles_mpf(df: pd.DataFrame, pattern: PatternResult, title: str = ""):
    """Wykres świecowy przy użyciu mplfinance (bez mpf.gcf)."""
    if df is None or df.empty:
        st.info("Brak danych do wykresu.")
        return

    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            st.info("Brak wymaganych kolumn OHLC do wykresu.")
            return

    dfp = df.copy().dropna(subset=["Open", "High", "Low", "Close"])
    if dfp.empty:
        st.info("Brak niepustych świec do wykresu.")
        return

    # ogranicz do 200 ostatnich świec
    if len(dfp) > 200:
        dfp = dfp.iloc[-200:]

    pat_txt = f" | Formacja: {pattern.name}" if pattern and pattern.name else ""
    chart_title = f"{title}{pat_txt}"

    fig, _ax = mpf.plot(
        dfp,
        type="candle",
        style="charles",
        volume=False,
        figratio=(10, 5),
        title=chart_title,
        mav=(7, 14),
        returnfig=True,   # KLUCZOWE
    )
    st.pyplot(fig, clear_figure=True, use_container_width=True)


# ========= UI =========

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Źródło danych")
    data_source = st.radio(
        " ",
        options=["Yahoo (opóźnione)", "MT5 (real-time, lokalnie)"],
        index=0,
        help="MT5 będzie działał lokalnie w przyszłości (nieaktywne w tej wersji).",
        label_visibility="collapsed",
    )

    rsi_buy = st.number_input("Próg RSI dla KUP (≤)", min_value=1, max_value=50, value=30, step=1)
    rsi_sell = st.number_input("Próg RSI dla SPRZEDAJ (≥)", min_value=50, max_value=99, value=70, step=1)
    rsi_period = st.number_input("Okres RSI", min_value=2, max_value=50, value=DEFAULT_RSI_PERIOD, step=1)

    intervals = st.multiselect("Interwały", ["15m", "30m", "60m"], default=DEFAULT_INTERVALS)
    only_active = st.checkbox("Pokaż tylko wiersze z aktywnym sygnałem", value=False)

    if data_source.startswith("MT5"):
        st.caption("Połączenie z MT5 (lokalnie na Windows) — w przygotowaniu.")
        st.text_input("Login (numer)", value="", disabled=True)
        st.text_input("Serwer", value="", disabled=True)
        st.text_input("Hasło", value="", type="password", disabled=True)

# Ostrzeżenie dla MT5
if data_source.startswith("MT5"):
    st.warning("Źródło MT5 w tej wersji jest wyłączone. Wybierz „Yahoo (opóźnione)”.")
    # Bezpieczny rerun (Streamlit 1.26+: st.rerun, starsze — brak)
    if hasattr(st, "rerun"):
        pass  # użytkownik sam przełączy; nie wymuszamy rerun
    st.stop()

# ========= GŁÓWNA TABELA =========

rows: List[Dict] = []
for sym in SYMBOLS_YF:
    for itv in intervals:
        row = build_row(sym, itv, rsi_buy, rsi_sell, rsi_period)
        if row:
            rows.append(row)

df_view = pd.DataFrame(rows)

if only_active and not df_view.empty:
    df_view = df_view[df_view["_raw_signal"].isin(["BUY", "SELL"])]

# Przyjazne nazwy + porządek kolumn
if not df_view.empty:
    df_view["Symbol"] = df_view["Symbol"].map(NICE).fillna(df_view["Symbol"])
    show_cols = ["Symbol", "Interval", "RSI", "Formacja", "Sygnał", "Cena", "Czas"]
    df_show = df_view[show_cols].sort_values(["Symbol", "Interval"])
else:
    df_show = pd.DataFrame(columns=["Symbol", "Interval", "RSI", "Formacja", "Sygnał", "Cena", "Czas"])

st.dataframe(
    df_show,
    use_container_width=True,
    height=420,
)

# ========= SZCZEGÓŁY + WYKRES =========

st.subheader("Szczegóły")

if df_show.empty:
    st.info("Brak danych do podglądu. Zmień interwały lub usuń filtr aktywnych sygnałów.")
else:
    # lista opcji do wyboru
    options = [f"{r.Symbol} × {r.Interval}" for r in df_show.itertuples(index=False)]
    selected = st.selectbox("Wybierz wiersz do podglądu", options, index=0)

    # rozbij wybór
    try:
        sel_sym, sel_itv = selected.split(" × ")
    except Exception:
        sel_sym, sel_itv = df_show.iloc[0]["Symbol"], df_show.iloc[0]["Interval"]

    # odszukaj rekord „źródłowy” z ukrytymi kolumnami
    mask = (df_view["Symbol"] == sel_sym) & (df_view["Interval"] == sel_itv)
    if not mask.any():
        st.info("Nie znaleziono danych dla wyboru.")
    else:
        row = df_view[mask].iloc[0]
        st.markdown(f"**Symbol**: {sel_sym} &nbsp;&nbsp;&nbsp; **Interwał**: {sel_itv} &nbsp;&nbsp;&nbsp; **Formacja**: {row['Formacja']}")
        st.markdown("#### Wykres świecowy")

        # wykres
        try:
            plot_candles_mpf(row["_raw_df"], row["_pattern"], title=f"{sel_sym} ({sel_itv})")
        except Exception as e:
            st.info(f"Nie udało się narysować wykresu: {e}")

st.caption("v1.4 — Dane: Yahoo Finance (opóźnione) • MT5 lokalnie (w przygotowaniu)")
