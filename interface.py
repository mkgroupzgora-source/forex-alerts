# interface.py
# Streamlit UI: sygnały RSI + formacje świecowe (M15, M30, H1)
# Bez TA‑Lib, działa w chmurze (używa "ta" + własne reguły formacji)

import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf
import streamlit as st

from ta.momentum import RSIIndicator
from candle_patterns import last_pattern, scan_patterns

# ========= USTAWIENIA =========
PAIRS: List[str] = [
    "EURUSD=X", "GBPUSD=X", "USDCHF=X", "USDJPY=X",
    "AUDUSD=X", "NZDUSD=X", "USDCAD=X", "USDPLN=X",
    # metale/crypto na Yahoo: XAUUSD i XAGUSD bywają pod innymi tickerami.
    # Popularne zamienniki:
    # Złoto (spot): "GC=F" lub "XAUUSD=X" (czasem brak). My spróbujemy oba.
    # Srebro (spot): "SI=F" lub "XAGUSD=X"
]
# zamienniki tickera jeśli podstawowy nie działa
TICKER_ALIASES: Dict[str, List[str]] = {
    "XAUUSD=X": ["XAUUSD=X", "GC=F"],
    "XAGUSD=X": ["XAGUSD=X", "SI=F"],
}

INTERVALS: Dict[str, Tuple[str, str]] = {
    # nazwa → (yfinance interval, okres)
    "M15": ("15m", "7d"),
    "M30": ("30m", "7d"),
    "H1":  ("60m", "30d"),
}

DEFAULT_RSI_BUY = 30
DEFAULT_RSI_SELL = 70

BULLISH_NAMES = {
    "Hammer",
    "Bullish Engulfing",
    "Piercing",
    "Morning Star",
}
BEARISH_NAMES = {
    "Shooting Star",
    "Bearish Engulfing",
    "Dark Cloud Cover",
    "Evening Star",
}


# ========= FUNKCJE POMOCNICZE =========

@st.cache_data(show_spinner=False, ttl=60*10)  # cache 10 min
def fetch_ohlc(symbol: str, yf_interval: str, yf_period: str) -> pd.DataFrame:
    """Pobiera OHLC dla symbolu; próbuje aliasów, czyści duplikaty."""
    candidates = TICKER_ALIASES.get(symbol, [symbol])
    last_err = None
    for tick in candidates:
        try:
            df = yf.download(
                tick, interval=yf_interval, period=yf_period, auto_adjust=True, progress=False
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df[["Open", "High", "Low", "Close"]].copy()
                df = df[~df.index.duplicated(keep="last")]
                df.dropna(inplace=True)
                return df
        except Exception as e:
            last_err = e
            continue
    # pusta ramka jako bezpieczny fallback
    return pd.DataFrame(columns=["Open", "High", "Low", "Close"])


def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    if df is None or df.empty:
        return pd.Series([], dtype="float64")
    rsi = RSIIndicator(close=df["Close"], window=window).rsi()
    return rsi


def evaluate_signal(rsi_val: float, pattern: str, rsi_buy: int, rsi_sell: int) -> str:
    """
    BUY: jeśli jest formacja bycza i RSI < rsi_buy
    SELL: jeśli jest formacja niedźwiedzia i RSI > rsi_sell
    NO SIGNAL w pozostałych przypadkach lub brak danych.
    """
    if pattern:
        # jeśli w nazwie jest kilka, rozdziel po przecinku
        labels = {p.strip() for p in pattern.split(",") if p.strip()}
        if rsi_val is not None and pd.notna(rsi_val):
            if labels & BULLISH_NAMES and rsi_val < rsi_buy:
                return "BUY"
            if labels & BEARISH_NAMES and rsi_val > rsi_sell:
                return "SELL"
    return "NO SIGNAL"


def analyze_pair_for_interval(symbol: str, interval_name: str, rsi_buy: int, rsi_sell: int) -> dict:
    yf_interval, yf_period = INTERVALS[interval_name]
    df = fetch_ohlc(symbol, yf_interval, yf_period)

    if df.empty:
        return {
            "pair": symbol,
            "interval": interval_name,
            "rsi": None,
            "pattern": "",
            "signal": "NO DATA",
            "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }

    # RSI
    df["RSI"] = compute_rsi(df)
    rsi_val = float(df["RSI"].dropna().iloc[-1]) if not df["RSI"].dropna().empty else None

    # Formacje (ostatnia świeca)
    patt = last_pattern(df) or ""

    # Sygnał wg warunku „formacja + rsi”
    sig = evaluate_signal(rsi_val, patt, rsi_buy, rsi_sell)

    # znacznik czasu wg indeksu
    last_ts = df.index[-1]
    if hasattr(last_ts, "tz_localize") or getattr(last_ts, "tzinfo", None) is not None:
        ts_str = last_ts.strftime("%Y-%m-%d %H:%M %Z")
    else:
        ts_str = last_ts.strftime("%Y-%m-%d %H:%M")

    return {
        "pair": symbol,
        "interval": interval_name,
        "rsi": round(rsi_val, 2) if rsi_val is not None else None,
        "pattern": patt,
        "signal": sig,
        "updated": ts_str,
    }


def build_table(pairs: List[str], intervals: List[str], rsi_buy: int, rsi_sell: int) -> pd.DataFrame:
    rows = []
    for p in pairs:
        for itv in intervals:
            rows.append(analyze_pair_for_interval(p, itv, rsi_buy, rsi_sell))
    df = pd.DataFrame(rows)
    # sort: sygnały na górę, potem para i interwał
    sig_order = {"BUY": 0, "SELL": 1, "NO SIGNAL": 2, "NO DATA": 3}
    df["__order"] = df["signal"].map(sig_order).fillna(9)
    df.sort_values(["__order", "pair", "interval"], inplace=True)
    df.drop(columns="__order", inplace=True)
    return df


# ========= UI =========

st.set_page_config(page_title="Forex Alerts – Patterns + RSI", layout="wide")

st.title("📈 Forex Alerts – Formacje świecowe + RSI")
st.caption("Sygnał pojawia się **tylko** jeśli jest rozpoznana formacja i RSI spełnia próg.")

with st.sidebar:
    st.subheader("Ustawienia")
    rsi_buy = st.number_input("Próg RSI (BUY, poniżej):", min_value=5, max_value=50, value=DEFAULT_RSI_BUY, step=1)
    rsi_sell = st.number_input("Próg RSI (SELL, powyżej):", min_value=50, max_value=95, value=DEFAULT_RSI_SELL, step=1)

    chosen_pairs = st.multiselect("Pary:", options=PAIRS, default=PAIRS)
    chosen_itv = st.multiselect("Interwały:", options=list(INTERVALS.keys()), default=["M15", "M30", "H1"])

    col_a, col_b = st.columns(2)
    with col_a:
        refresh = st.button("🔄 Odśwież teraz", use_container_width=True)
    with col_b:
        st.write("")  # odstęp
        st.caption("Dane cache’owane 10 min. Odśwież wymusza pobranie.")

# Wymuszenie mini‑busta cache przy odświeżeniu
if refresh:
    # param zależny od czasu do „przełamania” cache_data
    st.session_state["__refresh_key"] = time.time()

st.markdown("### 🧭 Przegląd sygnałów")
table = build_table(chosen_pairs, chosen_itv, rsi_buy, rsi_sell)
st.dataframe(
    table.rename(columns={
        "pair": "Para",
        "interval": "Interwał",
        "rsi": "RSI",
        "pattern": "Formacja",
        "signal": "Sygnał",
        "updated": "Aktualizacja"
    }),
    use_container_width=True,
    hide_index=True
)

# Detal po kliknięciu – proste rozwijane szczegóły
st.markdown("---")
st.markdown("### 🔍 Szczegóły pary")
col1, col2 = st.columns([1, 2])
with col1:
    symbol_pick = st.selectbox("Wybierz parę:", options=chosen_pairs)
    itv_pick = st.selectbox("Interwał:", options=chosen_itv)

with col2:
    yf_interval, yf_period = INTERVALS[itv_pick]
    df_det = fetch_ohlc(symbol_pick, yf_interval, yf_period)
    if df_det.empty:
        st.warning("Brak danych dla wybranego instrumentu/interwału.")
    else:
        df_det = df_det.copy()
        df_det["RSI"] = compute_rsi(df_det)
        patt_series = scan_patterns(df_det)
        last_patt = last_pattern(df_det) or ""
        last_rsi = float(df_det["RSI"].dropna().iloc[-1]) if not df_det["RSI"].dropna().empty else None
        signal = evaluate_signal(last_rsi, last_patt, rsi_buy, rsi_sell)

        st.metric(
            label=f"{symbol_pick} · {itv_pick}",
            value=signal,
            help="Sygnał = formacja + RSI względem progów"
        )
        st.write(f"**Ostatnia formacja:** {last_patt or '—'}")
        st.write(f"**Ostatni RSI:** {round(last_rsi,2) if last_rsi is not None else '—'}")

        st.markdown("#### Ostatnie świeczki")
        show_n = st.slider("Ile wierszy pokazać:", min_value=20, max_value=300, value=120, step=10)
        view = df_det.tail(show_n)[["Open", "High", "Low", "Close", "RSI"]].copy()
        view.index.name = "Time"
        st.dataframe(view, use_container_width=True)
