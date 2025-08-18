# interface.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# --- NEWS / NLP ---
try:
    # Twój moduł z poprzedniej sekcji
    from news_parsers import fetch_all_news
except Exception:
    fetch_all_news = None

# Sentiment: użyj modułu jeśli istnieje, w przeciwnym wypadku fallback VADER
try:
    from sentiment_analysis import score_sentiment  # funkcja powinna zwracać [-1..1]
except Exception:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()

    def score_sentiment(text: str) -> float:
        if not text:
            return 0.0
        s = _vader.polarity_scores(text or "")
        return float(s.get("compound", 0.0))

# ------------------ KONFIG ------------------

# domyślne progi i interwały – mogą zostać nadpisane przez config.json
DEFAULT_CONFIG = {
    "pairs": [
        # lista z Twojego zakresu – możesz skrócić dla testów
        "EURUSD", "GBPUSD", "USDCHF", "USDJPY", "USDCNH", "USDRUB",
        "AUDUSD", "NZDUSD", "USDCAD", "USDSEK", "USDPLN",
        "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDPLN",
        "CADCHF", "CADJPY", "CADPLN",
        "CHFJPY", "CHFPLN", "CNHJPY",
        "EURAUD", "EURCAD", "EURCHF", "EURCNH", "EURGBP",
        "EURJPY", "EURNZD", "EURPLN",
        "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPPLN",
        "XAGUSD", "XAUUSD", "XPDUSD", "XPTUSD",
        # opcjonalnie ropa / krypto:
        # "WTIUSD", "BRENTUSD", "BTCUSD"
    ],
    "rsi_buy_threshold": 30,
    "rsi_sell_threshold": 70,
    "rsi_period": 14,
    "intervals": ["15m", "30m", "60m"],
    "timezone": "Europe/Warsaw"
}

def load_config() -> Dict:
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # dopełnij brakujące klucze
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        return cfg
    except Exception:
        return DEFAULT_CONFIG.copy()

CFG = load_config()
PAIRS: List[str] = CFG["pairs"]
RSI_BUY = int(CFG["rsi_buy_threshold"])
RSI_SELL = int(CFG["rsi_sell_threshold"])
RSI_PERIOD = int(CFG["rsi_period"])
INTERVALS = list(CFG["intervals"])

# ------------------ NARZĘDZIA RYNKOWE ------------------

# mapowanie par do tickerów Yahoo Finance
def to_yf_symbol(pair: str) -> Optional[str]:
    """
    EURUSD -> 'EURUSD=X', XAUUSD -> 'XAUUSD=X', itp.
    Wiele krzyżówek zadziała, ale nie wszystkie mają dane intraday.
    """
    base = pair.upper()
    # najczęściej wspierane przez Yahoo:
    return f"{base}=X"

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Wilder) bez ta-lib (EMA na wzrostach/spadkach)."""
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    up_ema = pd.Series(up, index=close.index).ewm(alpha=1/period, adjust=False).mean()
    down_ema = pd.Series(down, index=close.index).ewm(alpha=1/period, adjust=False).mean()
    rs = up_ema / down_ema.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)

# proste formacje świecowe na ostatniej świecy
def detect_candle_pattern(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """
    Zwraca ('HAMMER' / 'SHOOTING_STAR' / 'ENGULF_BULL' / 'ENGULF_BEAR' / 'NONE', sygnał BUY/SELL/None)
    Wystarczy do sygnałów filtrujących RSI.
    """
    if df is None or df.empty or len(df) < 3:
        return "NONE", None

    o = df["Open"].iloc[-1]
    h = df["High"].iloc[-1]
    l = df["Low"].iloc[-1]
    c = df["Close"].iloc[-1]

    body = abs(c - o)
    full = max(h, o, c) - min(l, o, c)
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l

    # młotek: długi dolny cień, mały korpus, górny cień krótki
    if lower_shadow > 2 * body and upper_shadow <= body and body / full < 0.35:
        return "HAMMER", "BUY"

    # spadająca gwiazda: długi górny cień, mały korpus, dolny krótki
    if upper_shadow > 2 * body and lower_shadow <= body and body / full < 0.35:
        return "SHOOTING_STAR", "SELL"

    # objęcie – porównujemy dwie ostatnie świece
    prev_o = df["Open"].iloc[-2]
    prev_c = df["Close"].iloc[-2]
    # Bullish Engulfing
    if (prev_c < prev_o) and (c > o) and (c >= prev_o) and (o <= prev_c):
        return "ENGULF_BULL", "BUY"
    # Bearish Engulfing
    if (prev_c > prev_o) and (c < o) and (c <= prev_o) and (o >= prev_c):
        return "ENGULF_BEAR", "SELL"

    return "NONE", None

def fetch_ohlc(pair: str, yf_interval: str, period: str = "7d") -> Optional[pd.DataFrame]:
    """
    Pobiera OHLC dla pary w danym interwale z Yahoo.
    Interwały 15m/30m/60m wymagają krótkiego period (np. 7d).
    """
    ticker = to_yf_symbol(pair)
    if not ticker:
        return None
    try:
        df = yf.download(ticker, period=period, interval=yf_interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.title)  # O/H/L/C/Adj Close/Volume -> tytułowane
        # niektóre tickery nie zwracają Open/High/Low – spróbuj wyliczyć z Close gdy brak
        for col in ("Open", "High", "Low", "Close"):
            if col not in df.columns:
                df[col] = df["Adj Close"]
        df = df[["Open", "High", "Low", "Close"]].dropna()
        return df
    except Exception:
        return None

def signal_from_rsi_and_pattern(rsi_value: float,
                                pattern_signal: Optional[str],
                                rsi_buy: int,
                                rsi_sell: int) -> Optional[str]:
    """
    Zwraca końcowy sygnał tylko gdy spełniony WARUNEK:
    - jest formacja oraz RSI < rsi_buy (dla BUY) lub RSI > rsi_sell (dla SELL).
    """
    if pattern_signal == "BUY" and rsi_value is not None and rsi_value <= rsi_buy:
        return "KUP"
    if pattern_signal == "SELL" and rsi_value is not None and rsi_value >= rsi_sell:
        return "SPRZEDAJ"
    return None

# ------------------ UI ------------------

st.set_page_config(page_title="SEP Forex Signals", page_icon="📈", layout="wide")

# Sidebar – ustawienia
with st.sidebar:
    st.header("⚙️ Ustawienia strategii")
    RSI_BUY = st.number_input("Próg RSI dla KUP (≤)", min_value=5, max_value=50, value=RSI_BUY, step=1)
    RSI_SELL = st.number_input("Próg RSI dla SPRZEDAJ (≥)", min_value=50, max_value=95, value=RSI_SELL, step=1)
    RSI_PERIOD = st.number_input("Okres RSI", min_value=5, max_value=50, value=RSI_PERIOD, step=1)
    INTERVALS = st.multiselect("Interwały", ["15m", "30m", "60m"], default=INTERVALS)
    show_only_signals = st.checkbox("Pokaż tylko wiersze z aktywnym sygnałem", value=True)

st.title("📊 SEP Forex Signals")
st.caption("Analiza RSI + formacje świecowe + przypisane newsy. (Intra: M15/M30/H1, Yahoo Finance)")

colL, colR = st.columns([1, 3])
with colL:
    if st.button("🔁 Odśwież dane", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with colR:
    st.write(f"Ostatnia aktualizacja: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")

# ----------- Tabela sygnałów -----------

@st.cache_data(ttl=60)
def build_signals(pairs: List[str], intervals: List[str]) -> pd.DataFrame:
    rows = []
    for pair in pairs:
        for itv in intervals:
            # Yahoo interwał = itv, period: 7d dla intraday
            df = fetch_ohlc(pair, yf_interval=itv, period="7d")
            if df is None or df.empty:
                rows.append({
                    "Para": pair, "Interwał": itv, "RSI": None,
                    "Formacja": "BRAK DANYCH", "Sygnał": None, "Godzina": None
                })
                continue

            rsi = compute_rsi(df["Close"], period=RSI_PERIOD).iloc[-1]
            pattern, patt_sig = detect_candle_pattern(df)
            signal = signal_from_rsi_and_pattern(float(rsi), patt_sig, RSI_BUY, RSI_SELL)

            # czas ostatniej świecy – w UTC -> lokalny string
            ts = df.index[-1]
            if getattr(ts, "tzinfo", None) is None:
                ts = ts.tz_localize("UTC")
            ts_str = ts.tz_convert("Europe/Warsaw").strftime("%Y-%m-%d %H:%M")

            rows.append({
                "Para": pair,
                "Interwał": itv,
                "RSI": round(float(rsi), 2) if rsi is not None else None,
                "Formacja": pattern if pattern != "NONE" else "",
                "Sygnał": signal if signal else "",
                "Godzina": ts_str,
            })
    df_out = pd.DataFrame(rows)
    return df_out

signals_df = build_signals(PAIRS, INTERVALS)

if show_only_signals:
    view_df = signals_df[signals_df["Sygnał"] != ""].copy()
else:
    view_df = signals_df.copy()

st.subheader("📈 Sygnały (RSI + formacja)")
if view_df.empty:
    st.info("Brak aktywnych sygnałów dla wybranych ustawień.")
else:
    # Dodaj przyciski „Szczegóły” dla każdej pary (ostatni wiersz per para)
    # Aby nie mnożyć przycisków przy wielu interwałach, pokaż przycisk w każdej linii.
    def _detail_key(idx: int) -> str:
        return f"details_{idx}"

    # render
    for i, row in view_df.reset_index(drop=True).iterrows():
        c1, c2, c3, c4, c5, c6, c7 = st.columns([1.3, 0.7, 0.7, 1.2, 1, 1.2, 0.9])
        c1.write(f"**{row['Para']}**")
        c2.write(row["Interwał"])
        c3.write(row["RSI"] if row["RSI"] is not None else "—")
        c4.write(row["Formacja"] or "—")
        c5.write(f"**{row['Sygnał']}**" if row["Sygnał"] else "—")
        c6.write(row["Godzina"] or "—")
        if c7.button("Szczegóły", key=_detail_key(i)):
            st.session_state["detail_pair"] = row["Para"]
            st.session_state["detail_interval"] = row["Interwał"]
            st.session_state["detail_time"] = row["Godzina"]
            st.rerun()

# ----------- Szczegóły instrumentu -----------

def render_details():
    pair = st.session_state.get("detail_pair")
    itv = st.session_state.get("detail_interval", "60m")
    if not pair:
        return
    st.divider()
    st.subheader(f"🔎 Szczegóły: {pair} ({itv})")

    df = fetch_ohlc(pair, yf_interval=itv, period="7d")
    if df is None or df.empty:
        st.warning("Brak danych do podglądu wykresu.")
        return

    # mini-wykres
    chart = df["Close"].rename("Kurs")
    st.line_chart(chart)

    # ostatnie 10 świec
    st.markdown("**Ostatnie 10 świec (OHLC):**")
    st.dataframe(df.tail(10))

    # związane newsy
    if fetch_all_news:
        with st.spinner("Pobieram newsy i przypisuję do instrumentów…"):
            items = fetch_all_news(supported_symbols=[pair], per_feed_limit=30)
        st.markdown("### 📰 Przypięte newsy")
        if not items:
            st.write("Brak bieżących wiadomości dla tego symbolu.")
        else:
            for it in items[:10]:
                sent = score_sentiment(f"{it['title']} {it.get('summary','')}")
                badge = "🟢" if sent >= 0.2 else "🔴" if sent <= -0.2 else "🟡"
                ts = it["published"].astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                st.markdown(
                    f"{badge} **[{it['title']}]({it['link']})**  "
                    f"<small>({it['source']} • {ts})</small>",
                    unsafe_allow_html=True
                )
                if it.get("summary"):
                    st.caption(it["summary"])

if "detail_pair" in st.session_state:
    render_details()

# ----------- NEWS agregacja globalna -----------

st.divider()
st.subheader("📰 Ważne wiadomości (Investing, ForexFactory, MarketWatch)")

if fetch_all_news:
    with st.spinner("Ładowanie newsów…"):
        all_news = fetch_all_news(supported_symbols=PAIRS, per_feed_limit=20)
    if not all_news:
        st.write("Brak świeżych wiadomości.")
    else:
        # mała lista – top 12
        for n in all_news[:12]:
            syms = ", ".join(n.get("symbols") or [])
            sent = score_sentiment(f"{n['title']} {n.get('summary','')}")
            badge = "🟢" if sent >= 0.2 else "🔴" if sent <= -0.2 else "🟡"
            ts = n["published"].astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            st.markdown(
                f"{badge} **[{n['title']}]({n['link']})** "
                f"<small>({n['source']} • {ts}{' • ' + syms if syms else ''})</small>",
                unsafe_allow_html=True
            )
else:
    st.info("Moduł parsowania newsów jest niedostępny. Upewnij się, że plik **news_parsers.py** znajduje się w repozytorium.")

# stopka
st.caption("v1 • dane: Yahoo Finance • RSI + formacje świecowe + przypięte newsy")
