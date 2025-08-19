# interface.py
# SEP Forex Signals – Streamlit (RSI + formacje świecowe + fallback dla metali)
# Autor: przygotowane pod Twoją instalację (Python 3.13, bez TA-Lib, z 'ta')

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf
from ta.momentum import RSIIndicator

# -----------------------------
# Konfiguracja / listy symboli
# -----------------------------

DEFAULT_PAIRS: List[str] = [
    # Główne + krzyżowe (w tickerach Yahoo używamy przyrostka =X)
    "EURUSD=X","GBPUSD=X","USDCHF=X","USDJPY=X","USDCNH=X","USDRUB=X",
    "AUDUSD=X","NZDUSD=X","USDCAD=X","USDSEK=X","USDPLN=X",
    "AUDCAD=X","AUDCHF=X","AUDJPY=X","AUDNZD=X","AUDPLN=X",
    "CADCHF=X","CADJPY=X","CADPLN=X","CHFJPY=X","CHFPLN=X","CNHJPY=X",
    "EURAUD=X","EURCAD=X","EURCHF=X","EURCNH=X","EURGBP=X","EURJPY=X","EURNZD=X","EURPLN=X",
    "GBPAUD=X","GBPCAD=X","GBPCHF=X","GBPJPY=X","GBPPLN=X",
    # Metale – spot tickery często nie mają intraday; damy fallback
    "XAGUSD=X","XAUUSD=X","XPDUSD=X","XPTUSD=X",
]

# Fallback: jeśli spot nie ma danych, spróbuj kontraktu futures
FUTURES_FALLBACK: Dict[str, str] = {
    "XAUUSD=X": "GC=F",   # Gold
    "XAGUSD=X": "SI=F",   # Silver
    "XPTUSD=X": "PL=F",   # Platinum
    "XPDUSD=X": "PA=F",   # Palladium
}

# ---------------
# Pomocnicze I/O
# ---------------

def load_pairs_from_config() -> List[str]:
    """Wczytaj listę par z config.json (jeśli istnieje), w przeciwnym razie DEFAULT_PAIRS."""
    cfg_path = Path(__file__).with_name("config.json")
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            pairs = data.get("pairs") or data.get("PAIRS") or data.get("Pairs")
            if isinstance(pairs, list) and pairs:
                return pairs
        except Exception:
            pass
    return DEFAULT_PAIRS

# -----------------------------
# Pobieranie danych z yfinance
# -----------------------------

@st.cache_data(show_spinner=False, ttl=300)
def fetch_series(
    symbol: str,
    period: str = "5d",
    interval: str = "60m",
) -> Tuple[pd.DataFrame, str]:
    """
    Pobierz dane OHLCV dla symbolu. Jeśli dla metali spot brak danych,
    automatycznie użyjemy kontraktów futures z FUTURES_FALLBACK.
    Zwraca: (df, użyty_symbol)
    """
    used = symbol
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.dropna(), used
    except Exception:
        pass

    # Fallback dla metali
    if symbol in FUTURES_FALLBACK:
        alt = FUTURES_FALLBACK[symbol]
        try:
            df = yf.download(alt, period=period, interval=interval, progress=False, auto_adjust=True)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df.dropna(), alt
        except Exception:
            pass

    # Koniec – pusto
    return pd.DataFrame(), used

# -----------------------------
# RSI + formacje świecowe
# -----------------------------

def compute_rsi(df: pd.DataFrame, length: int = 14) -> Optional[pd.Series]:
    if df is None or df.empty or "Close" not in df.columns:
        return None
    try:
        rsi = RSIIndicator(close=df["Close"], window=length).rsi()
        return rsi
    except Exception:
        return None

def detect_candle_pattern(df: pd.DataFrame) -> str:
    """
    Bardzo lekkie heurystyki:
    - Hammer / Spadająca Gwiazda
    - Bullish/Bearish Engulfing
    - Piercing / Dark Cloud Cover
    Zwraca nazwę formacji lub "".
    """
    if df is None or df.empty:
        return ""

    # Bierzemy 2–3 ostatnie świece
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None

    def body(o, c): return abs(c - o)
    def upper(o, h, c): return float(h - max(o, c))
    def lower(o, l, c): return float(min(o, c) - l)

    try:
        o1, h1, l1, c1 = float(last["Open"]), float(last["High"]), float(last["Low"]), float(last["Close"])
        b1 = body(o1, c1)
        u1 = upper(o1, h1, c1)
        w1 = lower(o1, l1, c1)

        # Hammer / Shooting Star (korpos max 30% całej świecy)
        full = (h1 - l1) if (h1 - l1) != 0 else 1e-9
        if b1 / full <= 0.3:
            if w1 > 2 * b1 and u1 < b1:   # długi dolny cień
                return "Młotek (Hammer)"
            if u1 > 2 * b1 and w1 < b1:   # długi górny cień
                return "Spadająca Gwiazda"

        if prev is not None:
            o0, h0, l0, c0 = float(prev["Open"]), float(prev["High"]), float(prev["Low"]), float(prev["Close"])
            # Engulfing
            if c0 < o0 and c1 > o1 and o1 <= c0 and c1 >= o0:
                return "Objęcie Hossy"
            if c0 > o0 and c1 < o1 and o1 >= c0 and c1 <= o0:
                return "Objęcie Bessy"
            # Piercing / Dark Cloud
            mid0 = o0 + (c0 - o0) * 0.5
            if c0 < o0 and c1 > o1 and c1 >= mid0 and o1 < c0:
                return "Przenikanie"
            if c0 > o0 and c1 < o1 and c1 <= mid0 and o1 > c0:
                return "Zasłona Ciemnej Chmury"
    except Exception:
        return ""

    return ""

def build_row(symbol: str, interval_min: int, rsi_len: int, rsi_buy: float, rsi_sell: float) -> Dict:
    interval_code = f"{interval_min}m"
    # okres danych – 5 dni dla M15/M30/M60 wystarcza
    df, used_symbol = fetch_series(symbol, period="7d", interval=interval_code)
    if df.empty:
        return {
            "Symbol": symbol,
            "Użyty ticker": used_symbol,
            "Interwał": interval_code,
            "Cena": "-",
            "RSI": "-",
            "Formacja": "-",
            "Sygnał": "BRAK DANYCH",
            "_df": pd.DataFrame()
        }

    # RSI
    rsi = compute_rsi(df, length=rsi_len)
    rsi_last = float(rsi.dropna().iloc[-1]) if rsi is not None and not rsi.dropna().empty else None

    # Formacja (ostatnia świeca)
    pattern = detect_candle_pattern(df)

    price = float(df["Close"].iloc[-1])

    # Sygnał wg RSI + formacja (pokaż tylko, gdy jest formacja i RSI w strefie)
    signal = ""
    if rsi_last is not None and pattern:
        if rsi_last <= rsi_buy:
            signal = "KUP (RSI & formacja)"
        elif rsi_last >= rsi_sell:
            signal = "SPRZEDAJ (RSI & formacja)"

    return {
        "Symbol": symbol,
        "Użyty ticker": used_symbol,
        "Interwał": interval_code,
        "Cena": round(price, 5) if price < 1000 else round(price, 2),
        "RSI": round(rsi_last, 2) if rsi_last is not None else "-",
        "Formacja": pattern if pattern else "-",
        "Sygnał": signal if signal else "—",
        "_df": df
    }

# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="SEP Forex Signals", layout="wide")
st.title("SEP Forex Signals")
st.caption("Analiza RSI + formacje świecowe + przypisane newsy (intraday: M15/M30/H1, Yahoo Finance).")

# Sidebar – ustawienia strategii
with st.sidebar:
    st.header("Ustawienia strategii")
    rsi_buy = st.number_input("Próg RSI dla KUP (≤)", min_value=1, max_value=50, value=30, step=1)
    rsi_sell = st.number_input("Próg RSI dla SPRZEDAJ (≥)", min_value=50, max_value=99, value=70, step=1)
    rsi_len = st.number_input("Okres RSI", min_value=5, max_value=50, value=14, step=1)
    chosen = st.multiselect("Interwały", options=["15m","30m","60m"], default=["15m","30m","60m"])
    show_only_active = st.checkbox("Pokaż tylko wiersze z aktywnym sygnałem", value=False)

# Główne sterowanie
col1, col2 = st.columns([1, 2])
with col1:
    refresh = st.button("Odśwież dane", type="primary")
with col2:
    st.caption(f"Ostatnia aktualizacja: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

intervals_min = sorted({int(x.replace("m","")) for x in chosen})
pairs = load_pairs_from_config()

# Budowa tabeli
rows: List[Dict] = []
progress = st.empty()
total = len(pairs) * max(1, len(intervals_min))
done = 0

for sym in pairs:
    for im in intervals_min:
        row = build_row(sym, im, rsi_len, rsi_buy, rsi_sell)
        rows.append(row)
        done += 1
        if done % 5 == 0:
            progress.text(f"Pobieranie: {done}/{total}")

progress.empty()

if not rows:
    st.info("Brak danych do wyświetlenia.")
    st.stop()

# DataFrame do prezentacji
df_view = pd.DataFrame([{k: v for k, v in r.items() if k != "_df"} for r in rows])

if show_only_active:
    df_view = df_view[df_view["Sygnał"].isin(["KUP (RSI & formacja)", "SPRZEDAJ (RSI & formacja)"])]

# Sortowanie: aktywne sygnały na górze
df_view["__rank"] = df_view["Sygnał"].apply(lambda s: 0 if s in ("KUP (RSI & formacja)", "SPRZEDAJ (RSI & formacja)") else 1)
df_view = df_view.sort_values(["__rank", "Symbol", "Interwał"]).drop(columns="__rank")

st.dataframe(
    df_view[["Symbol","Użyty ticker","Interwał","Cena","RSI","Formacja","Sygnał"]],
    use_container_width=True,
    hide_index=True
)

# Szczegóły – ekspandery per wiersz
st.subheader("Szczegóły")
for r in rows:
    if r["_df"].empty:
        continue
    label = f"{r['Symbol']} • {r['Interwał']} • {r['Sygnał'] if r['Sygnał']!='—' else 'brak sygnału'}"
    with st.expander(label):
        st.write(f"**Użyty ticker:** {r['Użyty ticker']}")
        st.write(f"**Ostatnia cena:** {r['Cena']}")
        st.write(f"**RSI:** {r['RSI']}")
        st.write(f"**Formacja:** {r['Formacja']}")
        st.line_chart(r["_df"][["Close"]].rename(columns={"Close": r["Symbol"]}))

# Sekcja newsów – pokaż komunikat gdy parser nie jest dostępny
st.subheader("Ważne wiadomości (Investing, ForexFactory, MarketWatch)")
try:
    from news_parsers import get_top_news  # opcjonalny moduł
    news_items = get_top_news(max_items=15)
    if news_items:
        for n in news_items:
            st.markdown(f"- [{n['title']}]({n['url']}) — {n.get('source','')}")
    else:
        st.info("Brak świeżych wiadomości do wyświetlenia.")
except Exception:
    st.info("Moduł parsowania newsów jest niedostępny. Upewnij się, że plik **news_parsers.py** znajduje się w repozytorium.")
