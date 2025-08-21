# interface.py
# SEP Forex Signals – RSI + formacje świecowe
# Źródła: Yahoo (opóźnione) lub lokalny MT5 (opcjonalnie).
# Autor: (Twoje)

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st

# wykres świecowy
import mplfinance as mpf

# dane Yahoo
import yfinance as yf

# ====== Opcjonalny MT5 (lokalnie) ======
MT5_AVAILABLE = False
try:
    from mt5_handler import mt5_fetch_ohlc  # funkcja opcjonalna w Twoim repo
    MT5_AVAILABLE = True
except Exception:
    MT5_AVAILABLE = False

# ====== Symbole ======
try:
    # opcjonalny moduł z Twojego repo (jeśli jest)
    from fx_symbols import SYMBOLS_YF
except Exception:
    # Rezerwowa lista – możesz rozszerzyć
    SYMBOLS_YF = [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "USDCAD=X",
        "XAUUSD=X", "XAGUSD=X"  # złoto/srebro (czasem Yahoo zwraca XAGUSD=X)
    ]

# ====== Ustawienia strony ======
st.set_page_config(page_title="SEP Forex Signals", layout="wide")

# ====== Pomocnicze ===========================================================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Prosta implementacja RSI (bez ta-lib)."""
    series = series.astype(float)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wykładnicze średnie
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def _clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Porządkuje dane pod wykresy i obliczenia."""
    if df is None or df.empty:
        return pd.DataFrame()
    # Yahoo potrafi zwrócić kolumny wielopoziomowe – spłaszczamy jeśli trzeba
    if isinstance(df.columns, pd.MultiIndex):
        # spróbujmy znaleźć ('Close', '') etc.
        df = df.copy()
        df.columns = ['_'.join([c for c in col if c]) for col in df.columns.values]
        rename_map = {
            "Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Adj Close": "Adj Close", "Volume":"Volume",
            "Open_": "Open", "High_": "High", "Low_": "Low", "Close_": "Close", "Adj Close_": "Adj Close", "Volume_":"Volume",
            "Open_EURUSD=X": "Open", "High_EURUSD=X": "High", "Low_EURUSD=X": "Low", "Close_EURUSD=X": "Close",
        }
        for k, v in list(rename_map.items()):
            if k in df.columns:
                df.rename(columns={k: v}, inplace=True)

    # Zachowujemy standardowy zestaw
    cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[cols].copy()

    # indeks jako DatetimeIndex
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass

    # usuwamy wiersze z brakami cen
    must_have = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    if must_have:
        df = df.dropna(subset=must_have)

    # posortuj i usuń duplikaty indexu
    df = df[~df.index.duplicated(keep="last")].sort_index()

    return df


@dataclass
class PatternResult:
    name: Optional[str] = None
    bullish: bool = False
    bearish: bool = False


def detect_candle_pattern(df: pd.DataFrame) -> PatternResult:
    """
    Bardzo lekka detekcja kilku formacji na ostatniej świecy:
    - Bullish Engulfing
    - Bearish Engulfing
    - Hammer
    - Shooting Star
    Zwraca PatternResult (nazwa + flaga bullish/bearish).
    """
    res = PatternResult()
    if df is None or df.empty or len(df) < 2:
        return res

    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    # ostatnia i poprzednia świeca
    o1, c1, h1, l1 = o.iloc[-1], c.iloc[-1], h.iloc[-1], l.iloc[-1]
    o2, c2 = o.iloc[-2], c.iloc[-2]

    body1 = abs(c1 - o1)
    range1 = (h1 - l1) if (h1 - l1) != 0 else np.nan
    upper_shadow = h1 - max(c1, o1)
    lower_shadow = min(c1, o1) - l1

    # Bullish Engulfing: poprzednia świeca spadkowa, obecna wzrostowa i korpus obejmuje poprzedni
    if (c2 < o2) and (c1 > o1) and (o1 <= c2) and (c1 >= o2):
        res.name = "Bullish Engulfing"
        res.bullish = True
        return res

    # Bearish Engulfing
    if (c2 > o2) and (c1 < o1) and (o1 >= c2) and (c1 <= o2):
        res.name = "Bearish Engulfing"
        res.bearish = True
        return res

    # Hammer: mały korpus, długi dolny cień
    if range1 and body1 / range1 < 0.3 and lower_shadow > 2 * body1 and upper_shadow < body1:
        res.name = "Hammer"
        res.bullish = True
        return res

    # Shooting Star: mały korpus, długi górny cień
    if range1 and body1 / range1 < 0.3 and upper_shadow > 2 * body1 and lower_shadow < body1:
        res.name = "Shooting Star"
        res.bearish = True
        return res

    return res


@st.cache_data(show_spinner=False, ttl=60)
def fetch_yahoo(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Pobiera OHLC z Yahoo Finance i porządkuje."""
    df = yf.download(
        symbol, period=period, interval=interval, progress=False, auto_adjust=True,
        group_by="column", threads=True
    )
    return _clean_ohlc(df)


def fetch_data(source: str, symbol: str, interval: str, mt5_login: Optional[str] = None,
               mt5_server: Optional[str] = None, mt5_password: Optional[str] = None) -> pd.DataFrame:
    """Pobiera dane z wybranego źródła."""
    if source == "Yahoo":
        # dla intra sensownie 7d – 15/30/60m
        return fetch_yahoo(symbol, period="7d", interval=interval)

    # MT5 (opcjonalnie lokalnie)
    if source == "MT5":
        if not MT5_AVAILABLE:
            st.info("MT5: moduł lokalny niedostępny (mt5_handler). Pozostaję przy Yahoo.")
            return pd.DataFrame()
        try:
            df = mt5_fetch_ohlc(symbol=symbol, timeframe=interval, login=mt5_login,
                                server=mt5_server, password=mt5_password)
            return _clean_ohlc(df)
        except Exception as e:
            st.warning(f"MT5: problem z pobraniem {symbol}/{interval}: {e}")
            return pd.DataFrame()

    return pd.DataFrame()


def last_safe(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    val = series.dropna()
    if val.empty:
        return None
    try:
        return float(val.iloc[-1])
    except Exception:
        return None


def decide_signal(rsi_value: Optional[float], pattern: PatternResult,
                  rsi_buy_thr: float, rsi_sell_thr: float) -> Optional[str]:
    """Sygnal generowany tylko przy zgodności RSI + formacji."""
    if rsi_value is None or math.isnan(rsi_value):
        return None

    # BUY: RSI < próg kupna i formacja bycza
    if rsi_value <= rsi_buy_thr and pattern.bullish:
        return "BUY"
    # SELL: RSI > próg sprzedaży i formacja niedźwiedzia
    if rsi_value >= rsi_sell_thr and pattern.bearish:
        return "SELL"

    return None


def plot_candles_mpf(df: pd.DataFrame, pattern: PatternResult, title: str = ""):
    """Wykres świecowy przy użyciu mplfinance. Zabezpieczenia przed błędami."""
    if df is None or df.empty:
        st.info("Brak danych do wykresu.")
        return

    # bezpieczeństwo: wymagane kolumny
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            st.info("Brak wymaganych kolumn OHLC do wykresu.")
            return

    # czyszczenie
    dfp = df.copy()
    dfp.index = pd.to_datetime(dfp.index)
    dfp = dfp.dropna(subset=["Open", "High", "Low", "Close"])
    if dfp.empty:
        st.info("Brak niepustych świec do wykresu.")
        return

    # ogranicz wykres do ~200 ostatnich świec dla czytelności
    if len(dfp) > 200:
        dfp = dfp.iloc[-200:]

    # rysuj
    try:
        mpf.plot(
            dfp,
            type="candle",
            style="charles",
            volume=False,
            figratio=(10, 5),
            title=title or "",
            mav=(7, 14)  # proste średnie, lepsza czytelność
        )
        st.pyplot(mpf.gcf(), clear_figure=True, use_container_width=True)
    except Exception as e:
        st.warning(f"Nie udało się narysować wykresu: {e}")


# ====== UI ===================================================================

with st.sidebar:
    st.markdown("### Źródło danych")
    source = st.radio("",
                      options=["Yahoo (opóźnione)", "MT5 (real-time, lokalnie)"],
                      index=0,
                      label_visibility="collapsed")
    if source.startswith("Yahoo"):
        source_key = "Yahoo"
    else:
        source_key = "MT5"

    st.markdown("---")
    rsi_buy = st.number_input("Próg RSI dla KUP (≤)", min_value=1, max_value=99, value=30, step=1)
    rsi_sell = st.number_input("Próg RSI dla SPRZEDAJ (≥)", min_value=1, max_value=99, value=70, step=1)
    rsi_period = st.number_input("Okres RSI", min_value=2, max_value=100, value=14, step=1)

    intervals = st.multiselect("Interwały", options=["15m", "30m", "60m"], default=["15m", "30m", "60m"])
    only_active = st.checkbox("Pokaż tylko wiersze z aktywnym sygnałem", value=False)

    # Pola MT5 (opcjonalnie lokalnie)
    mt5_login = mt5_server = mt5_password = None
    if source_key == "MT5":
        st.markdown("---")
        st.caption("Połącz z MT5 (lokalnie na Windows)")
        mt5_login = st.text_input("Login (numer)")
        mt5_server = st.text_input("Server (np. XTB Demo)")
        mt5_password = st.text_input("Hasło", type="password")


st.title("🧠 SEP Forex Signals")
st.caption("RSI + formacje świecowe • Źródło danych: Yahoo (opóźnione) lub MT5 (real-time, lokalnie)")

# ====== Budowa tabeli z sygnałami ============================================

rows: List[Dict] = []

for sym in SYMBOLS_YF:
    for itv in intervals:
        df = fetch_data(source_key, sym, itv, mt5_login, mt5_server, mt5_password)
        if df is None or df.empty:
            rows.append({
                "Symbol": sym, "Interwał": itv, "RSI": None, "Formacja": None,
                "Sygnał": None, "Cena": None, "Czas": None, "_df": None
            })
            continue

        # RSI
        r = rsi(df["Close"], period=rsi_period)
        r_last = last_safe(r)

        # Formacja (na ostatniej świecy)
        patt = detect_candle_pattern(df)

        # sygnał tylko przy zgodności RSI + formacji
        sig = decide_signal(r_last, patt, rsi_buy, rsi_sell)

        # cena i czas
        price = last_safe(df["Close"])
        ts = None
        if not df.empty:
            ts = df.index[-1]

        rows.append({
            "Symbol": sym,
            "Interwał": itv,
            "RSI": None if r_last is None else round(float(r_last), 2),
            "Formacja": patt.name if patt.name else None,
            "Sygnał": sig,
            "Cena": price,
            "Czas": ts,
            "_df": df
        })

df_view = pd.DataFrame(rows)

# filtr aktywnych
if only_active and not df_view.empty:
    df_view = df_view[df_view["Sygnał"].notna()]

# Porządkowanie i prezentacja
show_cols = ["Symbol", "Interwał", "RSI", "Formacja", "Sygnał", "Cena", "Czas"]
st.dataframe(df_view[show_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

# ====== Szczegóły + wykres świecowy =========================================

st.markdown("### Szczegóły")
# wybór wiersza (symbol + interwał)
if not df_view.empty:
    options = [f"{r.Symbol} × {r.Interwał}" for r in df_view.itertuples()]
    choice = st.selectbox("Wybierz wiersz do podglądu", options=options, index=0)
    # znajdź rekord
    sym_sel, itv_sel = choice.split(" × ")
    row_sel = df_view[(df_view["Symbol"] == sym_sel) & (df_view["Interwał"] == itv_sel)]
    if not row_sel.empty:
        df_sel: pd.DataFrame = row_sel["_df"].iloc[0]
        patt_sel = detect_candle_pattern(df_sel)

        # podsumowanie
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Symbol", sym_sel)
        with c2:
            st.metric("Interwał", itv_sel)
        with c3:
            st.metric("Formacja", patt_sel.name if patt_sel.name else "—")

        # wykres
        st.markdown("#### Wykres świecowy")
        plot_candles_mpf(df_sel, patt_sel, title=f"{sym_sel} • {itv_sel}")

# ====== Kolorowanie sygnału ==================================================

# (streamlit dataframe nie ma natywnego warunkowego kolorowania pojedynczego pola,
# więc sygnał pokolorujemy w HTML, a dla prostoty pozostawiamy plain text w tabeli powyżej)
# Na żądanie można przejść na st.data_editor z kolumną stylowaną.


# ====== Akcje pomocnicze =====================================================

st.caption("v1.3 • Dane: Yahoo Finance (opóźnione) / MT5 lokalnie (jeśli dostępny).")

# Zamiennik starego experimental_rerun (jeśli kiedyś używałeś)
def _rerun_safe():
    try:
        st.rerun()
    except Exception:
        pass
