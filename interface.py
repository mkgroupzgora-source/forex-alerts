# interface.py
from __future__ import annotations

import math
from typing import List, Tuple, Optional, Dict

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

# opcjonalnie ładne świece
try:
    import mplfinance as mpf
    _HAS_MPF = True
except Exception:
    _HAS_MPF = False

# MT5 handler (lokalnie)
try:
    import mt5_handler as mt5h
except Exception:
    mt5h = None

st.set_page_config(page_title="SEP Forex Signals — RT via MT5", layout="wide")
st.title("📈 SEP Forex Signals")
st.caption("RSI + formacje świecowe • Źródło danych: Yahoo (opóźnione) lub MT5 (real‑time, lokalnie)")

# ====== LISTA SYMBOLI (Yahoo tickery) ======
SYMBOLS: List[str] = [
    "EURUSD=X", "GBPUSD=X", "USDCHF=X", "USDJPY=X", "USDCNH=X", "USDRUB=X",
    "AUDUSD=X", "NZDUSD=X", "USDCAD=X", "USDSEK=X", "USDPLN=X",
    "AUDCAD=X", "AUDCHF=X", "AUDJPY=X", "AUDNZD=X", "AUDPLN=X",
    "CADCHF=X", "CADJPY=X", "CADPLN=X",
    "CHFJPY=X", "CHFPLN=X", "CNHJPY=X",
    "EURAUD=X", "EURCAD=X", "EURCHF=X", "EURCNH=X", "EURGBP=X",
    "EURJPY=X", "EURNZD=X", "EURPLN=X",
    "GBPAUD=X", "GBPCAD=X", "GBPCHF=X", "GBPJPY=X", "GBPPLN=X",
    "XAGUSD=X", "XAUUSD=X", "XPDUSD=X", "XPTUSD=X"
]
INTERVALS = ["15m", "30m", "60m"]

# ====== Alias symboli Yahoo -> MT5 (zmień jeśli broker używa sufiksów) ======
# np. "EURUSD=X": "EURUSD", "XAUUSD=X": "XAUUSD."
SYMBOL_ALIAS: Dict[str, str] = {
    # waluty
    "EURUSD=X": "EURUSD",
    "GBPUSD=X": "GBPUSD",
    "USDCHF=X": "USDCHF",
    "USDJPY=X": "USDJPY",
    "USDCAD=X": "USDCAD",
    "USDPLN=X": "USDPLN",
    "EURPLN=X": "EURPLN",
    # metale – zmień na nazwę u brokera, np. "XAUUSD"
    "XAUUSD=X": "XAUUSD",
    "XAGUSD=X": "XAGUSD",
    "XPTUSD=X": "XPTUSD",  # Platinum
    "XPDUSD=X": "XPDUSD",  # Palladium
    # dodawaj kolejne w razie potrzeby...
}

# ================== POBIERANIE DANYCH ==================

@st.cache_data(ttl=60, show_spinner=False)
def yf_ohlc(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """OHLC z Yahoo (opóźnione ~15 min)."""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[keep].dropna()
    except Exception:
        return pd.DataFrame()

def rt_ohlc_mt5(symbol_yf: str, interval: str, bars: int = 500) -> pd.DataFrame:
    """OHLC z MT5 (real‑time) — wymaga uruchomionego terminala i poprawnego symbolu."""
    if mt5h is None or not mt5h.is_available():
        return pd.DataFrame()
    symbol_mt5 = SYMBOL_ALIAS.get(symbol_yf, symbol_yf.replace("=X", ""))  # prosta normalizacja
    return mt5h.get_ohlc(symbol_mt5, interval=interval, bars=bars)

# ================== RSI + FORMACJE ==================

def rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce").dropna()
    if c.size < period + 1:
        return pd.Series(index=close.index, dtype=float)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean().replace(0, 1e-12)
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.reindex(close.index)

def _body(o, c): return abs(c - o)
def _upper(o, h, c): return h - max(o, c)
def _lower(o, l, c): return min(o, c) - l

def detect_last_pattern(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if df is None or df.empty or len(df) < 2:
        return None, None
    o, h, l, c = [df[x].astype(float) for x in ["Open","High","Low","Close"]]
    i0, i1, i2 = -1, -2, -3
    # Engulfing
    if len(df) >= 2:
        prev_bear = c.iloc[i1] < o.iloc[i1]
        prev_bull = c.iloc[i1] > o.iloc[i1]
        bull = (c.iloc[i0] > o.iloc[i0]) and (o.iloc[i0] < c.iloc[i1]) and (c.iloc[i0] > o.iloc[i1]) and prev_bear
        bear = (c.iloc[i0] < o.iloc[i0]) and (o.iloc[i0] > c.iloc[i1]) and (c.iloc[i0] < o.iloc[i1]) and prev_bull
        if bull: return "Objęcie Hossy", "bull"
        if bear: return "Objęcie Bessy", "bear"
    # Stars
    if len(df) >= 3:
        small1 = _body(o.iloc[i1], c.iloc[i1]) < 0.3 * (h.iloc[i1] - l.iloc[i1] + 1e-12)
        downtrend = c.iloc[i2] < c.iloc[i2-1] if len(df) >= 4 else (c.iloc[i2] < o.iloc[i2])
        uptrend  = c.iloc[i2] > c.iloc[i2-1] if len(df) >= 4 else (c.iloc[i2] > o.iloc[i2])
        strong_bull = (c.iloc[i0] > (o.iloc[i2] + c.iloc[i2]) / 2) and (c.iloc[i0] > o.iloc[i0])
        strong_bear = (c.iloc[i0] < (o.iloc[i2] + c.iloc[i2]) / 2) and (c.iloc[i0] < o.iloc[i0])
        if downtrend and small1 and strong_bull:
            return "Gwiazda Poranna", "bull"
        if uptrend and small1 and strong_bear:
            return "Gwiazda Wieczorna", "bear"
    # Hammer / Shooting Star
    body = _body(o.iloc[i0], c.iloc[i0]); up = _upper(o.iloc[i0], h.iloc[i0], c.iloc[i0]); lo = _lower(o.iloc[i0], l.iloc[i0], c.iloc[i0])
    if body <= 0.3 * (h.iloc[i0] - l.iloc[i0] + 1e-12):
        if lo > 2 * body and up < body:  return "Młotek", "bull"
        if up > 2 * body and lo < body:  return "Spadająca Gwiazda", "bear"
    # Doji
    if _body(o.iloc[i0], c.iloc[i0]) <= 0.1 * (h.iloc[i0] - l.iloc[i0] + 1e-12):
        return "Doji", None
    return None, None

def detect_patterns_all(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    if df is None or df.empty: 
        return pd.DataFrame(columns=["time","pattern","dir","price"])
    c = df["Close"].astype(float)
    for i in range(1, len(df)):
        name, direction = detect_last_pattern(df.iloc[:i+1])
        if name:
            out.append({"time": df.index[i], "pattern": name, "dir": direction, "price": float(c.iloc[i])})
    return pd.DataFrame(out)

def decide_signal(rsi_value: Optional[float], pattern_dir: Optional[str], rsi_buy: float, rsi_sell: float) -> str:
    if rsi_value is None or math.isnan(rsi_value):
        return "—"
    if pattern_dir == "bull" and rsi_value <= rsi_buy:
        return "🟢 KUP"
    if pattern_dir == "bear" and rsi_value >= rsi_sell:
        return "🔴 SPRZEDAJ"
    return "—"

# ================== SIDEBAR: USTAWIENIA ==================

with st.sidebar:
    source = st.radio("Źródło danych", ["Yahoo (opóźnione)", "MT5 (real‑time, lokalnie)"], index=0)
    rsi_buy = st.number_input("Próg RSI dla KUP (≤)", value=30, step=1, min_value=1, max_value=99)
    rsi_sell = st.number_input("Próg RSI dla SPRZEDAJ (≥)", value=70, step=1, min_value=1, max_value=99)
    rsi_period = st.number_input("Okres RSI", value=14, step=1, min_value=2, max_value=200)
    intervals = st.multiselect("Interwały", options=INTERVALS, default=INTERVALS)
    only_active = st.checkbox("Pokaż tylko wiersze z aktywnym sygnałem", value=False)

    # Panel połączenia MT5 (tylko gdy wybrano MT5)
    if source.startswith("MT5"):
        st.markdown("---")
        st.caption("Połącz z MT5 (lokalnie na Windows):")
        login = st.text_input("Login (numer)", value="", placeholder="np. 1234567")
        server = st.text_input("Server", value="", placeholder="np. XTB-Demo")
        password = st.text_input("Hasło", value="", type="password")
        colA, colB = st.columns(2)
        connect_clicked = colA.button("🔌 Połącz z MT5")
        disconnect_clicked = colB.button("❌ Rozłącz MT5")

        if "mt5_connected" not in st.session_state:
            st.session_state.mt5_connected = False

        if connect_clicked:
            ok = (mt5h is not None and mt5h.initialize(
                login=int(login) if login.strip().isdigit() else None,
                password=password or None,
                server=server or None,
                path_terminal=None  # zostaw puste, jeśli MT5 jest w standardowej lokalizacji
            ))
            st.session_state.mt5_connected = bool(ok)
            st.toast("Połączono z MT5 ✅" if ok else "Nie udało się połączyć z MT5 ❌", icon="✅" if ok else "❌")

        if disconnect_clicked and mt5h is not None:
            mt5h.shutdown()
            st.session_state.mt5_connected = False
            st.toast("Rozłączono z MT5", icon="📴")

# Autoodświeżanie w trybie MT5 (np. co 5 sekund)
if source.startswith("MT5"):
    st.experimental_rerun  # no-op (tylko info dla lintera)
    st.autorefresh = st.experimental_rerun  # retro-compat alias
    st.experimental_set_query_params(rt="1")  # aby nie buforował URLa
    st_autoref = st.experimental_rerun  # alias
    st.write("")  # placeholder, nic nie robi
    st.experimental_memo.clear()  # nic krytycznego — odśwież cache
    st_autorefresh = st.empty()
    st_autorefresh = st.autorefresh if hasattr(st, "autorefresh") else None
    if hasattr(st, "autorefresh"):
        st.autorefresh(interval=5000, key="mt5-autorefresh", rerun=True)

if not intervals:
    st.info("Wybierz przynajmniej jeden interwał z lewej.")
    st.stop()

# ================== TABELA ==================

rows = []
for sym in SYMBOLS:
    for itv in intervals:
        if source.startswith("MT5") and mt5h is not None and st.session_state.get("mt5_connected", False):
            df = rt_ohlc_mt5(sym, interval=itv, bars=500)
        else:
            df = yf_ohlc(sym, period="7d", interval=itv)

        if df.empty or "Close" not in df.columns:
            rows.append({"Symbol": sym, "Interwał": itv, "RSI": "—", "Formacja": "—", "Sygnał": "—", "Cena": "—", "Czas": "—", "_df": pd.DataFrame(), "_pats": pd.DataFrame()})
            continue

        rsi = rsi_series(df["Close"], period=rsi_period)
        rsi_last = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else None
        patt_name, patt_dir = detect_last_pattern(df)
        signal = decide_signal(rsi_last, patt_dir, rsi_buy, rsi_sell)

        rows.append({
            "Symbol": sym,
            "Interwał": itv,
            "RSI": "—" if rsi_last is None else round(rsi_last, 2),
            "Formacja": patt_name if patt_name else "—",
            "Sygnał": signal,
            "Cena": float(df["Close"].iloc[-1]),
            "Czas": df.index[-1],
            "_df": df,
            "_pats": detect_patterns_all(df)
        })

tbl = pd.DataFrame(rows, columns=["Symbol","Interwał","RSI","Formacja","Sygnał","Cena","Czas","_df","_pats"])
if only_active:
    tbl = tbl[tbl["Sygnał"].isin(["🟢 KUP","🔴 SPRZEDAJ"])]

st.dataframe(tbl.drop(columns=["_df","_pats"]), use_container_width=True, hide_index=True)

# ================== SZCZEGÓŁY: ŚWIECOWY + MARKERY ==================

st.subheader("Szczegóły")
if tbl.empty:
    st.info("Brak wyników do podglądu.")
    st.stop()

opts = [f"{r.Symbol} × {r.Interwał}" for r in tbl.itertuples(index=False)]
pick = st.selectbox("Wybierz wiersz", options=opts, index=0)
sym, itv = [x.strip() for x in pick.split("×")]

row = tbl[(tbl["Symbol"] == sym) & (tbl["Interwał"] == itv)].iloc[0]
df_sel: pd.DataFrame = row["_df"]
pat_sel: pd.DataFrame = row["_pats"]

def plot_candles_mpf(df: pd.DataFrame, pats: pd.DataFrame, title: str):
    ap = []
    if pats is not None and not pats.empty:
        bulls = pats[pats["dir"] == "bull"]
        bears = pats[pats["dir"] == "bear"]
        if not bulls.empty:
            ap.append(mpf.make_addplot(bulls.set_index("time")["price"], type="scatter", markersize=80, marker="^", color="g"))
        if not bears.empty:
            ap.append(mpf.make_addplot(bears.set_index("time")["price"], type="scatter", markersize=80, marker="v", color="r"))
    fig, axlist = mpf.plot(
        df, type="candle", style="charles", addplot=ap, returnfig=True, figsize=(10, 5),
        title=title, volume=False
    )
    st.pyplot(fig, clear_figure=True)

def plot_candles_fallback(df: pd.DataFrame, pats: pd.DataFrame, title: str):
    o = df["Open"].values; h = df["High"].values; l = df["Low"].values; c = df["Close"].values
    x = np.arange(len(df)); width = 0.6
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(len(df)):
        ax.vlines(x[i], l[i], h[i], color="black", linewidth=1)
        color = "green" if c[i] >= o[i] else "red"
        lower = min(o[i], c[i]); height = abs(c[i] - o[i])
        ax.add_patch(plt.Rectangle((x[i]-width/2, lower), width, max(height, 1e-6), color=color, alpha=0.85))
    if pats is not None and not pats.empty:
        pos = {t: i for i, t in enumerate(df.index)}
        bulls = pats[pats["dir"] == "bull"]; bears = pats[pats["dir"] == "bear"]
        bx = [pos.get(t) for t in bulls["time"] if t in pos]; by = bulls["price"].tolist()
        rx = [pos.get(t) for t in bears["time"] if t in pos]; ry = bears["price"].tolist()
        if bx and by: ax.scatter(bx, by, marker="^", s=80, color="green", label="Formacja wzrostowa")
        if rx and ry: ax.scatter(rx, ry, marker="v", s=80, color="red",   label="Formacja spadkowa")
        if (bx and by) or (rx and ry): ax.legend(loc="best", fontsize=8)
    ax.set_title(title); ax.grid(True, alpha=0.25); ax.set_xlim(-0.5, len(df)-0.5)
    st.pyplot(fig, clear_figure=True)

if df_sel.empty:
    st.info("Brak danych do wykresu.")
else:
    title = f"{sym} • {itv} • {'MT5' if source.startswith('MT5') and st.session_state.get('mt5_connected', False) else 'Yahoo'}"
    if _HAS_MPF: plot_candles_mpf(df_sel, pat_sel, title)
    else:        plot_candles_fallback(df_sel, pat_sel, title)

    close = pd.to_numeric(df_sel["Close"], errors="coerce").dropna()
    last_rsi_val = rsi_series(close).dropna()
    last_rsi_val = float(last_rsi_val.iloc[-1]) if not last_rsi_val.empty else None
    c1, c2, c3 = st.columns(3)
    c1.metric("RSI (ostatnia)", "—" if last_rsi_val is None else f"{last_rsi_val:.2f}")
    c2.metric("Cena (Close)", f"{float(close.iloc[-1]):.6f}")
    c3.metric("Ostatnia świeca", str(close.index[-1]))

    st.markdown("**Ostatnie formacje (z widocznego zakresu)**")
    if pat_sel is not None and not pat_sel.empty:
        view = pat_sel.sort_values("time", ascending=False).head(12)
        view["time"] = view["time"].astype(str)
        st.dataframe(view.rename(columns={"time":"Czas","pattern":"Formacja","dir":"Kierunek","price":"Cena"}), use_container_width=True, hide_index=True)
    else:
        st.info("Brak wykrytych formacji.")
