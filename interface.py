# interface.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ====== próby importu modułów pomocniczych (miękkie fallbacki) ===============

# data_fetcher: ma zwracać OHLC dla par w interwałach M15/M30/H1
try:
    from data_fetcher import fetch_ohlc  # (pair:str, interval:str, lookback:int)->pd.DataFrame
except Exception:
    fetch_ohlc = None  # type: ignore

# candle patterns: wykrywanie formacji; oczekujemy funkcji detect_patterns(df)->str|None
try:
    from candle_patterns import detect_patterns
except Exception:
    def detect_patterns(df: pd.DataFrame) -> Optional[str]:
        # prosty fallback (brak formacji)
        return None

# news parser(s): zwraca listę newsów {title, link, source, published, mapped_symbol}
_news_module_loaded = False
NEWS_SOURCES_NAME = None
try:
    from news_parser import fetch_all_news  # type: ignore
    _news_module_loaded = True
    NEWS_SOURCES_NAME = "news_parser"
except Exception:
    try:
        from news_parsers import fetch_all_news  # type: ignore
        _news_module_loaded = True
        NEWS_SOURCES_NAME = "news_parsers"
    except Exception:
        _news_module_loaded = False
        fetch_all_news = None  # type: ignore

# sentiment (VADER)
try:
    from sentiment_analysis import score_sentiment  # (text)->float [-1..1]
except Exception:
    def score_sentiment(text: str) -> float:
        return 0.0

# mt5 handler (nasz nowy bezpieczny wrapper)
try:
    from mt5_handler import get_handler
except Exception:
    get_handler = None  # type: ignore


# ========================= USTAWIENIA / CONFIG ===============================

APP_DIR = Path(__file__).parent
CONFIG_PATH = APP_DIR / "config.json"

DEFAULT_CONFIG = {
    "pairs": [
        "EURUSD", "GBPUSD", "USDCHF", "USDJPY", "USDCNH", "USDRUB",
        "AUDUSD", "NZDUSD", "USDCAD", "USDSEK", "USDPLN",
        "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDPLN",
        "CADCHF", "CADJPY", "CADPLN", "CHFJPY", "CHFPLN", "CNHJPY",
        "EURAUD", "EURCAD", "EURCHF", "EURCNH", "EURGBP", "EURJPY",
        "EURNZD", "EURPLN", "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY",
        "GBPPLN",
        "XAGUSD", "XAUUSD", "XPDUSD", "XPTUSD"
    ],
    "intervals": ["15m", "30m", "1h"],  # M15, M30, H1
    "rsi_buy_threshold": 30,
    "rsi_sell_threshold": 70,
    "lookback_bars": 250,       # ile świec ściągać na wykres/RSI
    "tz": "Europe/Warsaw",
    # parametry trailing (domyślne)
    "trailing": {
        "step_pips": 10.0,
        "arm_after_pips": 10.0,
        "pip_size_fx": 0.0001,
        "pip_size_jpy": 0.01,
        "pip_size_metals": 0.1
    },
    # czy pokazywać MT5 przyciski (w chmurze i tak się ukryją gdy MT5 niedostępny)
    "mt5": {
        "show_controls": True,
        "default_volume": 0.1
    }
}

def load_config() -> Dict:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # scal z domyślną (dla brakujących kluczy)
            merged = DEFAULT_CONFIG.copy()
            for k, v in data.items():
                merged[k] = v
            # trailing/mt5 merge
            for block in ("trailing", "mt5"):
                if block in DEFAULT_CONFIG and block in data:
                    merged[block] = {**DEFAULT_CONFIG[block], **data[block]}
            return merged
        except Exception:
            pass
    return DEFAULT_CONFIG

CONFIG = load_config()


# ====================== NARZĘDZIA: cache, RSI, sygnał ========================

@st.cache_data(ttl=300, show_spinner=False)
def cached_fetch(pair: str, interval: str, lookback: int) -> Optional[pd.DataFrame]:
    if fetch_ohlc is None:
        return None
    try:
        df = fetch_ohlc(pair, interval=interval, lookback=lookback)
        if df is None or df.empty:
            return None
        # oczekujemy kolumn: ["open","high","low","close","volume"]
        # i indeksu w czasie
        return df
    except Exception:
        return None

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1.0 * delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, 1e-9))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def pick_signal(rsi_value: float, pattern: Optional[str], rsi_buy: int, rsi_sell: int) -> str:
    """
    Sygnał pojawia się TYLKO jeśli mamy jednocześnie:
      - formację świecową (pattern != None)
      - rsi poza progami
    """
    if pattern is None:
        return "BRAK"

    if rsi_value <= rsi_buy:
        return "KUP"
    if rsi_value >= rsi_sell:
        return "SPRZEDAJ"
    return "OCZEKAJ"


# =========================== NEWSY + SENTYMENT ===============================

@st.cache_data(ttl=600, show_spinner=False)
def cached_news() -> List[Dict]:
    if not _news_module_loaded or fetch_all_news is None:
        return []
    try:
        return fetch_all_news()  # lista: {title, link, source, published, mapped_symbol}
    except Exception:
        return []

def match_news_for_symbol(all_news: List[Dict], symbol: str) -> List[Dict]:
    s = symbol.upper()
    return [n for n in all_news if str(n.get("mapped_symbol", "")).upper() == s]


# =========================== POMOC: pip-size dla trailing ====================

def infer_pip_size(symbol: str) -> float:
    s = symbol.upper()
    if s.endswith("JPY") or s.startswith("JPY"):
        return CONFIG["trailing"]["pip_size_jpy"]
    if s in ("XAUUSD", "XAGUSD", "XPDUSD", "XPTUSD"):
        return CONFIG["trailing"]["pip_size_metals"]
    return CONFIG["trailing"]["pip_size_fx"]


# =========================== MT5 init ========================================

MT5_AVAILABLE = False
mt5 = None
if get_handler is not None:
    # W chmurze i tak stanie się "unavailable" -> UI to pokaże
    mt5 = get_handler(dry_run=False)
    MT5_AVAILABLE = mt5.is_available()
else:
    MT5_AVAILABLE = False


# =========================== UI / APLIKACJA ==================================

st.set_page_config(page_title="SEP Forex Signals", page_icon="📊", layout="wide")

st.title("📊 SEP Forex Signals")
st.caption("Analiza RSI + formacje świecowe + NLP (VADER) + przypisane newsy do aktywów.")

# Panel boczny – ustawienia strategii / interwały / MT5
with st.sidebar:
    st.header("⚙️ Ustawienia strategii")
    pairs = CONFIG["pairs"]
    intervals = CONFIG["intervals"]
    rsi_buy_def = CONFIG["rsi_buy_threshold"]
    rsi_sell_def = CONFIG["rsi_sell_threshold"]
    lookback = CONFIG["lookback_bars"]

    sel_intervals = st.multiselect("Interwały", options=["15m", "30m", "1h"], default=intervals)
    rsi_buy = st.slider("RSI – próg KUP", 5, 50, rsi_buy_def, 1)
    rsi_sell = st.slider("RSI – próg SPRZEDAJ", 50, 95, rsi_sell_def, 1)

    st.markdown("---")
    st.subheader("📈 Trailing SL (domyślne)")
    step_pips = st.number_input("Krok (pips)", value=float(CONFIG["trailing"]["step_pips"]), step=1.0, min_value=1.0)
    arm_after = st.number_input("Uzbrój po zysku (pips)", value=float(CONFIG["trailing"]["arm_after_pips"]), step=1.0, min_value=1.0)

    st.markdown("---")
    if MT5_AVAILABLE and CONFIG["mt5"]["show_controls"]:
        st.subheader("🧪 MT5 (logowanie opcjonalne)")
        login_id = st.text_input("Login", value="", placeholder="np. 5038837590")
        password = st.text_input("Hasło", value="", type="password")
        server = st.text_input("Serwer", value="MetaQuotes-Demo")
        colm = st.columns(2)
        if colm[0].button("Połącz z MT5", use_container_width=True):
            c = mt5.connect()
            if not c.connected:
                st.error(c.message)
            else:
                st.success(c.message)
        if colm[1].button("Zaloguj", use_container_width=True):
            if not login_id or not password or not server:
                st.warning("Uzupełnij login/hasło/serwer.")
            else:
                try:
                    lid = int(login_id)
                except Exception:
                    st.error("Login musi być liczbą.")
                else:
                    r = mt5.login(login=lid, password=password, server=server)
                    st.success(r.message) if r.connected else st.error(r.message)
        st.caption("Jeśli jesteś na Streamlit Cloud, MT5 jest niedostępny – użyj aplikacji lokalnie.")

st.markdown("—")
refresh = st.button("🔄 Odśwież dane teraz", type="primary")
if refresh:
    # Inwaliduj cache
    cached_fetch.clear()
    cached_news.clear()
    st.toast("Odświeżono dane.", icon="✅")

st.caption(f"Ostatnia aktualizacja: {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("## 📑 Sygnały")

# Pobierz newsy (1x)
all_news = cached_news()

# Tabela wyników
rows: List[Dict] = []

for pair in CONFIG["pairs"]:
    for interval in sel_intervals:
        df = cached_fetch(pair, interval, lookback)
        if df is None or df.empty or "close" not in df.columns:
            rows.append({
                "Aktywum": pair, "Interwał": interval, "RSI": None,
                "Formacja": None, "Sygnał": "Brak danych", "Godzina": "-",
                "Sentyment": None
            })
            continue

        # RSI
        rsi = compute_rsi(df["close"]).iloc[-1]
        rsi_rounded = round(float(rsi), 2)

        # Formacja
        try:
            pattern = detect_patterns(df)
        except Exception:
            pattern = None

        # Sygnał
        signal = pick_signal(rsi_rounded, pattern, rsi_buy, rsi_sell)

        # Ostatni czas
        ts = df.index[-1]
        if isinstance(ts, pd.Timestamp):
            last_ts = ts.tz_localize(None).strftime("%Y-%m-%d %H:%M")
        else:
            last_ts = str(ts)

        # Sentyment (na podstawie newsów przypisanych do pary)
        sym_news = match_news_for_symbol(all_news, pair)
        sentiments = [score_sentiment(n.get("title", "")) for n in sym_news]
        sent_val = round(float(sum(sentiments) / len(sentiments)), 3) if sentiments else 0.0

        rows.append({
            "Aktywum": pair, "Interwał": interval, "RSI": rsi_rounded,
            "Formacja": pattern or "-", "Sygnał": signal, "Godzina": last_ts,
            "Sentyment": sent_val
        })

df_tbl = pd.DataFrame(rows)

# ładny kolor sygnału
def color_signal(val: str):
    if val == "KUP":
        return "color: #0f8b0f; font-weight: 700;"
    if val == "SPRZEDAJ":
        return "color: #c00000; font-weight: 700;"
    if val == "Brak danych":
        return "color: #999;"
    return "color: inherit;"

st.dataframe(
    df_tbl.style.applymap(color_signal, subset=["Sygnał"]),
    use_container_width=True, height=420
)

st.markdown("## 📰 Ważne wiadomości")
if not all_news:
    st.info("Brak newsów (lub parser niedostępny w tym środowisku).")
else:
    # pokaż kilka najnowszych
    show_n = min(20, len(all_news))
    for i, item in enumerate(all_news[:show_n], 1):
        title = item.get("title", "")
        src = item.get("source", "")
        link = item.get("link", "")
        mapped = item.get("mapped_symbol", "")
        published = item.get("published", "")
        sent = score_sentiment(title)
        tone = "✅ pozytywny" if sent > 0.15 else ("⚠️ negatywny" if sent < -0.15 else "🟨 neutralny")
        st.markdown(f"**{i}. [{title}]({link})**  \n*{src}* · {published} · `{mapped}` · sentyment: **{tone} ({sent:.2f})**")

st.markdown("## 🔍 Szczegóły instrumentu")

# Wybór jednej pary do podglądu szczegółów/handlu
colA, colB = st.columns([1, 3])
with colA:
    pair_pick = st.selectbox("Wybierz aktywo", CONFIG["pairs"])
    interval_pick = st.selectbox("Interwał", sel_intervals)

with colB:
    df_detail = cached_fetch(pair_pick, interval_pick, lookback)
    if df_detail is None or df_detail.empty:
        st.warning("Brak danych do wizualizacji dla tego instrumentu / interwału.")
    else:
        # mini-wykres
        st.line_chart(df_detail["close"], height=260)
        # najnowsze RSI i formacja
        rsi_d = compute_rsi(df_detail["close"]).iloc[-1]
        pattern_d = detect_patterns(df_detail) if detect_patterns else None
        signal_d = pick_signal(float(rsi_d), pattern_d, rsi_buy, rsi_sell)
        st.caption(f"RSI: **{float(rsi_d):.2f}** · Formacja: **{pattern_d or '-'}** · Sygnał: **{signal_d}**")

        # NEWS dla tej pary
        sy_news = match_news_for_symbol(all_news, pair_pick)
        if sy_news:
            with st.expander("Powiązane newsy (ostatnie)"):
                for n in sy_news[:10]:
                    sent = score_sentiment(n.get("title", ""))
                    tone = "✅" if sent > 0.15 else ("⚠️" if sent < -0.15 else "🟨")
                    st.markdown(f"- [{n.get('title')}]({n.get('link')}) · {n.get('source')} · {n.get('published')} · sent: {tone} ({sent:.2f})")
        else:
            st.caption("Brak przypisanych newsów do tej pary.")

        # ======================== MT5 – manualne zlecenia =====================
        if MT5_AVAILABLE and CONFIG["mt5"]["show_controls"]:
            st.markdown("### 🧭 Handel (MT5 – manualnie)")
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            volume = col1.number_input("Wolumen", value=float(CONFIG["mt5"]["default_volume"]), step=0.01, min_value=0.01)
            price_info = df_detail["close"].iloc[-1]
            col1.caption(f"Ostatnia cena: {price_info:.5f}" if isinstance(price_info, (float, int)) else str(price_info))

            # ustawienia wstępnego SL/TP
            pip = infer_pip_size(pair_pick)
            sl_pips = col2.number_input("SL (pips)", value=20.0, step=1.0, min_value=1.0)
            tp_pips = col3.number_input("TP (pips)", value=40.0, step=1.0, min_value=1.0)

            # przyciski
            bt_buy = col2.button("BUY", type="primary", use_container_width=True)
            bt_sell = col3.button("SELL", type="secondary", use_container_width=True)

            # trailing
            with st.expander("Trailing SL/TP"):
                tcol = st.columns(3)
                tr_step = tcol[0].number_input("Krok (pips)", value=float(step_pips), step=1.0, min_value=1.0)
                tr_arm = tcol[1].number_input("Uzbrój po (pips)", value=float(arm_after), step=1.0, min_value=1.0)
                tr_max_tp = tcol[2].text_input("Stały TP (opcjonalnie)", value="", placeholder="np. 1.0950")
                tr_start = st.button("Start Trailing", use_container_width=True, key="trail_start_btn")
                tr_stop_ticket = st.text_input("Zatrzymaj trailing dla ticketu", value="", placeholder="ticket id")
                tr_stop = st.button("Stop Trailing", use_container_width=True, key="trail_stop_btn")

            # zlecenie
            if bt_buy or bt_sell:
                if not mt5.connected:
                    st.warning("Najpierw połącz i zaloguj się do MT5 (panel boczny).")
                else:
                    # wylicz SL/TP w cenie
                    curr_price = float(df_detail["close"].iloc[-1])
                    side = "BUY" if bt_buy else "SELL"
                    if side == "BUY":
                        sl = curr_price - sl_pips * pip
                        tp = curr_price + tp_pips * pip
                    else:
                        sl = curr_price + sl_pips * pip
                        tp = curr_price - tp_pips * pip

                    res = mt5.place_market_order(
                        symbol=pair_pick,
                        volume=float(volume),
                        side=side,
                        sl=float(sl),
                        tp=float(tp),
                        comment=f"SEP {side} {pair_pick}"
                    )
                    if res.success:
                        st.success(f"{side} wysłane. Ticket: {res.ticket}")
                        st.json(res.request)
                        # trailing on start?
                        if tr_start and res.ticket:
                            tp_val = None
                            if tr_max_tp:
                                try:
                                    tp_val = float(tr_max_tp)
                                except Exception:
                                    tp_val = None
                            msg = mt5.start_trailing(
                                ticket=int(res.ticket),
                                symbol=pair_pick,
                                side=side,
                                step_pips=float(tr_step),
                                arm_after_pips=float(tr_arm),
                                pip_size=float(pip),
                                max_tp=tp_val,
                                update_secs=2.0
                            )
                            st.info(msg)
                    else:
                        st.error(f"Zlecenie nieudane: {res.message}")
                        if res.request:
                            with st.expander("Szczegóły requestu"):
                                st.json(res.request)
                        if res.result_raw:
                            with st.expander("Odpowiedź MT5"):
                                st.json(res.result_raw)

            if tr_stop and tr_stop_ticket.strip():
                try:
                    tkt = int(tr_stop_ticket.strip())
                    msg = mt5.stop_trailing(tkt)
                    st.info(msg)
                except Exception:
                    st.warning("Ticket musi być liczbą.")

        else:
            st.info("MT5 niedostępny w tym środowisku lub ukryty w konfiguracji – handel wyłączony.")
