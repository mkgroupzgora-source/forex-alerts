import streamlit as st
import pandas as pd
import datetime
import yfinance as yf
from ta.momentum import RSIIndicator
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -------------------- Konfiguracja interfejsu --------------------
st.set_page_config(page_title="SEP Forex Signals", page_icon="📊")
st.title("📊 SEP Forex Signals – Dashboard")
st.caption("Analiza RSI + NLP (VADER) + przypisane newsy do aktywów.")

# -------------------- Panel ustawień RSI --------------------
st.sidebar.header("⚙️ Ustawienia RSI")
rsi_buy = st.sidebar.slider("Próg RSI – KUP", min_value=10, max_value=50, value=30)
rsi_sell = st.sidebar.slider("Próg RSI – SPRZEDAJ", min_value=50, max_value=90, value=70)

# -------------------- Lista aktywów --------------------
assets = {
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "USDJPY=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X",
    "XAU/USD": "XAUUSD=X",
    "XAG/USD": "XAGUSD=X",
    "WTI/USD": "CL=F",
    "Brent/USD": "BZ=F",
    "BTC/USD": "BTC-USD",
    "USD/PLN": "USDPLN=X"
}

# -------------------- RSI SYGNAŁ --------------------
def fetch_rsi(ticker):
    try:
        df = yf.download(ticker, period="7d", interval="1h", progress=False)
        df.dropna(inplace=True)
        rsi = RSIIndicator(df["Close"]).rsi()
        last_rsi = round(rsi.dropna().iloc[-1], 2)
        signal = "KUP" if last_rsi < rsi_buy else "SPRZEDAJ" if last_rsi > rsi_sell else "BRAK"
        return last_rsi, signal
    except:
        return None, "BŁĄD"

# -------------------- Odświeżanie --------------------
if st.button("🔁 Odśwież dane teraz"):
    st.session_state["last_refresh"] = datetime.datetime.now()
last_update = st.session_state.get("last_refresh", datetime.datetime.now())
st.caption(f"Ostatnia aktualizacja: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")

# -------------------- Tabela RSI --------------------
results = []
for name, symbol in assets.items():
    rsi, signal = fetch_rsi(symbol)
    results.append({
        "Aktywum": name,
        "RSI": rsi,
        "Sygnał": signal,
        "Godzina": last_update.strftime("%H:%M")
    })
df = pd.DataFrame(results)

def color_signal(val):
    if val == "KUP":
        return "color: green"
    elif val == "SPRZEDAJ":
        return "color: red"
    elif val == "BŁĄD":
        return "color: orange"
    return ""

st.subheader("📈 Sygnały RSI")
st.dataframe(df.style.applymap(color_signal, subset=["Sygnał"]))

# -------------------- NLP VADER: analiza sentymentu --------------------
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.2:
        return "✅", "green"
    elif score <= -0.2:
        return "❌", "red"
    else:
        return "⚪", "gray"

# -------------------- Przypisywanie newsów do aktywów --------------------
asset_keywords = {
    "EUR/USD": ["eur", "euro"],
    "USD/JPY": ["jpy", "yen"],
    "GBP/USD": ["gbp", "pound"],
    "AUD/USD": ["aud", "australian"],
    "USD/CHF": ["chf", "swiss"],
    "USDCAD": ["cad", "canadian"],
    "NZD/USD": ["nzd", "kiwi"],
    "XAU/USD": ["gold", "xau"],
    "XAG/USD": ["silver", "xag"],
    "WTI/USD": ["wti", "crude", "oil"],
    "Brent/USD": ["brent", "oil"],
    "BTC/USD": ["btc", "bitcoin", "crypto"],
    "USD/PLN": ["pln", "poland", "zloty"]
}

def match_asset(news_text):
    matched = []
    news_text = news_text.lower()
    for asset, keywords in asset_keywords.items():
        if any(k in news_text for k in keywords):
            matched.append(asset)
    return matched if matched else ["Ogólne"]

# -------------------- Parser RSS newsów --------------------
st.subheader("📰 Ważne wiadomości (Investing, ForexFactory, MarketWatch)")

news_sources = {
    "Investing.com": "https://www.investing.com/rss/news_25.rss",
    "Forex Factory": "https://www.forexfactory.com/ffcal_week_this.xml",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/"
}

for source, url in news_sources.items():
    st.markdown(f"### {source}")
    feed = feedparser.parse(url)
    if not feed.entries:
        st.warning("Brak danych z tego źródła.")
        continue

    for entry in feed.entries[:5]:
        title = entry.title
        sentiment_icon, sentiment_color = analyze_sentiment(title)
        assets_matched = match_asset(title)
        st.markdown(
            f"<span style='color:{sentiment_color}'>{sentiment_icon}</span> "
            f"<strong>{title}</strong> "
            f"<br/><small>Dotyczy: {', '.join(assets_matched)}</small><br/>",
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("SEP Forex Signals © 2025 – analiza RSI + VADER NLP + newsy fundamentalne")
