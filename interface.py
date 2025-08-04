import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="SEP Forex Signals", layout="centered")

# Mapa tickerów Yahoo Finance
symbol_map = {
    "EURUSD": "EURUSD=X",
    "AUDUSD": "AUDUSD=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "CAD=X",
    "NZDUSD": "NZDUSD=X",
    "XAUUSD": "XAUUSD=X",
    "XAGUSD": "XAGUSD=X",
    "WTIUSD": "CL=F",
    "BrentUSD": "BZ=F",
    "BTCUSD": "BTC-USD",
    "USDPLN": "USDPLN=X"
}

# Funkcja RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Funkcja określająca sygnał
def get_rsi_signal(rsi):
    if pd.isna(rsi):
        return "Brak danych"
    elif rsi < 30:
        return "KUP"
    elif rsi > 70:
        return "SPRZEDAJ"
    else:
        return "BRAK SYGNAŁU"

# Funkcja do analizy sentymentu VADER
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return "Pozytywny"
    elif score['compound'] <= -0.05:
        return "Negatywny"
    else:
        return "Neutralny"

# Funkcja do pobierania newsów RSS
def get_news():
    urls = {
        "Investing": "https://www.investing.com/rss/news_25.rss",
        "ForexFactory": "https://www.forexfactory.com/news.atom",
        "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/"
    }

    all_news = []

    for source, url in urls.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                title = entry.title
                published = entry.get("published", "")
                sentiment = analyze_sentiment(title)
                all_news.append({
                    "Źródło": source,
                    "Tytuł": title,
                    "Data": published,
                    "Sentyment": sentiment
                })
        except Exception as e:
            all_news.append({"Źródło": source, "Tytuł": f"Błąd: {e}", "Data": "", "Sentyment": "Brak"})

    return pd.DataFrame(all_news)

# Główna funkcja
def main():
    st.title("📈 SEP Forex Signals")
    st.markdown("Analiza RSI + NLP (VADER) + przypisane newsy do aktywów.")
    if st.button("🔄 Odśwież dane teraz"):
        st.rerun()

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"Ostatnia aktualizacja: **{now}**")

    rsi_data = []

    for asset, ticker in symbol_map.items():
        try:
            df = yf.download(ticker, period="1mo", interval="1d")
            if df.empty:
                rsi = None
            else:
                df['RSI'] = calculate_rsi(df)
                rsi = df['RSI'].iloc[-1]
            signal = get_rsi_signal(rsi)
        except Exception:
            rsi = None
            signal = "BŁĄD"

        rsi_data.append({
            "Aktywum": asset,
            "RSI": round(rsi, 2) if rsi else None,
            "Sygnał": signal,
            "Godzina": now.split(" ")[1]
        })

    st.subheader("📊 Sygnały RSI")
    st.dataframe(pd.DataFrame(rsi_data))

    st.subheader("📰 Ważne wiadomości (Investing, ForexFactory, MarketWatch)")
    news_df = get_news()
    st.dataframe(news_df)

if __name__ == "__main__":
    main()