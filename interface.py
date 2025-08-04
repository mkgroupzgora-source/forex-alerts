import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
import datetime
import time

analyzer = SentimentIntensityAnalyzer()

assets = [
    "EURUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD",
    "XAUUSD", "XAGUSD", "WTIUSD", "BrentUSD",
    "BTCUSD", "USDPLN"
]

rss_feeds = {
    "EURUSD": ["https://www.investing.com/rss/news_25.rss"],
    "AUDUSD": ["https://www.investing.com/rss/news_25.rss"],
    "USDCHF": ["https://www.investing.com/rss/news_25.rss"],
    "USDCAD": ["https://www.investing.com/rss/news_25.rss"],
    "NZDUSD": ["https://www.investing.com/rss/news_25.rss"],
    "XAUUSD": ["https://www.investing.com/rss/news_301.rss"],
    "XAGUSD": ["https://www.investing.com/rss/news_301.rss"],
    "WTIUSD": ["https://www.investing.com/rss/news_92.rss"],
    "BrentUSD": ["https://www.investing.com/rss/news_92.rss"],
    "BTCUSD": ["https://www.investing.com/rss/news_301.rss"],
    "USDPLN": ["https://www.investing.com/rss/news_25.rss"]
}

def get_rsi_signal(symbol):
    try:
        df = yf.download(symbol, period="7d", interval="1h", progress=False)
        if df.empty or len(df) < 15:
            return None, "Brak danych"
        df.dropna(inplace=True)
        rsi = RSIIndicator(df["Close"]).rsi().iloc[-1]
        if rsi < 30:
            return rsi, "Kupno"
        elif rsi > 70:
            return rsi, "Sprzedaż"
        else:
            return rsi, "Brak sygnału"
    except Exception as e:
        return None, "Błąd"

def get_news_sentiment(feed_urls):
    news_items = []
    try:
        for url in feed_urls:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                title = entry.title
                link = entry.link
                score = analyzer.polarity_scores(title)['compound']
                sentiment = "Pozytywny" if score > 0.2 else "Negatywny" if score < -0.2 else "Neutralny"
                news_items.append({
                    "tytuł": title,
                    "link": link,
                    "sentyment": sentiment
                })
        return news_items
    except Exception as e:
        return []

def main():
    st.title("📈 SEP Forex Signals")
    st.subheader("Analiza RSI + NLP (VADER) + przypisane newsy do aktywów.")
    if st.button("🔄 Odśwież dane teraz"):
        st.rerun()

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"Ostatnia aktualizacja: **{now}**")

    st.subheader("📊 Sygnały RSI")

    signals = []
    for asset in assets:
        rsi, signal = get_rsi_signal(asset)
        signals.append({
            "Aktywum": asset,
            "RSI": round(rsi, 2) if rsi else "None",
            "Sygnał": signal,
            "Godzina": now[-8:]
        })

    st.dataframe(pd.DataFrame(signals))

    st.subheader("📰 Ważne wiadomości")
    for asset in assets:
        st.markdown(f"#### {asset}")
        news = get_news_sentiment(rss_feeds.get(asset, []))
        if not news:
            st.write("Brak newsów lub błąd pobierania.")
            continue
        for item in news:
            st.markdown(f"- [{item['tytuł']}]({item['link']}) – *{item['sentyment']}*")

if __name__ == "__main__":
    main()