import streamlit as st
import pandas as pd
import datetime
from data_fetcher import fetch_data
from strategy import analyze_strategy
from news_parser import fetch_news
from sentiment_analysis import analyze_sentiment
from config import SYMBOLS

st.set_page_config(layout="wide")
st.title("📈 SEP Forex Signals + News + Candle Patterns")

st.sidebar.header("⚙️ Ustawienia")
refresh = st.sidebar.button("🔄 Odśwież dane")

st.sidebar.markdown("---")
st.sidebar.write("Wybierz parę do analizy szczegółowej:")
selected_pair = st.sidebar.selectbox("Para walutowa", SYMBOLS)

# Główna tabela z sygnałami
data = []
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

for symbol in SYMBOLS:
    df = fetch_data(symbol)
    strategy = analyze_strategy(df)
    data.append({
        "Para": symbol,
        "RSI": strategy["rsi"],
        "RSI Sygnał": strategy["rsi_signal"],
        "Formacja": strategy["pattern"],
        "Sygnał ze świec": strategy["pattern_signal"],
        "Godzina": timestamp
    })

df_signals = pd.DataFrame(data)
st.subheader("📊 Sygnały rynkowe (RSI + formacje świecowe)")
st.dataframe(df_signals, use_container_width=True)

# Szczegóły dla wybranej pary
st.subheader(f"🔍 Szczegóły dla: {selected_pair}")
df_selected = fetch_data(selected_pair)
strategy = analyze_strategy(df_selected)

st.write(f"**RSI:** {strategy['rsi']} — _{strategy['rsi_signal']}_")
st.write(f"**Formacja świecowa:** {strategy['pattern']} — _{strategy['pattern_signal']}_")
st.line_chart(df_selected['Close'])

# Sekcja newsów
st.subheader("📰 Ważne wiadomości (Investing, MarketWatch, ForexFactory)")

news = fetch_news()
for n in news:
    sent = analyze_sentiment(n["title"] + " " + n["summary"])
    with st.expander(f"{n['title']} ({n['source']}, {sent})"):
        st.write(n["summary"])
        st.markdown(f"[Link do źródła]({n['link']})")

