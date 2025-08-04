import streamlit as st
import pandas as pd
import datetime
from login import require_login

# --- Konfiguracja strony ---
st.set_page_config(
    page_title="SEP Forex Signals",
    page_icon="📊",
    layout="centered"
)

# --- Logowanie ---
require_login()

# --- Nagłówek ---
st.title("📊 SEP Forex Signals – Dashboard")
st.caption("Wersja testowa z RSI i analizą fundamentalną.")
st.success(f"Zalogowano jako {st.session_state.get('username', 'Użytkownik')}")

# --- Przykładowe dane RSI ---
st.subheader("🧮 Ostatnie sygnały RSI")

# Przykładowe dane
data = {
    "Para walutowa": ["EUR/USD", "USD/JPY", "GBP/USD", "XAU/USD", "USD/PLN"],
    "RSI": [28.5, 71.2, 49.8, 75.3, 26.9],
    "Sygnał": ["KUP", "SPRZEDAJ", "BRAK", "SPRZEDAJ", "KUP"],
    "Data": [datetime.date.today()] * 5
}

df = pd.DataFrame(data)

# Kolorowanie sygnałów
def highlight_signal(val):
    if val == "KUP":
        return "color: green; font-weight: bold"
    elif val == "SPRZEDAJ":
        return "color: red; font-weight: bold"
    return ""

st.dataframe(df.style.applymap(highlight_signal, subset=["Sygnał"]))

# --- Wykres RSI ---
st.subheader("📈 Wizualizacja RSI (testowa)")

rsi_chart_data = pd.DataFrame({
    "EUR/USD": [30, 33, 40, 45, 38, 29],
    "USD/JPY": [60, 65, 68, 70, 73, 71],
}, index=pd.date_range(end=datetime.date.today(), periods=6))

st.line_chart(rsi_chart_data)

# --- Wiadomości fundamentalne (mock) ---
st.subheader("📰 Analiza fundamentalna (testowa)")

news = [
    "🔺 Rezerwa Federalna sygnalizuje możliwą podwyżkę stóp procentowych we wrześniu.",
    "📉 Słabsze dane PMI z Niemiec pogłębiają obawy o recesję w strefie euro.",
    "🪙 Złoto traci na wartości po umocnieniu dolara amerykańskiego.",
]

for n in news:
    st.info(n)

# --- Stopka ---
st.markdown("---")
st.caption("SEP Forex Signals © 2025 | wersja testowa")
