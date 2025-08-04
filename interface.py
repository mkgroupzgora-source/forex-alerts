
import streamlit as st
import pandas as pd
import datetime
from login import require_login
import random

require_login()

st.set_page_config(page_title="SEP Forex Signals", page_icon="📊", layout="centered")
st.title("📊 SEP Forex Signals – Dashboard")

st.success(f"Zalogowano jako {st.session_state.get('username', 'Użytkownik')}")
st.caption("Wersja z odświeżaniem i dodatkowymi aktywami")

st.subheader("📈 Sygnały RSI dla par walutowych i aktywów")

assets = [
    "EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD", "USD/CHF", "USDCAD", "NZD/USD",
    "XAU/USD", "XAG/USD", "WTI/USD", "BTC/USD", "USD/PLN"
]

# Przycisk ręcznego odświeżania
if st.button("🔁 Odśwież dane teraz"):
    st.session_state['last_refresh'] = datetime.datetime.now()

# Pobranie czasu ostatniej aktualizacji
now = datetime.datetime.now()
last_refresh = st.session_state.get('last_refresh', now)
st.caption(f"Dane odświeżono: {last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")

# Generowanie przykładowych danych RSI
data = {
    "Aktywum": [],
    "RSI": [],
    "Sygnał": [],
    "Godzina": []
}

for asset in assets:
    rsi = round(random.uniform(20, 80), 2)
    signal = "KUP" if rsi < 30 else "SPRZEDAJ" if rsi > 70 else "BRAK"
    data["Aktywum"].append(asset)
    data["RSI"].append(rsi)
    data["Sygnał"].append(signal)
    data["Godzina"].append(last_refresh.strftime("%H:%M"))

df = pd.DataFrame(data)

def highlight(val):
    if val == "KUP":
        return "color: green"
    elif val == "SPRZEDAJ":
        return "color: red"
    return ""

st.dataframe(df.style.applymap(highlight, subset=["Sygnał"]))

st.markdown("---")
st.caption("SEP Forex Signals © 2025 – wersja demo")
