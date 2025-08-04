import streamlit as st
from login import require_login

require_login()

st.title("📊 SEP Forex Signals – Dashboard")
st.write("Wersja testowa z RSI i analizą fundamentalną.")

st.success("Zalogowano jako sepuser")

# Tutaj możesz dodać wykresy, dane RSI, newsy itd.
