import streamlit as st

def login():
    st.title("🔐 SEP Forex Signals – Logowanie")
    username = st.text_input("Login")
    password = st.text_input("Hasło", type="password")

    if st.button("Zaloguj"):
        if username == "sepuser" and password == "sep2025":
            st.session_state["authenticated"] = True
            st.experimental_rerun()
        else:
            st.error("Nieprawidłowe dane logowania")

def require_login():
    if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
        login()
        st.stop()
