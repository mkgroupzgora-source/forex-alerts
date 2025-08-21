# mt5_handler.py
from __future__ import annotations
import os
from typing import Optional, Dict
import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception as e:
    mt5 = None

# Mapowanie interwałów na timeframy MT5
TIMEFRAME_MAP: Dict[str, int] = {
    "1m":  mt5.TIMEFRAME_M1  if mt5 else 1,
    "5m":  mt5.TIMEFRAME_M5  if mt5 else 5,
    "15m": mt5.TIMEFRAME_M15 if mt5 else 15,
    "30m": mt5.TIMEFRAME_M30 if mt5 else 30,
    "60m": mt5.TIMEFRAME_H1  if mt5 else 60,
}

def is_available() -> bool:
    return mt5 is not None

def initialize(login: Optional[int] = None,
               password: Optional[str] = None,
               server: Optional[str] = None,
               path_terminal: Optional[str] = None) -> bool:
    """
    Inicjalizacja połączenia z MT5. 
    Jeżeli login/password/server są None – spróbujemy użyć już zalogowanego terminala.
    """
    if not is_available():
        return False

    # Jeśli terminal nie jest uruchomiony – spróbuj zainicjalizować
    if not mt5.initialize(path=path_terminal or ""):
        # czasem initialize(None) działa, nawet gdy path jest nieznane
        if not mt5.initialize():
            return False

    # Opcjonalne logowanie
    if login and password and server:
        ok = mt5.login(login=login, password=password, server=server)
        if not ok:
            # zakończ i zwróć False
            mt5.shutdown()
            return False

    return True

def shutdown():
    if is_available():
        try:
            mt5.shutdown()
        except Exception:
            pass

def symbol_select(symbol: str) -> bool:
    """Upewnij się, że instrument jest widoczny w "Market Watch"."""
    if not is_available():
        return False
    info = mt5.symbol_info(symbol)
    if info is None:
        return False
    if not info.visible:
        return mt5.symbol_select(symbol, True)
    return True

def get_ohlc(symbol: str, interval: str = "15m", bars: int = 500) -> pd.DataFrame:
    """
    Zwraca DataFrame OHLCV (Open,High,Low,Close,Volume) z MT5 dla danego symbolu i interwału.
    Index = DatetimeIndex w lokalnej strefie (czas terminala).
    """
    if not is_available():
        return pd.DataFrame()

    if not symbol_select(symbol):
        return pd.DataFrame()

    tf = TIMEFRAME_MAP.get(interval, mt5.TIMEFRAME_M15)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    # konwersja czasu na datetime
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")[["open", "high", "low", "close", "tick_volume"]].rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume"}
    )
    # posprzątaj NaN
    df = df.dropna()
    return df

def last_tick(symbol: str) -> Optional[float]:
    """Odczyt ostatniego ticka (do overlay’u na świecach, jeśli zechcesz)."""
    if not is_available():
        return None
    if not symbol_select(symbol):
        return None
    t = mt5.symbol_info_tick(symbol)
    return None if t is None else float(t.last)
