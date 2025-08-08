# mt5_handler.py
"""
MetaTrader 5: logowanie, zlecenia, trailing SL.
Uwaga: działa tylko lokalnie na maszynie, gdzie zainstalowany i uruchomiony jest MT5.
"""

from typing import Optional
import time
import MetaTrader5 as mt5
from config import TRAILING_STOP_PIPS, TRAILING_TRIGGER_PIPS


def initialize_mt5(login: int, password: str, server: str) -> None:
    if not mt5.initialize(login=login, password=password, server=server):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    # ok


def shutdown_mt5() -> None:
    try:
        mt5.shutdown()
    except Exception:
        pass


def _pip_size(symbol: str) -> float:
    s = symbol.upper()
    if s.endswith(".i"): # czasem broker dodaje sufiksy
        s = s[:-2]
    if "JPY" in s:
        return 0.01
    if s.startswith(("XAU", "XPT", "XPD")):
        return 0.1
    if s.startswith("XAG"):
        return 0.01
    return 0.0001


def _ensure_symbol(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Symbol {symbol} not found/enable failed")
    elif not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Symbol {symbol} not visible enable failed")


def place_order(symbol: str, side: str, lot: float = 0.1,
                sl_pips: Optional[int] = None, tp_pips: Optional[int] = None):
    """
    side: "BUY" / "SELL"
    Zwraca result z MT5 (dict-like).
    """
    _ensure_symbol(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError("No tick for symbol")

    pip = _pip_size(symbol)

    if side.upper() == "BUY":
        price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY
        sl = price - (sl_pips or TRAILING_STOP_PIPS) * pip
        tp = price + (tp_pips or TRAILING_STOP_PIPS) * pip
    else:
        price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL
        sl = price + (sl_pips or TRAILING_STOP_PIPS) * pip
        tp = price - (tp_pips or TRAILING_STOP_PIPS) * pip

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 987654,
        "comment": "SEP order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    return result


def trail_once(symbol: str, side: str,
               trigger_pips: int = TRAILING_TRIGGER_PIPS,
               trail_pips: int = TRAILING_STOP_PIPS) -> bool:
    """
    Jednorazowe podciągnięcie SL dla otwartej pozycji, jeśli zysk >= trigger.
    Zwraca True jeśli SL został przesunięty.
    """
    pip = _pip_size(symbol)

    # znajdź pozycję
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return False
    pos = positions[0]

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False

    if side.upper() == "BUY":
        current = tick.bid
        profit_pips = (current - pos.price_open) / pip
        if profit_pips >= trigger_pips:
            new_sl = current - trail_pips * pip
            if pos.sl is None or new_sl > pos.sl + 0.1 * pip:
                req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": pos.tp,
                }
                r = mt5.order_send(req)
                return r.retcode == mt5.TRADE_RETCODE_DONE
    else:
        current = tick.ask
        profit_pips = (pos.price_open - current) / pip
        if profit_pips >= trigger_pips:
            new_sl = current + trail_pips * pip
            if pos.sl is None or new_sl < pos.sl - 0.1 * pip:
                req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": pos.tp,
                }
                r = mt5.order_send(req)
                return r.retcode == mt5.TRADE_RETCODE_DONE

    return False


def trail_watch(symbol: str, side: str,
                trigger_pips: int = TRAILING_TRIGGER_PIPS,
                trail_pips: int = TRAILING_STOP_PIPS,
                interval_sec: int = 5, timeout_sec: int = 3600):
    """
    Prosta pętla trailing SL (do wywołania po zleceniu).
    Przerywa, gdy pozycja zniknie lub minie timeout.
    """
    start = time.time()
    while time.time() - start < timeout_sec:
        if not mt5.positions_get(symbol=symbol):
            break
        try:
            trail_once(symbol, side, trigger_pips, trail_pips)
        except Exception:
            pass
        time.sleep(interval_sec)
