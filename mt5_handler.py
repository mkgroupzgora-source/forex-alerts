# mt5_handler.py
# -*- coding: utf-8 -*-

"""
Warstwa integracji z MetaTrader 5 z bezpiecznym fallbackiem:
- Gdy MetaTrader5 jest dostępny (lokalnie) -> realne zlecenia.
- Gdy niedostępny (np. Streamlit Cloud)      -> brak twardych błędów, czytelne komunikaty.

Wymagania (lokalnie):
  pip install MetaTrader5

Uwaga: Pakiet MetaTrader5 nie ma kół dla większości środowisk linuksowych/chmurowych.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

# --- próba importu MT5 --------------------------------------------------------
try:
    import MetaTrader5 as MT5  # type: ignore
    _MT5_AVAILABLE = True
except Exception:
    # Pakiet niedostępny (np. Streamlit Cloud)
    MT5 = None  # type: ignore
    _MT5_AVAILABLE = False


# ====== Pomocnicze struktury danych ==========================================

@dataclass
class ConnectResult:
    available: bool
    connected: bool
    message: str

@dataclass
class OrderResult:
    success: bool
    ticket: Optional[int]
    message: str
    request: Optional[Dict[str, Any]] = None
    result_raw: Optional[Dict[str, Any]] = None

@dataclass
class PositionInfo:
    ticket: int
    symbol: str
    volume: float
    type: int       # 0=BUY, 1=SELL
    price_open: float
    sl: float
    tp: float
    comment: str


# ====== Klasa integracji ======================================================

class MT5Handler:
    """
    Prosta otoczka na MetaTrader5 z:
    - connect/login
    - market order
    - modyfikacja SL/TP
    - zamknięcie pozycji
    - trailing SL/TP (manualny)
    - tryb DRY-RUN (symulacja bez wysyłania do brokera)
    """

    def __init__(self, dry_run: bool = False, tz: str = "Europe/Warsaw"):
        self.available = _MT5_AVAILABLE
        self.connected = False
        self.login_id: Optional[int] = None
        self.server: Optional[str] = None
        self.tz = tz
        self._trailing_threads: Dict[int, threading.Thread] = {}
        self._trailing_flags: Dict[int, Dict[str, Any]] = {}
        self.dry_run = dry_run

    # ------------------------------------------------------------------ utils

    def is_available(self) -> bool:
        """Czy pakiet MetaTrader5 jest dostępny w tym środowisku."""
        return self.available

    # ---------------------------------------------------------------- connect

    def connect(self) -> ConnectResult:
        if not self.available:
            return ConnectResult(False, False, "MetaTrader5 package not available in this environment.")

        if self.connected:
            return ConnectResult(True, True, "Already connected.")

        ok = MT5.initialize()  # type: ignore
        if not ok:
            return ConnectResult(True, False, f"MT5.initialize() failed: {MT5.last_error()}")
        self.connected = True
        return ConnectResult(True, True, "MT5 initialized.")

    def login(self, login: int, password: str, server: str) -> ConnectResult:
        if not self.available:
            return ConnectResult(False, False, "MetaTrader5 not available here.")

        if not self.connected:
            c = self.connect()
            if not c.connected:
                return c

        ok = MT5.login(login, password=password, server=server)  # type: ignore
        if not ok:
            return ConnectResult(True, False, f"MT5.login failed: {MT5.last_error()}")
        self.login_id = login
        self.server = server
        return ConnectResult(True, True, "Logged in to MT5.")

    # ------------------------------------------------------------- symbol info

    def _ensure_symbol(self, symbol: str) -> Tuple[bool, str]:
        """Upewnia się, że symbol jest dostępny (subskrypcja)."""
        if self.dry_run or not self.available:
            return True, "DRY-RUN or not available -> skip symbol check."
        info = MT5.symbol_info(symbol)  # type: ignore
        if info is None:
            return False, f"Symbol {symbol} not found."
        if not info.visible:
            if not MT5.symbol_select(symbol, True):  # type: ignore
                return False, f"Failed to select symbol {symbol}."
        return True, "OK"

    # --------------------------------------------------------------- market order

    def place_market_order(
        self,
        symbol: str,
        volume: float,
        side: str,  # "BUY" | "SELL"
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "SEP Order",
        deviation: int = 10,
    ) -> OrderResult:
        """
        Market order. Zwraca OrderResult z ticketem (o ile sukces).
        """
        if not self.available:
            return OrderResult(False, None, "MT5 not available in this environment.")

        ok, msg = self._ensure_symbol(symbol)
        if not ok:
            return OrderResult(False, None, msg)

        if self.dry_run:
            # Symulacja: „udawany” ticket
            fake_ticket = int(time.time())
            req = {
                "symbol": symbol, "volume": volume, "type": side,
                "sl": sl, "tp": tp, "comment": comment, "deviation": deviation
            }
            return OrderResult(True, fake_ticket, "DRY-RUN: placed", req, {"retcode": 0})

        # Realny request
        order_type = MT5.ORDER_TYPE_BUY if side.upper() == "BUY" else MT5.ORDER_TYPE_SELL  # type: ignore
        price = MT5.symbol_info_tick(symbol).ask if side.upper() == "BUY" else MT5.symbol_info_tick(symbol).bid  # type: ignore

        request = {
            "action": MT5.TRADE_ACTION_DEAL,  # type: ignore
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(price),
            "sl": float(sl) if sl else 0.0,
            "tp": float(tp) if tp else 0.0,
            "deviation": int(deviation),
            "magic": 778899,
            "comment": comment,
            "type_time": MT5.ORDER_TIME_GTC,  # type: ignore
            "type_filling": MT5.ORDER_FILLING_FOK,  # type: ignore (zmień pod brokera jeśli potrzeba: IOC)
        }

        result = MT5.order_send(request)  # type: ignore
        if result is None:
            return OrderResult(False, None, f"order_send returned None: {MT5.last_error()}", request, None)  # type: ignore

        if result.retcode != MT5.TRADE_RETCODE_DONE:  # type: ignore
            return OrderResult(
                False,
                getattr(result, "order", None),
                f"MT5 send failed: retcode={result.retcode}",
                request,
                result._asdict() if hasattr(result, "_asdict") else None,
            )

        return OrderResult(
            True,
            getattr(result, "order", None),
            "Order placed.",
            request,
            result._asdict() if hasattr(result, "_asdict") else None,
        )

    # --------------------------------------------------------------- modify SL/TP

    def modify_sl_tp(self, symbol: str, ticket: int, sl: Optional[float], tp: Optional[float]) -> OrderResult:
        if not self.available:
            return OrderResult(False, None, "MT5 not available.")

        if self.dry_run:
            return OrderResult(True, ticket, "DRY-RUN: SL/TP modified.", {"sl": sl, "tp": tp}, None)

        pos = self._find_position(ticket)
        if pos is None:
            return OrderResult(False, None, f"Position {ticket} not found.")

        req = {
            "action": MT5.TRADE_ACTION_SLTP,  # type: ignore
            "position": ticket,
            "symbol": symbol,
            "sl": float(sl) if sl else 0.0,
            "tp": float(tp) if tp else 0.0,
            "magic": 778899,
            "comment": "SEP modify SL/TP",
        }
        result = MT5.order_send(req)  # type: ignore
        if result is None:
            return OrderResult(False, None, f"order_send returned None: {MT5.last_error()}", req, None)  # type: ignore
        if result.retcode != MT5.TRADE_RETCODE_DONE:  # type: ignore
            return OrderResult(False, None, f"Modify SL/TP failed: {result.retcode}", req, result._asdict())
        return OrderResult(True, ticket, "SL/TP modified.", req, result._asdict())

    # ---------------------------------------------------------------- close

    def close_position(self, ticket: int) -> OrderResult:
        if not self.available:
            return OrderResult(False, None, "MT5 not available.")

        if self.dry_run:
            return OrderResult(True, ticket, "DRY-RUN: position closed.", {"ticket": ticket}, None)

        pos = self._find_position(ticket)
        if pos is None:
            return OrderResult(False, None, f"Position {ticket} not found.")

        # Zamykanie: transakcja odwrotna
        symbol = pos.symbol
        vol = pos.volume
        order_type = MT5.ORDER_TYPE_SELL if pos.type == MT5.ORDER_TYPE_BUY else MT5.ORDER_TYPE_BUY  # type: ignore
        price = MT5.symbol_info_tick(symbol).bid if order_type == MT5.ORDER_TYPE_SELL else MT5.symbol_info_tick(symbol).ask  # type: ignore

        req = {
            "action": MT5.TRADE_ACTION_DEAL,  # type: ignore
            "position": ticket,
            "symbol": symbol,
            "volume": float(vol),
            "type": order_type,
            "price": float(price),
            "deviation": 10,
            "magic": 778899,
            "comment": "SEP close",
            "type_time": MT5.ORDER_TIME_GTC,  # type: ignore
            "type_filling": MT5.ORDER_FILLING_FOK,  # type: ignore
        }
        result = MT5.order_send(req)  # type: ignore
        if result is None:
            return OrderResult(False, None, f"order_send returned None: {MT5.last_error()}", req, None)  # type: ignore
        if result.retcode != MT5.TRADE_RETCODE_DONE:  # type: ignore
            return OrderResult(False, None, f"Close failed: {result.retcode}", req, result._asdict())
        return OrderResult(True, ticket, "Position closed.", req, result._asdict())

    # ----------------------------------------------------------- helper: find pos

    def _find_position(self, ticket: int) -> Optional[PositionInfo]:
        if not self.available:
            return None
        positions = MT5.positions_get()  # type: ignore
        if positions is None:
            return None
        for p in positions:
            if int(p.ticket) == int(ticket):
                return PositionInfo(
                    ticket=int(p.ticket),
                    symbol=p.symbol,
                    volume=float(p.volume),
                    type=int(p.type),
                    price_open=float(p.price_open),
                    sl=float(p.sl),
                    tp=float(p.tp),
                    comment=str(p.comment),
                )
        return None

    # ====================================================== TRAILING (manualny)

    def start_trailing(
        self,
        ticket: int,
        symbol: str,
        side: str,               # "BUY" | "SELL"
        step_pips: float = 10.0, # co ile pipsów przesuwać
        arm_after_pips: float = 10.0,  # dopiero po osiągnięciu zysku X pipsów
        pip_size: float = 0.0001,      # EURUSD 0.0001, USDJPY 0.01, złoto 0.1/0.01 itd.
        max_tp: Optional[float] = None,
        update_secs: float = 2.0,
    ) -> str:
        """
        Uruchamia w tle wątek, który:
         - gdy zysk > arm_after_pips, przesuwa SL co step_pips,
         - opcjonalnie utrzymuje TP (jeśli max_tp podane),
         - działa aż do stopu (stop_trailing) lub gdy pozycja zniknie.
        """
        if not self.available and not self.dry_run:
            return "MT5 not available. Cannot start trailing."

        if ticket in self._trailing_threads:
            return f"Trailing already running for ticket {ticket}."

        self._trailing_flags[ticket] = {
            "running": True,
            "symbol": symbol,
            "side": side.upper(),
            "step": float(step_pips),
            "arm": float(arm_after_pips),
            "pip": float(pip_size),
            "max_tp": max_tp,
            "update": float(update_secs),
        }

        t = threading.Thread(target=self._trailing_worker, args=(ticket,), daemon=True)
        self._trailing_threads[ticket] = t
        t.start()
        return "Trailing started."

    def stop_trailing(self, ticket: int) -> str:
        flag = self._trailing_flags.get(ticket)
        if not flag:
            return "No trailing found for this ticket."
        flag["running"] = False
        return "Trailing stop requested to stop."

    def _trailing_worker(self, ticket: int) -> None:
        flag = self._trailing_flags.get(ticket)
        if not flag:
            return

        symbol = flag["symbol"]
        side = flag["side"]
        step = flag["step"]
        arm = flag["arm"]
        pip = flag["pip"]
        max_tp = flag["max_tp"]
        update = flag["update"]

        last_armed = False
        last_reference_sl: Optional[float] = None

        while flag["running"]:
            # Pobierz pozycję / ceny
            pos = self._find_position(ticket)
            if pos is None:
                # w DRY-RUN możemy tylko „udawać”
                if self.dry_run:
                    time.sleep(update)
                    continue
                # realnie – kończ
                break

            tick = None
            if self.available and not self.dry_run:
                tick = MT5.symbol_info_tick(symbol)  # type: ignore

            # Bieżąca cena i zysk w pipsach
            if side == "BUY":
                curr_price = (tick.bid if tick else pos.price_open + (step * pip))  # prosta symulacja dla DRY-RUN
                profit_pips = (curr_price - pos.price_open) / pip
            else:
                curr_price = (tick.ask if tick else pos.price_open - (step * pip))
                profit_pips = (pos.price_open - curr_price) / pip

            # uzbrojenie trailing dopiero po arm_after_pips
            if profit_pips >= arm:
                last_armed = True

            if last_armed:
                # docelowy SL zgodnie z krokiem
                if side == "BUY":
                    target_sl = curr_price - (step * pip)
                    if last_reference_sl is None or target_sl > last_reference_sl:
                        # przesuń SL w górę
                        self.modify_sl_tp(symbol, ticket, sl=target_sl, tp=max_tp)
                        last_reference_sl = target_sl
                else:
                    target_sl = curr_price + (step * pip)
                    if last_reference_sl is None or target_sl < last_reference_sl:
                        self.modify_sl_tp(symbol, ticket, sl=target_sl, tp=max_tp)
                        last_reference_sl = target_sl

            time.sleep(update)

        # sprzątanie
        self._trailing_threads.pop(ticket, None)
        self._trailing_flags.pop(ticket, None)

# ====== Fabryka (wygodny konstruktor) =========================================

def get_handler(dry_run: bool = False) -> MT5Handler:
    """
    Zwraca gotowy obiekt obsługujący zarówno tryb lokalny (MT5) jak i chmurowy (fallback).
    Użycie:
        mt5 = get_handler(dry_run=False)
        if not mt5.is_available():
            st.warning("MT5 niedostępne w tym środowisku – przyciski tradingowe ukryte")
    """
    return MT5Handler(dry_run=dry_run)
