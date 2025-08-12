# mt5_handler.py
# Hybrydowa obsługa MT5: realna (gdy biblioteka MetaTrader5 jest dostępna)
# oraz tryb MOCK (gdy pracujemy w chmurze / bez MT5).
#
# Użycie:
#   from mt5_handler import MT5
#   mt5 = MT5()
#   mt5.connect(login=..., password=..., server=...)
#   mt5.place_order(symbol="EURUSD", action="BUY", volume=0.1)
#   mt5.set_trailing(symbol="EURUSD", profit_trigger_pips=10, trail_step_pips=10)
#   mt5.close_position(ticket=123456)
#   mt5.shutdown()

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


# ------------------------------------------------------------
# Czy MetaTrader5 jest dostępny?
# ------------------------------------------------------------
MT5_AVAILABLE = False
try:
    # Pozwalamy wymusić MOCK przez zmienną środowiskową
    if os.environ.get("USE_MT5", "1") == "1":
        import MetaTrader5 as _mt5  # type: ignore
        MT5_AVAILABLE = True
except Exception:
    MT5_AVAILABLE = False


@dataclass
class OrderResult:
    ok: bool
    ticket: Optional[int] = None
    comment: str = ""


@dataclass
class Position:
    ticket: int
    symbol: str
    type: str  # "BUY" albo "SELL"
    volume: float
    price_open: float
    sl: Optional[float]
    tp: Optional[float]
    profit: float


# ------------------------------------------------------------
# Implementacja MOCK (dla chmury / bez MT5)
# ------------------------------------------------------------
class _MockMT5:
    """
    Bardzo prosty mock. Trzyma pozycje w pamięci procesu.
    Nie łączy się z żadnym brokerem. Przydaje się na Streamlit Cloud,
    gdzie nie możemy zainstalować MetaTrader5.
    """

    def __init__(self) -> None:
        self.connected = False
        self._positions: Dict[int, Position] = {}
        self._next_ticket = 1
        self._balance = 100000.0

    # -- API zbliżone do prawdziwego handlera --

    def connect(self, login: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None) -> bool:
        self.connected = True
        print("[MOCK MT5] Connected (symulacja).")
        return True

    def is_connected(self) -> bool:
        return self.connected

    def shutdown(self) -> None:
        self.connected = False
        print("[MOCK MT5] Shutdown (symulacja).")

    def get_balance(self) -> float:
        return self._balance

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        positions = list(self._positions.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions

    def place_order(
        self,
        symbol: str,
        action: str,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        deviation: int = 20,
        comment: str = "streamlit-order",
    ) -> OrderResult:
        if not self.connected:
            return OrderResult(False, None, "Not connected (MOCK)")
        ticket = self._next_ticket
        self._next_ticket += 1
        price = 1.0  # symulowana cena
        pos = Position(
            ticket=ticket,
            symbol=symbol,
            type=action.upper(),
            volume=volume,
            price_open=price,
            sl=sl,
            tp=tp,
            profit=0.0,
        )
        self._positions[ticket] = pos
        print(f"[MOCK MT5] Placed {action} {symbol} {volume} lots, ticket={ticket}")
        return OrderResult(True, ticket, "OK (MOCK)")

    def close_position(self, ticket: int) -> bool:
        if ticket in self._positions:
            del self._positions[ticket]
            print(f"[MOCK MT5] Closed position ticket={ticket}")
            return True
        print(f"[MOCK MT5] No position with ticket={ticket}")
        return False

    def set_trailing(self, symbol: str, profit_trigger_pips: int, trail_step_pips: int) -> None:
        # W mocku tylko wypisujemy w logu.
        print(
            f"[MOCK MT5] Trailing set for {symbol}: trigger={profit_trigger_pips} pips, step={trail_step_pips} pips"
        )


# ------------------------------------------------------------
# Prawdziwa implementacja na MetaTrader5 (do użycia lokalnie)
# ------------------------------------------------------------
class _RealMT5:
    def __init__(self) -> None:
        self.connected = False

    def connect(self, login: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None) -> bool:
        # start terminala (jeśli potrzeba – często wystarczy sama inicjalizacja)
        if not _mt5.initialize():
            print(f"[MT5] initialize() failed, error: {_mt5.last_error()}")
            return False

        if all(v is not None for v in (login, password, server)):
            authorized = _mt5.login(login, password=password, server=server)
            if not authorized:
                print(f"[MT5] login failed, error: {_mt5.last_error()}")
                return False

        self.connected = True
        print("[MT5] Connected.")
        return True

    def is_connected(self) -> bool:
        return self.connected

    def shutdown(self) -> None:
        try:
            _mt5.shutdown()
        finally:
            self.connected = False
        print("[MT5] Shutdown.")

    def get_balance(self) -> float:
        account_info = _mt5.account_info()
        if account_info is None:
            return 0.0
        return float(account_info.balance)

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        if symbol:
            mt5_positions = _mt5.positions_get(symbol=symbol)
        else:
            mt5_positions = _mt5.positions_get()
        result: List[Position] = []
        if mt5_positions is None:
            return result

        for p in mt5_positions:
            result.append(
                Position(
                    ticket=p.ticket,
                    symbol=p.symbol,
                    type="BUY" if p.type == 0 else "SELL",
                    volume=float(p.volume),
                    price_open=float(p.price_open),
                    sl=float(p.sl) if p.sl != 0 else None,
                    tp=float(p.tp) if p.tp != 0 else None,
                    profit=float(p.profit),
                )
            )
        return result

    def _symbol_check(self, symbol: str) -> bool:
        if not _mt5.symbol_select(symbol, True):
            print(f"[MT5] symbol_select({symbol}) failed, error: {_mt5.last_error()}")
            return False
        return True

    def place_order(
        self,
        symbol: str,
        action: str,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        deviation: int = 20,
        comment: str = "streamlit-order",
    ) -> OrderResult:
        if not self.connected:
            return OrderResult(False, None, "Not connected")

        if not self._symbol_check(symbol):
            return OrderResult(False, None, f"Symbol {symbol} not available")

        action = action.upper()
        if action not in ("BUY", "SELL"):
            return OrderResult(False, None, "action must be BUY or SELL")

        # bieżący tick
        tick = _mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(False, None, "No tick")

        if action == "BUY":
            price = tick.ask
            order_type = _mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            order_type = _mt5.ORDER_TYPE_SELL

        request = {
            "action": _mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(price),
            "deviation": int(deviation),
            "magic": 123456,
            "comment": comment,
            "type_time": _mt5.ORDER_TIME_GTC,
            "type_filling": _mt5.ORDER_FILLING_FOK,
        }
        if sl:
            request["sl"] = float(sl)
        if tp:
            request["tp"] = float(tp)

        result = _mt5.order_send(request)
        if result is None:
            return OrderResult(False, None, f"order_send failed: {_mt5.last_error()}")

        if result.retcode != _mt5.TRADE_RETCODE_DONE:
            return OrderResult(False, None, f"retcode={result.retcode} {result.comment}")

        ticket = int(result.order) if result.order != 0 else int(getattr(result, "deal", 0))
        print(f"[MT5] Placed {action} {symbol} {volume} lots, ticket={ticket}")
        return OrderResult(True, ticket, "OK")

    def close_position(self, ticket: int) -> bool:
        pos_list = _mt5.positions_get(ticket=ticket)
        if pos_list is None or len(pos_list) == 0:
            print(f"[MT5] close_position: no position for ticket={ticket}")
            return False

        pos = pos_list[0]
        symbol = pos.symbol
        volume = pos.volume

        # cena zamknięcia zależnie od kierunku
        tick = _mt5.symbol_info_tick(symbol)
        if tick is None:
            print("[MT5] No tick on close.")
            return False

        if pos.type == _mt5.POSITION_TYPE_BUY:
            price = tick.bid  # zamykamy po przeciwnej stronie
            order_type = _mt5.ORDER_TYPE_SELL
        else:
            price = tick.ask
            order_type = _mt5.ORDER_TYPE_BUY

        request = {
            "action": _mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "position": int(ticket),
            "price": float(price),
            "deviation": 30,
            "magic": 123456,
            "comment": "close-by-handler",
            "type_time": _mt5.ORDER_TIME_GTC,
            "type_filling": _mt5.ORDER_FILLING_FOK,
        }
        result = _mt5.order_send(request)
        ok = result and result.retcode == _mt5.TRADE_RETCODE_DONE
        print(f"[MT5] close_position ticket={ticket} -> {'OK' if ok else result.retcode}")
        return bool(ok)

    def set_trailing(self, symbol: str, profit_trigger_pips: int, trail_step_pips: int) -> None:
        """
        Prosty trailing: jeśli pozycja ma zysk >= trigger, to podnosi SL o 'step' pipsów
        od aktualnej ceny (dla BUY) lub ponad cenę (dla SELL). W praktyce trailing należy
        wywoływać cyklicznie (np. co 10–30s) w osobnym wątku/cron.
        """
        positions = self.get_positions(symbol=symbol)
        if not positions:
            return

        info = _mt5.symbol_info(symbol)
        if info is None or info.point == 0:
            print(f"[MT5] No symbol info for {symbol}")
            return
        point = info.point
        pips = lambda x: x * 10 * point if info.digits >= 3 else x * point

        tick = _mt5.symbol_info_tick(symbol)
        if tick is None:
            return

        for p in positions:
            if p.type == "BUY":
                current_price = tick.bid
                gained = (current_price - p.price_open) / pips(1)
                if gained >= profit_trigger_pips:
                    new_sl = current_price - pips(trail_step_pips)
                    if p.sl is None or new_sl > p.sl:
                        self._modify_sl(p.ticket, new_sl)
            else:  # SELL
                current_price = tick.ask
                gained = (p.price_open - current_price) / pips(1)
                if gained >= profit_trigger_pips:
                    new_sl = current_price + pips(trail_step_pips)
                    if p.sl is None or new_sl < p.sl:
                        self._modify_sl(p.ticket, new_sl)

    def _modify_sl(self, ticket: int, new_sl: float) -> None:
        pos_list = _mt5.positions_get(ticket=ticket)
        if not pos_list:
            return
        pos = pos_list[0]
        request = {
            "action": _mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": pos.symbol,
            "sl": float(new_sl),
            "tp": pos.tp if pos.tp else 0.0,
            "magic": 123456,
            "comment": "trail-sl",
        }
        result = _mt5.order_send(request)
        if result and result.retcode == _mt5.TRADE_RETCODE_DONE:
            print(f"[MT5] SL modified ticket={ticket} -> {new_sl}")
        else:
            print(f"[MT5] SL modify failed ticket={ticket}: {getattr(result, 'retcode', 'none')}")


# ------------------------------------------------------------
# Publiczna klasa, która wybiera backend (Real vs Mock)
# ------------------------------------------------------------
class MT5:
    def __init__(self) -> None:
        if MT5_AVAILABLE:
            self._backend: Any = _RealMT5()
            self.mode = "REAL"
        else:
            self._backend = _MockMT5()
            self.mode = "MOCK"

    # Proxy do metod backendu
    def connect(self, *args, **kwargs) -> bool:
        return self._backend.connect(*args, **kwargs)

    def is_connected(self) -> bool:
        return self._backend.is_connected()

    def shutdown(self) -> None:
        return self._backend.shutdown()

    def get_balance(self) -> float:
        return self._backend.get_balance()

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        return self._backend.get_positions(symbol=symbol)

    def place_order(
        self,
        symbol: str,
        action: str,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        deviation: int = 20,
        comment: str = "streamlit-order",
    ) -> OrderResult:
        return self._backend.place_order(symbol, action, volume, sl, tp, deviation, comment)

    def close_position(self, ticket: int) -> bool:
        return self._backend.close_position(ticket)

    def set_trailing(self, symbol: str, profit_trigger_pips: int, trail_step_pips: int) -> None:
        return self._backend.set_trailing(symbol, profit_trigger_pips, trail_step_pips)
