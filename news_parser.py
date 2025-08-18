# news_parsers.py
# -*- coding: utf-8 -*-
"""
Agregator newsów z Investing.com, ForexFactory i MarketWatch.
- Bez zależności od lxml (korzysta z html.parser).
- Oparty o feedparser (RSS/Atom) + opcjonalny parsing HTML opisów.
- Automatyczne przypinanie newsów do instrumentów (waluty, metale, surowce).

Zwracane rekordy: dict(
    title: str,
    link: str,
    source: str,          # 'Investing', 'ForexFactory', 'MarketWatch'...
    published: datetime,  # w UTC (aware)
    summary: str,
    symbols: List[str],   # np. ['EURUSD', 'XAUUSD']
)
"""

from __future__ import annotations

import re
import time
import logging
from datetime import datetime, timezone
from typing import Iterable, List, Dict, Any, Optional

import requests
import feedparser
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

# -----------------------
# Ustawienia / nagłówki
# -----------------------

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)
REQ_TIMEOUT = 12  # sekundy

DEFAULT_FEEDS: Dict[str, List[str]] = {
    # INVESTING – forex + commodities (RSS Investing potrafi się zmieniać,
    # poniższe działają stabilnie, ale możesz dodać własne).
    "Investing": [
        "https://www.investing.com/rss/news_25.rss",        # Forex News
        "https://www.investing.com/rss/commodities.rss",    # Commodities
    ],
    # FOREXFACTORY – kalendarz na bieżący tydzień (XML)
    "ForexFactory": [
        "https://www.forexfactory.com/ffcal_week_this.xml"
    ],
    # MARKETWATCH – szybkie krótkie newsy (MarketPulse)
    "MarketWatch": [
        "https://feeds.marketwatch.com/marketwatch/marketpulse/"
    ],
}

# ------------------------------------------------------
# Mapowanie słów kluczowych -> instrumenty (symbole)
# Działa językowo niezależnie: kody walut, nazwy metali,
# potoczne nazwy (np. 'złoto', 'gold', 'brent', 'wt i').
# ------------------------------------------------------

# Dozwolone symbole końcowe – jeśli przekażesz własną listę do
# fetch_all_news(..., supported_symbols=[...]), zostaną przefiltrowane.
ALL_CANONICAL_SYMBOLS = {
    # Majory / minors
    "EURUSD", "GBPUSD", "USDCHF", "USDJPY", "USDCNH", "USDRUB",
    "AUDUSD", "NZDUSD", "USDCAD", "USDSEK", "USDPLN",
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDPLN",
    "CADCHF", "CADJPY", "CADPLN",
    "CHFJPY", "CHFPLN", "CNHJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURCNH", "EURGBP",
    "EURJPY", "EURNZD", "EURPLN",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPPLN",
    # Metale / surowce
    "XAGUSD", "XAUUSD", "XPDUSD", "XPTUSD",
    # (opcjonalnie) ropa:
    "WTIUSD", "BRENTUSD",
    # (opcjonalnie) krypto (jeśli w projekcie jest obsługa)
    "BTCUSD", "ETHUSD",
}

# Słownik słów kluczowych powiązanych z instrumentami
KEYWORD_MAP: Dict[str, List[str]] = {
    # Waluty – kody ISO i nazwy w PL/EN:
    "USD": ["usd", "dollar", "us dollar", "amerykański", "usa"],
    "EUR": ["eur", "euro", "european"],
    "GBP": ["gbp", "pound", "british", "funt"],
    "CHF": ["chf", "franc", "swiss"],
    "JPY": ["jpy", "yen", "japanese", "jen"],
    "PLN": ["pln", "złoty", "zloty", "poland"],
    "CAD": ["cad", "canadian", "loonie"],
    "AUD": ["aud", "aussie", "australian"],
    "NZD": ["nzd", "kiwi", "new zealand"],
    "CNH": ["cnh", "offshore yuan", "yuan", "renminbi", "cny"],
    "RUB": ["rub", "ruble", "russian"],

    # Metale / surowce:
    "XAU": ["xau", "gold", "złoto", "zloto", "bullion"],
    "XAG": ["xag", "silver", "srebro"],
    "XPT": ["xpt", "platinum", "platyna"],
    "XPD": ["xpd", "palladium", "pallad"],
    "WTI": ["wti", "crude", "us crude", "west texas intermediate"],
    "BRENT": ["brent", "brent crude"],

    # Krypto – jeśli potrzebne:
    "BTC": ["btc", "bitcoin"],
    "ETH": ["eth", "ethereum"],
}

# gotowe pary, jakie potrafimy złożyć z dwóch kluczy walutowych
PAIR_TEMPLATES = [
    # Majory i krzyżówki – najczęściej używane
    ("EUR", "USD"), ("GBP", "USD"), ("USD", "CHF"), ("USD", "JPY"),
    ("AUD", "USD"), ("NZD", "USD"), ("USD", "CAD"), ("USD", "SEK"),
    ("USD", "PLN"), ("EUR", "GBP"), ("EUR", "JPY"), ("EUR", "CHF"),
    ("EUR", "CAD"), ("EUR", "AUD"), ("EURNZD", "USD"),  # placeholder, niżej robimy generycznie
    ("GBP", "JPY"), ("GBP", "CHF"), ("GBP", "CAD"), ("GBP", "PLN"),
    ("CHF", "JPY"), ("CAD", "JPY"), ("AUD", "JPY"),
    ("AUD", "CAD"), ("AUD", "CHF"), ("AUD", "NZD"), ("AUD", "PLN"),
    ("CAD", "CHF"), ("CAD", "PLN"), ("CHF", "PLN"),
    ("EUR", "PLN"), ("EUR", "NZD"), ("EUR", "CNH"),
    ("CNH", "JPY"),
]

# ------------------------------------------------------
# Narzędzia
# ------------------------------------------------------

SAFE_ASCII = re.compile(r"\s+")

def _norm_text(txt: str) -> str:
    return SAFE_ASCII.sub(" ", (txt or "")).strip()

def _to_utc(dt_struct) -> Optional[datetime]:
    """Spróbuj zamienić parsed time z feedparsera na datetime UTC."""
    try:
        if not dt_struct:
            return None
        # feedparser podaje time.struct_time
        ts = time.mktime(dt_struct)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        return None

def _detect_tokens(text: str) -> Dict[str, bool]:
    """Zwraca obecność tokenów/skrótów walutowych/metali w tekście."""
    text_l = (text or "").lower()
    present: Dict[str, bool] = {}
    for code, keys in KEYWORD_MAP.items():
        flag = any(k in text_l for k in keys) or (code.lower() in text_l)
        present[code] = flag
    return present

def _make_pairs_from_tokens(tokens: Dict[str, bool]) -> List[str]:
    """Z komend TRUE buduje listę możliwych par walutowych w standardzie XXXYYY."""
    pairs: List[str] = []

    # waluty
    for base, quote in PAIR_TEMPLATES:
        # specjalny przypadek 'EURNZD' z powyższego komentarza – ignorujemy
        if base == "EURNZD":
            continue
        if tokens.get(base) and tokens.get(quote):
            pairs.append(f"{base}{quote}")

    # metale/surowce – łączymy z USD
    if tokens.get("XAU"):
        pairs.append("XAUUSD")
    if tokens.get("XAG"):
        pairs.append("XAGUSD")
    if tokens.get("XPT"):
        pairs.append("XPTUSD")
    if tokens.get("XPD"):
        pairs.append("XPDUSD")

    # ropa
    if tokens.get("WTI"):
        pairs.append("WTIUSD")
    if tokens.get("BRENT"):
        pairs.append("BRENTUSD")

    # krypto – jeśli w projekcie przewidziano
    if tokens.get("BTC"):
        pairs.append("BTCUSD")
    if tokens.get("ETH"):
        pairs.append("ETHUSD")

    # deduplikacja
    pairs = list(dict.fromkeys(pairs))
    return pairs

def assign_symbols_by_text(text: str,
                           supported_symbols: Optional[Iterable[str]] = None) -> List[str]:
    """Zwraca listę symboli (np. ['EURUSD','XAUUSD']) wykrytych w tekście."""
    tokens = _detect_tokens(text)
    pairs = _make_pairs_from_tokens(tokens)

    # Filtruj do dozwolonych jeśli podano
    if supported_symbols:
        allowed = set(supported_symbols)
    else:
        allowed = ALL_CANONICAL_SYMBOLS
    return [p for p in pairs if p in allowed]

# ------------------------------------------------------
# Pobieranie / parsowanie feedów
# ------------------------------------------------------

def _fetch_url(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQ_TIMEOUT)
        if r.status_code == 200:
            return r.content
        log.warning("HTTP %s for %s", r.status_code, url)
        return None
    except requests.RequestException as e:
        log.warning("Request error for %s: %s", url, e)
        return None

def _parse_feed_generic(source_name: str,
                        url: str,
                        supported_symbols: Optional[Iterable[str]],
                        limit: int = 30) -> List[Dict[str, Any]]:
    """Parser generczny RSS/Atom via feedparser (Investing, MarketWatch...)."""
    out: List[Dict[str, Any]] = []
    raw = _fetch_url(url)
    if not raw:
        return out

    fp = feedparser.parse(raw)
    for entry in fp.entries[:limit]:
        title = _norm_text(getattr(entry, "title", ""))
        link = getattr(entry, "link", "")
        summary = _norm_text(getattr(entry, "summary", "") or getattr(entry, "description", ""))

        # niektóre feedy pakują HTML do summary – oczyść lekko
        if "<" in summary and ">" in summary:
            try:
                soup = BeautifulSoup(summary, "html.parser")
                summary = _norm_text(soup.get_text(" "))
            except Exception:
                summary = _norm_text(summary)

        published = _to_utc(getattr(entry, "published_parsed", None)) \
                    or _to_utc(getattr(entry, "updated_parsed", None)) \
                    or datetime.now(tz=timezone.utc)

        # przypisz symbole po tytule+streszczeniu
        symbols = assign_symbols_by_text(f"{title} {summary}", supported_symbols)

        out.append({
            "title": title,
            "link": link,
            "source": source_name,
            "published": published,
            "summary": summary,
            "symbols": symbols,
        })
    return out

def _parse_forexfactory_calendar(url: str,
                                 supported_symbols: Optional[Iterable[str]],
                                 limit: int = 50) -> List[Dict[str, Any]]:
    """
    ForexFactory: tygodniowy kalendarz w XML (ffcal_week_this.xml).
    Format bywa różny – feedparser poradzi sobie, a my wyciągamy najważniejsze pola.
    """
    out: List[Dict[str, Any]] = []
    raw = _fetch_url(url)
    if not raw:
        return out

    fp = feedparser.parse(raw)

    # ForexFactory zwykle w tytule ma [CCY] Nazwa, a w opisie szczegóły.
    for entry in fp.entries[:limit]:
        title = _norm_text(getattr(entry, "title", ""))
        link = getattr(entry, "link", "") or "https://www.forexfactory.com/calendar"
        summary = _norm_text(getattr(entry, "summary", "") or getattr(entry, "description", ""))

        if "<" in summary and ">" in summary:
            try:
                soup = BeautifulSoup(summary, "html.parser")
                summary = _norm_text(soup.get_text(" "))
            except Exception:
                summary = _norm_text(summary)

        published = _to_utc(getattr(entry, "published_parsed", None)) \
                    or _to_utc(getattr(entry, "updated_parsed", None)) \
                    or datetime.now(tz=timezone.utc)

        # W wielu tytułach jest np. "[USD] Nonfarm Payrolls" – wychwyć skrót w []:
        m = re.search(r"\[([A-Z]{3})\]", title)
        symbols: List[str] = []
        if m:
            ccy = m.group(1)
            # jeśli jest ccy, zbuduj kilka podstawowych krzyżówek z USD/EUR/JPY itp.
            # (zależnie od supported_symbols)
            candidates = [
                f"{ccy}USD", f"EUR{ccy}", f"USD{ccy}", f"{ccy}JPY", f"GBP{ccy}",
                f"{ccy}PLN", f"{ccy}CHF", f"{ccy}CAD", f"{ccy}AUD", f"{ccy}NZD",
            ]
            if supported_symbols:
                allowed = set(supported_symbols)
            else:
                allowed = ALL_CANONICAL_SYMBOLS
            symbols = [c for c in candidates if c in allowed]

        # Dorzuć heurystykę z tekstu (gdy brak [CCY] w tytule)
        if not symbols:
            symbols = assign_symbols_by_text(f"{title} {summary}", supported_symbols)

        out.append({
            "title": title,
            "link": link,
            "source": "ForexFactory",
            "published": published,
            "summary": summary,
            "symbols": symbols,
        })
    return out

# ------------------------------------------------------
# API publiczne
# ------------------------------------------------------

def fetch_all_news(supported_symbols: Optional[Iterable[str]] = None,
                   per_feed_limit: int = 30) -> List[Dict[str, Any]]:
    """
    Zbierz newsy z domyślnych źródeł i przypisz symbole.
    Zwraca posortowane malejąco po `published`.
    """
    items: List[Dict[str, Any]] = []

    feeds = DEFAULT_FEEDS

    # Investing + MarketWatch – generyczny RSS
    for source in ("Investing", "MarketWatch"):
        for url in feeds.get(source, []):
            try:
                items.extend(
                    _parse_feed_generic(source, url, supported_symbols, per_feed_limit)
                )
            except Exception as e:
                log.warning("Feed error (%s): %s", source, e)

    # ForexFactory – kalendarz
    for url in feeds.get("ForexFactory", []):
        try:
            items.extend(
                _parse_forexfactory_calendar(url, supported_symbols, per_feed_limit)
            )
        except Exception as e:
            log.warning("Feed error (ForexFactory): %s", e)

    # Sortuj po dacie malejąco i utnij do rozsądnej liczby (np. 200)
    items.sort(key=lambda x: x.get("published") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return items[:200]


# ------------------------------------------------------
# Quick manual test
# ------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    news = fetch_all_news(
        supported_symbols=[
            # Przykładowy filtr, jeśli chcesz tylko część:
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "XAUUSD", "XAGUSD", "USDPLN",
        ],
        per_feed_limit=10,
    )
    for n in news[:10]:
        print(
            f"[{n['source']}] {n['published']:%Y-%m-%d %H:%M}  {', '.join(n['symbols']) or '-'}\n"
            f"{n['title']}\n{n['link']}\n"
        )
            items.append({
                "source": source,
                "title": e.get("title", ""),
                "summary": e.get("summary", ""),
                "published": e.get("published", ""),
                "link": e.get("link", ""),
            })
    return items


def news_for_symbol(symbol: str, all_news: List[Dict]) -> List[Dict]:
    keys = _keywords_for_symbol(symbol)
    out: List[Dict] = []
    for n in all_news:
        text = f"{n.get('title','')} {n.get('summary','')}".lower()
        if any(re.search(rf"\b{re.escape(k)}\b", text) for k in keys):
            out.append(n)
    return out[:10]
