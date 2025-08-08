# news_parser.py
"""
RSS parser: Investing, ForexFactory, MarketWatch.
Funkcje:
- fetch_news() -> list[dict]
- news_for_symbol(symbol, all_news) -> list[dict] (filtrowanie po słowach kluczowych)
"""

from typing import List, Dict
import re
import feedparser

NEWS_FEEDS = {
    "Investing": "https://www.investing.com/rss/news_301.rss",
    "ForexFactory": "https://www.forexfactory.com/news.atom",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
}

CCY_NAMES = {
    "USD": ["usd", "dollar", "us dollar", "u.s. dollar"],
    "EUR": ["eur", "euro"],
    "GBP": ["gbp", "pound", "british pound", "sterling"],
    "JPY": ["jpy", "yen", "japanese yen"],
    "CHF": ["chf", "swiss franc"],
    "CAD": ["cad", "canadian dollar"],
    "AUD": ["aud", "australian dollar"],
    "NZD": ["nzd", "new zealand dollar"],
    "PLN": ["pln", "polish zloty", "zloty"],
    "CNH": ["cnh", "offshore yuan", "yuan", "renminbi"],
    "RUB": ["rub", "ruble", "rouble"],
    "SEK": ["sek", "swedish krona"],
}

COMMODITY_NAMES = {
    "XAUUSD": ["gold", "xau", "bullion"],
    "XAGUSD": ["silver", "xag"],
    "XPTUSD": ["platinum", "xpt"],
    "XPDUSD": ["palladium", "xpd"],
}


def _keywords_for_symbol(symbol: str) -> List[str]:
    symbol = symbol.replace("=X", "").upper()
    keys: List[str] = []
    # metals
    if symbol in COMMODITY_NAMES:
        return COMMODITY_NAMES[symbol]

    # FX pairs
    if len(symbol) >= 6:
        base = symbol[:3]
        quote = symbol[3:6]
        keys += CCY_NAMES.get(base, [base.lower()])
        keys += CCY_NAMES.get(quote, [quote.lower()])
    else:
        keys.append(symbol.lower())
    return list(set(keys))


def fetch_news(limit_per_feed: int = 15) -> List[Dict]:
    items: List[Dict] = []
    for source, url in NEWS_FEEDS.items():
        feed = feedparser.parse(url)
        for e in feed.entries[:limit_per_feed]:
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
