# news_parsers.py
# Minimalny parser newsów: Investing (kanały RSS społeczności), ForexFactory (kalendarz/wiadomości), MarketWatch
# Brak scrapingu stron — tylko oficjalne kanały RSS/Atom. Zwraca listę słowników.

from __future__ import annotations
import time
from typing import List, Dict
import feedparser

# --- Lista kanałów RSS/Atom ---
# Jeśli któryś kanał przestanie działać, po prostu podmień URL na aktualny.
FEEDS: List[Dict[str, str]] = [
    # MarketWatch – top stories
    {"source": "MarketWatch", "url": "https://www.marketwatch.com/feeds/topstories"},
    # MarketWatch – Economy/Politics (bardziej makro)
    {"source": "MarketWatch (Economy)", "url": "https://www.marketwatch.com/feeds/Economy"},
    # ForexFactory – ostatnie wiadomości (forum/news), prosty RSS
    {"source": "ForexFactory", "url": "https://www.forexfactory.com/ff.xml"},
    # ForexFactory – kalendarz tygodniowy XML (wydarzenia makro)
    {"source": "ForexFactory Calendar", "url": "https://www.forexfactory.com/ffcal_week_this.xml"},
    # Investing.com – publiczny kanał RSS dla forex (community / articles)
    # Uwaga: Investing często zmienia kanały; jeśli ten nie zadziała, podmień na inny z /rss/
    {"source": "Investing (Forex)", "url": "https://www.investing.com/rss/news_25.rss"},
    # Alternatywne/kierunkowe (możesz włączyć, jeśli chcesz)
    # {"source": "Reuters (Business)", "url": "https://feeds.reuters.com/reuters/businessNews"},
    # {"source": "Reuters (Markets)", "url": "https://feeds.reuters.com/reuters/marketsNews"},
]

# Proste filtry słów kluczowych – tylko jeśli chcesz zawężać (zostaw pustą listę = wyłącz filtrowanie)
KEYWORDS: List[str] = []  # np. ["EUR", "USD", "inflation", "CPI", "Fed", "NFP", "RBNZ", "BoE", "BoJ"]

def _passes_keywords(title: str) -> bool:
    if not KEYWORDS:
        return True
    t = (title or "").lower()
    return any(k.lower() in t for k in KEYWORDS)

def _parse_feed(url: str) -> List[Dict]:
    out: List[Dict] = []
    d = feedparser.parse(url)
    for e in d.get("entries", []):
        title = e.get("title") or ""
        link = e.get("link") or e.get("id") or ""
        # published_parsed to struct_time – konwertujemy do ISO
        if "published_parsed" in e and e["published_parsed"]:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", e["published_parsed"])
        elif "updated_parsed" in e and e["updated_parsed"]:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", e["updated_parsed"])
        else:
            ts = ""
        out.append({"title": title.strip(), "url": link.strip(), "published": ts})
    return out

def get_top_news(max_items: int = 20) -> List[Dict]:
    """
    Zwraca posortowaną listę newsów z wielu kanałów:
    [
      {"title": "...", "url": "...", "source": "MarketWatch", "published": "YYYY-mm-dd HH:MM:SS"}
    ]
    """
    items: List[Dict] = []
    seen_urls = set()

    for feed in FEEDS:
        try:
            chunk = _parse_feed(feed["url"])
            for it in chunk:
                if not it.get("title") or not it.get("url"):
                    continue
                if not _passes_keywords(it["title"]):
                    continue
                # deduplikacja po URL
                if it["url"] in seen_urls:
                    continue
                seen_urls.add(it["url"])
                it["source"] = feed["source"]
                items.append(it)
        except Exception:
            # nie blokujemy – w razie problemu z jednym kanałem lecimy dalej
            continue

    # sortowanie: dostępne daty (string) malejąco, potem alfabetycznie po źródle
    def _sort_key(x: Dict):
        return (x.get("published", ""), x.get("source", ""))
    items.sort(key=_sort_key, reverse=True)

    return items[:max_items]
