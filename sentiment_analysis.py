# sentiment_analysis.py
"""
NLP: VADER. Zwraca etykietę oraz wynik.
- sentiment_label(text) -> ("Positive"/"Negative"/"Neutral", score)
- annotate_news(news_list) -> dodaje pola sentiment, score.
"""

from typing import Tuple, List, Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

def sentiment_label(text: str) -> Tuple[str, float]:
    score = _analyzer.polarity_scores(text or "")["compound"]
    if score >= 0.05:
        return "Positive", score
    if score <= -0.05:
        return "Negative", score
    return "Neutral", score

def annotate_news(news_list: List[Dict]) -> List[Dict]:
    out = []
    for n in news_list:
        label, score = sentiment_label(f"{n.get('title','')} {n.get('summary','')}")
        m = dict(n)
        m["sentiment"] = label
        m["score"] = score
        out.append(m)
    return out
