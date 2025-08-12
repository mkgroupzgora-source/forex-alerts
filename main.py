from typing import Optional
from fastapi import FastAPI, Query, HTTPException

app = FastAPI()

# Przykładowe źródło danych (zastąp własnym)
OFFERS = [
    {"id": 1, "city": "Warszawa", "price": 520000, "rooms": 2},
    {"id": 2, "city": "Gdańsk",   "price": 780000, "rooms": 3},
    {"id": 3, "city": "Kraków",   "price": 410000, "rooms": 1},
]

@app.get("/offers")
def search_offers(
    max_price: Optional[float] = Query(None, ge=0, description="Maksymalna cena"),
    city: Optional[str] = Query(None, min_length=2, description="Miasto"),
    min_rooms: Optional[int] = Query(0, ge=0, description="Minimalna liczba pokoi"),
):
    """
    Zwraca oferty z opcjonalnym filtrowaniem.
    Wszystkie parametry są opcjonalne i bezpiecznie normalizowane.
    """

    # --- Normalizacja / wartości domyślne ---
    # jeśli max_price nie podane -> nieskończoność (brak górnego limitu)
    max_price_val = float("inf") if max_price is None else float(max_price)
    # jeśli min_rooms nie podane -> 0
    min_rooms_val = 0 if min_rooms is None else int(min_rooms)
    # jeśli city podane -> porównujemy case-insensitive
    city_norm = city.lower() if city else None

    # --- Walidacje dodatkowe (opcjonalnie) ---
    if max_price_val < 0 or min_rooms_val < 0:
        raise HTTPException(status_code=400, detail="Parametry nie mogą być ujemne.")

    # --- Filtrowanie bez None w działaniach matematycznych ---
    results = []
    for o in OFFERS:
        if o["price"] > max_price_val:
            continue
        if o["rooms"] < min_rooms_val:
            continue
        if city_norm and o["city"].lower() != city_norm:
            continue
        results.append(o)

    return {"count": len(results), "items": results}
