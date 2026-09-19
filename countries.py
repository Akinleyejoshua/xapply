"""Country filtering for job discovery.

Postings rarely name a country. They say "San Francisco", "Remote, EMEA" or
"Bengaluru". So each country carries the aliases a posting is likely to use:
its own name, common short forms, and the cities that actually appear in tech
job listings. Matching is done on a normalised copy of the location text.

`ANYWHERE` is a pseudo-country meaning "remote with no stated country", which is
how fully-distributed roles are usually written.
"""
from __future__ import annotations

import re

ANYWHERE = "Anywhere / Worldwide"

#: country -> phrases that imply it. Lower case; matched on word boundaries.
COUNTRY_ALIASES: dict[str, list[str]] = {
    ANYWHERE: ["anywhere", "worldwide", "global", "fully remote", "remote - global",
               "any location", "distributed"],
    "United States": ["united states", "usa", "u.s.", "u.s.a", "us-remote", "us remote",
                      "remote us", "america", "san francisco", "new york", "nyc", "seattle",
                      "austin", "boston", "chicago", "denver", "los angeles", "san jose",
                      "palo alto", "mountain view", "atlanta", "miami", "washington, d.c.",
                      "bay area", "california", "texas", "new jersey"],
    "Canada": ["canada", "toronto", "vancouver", "montreal", "ottawa", "calgary", "waterloo"],
    "United Kingdom": ["united kingdom", "uk", "u.k.", "england", "scotland", "wales",
                       "london", "manchester", "edinburgh", "bristol", "cambridge"],
    "Ireland": ["ireland", "dublin", "cork", "galway"],
    "Germany": ["germany", "berlin", "munich", "münchen", "hamburg", "frankfurt", "cologne"],
    "France": ["france", "paris", "lyon", "toulouse", "bordeaux"],
    "Netherlands": ["netherlands", "holland", "amsterdam", "rotterdam", "utrecht", "eindhoven"],
    "Spain": ["spain", "madrid", "barcelona", "valencia", "malaga"],
    "Portugal": ["portugal", "lisbon", "lisboa", "porto"],
    "Italy": ["italy", "milan", "rome", "turin"],
    "Poland": ["poland", "warsaw", "krakow", "kraków", "wroclaw", "gdansk"],
    "Switzerland": ["switzerland", "zurich", "zürich", "geneva", "lausanne"],
    "Sweden": ["sweden", "stockholm", "gothenburg", "malmo"],
    "Norway": ["norway", "oslo", "bergen"],
    "Denmark": ["denmark", "copenhagen", "aarhus"],
    "Finland": ["finland", "helsinki", "espoo"],
    "Austria": ["austria", "vienna", "wien"],
    "Belgium": ["belgium", "brussels", "antwerp", "ghent"],
    "Czechia": ["czechia", "czech republic", "prague", "brno"],
    "Romania": ["romania", "bucharest", "cluj", "timisoara"],
    "Estonia": ["estonia", "tallinn", "tartu"],
    "Ukraine": ["ukraine", "kyiv", "kiev", "lviv"],
    "Turkey": ["turkey", "türkiye", "istanbul", "ankara", "izmir"],
    "Israel": ["israel", "tel aviv", "haifa", "jerusalem"],
    "United Arab Emirates": ["united arab emirates", "uae", "dubai", "abu dhabi"],
    "Nigeria": ["nigeria", "lagos", "abuja", "ibadan", "port harcourt"],
    "Ghana": ["ghana", "accra", "kumasi"],
    "Kenya": ["kenya", "nairobi", "mombasa"],
    "South Africa": ["south africa", "cape town", "johannesburg", "durban", "pretoria"],
    "Egypt": ["egypt", "cairo", "alexandria", "giza"],
    "Morocco": ["morocco", "casablanca", "rabat", "marrakesh"],
    "India": ["india", "bengaluru", "bangalore", "hyderabad", "mumbai", "delhi", "gurgaon",
              "gurugram", "noida", "pune", "chennai", "kolkata"],
    "Pakistan": ["pakistan", "karachi", "lahore", "islamabad"],
    "Bangladesh": ["bangladesh", "dhaka", "chittagong"],
    "Singapore": ["singapore"],
    "Japan": ["japan", "tokyo", "osaka", "kyoto"],
    "South Korea": ["south korea", "korea", "seoul", "busan"],
    "China": ["china", "beijing", "shanghai", "shenzhen", "guangzhou", "hangzhou"],
    "Hong Kong": ["hong kong"],
    "Taiwan": ["taiwan", "taipei"],
    "Vietnam": ["vietnam", "viet nam", "hanoi", "ho chi minh", "saigon", "da nang"],
    "Philippines": ["philippines", "manila", "cebu", "makati"],
    "Indonesia": ["indonesia", "jakarta", "bandung", "surabaya"],
    "Malaysia": ["malaysia", "kuala lumpur", "penang"],
    "Thailand": ["thailand", "bangkok", "chiang mai"],
    "Australia": ["australia", "sydney", "melbourne", "brisbane", "perth", "canberra"],
    "New Zealand": ["new zealand", "auckland", "wellington", "christchurch"],
    "Brazil": ["brazil", "brasil", "sao paulo", "são paulo", "rio de janeiro", "belo horizonte"],
    "Argentina": ["argentina", "buenos aires", "cordoba"],
    "Chile": ["chile", "santiago"],
    "Colombia": ["colombia", "bogota", "bogotá", "medellin", "medellín"],
    "Mexico": ["mexico", "méxico", "mexico city", "guadalajara", "monterrey"],
}

#: Regional shorthands that appear in postings and cover several countries.
REGIONS: dict[str, list[str]] = {
    "emea": ["United Kingdom", "Ireland", "Germany", "France", "Netherlands", "Spain", "Portugal",
             "Italy", "Poland", "Switzerland", "Sweden", "Norway", "Denmark", "Finland", "Austria",
             "Belgium", "Czechia", "Romania", "Estonia", "Ukraine", "Turkey", "Israel",
             "United Arab Emirates", "Nigeria", "Ghana", "Kenya", "South Africa", "Egypt", "Morocco"],
    "eu": ["Ireland", "Germany", "France", "Netherlands", "Spain", "Portugal", "Italy", "Poland",
           "Sweden", "Denmark", "Finland", "Austria", "Belgium", "Czechia", "Romania", "Estonia"],
    "europe": ["United Kingdom", "Ireland", "Germany", "France", "Netherlands", "Spain", "Portugal",
               "Italy", "Poland", "Switzerland", "Sweden", "Norway", "Denmark", "Finland", "Austria",
               "Belgium", "Czechia", "Romania", "Estonia", "Ukraine"],
    "apac": ["India", "Singapore", "Japan", "South Korea", "China", "Hong Kong", "Taiwan", "Vietnam",
             "Philippines", "Indonesia", "Malaysia", "Thailand", "Australia", "New Zealand"],
    "latam": ["Brazil", "Argentina", "Chile", "Colombia", "Mexico"],
    "americas": ["United States", "Canada", "Brazil", "Argentina", "Chile", "Colombia", "Mexico"],
    "north america": ["United States", "Canada", "Mexico"],
    "africa": ["Nigeria", "Ghana", "Kenya", "South Africa", "Egypt", "Morocco"],
}

COUNTRIES: list[str] = [ANYWHERE] + sorted(k for k in COUNTRY_ALIASES if k != ANYWHERE)

_WORD = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    return " " + _WORD.sub(" ", (text or "").lower()).strip() + " "


def _alias_hit(blob: str, alias: str) -> bool:
    return f" {_WORD.sub(' ', alias).strip()} " in blob


def country_matches(location_text: str, wanted: list[str] | None) -> bool:
    """True when the posting's location names one of the wanted countries.

    Regional shorthands count: a posting in "Remote, EMEA" matches Germany, and a
    posting anywhere matches when `Anywhere / Worldwide` is selected.
    """
    if not wanted:
        return True
    blob = _norm(location_text)
    chosen = {c.strip() for c in wanted if c.strip()}

    for country in chosen:
        for alias in COUNTRY_ALIASES.get(country, [country]):
            if _alias_hit(blob, alias):
                return True

    for region, members in REGIONS.items():          # "Remote, EMEA" covers its members
        if _alias_hit(blob, region) and chosen & set(members):
            return True

    if ANYWHERE in chosen and not blob.strip():      # no location given at all
        return True
    return False


def countries_for(location_text: str) -> list[str]:
    """Which countries a posting's location appears to name. Used for diagnostics."""
    blob = _norm(location_text)
    return [c for c, aliases in COUNTRY_ALIASES.items()
            if c != ANYWHERE and any(_alias_hit(blob, a) for a in aliases)]
