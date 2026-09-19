"""How closely a job title resembles what you searched for.

Counting shared words is too blunt for job titles. "Data Analytics" has to find
"Analytics Engineer Intern" and "Business Analyst", which share only part of the
phrase, while "Backend Engineer" must not drag in every "Sales Engineer".

So words are first folded into concepts (`analyst`, `analytics`, `analysis`,
`insights` and `BI` are all one idea), and concepts are weighted by how much they
narrow a search. `engineer` barely narrows anything, so it carries little weight;
`backend` or `kubernetes` carries full weight. The score is the share of the
query's weight that the title covers, which makes it a tunable dial rather than a
yes/no rule.
"""
from __future__ import annotations

import difflib
import re
from typing import Iterable, Optional

#: Words that say nothing about the role.
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "with", "in", "at", "to", "on", "our", "you",
    "senior", "junior", "staff", "lead", "principal", "mid", "level", "sr", "jr",
    "i", "ii", "iii", "iv", "new", "grad", "entry", "full", "part", "time", "remote",
}

#: concept -> the words that express it. First match wins, so order does not matter.
CONCEPTS: dict[str, list[str]] = {
    "analytics": ["analytics", "analytic", "analyst", "analysis", "analytical", "analyse",
                  "analyze", "insights", "insight", "bi", "businessintelligence", "reporting",
                  "reports", "dashboard", "dashboards", "tableau", "powerbi", "looker", "metrics"],
    "data": ["data", "dataset", "datasets", "database", "databases", "warehouse", "warehousing",
             "etl", "elt", "pipeline", "pipelines", "bigdata", "sql", "snowflake", "dbt"],
    "datascience": ["datascience", "scientist", "science", "statistics", "statistical",
                    "statistician", "modeling", "modelling", "econometrics", "quant"],
    "ml": ["ml", "machinelearning", "ai", "artificialintelligence", "deeplearning", "deep",
           "learning", "machine", "nlp", "llm", "llms", "genai", "generative", "computervision",
           "vision", "pytorch", "tensorflow", "mlops"],
    "software": ["software", "engineer", "engineering", "developer", "development", "develop",
                 "programmer", "programming", "swe", "sde", "coder", "code", "technical"],
    "backend": ["backend", "back", "server", "serverside", "api", "apis", "microservices",
                "distributed", "golang", "django", "fastapi", "rails", "spring"],
    "frontend": ["frontend", "front", "ui", "react", "angular", "vue", "javascript", "typescript",
                 "css", "html", "web", "webapp"],
    "fullstack": ["fullstack", "stack"],
    "mobile": ["mobile", "ios", "android", "swift", "kotlin", "flutter", "reactnative"],
    "devops": ["devops", "sre", "reliability", "infrastructure", "infra", "platform", "cloud",
               "kubernetes", "docker", "terraform", "aws", "gcp", "azure", "deployment"],
    "security": ["security", "infosec", "appsec", "cybersecurity", "cyber", "cryptography"],
    "qa": ["qa", "quality", "test", "testing", "tester", "sdet", "automation"],
    "product": ["product", "pm", "roadmap"],
    "design": ["design", "designer", "ux", "ui", "usability", "graphic", "visual"],
    "management": ["manager", "management", "director", "head", "chief", "vp", "supervisor"],
    "sales": ["sales", "account", "revenue", "gtm", "quota", "sdr", "ae"],
    "marketing": ["marketing", "growth", "seo", "brand", "campaign", "content"],
    "finance": ["finance", "financial", "accounting", "accountant", "treasury", "audit",
                "controller", "fp&a"],
    "people": ["people", "hr", "recruiting", "recruiter", "talent", "hiring"],
    "support": ["support", "customer", "success", "helpdesk", "servicedesk"],
    "operations": ["operations", "operational", "ops", "logistics", "supply"],
    "research": ["research", "researcher", "scientific"],
    "python": ["python", "pythonic"],
    "java": ["java"],
    "dotnet": ["dotnet", "csharp", "net"],
    "php": ["php", "laravel", "wordpress"],
    "ruby": ["ruby"],
    "blockchain": ["blockchain", "web3", "crypto", "solidity", "smartcontract", "defi"],
    "internship": ["intern", "internship", "coop", "apprentice", "trainee", "graduate", "campus",
                   "student", "university", "placement"],
}

#: How much a concept narrows a search. Low means "almost every posting has this".
CONCEPT_WEIGHT: dict[str, float] = {
    "software": 0.40, "management": 0.40, "operations": 0.45, "product": 0.55,
    "support": 0.55, "sales": 0.60, "marketing": 0.60, "people": 0.60, "research": 0.55,
    "internship": 0.50, "qa": 0.75, "design": 0.75, "finance": 0.75,
}
DEFAULT_WEIGHT = 1.0        # domain and technology concepts narrow a search a lot
UNKNOWN_WEIGHT = 0.85       # a word we do not recognise is probably distinctive

#: Confidence that two things mean the same, highest first.
EXACT, STEM, CONCEPT, FUZZY = 1.0, 0.94, 0.86, 0.72
STEM_PREFIX = 5             # "analytics" / "analyst" share five characters
CONTAINS_PREFIX = 4         # "engineer" inside "engineering"
FUZZY_CUTOFF = 0.84

_SPLIT = re.compile(r"[^a-z0-9+#]+")

WORD_TO_CONCEPT: dict[str, str] = {
    word: concept for concept, words in CONCEPTS.items() for word in words
}


def tokenize(text: str) -> list[str]:
    """Words worth comparing, with separators removed so 'back-end' reads as 'backend'."""
    raw = _SPLIT.split((text or "").lower())
    out: list[str] = []
    for i, word in enumerate(raw):
        if not word or word in STOPWORDS:
            continue
        out.append(word)
        # glue adjacent words so "machine learning" and "full stack" resolve as one idea
        if i + 1 < len(raw) and raw[i + 1]:
            joined = word + raw[i + 1]
            if joined in WORD_TO_CONCEPT:
                out.append(joined)
    return out


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def stem_equal(a: str, b: str) -> bool:
    """Whether two words are inflections of each other."""
    if a == b:
        return True
    if len(a) >= CONTAINS_PREFIX and len(b) >= CONTAINS_PREFIX and (a.startswith(b) or b.startswith(a)):
        return True
    return (_common_prefix(a, b) >= STEM_PREFIX
            and len(a) >= STEM_PREFIX and len(b) >= STEM_PREFIX)


def concept_of(word: str) -> Optional[str]:
    """The idea a word expresses, if we know one."""
    hit = WORD_TO_CONCEPT.get(word)
    if hit:
        return hit
    for known, concept in WORD_TO_CONCEPT.items():
        if stem_equal(word, known):
            return concept
    return None


def weight_of(word: str) -> float:
    concept = concept_of(word)
    if concept is None:
        return UNKNOWN_WEIGHT
    return CONCEPT_WEIGHT.get(concept, DEFAULT_WEIGHT)


def word_similarity(query_word: str, title_word: str) -> float:
    """0 to 1: how confidently one title word answers one query word."""
    if query_word == title_word:
        return EXACT
    if stem_equal(query_word, title_word):
        return STEM
    q_concept, t_concept = concept_of(query_word), concept_of(title_word)
    if q_concept is not None and q_concept == t_concept:
        return CONCEPT
    ratio = difflib.SequenceMatcher(None, query_word, title_word).ratio()
    return FUZZY if ratio >= FUZZY_CUTOFF else 0.0


def relevance(title: str, query: str | Iterable[str]) -> float:
    """How well `title` answers `query`, from 0 (unrelated) to 1 (says the same thing).

    The score is the share of the query's weighted meaning that the title covers, so
    missing a throwaway word like "engineer" costs far less than missing "backend".
    """
    query_words = tokenize(" ".join(query) if not isinstance(query, str) else query)
    title_words = tokenize(title)
    if not query_words:
        return 1.0
    if not title_words:
        return 0.0

    # Collapse the query to one entry per concept so "machine learning" is not counted twice.
    seen: dict[str, str] = {}
    for word in query_words:
        key = concept_of(word) or word
        if key not in seen or len(word) > len(seen[key]):
            seen[key] = word
    distinct = list(seen.values())

    total = sum(weight_of(w) for w in distinct)
    if not total:
        return 0.0
    earned = sum(
        weight_of(q) * max((word_similarity(q, t) for t in title_words), default=0.0)
        for q in distinct
    )
    return round(min(1.0, earned / total), 3)


def best_relevance(title: str, queries: Iterable[str]) -> float:
    """The best score across every search term the user entered."""
    scores = [relevance(title, q) for q in queries if q and q.strip()]
    return max(scores) if scores else 1.0


def explain(title: str, query: str) -> dict[str, object]:
    """Per-word breakdown, for debugging a surprising score."""
    query_words = tokenize(query)
    title_words = tokenize(title)
    rows = []
    for q in query_words:
        best, word = 0.0, None
        for t in title_words:
            sim = word_similarity(q, t)
            if sim > best:
                best, word = sim, t
        rows.append({"query_word": q, "concept": concept_of(q), "weight": weight_of(q),
                     "matched": word, "similarity": best})
    return {"title": title, "query": query, "score": relevance(title, query), "words": rows}
