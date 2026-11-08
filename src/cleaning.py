"""Reusable cleaning / normalization pipeline for Amazon Fine Food Reviews.

Single entry point:  clean_text(text) -> str

Step order (and why the order matters) is documented in README.md and in the
`clean_text` docstring. Everything is deterministic, so running it twice
gives identical output.
"""
from __future__ import annotations

import html
import re
import unicodedata

# --------------------------------------------------------------------------
# Regular expressions (each one is explained in the report / notebook)
# --------------------------------------------------------------------------
HTML_TAG_RE = re.compile(r"<[^>]+>")                         # <br />, <a href=...>, </b>
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)  # http://..., www....
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PRICE_RE = re.compile(r"\$\s?\d+(?:[.,]\d+)*")               # $3.99, $ 12, $1,200.50
REPEAT_PUNCT_RE = re.compile(r"([!?.])\1{1,}")               # "!!!" -> "!"
WORD_RE = re.compile(r"[a-z]+")                              # tokens after cleaning
WS_RE = re.compile(r"\s+")

# Contractions are expanded so that negation survives as the token "not".
CONTRACTIONS = [
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bain't\b"), "is not"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'m\b"), " am"),
    (re.compile(r"'d\b"), " would"),
    (re.compile(r"'s\b"), ""),          # possessive / "it's": carries no sentiment
]

# --------------------------------------------------------------------------
# Stop words: spaCy's list MINUS negations / sentiment-bearing function words
# --------------------------------------------------------------------------
KEEP_WORDS = {
    "no", "not", "nor", "never", "none", "nothing", "nobody", "nowhere", "neither",
    "cannot", "without", "n't",
    "very", "too", "really", "quite", "more", "most", "less", "least", "enough",
    "but", "however", "although", "though", "only", "just", "against", "down", "up",
    "off", "over", "under", "again", "always", "never",
}


def _build_stopwords() -> frozenset[str]:
    from spacy.lang.en.stop_words import STOP_WORDS
    return frozenset(w for w in STOP_WORDS if w not in KEEP_WORDS)


STOP_WORDS = _build_stopwords()

# --------------------------------------------------------------------------
# Lemmatizer: spaCy lookup-table lemmatizer (needs spacy-lookups-data), cached
# per word type. A few overrides fix lookup errors that would hurt sentiment
# ("better"/"best" -> "well", "worse" -> "wrong", "cookies" -> "cooky").
# --------------------------------------------------------------------------
LEMMA_OVERRIDES = {
    "better": "better", "best": "best", "worse": "worse", "worst": "worst",
    "cookies": "cookie", "cooky": "cookie", "brownies": "brownie",
    "smoothies": "smoothie", "veggies": "veggie", "goodies": "goodie",
    "less": "less", "more": "more", "most": "most",
}
_NLP = None
_LEMMA_CACHE: dict[str, str] = dict(LEMMA_OVERRIDES)


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.blank("en")
        _NLP.add_pipe("lemmatizer", config={"mode": "lookup"})
        _NLP.initialize()
    return _NLP


def lemmatize_types(words) -> None:
    """Fill the lemma cache for many word types at once."""
    todo = [w for w in set(words) if w not in _LEMMA_CACHE]
    if not todo:
        return
    nlp = _get_nlp()
    for w, doc in zip(todo, nlp.pipe(todo, batch_size=2000)):
        _LEMMA_CACHE[w] = doc[0].lemma_.lower() if len(doc) else w


def lemma(word: str) -> str:
    if word not in _LEMMA_CACHE:
        lemmatize_types([word])
    return _LEMMA_CACHE[word]


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------
def normalize_unicode(text: str) -> str:
    """Step 1 - Unicode NFC + typographic quote/dash folding."""
    text = unicodedata.normalize("NFC", text)
    return (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"')
                .replace("–", "-").replace("—", "-")
                .replace(" ", " "))


def remove_noise(text: str) -> str:
    """Step 2 - HTML tags/entities, URLs, e-mails, prices, repeated punctuation."""
    text = html.unescape(text)            # &quot; &#34; &amp; -> real characters
    text = HTML_TAG_RE.sub(" ", text)     # tags removed AFTER unescape (catches &lt;br&gt;)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PRICE_RE.sub(" ", text)
    text = REPEAT_PUNCT_RE.sub(r"\1", text)
    return text


def expand_contractions(text: str) -> str:
    for pattern, repl in CONTRACTIONS:
        text = pattern.sub(repl, text)
    return text


def tokenize(text: str) -> list[str]:
    """Letters-only tokens. Digits and punctuation are dropped."""
    return WORD_RE.findall(text)


def clean_text(text: str, *, remove_stopwords: bool = True, lemmatize: bool = True,
               min_len: int = 2) -> str:
    """Clean one review. ORDER IS IMPORTANT:

    1. Unicode NFC      - before anything else, so later regexes see canonical chars
    2. Noise removal    - HTML/URL/etc. BEFORE case folding and tokenization, because
                          tags and URLs would otherwise be split into junk tokens
    3. Case folding     - lower-case (after noise removal: URLs/tags are case-insensitive anyway)
    4. Contractions     - after lowercasing, before tokenizing so "don't" keeps its "not"
    5. Tokenization     - letters-only regex tokens (drops digits / punctuation)
    6. Stop words       - removed AFTER contraction expansion so negations are known;
                          negations are protected via KEEP_WORDS
    7. Lemmatization    - last, on the surviving tokens
    8. Whitespace       - single spaces on join
    """
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    text = remove_noise(text)
    text = text.lower()
    text = expand_contractions(text)
    tokens = tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if lemmatize:
        lemmatize_types(tokens)
        tokens = [_LEMMA_CACHE[t] for t in tokens]
    tokens = [t for t in tokens if len(t) >= min_len or t in ("i",)]
    return " ".join(tokens)


def surface_features(text: str) -> dict:
    """Signals that case folding / punctuation removal would destroy - kept as columns."""
    if not isinstance(text, str):
        return {"n_exclaim": 0, "caps_ratio": 0.0}
    letters = [c for c in text if c.isalpha()]
    caps = sum(c.isupper() for c in letters)
    return {"n_exclaim": text.count("!"),
            "caps_ratio": round(caps / len(letters), 4) if letters else 0.0}
