"""
Text preprocessing module for the Spam Message Detector.

Each function below corresponds to one step in the NLP preprocessing
pipeline.  The functions are intentionally simple and composable so
that the same pipeline can be applied during training and inference.

Preprocessing steps (in order):
1. Lowercasing            - normalises case so "FREE" and "free" match
2. Remove special chars   - strips punctuation/symbols, keeps letters/numbers/spaces
3. Whitespace normalisation - collapses multiple spaces into one, trims ends
4. Tokenisation           - splits the text into individual words
5. Stopword removal       - drops common words ("the", "is", "at") that add noise
6. Rejoin tokens          - turns the token list back into a single string
"""

import re
import string

import nltk
from nltk.corpus import stopwords

# ---------------------------------------------------------------------------
# NLTK resource bootstrap
# ---------------------------------------------------------------------------
# Download stopwords the first time the module is imported.  We use
# quiet=True and a try/except so this works in offline environments
# where the data may already be present or the download may be blocked.
_NLTK_DOWNLOADED = False

def _ensure_nltk_resources():
    global _NLTK_DOWNLOADED
    if _NLTK_DOWNLOADED:
        return
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
        except Exception:
            pass
    _NLTK_DOWNLOADED = True


_ensure_nltk_resources()

# Cache the English stopword set once (frozenset = O(1) lookup).
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    # Fallback: a small hard-coded set if NLTK data is unavailable.
    STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "on", "at", "by", "for", "with", "about", "against",
        "between", "into", "through", "during", "before", "after", "above",
        "below", "from", "up", "down", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "s", "t", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
        "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "what", "which", "who",
        "whom", "this", "that", "these", "those", "am",
    }


def lowercase(text: str) -> str:
    """Convert all characters to lowercase."""
    return text.lower()


def remove_special_characters(text: str) -> str:
    """
    Remove punctuation and special symbols.

    We keep:
      - letters (a-z after lowercasing)
      - digits  (0-9)
      - whitespace

    Everything else (punctuation, emojis, currency symbols, URLs
    fragments) is replaced with a single space so that words that
    were separated by punctuation do not get glued together.
    """
    # Replace any character that is NOT a letter, digit, or whitespace
    # with a space.  This prevents "free!!!prize" becoming "freeprize".
    pattern = r"[^a-z0-9\s]"
    cleaned = re.sub(pattern, " ", text)
    return cleaned


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs into one and strip leading/trailing space."""
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split text on whitespace into a list of tokens."""
    if not text:
        return []
    return text.split(" ")


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Drop tokens that appear in the English stopword list."""
    return [t for t in tokens if t and t not in STOP_WORDS]


def preprocess_text(text: str) -> str:
    """
    Apply the full preprocessing pipeline to a single raw message.

    Order matters:
      lowercase -> remove_special_characters -> normalize_whitespace
      -> tokenize -> remove_stopwords -> rejoin
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    if not text.strip():
        return ""

    text = lowercase(text)
    text = remove_special_characters(text)
    text = normalize_whitespace(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)


def preprocess_series(messages) -> "pd.Series":
    """Apply preprocess_text to a pandas Series of messages."""
    import pandas as pd
    return messages.astype(str).apply(preprocess_text)
