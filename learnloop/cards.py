"""Write cards from source text, and keep only cards whose quote is verbatim in the source."""
import re
import sys
import unicodedata

from . import llm
from .scheduler import KINDS

SYSTEM = """You write spaced-repetition cards that help one person learn the material you are given. They are smart and new to the subject; they want to understand and reason with it, not recite trivia.

Rules:
- Every card must be answerable from the source text. Add nothing from outside it.
- Prefer cards that test understanding: why something is true, how two ideas differ, how a process works, when an idea applies.
- Use 'number' cards only for quantities that matter to understanding. Never ask for dates of publication, author names or other incidental details unless they are the point of the material.
- Questions are short, specific and self-contained (name the subject; never say "this text" or "the author"). Free-text recall, not multiple choice.
- answer: 1-3 sentences. key_points: the 1-4 ideas a correct answer must contain; grading checks these.
- quote: copy one passage from the source, verbatim, character for character, 40-300 characters, that best supports the answer. It is checked mechanically; a paraphrase gets the card discarded.
- Cover the most important ideas first. No em dashes."""

SCHEMA = {
    "type": "object",
    "properties": {
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": KINDS},
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                    "quote": {"type": "string"},
                },
                "required": ["kind", "question", "answer", "key_points", "quote"],
            },
        }
    },
    "required": ["cards"],
}


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "}))
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_in_source(quote: str, source_norm: str) -> bool:
    q = normalize(quote).strip(" .…\"'")
    return len(q) >= 20 and q in source_norm


def write_cards(title: str, chunk: str, n: int) -> tuple[list[dict], int]:
    """Return (verified cards, number dropped)."""
    out = llm.structured(
        llm.DEEP_MODEL, SYSTEM,
        f"Source: {title}\n\nWrite about {n} cards.\n\n<source>\n{chunk}\n</source>",
        "cards", SCHEMA, max_tokens=16000,
    )
    src = normalize(chunk)
    kept = [c for c in out["cards"] if quote_in_source(c["quote"], src)]
    dropped = len(out["cards"]) - len(kept)
    if dropped:
        print(f"    dropped {dropped} card(s) whose quote was not verbatim in the source", file=sys.stderr)
    return kept, dropped
