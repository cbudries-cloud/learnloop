"""Storage, queue, grading, explanations and scheduler tuning."""
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from . import llm
from . import scheduler as sch

HOME = Path(os.environ.get("LEARNLOOP_HOME", "~/.learnloop")).expanduser()
DB_PATH = Path(os.environ.get("LEARNLOOP_DB", HOME / "learnloop.db"))
NEW_PER_DAY = int(os.environ.get("LEARNLOOP_NEW_PER_DAY", "10"))


# ---------- storage ----------

def db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY, deck TEXT NOT NULL, title TEXT, uri TEXT NOT NULL, chars INTEGER, ts REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS cards (
            id INTEGER PRIMARY KEY, deck TEXT NOT NULL, source_id INTEGER NOT NULL REFERENCES sources(id),
            kind TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL, key_points TEXT NOT NULL,
            quote TEXT, ts REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY, card_id INTEGER NOT NULL, ts REAL NOT NULL, grade TEXT NOT NULL,
            model_grade TEXT, answer TEXT, missed_points TEXT, misconception TEXT, ms INTEGER);
        CREATE TABLE IF NOT EXISTS explanations (
            card_id INTEGER PRIMARY KEY, text TEXT NOT NULL, misses_covered INTEGER NOT NULL, ts REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS params (
            id INTEGER PRIMARY KEY, ts REAL NOT NULL, params TEXT NOT NULL, rationale TEXT,
            loss_before REAL, loss_after REAL, n INTEGER, status TEXT NOT NULL);
        """)


def add_source(deck: str, uri: str, title: str, chars: int, cards: list[dict]) -> int:
    now = time.time()
    with db() as c:
        sid = c.execute("INSERT INTO sources (deck, title, uri, chars, ts) VALUES (?,?,?,?,?)",
                        (deck, title, uri, chars, now)).lastrowid
        c.executemany("INSERT INTO cards (deck, source_id, kind, question, answer, key_points, quote, ts) VALUES (?,?,?,?,?,?,?,?)",
                      [(deck, sid, k["kind"], k["question"], k["answer"], json.dumps(k["key_points"]), k["quote"], now)
                       for k in cards])
    return sid


def load_cards(deck: str | None = None) -> dict:
    q = ("SELECT cards.*, sources.title AS source_title, sources.uri AS source_uri FROM cards "
         "JOIN sources ON sources.id = cards.source_id")
    with db() as c:
        rows = c.execute(q + (" WHERE cards.deck=?" if deck else "") + " ORDER BY cards.id", (deck,) if deck else ())
        return {r["id"]: {**dict(r), "key_points": json.loads(r["key_points"])} for r in rows}


def decks() -> list[dict]:
    with db() as c:
        return [dict(r) for r in c.execute(
            "SELECT deck, COUNT(*) AS cards, (SELECT COUNT(*) FROM sources s WHERE s.deck = cards.deck) AS sources "
            "FROM cards GROUP BY deck ORDER BY deck")]


def active_params() -> dict:
    with db() as c:
        row = c.execute("SELECT params FROM params WHERE status='active' ORDER BY id DESC LIMIT 1").fetchone()
    return sch.clamp_params(json.loads(row["params"]) if row else {})


def all_reviews() -> list[dict]:
    with db() as c:
        return [dict(r) for r in c.execute("SELECT * FROM reviews ORDER BY ts, id")]


def card_states(reviews, kinds, p) -> dict:
    """Derive every card's state from the review log under params p."""
    states = {}
    for rv in reviews:
        cid = rv["card_id"]
        prev = states.get(cid)
        elapsed = (rv["ts"] - prev["last_ts"]) / 86400 if prev else 0.0
        st = sch.update(prev, rv["grade"], elapsed, kinds.get(cid, "concept"), p)
        st["last_ts"] = rv["ts"]
        st["due"] = rv["ts"] + sch.next_due_days(st, rv["grade"], p) * 86400
        st["reviews"] = (prev["reviews"] if prev else 0) + 1
        st["misses"] = (prev["misses"] if prev else 0) + (rv["grade"] in ("again", "hard"))
        states[cid] = st
    return states


# ---------- model calls ----------

GRADE_SYSTEM = """You grade one free-text recall answer on a spaced-repetition card.
Grade against the key points, not wording. Synonyms and paraphrase count.
- again: the core idea is missing or the answer states something wrong.
- hard: partly right; at least one key point missing or vague.
- good: every key point present and nothing wrong.
Never award 'easy'; the learner decides that.
If the answer contains a wrong belief, name it in misconception (else empty string).
feedback: 1-2 plain sentences telling the learner what they missed or confirming what they got. No em dashes."""

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "grade": {"type": "string", "enum": ["again", "hard", "good"]},
        "missed_points": {"type": "array", "items": {"type": "string"}},
        "misconception": {"type": "string"},
        "feedback": {"type": "string"},
    },
    "required": ["grade", "missed_points", "misconception", "feedback"],
}

EXPLAIN_SYSTEM = """A learner keeps getting the same spaced-repetition card wrong. Write a short explanation aimed at the specific thing they keep missing, using their actual past answers to diagnose the confusion.
- Start with the diagnosis in one sentence (what their answers have in common).
- Then explain the idea a different way than the card answer does: a concrete example, an analogy, or the mechanism. Under 150 words total.
- Close with a one-line hook that makes the right answer easy to retrieve next time.
- Stay within the source material provided. No em dashes. Plain text, short paragraphs."""

EXPLAIN_SCHEMA = {"type": "object", "properties": {"explanation": {"type": "string"}}, "required": ["explanation"]}

TUNE_SYSTEM = """You tune the scheduler of a personal spaced-repetition app. You get the current parameters, their allowed bounds, and a replay of the learner's full review history: log loss of predicted recall vs actual outcome, calibration bins (predicted vs observed recall), calibration by card kind, and the cards they struggle with.

Your job: propose parameters that would have predicted this learner's recall better, so reviews land near the target retention without wasted repetitions.
- Reason from the evidence. If observed recall is above predicted, memories are lasting longer than modeled (raise growth or initial stability); below, the reverse.
- Use kind_multiplier when one card kind decays differently from the others.
- Make conservative moves when there are few reviews. Under ~30 scored reviews, change at most one or two values slightly, or nothing, and say the data is thin.
- Every proposal is replayed against history and rejected if it does not lower log loss, so do not guess wildly.
rationale: 2-4 sentences naming the evidence for each change. No em dashes."""


def tune_schema():
    num = {"type": "number"}
    return {
        "type": "object",
        "properties": {
            "params": {
                "type": "object",
                "properties": {
                    "init_stability": {"type": "object", "properties": {g: num for g in sch.GRADES}, "required": list(sch.GRADES)},
                    "kind_multiplier": {"type": "object", "properties": {k: num for k in sch.KINDS}, "required": sch.KINDS},
                    **{k: num for k in sch.BOUNDS},
                },
                "required": ["init_stability", "kind_multiplier", *sch.BOUNDS],
            },
            "rationale": {"type": "string"},
        },
        "required": ["params", "rationale"],
    }


def grade_answer(card_id: int, answer: str) -> dict:
    card = load_cards()[card_id]
    user = json.dumps({"question": card["question"], "reference_answer": card["answer"],
                       "key_points": card["key_points"], "learner_answer": answer}, indent=1)
    return llm.structured(llm.FAST_MODEL, GRADE_SYSTEM, user, "grade", GRADE_SCHEMA, 2000, effort="low")


def maybe_explain(card, reviews_for_card):
    misses = [r for r in reviews_for_card if r["grade"] in ("again", "hard")]
    if len(misses) < 2:
        return None
    with db() as c:
        row = c.execute("SELECT * FROM explanations WHERE card_id=?", (card["id"],)).fetchone()
    if row and row["misses_covered"] >= len(misses):
        return row["text"]
    history = [{"answer": r["answer"], "grade": r["grade"], "missed": json.loads(r["missed_points"] or "[]"),
                "misconception": r["misconception"]} for r in misses[-5:]]
    user = json.dumps({"question": card["question"], "reference_answer": card["answer"], "key_points": card["key_points"],
                       "source_quote": card["quote"], "source": card["source_title"], "past_wrong_answers": history}, indent=1)
    text = llm.structured(llm.FAST_MODEL, EXPLAIN_SYSTEM, user, "explanation", EXPLAIN_SCHEMA, 3000)["explanation"]
    with db() as c:
        c.execute("INSERT OR REPLACE INTO explanations VALUES (?,?,?,?)", (card["id"], text, len(misses), time.time()))
    return text


# ---------- API ----------

def local_day(ts):
    return datetime.fromtimestamp(ts).date()


def _explanations():
    with db() as c:
        return {r["card_id"]: r["text"] for r in c.execute("SELECT card_id, text FROM explanations")}


def queue(deck=None):
    cards, p, reviews = load_cards(deck), active_params(), all_reviews()
    allcards = load_cards()
    states = card_states(reviews, {i: c["kind"] for i, c in allcards.items()}, p)
    now = time.time()
    due = sorted((s["due"], cid) for cid, s in states.items() if s["due"] <= now and cid in cards)
    first_seen = {}
    for r in reviews:
        first_seen.setdefault(r["card_id"], r["ts"])
    new_today = sum(1 for ts in first_seen.values() if local_day(ts) == local_day(now))
    unseen = [cid for cid in cards if cid not in states]
    by_source = {}
    for cid in unseen:  # rotate new cards across sources
        by_source.setdefault(cards[cid]["source_id"], []).append(cid)
    interleaved = []
    while any(by_source.values()):
        for lst in by_source.values():
            if lst:
                interleaved.append(lst.pop(0))
    new = interleaved[: max(0, NEW_PER_DAY - new_today)]
    expl = _explanations()
    items = [{**cards[cid], "is_new": False, "explanation": expl.get(cid)} for _, cid in due]
    items += [{**cards[cid], "is_new": True, "explanation": None} for cid in new]
    upcoming = sorted(s["due"] for cid, s in states.items() if s["due"] > now and cid in cards)
    return {"queue": items, "next_due": upcoming[0] if upcoming else None,
            "unseen_remaining": len(unseen) - len(new), "new_per_day": NEW_PER_DAY}


def review(body):
    card_id, grade = int(body["card_id"]), body["grade"]
    if grade not in sch.GRADES:
        raise ValueError("bad grade")
    cards = load_cards()
    card = cards[card_id]
    with db() as c:
        c.execute("INSERT INTO reviews (card_id, ts, grade, model_grade, answer, missed_points, misconception, ms) VALUES (?,?,?,?,?,?,?,?)",
                  (card_id, time.time(), grade, body.get("model_grade"), body.get("answer", ""),
                   json.dumps(body.get("missed_points", [])), body.get("misconception", ""), body.get("ms")))
    reviews = all_reviews()
    st = card_states(reviews, {i: c["kind"] for i, c in cards.items()}, active_params())[card_id]
    explanation = maybe_explain(card, [r for r in reviews if r["card_id"] == card_id])
    return {"due": st["due"], "interval_days": (st["due"] - time.time()) / 86400, "explanation": explanation}


def stats(deck=None):
    allcards, p, reviews = load_cards(), active_params(), all_reviews()
    kinds = {i: c["kind"] for i, c in allcards.items()}
    states = card_states(reviews, kinds, p)
    cards = {i: c for i, c in allcards.items() if not deck or c["deck"] == deck}
    now = time.time()
    groups = {}
    for cid, c in cards.items():
        e = groups.setdefault(c["source_id"], {"title": c["source_title"], "uri": c["source_uri"], "deck": c["deck"],
                                               "cards": 0, "seen": 0, "recall": []})
        e["cards"] += 1
        if cid in states:
            e["seen"] += 1
            s = states[cid]
            e["recall"].append(sch.recall_prob(s["stability"], (now - s["last_ts"]) / 86400))
    for e in groups.values():
        e["recall"] = round(sum(e["recall"]) / len(e["recall"]), 3) if e["recall"] else None
    with db() as c:
        history = [dict(r) for r in c.execute(
            "SELECT id, ts, rationale, loss_before, loss_after, n, status FROM params ORDER BY id DESC LIMIT 20")]
    expl = _explanations()
    mine = {cid: s for cid, s in states.items() if cid in cards}
    trouble = sorted(((s["misses"], cid) for cid, s in mine.items() if s["misses"] >= 2), reverse=True)
    week = [0] * 7
    for s in mine.values():
        d = max(0, (local_day(s["due"]) - local_day(now)).days)
        if d < 7:
            week[d] += 1
    return {
        "totals": {"cards": len(cards), "seen": len(mine),
                   "reviews": sum(1 for r in reviews if r["card_id"] in cards),
                   "due_now": sum(1 for s in mine.values() if s["due"] <= now)},
        "forecast": week,
        "sources": list(groups.values()),
        "replay": sch.replay(reviews, kinds, p),
        "params": p,
        "tuning_history": history,
        "trouble": [{"card": cards[cid], "misses": m, "explanation": expl.get(cid)} for m, cid in trouble[:12]],
    }


def propose_params(reviews, kinds, cards, p) -> dict:
    """Ask the model for new params given the replayed history. Pure: no DB writes."""
    before = sch.replay(reviews, kinds, p)
    states = card_states(reviews, kinds, p)
    struggles = [{"kind": kinds.get(cid), "question": cards[cid]["question"] if cid in cards else None,
                  "reviews": s["reviews"], "misses": s["misses"], "stability_days": round(s["stability"], 2)}
                 for cid, s in states.items() if s["misses"]][:25]
    user = json.dumps({"current_params": p, "bounds": sch.BOUNDS,
                       "bounds_note": "init_stability 0.05-60 days; kind_multiplier 0.25-4",
                       "replay": before, "struggling_cards": struggles,
                       "total_reviews": len(reviews), "cards_seen": len(states)}, indent=1)
    out = llm.structured(llm.DEEP_MODEL, TUNE_SYSTEM, user, "proposal", tune_schema(), 8000, effort="high")
    proposed = sch.clamp_params(out["params"])
    after = sch.replay(reviews, kinds, proposed)
    return {"params": proposed, "rationale": out["rationale"], "loss_before": before["log_loss"],
            "loss_after": after["log_loss"], "n": before["n"],
            "improves": after["log_loss"] is not None and after["log_loss"] < before["log_loss"]}


def tune():
    cards, p, reviews = load_cards(), active_params(), all_reviews()
    kinds = {i: c["kind"] for i, c in cards.items()}
    if sch.replay(reviews, kinds, p)["n"] == 0:
        raise ValueError("No scored reviews yet. A card needs at least two reviews before it can test the scheduler.")
    r = propose_params(reviews, kinds, cards, p)
    with db() as c:
        r["id"] = c.execute("INSERT INTO params (ts, params, rationale, loss_before, loss_after, n, status) VALUES (?,?,?,?,?,?,?)",
                            (time.time(), json.dumps(r["params"]), r["rationale"], r["loss_before"], r["loss_after"],
                             r["n"], "proposed" if r["improves"] else "rejected")).lastrowid
    r["current"] = p
    return r


def apply_params(body):
    with db() as c:
        row = c.execute("SELECT status FROM params WHERE id=?", (body["id"],)).fetchone()
        if not row or row["status"] != "proposed":
            raise ValueError("Only an improving, unapplied proposal can be applied.")
        c.execute("UPDATE params SET status='retired' WHERE status='active'")
        c.execute("UPDATE params SET status='active' WHERE id=?", (body["id"],))
    return {"ok": True}
