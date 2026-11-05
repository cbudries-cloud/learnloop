"""Tunable memory model.

Each card has stability S (days until recall probability falls to 90%) and
difficulty D (1-10). Recall probability after t days uses the FSRS power curve
R(t) = (1 + t / (9 S))^-1. The next interval is the t at which R hits the
target retention.

Every number that shapes scheduling lives in PARAMS so the weekly tuning pass
can change it, and replay() scores any parameter set against the real review
log, so a proposed change is only accepted if it predicts past recall better.
"""
import math

GRADES = {"again": 1, "hard": 2, "good": 3, "easy": 4}
KINDS = ["concept", "fact", "process", "distinction", "number", "application"]

DEFAULT_PARAMS = {
    "init_stability": {"again": 0.3, "hard": 1.0, "good": 3.0, "easy": 7.0},
    "kind_multiplier": {k: 1.0 for k in KINDS},
    "growth": 1.6,
    "recall_bonus": 1.0,
    "hard_factor": 0.6,
    "easy_factor": 1.4,
    "lapse_factor": 0.3,
    "difficulty_step": 1.0,
    "init_difficulty": 5.0,
    "target_retention": 0.9,
    "min_stability": 0.2,
}

BOUNDS = {
    "growth": (0.2, 5.0),
    "recall_bonus": (0.0, 4.0),
    "hard_factor": (0.2, 1.0),
    "easy_factor": (1.0, 3.0),
    "lapse_factor": (0.05, 0.9),
    "difficulty_step": (0.0, 3.0),
    "init_difficulty": (1.0, 10.0),
    "target_retention": (0.75, 0.97),
    "min_stability": (0.05, 2.0),
}


def clamp_params(p: dict) -> dict:
    out = {**DEFAULT_PARAMS, **{k: v for k, v in p.items() if k in DEFAULT_PARAMS}}
    for k, (lo, hi) in BOUNDS.items():
        out[k] = min(hi, max(lo, float(out[k])))
    out["init_stability"] = {g: min(60.0, max(0.05, float(out["init_stability"].get(g, DEFAULT_PARAMS["init_stability"][g]))))
                             for g in GRADES}
    out["kind_multiplier"] = {k: min(4.0, max(0.25, float(out["kind_multiplier"].get(k, 1.0)))) for k in KINDS}
    return out


def recall_prob(stability: float, elapsed_days: float) -> float:
    return (1 + max(0.0, elapsed_days) / (9 * stability)) ** -1


def interval_days(stability: float, target: float) -> float:
    return 9 * stability * (1 / target - 1)


def update(state: dict | None, grade: str, elapsed_days: float, kind: str, p: dict) -> dict:
    """Return the new card state after a review. state=None means first review."""
    g = GRADES[grade]
    mult = p["kind_multiplier"].get(kind, 1.0)
    if state is None:
        s = p["init_stability"][grade] * mult
        d = p["init_difficulty"] - p["difficulty_step"] * (g - 3)
        lapses = 1 if g == 1 else 0
    else:
        s, d, lapses = state["stability"], state["difficulty"], state["lapses"]
        r = recall_prob(s, elapsed_days)
        if g == 1:
            s = s * p["lapse_factor"]
            lapses += 1
        else:
            factor = {2: p["hard_factor"], 3: 1.0, 4: p["easy_factor"]}[g]
            s = s * (1 + p["growth"] * mult * (11 - d) / 10 * (1 + p["recall_bonus"] * (1 - r)) * factor)
        d = d - p["difficulty_step"] * (g - 3)
    return {
        "stability": max(p["min_stability"], s),
        "difficulty": min(10.0, max(1.0, d)),
        "lapses": lapses,
    }


def next_due_days(state: dict, grade: str, p: dict) -> float:
    if grade == "again":
        return 10 / 1440  # re-show in the same session
    return max(1.0, interval_days(state["stability"], p["target_retention"]))


def replay(reviews: list[dict], kinds: dict, p: dict, score_from: float | None = None) -> dict:
    """Score params against history. reviews: ordered dicts with card_id, grade, ts (epoch s).

    For every non-first review, the model predicts recall probability; the
    outcome is 1 for hard/good/easy and 0 for again. Returns log loss, count,
    and calibration bins (predicted vs observed). With score_from, the whole
    history still drives card state but only reviews at or after that
    timestamp are scored (used for held-out evaluation).
    """
    states, last_ts = {}, {}
    loss, n = 0.0, 0
    bins = [{"lo": i / 10, "hi": (i + 1) / 10, "n": 0, "pred": 0.0, "obs": 0} for i in range(10)]
    by_kind = {}
    for rv in reviews:
        cid, grade, ts = rv["card_id"], rv["grade"], rv["ts"]
        kind = kinds.get(cid, "concept")
        st = states.get(cid)
        elapsed = (ts - last_ts[cid]) / 86400 if cid in last_ts else 0.0
        if st is not None and (score_from is None or ts >= score_from):
            r = min(0.999, max(0.001, recall_prob(st["stability"], elapsed)))
            y = 0 if grade == "again" else 1
            loss -= y * math.log(r) + (1 - y) * math.log(1 - r)
            n += 1
            b = bins[min(9, int(r * 10))]
            b["n"] += 1
            b["pred"] += r
            b["obs"] += y
            k = by_kind.setdefault(kind, {"n": 0, "pred": 0.0, "obs": 0})
            k["n"] += 1
            k["pred"] += r
            k["obs"] += y
        states[cid] = update(st, grade, elapsed, kind, p)
        last_ts[cid] = ts
    for b in bins:
        if b["n"]:
            b["pred"] = round(b["pred"] / b["n"], 3)
            b["obs"] = round(b["obs"] / b["n"], 3)
    for k in by_kind.values():
        k["pred"] = round(k["pred"] / k["n"], 3)
        k["obs"] = round(k["obs"] / k["n"], 3)
    return {
        "log_loss": round(loss / n, 4) if n else None,
        "n": n,
        "bins": [b for b in bins if b["n"]],
        "by_kind": by_kind,
    }
