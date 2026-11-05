"""Does model-driven scheduler tuning help? Test it on simulated learners.

Each scenario is a learner whose memory follows hidden "true" parameters that
differ from the defaults in a known way. We simulate 120 days of reviews
scheduled by the default parameters, let the model tune on the first 80 days,
and score both parameter sets on the last 40 days, which the tuner never saw.
Metric: log loss of predicted recall vs actual outcome (lower is better), plus
the calibration gap (mean predicted minus observed recall).
"""
import json
import random

from . import engine
from . import scheduler as sch

DAY = 86400.0
D = sch.DEFAULT_PARAMS


def _p(**over):
    p = json.loads(json.dumps(D))
    for k, v in over.items():
        if isinstance(v, dict):
            p[k].update(v)
        else:
            p[k] = v
    return sch.clamp_params(p)


SCENARIOS = {
    "matches-defaults": ("Learner's memory matches the default model (control: tuning should not hurt).", _p()),
    "durable": ("Remembers much longer than the defaults assume.",
                _p(growth=3.0, init_stability={"again": 0.6, "hard": 2.5, "good": 7.0, "easy": 15.0})),
    "fragile": ("Forgets faster than the defaults assume.",
                _p(growth=0.9, lapse_factor=0.15, init_stability={"again": 0.2, "hard": 0.5, "good": 1.2, "easy": 3.0})),
    "numbers-fade": ("Concepts stick but numbers fade fast.",
                     _p(growth=2.2, kind_multiplier={"number": 0.3, "fact": 0.7})),
}


def simulate(true_p, days=120, n_cards=160, new_per_day=8, seed=7):
    rng = random.Random(seed)
    kinds = {i: rng.choice(sch.KINDS) for i in range(n_cards)}
    sched_p = sch.clamp_params({})
    truth, sched, last, due, log = {}, {}, {}, {}, []
    introduced = 0
    for day in range(days):
        t0 = day * DAY + 9 * 3600
        for _ in range(new_per_day):
            if introduced < n_cards:
                due[introduced] = t0
                introduced += 1
        today = sorted(c for c, d in due.items() if d <= t0 + DAY - 1)
        for k, cid in enumerate(today):
            ts = t0 + k * 30
            elapsed = (ts - last[cid]) / DAY if cid in last else 0.0
            if cid in truth:
                ok = rng.random() < sch.recall_prob(truth[cid]["stability"], elapsed)
                grade = "good" if ok else "again"
            else:
                grade = rng.choice(["good", "good", "hard", "again"])
            truth[cid] = sch.update(truth.get(cid), grade, elapsed, kinds[cid], true_p)
            sched[cid] = sch.update(sched.get(cid), grade, elapsed, kinds[cid], sched_p)
            last[cid] = ts
            due[cid] = ts + sch.next_due_days(sched[cid], grade, sched_p) * DAY
            log.append({"card_id": cid, "ts": ts, "grade": grade})
    return log, kinds


def _gap(r):
    if not r["n"]:
        return None
    pred = sum(b["pred"] * b["n"] for b in r["bins"]) / r["n"]
    obs = sum(b["obs"] * b["n"] for b in r["bins"]) / r["n"]
    return round(pred - obs, 3)


def run(names=None, out=None):
    results = []
    for name in names or SCENARIOS:
        desc, true_p = SCENARIOS[name]
        log, kinds = simulate(true_p)
        split = 80 * DAY
        train = [r for r in log if r["ts"] < split]
        cards = {cid: {"question": f"[simulated {k} card]"} for cid, k in kinds.items()}
        base = sch.clamp_params({})
        prop = engine.propose_params(train, kinds, cards, base)
        tuned = prop["params"] if prop["improves"] else base
        h_base = sch.replay(log, kinds, base, score_from=split)
        h_tuned = sch.replay(log, kinds, tuned, score_from=split)
        h_true = sch.replay(log, kinds, true_p, score_from=split)
        row = {"scenario": name, "description": desc, "train_reviews": len(train),
               "holdout_reviews": h_base["n"], "accepted": prop["improves"],
               "holdout_log_loss": {"default": h_base["log_loss"], "tuned": h_tuned["log_loss"], "oracle": h_true["log_loss"]},
               "calibration_gap": {"default": _gap(h_base), "tuned": _gap(h_tuned)},
               "rationale": prop["rationale"]}
        results.append(row)
        print(f"\n{name}: {desc}")
        print(f"  train {len(train)} reviews, holdout {h_base['n']}; proposal {'accepted' if prop['improves'] else 'rejected'}")
        print(f"  holdout log loss  default {h_base['log_loss']}  tuned {h_tuned['log_loss']}  oracle {h_true['log_loss']}")
        print(f"  calibration gap   default {_gap(h_base)}  tuned {_gap(h_tuned)}")
        print(f"  tuner: {prop['rationale']}")
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
    return results
