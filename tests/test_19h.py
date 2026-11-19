"""19-hour real-data out-of-sample test for gta6-trophy-rarity-forecast-01.

The model was built when the only data was launch night (flat page +
one refresh).  Now ~15 h of real progression exist on the public
Exophase page (crawled 2026-11-19 14:48 UTC, saved in
tests/live_20261119_1448z.json: N ~= 12,000, up from 3,868 at 04:30Z).

Test A (the specified test): history = the practice-case history
(flat launch-night page, kit input/p00.in, checkpoint 03:00Z),
target = now (14:48Z, 708 min horizon).  Scores the CURRENT program
against the actual current rates with the exact pinball-loss formula
from RULES.txt.  With a flat history there is no trend signal, so
this tests the prior interpolation + refresh allowance + spread
against REAL early progression.

Test B (diagnostic, trend machinery): history = practice history +
the 04:30Z refresh read (N=3,868, rates from tests/practice_answer.json),
checkpoint 04:30Z, target = now (618 min horizon).  Two distinct
reads, so the earners/players trend fits engage -- the machinery
that decides the FINAL case.  Tests whether the split extrapolates
real dilution + real progression, or misreads it.

Both tests also score the house persistence baseline for context.
Run: python3 tests/test_19h.py   (from the repo root)
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = Path.home() / "workspace/make-fun-bounty/kits/gta6-trophy-rarity-forecast-01-kit-fresh"
sys.path.insert(0, str(ROOT))
import main as entry  # noqa: E402

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(p):
    p = clamp(p)
    return math.log(p / (100 - p))


def pinball(forecast, actual):
    tot, n, miss, cov = 0.0, 0, 0.0, 0
    by_id = {r["id"]: r["q"] for r in forecast["trophies"]}
    for tid, y in actual.items():
        q = by_id[tid]
        miss += abs(y - q[2])
        cov += 1 if q[0] <= y <= q[4] else 0
        for tau, qq in zip(LEVELS, q):
            d = logit(y) - logit(qq)
            tot += 2 * max(tau * d, (tau - 1) * d)
            n += 1
    return tot / n, miss / len(actual), cov / len(actual)


def pinball_by_cat(forecast, actual, case):
    from collections import defaultdict
    tot = defaultdict(lambda: [0.0, 0])
    mission_seen = 0
    cats = {}
    for tr in case["trophies"]:
        cat, _ = entry.classify(tr, mission_seen)
        if cat == "story_mission":
            mission_seen += 1
        cats[tr["id"]] = cat
    by_id = {r["id"]: r["q"] for r in forecast["trophies"]}
    for tid, y in actual.items():
        for tau, qq in zip(LEVELS, by_id[tid]):
            d = logit(y) - logit(qq)
            tot[cats[tid]][0] += 2 * max(tau * d, (tau - 1) * d)
            tot[cats[tid]][1] += 1
    return {c: v[0] / v[1] for c, v in sorted(tot.items())}


def baseline_forecast(case):
    last = case["history"][-1]
    d = case["horizon_minutes"] / 1440
    spread = 0.5 * math.sqrt(d)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        lg = logit(last["rates"][i] if last["rates"][i] is not None else 0)
        rows.append({"id": tr["id"], "q": [
            round(100 / (1 + math.exp(-(lg + z * spread))), 4) for z in Z]})
    return {"trophies": rows}


def main():
    live = json.loads((ROOT / "tests/live_20261119_1448z.json").read_text())
    actual = {int(k): v for k, v in live["rates"].items()}
    p00 = json.loads((KIT / "input/p00.in").read_text())
    answer = json.loads((ROOT / "tests/practice_answer.json").read_text())
    refresh_rates = [answer["rates"][str(t["id"])] for t in p00["trophies"]]

    # --- Test A: flat launch-night history, target now ---
    case_a = dict(p00)
    case_a["id"] = "t19h"
    case_a["at"] = "2026-11-19T03:00:00Z"
    case_a["horizon_minutes"] = 708
    case_a["target"] = "2026-11-19T14:48:00Z"
    fc_a = entry.forecast_case(case_a, "t19h")
    loss, miss, cov = pinball(fc_a, actual)
    bl, bm, bc = pinball(baseline_forecast(case_a), actual)
    print("== Test A: flat history (p00) -> now 14:48Z, 708 min ==")
    print(f" ours loss {loss:.4f} miss {miss:.2f}pp cov {cov:.2f} | "
          f"persistence {bl:.4f} miss {bm:.2f}pp cov {bc:.2f}")
    for cat, l in pinball_by_cat(fc_a, actual, case_a).items():
        print(f"   {cat}: {l:.4f}")
    by_id = {r["id"]: r["q"] for r in fc_a["trophies"]}
    print("   worst trophies (id: actual vs q50, loss):")
    worst = []
    for tid, y in actual.items():
        q = by_id[tid]
        l = sum(2 * max(t * (logit(y) - logit(qq)), (t - 1) * (logit(y) - logit(qq)))
                for t, qq in zip(LEVELS, q)) / 5
        worst.append((l, tid, y, q[2]))
    for l, tid, y, q50 in sorted(worst, reverse=True)[:10]:
        print(f"    id {tid}: actual {y}% q50 {q50}% loss {l:.3f}")
    print("   story chain (id: actual vs q50): " +
          ", ".join(f"{i}:{actual[i]} vs {by_id[i][2]}" for i in (3, 4, 5, 6, 7)))

    # --- Test B: history + 04:30Z refresh read, target now ---
    case_b = dict(p00)
    case_b["id"] = "t19h_b"
    case_b["at"] = "2026-11-19T04:30:00Z"
    case_b["horizon_minutes"] = 618
    case_b["target"] = "2026-11-19T14:48:00Z"
    hist_b = list(p00["history"]) + [{
        "slot": "2026-11-19T04:30:00Z", "at": "2026-11-19T04:30:00Z",
        "players_tracked": 3868, "trophies_earned": 4276,
        "rates": refresh_rates}]
    case_b["history"] = hist_b
    fc_b = entry.forecast_case(case_b, "t19h_b")
    loss, miss, cov = pinball(fc_b, actual)
    bl, bm, bc = pinball(baseline_forecast(case_b), actual)
    print("== Test B: history + 04:30Z refresh -> now 14:48Z, 618 min ==")
    print(f" ours loss {loss:.4f} miss {miss:.2f}pp cov {cov:.2f} | "
          f"persistence {bl:.4f} miss {bm:.2f}pp cov {bc:.2f}")
    for cat, l in pinball_by_cat(fc_b, actual, case_b).items():
        print(f"   {cat}: {l:.4f}")
    by_id = {r["id"]: r["q"] for r in fc_b["trophies"]}
    print("   story chain (id: actual vs q50): " +
          ", ".join(f"{i}:{actual[i]} vs {by_id[i][2]}" for i in (3, 4, 5, 6, 7)))
    print(f"   Beat Cal id20: actual {actual[20]} vs q50 {by_id[20][2]}; "
          f"arcade id21: {actual[21]} vs {by_id[21][2]}")


if __name__ == "__main__":
    main()
