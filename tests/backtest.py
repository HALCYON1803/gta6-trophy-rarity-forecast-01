"""Local backtests for gta6-trophy-rarity-forecast-01 (not the entrypoint).

1. Practice: scores main.py on p00/p01 against the reconstructed
   practice answer — the Exophase page as crawled 2026-11-19 11:42 UTC,
   which is the 04:30Z refresh read: players tracked still 3,868, and
   the counts implied by its rates sum to exactly 4,276 trophies
   earned, the figure the kit's competition.yaml reports for the
   04:30Z read (and the 05:00Z read was unchanged).  Every rate is
   exactly count/3,868 rounded to 2 dp.  See tests/practice_answer.json.
2. Synthetic final: a simulated launch (cohort-delay model with known
   day-7 truth, refresh-step players growth, dilution then progression)
   cut at a Nov-21-like checkpoint; scores main.py vs the persistence
   baseline and the house curve baseline under slow/fast truth.
3. Determinism + validity: two runs byte-identical, output shape valid.

Run: python3 tests/backtest.py   (from the repo root)
"""
import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
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
    """Mean pinball loss on the logit scale; also typical miss, coverage."""
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


def baseline_forecast(case):
    """House persistence baseline (starter, same numbers)."""
    last = case["history"][-1]
    d = case["horizon_minutes"] / 1440
    spread = 0.5 * math.sqrt(d)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        lg = logit(last["rates"][i] if last["rates"][i] is not None else 0)
        rows.append({"id": tr["id"], "q": [
            round(100 / (1 + math.exp(-(lg + z * spread))), 4) for z in Z]})
    return {"trophies": rows}


def curve_baseline_forecast(case):
    """House curve baseline: per-trophy OLS logit rate on ln(1+h) over
    the last 24 h of reads, read at the target; spread = random walk +
    residual. Falls back to persistence with <3 reads."""
    release = datetime(2026, 11, 19, tzinfo=timezone.utc)
    def hrs(s):
        return max(0.0, (datetime.fromisoformat(s.replace("Z", "+00:00")) - release).total_seconds() / 3600)
    hist = case["history"]
    t_last = hrs(hist[-1]["at"])
    win = [h for h in hist if hrs(h["at"]) >= t_last - 24] or hist
    xs = [math.log1p(hrs(h["at"])) for h in win]
    x_t = math.log1p(hrs(case["target"]))
    d = case["horizon_minutes"] / 1440
    rows = []
    for i, tr in enumerate(case["trophies"]):
        ys = [logit(h["rates"][i]) for h in win if h["rates"][i] is not None]
        xx = xs[:len(ys)]
        if len(ys) >= 3 and len(set(xx)) > 1:
            mx, my = sum(xx) / len(xx), sum(ys) / len(ys)
            sxx = sum((x - mx) ** 2 for x in xx)
            b = sum((x - mx) * (y - my) for x, y in zip(xx, ys)) / sxx if sxx else 0.0
            a = my - b * mx
            resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xx, ys)) / len(ys))
            med, s = a + b * x_t, math.sqrt((0.5 * math.sqrt(d)) ** 2 + resid ** 2)
        else:
            med, s = logit(hist[-1]["rates"][i] or 0), 0.5 * math.sqrt(d)
        rows.append({"id": tr["id"], "q": [
            round(100 / (1 + math.exp(-(med + z * s))), 4) for z in Z]})
    return {"trophies": rows}


def run_entry_on(case_dir, out_dir):
    env = dict(__import__("os").environ,
               MAKEFUN_INPUT_DIR=str(case_dir), MAKEFUN_OUTPUT_DIR=str(out_dir))
    subprocess.run([sys.executable, str(ROOT / "main.py")],
                   check=True, env=env, capture_output=True, text=True)


def practice_test():
    answer = json.loads((ROOT / "tests/practice_answer.json").read_text())
    actual = {int(k): v for k, v in answer["rates"].items()}
    print("== practice (answer = 04:30Z refresh, N=3868, 4276 earned) ==")
    for cid in ("p00", "p01"):
        case = json.loads((KIT / f"input/{cid}.in").read_text())
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "in").mkdir(); (td / "out").mkdir()
            (td / "in" / f"{cid}.in").write_text(json.dumps(case))
            run_entry_on(td / "in", td / "out")
            fc = json.loads((td / "out" / f"{cid}.out").read_text())
        loss, miss, cov = pinball(fc, actual)
        bl, bm, bc = pinball(baseline_forecast(case), actual)
        print(f" {cid}: ours loss {loss:.4f} miss {miss:.2f}pp cov {cov:.2f} | "
              f"baseline loss {bl:.4f} miss {bm:.2f}pp cov {bc:.2f}")
    return True


# ---- synthetic final -------------------------------------------------
def simulate_launch(seed=7, speed=1.0, n_players_day7=300_000):
    """Cohort-delay launch, calibrated to the observed early curve:
    at ~12 h after release the real page read Ch1 0.49%, story
    trophies 0.23%, fishing challenge 1.16% (N=3,868).  Players join
    by a power law; each cohort earns trophy i after a lognormal
    delay (calendar hours).  The eventual share is set so the
    day-7 truth equals the entry's category prior at speed 1.0
    (speed scales the delays: 0.7 slow, 1.4 fast), so the test asks
    whether the entry recovers the day-7 level from a Nov-21 cut,
    not whether it agrees with its own prior by construction."""
    import random
    rng = random.Random(seed)
    release = datetime(2026, 11, 19, tzinfo=timezone.utc)
    trophies = json.loads((KIT / "input/p00.in").read_text())["trophies"]
    cats = {}
    mission_seen = 0
    for tr in trophies:
        cat, prior = entry.classify(tr, mission_seen)
        if cat == "story_mission":
            mission_seen += 1
        cats[tr["id"]] = (cat, prior)
    # Median calendar delay (h) for a cohort to earn the trophy.
    # Ch1 ~70 h: the average tracked player plays a few hours a day
    # and reaches the end of chapter 1 after ~10-15 h of play.
    delay = {"story_ch1": 70, "story_ch2": 110, "story_ch3": 150,
             "story_ch4": 190, "story_final": 230, "story_mission": 150,
             "stranger_final": 200, "activity_try": 60, "easy_oneoff": 50,
             "grind_small": 300, "grind_big": 800, "money": 250,
             "completion": 1500, "platinum": 1500, "first_mission": 1}

    def n_of(t):
        return 638 + (n_players_day7 - 638) * ((1 + t) / 169.0) ** 1.35

    def earners_at(t, cat, eventual_pct):
        med = delay[cat] / speed
        eventual = eventual_pct / 100.0
        total = 0.0
        s = 0.0
        while s < t:
            dn = n_of(min(t, s + 1.0)) - n_of(s)
            age = t - (s + 0.5)
            if age > 0:
                from math import erf, sqrt
                z = (math.log(max(age, 0.1)) - math.log(med)) / 0.7
                cdf = 0.5 * (1 + erf(z / sqrt(2)))
                total += dn * eventual * cdf
            s += 1.0
        return total

    # Eventual shares: day-7 truth == prior at speed 1.
    n7 = n_of(168.0)
    eventual = {}
    for tr in trophies:
        cat, prior = cats[tr["id"]]
        if tr["id"] == 2:
            eventual[tr["id"]] = 100.0
            continue
        f7 = earners_at(168.0, cat, 100.0) / n7
        eventual[tr["id"]] = min(95.0, prior / max(f7, 0.05))
    slots = []
    t = 0.0
    while t <= 59.5:
        slots.append(t)
        t += 6.0
    history = []
    for t in slots:
        n = n_of(t)
        rates = []
        for tr in trophies:
            cat, prior = cats[tr["id"]]
            e = earners_at(t, cat, eventual[tr["id"]])
            rates.append(round(100 * e / n, 2) if tr["id"] != 2 else 100.0)
        history.append({"slot": (release + timedelta(hours=t)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "at": (release + timedelta(hours=t)).isoformat().replace("+00:00", "Z"),
                        "players_tracked": int(n),
                        "trophies_earned": int(sum(r * n / 100 for r in rates)),
                        "rates": rates})
    case = {"schema": "makefun/gta6-trophies-case/v1", "id": "f00",
            "at": "2026-11-21T11:30:00Z", "horizon_minutes": 6510,
            "target": "2026-11-26T00:00:00Z",
            "levels": LEVELS,
            "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                     "release_date": "2026-11-19",
                     "source": "https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/"},
            "trophies": trophies, "history": history, "notes": "synthetic"}
    truth = {}
    for tr in trophies:
        cat, prior = cats[tr["id"]]
        e = earners_at(168.0, cat, eventual[tr["id"]])
        truth[tr["id"]] = 100.0 if tr["id"] == 2 else round(100 * e / n7, 2)
    return case, truth


def synthetic_test():
    print("== synthetic final (Nov-21 cut -> Nov-26 truth) ==")
    for speed, label in ((0.7, "slow progression"), (1.0, "base"), (1.4, "fast progression")):
        case, truth = simulate_launch(speed=speed)
        fc = entry.forecast_case(case, "f00")
        loss, miss, cov = pinball(fc, truth)
        bl, bm, bc = pinball(baseline_forecast(case), truth)
        cl, cm, cc = pinball(curve_baseline_forecast(case), truth)
        print(f" {label}: ours {loss:.4f} (miss {miss:.2f}pp cov {cov:.2f}) | "
              f"persistence {bl:.4f} | curve {cl:.4f}")
        if label == "base":
            for tid in (3, 4, 7, 1, 14, 15, 20):
                q = {r["id"]: r["q"] for r in fc["trophies"]}[tid]
                print(f"   id {tid}: truth {truth[tid]}%  q50 {q[2]}%  band [{q[0]}, {q[4]}]")


def validity_test():
    print("== validity / determinism ==")
    case = json.loads((KIT / "input/p00.in").read_text())
    outs = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "in").mkdir(); (td / "out").mkdir()
            (td / "in" / "p00.in").write_text(json.dumps(case))
            run_entry_on(td / "in", td / "out")
            outs.append((td / "out" / "p00.out").read_bytes())
    assert outs[0] == outs[1], "non-deterministic output"
    fc = json.loads(outs[0])
    assert len(fc["trophies"]) == 55
    assert all(len(r["q"]) == 5 and r["q"] == sorted(r["q"])
               and all(0 <= q <= 100 for q in r["q"]) for r in fc["trophies"])
    assert len(outs[0]) <= 65536
    print(f" ok: 55 rows, monotone, {len(outs[0])} bytes, byte-identical reruns")


if __name__ == "__main__":
    validity_test()
    practice_test()
    synthetic_test()
    # 19-hour real-data test (tests/test_19h.py): flat launch-night
    # history and history+04:30Z refresh, both scored against the
    # live Exophase page crawled 2026-11-19 14:48 UTC.
    sys.path.insert(0, str(ROOT / "tests"))
    from test_19h import main as test_19h_main
    test_19h_main()
