"""Experiment grid for gta6-trophy-rarity-forecast-01 v2 candidates.

NOT part of the entry: runs variant copies of main.py (in /tmp) against
four validations and prints one line per variant:
  Test A (flat history -> live 14:48Z rates), Test B (history+04:30Z
  refresh -> live rates), practice mean (p00/p01 vs reconstructed
  answer), synthetic-final mean (slow/base/fast).
Variants edit main.py source by string replacement; each variant is
run as a subprocess on real case files so the numbers are end-to-end.
Run: python3 tests/exp_grid.py   (from the repo root)
"""
import copy
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = Path.home() / "workspace/make-fun-bounty/kits/gta6-trophy-rarity-forecast-01-kit-fresh"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import main as entry  # noqa: E402
from backtest import simulate_launch, pinball, baseline_forecast  # noqa: E402

LIVE = json.loads((ROOT / "tests/live_20261119_1448z.json").read_text())
ACTUAL_LIVE = {int(k): v for k, v in LIVE["rates"].items()}
ANSWER = json.loads((ROOT / "tests/practice_answer.json").read_text())
ACTUAL_PRAC = {int(k): v for k, v in ANSWER["rates"].items()}


def build_cases():
    p00 = json.loads((KIT / "input/p00.in").read_text())
    p01 = json.loads((KIT / "input/p01.in").read_text())
    refresh_rates = [ANSWER["rates"][str(t["id"])] for t in p00["trophies"]]
    case_a = dict(p00)
    case_a.update({"id": "t19h", "at": "2026-11-19T03:00:00Z",
                   "horizon_minutes": 708, "target": "2026-11-19T14:48:00Z"})
    case_b = dict(p00)
    case_b.update({"id": "t19h_b", "at": "2026-11-19T04:30:00Z",
                   "horizon_minutes": 618, "target": "2026-11-19T14:48:00Z"})
    case_b["history"] = list(p00["history"]) + [{
        "slot": "2026-11-19T04:30:00Z", "at": "2026-11-19T04:30:00Z",
        "players_tracked": 3868, "trophies_earned": 4276,
        "rates": refresh_rates}]
    syn = {}
    for speed, label in ((0.7, "slow"), (1.0, "base"), (1.4, "fast")):
        case, truth = simulate_launch(speed=speed)
        syn[label] = (case, truth)
    return {"p00": (p00, ACTUAL_PRAC), "p01": (p01, ACTUAL_PRAC),
            "tA": (case_a, ACTUAL_LIVE), "tB": (case_b, ACTUAL_LIVE),
            "syn-slow": syn["slow"], "syn-base": syn["base"],
            "syn-fast": syn["fast"]}


def run_variant(src_text, cases):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "main.py").write_text(src_text)
        (td / "in").mkdir()
        (td / "out").mkdir()
        for cid, (case, _) in cases.items():
            (td / "in" / f"{cid}.in").write_text(json.dumps(case))
        env = dict(__import__("os").environ,
                   MAKEFUN_INPUT_DIR=str(td / "in"),
                   MAKEFUN_OUTPUT_DIR=str(td / "out"))
        r = subprocess.run([sys.executable, str(td / "main.py")],
                           capture_output=True, text=True, env=env)
        if r.returncode != 0:
            return None, r.stderr[-500:]
        out = {}
        for cid, (case, actual) in cases.items():
            fc = json.loads((td / "out" / f"{cid}.out").read_text())
            loss, _, _ = pinball(fc, actual)
            out[cid] = loss
        return out, ""


SRC = (ROOT / "main.py").read_text()

# --- variant definitions: (name, list of (old, new) replacements) ---
FUNNEL_OLD = "            medians[k] = 0.5 * medians[k] + 0.5 * funnel_logit"
SPREAD_OLD = "            spreads[k] = math.sqrt(spreads[k] ** 2 + 0.25 ** 2)"
REFRESH_OLD = "    refresh_factor = flatness * (1.0 - min(1.0, horizon_h / 24.0))"

VARIANTS = {
    "V0 baseline": [],
    # F1: funnel weight grows with horizon, saturating at 0.5 by 2 days.
    "V1 funnel-h2d": [
        (FUNNEL_OLD, "            fw = 0.5 * min(1.0, horizon_d / 2.0)\n"
                     "            medians[k] = (1.0 - fw) * medians[k] + fw * funnel_logit"),
        (SPREAD_OLD, "            spreads[k] = math.sqrt(spreads[k] ** 2 + (0.25 * fw / 0.5) ** 2)"),
    ],
    # F1b: funnel weight keeps rising past 0.5: 0.15 at <1d -> 0.60 at 4.5d.
    "V1b funnel-rising": [
        (FUNNEL_OLD, "            fw = min(0.75, 0.15 + 0.10 * horizon_d)\n"
                     "            medians[k] = (1.0 - fw) * medians[k] + fw * funnel_logit"),
        (SPREAD_OLD, "            spreads[k] = math.sqrt(spreads[k] ** 2 + (0.25 * fw / 0.5) ** 2)"),
    ],
    # F3: refresh allowance scales with log-time extrapolation distance,
    # calibrated so the 90-min practice allowance is unchanged
    # (extrap_practice = ln(5.5)-ln(3.5) = 0.452; current factor 0.569).
    "V2 refresh-logtime": [
        (REFRESH_OLD, "    refresh_factor = flatness * min(1.5, 1.259 * max(0.0, x_target - x_last))"),
    ],
    "V3 V1+V2": [
        (FUNNEL_OLD, "            fw = 0.5 * min(1.0, horizon_d / 2.0)\n"
                     "            medians[k] = (1.0 - fw) * medians[k] + fw * funnel_logit"),
        (SPREAD_OLD, "            spreads[k] = math.sqrt(spreads[k] ** 2 + (0.25 * fw / 0.5) ** 2)"),
        (REFRESH_OLD, "    refresh_factor = flatness * min(1.5, 1.259 * max(0.0, x_target - x_last))"),
    ],
    "V4 V1b+V2": [
        (FUNNEL_OLD, "            fw = min(0.75, 0.15 + 0.10 * horizon_d)\n"
                     "            medians[k] = (1.0 - fw) * medians[k] + fw * funnel_logit"),
        (SPREAD_OLD, "            spreads[k] = math.sqrt(spreads[k] ** 2 + (0.25 * fw / 0.5) ** 2)"),
        (REFRESH_OLD, "    refresh_factor = flatness * min(1.5, 1.259 * max(0.0, x_target - x_last))"),
    ],
    # F4: cap the players-growth slope at 2.0 instead of 3.0 (early
    # launch transient slope was ~4.0 in log-time; real later growth ~1.1).
    "V5 ncap-2.0": [
        ("        b_n = min(max(b_n, 0.0), 3.0)", "        b_n = min(max(b_n, 0.0), 2.0)"),
        ("                    b_nr = min(max(fit_nr[1], 0.0), 3.0) if fit_nr else b_n",
         "                    b_nr = min(max(fit_nr[1], 0.0), 2.0) if fit_nr else b_n"),
    ],
    "V6 V1b+V2+ncap": [
        (FUNNEL_OLD, "            fw = min(0.75, 0.15 + 0.10 * horizon_d)\n"
                     "            medians[k] = (1.0 - fw) * medians[k] + fw * funnel_logit"),
        (SPREAD_OLD, "            spreads[k] = math.sqrt(spreads[k] ** 2 + (0.25 * fw / 0.5) ** 2)"),
        (REFRESH_OLD, "    refresh_factor = flatness * min(1.5, 1.259 * max(0.0, x_target - x_last))"),
        ("        b_n = min(max(b_n, 0.0), 3.0)", "        b_n = min(max(b_n, 0.0), 2.0)"),
        ("                    b_nr = min(max(fit_nr[1], 0.0), 3.0) if fit_nr else b_n",
         "                    b_nr = min(max(fit_nr[1], 0.0), 2.0) if fit_nr else b_n"),
    ],
}


def main():
    cases = build_cases()
    print(f"{'variant':<18} {'tA19h':>7} {'tB19h':>7} {'prac':>7} "
          f"{'synS':>7} {'synB':>7} {'synF':>7} {'synMean':>8}")
    for name, reps in VARIANTS.items():
        src = SRC
        for old, new in reps:
            assert old in src, f"{name}: pattern not found: {old[:60]}"
            src = src.replace(old, new, 1)
        out, err = run_variant(src, cases)
        if out is None:
            print(f"{name:<18} FAILED: {err}")
            continue
        prac = (out["p00"] + out["p01"]) / 2
        synm = (out["syn-slow"] + out["syn-base"] + out["syn-fast"]) / 3
        print(f"{name:<18} {out['tA']:7.4f} {out['tB']:7.4f} {prac:7.4f} "
              f"{out['syn-slow']:7.4f} {out['syn-base']:7.4f} {out['syn-fast']:7.4f} {synm:8.4f}")


if __name__ == "__main__":
    main()
