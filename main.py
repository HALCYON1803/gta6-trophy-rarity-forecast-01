"""gta6-trophy-rarity-forecast-01 entry: week-one Exophase trophy rates.

Model (fit fresh at runtime from the case file it is given; nothing in it
hardcodes a GTA VI rate — the only GTA-VI-specific numbers are the
category priors in CATEGORY, documented in METHODOLOGY.md, which act as
shrinkage anchors, never as the forecast itself):

  The rate of a trophy is earners / players-tracked.  Both move, in
  opposite directions, and the history shows it: on launch night the
  page was frozen at 638 players, then one refresh took it to 3,868 and
  every rate but the one at 100% fell, because 3,230 new players had
  earned almost nothing yet.  A persistence forecast therefore inherits
  the last refresh's dilution, and a logit line fitted over the whole
  history inherits the early dilution phase too.  This program instead
  separates the two moving parts and fits the *recent* regime:

  * players N(t): log N against ln(1 + hours since release), fitted on
    the distinct reads (identical reads, of which a frozen page produces
    many, are collapsed first so they cannot swamp the fit);
  * earners E_i(t) = rate_i * N / 100: log E_i against ln(1 + hours),
    fitted on the last 24 hours of distinct reads, where the current
    progression regime lives;
  * the rate's own logit against ln(1 + hours), fitted on the same
    recent window, as a cross-check.

  The median is a blend of the trend extrapolation, the earners-over-
  players extrapolation, and a category prior anchored at day 7 (168 h
  after release).  The blend weight on the data grows with the number of
  distinct reads and with the recency of the trend, so a flat launch-
  night history (one distinct read, as in the practice cases) forecasts
  mostly from the prior and a wide spread, while a two-day history
  (the final case) forecasts mostly from its own trend.  When the
  history is flat, the median is also shifted down by a refresh
  allowance: Exophase refreshes hours apart, and a refresh dilutes
  every rate by roughly the jump in players tracked.  The allowance
  scales with the log-time distance the forecast spans (player growth
  is roughly linear in log time early on, so the dilution over a span
  grows with it) and is capped; it decays as distinct reads
  accumulate, so it is ~0 for the final case, whose history prices
  dilution through its own trend.  v1 decayed the allowance with
  horizon, which is backwards — dilution accumulates while players
  keep joining — and under-shifted everywhere it applied: practice
  0.42, 19-hour real-data test 1.68 (see METHODOLOGY.md).

  Spread: a base that grows with horizon, plus the trend fit's residual
  SD scaled by how far the target is extrapolated, plus a refresh term
  when the history is flat.  The top quantile uses z = 1.9, not 1.645:
  progression surprises are right-skewed (a trophy's earners can surge
  as players reach it; they can only fall by dilution, which is
  bounded by how fast players join).  Quantiles are enforced monotone
  by construction, and the story chain (Chapter 1 >= Chapter 2 >=
  Chapter 3 >= Chapter 4 >= final mission) and the platinum (<= every
  other trophy, <= the 100%-completion trophy) are enforced per level,
  because a later story trophy cannot be held by more players than an
  earlier one, and the platinum cannot be held by anyone who has not
  earned every other trophy.

  "Complete Gone Fishing" is special: it reads 100.00% in every read
  on record (638 players, then 3,868), so its quantiles sit at the top
  with only a small lower tail.

Standard library only, deterministic (reads sorted, no randomness, no
wall-clock use — every time comes from the case file), runs in well
under a second per case.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.9]  # top z widened: right-skewed progression

# Blend controls (tuned on the practice cases and the synthetic-final
# backtest in tests/backtest.py; see METHODOLOGY.md): W_CAP is the most
# weight the data extrapolation ever gets against the day-7 category
# prior, and SHRINK further shrinks the data extrapolation's deviation
# from that prior, because both extrapolations (logit-linear and
# earners-over-players power law) keep rising where the truth
# saturates as cohorts finish a trophy.
W_CAP = 0.65
SHRINK = 0.75
# Extrapolation damping: the recent-regime slope is fitted over the
# last 24 h; only this fraction of the log-time distance to the target
# is extrapolated at full strength.  A first, saturating generator
# (cohorts earning within hours of joining) favoured 0.3-0.5, but
# that generator is inconsistent with the observed launch night
# (chapter 1 was still 0.49% at 12 h); the generator calibrated to
# that curve favours 1.0, and the two roughly cancel, so the recent
# slope is extrapolated unchanged.
EXTRAP_DAMP = 1.0

RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)  # game release_date
WEEK_HOURS = 168.0  # day-7 target is 168 h after release

# --- category priors: (day-7 rate in percent, extra spread, refresh shift) ---
# Day-7 rates are shrinkage anchors grounded in METHODOLOGY.md (Phantom
# Blade Zero's Steam curve at day 11 for a 30-40 h game, God of War
# Ragnarok 24.8% story completion at two weeks, the ~50 h GTA VI
# campaign and ~200 h platinum from PowerPyx/Polygon, and the launch-
# night counts: only ~9 of 638 tracked players had finished the story
# before launch, and the 3,230 players added by the first refresh had
# earned almost nothing).  Grind/100%/platinum trophies need far more
# hours than a week allows, so their day-7 rates are set by dilution
# of the handful of pre-launch earners, not by progression.
PRIOR = {
    "first_mission": (100.0, 0.05, 0.0),
    "story_ch1": (50.0, 0.10, 0.7),
    "story_ch2": (35.0, 0.12, 1.0),
    "story_ch3": (25.0, 0.15, 1.2),
    "story_ch4": (18.0, 0.18, 1.2),
    "story_final": (13.0, 0.20, 1.2),
    "story_mission": (20.0, 0.20, 1.2),
    "stranger_final": (8.0, 0.25, 1.4),
    "activity_try": (18.0, 0.20, 0.9),
    "easy_oneoff": (8.0, 0.25, 1.4),
    "grind_small": (4.0, 0.30, 1.5),
    "grind_big": (0.8, 0.35, 1.6),
    "money": (4.0, 0.30, 1.5),
    "completion": (0.03, 0.35, 1.6),
    "platinum": (0.03, 0.35, 1.6),
}

# Story-mission priors decline along the list order (the missions are
# ordered roughly by where they fall in the story).
STORY_MISSION_PRIORS = [26.0, 22.0, 20.0, 18.0, 16.0, 14.0]


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    lg = min(12.0, max(-12.0, lg))
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_ts(s):
    if not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def hours_since_release(ts):
    if ts is None:
        return 0.0
    return max(0.0, (ts - RELEASE).total_seconds() / 3600.0)


def ols(xs, ys):
    """OLS y ~ a + b*x; returns (a, b, residual sd). None if degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n
    return a, b, math.sqrt(max(0.0, var))


def classify(trophy, story_mission_index):
    """Return (category, day-7 prior rate) from name + description."""
    text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    tier = trophy.get("tier", "")
    if tier == "platinum":
        return "platinum", PRIOR["platinum"][0]
    if "100% completion" in text:
        return "completion", PRIOR["completion"][0]
    if "gone fishing" in text:
        return "first_mission", 100.0
    m = re.search(r"complete chapter (\d)", text)
    if m:
        cat = {1: "story_ch1", 2: "story_ch2", 3: "story_ch3", 4: "story_ch4"}[int(m.group(1))]
        return cat, PRIOR[cat][0]
    if "complete the final mission" in text and "for " not in text:
        return "story_final", PRIOR["story_final"][0]
    if "complete the final mission for" in text:
        return "stranger_final", PRIOR["stranger_final"][0]
    if "try each type of activity" in text:
        return "activity_try", PRIOR["activity_try"][0]
    if "beat cal in a fishing challenge" in text:
        return "activity_try", 15.0
    if "complete an arcade game" in text:
        return "activity_try", 12.0
    if "gold medal in all shooting range" in text:
        return "grind_big", 1.5
    if "meditate at every spot" in text:
        return "grind_big", 1.0
    if "8 mission strands for strangers" in text:
        return "grind_big", 1.5
    if "3 collectible card sets" in text:
        return "grind_big", 0.8
    if "250 stunts" in text:
        return "grind_big", 0.05
    if "waink it at 100 locations" in text:
        return "grind_big", 0.5
    if "complete 50 challenges" in text:
        return "grind_small", 2.0
    if "win 20 different races" in text:
        return "grind_small", 3.0
    if "discover 100 vehicles" in text:
        return "grind_small", 8.0
    if "rob 5 bank vaults" in text:
        return "grind_small", 2.5
    if "explore all of leonida" in text:
        return "grind_big", 2.0
    if "win a top fun drone race at every location" in text:
        return "grind_big", 2.0
    if "laundering marked bills" in text:
        return "money", 2.0
    if "spend $100,000" in text:
        return "money", 6.0
    if "purchase all garage properties" in text:
        return "money", 1.5
    if "register 10 vehicles" in text:
        return "grind_small", 5.0
    if text.startswith("complete ") and "complete" in text:
        # A named story mission ("Complete Breakfast Run." etc.).
        prior = STORY_MISSION_PRIORS[min(story_mission_index, len(STORY_MISSION_PRIORS) - 1)]
        return "story_mission", prior
    return "easy_oneoff", PRIOR["easy_oneoff"][0]


def distinct_reads(case):
    """Collapse identical reads (a frozen page repeats) keeping first time."""
    seen = {}
    for read in case.get("history") or []:
        at = parse_ts(read.get("at"))
        rates = tuple(read.get("rates") or [])
        key = (read.get("players_tracked"), rates)
        if key not in seen:
            seen[key] = (at, read)
    out = sorted(seen.values(), key=lambda r: (r[0] or RELEASE))
    return out


def forecast_case(case, name):
    trophies = case.get("trophies") or []
    if not trophies:
        raise ValueError(f"{name} has no trophies")
    target = parse_ts(case.get("target"))
    horizon_h = (case.get("horizon_minutes") or 0) / 60.0
    horizon_d = horizon_h / 24.0
    t_target = hours_since_release(target) if target else WEEK_HOURS
    x_target = math.log1p(t_target)

    reads = distinct_reads(case)
    n_distinct = len(reads)
    if n_distinct == 0:
        raise ValueError(f"{name} has no history")
    last_at, last_read = reads[-1]
    t_last = hours_since_release(last_at)
    x_last = math.log1p(t_last)
    last_rates = last_read.get("rates") or []
    last_n = last_read.get("players_tracked") or 0

    # Recent window for trend fits: the last 24 h of distinct reads.
    # With a single distinct read there is no trend to fit.
    recent = [(at, r) for (at, r) in reads
              if hours_since_release(at) >= t_last - 24.0] or reads
    if len(recent) < 2:
        recent = reads

    xs_recent = [math.log1p(hours_since_release(at)) for at, _ in recent]
    # Damped extrapolation point: only part of the log-time distance
    # from the last read to the target is run at the recent slope.
    x_pred = xs_recent[-1] + EXTRAP_DAMP * (x_target - xs_recent[-1]) if xs_recent else x_target
    xs_all = [math.log1p(hours_since_release(at)) for at, _ in reads]

    # Players-tracked growth, fitted on all distinct reads.
    logn = [math.log(max(1.0, float(r.get("players_tracked") or 1)))
            for _, r in reads]
    fit_n = ols(xs_all, logn)
    if fit_n is not None:
        a_n, b_n, _ = fit_n
        b_n = min(max(b_n, 0.0), 3.0)
        logn_target = a_n + b_n * x_pred
    else:
        logn_target = math.log(max(1.0, float(last_n or 1)))
        b_n = 0.0

    # Story-mission index for declining priors (count in id order).
    mission_seen = 0
    categories = []
    for tr in trophies:
        cat, prior = classify(tr, mission_seen)
        if cat == "story_mission":
            mission_seen += 1
        categories.append((cat, prior))

    # Refresh allowance: with a flat history a refresh in the horizon
    # dilutes every rate.  It decays as distinct reads accumulate (a
    # history with a real trend prices dilution itself), but it must
    # NOT decay with horizon: dilution accumulates while players keep
    # joining, so a longer horizon spans more refreshes, not fewer.
    # v1 decayed it as (1 - horizon/24h) and under-shifted badly: on
    # the 19-hour real-data test (flat launch-night history -> live
    # 14:48Z page, N 638 -> ~12,000) v1 scored 1.68, and even on the
    # 90-minute practice it scored 0.42 with a shift of only ~0.75
    # logit against a ~1.6 logit refresh.  The allowance now scales
    # with the log-time distance the forecast spans (x_target -
    # x_last: player growth is roughly linear in log time early on,
    # so the dilution over a span grows with it), capped at 1.5 x
    # flatness.  On the flat launch-night history x_last is 0 (the
    # first read predates release), so the cap binds and the allowance
    # is flatness x 1.5 at every horizon; on a history with real reads
    # it grows with the spanned distance.  With many distinct reads
    # (the final case) flatness is ~0 and the allowance is ~0, so the
    # final forecast is unchanged by this change.
    flatness = math.exp(-n_distinct / 2.0)
    refresh_factor = flatness * min(1.5, 1.259 * max(0.0, x_target - x_last))

    medians = []
    spreads = []
    for i, tr in enumerate(trophies):
        cat, prior7 = categories[i]
        _, spread_add, refresh_scale = PRIOR[cat]
        rates_i = []
        for _, r in recent:
            v = (r.get("rates") or [None] * len(trophies))[i]
            if isinstance(v, (int, float)):
                rates_i.append(v)
        last_rate = None
        for _, r in reversed(reads):
            v = (r.get("rates") or [None] * len(trophies))[i]
            if isinstance(v, (int, float)):
                last_rate = v
                break
        if last_rate is None:
            last_rate = 0.0

        # --- data extrapolations (recent regime) ---
        pred_data = None
        resid = 0.0
        if len(rates_i) >= 2:
            ys_logit = [logit(v) for v in rates_i]
            fit_r = ols(xs_recent[:len(ys_logit)], ys_logit)
            # Earners and players over the same recent window.
            ys_loge = []
            ns_recent = []
            ok = True
            for (at, r), v in zip(recent, rates_i):
                n = r.get("players_tracked") or 0
                if n <= 0:
                    ok = False
                    break
                ns_recent.append(math.log(float(n)))
                ys_loge.append(math.log(max(0.25, v * n / 100.0)))
            fit_e = ols(xs_recent[:len(ys_loge)], ys_loge) if ok else None
            if fit_r is not None:
                a_r, b_r, resid = fit_r
                b_r = min(max(b_r, -2.0), 3.0)
                pred_trend = a_r + b_r * x_pred
                pred_en = None
                if fit_e is not None:
                    a_e, b_e, _ = fit_e
                    b_e = min(max(b_e, 0.0), 4.0)
                    # Recent players growth for the earners/players split.
                    fit_nr = ols(xs_recent[:len(ns_recent)], ns_recent)
                    b_nr = min(max(fit_nr[1], 0.0), 3.0) if fit_nr else b_n
                    a_nr = fit_nr[0] if fit_nr else (ns_recent[-1] - b_nr * xs_recent[-1])
                    loge_t = a_e + b_e * x_pred
                    logn_t = a_nr + b_nr * x_pred
                    pred_en = logit(100.0 * math.exp(loge_t - logn_t))
                if pred_en is not None:
                    pred_data = 0.55 * pred_trend + 0.45 * pred_en
                else:
                    pred_data = pred_trend
                # Saturation cap: the trend may not run far past the
                # category prior (rates saturate; logit lines do not).
                cap = logit(prior7) + 0.3 if prior7 < 99.0 else 6.0
                pred_data = min(pred_data, cap)

        # --- category prior at the target time ---
        # Interpolate in logit space from the last rate to the day-7
        # prior, by how much of the week remains between last read and
        # target (practice: ~0, so the prior is the last rate; final:
        # 1, so it is the full day-7 prior).
        span = max(1.0, WEEK_HOURS - t_last)
        g = min(1.0, max(0.0, (t_target - t_last) / span))
        prior_logit = logit(last_rate) + (logit(prior7) - logit(last_rate)) * g

        # --- blend ---
        if pred_data is None:
            w_data = 0.0
        else:
            w_data = n_distinct / (n_distinct + 4.0)
            w_data = min(w_data, W_CAP)
            w_data *= 1.0 / (1.0 + resid)
            if horizon_h < 6.0:
                w_data *= 0.6
        med_logit = prior_logit + w_data * SHRINK * (pred_data - prior_logit) \
            if pred_data is not None else prior_logit
        # Refresh allowance on a flat history.
        med_logit -= refresh_scale * refresh_factor * 1.1

        # --- spread ---
        extrap = max(0.0, x_target - x_last)
        base = 0.55 + 0.35 * min(1.0, horizon_d / 4.5) + spread_add
        s = math.sqrt(base * base + (resid * (0.5 + extrap)) ** 2
                      + (0.9 * refresh_factor) ** 2)
        s = min(max(s, 0.50), 1.35)
        medians.append(med_logit)
        spreads.append(s)

    by_id = {tr["id"]: k for k, tr in enumerate(trophies)}
    chain = [by_id[i] for i in (3, 4, 5, 6, 7) if i in by_id]

    # --- story-chain funnel: later chapters share one progression
    # process with chapter 1, so their medians are blended toward
    # what chapter 1's median implies through the funnel ratios of
    # the category priors (prior_k / prior_Ch1).  A late chapter's
    # own trend is nearly uninformative at the cut — its earners are
    # still ~0 while chapter 1 is already moving — and extrapolating
    # it alone badly under-forecasts the coming wave (in the synthetic
    # backtest the final mission's own-trend median was 0.7% against
    # a 7.7% truth, while the funnel implied ~15%).  The ratios match
    # the funnel seen in a comparable launch: Phantom Blade Zero's
    # Steam story steps at day 11 ran 94.9 / 84.1 / 55.3 / 30.0 /
    # 24.2 / 21.6% (see METHODOLOGY.md).
    if 3 in by_id:
        m1 = inv_logit(medians[by_id[3]])
        for k in chain[1:]:
            ratio = categories[k][1] / categories[by_id[3]][1]
            funnel_logit = logit(m1 * ratio)
            medians[k] = 0.5 * medians[k] + 0.5 * funnel_logit
            # The funnelled median inherits chapter 1's error as well
            # as its own, so its spread widens.
            spreads[k] = math.sqrt(spreads[k] ** 2 + 0.25 ** 2)

    # --- structural enforcement, per quantile level ---
    n = len(trophies)
    quantiles = []
    for k in range(n):
        row = []
        prev = 0.0
        for z in Z:
            q = inv_logit(medians[k] + z * spreads[k])
            q = min(max(round4(q), 0.01), 100.0)
            q = max(q, prev)
            row.append(q)
            prev = q
        quantiles.append(row)

    # "Complete Gone Fishing" reads 100% in every read on record.
    if 2 in by_id:
        quantiles[by_id[2]] = [99.0, 99.9, 100, 100, 100]

    # Story chain: later chapters cannot exceed earlier ones, per level.
    for level in range(5):
        running = 100.0
        for k in chain:
            running = min(running, quantiles[k][level])
            quantiles[k][level] = running
    # Platinum <= 100%-completion trophy <= 250-stunt trophy, per level;
    # platinum <= every other trophy, per level.
    for level in range(5):
        if 14 in by_id and 15 in by_id:
            quantiles[by_id[14]][level] = min(quantiles[by_id[14]][level],
                                              quantiles[by_id[15]][level])
        if 1 in by_id:
            others = min(quantiles[k][level] for k in range(n) if k != by_id[1])
            quantiles[by_id[1]][level] = min(quantiles[by_id[1]][level], others)
            if 14 in by_id:
                quantiles[by_id[1]][level] = min(quantiles[by_id[1]][level],
                                                  quantiles[by_id[14]][level])
    # Re-enforce monotonicity inside each row after the caps.
    for k in range(n):
        prev = 0.0
        for level in range(5):
            quantiles[k][level] = max(quantiles[k][level], prev)
            prev = quantiles[k][level]

    rows = [{"id": tr["id"], "q": quantiles[k]} for k, tr in enumerate(trophies)]
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
