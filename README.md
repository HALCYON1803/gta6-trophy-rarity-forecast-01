# gta6-trophy-rarity-forecast-01

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/HALCYON1803/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/HALCYON1803/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts, for every GTA VI PlayStation 5 trophy, the percent of the
players Exophase tracks who will have earned it at 2026-11-26T00:00Z
— one week after launch — as five quantiles per trophy, scored by
pinball loss on the logit scale.

## The entry

`main.py` (Python 3.13, standard library only, deterministic, no
network) is run once per case: it reads the case file, fits the model
fresh from that case's own history, and writes one forecast file.

The rate of a trophy is earners / players-tracked, and both move in
opposite directions: Exophase refreshes its counts hours apart, and
each refresh dilutes every rate as thousands of new players — who
have earned almost nothing yet — join the denominator. The program
therefore separates the two moving parts (log earners and log players
against log time since release, fitted on the last 24 h of distinct
reads), blends that extrapolation with a category prior anchored at
day 7, and enforces the story chain (Chapter 1 ≥ Chapter 2 ≥ Chapter 3
≥ Chapter 4 ≥ final mission) and the platinum (≤ every other trophy).
With a flat history it expects a refresh and forecasts wide; with a
two-day history it forecasts mostly from its own trend. Full detail:
[METHODOLOGY.md](METHODOLOGY.md).

## Reproduce

```sh
# Practice cases (from the kit's input/)
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
# Scores against the reconstructed practice answer:
python3 tests/backtest.py
```

## Validation (local)

Practice answers reconstructed from the live Exophase page (see
`tests/practice_answer.json` — the 04:30Z refresh read, verified
exactly: counts sum to 4,276 trophies earned at 3,868 players tracked):

- Practice mean pinball: **0.421** vs house baseline **1.508**
  (p00 0.420, p01 0.423; typical miss 0.20 pp vs 0.68 pp; 90% coverage 1.00 vs 0.00)
- Synthetic final (Nov-21 cut → Nov-26 truth, cohort-delay launch
  calibrated to the observed launch-night curve; slow / base / fast
  progression): **0.578 / 0.529 / 0.462** vs persistence baseline
  1.493 / 1.281 / 1.149 and curve baseline 0.868 / 0.609 / 0.719
- Output is byte-identical across reruns, 55 rows, monotone quantiles,
  under the 65,536-byte limit.

Beating the house baselines on practice and on a synthetic final does
not guarantee beating the house superbot model forecast on the real
final — a prize needs a loss strictly below the best house entry.
This is a lottery ticket.

## License

MIT — see [LICENSE](LICENSE).
