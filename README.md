# Roblox Up-and-Coming top-10 forecaster

Entry for the [make.fun](https://make.fun) bounty
**"Predict Roblox's Up-and-Coming top 10"**
(`roblox-up-and-coming-forecast-01`, $500 prize pool, 20 places).

## What it does

`main.py` reads the chart history at `/input/history.ndjson` (plus
`/input/task.json`) and writes one forecast to
`/output/forecast.json`: for each of the top 10 places, a probability
for each game it names being at that place, and its top-10 guess in
order. The forecast is scored by log score against every hourly read
of Roblox's Up-and-Coming chart in the 72 hours after the deadline
(2026-11-13T06:00:00Z to 2026-11-16T06:00:00Z).

The program runs with no network in the grading sandbox, so
everything it knows comes from the history it is given. It re-reads
that history on every run: nothing about the current chart is
baked in.

## Method

Monte Carlo rank diffusion in latent-appeal space:

- Each game gets a latent appeal `a = -log(rank)`. Appeal follows a
  random walk whose step volatility and per-game drift are estimated
  from the history itself: the pooled rank-step volatility
  (sd of `da * rank / sqrt(dt)` over consecutive reads), and each
  game's recent appeal slope over the last 48 hours, shrunk and
  capped, with a weak pull back toward the chart mean.
- Games age out of the chart 28 days after their creation date when
  the history's details carry it.
- New games keep joining the chart (about 0.5/hour in the archive),
  so each simulated hour also draws phantom entrants from the
  empirical entry-rank distribution. Their top-10 mass is left
  unnamed on purpose — the scoring rule spreads a place's unnamed
  share over that read's unnamed games, which is exactly the hedge
  a brand-new climber needs.
- 1,200 seeded simulations x 72 hourly slots are counted: how often
  each real game sits at each place. Counts are blended with a small
  flat prior, capped per place, and every place keeps at least 5%
  unnamed. No game is ever staked at p = 1 and no plausible climber
  is left at zero.

## Validation

Holdout scoring with the official rule (fit on the early snapshots,
score on the later ones, log score averaged over the held-out reads):

- Fit on snapshots 1–24, score on 25–34 (11-05 to 11-07):
  **-40.3**, vs **-56.9** for the persistence baseline.
- Fit on snapshots 1–28, score on 29–34: **-34.0**, in the same band
  as the practice leaderboard's best entries (~-33.6 to -34.8).

Two runs on the same history write byte-identical output
(deterministic seed, ~20 s per run).

## Run it locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
cat out/forecast.json
```

Standard library only; Python 3.13 in the grading sandbox.

## License

MIT — see [LICENSE](LICENSE).
