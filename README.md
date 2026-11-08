# Ball possession from automatic tracking — More than a Hack 2027

Our entry to **[More than a Hack 2027, Phase 1: Ball Possession](https://github.com/metrica-sports/more-than-a-hack-2027)**
(Metrica Sports · FC Barcelona · Mobile World Capital Barcelona).

For every frame of FC Barcelona vs Athletic Club, the pipeline says who has the ball: `H` (home),
`A` (away), `D` (dead ball) or `X` (not graded). It turns the provided automatic tracking and Smart
Tagging into a 199,319-row CSV, deterministically, with no learnt parameters and no hand labels.

**How it works, in one line:** we detect events (control spells, the ball going out, restart kicks)
and run a possession state machine that changes hands only on *confirmed* opposing control. The
full method is in **[TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md)**.

## Reproduce the submission

Python 3.9+.

```bash
git clone https://github.com/joaquimmulaza/mtah2027-possession.git && cd mtah2027-possession
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt

# Unpack the match bundle into ./data (see DATA.md), or: export MTAH_DATA_DIR=/path/to/bundle
python -m possession.pipeline --team <TEAM_NAME>
python evaluation/validate_submission.py evaluation/results/<TEAM_NAME>.csv
```

The same data produces the same file, byte for byte: no randomness, no network access, no model
calls. The first run parses the raw tracking (1–3 min) and caches it under `data/cache/`.

## Check it without the data

```bash
python -m tests.test_pipeline
```

This generates a synthetic match in the exact file formats of the bundle, with its own ground
truth, and checks the file contract, the determinism, and that the pipeline beats the reference
baseline. CI runs it on every push. It uses no match data. Scores on it test the mechanics only and
say nothing about the real match.

| synthetic match | macroF1 | spellF1 | possessions (truth 51) |
|---|---|---|---|
| reference baseline (`smooth50`) | 0.47 | 0.22 | 161 |
| this pipeline | 0.94 | 0.83 | 46 |

## Repository layout

```
possession/                 our solution
  config.py                 every threshold, with the football reason behind it
  atd.py                    raw ATD -> arrays (cached in data/, never committed)
  features.py               metres, ball gap interpolation, ball speed, nearest player, intensity
  dead_ball.py              D: ball out, set pieces, nobody playing, goals/kickoffs; restart kicks
  states.py                 control spells -> confirmed possession changes
  tagging.py                Smart Tagging: restarts + a self-calibrated team prior
  pipeline.py               entry point
tests/                      synthetic match generator + end-to-end test
TECHNICAL_DOCUMENT.md       the technical document (deliverable 3)
requirements-lock.txt       pinned dependencies
```

Everything else is the organisers' starter repository, unchanged: the scorer (`evaluation/`), the
loaders (`starter/`, `games.py`) and the challenge documentation (`README_CHALLENGE.md`,
`CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `SUBMISSION.md`, `RULES.md`, `FAQ.md`).

## Compliance with the rules

- **Labels come from an algorithm only.** No hand labels or hand corrections on the provided game,
  in any amount, for any purpose.
- **No private ground truth and no human-produced data about this game** (no event data).
- **Not tuned on the public 5 minutes.** It was used only to check frame alignment and the file
  format. Every threshold is a prior, documented in `possession/config.py`.
- **No external training data.** The method has no learnt parameters.
- **The match data is not in this repository**, and neither is anything derived from it (`data/` is
  gitignored, the cache included). It has never been uploaded to any hosted LLM or cloud service.
- **Reproducible:** pinned dependencies, deterministic code, one command.

## Licence

MIT, see [LICENSE](LICENSE). The scorer, loaders and challenge documentation are © Metrica Sports
under the same MIT licence. The match data is not covered and is not distributed here.
