# <TEAM NAME> — Technical document

More than a Hack 2027, Phase 1: Ball Possession · FC Barcelona vs Athletic Club

## 1. Approach in one paragraph

We do not classify frames. We detect **events** — the ball being controlled by a player, the ball
leaving the pitch, a restart kick — and run a **possession state machine** over them. Possession
changes only when the opponent gains *controlled* possession (holds the ball, or passes it among
themselves), exactly as CHALLENGE.md defines it. Because the whole match is processed offline, a
confirmed change is dated back to the first touch of the new team rather than to the moment we
became sure. Dead ball is detected separately from four independent signals and always ends on the
restart kick. This targets `spellF1` directly: few, whole possessions with sharp boundaries,
instead of a per-frame guess that flickers. Every tunable parameter was tuned on **external games
only** (§4), as the organisers' rules clarification requires.

## 2. Pipeline

| step | module | what it does |
|---|---|---|
| read | `possession/atd.py` | streams the raw ATD into dense arrays (cached under `data/cache/`, gitignored) |
| geometry | `possession/features.py` | metres, ball gaps ≤ 0.4 s interpolated, smoothed ball speed, nearest player per team, collective player speed |
| dead ball | `possession/dead_ball.py` | `D` from: ball out of the pitch; tagged set pieces; "nobody is playing" (collective speed); goals and the pre-kickoff stretch of each half |
| possession | `possession/states.py` | control spells → confirmed changes → back-dated hand-overs → merge possessions < 1.5 s |
| prior | `possession/tagging.py` | Smart Tagging team tags, sign learnt automatically, only where the tracking gives no evidence |
| output | `possession/pipeline.py` | `H/A/D` on graded frames, `X` elsewhere; 199,319 rows |

The algorithm itself is one function, `pipeline.infer`, shared by the submission run and by the
external tuning harness, so the tuned parameters are exactly the ones that label the provided game.

### 2.1 Control

A team controls the ball on a frame when its nearest player is within 2.5 m of the ball, the nearest
opponent is at least 0.5 m further, and the ball moves slower than 10 m/s (a pass in flight passing
close to someone is not control). Consecutive control frames, with gaps ≤ 4 frames, form a
**control spell**.

### 2.2 Change of possession

An opposing spell confirms a change if it lasts ≥ 0.48 s (the opponent *kept* the ball), or if two
**distinct** opposing spells (different track slot or > 3 m apart) occur with no spell of the
possessing team in between (the opponent *passed* it). Spells shorter than 3 frames are never
evidence. This encodes the challenge's "does not count" list: a touch without control, a deflection,
an aerial duel without subsequent control.

When confirmed, the change is placed at the first frame of the first confirming spell. If the ball
was invisible for most of the time between the old team's last touch and the new team's first, the
change goes in the middle of that hole.

### 2.3 Dead ball

1. **Out of the pitch**: ball > 0.5 m beyond a line → dead until the next restart kick.
2. **Tagged set pieces**: dead for ≥ 2 s before the restart kick found inside the tag window ± 3 s,
   extended back over any already-dead stretch.
3. **Nobody playing**: median player speed over 2 s below 40 % of the match's own median
   intensity, or below 85 % of it with the ball still or untracked. This catches the fouls and
   stoppages the tagging does not. The thresholds are expressed relative to the match's own median
   — a normalisation of the input computed at run time, like scaling coordinates to metres —
   because absolute speeds do not transfer between matches and tracking systems. The two
   fractions themselves were tuned on external games (§4).
4. **Goals and kickoffs**: dead from the ball crossing the line to the next kickoff; inside each
   graded half, everything before its kickoff (the second half is graded before it kicks off).

A **restart kick** is the ball, still or invisible for ≥ 0.5 s, moving faster than 3 m/s for 3 frames.
Every stoppage ends on one, so the first possession after a restart starts on the right frame. The
dead mask is cleaned: dead runs < 1 s are dropped, live gaps < 1 s between stoppages are filled.

### 2.4 Smart Tagging as a prior

Some codes name the attacking team, some the defending one. We do not hard-code which: at run time
each code's team tag is compared with the possession our own tracking-only pass produced, and the
code is kept (≥ 60 % agreement), flipped (≤ 40 %) or dropped. This uses only generated data — our
output and the automatic tagging — never the ground truth or the video. The resulting vote is only
used for live stretches where the tracking gives no control evidence at all. The restarting team of
a tagged set piece seeds the first possession after it.

### 2.5 Automatic checks on the provided game (no ground truth, no video)

- **Smart Tagging agreement**: codes describing the team on the ball agree with our labels on
  69–91 % of frames (`build up` 0.91, `progression` 0.87), and every `defending …` / `defensive
  transition` code comes out flipped, i.e. it names the team without the ball, as its meaning implies.
- **Plausibility of aggregates**: 37 % of graded frames dead, 332 possessions of 11.9 s on average.
- **Robustness**: `python -m tools.stability` moves each of eight thresholds by ±25 % and compares
  with the default run. In 14 of 16 variants the possession structure is 92–100 % the same
  (spellF1 of variant vs default 0.92–1.00, frame agreement ≥ 97 %). The sensitive one is
  `ACTIVITY_SLOW_REL`, which moves the dead share between 30 % and 45 %.

### 2.6 Where we write `X`

Outside the two graded halves, always. Inside them, only on live stretches where neither the
tracking nor the Smart Tagging gives any evidence for either team (1,430 of 158,857 graded frames).
`spellF1` forward-fills these, so they cost nothing there; inventing a team would risk splitting a
possession.

## 3. Assets used

| asset | used | how |
|---|---|---|
| tactical ATD | yes | all geometry |
| Smart Tagging | yes | restarts, kickoffs, goals, weak team prior |
| homography | no | tactical ATD is already in pitch coordinates |
| tactical video | no | see §6 |
| broadcast video / ATD | no | not synchronised; mapping it back was not worth the time |
| public 5-minute ground truth | check only | file format and frame alignment; never used to choose, tune or calibrate anything |
| Metrica public sample games 1–3 | yes | external tuning and validation (§4) |

## 4. Training and tuning data — external only

**External dataset.** Metrica Sports' public sample data
(<https://github.com/metrica-sports/sample-data>): three anonymised professional matches with
synchronised tracking and event data, published as "sample data to play around with to the broader
football analytics community", with the request to acknowledge the source — which we do here. It is
downloaded into `data/external/` (gitignored) by `python -m tools.external --download`.

**Reference possession on those games** is derived *mechanically* from their event data, with no
human judgement of ours: possession is the team of the latest on-ball event (pass, carry, recovery,
shot, ball lost, set piece, ball out), duels and fouls received do not change it; dead ball runs
from the end of the last event before each set piece to that set piece, plus the stretch before each
kick-off. This is a proxy: the challenge's ground truth comes from corrected tracking, and
event-derived possessions are somewhat shorter (every recovery counts), so absolute scores on the
two are not comparable.

**Matching the provided game's conditions.** The sample tracking is cleaner than the provided
automatic tracking, so it is degraded deterministically (fixed seed) to the conditions DATA.md
publishes for the provided game: the ball hidden in bursts on ~41 % of frames, ~0.3 m per-axis
position noise, ~7 % of player observations missing.

**Protocol.** Coordinate descent over ten parameters (`tools/external.py`, `GRID`), two rounds,
maximising mean `spellF1` on games 1 and 2; game 3 held out. `python -m tools.external --tune`
reproduces it.

| | game 1 (train) | game 2 (train) | game 3 (held out) |
|---|---|---|---|
| reference baseline `smooth50` (nearest player, ±2 s vote) | 0.298 | 0.272 | 0.254 |
| our priors, before tuning | 0.424 | 0.472 | 0.448 |
| **tuned** | **0.464** | **0.497** | **0.459** |

The held-out game improves too, so the tuning generalises rather than memorising the two training
games. Seven parameters moved (marked `[tuned]` in `possession/config.py`, with their prior), three
kept their prior.

**No hand labels** of any kind, on this game or any other, were used. Nothing was trained, fitted
or calibrated on the provided game's ground truth or video.

## 5. Possession rules implemented

- **Loose ball**: stays with the last team in possession until the opponent confirms control.
- **Challenges / duels**: opponent within 0.5 m of the same distance to the ball → no evidence.
- **Deflections and blocks**: shorter than 0.48 s and not followed by an opposing pass → ignored.
- **Restarts**: possession starts at the restart kick with the team that takes it (tracking first,
  Smart Tagging team as fallback).
- **Shortest possession**: 1.5 s; shorter ones are merged into the previous possession.
- **Ball invisible**: the state is held; hand-overs inside a hole are placed at its midpoint.

## 6. Limitations and what we would do with more time

- **41 % of graded frames have no ball.** We hold state through them; a ball detector trained on
  *other* games and run on the provided video could recover part of it and is the largest expected
  gain.
- **The external reference is a proxy.** Possession derived from events is not the same definition
  as the challenge's corrected-tracking ownership; three games is also a small tuning set.
- **No player identities**: "distinct receiver" uses track slot or distance, so identity switches
  can occasionally look like a pass.
- **Free kicks without a tag** rely on collective speed; a quick free kick can be missed.

## 7. Reproduction

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt
# unpack the data bundle into ./data (or set MTAH_DATA_DIR)
python -m possession.pipeline --team <TEAM_NAME>
python evaluation/validate_submission.py evaluation/results/<TEAM_NAME>.csv

# optional: reproduce the external tuning
python -m tools.external --download
python -m tools.external --tune
```

The pipeline is deterministic (no randomness, no network, no model calls): the same data produces
the same file, byte for byte — checked from a clean clone, with and without the cache.
`python -m tests.test_pipeline` checks this, and the file contract, on a synthetic match.
