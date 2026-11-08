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
instead of a per-frame guess that flickers.

## 2. Pipeline

| step | module | what it does |
|---|---|---|
| read | `possession/atd.py` | streams the raw ATD into dense arrays (cached under `data/cache/`, gitignored) |
| geometry | `possession/features.py` | metres, ball gaps ≤ 0.4 s interpolated, smoothed ball speed, nearest player per team, collective player speed |
| dead ball | `possession/dead_ball.py` | `D` from: ball out of the pitch; tagged set pieces; "nobody is playing" (median player speed); goals and the pre-kickoff stretch of each half |
| possession | `possession/states.py` | control spells → confirmed changes → back-dated hand-overs → merge possessions < 1 s |
| prior | `possession/tagging.py` | Smart Tagging team tags, self-calibrated, only where the tracking gives no evidence |
| output | `possession/pipeline.py` | `H/A/D` on graded frames, `X` elsewhere; 199,319 rows |

### 2.1 Control

A team controls the ball on a frame when its nearest player is within 2.0 m of the ball, the nearest
opponent is at least 0.7 m further, and the ball moves slower than 8 m/s (a pass in flight passing
close to someone is not control). Consecutive control frames, with gaps ≤ 4 frames, form a
**control spell**.

### 2.2 Change of possession

An opposing spell confirms a change if it lasts ≥ 0.6 s (the opponent *kept* the ball), or if two
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
3. **Nobody playing**: median player speed over 2 s below 1.3 m/s, or below 2.0 m/s with the ball
   still or untracked. This catches the fouls and stoppages the tagging does not.
4. **Goals and kickoffs**: dead from the ball crossing the line to the next kickoff; inside each
   graded half, everything before its kickoff (the second half is graded before it kicks off).

A **restart kick** is the ball, still or invisible for ≥ 0.5 s, moving faster than 4 m/s for 3 frames.
Every stoppage ends on one, so the first possession after a restart starts on the right frame. The
dead mask is cleaned: dead runs < 1 s are dropped, live gaps < 1 s between stoppages are filled.

### 2.4 Smart Tagging as a calibrated prior

Some codes name the attacking team, some the defending one. We do not hard-code which: each code's
team tag is compared with the possession our tracking-only pass already produced, and the code is
kept (≥ 60 % agreement), flipped (≤ 40 %) or dropped. This is self-supervision on the game — no
human label is involved. The resulting vote is only used for live stretches where the tracking
gives no control evidence at all. The restarting team of a tagged set piece seeds the first
possession after it.

## 3. Assets used

| asset | used | how |
|---|---|---|
| tactical ATD | yes | all geometry |
| Smart Tagging | yes | restarts, kickoffs, goals, weak team prior |
| homography | no | tactical ATD is already in pitch coordinates; reserved for video-based ball recovery |
| tactical video | no | see §6 |
| broadcast video / ATD | no | not synchronised; mapping it back was not worth the time |

## 4. Training data

**None.** The method has no learnt parameters. Every threshold in `possession/config.py` is a
physical or footballing prior (control radius, sprint speed, restart duration) with its reason
written next to it. None was tuned on the public 5-minute ground truth; it was only used to check
that frames are aligned and labels mean what we think (RULES.md). No hand labels of any kind, on
this game or any other, were used.

## 5. Possession rules implemented

- **Loose ball**: stays with the last team in possession until the opponent confirms control.
- **Challenges / duels**: both players within 0.7 m of each other relative to the ball → no evidence.
- **Deflections and blocks**: shorter than 0.6 s and not followed by an opposing pass → ignored.
- **Restarts**: possession starts at the restart kick with the team that takes it (tracking first,
  Smart Tagging team as fallback).
- **Shortest possession**: 1 s; shorter ones are merged into the previous possession.
- **Ball invisible**: the state is held; hand-overs inside a hole are placed at its midpoint.

## 6. Limitations and what we would do with more time

- **41 % of graded frames have no ball.** We hold state through them; a detector on the tactical
  video (e.g. a YOLO ball model trained on *other* games, run on the provided video) could recover
  part of it and is the largest expected gain.
- **No player identities**: "distinct receiver" uses track slot or distance, so identity switches
  can occasionally look like a pass.
- **Thresholds are priors, not fitted.** A self-supervised calibration (e.g. fit the control radius
  to the distribution of ball-to-nearest-player distances at low ball speed) would be the next step.
- **Free kicks without a tag** rely on collective speed; a quick free kick can be missed.

## 7. Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# unpack the data bundle into ./data (or set MTAH_DATA_DIR)
python -m possession.pipeline --team <TEAM_NAME>
python evaluation/validate_submission.py evaluation/results/<TEAM_NAME>.csv
```

The pipeline is deterministic (no randomness, no network, no model calls): the same data produces
the same file, byte for byte. `python -m tests.test_pipeline` checks this, and the file contract, on
a synthetic match.
