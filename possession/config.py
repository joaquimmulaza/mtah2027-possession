"""Every tunable number of the pipeline, in one place, with the football reason behind it.

TUNING (organisers' clarification: parameters MUST be tuned on external data only). The ten
parameters in tools/external.py GRID were tuned by coordinate descent on Metrica's public sample
games 1 and 2, with game 3 held out, using reference possession derived mechanically from their
event data: `python -m tools.external --tune` reproduces it. Values marked [tuned] below come from
that run; the others kept their prior because no grid value beat it. Nothing here was fitted to
the provided game, its ground truth or its video.

None of these values was fitted to the public 5-minute ground truth. RULES.md forbids it, and
EVALUATION.md shows why it would pick the wrong answer anyway: 17 possessions cannot rank two
settings. Each value below is a physical or footballing prior -- how fast players move, how close
a player must be to control a ball, how long a restart takes. If you change one, write the reason
next to it, and never the reason "it scored better on the public window".
"""

FPS = 25

# ---------------------------------------------------------------------------------------------
# Ball reconstruction
# ---------------------------------------------------------------------------------------------
# Gaps in the ball track shorter than this are bridged by straight-line interpolation. 0.4 s is
# about the time a ground pass needs to cross 4-6 m: short enough that a straight line is a fair
# description of what the ball did while the tracker lost it.
BALL_INTERP_MAX_GAP = 10

# Centred moving average applied to the ball before speeds are computed. Tracking jitter of a few
# centimetres per frame becomes metres per second once multiplied by 25 fps.
BALL_SMOOTH_FRAMES = 5

# ---------------------------------------------------------------------------------------------
# Control: "this team's player has the ball at its feet on this frame"
# ---------------------------------------------------------------------------------------------
# A player controls the ball when it is within this radius of them (metres). Feet are ~0.5 m from
# the tracked body centre and automatic tracking adds ~0.5-1 m of noise.
CONTROL_RADIUS_M = 2.5            # [tuned] prior 2.0

# ...and the nearest opponent is at least this much further away. Inside that margin it is a duel,
# which by the challenge's rules does not change possession on its own.
CONTROL_MARGIN_M = 0.5            # [tuned] prior 0.7

# ...and the ball is slower than this (m/s). A sprinting dribble is ~7-8 m/s; a pass in flight
# passing close to a player is 10-25 m/s and is not control.
CONTROL_MAX_BALL_SPEED = 10.0     # [tuned] prior 8.0

# Control frames of the same team separated by a gap this short are one spell (tracking flicker).
SPELL_JOIN_GAP = 4

# ---------------------------------------------------------------------------------------------
# Change of possession (CHALLENGE.md: "controlled possession by the opponent")
# ---------------------------------------------------------------------------------------------
# One opposing spell this long confirms a change: the opponent kept the ball, not just touched it.
CONFIRM_HOLD_FRAMES = 12         # [tuned] 0.48 s; prior 15

# Or: this many distinct opposing spells in a row with no spell of the possessing team in between,
# i.e. the opponent made a pass and someone of theirs received it. Distinct = different track slot
# or more than SPELL_DISTINCT_M apart.
CONFIRM_SPELLS = 2
SPELL_DISTINCT_M = 3.0

# Opposing spells shorter than this are never evidence: deflections, blocks, a toe on the ball.
MIN_EVIDENCE_FRAMES = 3

# When the ball was invisible between the last spell of the old team and the first spell of the
# new one, the hand-over happened somewhere in that hole. Longer than this, put the change in the
# middle of the hole instead of at the reappearance.
GAP_MIDPOINT_MIN = 10

# A possession shorter than this (inside live play) is merged into the one before it. A real
# possession needs at least a controlled touch and a decision; 1 s is the shortest a coach would
# count. It is also what keeps spellF1 from being flooded by flicker.
MIN_POSSESSION_FRAMES = 38        # [tuned] 1.5 s; prior 25

# ---------------------------------------------------------------------------------------------
# Dead ball
# ---------------------------------------------------------------------------------------------
# Ball outside the pitch lines by more than this (metres) is out of play. The margin absorbs
# tracking noise on balls running along the line.
OUT_MARGIN_M = 0.5

# Collective intensity: median speed of the tracked players, over a centred window. Absolute
# thresholds do not transfer between matches (players walk far more than intuition says -- the
# median frame of a match is ~1.4 m/s, not 2-3), so the thresholds are fractions of the match's own
# median intensity over the graded frames: a self-calibration with no labels involved. A stoppage
# is the whole pitch clearly below its usual pace.
ACTIVITY_WINDOW = 50              # 2 s
ACTIVITY_DEAD_REL = 0.4           # [tuned] below 40% of the match median: nobody is playing; prior 0.6
ACTIVITY_SLOW_REL = 0.85          # below 85% of it, and the ball is still or untracked: a set piece
                                  # being set up

# Ball "still": smoothed speed below this (m/s), as a fraction of the visible frames of a window.
BALL_STILL_MS = 0.8
BALL_STILL_WINDOW = 25
BALL_STILL_FRACTION = 0.8

# A restart is the ball being struck: speed above this for RESTART_RUN frames after being still.
RESTART_KICK_MS = 3.0             # [tuned] prior 4.0
RESTART_RUN = 3

# How long after the ball leaves the pitch to look for the restart before giving up (frames).
OUT_MAX_DEAD = 90 * FPS

# Morphology on the dead mask: drop dead runs shorter than this, fill live gaps shorter than this.
MIN_DEAD_FRAMES = 25              # 1 s: shorter "stoppages" are tracking noise
MIN_LIVE_FRAMES = 25              # 1 s of play between two stoppages is not a real restart

# Smart Tagging set-piece windows are ~5 s around the restart; search this far either side.
TAG_RESTART_SLACK = 75            # 3 s

# Before a tagged set piece the ball has been dead for at least this long.
SETPIECE_MIN_DEAD = 50            # 2 s

# ---------------------------------------------------------------------------------------------
# Smart Tagging team prior (used only where the tracking gives no evidence at all)
# ---------------------------------------------------------------------------------------------
# A tag code's team is kept as possession evidence only if it agrees with our own tracking-derived
# possession on at least this fraction of frames -- measured on the game itself, algorithmically,
# with no human label involved. Codes below 1 - this are flipped (the tag names the defending team).
TAG_AGREEMENT_KEEP = 0.6
