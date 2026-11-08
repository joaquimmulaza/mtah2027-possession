"""Which frames is the ball out of play on? (label D)

Four independent kinds of evidence, combined as a union and then cleaned up:

  1. the ball left the pitch          -> dead until the next restart kick
  2. a tagged set piece               -> dead for a while before its restart kick
  3. nobody is playing                -> collective speed is a walk, or a jog with a still ball
  4. a goal / the second-half kickoff -> dead until the kickoff restart

Every dead stretch ends at a *restart kick*: the ball, still (or untracked) for half a second,
suddenly moving faster than RESTART_KICK_MS. Ending stoppages on that instant rather than on a
smoothed signal is what makes the first possession after a restart start on the right frame,
which is most of what spellF1 rewards.
"""

import numpy as np

from . import config as C
from .features import moving_fraction, runs_of

QUIET_BEFORE_KICK = 12  # frames of a still or invisible ball that must precede a kick


def _forward_all(mask, n):
    """out[t] = mask[t:t+n].all()"""
    c = np.concatenate(([0], np.cumsum(mask.astype(np.int64))))
    idx = np.arange(len(mask))
    hi = np.minimum(len(mask), idx + n)
    return (c[hi] - c[idx]) == (hi - idx)


def _backward_all(mask, n):
    """out[t] = mask[t-n:t].all() (False when fewer than n frames precede t)"""
    c = np.concatenate(([0], np.cumsum(mask.astype(np.int64))))
    idx = np.arange(len(mask))
    lo = idx - n
    ok = lo >= 0
    lo = np.maximum(lo, 0)
    return ok & ((c[idx] - c[lo]) == n)


def kick_onsets(f):
    """Boolean per frame: a restart-like kick starts here."""
    speed = np.nan_to_num(f["ball_speed"], nan=0.0)
    fast = (speed > C.RESTART_KICK_MS) & f["have_ball"] & ~f["out_of_pitch"]
    quiet = ~f["have_ball"] | f["out_of_pitch"] | (speed < 2 * C.BALL_STILL_MS)
    return _forward_all(fast, C.RESTART_RUN) & _backward_all(quiet, QUIET_BEFORE_KICK)


def first_in(onsets, start, stop):
    start, stop = max(0, start), min(len(onsets), stop)
    hits = np.flatnonzero(onsets[start:stop])
    return start + int(hits[0]) if len(hits) else None


def last_in(onsets, start, stop):
    start, stop = max(0, start), min(len(onsets), stop)
    hits = np.flatnonzero(onsets[start:stop])
    return start + int(hits[-1]) if len(hits) else None


def clean(mask, min_true, min_false):
    """Fill short False gaps, then drop short True runs."""
    out = mask.copy()
    for start, stop in runs_of(~out):
        if 0 < start and stop < len(out) and stop - start < min_false:
            out[start:stop] = True
    for start, stop in runs_of(out):
        if stop - start < min_true:
            out[start:stop] = False
    return out


def detect(f, set_pieces, goals, blocks):
    """(dead mask, restarts) where restarts is {frame: side_or_None} -- the kicks that end a
    stoppage, with the restarting team when the Smart Tagging names it."""
    n = len(f["ball_speed"])
    onsets = kick_onsets(f)
    restarts = {}

    # 3. nobody is playing
    graded = np.zeros(n, dtype=bool)
    for first, last in blocks:
        graded[first:last] = True
    reference = float(np.nanmedian(f["activity"][graded])) if graded.any() else 1.0
    dead_ms = C.ACTIVITY_DEAD_REL * reference
    slow_ms = C.ACTIVITY_SLOW_REL * reference
    activity = np.nan_to_num(f["activity"], nan=0.0)
    speed = f["ball_speed"]
    still_frac = moving_fraction(speed < C.BALL_STILL_MS, f["have_ball"] & ~np.isnan(speed),
                                 C.BALL_STILL_WINDOW)
    still = np.nan_to_num(still_frac, nan=0.0) >= C.BALL_STILL_FRACTION
    no_ball = np.nan_to_num(moving_fraction(f["have_ball"], np.ones(n, bool), C.ACTIVITY_WINDOW),
                            nan=0.0) == 0
    dead = (activity < dead_ms) | ((activity < slow_ms) & (still | no_ball))

    # Each activity stoppage ends on the kick nearest its end, if there is one.
    for start, stop in runs_of(dead):
        kick = first_in(onsets, stop - C.ACTIVITY_WINDOW, stop + C.ACTIVITY_WINDOW)
        if kick is not None and kick > start:
            dead[start:kick] = True
            dead[kick:stop] = False
            restarts.setdefault(kick, None)

    # 1. ball left the pitch
    for start, stop in runs_of(f["out_of_pitch"]):
        if stop - start < 2:
            continue
        kick = first_in(onsets, stop, start + C.OUT_MAX_DEAD)
        if kick is None:
            # no visible kick: dead until the players are moving again
            later = np.flatnonzero(~dead[stop:min(n, start + C.OUT_MAX_DEAD)])
            kick = stop + int(later[0]) if len(later) else stop
        dead[start:kick] = True
        restarts.setdefault(kick, None)

    # 2. tagged set pieces
    kickoffs = []
    for start, stop, side, kind in set_pieces:
        kick = first_in(onsets, start - C.TAG_RESTART_SLACK, stop + C.TAG_RESTART_SLACK)
        if kick is None:
            kick = (start + stop) // 2
        begin = kick - C.SETPIECE_MIN_DEAD
        # extend back over an already-dead stretch that leads into it
        while begin > 0 and dead[begin - 1] and kick - begin < C.OUT_MAX_DEAD:
            begin -= 1
        dead[max(0, begin):kick] = True
        dead[kick:min(n, kick + C.RESTART_RUN)] = False
        restarts[kick] = side if side is not None else restarts.get(kick)
        if "kick off" in kind:
            kickoffs.append(kick)

    # 4. goals: dead from the ball crossing the line to the next kickoff
    for start, stop in goals:
        out = np.flatnonzero(f["out_of_pitch"][start:min(n, stop + C.TAG_RESTART_SLACK)])
        begin = start + int(out[0]) if len(out) else stop
        after = [k for k in kickoffs if k > begin]
        if after:
            dead[begin:min(after)] = True

    # Inside each graded block, everything before the first kickoff is dead (the second half is
    # graded from 110808 but kicks off later -- FAQ.md).
    for first, last in blocks:
        ks = [k for k in kickoffs if first <= k <= last]
        if ks:
            dead[first:min(ks)] = True

    dead = clean(dead, C.MIN_DEAD_FRAMES, C.MIN_LIVE_FRAMES)
    # restarts that no longer end a dead stretch are dropped
    restarts = {k: s for k, s in restarts.items() if 0 < k < n and dead[k - 1] and not dead[k]}
    return dead, restarts
