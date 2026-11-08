"""Who has the ball while it is alive (labels H / A).

Per frame we only ask a local question: does one team's player have the ball at their feet?
That gives *control spells* -- short runs of one team touching the ball. Possession is then a
state that changes only on *confirmed* opposing control, exactly as CHALLENGE.md defines it:

  * an opposing spell of CONFIRM_HOLD_FRAMES or more   (the opponent kept the ball), or
  * CONFIRM_SPELLS distinct opposing spells in a row     (the opponent passed it among themselves)

Touches, blocks, deflections and duels never confirm anything, so they never split a possession.
The decision is made offline over the whole match, so when a change is confirmed it is dated back
to the frame the new team first touched the ball -- not to the frame we became sure.
"""

import numpy as np

from . import config as C
from .features import runs_of

HOME, AWAY, NONE = 0, 1, -1


def control(f):
    """(team per frame or NONE, slot per frame)."""
    dh, da = f["d_home"], f["d_away"]
    speed = np.nan_to_num(f["ball_speed"], nan=np.inf)
    ok = f["have_ball"] & (speed < C.CONTROL_MAX_BALL_SPEED) & ~f["out_of_pitch"]
    with np.errstate(invalid="ignore"):  # inf - inf where a team has nobody tracked
        home = ok & (dh < C.CONTROL_RADIUS_M) & (da - dh > C.CONTROL_MARGIN_M)
        away = ok & (da < C.CONTROL_RADIUS_M) & (dh - da > C.CONTROL_MARGIN_M)
    team = np.full(len(dh), NONE, dtype=np.int8)
    team[home] = HOME
    team[away] = AWAY
    slot = np.where(home, f["slot_home"], np.where(away, f["slot_away"], -1))
    return team, slot


def spells(team, slot, ball, start, stop):
    """[(s, e, team, slot, x, y)] control spells inside [start, stop), short gaps joined."""
    out = []
    for side in (HOME, AWAY):
        mask = team[start:stop] == side
        rs = runs_of(mask)
        merged = []
        for s, e in rs:
            if merged and s - merged[-1][1] <= C.SPELL_JOIN_GAP:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        for s, e in merged:
            s, e = s + start, e + start
            sl = slot[s:e]
            sl = sl[sl >= 0]
            main = int(np.bincount(sl).argmax()) if len(sl) else -1
            x, y = np.nanmean(ball[s:e], axis=0)
            out.append((s, e, side, main, x, y))
    out.sort()
    return out


def _distinct(a, b):
    return a[3] != b[3] or np.hypot(a[4] - b[4], a[5] - b[5]) > C.SPELL_DISTINCT_M


def _change_frame(prev_end, new_start, have_ball):
    """Where to put a hand-over between the old team's last touch and the new team's first."""
    if prev_end is None or new_start - prev_end < C.GAP_MIDPOINT_MIN:
        return new_start
    hole = ~have_ball[prev_end:new_start]
    if hole.mean() > 0.5:
        return (prev_end + new_start) // 2
    return new_start


def segment_live(sp, start, stop, have_ball, first_team):
    """Possession labels (HOME/AWAY) for one live stretch [start, stop)."""
    labels = np.full(stop - start, NONE, dtype=np.int8)
    if first_team is None:
        first_team = next((s[2] for s in sp if s[1] - s[0] >= C.MIN_EVIDENCE_FRAMES), None)
    if first_team is None:
        return labels  # no evidence at all; caller fills from the prior
    current = first_team
    changes = [(start, current)]
    last_own_end = None
    pending = []  # opposing spells since the last own spell

    for s in sp:
        s_start, s_end, side = s[0], s[1], s[2]
        if s_end - s_start < C.MIN_EVIDENCE_FRAMES:
            continue
        if side == current:
            last_own_end = s_end
            pending = []
            continue
        if pending and not _distinct(pending[-1], s):
            pending[-1] = (pending[-1][0], s_end) + pending[-1][2:]
        else:
            pending.append(s)
        held = max(p[1] - p[0] for p in pending)
        if held >= C.CONFIRM_HOLD_FRAMES or len(pending) >= C.CONFIRM_SPELLS:
            frame = _change_frame(last_own_end, pending[0][0], have_ball)
            frame = max(frame, changes[-1][0] + 1)
            current = side
            changes.append((frame, current))
            last_own_end = pending[-1][1]
            pending = []

    for i, (frame, side) in enumerate(changes):
        end = changes[i + 1][0] if i + 1 < len(changes) else stop
        labels[frame - start:end - start] = side
    return labels


def merge_short(labels, live_runs):
    """Merge possessions shorter than MIN_POSSESSION_FRAMES into the previous one (or the next,
    at the head of a live stretch). Repeated until stable."""
    out = labels.copy()
    for start, stop in live_runs:
        changed = True
        while changed:
            changed = False
            window = out[start:stop]
            if not len(window):
                break
            breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
            bounds = np.concatenate(([0], breaks, [len(window)]))
            if len(bounds) <= 2:
                break
            lengths = np.diff(bounds)
            i = int(np.argmin(lengths))
            if lengths[i] >= C.MIN_POSSESSION_FRAMES:
                break
            a, b = bounds[i], bounds[i + 1]
            fill = window[a - 1] if a > 0 else window[b]
            window[a:b] = fill
            changed = True
    return out


def label_live(f, dead, restarts, prior=None):
    """HOME/AWAY/NONE per frame; NONE on dead frames and where no evidence exists."""
    n = len(dead)
    team, slot = control(f)
    out = np.full(n, NONE, dtype=np.int8)
    live_runs = runs_of(~dead)
    for start, stop in live_runs:
        sp = spells(team, slot, f["ball"], start, stop)
        first = restarts.get(start)
        out[start:stop] = segment_live(sp, start, stop, f["have_ball"], first)
        if prior is not None and np.any(out[start:stop] == NONE):
            # no tracking evidence for this whole stretch: fall back to the tagging vote
            vote = prior[start:stop].sum()
            if vote != 0:
                out[start:stop] = HOME if vote > 0 else AWAY
    return merge_short(out, live_runs), team
