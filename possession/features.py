"""Per-frame geometry: where the ball is, how fast it moves, who is nearest, how hard anyone runs."""

import numpy as np

from . import config as C


def runs_of(mask):
    """[(start, stop)] half-open runs where a boolean array is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def interpolate_short_gaps(xy, max_gap):
    """Linear interpolation across NaN runs of at most max_gap frames, bounded on both sides."""
    out = xy.copy()
    missing = np.isnan(xy[:, 0])
    n = len(xy)
    for start, stop in runs_of(missing):
        if start == 0 or stop == n or stop - start > max_gap:
            continue
        a, b = xy[start - 1], xy[stop]
        t = (np.arange(start, stop) - (start - 1)) / float(stop - (start - 1))
        out[start:stop] = a + (b - a) * t[:, None]
    return out


def nan_moving_mean(values, window):
    """Centred moving mean ignoring NaN; NaN where the window has no valid value."""
    values = np.asarray(values, dtype=np.float64)
    valid = ~np.isnan(values)
    filled = np.where(valid, values, 0.0)
    half = window // 2
    csum = np.concatenate(([0.0], np.cumsum(filled)))
    ccount = np.concatenate(([0], np.cumsum(valid)))
    idx = np.arange(len(values))
    lo = np.maximum(0, idx - half)
    hi = np.minimum(len(values), idx + half + 1)
    count = ccount[hi] - ccount[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (csum[hi] - csum[lo]) / np.maximum(count, 1), np.nan)


def moving_fraction(mask, valid, window):
    """Fraction of valid frames in a centred window where mask is True; NaN with no valid frame."""
    m = np.where(valid, mask, False).astype(np.float64)
    m[~valid] = np.nan
    return nan_moving_mean(m, window)


def compute(atd, pitch):
    """Everything downstream needs, in metres and m/s. All arrays are length n_frames."""
    length, width = pitch
    scale = np.array([length, width], dtype=np.float64)
    team = atd["team"]

    ball_raw = atd["ball"].astype(np.float64) * scale
    seen = ~np.isnan(ball_raw[:, 0])
    ball = interpolate_short_gaps(ball_raw, C.BALL_INTERP_MAX_GAP)
    have_ball = ~np.isnan(ball[:, 0])

    smooth = np.stack([nan_moving_mean(ball[:, k], C.BALL_SMOOTH_FRAMES) for k in (0, 1)], 1)
    smooth[~have_ball] = np.nan
    velocity = np.gradient(smooth, axis=0) * C.FPS
    ball_speed = np.hypot(velocity[:, 0], velocity[:, 1])

    players = atd["players"].astype(np.float64) * scale
    visible = ~np.isnan(players[:, :, 0])
    dist = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    dist = np.where(np.isnan(dist), np.inf, dist)

    nearest = {}
    for side in (0, 1):
        d_side = np.where(team[None, :] == side, dist, np.inf)
        slot = np.argmin(d_side, axis=1)
        nearest[side] = (d_side[np.arange(len(slot)), slot], slot)

    # Player speed: the ATD carries one; fall back to finite differences where it does not.
    speed = atd["speed"].astype(np.float64)
    if np.all(np.isnan(speed)):
        step = np.hypot(*np.moveaxis(np.diff(players, axis=0, prepend=players[:1]), 2, 0))
        speed = step * C.FPS
    import warnings
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with nobody tracked -> NaN
        frame_speed = np.nanmedian(np.where(visible, speed, np.nan), axis=1)
    activity = nan_moving_mean(frame_speed, C.ACTIVITY_WINDOW)

    in_pitch = ((ball[:, 0] >= -C.OUT_MARGIN_M) & (ball[:, 0] <= length + C.OUT_MARGIN_M)
                & (ball[:, 1] >= -C.OUT_MARGIN_M) & (ball[:, 1] <= width + C.OUT_MARGIN_M))

    return {
        "ball": ball,
        "ball_seen": seen,
        "have_ball": have_ball,
        "ball_speed": ball_speed,
        "out_of_pitch": have_ball & ~in_pitch,
        "d_home": nearest[0][0], "slot_home": nearest[0][1],
        "d_away": nearest[1][0], "slot_away": nearest[1][1],
        "players": players,
        "activity": activity,
        "n_players": visible.sum(axis=1),
    }
