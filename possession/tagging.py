"""Smart Tagging as frame intervals: set pieces (restarts) and team-tagged phases (a weak prior).

The tagging is automatic, not ground truth (DATA.md). It is used for two things only:
  * a set piece marks a restart, so the ball was dead just before it;
  * a phase's Team tag is evidence about who has the ball, used only where the tracking has none.
"""

import difflib
import xml.etree.ElementTree as ET

import numpy as np

from . import config as C

SET_PIECES = "set pieces"
GOALS = "goals"


def load(path, fps=C.FPS):
    """[{code, start, stop, tags}] with frames half-open, codes lower-cased."""
    out = []
    for instance in ET.parse(path).iter("instance"):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        out.append({
            "code": (instance.findtext("code") or "").strip().lower(),
            "start": int(round(start_s * fps)),
            "stop": int(round(end_s * fps)),
            "tags": {(label.findtext("group") or "").strip().lower():
                     (label.findtext("text") or "").strip()
                     for label in instance.findall("label")},
        })
    return out


def side_of_team_tag(value, home_name, away_name):
    """0 home, 1 away, None for N/A or anything unrecognisable. Fuzzy: the XML says
    'Atletic Club' where the tracking metadata may say 'Athletic Club'."""
    if not value or value.strip().lower() in ("n/a", "na", "none", ""):
        return None
    v = value.lower()
    scores = [difflib.SequenceMatcher(None, v, (name or "").lower()).ratio()
              for name in (home_name, away_name)]
    if max(scores) < 0.4 or scores[0] == scores[1]:
        return None
    return 0 if scores[0] > scores[1] else 1


def set_pieces(events, home_name, away_name):
    """[(start, stop, side_or_None, type_lower)] for every SET PIECES instance."""
    out = []
    for e in events:
        if e["code"] == SET_PIECES:
            out.append((e["start"], e["stop"],
                        side_of_team_tag(e["tags"].get("team"), home_name, away_name),
                        (e["tags"].get("type") or "").lower()))
    return out


def goals(events):
    return [(e["start"], e["stop"]) for e in events if e["code"] == GOALS]


def team_phases(events, home_name, away_name):
    """{code: [(start, stop, side)]} for every non-set-piece code with a usable Team tag."""
    out = {}
    for e in events:
        if e["code"] in (SET_PIECES,):
            continue
        side = side_of_team_tag(e["tags"].get("team"), home_name, away_name)
        if side is None:
            continue
        out.setdefault(e["code"], []).append((e["start"], e["stop"], side))
    return out


def calibrated_prior(phases, labels, n_frames):
    """Per-frame home-minus-away vote from the tags, with each code's meaning learnt from the game.

    Some codes name the attacking team (BUILD UP), some the defending one (DEFENDING IN ...). We do
    not hard-code which: for each code we measure how often its team agrees with the possession
    our tracking pipeline already produced on frames where it produced one, keep the code if it
    mostly agrees, flip it if it mostly disagrees, drop it otherwise. That is self-supervision --
    no human label on the game is involved (RULES.md).

    Returns (vote array, {code: (agreement, decision)}).
    """
    vote = np.zeros(n_frames, dtype=np.float64)
    report = {}
    is_h, is_a = labels == "H", labels == "A"
    for code, intervals in sorted(phases.items()):
        agree = total = 0
        for start, stop, side in intervals:
            start, stop = max(0, start), min(n_frames, stop)
            if stop <= start:
                continue
            own = (is_h if side == 0 else is_a)[start:stop].sum()
            other = (is_a if side == 0 else is_h)[start:stop].sum()
            agree += own
            total += own + other
        if total == 0:
            report[code] = (None, "dropped")
            continue
        rate = agree / float(total)
        if rate >= C.TAG_AGREEMENT_KEEP:
            sign, decision = 1.0, "kept"
        elif rate <= 1.0 - C.TAG_AGREEMENT_KEEP:
            sign, decision = -1.0, "flipped"
        else:
            report[code] = (rate, "dropped")
            continue
        report[code] = (rate, decision)
        for start, stop, side in intervals:
            start, stop = max(0, start), min(n_frames, stop)
            vote[start:stop] += sign * (1.0 if side == 0 else -1.0)
    return vote, report
