"""Read the raw ATD into dense numpy arrays, once, and cache them under data/cache/.

Raw line format (see starter/baseline.py): "<frame_id>:<slot>;<slot>;...:<ball>", each group
"x,y,speed" in normalised 0-1 pitch coordinates, NaN where the object is not seen. frame_id
counts from 1; arrays here are indexed by video frame (frame_id - 1), i.e. by submission row.

The cache lives inside data/, which is gitignored: it is derived frame-level tracking and must not
be published (RULES.md, "Do not publish derived data that would reconstruct it").
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

HOME, AWAY = 0, 1
NAN = float("nan")


def team_names(meta_path):
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))


def team_of_each_slot(meta_path):
    """0 for home slots, 1 for away. First <Team> is home (FIFA/EPTS; same rule as the baseline)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def _group(text):
    """(x, y, speed) of one 'x,y,speed' group; NaN for anything missing."""
    if not text or text[0] in "Nn":
        return NAN, NAN, NAN
    values = text.split(",")
    out = []
    for i in range(3):
        try:
            out.append(float(values[i]))
        except (IndexError, ValueError):
            out.append(NAN)
    return out[0], out[1], out[2]


def parse_raw(raw_path, n_frames, n_slots):
    """{'players': (F, S, 2), 'speed': (F, S), 'ball': (F, 2)} as float32, NaN where unseen."""
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)

    row = np.empty(n_slots * 3, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) < 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            row.fill(np.nan)
            for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                row[slot * 3:slot * 3 + 3] = _group(group)
            block = row.reshape(n_slots, 3)
            players[frame] = block[:, :2]
            speed[frame] = block[:, 2]
            bx, by, _ = _group(parts[2].strip())
            ball[frame] = (bx, by)
    return {"players": players, "speed": speed, "ball": ball}


def load(meta_path, raw_path, n_frames, cache_dir=None):
    """Parsed ATD plus slot teams, from the cache when it is newer than the raw file."""
    team = team_of_each_slot(meta_path)
    cache = None
    if cache_dir:
        cache = os.path.join(cache_dir, os.path.basename(raw_path) + ".npz")
        if os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(raw_path):
            with np.load(cache) as stored:
                data = {key: stored[key] for key in ("players", "speed", "ball")}
            if data["ball"].shape[0] == n_frames and data["speed"].shape[1] == len(team):
                data["team"] = team
                return data

    data = parse_raw(raw_path, n_frames, len(team))
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache, **data)
    data["team"] = team
    return data
