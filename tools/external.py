"""Tune and validate the pipeline on EXTERNAL games only -- never on the provided game.

    python -m tools.external            # score the current config on the three external games
    python -m tools.external --tune     # coordinate-descent tuning on games 1-2, test on game 3

The organisers' clarification (it takes precedence over all other documentation): "If your
algorithm has parameters that need to be tuned, you MUST tune them on external data, never on the
ground-truth possession data or on the provided video." This module is how we do that.

External data: Metrica Sports' public sample games (https://github.com/metrica-sports/sample-data,
"sample data to play around with to the broader football analytics community", source
acknowledged). Three anonymised matches with synchronised tracking and event data. Download them
into data/external/ (gitignored) with:  python -m tools.external --download

Reference possession on those games is derived *mechanically* from their event data -- no human
judgement of ours is involved:
  * possession  = team of the latest on-ball event (pass, carry, recovery, shot, ball lost,
                  set piece, ball out); duels and fouls received do not change it;
  * dead ball   = from the end of the last event before each set piece to that set piece, plus
                  the stretch before each period's kick-off;
  * graded      = from each period's kick-off to its last event.

The sample tracking is cleaner than the provided automatic tracking, so it is degraded first,
deterministically, to the conditions DATA.md publishes for the provided game: the ball hidden in
bursts on ~41% of frames, ~1 m of position noise, ~7% of player observations missing.
"""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import DATA_DIR  # noqa: E402
from possession import atd as atd_io, config as C, features, pipeline  # noqa: E402

EXT = os.path.join(DATA_DIR, "external")
PITCH = (105.0, 68.0)
BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
FILES = {
    1: ["Sample_Game_1/Sample_Game_1_RawEventsData.csv",
        "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
        "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"],
    2: ["Sample_Game_2/Sample_Game_2_RawEventsData.csv",
        "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
        "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv"],
    3: ["Sample_Game_3/Sample_Game_3_events.json",
        "Sample_Game_3/Sample_Game_3_metadata.xml",
        "Sample_Game_3/Sample_Game_3_tracking.txt"],
}
TRAIN, TEST = (1, 2), (3,)
ON_BALL = {"PASS", "CARRY", "RECOVERY", "SHOT", "BALL LOST", "SET PIECE", "BALL OUT"}

# Degradation to the provided game's published conditions (DATA.md).
TARGET_BALL_MISSING = 0.41
BALL_GAP_MEAN = 30          # frames per hidden burst
POSITION_NOISE_M = 0.3      # per axis, players; ball gets 2/3 of it
PLAYER_DROP = 0.07
SEED = 2027


# -- loading -------------------------------------------------------------------------------------
def _events_csv(path):
    e = pd.read_csv(path)
    return pd.DataFrame({"team": e["Team"].str.lower().map({"home": 0, "away": 1}),
                         "type": e["Type"].str.upper(), "subtype": e["Subtype"].fillna(""),
                         "period": e["Period"], "start": e["Start Frame"], "end": e["End Frame"]})


def _events_json(path, home_id):
    rows = []
    for x in json.load(open(path))["data"]:
        team = x.get("team") or {}
        rows.append({"team": 0 if team.get("id") == home_id else 1, "type": x["type"]["name"].upper(),
                     "subtype": (x.get("subtypes") or {}).get("name", "") if isinstance(
                         x.get("subtypes"), dict) else "",
                     "period": x["period"], "start": x["start"]["frame"], "end": x["end"]["frame"]})
    return pd.DataFrame(rows)


def _tracking_csv(home_path, away_path):
    """Metrica classic CSV: 3 header lines, then Period, Frame, Time, x/y pairs..., Ball x/y."""
    def read(path):
        df = pd.read_csv(path, skiprows=2)
        cols = list(df.columns)
        xy = []
        for i, c in enumerate(cols):
            if c.startswith("Player") or c == "Ball":
                xy.append((c, cols[i], cols[i + 1]))
        frame = df["Frame"].values.astype(int)
        players = np.stack([df[[a, b]].values for name, a, b in xy if name != "Ball"], axis=1)
        ball = df[[[a, b] for name, a, b in xy if name == "Ball"][0]].values
        return frame, players.astype(np.float64), ball.astype(np.float64)

    fh, ph, ball = read(home_path)
    fa, pa, _ = read(away_path)
    assert np.array_equal(fh, fa)
    n = int(fh.max()) + 1
    players = np.full((n, ph.shape[1] + pa.shape[1], 2), np.nan)
    players[fh] = np.concatenate([ph, pa], axis=1)
    b = np.full((n, 2), np.nan)
    b[fh] = ball
    team = np.array([0] * ph.shape[1] + [1] * pa.shape[1], dtype=np.int8)
    return players, b, team


def load_game(game):
    paths = [os.path.join(EXT, p) for p in FILES[game]]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise IOError("external data missing, run:  python -m tools.external --download\n  "
                      + "\n  ".join(missing))
    if game in (1, 2):
        events = _events_csv(paths[0])
        players, ball, team = _tracking_csv(paths[1], paths[2])
    else:
        root = ET.parse(paths[1]).getroot()
        home_id = [t.get("id") for t in root.iter("Team")][0]
        events = _events_json(paths[0], home_id)
        n = int(events["end"].max()) + 2000
        with open(paths[2]) as h:
            n = max(n, sum(1 for _ in h) + 2)
        team = atd_io.team_of_each_slot(paths[1])
        # This file changes its column order across 11 DataFormatSpecification segments
        # (substitutions), so columns are mapped to players per frame range. (The provided
        # game's metadata has one segment whose order equals the Player list -- checked.)
        player_ids = [p.get("id") for p in root.iter("Player")]
        channel = {pc.get("id"): pc.get("playerId") for pc in root.iter("PlayerChannel")}
        specs = []
        for s in root.iter("DataFormatSpecification"):
            order = []
            for ref in s.iter("PlayerChannelRef"):
                pid = channel.get(ref.get("playerChannelId"))
                if pid not in order:
                    order.append(pid)
            specs.append((int(s.get("startFrame")), int(s.get("endFrame")),
                          [player_ids.index(p) for p in order]))
        width = max(len(o) for *_, o in specs)
        raw = atd_io.parse_raw(paths[2], n, width)
        # EPTS frame ids count from 1; parse_raw stores id-1; index arrays by frame id like events
        players = np.full((n, len(team), 2), np.nan)
        for first, last, order in specs:
            rows = np.arange(first, min(last, n - 1) + 1)
            players[rows[:, None], np.array(order)[None, :]] = raw["players"][rows - 1, :len(order)]
        ball = np.full((n, 2), np.nan)
        ball[1:] = raw["ball"][:-1]
    return events, players, ball, team


# -- reference labels from events ----------------------------------------------------------------
def reference(events, n):
    """('H','A','D','X') per frame and the graded blocks, derived mechanically from events."""
    labels = np.full(n, "X", dtype="<U1")
    blocks = []
    ev = events.dropna(subset=["start"]).sort_values(["start", "end"]).reset_index(drop=True)
    for period in sorted(ev["period"].unique()):
        e = ev[ev["period"] == period]
        kick = e[(e["type"] == "SET PIECE") & e["subtype"].str.upper().str.contains("KICK OFF")]
        start = int(kick["start"].min()) if len(kick) else int(e["start"].min())
        stop = int(max(e["end"].max(), e["start"].max())) + 1
        stop = min(stop, n)
        blocks.append((start, stop))
        on = e[e["type"].isin(ON_BALL) & e["team"].notna()]
        current = None
        frames = on["start"].astype(int).values
        teams = on["team"].astype(int).values
        for i, (f0, side) in enumerate(zip(frames, teams)):
            f1 = frames[i + 1] if i + 1 < len(frames) else stop
            labels[max(start, f0):max(start, f1)] = "HA"[side]
            current = side
        if current is None:
            continue
        labels[start:stop][labels[start:stop] == "X"] = "D"
        # dead ball before every restart: from the end of the last event before it
        restarts = e[(e["type"] == "SET PIECE")]
        for _, r in restarts.iterrows():
            rs = int(r["start"])
            before = e[(e["start"] < rs)]
            if not len(before):
                continue
            last_end = int(max(before["end"].max(), before["start"].max()))
            if rs - last_end > 0:
                labels[max(start, last_end):rs] = "D"
    return labels, blocks


# -- degradation ---------------------------------------------------------------------------------
def degrade(players, ball, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(ball)
    speed = np.hypot(*np.moveaxis(np.gradient(players * PITCH[0] / PITCH[0] *
                                              np.array(PITCH), axis=0), 2, 0)) * C.FPS
    speed = np.where(np.isnan(speed), np.nan, speed)  # provided ATD carries a clean speed
    p = players + rng.normal(0, POSITION_NOISE_M, players.shape) / np.array(PITCH)
    drop = rng.random(players.shape[:2]) < PLAYER_DROP
    p[drop] = np.nan
    b = ball + rng.normal(0, POSITION_NOISE_M * 2 / 3, ball.shape) / np.array(PITCH)
    hidden = np.zeros(n, bool)
    gap_between = BALL_GAP_MEAN * (1 - TARGET_BALL_MISSING) / TARGET_BALL_MISSING
    t = 0
    while t < n:
        t += int(rng.geometric(1 / gap_between))
        g = int(rng.geometric(1 / BALL_GAP_MEAN))
        hidden[t:t + g] = True
        t += g
    b[hidden] = np.nan
    return p, b, speed


# -- scoring -------------------------------------------------------------------------------------
_CACHE = {}


def prepared(game, degraded=True):
    key = (game, degraded)
    if key not in _CACHE:
        events, players, ball, team = load_game(game)
        n = len(ball)
        ref, blocks = reference(events, n)
        if degraded:
            p, b, s = degrade(players, ball)
        else:
            p, b = players, ball
            s = np.hypot(*np.moveaxis(np.gradient(players * np.array(PITCH), axis=0), 2, 0)) * C.FPS
        data = {"players": p.astype(np.float32), "ball": b.astype(np.float32),
                "speed": s.astype(np.float32), "team": team}
        _CACHE[key] = (data, ref, blocks)
    return _CACHE[key]


_FEATURES = {}
FEATURE_PARAMS = ("BALL_INTERP_MAX_GAP", "BALL_SMOOTH_FRAMES", "ACTIVITY_WINDOW", "OUT_MARGIN_M")


def score(game, degraded=True):
    data, ref, blocks = prepared(game, degraded)
    key = (game, degraded) + tuple(getattr(C, p) for p in FEATURE_PARAMS)
    if key not in _FEATURES:  # features do not depend on the tuned thresholds
        _FEATURES[key] = features.compute(data, PITCH)
    f = _FEATURES[key]
    labels, *_ = pipeline.infer(f, blocks)
    r = pm.evaluate(pm.label_codes(ref), pm.label_codes(labels), bootstrap_n=0)
    return {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
            "poss": r["n_pred_spells"], "poss_ref": r["n_gt_spells"],
            "dead_pred": float((labels[ref != "X"] == "D").mean()),
            "dead_ref": float((ref[ref != "X"] == "D").mean())}


def table(games, degraded=True):
    rows = {g: score(g, degraded) for g in games}
    print("{:<8} {:>8} {:>8} {:>6} {:>6} {:>7} {:>7}".format(
        "game", "spellF1", "macroF1", "poss", "ref", "D%", "refD%"))
    for g, r in rows.items():
        print("{:<8} {:>8.3f} {:>8.3f} {:>6} {:>6} {:>7.1f} {:>7.1f}".format(
            "game {}".format(g), r["spellF1"], r["macroF1"], r["poss"], r["poss_ref"],
            100 * r["dead_pred"], 100 * r["dead_ref"]))
    return rows


# -- tuning --------------------------------------------------------------------------------------
GRID = {
    "CONTROL_RADIUS_M": [1.25, 1.5, 2.0, 2.5, 3.0],
    "CONTROL_MARGIN_M": [0.3, 0.5, 0.7, 1.0],
    "CONTROL_MAX_BALL_SPEED": [6.0, 8.0, 10.0],
    "CONFIRM_HOLD_FRAMES": [8, 12, 15, 20, 25],
    "CONFIRM_SPELLS": [2, 3],
    "MIN_EVIDENCE_FRAMES": [2, 3, 5],
    "MIN_POSSESSION_FRAMES": [12, 25, 38, 50],
    "ACTIVITY_DEAD_REL": [0.4, 0.5, 0.6, 0.7],
    "ACTIVITY_SLOW_REL": [0.7, 0.8, 0.85, 0.95],
    "RESTART_KICK_MS": [3.0, 4.0, 5.0, 6.0],
}


def objective(games):
    return float(np.mean([score(g)["spellF1"] for g in games]))


def tune(rounds=2):
    t0 = time.time()
    best = objective(TRAIN)
    print("start: mean spellF1 on train games {} = {:.4f}".format(TRAIN, best))
    for rnd in range(rounds):
        changed = False
        for name, values in GRID.items():
            original = getattr(C, name)
            results = []
            for v in values:
                setattr(C, name, v)
                results.append((objective(TRAIN), v))
            setattr(C, name, original)
            top, value = max(results, key=lambda r: (round(r[0], 4), -abs(r[1] - original)))
            if top > best + 1e-4 and value != original:
                setattr(C, name, value)
                best, changed = top, True
            print("  [{:5.0f}s] round {} {:<24} -> {:<6} train spellF1 {:.4f}".format(
                time.time() - t0, rnd + 1, name, getattr(C, name), best))
        if not changed:
            break
    print("\ntuned values (copy into possession/config.py):")
    for name in GRID:
        print("  {} = {}".format(name, getattr(C, name)))
    print("\nheld-out test game(s) {}:".format(TEST))
    table(TEST)


def download():
    import urllib.request
    for game, files in FILES.items():
        for rel in files:
            dest = os.path.join(EXT, rel)
            if os.path.isfile(dest):
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            print("downloading", rel)
            urllib.request.urlretrieve("{}/{}".format(BASE_URL, rel), dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--clean", action="store_true", help="score on undegraded tracking too")
    args = parser.parse_args()
    if args.download:
        download()
    if args.tune:
        tune()
    else:
        print("current config, degraded tracking:")
        table(TRAIN + TEST)
        if args.clean:
            print("\ncurrent config, clean tracking:")
            table(TRAIN + TEST, degraded=False)


if __name__ == "__main__":
    main()
