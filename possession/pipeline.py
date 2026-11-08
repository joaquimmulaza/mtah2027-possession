"""Provided data in, submission CSV out.

    python -m possession.pipeline --team <team_name>

writes evaluation/results/<team_name>.csv (199,319 rows), deterministically: the same data gives
the same file, byte for byte (RULES.md rule 3). No randomness, no network, no LLM calls.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR, half_video_blocks, require, source_frames  # noqa: E402

from . import atd as atd_io  # noqa: E402
from . import dead_ball, features, states, tagging  # noqa: E402

LETTERS = {states.HOME: "H", states.AWAY: "A"}


def to_letters(dead, live, blocks):
    n = len(dead)
    out = np.full(n, "X", dtype="<U1")
    graded = np.zeros(n, dtype=bool)
    for first, stop in blocks:
        graded[first:stop] = True
    out[graded & (live == states.HOME)] = "H"
    out[graded & (live == states.AWAY)] = "A"
    out[graded & dead] = "D"
    return out


def write_csv(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


def summarise(labels, blocks, fps):
    graded = np.concatenate([labels[a:b] for a, b in blocks])
    counts = {k: int((graded == k).sum()) for k in "HADX"}
    n_poss, durations = 0, []
    for a, b in blocks:
        w = labels[a:b]
        w = w[w != "X"]
        if not len(w):
            continue
        bounds = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1, [len(w)]))
        for i in range(len(bounds) - 1):
            if w[bounds[i]] in "HA":
                n_poss += 1
                durations.append((bounds[i + 1] - bounds[i]) / float(fps))
    return counts, n_poss, (float(np.mean(durations)) if durations else 0.0)


def run(game=GAME, data_dir=None, n_frames=None, blocks=None, verbose=True):
    """Labels array ('H','A','D','X') for every video frame, plus a diagnostics dict."""
    t0 = time.time()
    data_dir = data_dir or DATA_DIR
    meta, raw, st_path = require(["atd_meta", "atd_raw", "smart_tagging"], game, data_dir)
    n_frames = n_frames or source_frames("tactical", game)
    if blocks is None:  # published half boundaries (DATA.md) as half-open [first, stop)
        blocks = [(first, last + 1) for _, first, last in half_video_blocks(game)]
    fps = GAMES[game]["fps"]

    def log(msg):
        if verbose:
            print("[{:6.1f}s] {}".format(time.time() - t0, msg))

    log("reading tracking (first run parses the raw file, later runs use data/cache/)")
    data = atd_io.load(meta, raw, n_frames, cache_dir=os.path.join(data_dir, "cache"))
    home_name, away_name = atd_io.team_names(meta)
    log("teams: H = {}, A = {}; {} track slots".format(home_name, away_name, len(data["team"])))

    f = features.compute(data, GAMES[game]["pitch"])
    log("ball seen on {:.1f}% of frames, {:.1f}% after short-gap interpolation".format(
        100 * f["ball_seen"].mean(), 100 * f["have_ball"].mean()))

    events = tagging.load(st_path, fps)
    set_pieces = tagging.set_pieces(events, home_name, away_name)
    goals = tagging.goals(events)
    phases = tagging.team_phases(events, home_name, away_name)
    log("smart tagging: {} instances, {} set pieces, {} goals".format(
        len(events), len(set_pieces), len(goals)))

    dead, restarts = dead_ball.detect(f, set_pieces, goals, blocks)
    log("dead ball on {:.1f}% of graded frames, {} restarts".format(
        100 * np.concatenate([dead[a:b] for a, b in blocks]).mean(), len(restarts)))

    # Pass 1: tracking only. Pass 2: tagging prior calibrated against pass 1 fills the stretches
    # the tracking could not decide.
    live1, _ = states.label_live(f, dead, restarts)
    prior, report = tagging.calibrated_prior(phases, to_letters(dead, live1, blocks), n_frames)
    live, control_team = states.label_live(f, dead, restarts, prior)
    labels = to_letters(dead, live, blocks)

    counts, n_poss, mean_dur = summarise(labels, blocks, fps)
    log("graded frames: H {H}  A {A}  D {D}  X {X}".format(**counts))
    log("{} possessions, mean {:.1f} s".format(n_poss, mean_dur))
    if verbose and report:
        print("  smart-tagging codes, agreement with tracking-derived possession:")
        for code, (rate, decision) in sorted(report.items()):
            print("    {:<32} {:>6}  {}".format(
                code, "-" if rate is None else "{:.2f}".format(rate), decision))
    return labels, {"counts": counts, "n_possessions": n_poss, "mean_duration_s": mean_dur,
                    "tag_report": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", required=True, help="team name; the file is <team>.csv")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    parser.add_argument("--data-dir", default=None, help="defaults to MTAH_DATA_DIR or ./data")
    args = parser.parse_args()

    labels, _ = run(data_dir=args.data_dir)
    path = write_csv(os.path.join(args.out, "{}.csv".format(args.team)), labels)
    print("\nwrote {}".format(os.path.relpath(path, ROOT)))
    print("next:  python evaluation/validate_submission.py {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
