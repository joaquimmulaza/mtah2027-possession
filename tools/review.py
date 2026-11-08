"""A review pack for a football expert, built from the tracking and our labels. No video needed.

    python -m tools.review

writes, inside data/review/ (gitignored -- it contains derived tracking and must never be
published or uploaded to any cloud service or AI tool):

  * relatorio.txt  -- the match as our system sees it, in plain Portuguese
  * replay.html    -- animated top-down replays of sampled moments, with our label on screen

The review is for DEBUGGING, as RULES.md allows ("watching the video and inspecting the data to
understand them, debug your pipeline and design features"). The expert describes patterns in
words; there is deliberately no right/wrong tally, because human judgements on this game may not
be used to validate or tune the system. A pattern becomes a general rule in the code, never a
correction tied to a specific moment.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAMES, GAME  # noqa: E402
from possession import pipeline  # noqa: E402

FPS = 25
BEFORE, AFTER = 8 * FPS, 6 * FPS
N_CHANGES, N_STOPS, N_RESTARTS = 16, 7, 7
SEED = 2027


def kickoffs(labels, blocks):
    """First live frame of each graded block -- our own estimate of each kickoff."""
    out = []
    for a, b in blocks:
        live = np.flatnonzero(np.isin(labels[a:b], ["H", "A"]))
        out.append(a + int(live[0]) if len(live) else a)
    return out


def clock(frame, blocks, ko):
    """'1ª parte 23:14' style match clock for a video frame."""
    for i, (a, b) in enumerate(blocks):
        if a <= frame < b:
            s = max(0, (frame - ko[i]) / float(FPS)) + 45 * 60 * i
            return "{}ª parte {:02d}:{:02d}".format(i + 1, int(s // 60), int(s % 60))
    return "fora do jogo"


def runs(labels, a, b):
    w = labels[a:b]
    bounds = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1, [len(w)]))
    return [(a + int(bounds[i]), a + int(bounds[i + 1]), w[bounds[i]])
            for i in range(len(bounds) - 1)]


def report(labels, blocks, names):
    home, away = names
    ko = kickoffs(labels, blocks)
    lines = ["RELATÓRIO DO JOGO, como o nosso sistema o vê", "=" * 46, "",
             "H = {} (casa)   A = {} (fora)".format(home, away), ""]

    def share(a, b):
        w = labels[a:b]
        h, x = int((w == "H").sum()), int((w == "A").sum())
        return (100.0 * h / (h + x), 100.0 * x / (h + x)) if h + x else (0.0, 0.0)

    h_all = int(sum((labels[a:b] == "H").sum() for a, b in blocks))
    a_all = int(sum((labels[a:b] == "A").sum() for a, b in blocks))
    lines.append("POSSE DE BOLA (só tempo com a bola em jogo)")
    lines.append("  jogo inteiro:   {} {:.0f}%  |  {} {:.0f}%".format(
        home, 100.0 * h_all / max(1, h_all + a_all), away, 100.0 * a_all / max(1, h_all + a_all)))
    for i, (a, b) in enumerate(blocks):
        sh, sa = share(ko[i], b)
        lines.append("  {}ª parte:       {} {:.0f}%  |  {} {:.0f}%".format(i + 1, home, sh, away, sa))
    lines.append("  por períodos de 15 minutos:")
    for i, (a, b) in enumerate(blocks):
        for q in range(0, 60, 15):
            s, e = ko[i] + q * 60 * FPS, min(b, ko[i] + (q + 15) * 60 * FPS)
            if s >= b:
                break
            sh, sa = share(s, e)
            m0 = q + 45 * i
            label = ("{}'-{}'".format(m0, m0 + 15) if q < 45
                     else "{}'+ desc.".format(45 * (i + 1)))
            lines.append("    {:>8}   {} {:.0f}%  |  {} {:.0f}%".format(label, home, sh, away, sa))
    lines.append("")

    poss, stops = [], []
    for a, b in blocks:
        for s, e, k in runs(labels, a, b):
            (poss if k in "HA" else stops if k == "D" else []).append((s, e, k))
    dur = np.array([(e - s) / float(FPS) for s, e, _ in poss]) if poss else np.zeros(1)
    lines.append("POSSES")
    lines.append("  número de posses:        {}  ({} {}, {} {})".format(
        len(poss), sum(k == "H" for *_, k in poss), home, sum(k == "A" for *_, k in poss), away))
    lines.append("  duração média:           {:.1f} s   (mediana {:.1f} s)".format(
        dur.mean(), float(np.median(dur))))
    lines.append("  posses de menos de 3 s:  {}".format(int((dur < 3).sum())))
    for s, e, k in sorted(poss, key=lambda p: p[0] - p[1])[:3]:
        lines.append("  uma das mais longas:     {} {:.0f} s, a partir de {}".format(
            home if k == "H" else away, (e - s) / float(FPS), clock(s, blocks, ko)))
    lines.append("")

    live = sum(int(np.isin(labels[ko[i]:b], ["H", "A"]).sum()) for i, (a, b) in enumerate(blocks))
    total = sum(b - ko[i] for i, (a, b) in enumerate(blocks))
    sdur = np.array([(e - s) / float(FPS) for s, e, _ in stops]) if stops else np.zeros(1)
    lines.append("BOLA PARADA")
    lines.append("  tempo útil de jogo:      {:.0f} min de {:.0f} min ({:.0f}%)".format(
        live / FPS / 60, total / FPS / 60, 100.0 * live / max(1, total)))
    lines.append("  número de paragens:      {}".format(len(stops)))
    lines.append("  duração média:           {:.0f} s".format(sdur.mean()))
    lines.append("")
    lines.append("PERGUNTAS PARA QUEM PERCEBE DE FUTEBOL (responder por palavras, sem números)")
    lines.append("  1. A divisão da posse faz sentido para este jogo e estas equipas?")
    lines.append("  2. ~{:.0f} posses num jogo e ~{:.0f} s por posse parece-te normal?".format(
        len(poss), dur.mean()))
    lines.append("  3. ~{:.0f}% de tempo útil parece-te normal na LaLiga?".format(
        100.0 * live / max(1, total)))
    lines.append("  4. Os períodos em que uma equipa domina parecem-te plausíveis?")
    return "\n".join(lines) + "\n"


def pick_moments(labels, blocks, rng):
    changes, stops, restarts = [], [], []
    for a, b in blocks:
        w = labels[a:b]
        for i in (np.flatnonzero(w[1:] != w[:-1]) + 1).tolist():
            old, new, t = w[i - 1], w[i], a + i
            if t - BEFORE < a or t + AFTER > b:
                continue
            if old in "HA" and new in "HA":
                changes.append(t)
            elif old in "HA" and new == "D":
                stops.append(t)
            elif old == "D" and new in "HA":
                restarts.append(t)

    def take(xs, k):
        return sorted(rng.choice(xs, size=min(k, len(xs)), replace=False).tolist()) if xs else []

    out = [(t, "mudança de posse") for t in take(changes, N_CHANGES)]
    out += [(t, "a bola fica parada") for t in take(stops, N_STOPS)]
    out += [(t, "o jogo recomeça") for t in take(restarts, N_RESTARTS)]
    return sorted(out)


def clip(f, team, labels, t):
    a, b = t - BEFORE, t + AFTER
    players, ball = [], []
    for k in range(a, b):
        p = f["players"][k]
        vis = ~np.isnan(p[:, 0])
        players.append([[round(float(x), 1), round(float(y), 1), int(s)]
                        for x, y, s in zip(p[vis, 0], p[vis, 1], team[vis])])
        bx, by = f["ball"][k]
        ball.append(None if np.isnan(bx) else [round(float(bx), 1), round(float(by), 1)])
    return {"players": players, "ball": ball, "labels": "".join(labels[a:b]), "event": BEFORE}


def build(data_dir=None, n_frames=None, blocks=None, out_dir=None):
    labels, info = pipeline.run(data_dir=data_dir, n_frames=n_frames, blocks=blocks,
                                verbose=False)
    blocks, names = info["blocks"], info["names"]
    out_dir = out_dir or os.path.join(data_dir or DATA_DIR, "review")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "relatorio.txt"), "w") as h:
        h.write(report(labels, blocks, names))

    ko = kickoffs(labels, blocks)
    moments = pick_moments(labels, blocks, np.random.default_rng(SEED))
    clips = []
    for t, kind in moments:
        c = clip(info["features"], info["team"], labels, t)
        c.update({"kind": kind, "clock": clock(t, blocks, ko)})
        clips.append(c)
    pitch = GAMES[GAME]["pitch"]
    payload = json.dumps({"clips": clips, "names": names, "pitch": pitch, "fps": FPS},
                         separators=(",", ":"))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_template.html")) as h:
        html = h.read().replace("/*DATA*/null", payload)
    with open(os.path.join(out_dir, "replay.html"), "w") as h:
        h.write(html)
    return out_dir, len(clips)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    out_dir, n = build(data_dir=args.data_dir)
    print("pronto: {} momentos".format(n))
    print("  abre no Finder:  {}".format(os.path.join(out_dir, "replay.html")))
    print("  relatório:       {}".format(os.path.join(out_dir, "relatorio.txt")))
    print("NÃO partilhes estes ficheiros fora da equipa nem os envies para nenhuma IA.")


if __name__ == "__main__":
    main()
