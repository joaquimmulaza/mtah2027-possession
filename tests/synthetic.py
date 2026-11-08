"""A synthetic match in the exact file formats of the real bundle, with its own ground truth.

It exists to test the *mechanics* of the pipeline -- parsing, alignment, the dead-ball detector,
the state machine, the file contract -- without touching the real data (which may not be sent to
any cloud service, see FAQ.md). Scores on it say nothing about scores on the real match.

Simulated: dribbles, passes, interceptions, deflections that do not change possession, balls going
out followed by a walk-paced stoppage and a restart by the other team, untagged fouls, ~40% of
frames with the ball invisible in bursts, tracking noise, unused track slots, a misspelt team name
in the tagging, and phase tags that name the defending team rather than the attacking one.
"""

import os

import numpy as np

L, W = 105.3, 68.0
FPS = 25
HOME_NAME, AWAY_NAME = "Home Club", "Away Athletic"
TAG_NAMES = ("Home Club", "Away Atletic")   # misspelt on purpose, as in the real file


class Match:
    def __init__(self, n_frames=40000, blocks=((2000, 20000), (24000, 40000)),
                 kickoffs=(2000, 25000), seed=7):
        self.rng = np.random.default_rng(seed)
        self.n = n_frames
        self.blocks = list(blocks)
        self.kickoffs = list(kickoffs)
        self.n_slots = 30
        self.team = np.array([0] * 11 + [1] * 11 + [0] * 4 + [1] * 4)
        self.players = np.full((self.n, self.n_slots, 2), np.nan)
        self.speed = np.full((self.n, self.n_slots), np.nan)
        self.ball = np.full((self.n, 2), np.nan)
        self.gt = np.full(self.n, "X", dtype="<U1")
        self.tags = []  # (start_frame, stop_frame, code, {group: text})
        self.pos = self.rng.uniform([8, 6], [L - 8, W - 6], (22, 2))
        self.vel = np.zeros((22, 2))
        self.t = 0

    # -- low-level frame writer ----------------------------------------------------------------
    def step(self, ball_xy, label, pace):
        if self.t >= self.n:
            return False
        self.vel += self.rng.normal(0, 0.6, self.vel.shape)
        norm = np.linalg.norm(self.vel, axis=1, keepdims=True) + 1e-9
        self.vel = self.vel / norm * np.minimum(norm, pace * 1.6)
        self.vel = self.vel * (pace / max(np.linalg.norm(self.vel, axis=1).mean(), 1e-6))
        self.pos = np.clip(self.pos + self.vel / FPS, [1, 1], [L - 1, W - 1])
        self.players[self.t, :22] = self.pos + self.rng.normal(0, 0.25, (22, 2))
        self.speed[self.t, :22] = np.linalg.norm(self.vel, axis=1)
        self.ball[self.t] = np.asarray(ball_xy) + self.rng.normal(0, 0.1, 2)
        self.gt[self.t] = label
        self.t += 1
        return True

    def tag(self, start, stop, code, **groups):
        self.tags.append((start, stop, code, groups))

    def carrier_of(self, side):
        idx = np.flatnonzero(self.team[:22] == side)
        return int(self.rng.choice(idx))

    # -- play ----------------------------------------------------------------------------------
    def dribble(self, carrier, side):
        for _ in range(int(self.rng.integers(25, 80))):
            ball = self.pos[carrier] + np.array([0.5, 0.0])
            if not self.step(ball, "HA"[side], 2.6):
                return

    def travel(self, start, end, label, speed=15.0, deflect=None):
        dist = np.linalg.norm(end - start)
        frames = max(3, int(dist / speed * FPS))
        for i in range(1, frames + 1):
            p = start + (end - start) * i / frames
            if deflect is not None and abs(i - frames // 2) <= 1:
                self.pos[deflect] = p + np.array([0.6, 0.0])  # opponent gets a toe on it
            if not self.step(p, label, 2.6):
                return

    def stoppage(self, ball_xy, frames):
        for _ in range(frames):
            if not self.step(ball_xy, "D", 0.8):
                return

    def play_block(self, first, stop, kickoff, side):
        while self.t < first:
            self.t += 1
        self.stoppage(np.array([L / 2, W / 2]), kickoff - first)
        self.tag(kickoff - 60, kickoff + 65, "SET PIECES", Team=TAG_NAMES[side],
                 Type="Kick Off (Start)", Half="1st Half" if first == self.blocks[0][0]
                 else "2nd Half")
        carrier = self.carrier_of(side)
        self.pos[carrier] = [L / 2 - 0.5, W / 2]
        poss_start = self.t
        while self.t < stop:
            self.dribble(carrier, side)
            r = self.rng.random()
            here = self.pos[carrier] + np.array([0.5, 0.0])
            if r < 0.07 or r < 0.12:
                out = r < 0.07
                if out:  # ball runs over the touchline
                    target = np.array([here[0], -2.0 if here[1] < W / 2 else W + 2.0])
                    self.travel(here, target, "HA"[side])
                    rest = np.array([target[0], -1.0 if target[1] < 0 else W + 1.0])
                else:    # foul: ball stays where it is
                    rest = here
                self._phase_tags(poss_start, self.t, side)
                dead_frames = int(self.rng.integers(150, 375))
                self.stoppage(rest, dead_frames)
                side = 1 - side
                carrier = self.carrier_of(side)
                self.pos[carrier] = rest - np.array([0.5, 0.0])
                kick = self.t
                if out:
                    self.tag(kick - 60, kick + 65, "SET PIECES", Team="N/A", Type="Throw-in")
                elif self.rng.random() < 0.3:
                    self.tag(kick - 60, kick + 65, "SET PIECES", Team=TAG_NAMES[side],
                             Type="Free Kick")
                poss_start = self.t
                mate = self._mate(carrier, side)
                self.travel(rest, self.pos[mate] + np.array([0.5, 0]), "HA"[side])
                carrier = mate
                continue
            intercept = self.rng.random() < 0.17
            receiver = self.carrier_of(1 - side) if intercept else self._mate(carrier, side)
            deflect = None
            if not intercept and self.rng.random() < 0.06:
                deflect = self.carrier_of(1 - side)
            self.travel(here, self.pos[receiver] + np.array([0.5, 0]), "HA"[side],
                        deflect=deflect)
            if intercept:
                self._phase_tags(poss_start, self.t, side)
                side = 1 - side
                poss_start = self.t
            carrier = receiver

    def _mate(self, carrier, side):
        while True:
            m = self.carrier_of(side)
            if m != carrier:
                return m

    def _phase_tags(self, start, stop, side):
        if stop - start > 100 and self.rng.random() < 0.6:
            self.tag(start + 10, stop - 10, "BUILD UP", Team=TAG_NAMES[side])
        if stop - start > 100 and self.rng.random() < 0.4:
            self.tag(start + 10, stop - 10, "DEFENDING IN MIDDLE THIRD", Team=TAG_NAMES[1 - side])

    def simulate(self):
        for (first, stop), kickoff, side in zip(self.blocks, self.kickoffs, (0, 1)):
            self.play_block(first, stop, kickoff, side)
            self.t = stop
        graded = np.zeros(self.n, bool)
        for a, b in self.blocks:
            graded[a:b] = True
        self.gt[~graded] = "X"
        self.players[~graded] = np.nan
        self.ball[~graded] = np.nan
        # the tracker loses the ball in bursts, ~40% of the time
        hidden = np.zeros(self.n, bool)
        t = 0
        while t < self.n:
            t += int(self.rng.geometric(1 / 45.0))
            gap = int(self.rng.geometric(1 / 30.0))
            hidden[t:t + gap] = True
            t += gap
        self.ball[hidden] = np.nan
        return self

    # -- writers -------------------------------------------------------------------------------
    def write(self, data_dir):
        folder = os.path.join(data_dir, "tactical")
        os.makedirs(folder, exist_ok=True)
        stem = os.path.join(folder, "BAR-ATH_tactical")
        with open(stem + "_atd.xml", "w") as h:
            h.write('<?xml version="1.0"?>\n<MainMeta><Metadata><Teams>'
                    '<Team id="T1"><Name>{}</Name></Team><Team id="T2"><Name>{}</Name></Team>'
                    '</Teams><Players>'.format(HOME_NAME, AWAY_NAME))
            for s in range(self.n_slots):
                h.write('<Player id="P{0}" teamId="T{1}"><Name>Track {0}</Name>'
                        '<ShirtNumber>{0}</ShirtNumber></Player>'.format(s + 1, self.team[s] + 1))
            h.write("</Players></Metadata></MainMeta>\n")

        def g(x, y, s):
            if np.isnan(x):
                return "NaN,NaN,NaN"
            return "{:.5f},{:.5f},{:.2f}".format(x / L, y / W, s)

        with open(stem + "_atd.txt", "w") as h:
            for t in range(self.n):
                ps = ";".join(g(self.players[t, s, 0], self.players[t, s, 1],
                                self.speed[t, s] if not np.isnan(self.speed[t, s]) else 0)
                              for s in range(self.n_slots))
                b = self.ball[t]
                h.write("{}:{}:{}\n".format(t + 1, ps, g(b[0], b[1], 0.0)))

        with open(stem + "_smart_tagging.xml", "w") as h:
            h.write("<file><ALL_INSTANCES>\n")
            for i, (a, b, code, groups) in enumerate(sorted(self.tags)):
                labels = "".join("<label><group>{}</group><text>{}</text></label>".format(k, v)
                                 for k, v in groups.items())
                h.write("<instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end>"
                        "<code>{}</code>{}</instance>\n".format(i, a / FPS, b / FPS, code, labels))
            h.write("</ALL_INSTANCES></file>\n")

        gt_path = os.path.join(data_dir, "synthetic_gt.csv")
        with open(gt_path, "w") as h:
            h.write("frame,label\n")
            h.writelines("{},{}\n".format(i, x) for i, x in enumerate(self.gt))
        return gt_path
