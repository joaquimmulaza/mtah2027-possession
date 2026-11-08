"""How much does the answer move when each threshold moves by +/-25%?   python -m tools.stability

No ground truth is involved: every variant is compared with the default run of the pipeline.
A robust method changes little; a parameter that moves the result a lot is one whose value
matters and deserves a better-founded choice (never one fitted to the public 5 minutes).

Printed: possessions, dead-ball share, frame agreement with the default run and spellF1 of the
variant *against the default run* (1.0 = identical possession structure).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from possession import config as C, pipeline  # noqa: E402

PARAMS = ("CONTROL_RADIUS_M", "CONTROL_MARGIN_M", "CONFIRM_HOLD_FRAMES", "MIN_POSSESSION_FRAMES",
          "ACTIVITY_DEAD_REL", "ACTIVITY_SLOW_REL", "RESTART_KICK_MS", "BALL_INTERP_MAX_GAP")
FACTORS = (0.75, 1.25)


def describe(labels, blocks):
    graded = np.concatenate([labels[a:b] for a, b in blocks])
    _, n_poss, mean = pipeline.summarise(labels, blocks, C.FPS)
    return n_poss, mean, 100.0 * (graded == "D").mean()


def main(data_dir=None, n_frames=None, blocks=None):
    base, info = pipeline.run(data_dir=data_dir, n_frames=n_frames, blocks=blocks, verbose=False)
    blocks = info["blocks"]
    graded = np.zeros(len(base), bool)
    for a, b in blocks:
        graded[a:b] = True
    base_codes = pm.label_codes(base)
    runs = [(a, b) for a, b in blocks]
    n0, m0, d0 = describe(base, blocks)
    print("{:<24} {:>6} {:>7} {:>6} {:>7} {:>8}".format(
        "variant", "poss", "dur_s", "D%", "agree%", "spellF1"))
    print("{:<24} {:>6} {:>7.1f} {:>6.1f} {:>7} {:>8}".format("default", n0, m0, d0, "-", "-"))
    for name in PARAMS:
        original = getattr(C, name)
        for factor in FACTORS:
            value = original * factor
            if isinstance(original, int):
                value = int(round(value))
            setattr(C, name, value)
            try:
                labels, _ = pipeline.run(data_dir=data_dir, n_frames=n_frames, blocks=blocks,
                                         verbose=False)
            finally:
                setattr(C, name, original)
            n, m, d = describe(labels, blocks)
            agree = 100.0 * (labels[graded] == base[graded]).mean()
            spell = pm.spell_scores(base_codes, pm.label_codes(labels), runs)["mean_f1"]
            print("{:<24} {:>6} {:>7.1f} {:>6.1f} {:>7.1f} {:>8.3f}".format(
                "{} x{:.2f}".format(name[:17], factor), n, m, d, agree, spell))


if __name__ == "__main__":
    main()
