"""End-to-end check on a synthetic match:  python -m tests.test_pipeline

Asserts the file contract, determinism, and that the pipeline beats the reference baseline on a
match whose answer we know. Uses no real match data.
"""

import hashlib
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from possession import pipeline  # noqa: E402
from starter import baseline  # noqa: E402
from tests.synthetic import Match  # noqa: E402


def score(gt_path, labels):
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    return pm.evaluate(gt, pm.label_codes(labels), bootstrap_n=0)


def main():
    tmp = tempfile.mkdtemp(prefix="mtah_synth_")
    try:
        match = Match().simulate()
        gt_path = match.write(tmp)
        n = match.n

        labels, info = pipeline.run(data_dir=tmp, n_frames=n, blocks=match.blocks)
        assert len(labels) == n and set(np.unique(labels)) <= set("HADX")

        out = pipeline.write_csv(os.path.join(tmp, "team.csv"), labels)
        frame = pm.read_label_csv(out)
        assert pm.validate_submission(frame, n) is None, pm.validate_submission(frame, n)
        digest = hashlib.sha256(open(out, "rb").read()).hexdigest()

        # determinism: second run, from the cache this time, must give the identical file
        labels2, _ = pipeline.run(data_dir=tmp, n_frames=n, blocks=match.blocks, verbose=False)
        out2 = pipeline.write_csv(os.path.join(tmp, "team2.csv"), labels2)
        assert hashlib.sha256(open(out2, "rb").read()).hexdigest() == digest, "not deterministic"

        # the reference baseline on the same synthetic match
        tactical = os.path.join(tmp, "tactical")
        codes = baseline.nearest_player_labels(
            os.path.join(tactical, "BAR-ATH_tactical_atd.txt"),
            baseline.team_of_each_track(os.path.join(tactical, "BAR-ATH_tactical_atd.xml")),
            n, (105.3, 68.0))
        base = baseline.smooth(codes, 50)
        graded = np.zeros(n, bool)
        for a, b in match.blocks:
            graded[a:b] = True
        base[~graded] = "X"

        ours, ref = score(gt_path, labels), score(gt_path, base)
        print("\n{:<22} {:>8} {:>8} {:>6} {:>6}".format("synthetic match", "macroF1", "spellF1",
                                                      "poss", "gt"))
        for name, r in (("baseline_smooth50", ref), ("possession pipeline", ours)):
            print("{:<22} {:>8.4f} {:>8.4f} {:>6} {:>6}".format(
                name, r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"], r["n_gt_spells"]))
        assert ours["spell_f1_mean"] > ref["spell_f1_mean"], "pipeline does not beat baseline"
        print("\nOK: file contract, determinism, beats baseline on the synthetic match")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
