"""POST-HOC analysis of the saved test predictions (exploratory; added after the single test run, see plan Deviations).

It does not fit any model, does not touch the frozen configuration and does not re-evaluate anything: it reads
data/processed/test_predictions.csv (written by the one test run) and computes
  1. paired country-clustered bootstrap differences LR vs RF and LR vs HGB (the test script skipped them because the
     chosen model was itself LR, which made the pair (chosen, lr) degenerate);
  2. an income-group prior: every test row is scored with the pool base rate of its World Bank income group
     (a reference to show how much of the skill is "knowing the income group"; income group is NOT a model feature).
Output: results/test_posthoc.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import metrics as MT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, PROC = ROOT / "results", ROOT / "data" / "processed"


def main() -> None:
    tp = pd.read_csv(PROC / "test_predictions.csv")
    y = tp["y"].to_numpy()
    pool = F.get_pool(F.build_dataset())
    group_rate = pool.groupby("income_group")["y"].mean()
    tp["p_group_prior"] = tp["income_group"].map(group_rate).fillna(pool["y"].mean())
    boot = MT.ClusterBootstrap(tp["iso3"].to_numpy())
    rows = []
    for a, b in (("lr", "rf"), ("lr", "hgb"), ("lr", "group_prior"), ("rf", "group_prior"), ("persistence", "group_prior")):
        for metric, fn in (("pr_auc", MT.safe_ap), ("roc_auc", MT.safe_auc)):
            pa, pb = tp[f"p_{a}"].to_numpy(), tp[f"p_{b}"].to_numpy()
            lo, hi, d = boot.ci(lambda ix: fn(y[ix], pa[ix]) - fn(y[ix], pb[ix]))
            rows.append({"model_a": a, "model_b": b, "metric": metric, "value_a": fn(y, pa), "value_b": fn(y, pb),
                         "diff": fn(y, pa) - fn(y, pb), "ci_low": lo, "ci_high": hi, "share_boot_diff_le_0": float(np.mean(d <= 0))})
    pd.DataFrame(rows).to_csv(RES / "test_posthoc.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
