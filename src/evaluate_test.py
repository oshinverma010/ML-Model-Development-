"""THE single test run. Evaluates the frozen configuration on the held-out test years (t = 2017, 2018, 2019).

* Reads results/frozen_config.json (written by src/evaluate_cv.py): chosen model, preprocessing, hyperparameters and the
  decision threshold. Nothing is chosen here.
* Refuses to run a second time: results/test_run_lock.json is written at the end of the first successful run.
* `--rehearsal` runs exactly the same code on a pseudo-split carved out of the POOL (never the real test years) and writes
  to results/_rehearsal/ (git-ignored); it exists only to debug this script without touching the test labels.

Run: python src/evaluate_test.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import metrics as MT  # noqa: E402
import models as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, PROC = ROOT / "results", ROOT / "data" / "processed"
LOCK = RES / "test_run_lock.json"
SCORE_KEYS = ["pr_auc", "roc_auc", "brier", "ece", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc"]
N_JOBS = max(1, min(8, (os.cpu_count() or 2) - 1))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_models(cfg, train_by_variant, feature_cols=F.FEATURES, l1_select=False, only=None):
    """Fit the configured pipelines on the pool. Returns {model: fitted pipeline}."""
    out = {}
    for m in (only or M.MODELS):
        c = cfg["models"][m]
        T = train_by_variant[M.dataset_variant(c["imputer"])]
        pipe = M.make_pipeline(m, c["imputer"], c["class_weight"], c["params"], l1_select=l1_select)
        out[m] = pipe.fit(T[feature_cols], T["y"].to_numpy())
    return out


def predict(models, test_by_variant, cfg, cols=F.FEATURES):
    preds = {}
    for m, pipe in models.items():
        Te = test_by_variant[M.dataset_variant(cfg["models"][m]["imputer"])]
        preds[m] = pipe.predict_proba(Te[cols])[:, 1]
    return preds


def run_evaluation(cfg: dict, train_v: dict, test_v: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    chosen = cfg["chosen_model"]
    T0, E0 = train_v[False], test_v[False]
    assert (T0["year"].max() + F.H_DEFAULT) < E0["year"].min(), "train labels must end before test features begin"
    y = E0["y"].to_numpy()
    n_test = len(E0)

    # ---------------- fit on the pool, predict the test rows
    log("fitting frozen models on the pool")
    fitted = fit_models(cfg, train_v)
    pred = predict(fitted, test_v, cfg)
    pred["persistence"] = M.persistence_pred(E0)
    pred["prior"] = M.prior_pred(T0["y"].to_numpy(), n_test)
    thr = {m: cfg["models"][m]["threshold"] for m in M.MODELS} | {"persistence": 0.5, "prior": 0.5}
    order = ["lr", "rf", "hgb", "persistence", "prior"]

    tp = E0[["iso3", "year", "y", "income_group", "region", "small_pop"]].copy()
    for m in order:
        tp[f"p_{m}"] = pred[m]
    tp["p_chosen"] = pred[chosen]
    tp.to_csv(PROC / "test_predictions.csv" if out == RES else out / "test_predictions.csv", index=False)

    # ---------------- base rates
    br_test = F.base_rate_by_year(E0).assign(split="test")
    br_pool = F.base_rate_by_year(T0).assign(split="pool")
    pd.concat([br_pool, br_test]).to_csv(out / "label_base_rate_by_year.csv", index=False)
    allp = pd.concat([T0, E0])
    pd.DataFrame([{"scope": "all eligible pool + test rows", "n_rows": len(allp), "n_positive": int(allp.y.sum()),
                   "base_rate": allp.y.mean(), "n_countries": allp.iso3.nunique()},
                  {"scope": "pool", "n_rows": len(T0), "n_positive": int(T0.y.sum()), "base_rate": T0.y.mean(),
                   "n_countries": T0.iso3.nunique()},
                  {"scope": "test", "n_rows": n_test, "n_positive": int(y.sum()), "base_rate": y.mean(),
                   "n_countries": E0.iso3.nunique()}]).to_csv(out / "label_base_rate_overall.csv", index=False)

    # ---------------- headline metrics with country-clustered bootstrap CIs
    log("headline metrics and bootstrap")
    boot = MT.ClusterBootstrap(E0["iso3"].to_numpy())

    def all_models_metrics(ix):
        rec = {}
        for m in order:
            for k, v in MT.all_metrics(y[ix], pred[m][ix], thr[m]).items():
                if k in SCORE_KEYS:
                    rec[f"{m}|{k}"] = v
        return rec
    cis = boot.ci_dict(all_models_metrics)
    rows, conf_rows = [], []
    for m in order:
        point = MT.all_metrics(y, pred[m], thr[m])
        row = {"model": m, "threshold": thr[m], "n": n_test, "n_positive": int(y.sum()), **point}
        for k in SCORE_KEYS:
            row[f"{k}_lo"], row[f"{k}_hi"] = cis[f"{m}|{k}"]
        rows.append(row)
        tn, fp, fn, tpc = point["tn"], point["fp"], point["fn"], point["tp"]
        conf_rows.append({"model": m, "tn": tn, "fp": fp, "fn": fn, "tp": tpc,
                          "tn_rate": tn / max(tn + fp, 1), "fp_rate": fp / max(tn + fp, 1),
                          "fn_rate": fn / max(fn + tpc, 1), "tp_rate": tpc / max(fn + tpc, 1)})
    pd.DataFrame(rows).to_csv(out / "test_metrics.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(out / "test_confusion.csv", index=False)

    # curves and calibration (data for the figures)
    roc_rows, pr_rows, cal_rows = [], [], []
    for m in order:
        fpr, tpr, _ = roc_curve(y, pred[m])
        roc_rows.append(pd.DataFrame({"model": m, "fpr": fpr, "tpr": tpr}))
        pr, rc, _ = precision_recall_curve(y, pred[m])
        pr_rows.append(pd.DataFrame({"model": m, "precision": pr, "recall": rc}))
        if m in ("lr", "rf", "hgb"):
            cal_rows.append(MT.reliability_table(y, np.clip(pred[m], 0, 1)).assign(model=m))
    pd.concat(roc_rows).to_csv(out / "test_curve_roc.csv", index=False)
    pd.concat(pr_rows).to_csv(out / "test_curve_pr.csv", index=False)
    pd.concat(cal_rows).to_csv(out / "test_calibration.csv", index=False)

    # paired differences (country-clustered bootstrap)
    pair_rows = []
    for a, b in ((chosen, "persistence"), (chosen, "prior"), (chosen, "lr"), ("lr", "persistence"), ("hgb", "rf")):
        if a == b:
            continue
        for metric, fn in (("pr_auc", MT.safe_ap), ("roc_auc", MT.safe_auc)):
            point = fn(y, pred[a]) - fn(y, pred[b])
            lo, hi, d = boot.ci(lambda ix: fn(y[ix], pred[a][ix]) - fn(y[ix], pred[b][ix]))
            pair_rows.append({"model_a": a, "model_b": b, "metric": metric, "diff": point, "ci_low": lo, "ci_high": hi,
                              "share_boot_diff_le_0": float(np.mean(d <= 0))})
    pd.DataFrame(pair_rows).to_csv(out / "test_paired_differences.csv", index=False)

    # ---------------- subgroups
    log("subgroups")
    sub_rows = []
    for var in ("income_group", "region"):
        for g, idx in E0.groupby(var).indices.items():
            yy = y[idx]
            row = {"grouping": var, "group": g, "n": len(idx), "n_countries": E0.iloc[idx]["iso3"].nunique(),
                   "n_positive": int(yy.sum()), "base_rate": yy.mean()}
            for m in (chosen, "lr", "persistence"):
                pm = pred[m][idx]
                row[f"{m}_pr_auc"] = MT.safe_ap(yy, pm)
                row[f"{m}_roc_auc"] = MT.safe_auc(yy, pm)
                tm = MT.threshold_metrics(yy, pm, thr[m])
                row[f"{m}_precision"], row[f"{m}_recall"] = tm["precision"], tm["recall"]
            if len(idx) >= 10 and row["n_countries"] >= 3 and 0 < yy.sum() < len(yy):
                b = MT.ClusterBootstrap(E0.iloc[idx]["iso3"].to_numpy())
                for m in (chosen, "lr"):
                    pm = pred[m][idx]
                    lo, hi, _ = b.ci(lambda ix: MT.safe_ap(yy[ix], pm[ix]))
                    row[f"{m}_pr_auc_lo"], row[f"{m}_pr_auc_hi"] = lo, hi
            sub_rows.append(row)
    pd.DataFrame(sub_rows).to_csv(out / "test_subgroups.csv", index=False)

    # ---------------- robustness subsets of the single run
    rob_rows = []
    variants = {"primary (t = 2017-2019)": np.ones(n_test, bool),
                "exclude t = 2017 (window ends 2020)": (E0["year"] != 2017).to_numpy(),
                "exclude population < 1 million": (~E0["small_pop"]).to_numpy()}
    for name, mask in variants.items():
        idx = np.flatnonzero(mask)
        b = MT.ClusterBootstrap(E0.iloc[idx]["iso3"].to_numpy())
        for m in (chosen, "lr", "persistence", "prior"):
            ym, pm = y[idx], pred[m][idx]
            lo, hi, _ = b.ci(lambda ix: MT.safe_ap(ym[ix], pm[ix]))
            tm = MT.threshold_metrics(ym, pm, thr[m])
            rob_rows.append({"variant": name, "model": m, "n": len(idx), "n_positive": int(ym.sum()),
                             "base_rate": ym.mean(), "pr_auc": MT.safe_ap(ym, pm), "pr_auc_lo": lo, "pr_auc_hi": hi,
                             "roc_auc": MT.safe_auc(ym, pm), "f1": tm["f1"], "mcc": tm["mcc"]})
    pd.DataFrame(rob_rows).to_csv(out / "test_robustness.csv", index=False)

    # ---------------- ablation on the test set (frozen hyperparameters; threshold-free metrics)
    log("ablation")
    ab_rows = []
    for m in dict.fromkeys([chosen, "lr"]):
        for label, cols, l1 in (("A", F.FEATURE_SETS["A"], False), ("B", F.FEATURE_SETS["B"], False),
                                ("C", F.FEATURE_SETS["C"], False), ("C+L1", F.FEATURE_SETS["C"], True)):
            f = fit_models(cfg, train_v, cols, l1, only=[m])
            p = predict(f, test_v, cfg, cols)[m]
            lo, hi, _ = boot.ci(lambda ix: MT.safe_ap(y[ix], p[ix]))
            ab_rows.append({"model": m, "feature_set": label, "n_features": len(cols), "pr_auc": MT.safe_ap(y, p),
                            "pr_auc_lo": lo, "pr_auc_hi": hi, "roc_auc": MT.safe_auc(y, p)})
    pd.DataFrame(ab_rows).to_csv(out / "test_ablation.csv", index=False)

    # ---------------- interpretation
    log("interpretation")
    Te = test_v[M.dataset_variant(cfg["models"][chosen]["imputer"])]
    r = permutation_importance(fitted[chosen], Te[F.FEATURES], y, scoring="average_precision", n_repeats=20,
                               random_state=M.SEED, n_jobs=N_JOBS)
    pd.DataFrame({"feature": F.FEATURES, "importance_mean": r.importances_mean, "importance_sd": r.importances_std}
                 ).sort_values("importance_mean", ascending=False).to_csv(out / "test_permutation_importance.csv", index=False)

    cv_imp = pd.read_csv(RES / "permutation_importance_cv.csv")          # top 3 chosen from CV (no test labels)
    top3 = cv_imp["feature"].head(3).tolist()
    Tc = train_v[M.dataset_variant(cfg["models"][chosen]["imputer"])]
    pdp_rows = []
    for f in top3:
        pdp = partial_dependence(fitted[chosen], Tc[F.FEATURES], [f], kind="average", grid_resolution=20,
                                 percentiles=(0.05, 0.95), response_method="predict_proba")
        for v, a in zip(pdp["grid_values"][0], pdp["average"][0]):
            pdp_rows.append({"feature": f, "value": float(v), "partial_dependence": float(a)})
    pd.DataFrame(pdp_rows).to_csv(out / "pdp_top3.csv", index=False)

    lr_c = fitted["lr"]
    names = M.pipeline_feature_names(lr_c, F.FEATURES)
    coef = lr_c.named_steps["clf"].coef_.ravel()
    pd.DataFrame({"feature": names, "standardised_coefficient": coef, "zeroed": coef == 0}
                 ).to_csv(out / "lr_standardised_coefficients.csv", index=False)
    return {"n_test": n_test, "n_positive": int(y.sum())}


def datasets(rehearsal: bool):
    D = {False: F.build_dataset(ffill=False), True: F.build_dataset(ffill=True)}
    if rehearsal:   # pseudo-split inside the pool: train t <= 2005, pseudo-test t in 2009-2011 (3-year purge)
        train = {k: v[v["year"] <= 2005].reset_index(drop=True) for k, v in D.items()}
        test = {k: v[v["year"].isin([2009, 2010, 2011])].reset_index(drop=True) for k, v in D.items()}
    else:
        train = {k: F.get_pool(v) for k, v in D.items()}
        test = {k: F.get_test(v) for k, v in D.items()}
    return train, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearsal", action="store_true", help="debug run on a pseudo-split of the pool (not the test years)")
    args = ap.parse_args()
    cfg_path = RES / "frozen_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    oof_hash = hashlib.sha256((PROC / "oof_predictions.csv").read_bytes()).hexdigest()
    assert oof_hash == cfg["oof_sha256"], "OOF predictions changed after the freeze"

    if args.rehearsal:
        out = RES / "_rehearsal"
        train, test = datasets(True)
        info = run_evaluation(cfg, train, test, out)
        log(f"rehearsal finished: {info}")
        return

    if LOCK.exists():
        sys.exit("results/test_run_lock.json exists: the test set has already been evaluated once. Refusing to run again.")
    train, test = datasets(False)
    assert set(test[False]["year"]) == set(F.TEST_YEARS)
    log(f"TEST RUN: frozen config sha256 {hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:12]}..., "
        f"chosen model {cfg['chosen_model']}")
    info = run_evaluation(cfg, train, test, RES)
    LOCK.write_text(json.dumps({
        "run_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        "n_test_rows": info["n_test"], "n_test_positive": info["n_positive"],
        "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
        "note": "The test set was evaluated once. Re-running is blocked by the existence of this file.",
    }, indent=2), encoding="utf-8")
    log("test run complete; lock written")


if __name__ == "__main__":
    main()
