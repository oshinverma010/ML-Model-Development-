"""EDA on the training pool, model comparison and selection by rolling-origin CV. NEVER touches the test years.

Stages (all described in docs/ml_plan.md):
  0  EDA on the pool: base rate by year, missingness audit, correlations, VIF, feature summary
  1  baselines + preprocessing choice (imputation x class weighting, default hyperparameters)
  2  hyperparameter search (RandomizedSearchCV, n_iter <= 30) inside the rolling-origin folds
  3  out-of-fold predictions, model comparison, pre-specified selection rule, paired bootstrap
  4  single decision threshold from pooled out-of-fold predictions, secondary GroupKFold check
  5  feature selection views, ablation, robustness variants, reproducibility check
  6  freeze everything in results/frozen_config.json

Run: python src/evaluate_cv.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")   # inherited by joblib workers (IterativeImputer convergence chatter)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import models as M  # noqa: E402
import metrics as MT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, PROC = ROOT / "results", ROOT / "data" / "processed"
N_JOBS = max(1, min(8, (os.cpu_count() or 2) - 1))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- generic CV helpers
def _fit_predict(model, imputer, cw, params, X, y, tr, va, cols, l1_select=False):
    pipe = M.make_pipeline(model, imputer, cw, params, l1_select=l1_select)
    pipe.fit(X.iloc[tr][cols], y[tr])
    return pipe.predict_proba(X.iloc[va][cols])[:, 1]


def run_folds(model, imputer, cw, params, X, y, folds, cols=F.FEATURES, l1_select=False, n_jobs=N_JOBS):
    """Fit on each training block, predict the validation block. Returns list of probability arrays."""
    return Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict)(model, imputer, cw, params, X, y, tr, va, cols, l1_select) for tr, va in folds)


def fold_table(preds, y, folds) -> pd.DataFrame:
    rows = []
    for k, ((tr, va), p) in enumerate(zip(folds, preds), start=1):
        rows.append({"fold": k, **MT.score_metrics(y[va], p)})
    return pd.DataFrame(rows)


def summarise(ft: pd.DataFrame, prefix: str = "") -> dict:
    n = len(ft)
    out = {f"{prefix}pr_auc_mean": ft["pr_auc"].mean(), f"{prefix}pr_auc_sd": ft["pr_auc"].std(ddof=1)}
    out[f"{prefix}pr_auc_se"] = out[f"{prefix}pr_auc_sd"] / np.sqrt(n)
    for c in ("roc_auc", "brier", "ece"):
        out[f"{prefix}{c}_mean"] = ft[c].mean()
    return out


def choose_simplest_within_se(df: pd.DataFrame, mean_col: str, se_col: str, complexity_key) -> pd.Series:
    """Pre-specified rule: best mean; among rows within one SE of the best choose the simplest."""
    best = df.loc[df[mean_col].idxmax()]
    cand = df[df[mean_col] >= best[mean_col] - best[se_col]].copy()
    cand["_cx"] = cand.apply(complexity_key, axis=1)
    return cand.sort_values(["_cx", mean_col], ascending=[True, False]).iloc[0], best, cand


# --------------------------------------------------------------------------- stage 0: EDA on the pool
def stage0_eda(P: pd.DataFrame) -> None:
    log("stage 0: EDA on the pool")
    F.base_rate_by_year(P).to_csv(RES / "label_base_rate_pool.csv", index=False)
    pd.DataFrame([{"split": "pool", "t_min": int(P.year.min()), "t_max": int(P.year.max()), "n_rows": len(P),
                   "n_countries": P.iso3.nunique(), "n_positive": int(P.y.sum()), "base_rate": P.y.mean()}]
                 ).to_csv(RES / "pool_summary.csv", index=False)

    # missingness by feature, year and income group (pool rows only)
    X = P[F.FEATURES]
    miss_year = (X.isna().groupby(P["year"]).mean().T.rename_axis("feature").reset_index()
                 .melt(id_vars="feature", var_name="year", value_name="share_missing"))
    miss_year.to_csv(RES / "missingness_by_year.csv", index=False)
    miss_inc = (X.isna().groupby(P["income_group"]).mean().T.rename_axis("feature").reset_index()
                .melt(id_vars="feature", var_name="income_group", value_name="share_missing"))
    cnt = P.groupby("income_group").size().rename("n_rows").reset_index()
    miss_inc.merge(cnt, on="income_group").to_csv(RES / "missingness_by_income.csv", index=False)

    rows = []
    for f in F.FEATURES:
        m = X[f].isna()
        if m.sum() == 0:
            rows.append({"feature": f, "n_missing": 0, "share_missing": 0.0})
            continue
        tab = pd.crosstab(m, P["income_group"])
        p_inc = stats.chi2_contingency(tab)[1] if tab.shape[0] == 2 and tab.shape[1] > 1 else np.nan
        pop = np.log(P["population"].where(P["population"] > 0))
        a, b = pop[m].dropna(), pop[~m].dropna()
        p_pop = stats.mannwhitneyu(a, b).pvalue if len(a) > 5 and len(b) > 5 else np.nan
        rows.append({"feature": f, "n_missing": int(m.sum()), "share_missing": float(m.mean()),
                     "p_vs_income_group_chi2": p_inc, "median_ln_pop_missing": a.median(),
                     "median_ln_pop_observed": b.median(), "p_vs_population_mannwhitney": p_pop,
                     "countries_with_any_missing": int(P.loc[m, "iso3"].nunique())})
    pd.DataFrame(rows).to_csv(RES / "missingness_tests.csv", index=False)
    any_missing = X.isna().any(axis=1)
    pd.DataFrame([{"rows_with_any_missing": int(any_missing.sum()), "share_rows_with_any_missing": any_missing.mean(),
                   "rows_complete": int((~any_missing).sum())}]).to_csv(RES / "missingness_rows.csv", index=False)

    # Spearman matrix, near-duplicate filter on the whole pool, VIF
    rho = X.corr(method="spearman")
    rho.rename_axis("feature").reset_index().to_csv(RES / "feature_spearman.csv", index=False)
    flt = M.CorrelationFilter(0.9).fit(X)
    kept = [F.FEATURES[j] for j in flt.keep_]
    rows = []
    for j, name in enumerate(F.FEATURES):
        if j in set(flt.keep_):
            rows.append({"feature": name, "kept": True, "dropped_because_of": "", "abs_rho": np.nan})
        else:
            partners = [(F.FEATURES[k], abs(rho.iloc[j, k])) for k in flt.keep_ if abs(rho.iloc[j, k]) > 0.9]
            top = max(partners, key=lambda t: t[1])
            rows.append({"feature": name, "kept": False, "dropped_because_of": top[0], "abs_rho": top[1]})
    pd.DataFrame(rows).to_csv(RES / "corr_filter_pool.csv", index=False)

    def vif(cols):
        Z = X[cols].fillna(X[cols].median())
        Z = (Z - Z.mean()) / Z.std(ddof=0)
        Z = Z.assign(const=1.0)
        out = []
        for i, c in enumerate(cols):
            try:
                v = variance_inflation_factor(Z.to_numpy(), i)
            except Exception:
                v = np.inf
            out.append(v if np.isfinite(v) and v < 1e6 else np.inf)
        return dict(zip(cols, out))

    v_all, v_kept = vif(F.FEATURES), vif(kept)
    pd.DataFrame({"feature": F.FEATURES, "vif_all_20": [v_all[c] for c in F.FEATURES],
                  "vif_after_filter": [v_kept.get(c, np.nan) for c in F.FEATURES]}).to_csv(RES / "vif.csv", index=False)

    desc = X.describe(percentiles=[0.01, 0.5, 0.99]).T
    desc["share_missing"] = X.isna().mean()
    desc.rename_axis("feature").reset_index().to_csv(RES / "feature_summary_pool.csv", index=False)

    # fold design table
    rows = []
    for k, (tr, va) in enumerate(F.rolling_origin_folds(P), start=1):
        rows.append({"fold": k, "train_years": f"{P.year.iloc[tr].min()}-{P.year.iloc[tr].max()}",
                     "validation_years": f"{P.year.iloc[va].min()}-{P.year.iloc[va].max()}",
                     "n_train": len(tr), "n_validation": len(va), "positives_train": int(P.y.iloc[tr].sum()),
                     "positives_validation": int(P.y.iloc[va].sum()), "base_rate_validation": P.y.iloc[va].mean()})
    pd.DataFrame(rows).to_csv(RES / "cv_folds.csv", index=False)


# --------------------------------------------------------------------------- main
def main() -> None:
    RES.mkdir(exist_ok=True)
    PROC.mkdir(exist_ok=True)
    D0, D1 = F.build_dataset(ffill=False, write_flow=True), F.build_dataset(ffill=True)
    P0, P1 = F.get_pool(D0), F.get_pool(D1)
    assert (P0[["iso3", "year", "y"]].values == P1[["iso3", "year", "y"]].values).all()
    assert P0["year"].max() <= F.pool_end(), "test years must not be used here"
    y = P0["y"].to_numpy()
    folds = F.rolling_origin_folds(P0)
    pool_of = lambda imp: P1 if M.dataset_variant(imp) else P0  # noqa: E731
    log(f"pool rows {len(P0):,}, countries {P0.iso3.nunique()}, base rate {y.mean():.3f}; n_jobs={N_JOBS}")

    stage0_eda(P0)

    # ---------------- stage 1: baselines and preprocessing choice
    log("stage 1: baselines and preprocessing")
    base_rows = []
    for k, (tr, va) in enumerate(folds, start=1):
        prior = M.prior_pred(y[tr], len(va))
        pers = M.persistence_pred(P0.iloc[va])
        for name, p in (("prior", prior), ("persistence", pers)):
            base_rows.append({"model": name, "fold": k, **MT.score_metrics(y[va], p)})
    pd.DataFrame(base_rows).to_csv(RES / "cv_baselines_folds.csv", index=False)

    s1_rows = []
    for m in M.MODELS:
        for imp in M.IMPUTERS[m]:
            for cw in (None, "balanced"):
                P = pool_of(imp)
                preds = run_folds(m, imp, cw, {}, P[F.FEATURES], y, folds)
                ft = fold_table(preds, y, folds)
                s1_rows.append({"model": m, "imputer": imp, "class_weight": cw or "none",
                                **{f"fold{k}_pr_auc": v for k, v in enumerate(ft["pr_auc"], start=1)}, **summarise(ft)})
                log(f"  stage1 {m:4s} {imp:13s} {str(cw):9s} PR-AUC {s1_rows[-1]['pr_auc_mean']:.4f}")
    s1 = pd.DataFrame(s1_rows)
    s1.to_csv(RES / "stage1_preprocessing.csv", index=False)

    chosen_pre, sel_rows = {}, []
    for m in M.MODELS:
        sub = s1[s1["model"] == m].reset_index(drop=True)
        pick, best, cand = choose_simplest_within_se(
            sub, "pr_auc_mean", "pr_auc_se",
            lambda r: (M.IMPUTER_COMPLEXITY[r["imputer"]], M.WEIGHT_COMPLEXITY[None if r["class_weight"] == "none" else "balanced"]))
        chosen_pre[m] = {"imputer": pick["imputer"], "class_weight": None if pick["class_weight"] == "none" else "balanced"}
        sel_rows.append({"model": m, "best_imputer": best["imputer"], "best_class_weight": best["class_weight"],
                         "best_pr_auc": best["pr_auc_mean"], "best_se": best["pr_auc_se"],
                         "n_within_one_se": len(cand), "chosen_imputer": pick["imputer"],
                         "chosen_class_weight": pick["class_weight"], "chosen_pr_auc": pick["pr_auc_mean"]})
    pd.DataFrame(sel_rows).to_csv(RES / "stage1_selection.csv", index=False)
    log(f"  chosen preprocessing: {chosen_pre}")

    # ---------------- stage 2: tuning
    log("stage 2: RandomizedSearchCV")
    tuned, tune_rows = {}, []
    for m in M.MODELS:
        t0 = time.time()
        imp, cw = chosen_pre[m]["imputer"], chosen_pre[m]["class_weight"]
        P = pool_of(imp)
        search = RandomizedSearchCV(M.make_pipeline(m, imp, cw), M.GRIDS[m], n_iter=M.N_ITER, scoring="average_precision",
                                    cv=folds, random_state=M.SEED, n_jobs=N_JOBS, refit=False, error_score="raise")
        search.fit(P[F.FEATURES], y)
        cvr = pd.DataFrame(search.cv_results_)
        cvr = cvr.assign(model=m)
        keep_cols = ["model", "rank_test_score", "mean_test_score", "std_test_score", "params"]
        tune_rows.append(cvr[keep_cols].assign(params=cvr["params"].astype(str)))
        tuned[m] = M.params_from_search(search.best_params_)
        log(f"  {m}: best {tuned[m]} mean PR-AUC {search.best_score_:.4f} ({time.time()-t0:.0f}s)")
    pd.concat(tune_rows).sort_values(["model", "rank_test_score"]).to_csv(RES / "tuning_results.csv", index=False)

    # ---------------- stage 3: OOF predictions, comparison, selection
    log("stage 3: out-of-fold comparison")
    va_all = np.concatenate([va for _, va in folds])
    oof = P0.iloc[va_all][["iso3", "year", "y", "income_group", "region", "small_pop"]].copy()
    oof["fold"] = np.concatenate([np.full(len(va), k) for k, (_, va) in enumerate(folds, start=1)])
    cmp_rows, fold_rows = [], []
    for m in M.MODELS:
        imp, cw = chosen_pre[m]["imputer"], chosen_pre[m]["class_weight"]
        P = pool_of(imp)
        preds = run_folds(m, imp, cw, tuned[m], P[F.FEATURES], y, folds)
        ft = fold_table(preds, y, folds).assign(model=m)
        fold_rows.append(ft)
        oof[f"p_{m}"] = np.concatenate(preds)
        cmp_rows.append({"model": m, "label": M.MODEL_LABELS[m], "imputer": imp, "class_weight": cw or "none",
                         "params": str(tuned[m]), **summarise(ft)})
    for name in ("prior", "persistence"):
        ps = []
        for tr, va in folds:
            ps.append(M.prior_pred(y[tr], len(va)) if name == "prior" else M.persistence_pred(P0.iloc[va]))
        ft = fold_table(ps, y, folds).assign(model=name)
        fold_rows.append(ft)
        oof[f"p_{name}"] = np.concatenate(ps)
        cmp_rows.append({"model": name, "label": {"prior": "Prior (majority) baseline",
                                                  "persistence": "Persistence rule"}[name],
                         "imputer": "", "class_weight": "", "params": "", **summarise(ft)})
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(RES / "cv_model_comparison.csv", index=False)
    pd.concat(fold_rows).to_csv(RES / "cv_model_folds.csv", index=False)
    oof.to_csv(PROC / "oof_predictions.csv", index=False)

    trained = cmp[cmp["model"].isin(M.MODELS)].reset_index(drop=True)
    pick, best, cand = choose_simplest_within_se(trained, "pr_auc_mean", "pr_auc_se",
                                                 lambda r: M.COMPLEXITY[r["model"]])
    chosen = pick["model"]
    pd.DataFrame([{"best_model": best["model"], "best_pr_auc": best["pr_auc_mean"], "best_se": best["pr_auc_se"],
                   "threshold_for_simpler": best["pr_auc_mean"] - best["pr_auc_se"],
                   "models_within_one_se": ",".join(cand["model"]), "chosen_model": chosen,
                   "chosen_pr_auc": pick["pr_auc_mean"], "rule": "highest mean PR-AUC; simpler model within 1 SE wins"}]
                 ).to_csv(RES / "model_selection.csv", index=False)
    log(f"  selected model: {chosen} (best by mean: {best['model']})")

    # paired country-clustered bootstrap of PR-AUC differences on pooled OOF
    log("  paired bootstrap on OOF predictions")
    boot = MT.ClusterBootstrap(oof["iso3"].to_numpy())
    yo = oof["y"].to_numpy()
    cols = [f"p_{m}" for m in M.MODELS] + ["p_persistence"]
    pairs = [(a, b) for a in cols for b in cols if cols.index(a) < cols.index(b)]
    pair_rows = []
    for a, b in pairs:
        pa, pb = oof[a].to_numpy(), oof[b].to_numpy()
        point = MT.safe_ap(yo, pa) - MT.safe_ap(yo, pb)
        lo, hi, draws = boot.ci(lambda ix: MT.safe_ap(yo[ix], pa[ix]) - MT.safe_ap(yo[ix], pb[ix]))
        pair_rows.append({"model_a": a[2:], "model_b": b[2:], "pr_auc_a": MT.safe_ap(yo, pa), "pr_auc_b": MT.safe_ap(yo, pb),
                          "diff": point, "ci_low": lo, "ci_high": hi,
                          "share_boot_diff_le_0": float(np.mean(draws <= 0))})
    pd.DataFrame(pair_rows).to_csv(RES / "cv_paired_differences.csv", index=False)

    # ---------------- stage 4: threshold, pooled OOF metrics, GroupKFold
    log("stage 4: threshold and GroupKFold")
    thresholds = {m: MT.best_f1_threshold(yo, oof[f"p_{m}"]) for m in M.MODELS}
    thr_rows = []
    for m in M.MODELS:
        thr_rows.append({"model": m, "threshold": thresholds[m], **MT.all_metrics(yo, oof[f"p_{m}"], thresholds[m])})
    pers = oof["p_persistence"].to_numpy()
    thr_rows.append({"model": "persistence", "threshold": 0.5, **MT.all_metrics(yo, pers, 0.5)})
    thr_rows.append({"model": "prior", "threshold": 0.5, **MT.all_metrics(yo, oof["p_prior"], 0.5)})
    pd.DataFrame(thr_rows).to_csv(RES / "cv_oof_metrics.csv", index=False)

    gfolds = F.group_folds(P0)
    g_rows = []
    for m in M.MODELS:
        imp, cw = chosen_pre[m]["imputer"], chosen_pre[m]["class_weight"]
        P = pool_of(imp)
        preds = run_folds(m, imp, cw, tuned[m], P[F.FEATURES], y, gfolds)
        ft = fold_table(preds, y, gfolds)
        g_rows.append({"model": m, **summarise(ft)})
    gp = [M.persistence_pred(P0.iloc[va]) for _, va in gfolds]
    g_rows.append({"model": "persistence", **summarise(fold_table(gp, y, gfolds))})
    g_rows.append({"model": "prior", **summarise(fold_table([M.prior_pred(y[tr], len(va)) for tr, va in gfolds], y, gfolds))})
    pd.DataFrame(g_rows).to_csv(RES / "cv_groupkfold.csv", index=False)

    # ---------------- stage 5: feature selection views, ablation, robustness, reproducibility
    log("stage 5: feature selection, ablation, robustness")
    imp, cw = chosen_pre[chosen]["imputer"], chosen_pre[chosen]["class_weight"]
    P = pool_of(imp)
    X = P[F.FEATURES]

    # permutation importance of the chosen model on each validation fold
    def perm_fold(tr, va):
        pipe = M.make_pipeline(chosen, imp, cw, tuned[chosen]).fit(X.iloc[tr], y[tr])
        r = permutation_importance(pipe, X.iloc[va], y[va], scoring="average_precision", n_repeats=10,
                                   random_state=M.SEED, n_jobs=1)
        return r.importances_mean
    pi = np.array(Parallel(n_jobs=N_JOBS)(delayed(perm_fold)(tr, va) for tr, va in folds))
    pd.DataFrame({"feature": F.FEATURES, "importance_mean": pi.mean(0), "importance_sd_over_folds": pi.std(0, ddof=1),
                  "folds_positive": (pi > 0).sum(0)}).sort_values("importance_mean", ascending=False
                                                                  ).to_csv(RES / "permutation_importance_cv.csv", index=False)

    # embedded selection: elastic-net coefficients of the LR deliverable fitted on the whole pool
    lr_imp, lr_cw = chosen_pre["lr"]["imputer"], chosen_pre["lr"]["class_weight"]
    Plr = pool_of(lr_imp)
    lr_pipe = M.make_pipeline("lr", lr_imp, lr_cw, tuned["lr"]).fit(Plr[F.FEATURES], y)
    names = M.pipeline_feature_names(lr_pipe, F.FEATURES)
    coef = lr_pipe.named_steps["clf"].coef_.ravel()
    pd.DataFrame({"feature": names, "coefficient": coef, "zeroed_by_elastic_net": coef == 0}
                 ).to_csv(RES / "lr_coefficients_pool.csv", index=False)

    # ablation: feature sets A, B, C (+ C with L1 selection), fixed tuned hyperparameters, rolling-origin CV
    ab_rows = []
    for m in dict.fromkeys([chosen, "lr"]):
        mi, mc = chosen_pre[m]["imputer"], chosen_pre[m]["class_weight"]
        Pm = pool_of(mi)
        for label, cols_, l1 in (("A", F.FEATURE_SETS["A"], False), ("B", F.FEATURE_SETS["B"], False),
                                 ("C", F.FEATURE_SETS["C"], False), ("C+L1", F.FEATURE_SETS["C"], True)):
            preds = run_folds(m, mi, mc, tuned[m], Pm[F.FEATURES], y, folds, cols=cols_, l1_select=l1)
            ab_rows.append({"model": m, "feature_set": label, "n_features": len(cols_), **summarise(fold_table(preds, y, folds))})
    pd.DataFrame(ab_rows).to_csv(RES / "ablation_cv.csv", index=False)

    # robustness (CV level): (a) exclude small countries from validation rows, (b) horizon h = 5
    rob = []
    keep = ~oof["small_pop"].to_numpy()
    for name in dict.fromkeys([chosen, "lr", "persistence", "prior"]):
        col = f"p_{name}"
        rob.append({"variant": "exclude population < 1 million (OOF subset)", "model": name, "n_rows": int(keep.sum()),
                    "base_rate": yo[keep].mean(), "pr_auc": MT.safe_ap(yo[keep], oof[col].to_numpy()[keep]),
                    "roc_auc": MT.safe_auc(yo[keep], oof[col].to_numpy()[keep])})
        rob.append({"variant": "primary (all OOF rows)", "model": name, "n_rows": len(yo), "base_rate": yo.mean(),
                    "pr_auc": MT.safe_ap(yo, oof[col].to_numpy()), "roc_auc": MT.safe_auc(yo, oof[col].to_numpy())})
    h5 = 5
    nf5 = 4
    Q0, Q1 = F.get_pool(F.build_dataset(h=h5, ffill=False), h5), F.get_pool(F.build_dataset(h=h5, ffill=True), h5)
    y5 = Q0["y"].to_numpy()
    folds5 = F.rolling_origin_folds(Q0, h5, nf5)
    for name in dict.fromkeys([chosen, "lr"]):
        mi, mc = chosen_pre[name]["imputer"], chosen_pre[name]["class_weight"]
        Q = Q1 if M.dataset_variant(mi) else Q0
        preds = run_folds(name, mi, mc, tuned[name], Q[F.FEATURES], y5, folds5)
        ft = fold_table(preds, y5, folds5)
        rob.append({"variant": f"horizon h=5 (rolling-origin, {nf5} folds, 5-year purge)", "model": name, "n_rows": len(Q0),
                    "base_rate": y5.mean(), **{k: v for k, v in summarise(ft).items() if k in ("pr_auc_mean", "roc_auc_mean")},
                    "pr_auc": ft["pr_auc"].mean(), "roc_auc": ft["roc_auc"].mean()})
    ps5 = [M.persistence_pred(Q0.iloc[va]) for _, va in folds5]
    pr5 = [M.prior_pred(y5[tr], len(va)) for tr, va in folds5]
    for name, ps in (("persistence", ps5), ("prior", pr5)):
        ft = fold_table(ps, y5, folds5)
        rob.append({"variant": f"horizon h=5 (rolling-origin, {nf5} folds, 5-year purge)", "model": name, "n_rows": len(Q0),
                    "base_rate": y5.mean(), "pr_auc": ft["pr_auc"].mean(), "roc_auc": ft["roc_auc"].mean()})
    pd.DataFrame(rob).drop(columns=["pr_auc_mean", "roc_auc_mean"], errors="ignore").to_csv(RES / "robustness_cv.csv", index=False)

    # reproducibility: same seed, same predictions
    tr, va = folds[0]
    a = M.make_pipeline(chosen, imp, cw, tuned[chosen]).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    b = M.make_pipeline(chosen, imp, cw, tuned[chosen]).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    pd.DataFrame([{"model": chosen, "max_abs_difference_between_two_fits": float(np.max(np.abs(a - b)))}]
                 ).to_csv(RES / "reproducibility.csv", index=False)

    # ---------------- stage 6: freeze
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    cfg = {
        "frozen_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "git_commit_at_freeze": commit,
        "h": F.H_DEFAULT, "pool_end": F.pool_end(), "test_years": list(F.TEST_YEARS),
        "features": F.FEATURES, "chosen_model": chosen, "baseline_model": "lr",
        "models": {m: {"imputer": chosen_pre[m]["imputer"], "class_weight": chosen_pre[m]["class_weight"],
                       "params": {k: (v.item() if hasattr(v, "item") else v) for k, v in tuned[m].items()},
                       "threshold": thresholds[m]} for m in M.MODELS},
        "threshold_rule": "maximise F1 on pooled out-of-fold rolling-origin predictions of the pool",
        "oof_sha256": hashlib.sha256((PROC / "oof_predictions.csv").read_bytes()).hexdigest(),
    }
    (RES / "frozen_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    log(f"frozen config written; chosen model {chosen}, threshold {thresholds[chosen]:.4f}")


if __name__ == "__main__":
    main()
