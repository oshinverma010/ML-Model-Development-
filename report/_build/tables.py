"""Markdown tables for the report, all generated from results/*.csv (or from the code constants they describe).

A cell never contains a vertical bar: md_table() raises if one is present.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
import features as F  # noqa: E402
import models as M  # noqa: E402

MNAME = {"lr": "Elastic-net logistic regression", "rf": "Random forest", "hgb": "Gradient boosting (HGB)",
         "persistence": "Persistence rule", "prior": "Prior (majority) baseline"}


def rd(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / name).drop_duplicates().reset_index(drop=True)


def md_table(headers: list[str], rows: list[list], align: list[str] | None = None) -> str:
    def cell(x) -> str:
        s = str(x)
        if "|" in s:
            raise ValueError(f"vertical bar inside table cell: {s!r}")
        return s.replace("\n", " ")
    align = align or ["l"] + ["r"] * (len(headers) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    out = ["| " + " | ".join(cell(h) for h in headers) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f(x, d=3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def ci(x, lo, hi, d=3) -> str:
    return f"{f(x, d)} [{f(lo, d)}, {f(hi, d)}]"


def sci(x) -> str:
    return "n/a" if x is None or pd.isna(x) else "{:.1e}".format(x)


def pct(x, d=1) -> str:
    return f"{100 * x:.{d}f}%"


# --------------------------------------------------------------------------- feature tables
FEATURE_TEXT = {  # definition, rationale
    "ln_gdp_pc": ("ln GDP per capita (constant 2015 US$)", "Richer economies have more room for structural change."),
    "gdp_g1": ("1-year change in ln GDP per capita", "Decoupling needs growth to continue; recent growth is persistent."),
    "gdp_cagr3": ("3-year average annual change in ln GDP per capita", "Medium-run growth trend."),
    "gdp_vol5": ("standard deviation of the last five 1-year growth rates", "Unstable economies rarely decouple steadily."),
    "ln_pop": ("ln population", "Scale of the country; weak prior."),
    "pop_g3": ("3-year average annual change in ln population", "Fast population growth raises energy demand."),
    "urban_pct": ("urban population share (%)", "Stage of development."),
    "industry_pct": ("industry value added (% of GDP)", "Industry-heavy economies are harder to decarbonise."),
    "ln_co2_pc": ("ln CO2 per capita", "A high emitter has more to cut."),
    "co2_chg1": ("1-year change in ln CO2 per capita", "Emission momentum: recent falls tend to persist."),
    "co2_chg3": ("3-year change in ln CO2 per capita", "Medium-run emission momentum."),
    "co2_chg5": ("5-year change in ln CO2 per capita", "Long-run emission momentum."),
    "ln_co2_per_gdp": ("ln of CO2 per unit of GDP", "Carbon intensity of output (Kaya term)."),
    "ci_chg3": ("3-year change in ln CO2 per unit of GDP", "Carbon-intensity momentum."),
    "ln_energy_pc": ("ln primary energy per capita (OWID)", "Energy use level."),
    "ln_energy_per_gdp": ("ln energy per unit of GDP (OWID)", "Energy intensity of output (Kaya term)."),
    "co2_per_unit_energy": ("CO2 per unit of primary energy (g/kWh, OWID)", "Carbon intensity of energy (Kaya term)."),
    "coal_share": ("coal CO2 divided by total CO2", "Coal-heavy systems decouple less easily."),
    "oil_share": ("oil CO2 divided by total CO2", "Transport-heavy fuel mix; direction unclear."),
    "gas_share": ("gas CO2 divided by total CO2", "Gas often replaces coal."),
}


def feature_catalogue(V) -> str:
    rows = [[n, F.FEATURE_META[n][0], FEATURE_TEXT[n][0], FEATURE_TEXT[n][1], F.FEATURE_META[n][1]] for n in F.FEATURES]
    return md_table(["Feature", "Group", "Definition", "Why it matters", "Expected sign"], rows,
                    ["l", "l", "l", "l", "l"])


def feature_stats(V) -> str:
    d = rd("feature_summary_pool.csv").set_index("feature")
    rows = [[n, F.FEATURE_META[n][0], f(d.loc[n, "50%"], 3), f(d.loc[n, "1%"], 3), f(d.loc[n, "99%"], 3),
             pct(d.loc[n, "share_missing"])] for n in F.FEATURES]
    return md_table(["Feature", "Group", "Median", "1st percentile", "99th percentile", "Missing in pool"], rows,
                    ["l", "l", "r", "r", "r", "r"])


def feature_sets(V) -> str:
    rows = [[k, len(v), ", ".join(f"`{c}`" for c in v)] for k, v in F.FEATURE_SETS.items()]
    return md_table(["Set", "Features", "Members"], rows, ["l", "r", "l"])


def vif_table(V) -> str:
    d = rd("vif.csv")
    rows = []
    for _, r in d.iterrows():
        a = "infinite" if np.isinf(r["vif_all_20"]) else f(r["vif_all_20"], 1)
        b = "dropped by filter" if np.isnan(r["vif_after_filter"]) else ("infinite" if np.isinf(r["vif_after_filter"]) else f(r["vif_after_filter"], 1))
        rows.append([r["feature"], a, b])
    return md_table(["Feature", "VIF, all 20 features", "VIF after the near-duplicate filter"], rows)


def missingness(V) -> str:
    d = rd("missingness_tests.csv")
    d = d[d["n_missing"] > 0].sort_values("share_missing", ascending=False)
    rows = []
    for _, r in d.iterrows():
        rows.append([r["feature"], pct(r["share_missing"]), int(r["countries_with_any_missing"]),
                     sci(r["p_vs_income_group_chi2"]),
                     "n/a" if pd.isna(r["median_ln_pop_missing"]) else f"{r['median_ln_pop_missing']:.1f} vs {r['median_ln_pop_observed']:.1f}",
                     sci(r["p_vs_population_mannwhitney"])])
    return md_table(["Feature", "Missing in pool", "Countries affected", "p, missing vs income group (chi-square)",
                     "Median ln population, missing vs observed", "p, population (Mann-Whitney)"], rows)


# --------------------------------------------------------------------------- data tables
def sample_flow(V) -> str:
    d = rd("sample_flow.csv")
    return md_table(["Stage", "Rows remaining", "Rows dropped"],
                    [[r["stage"], f"{int(r['n_remaining']):,}", f"{int(r['n_dropped']):,}"] for _, r in d.iterrows()])


def cleaning_log(V) -> str:
    d = rd("cleaning_log.csv")
    return md_table(["Step", "Rule", "Rows before", "Rows after", "Note"],
                    [[r["step"], r["rule"], f"{int(r['rows_before']):,}", f"{int(r['rows_after']):,}",
                      "" if pd.isna(r["note"]) else r["note"]] for _, r in d.iterrows()], ["l", "l", "r", "r", "l"])


def cv_folds(V) -> str:
    d = rd("cv_folds.csv")
    return md_table(["Fold", "Training years t", "Validation years t", "Training rows", "Validation rows",
                     "Positives in validation", "Validation base rate"],
                    [[int(r["fold"]), r["train_years"], r["validation_years"], f"{int(r['n_train']):,}",
                      f"{int(r['n_validation']):,}", int(r["positives_validation"]), pct(r["base_rate_validation"])]
                     for _, r in d.iterrows()])


# --------------------------------------------------------------------------- algorithms
def algo_comparison(V) -> str:
    rows = [
        ["Elastic-net logistic regression", "Interpretable coefficients; convex; fast; L1 part selects features, L2 part handles correlated features; probabilities are well-behaved",
         "Linear in the features, so interactions and thresholds are missed; needs scaling and imputation", "Linear log-odds; independent rows (violated here, handled by clustered resampling)", "Trained; kept as the interpretable model"],
        ["Random forest", "Captures non-linearity and interactions; robust to outliers and scale; few settings to tune", "Probabilities are less calibrated; less interpretable; cannot extrapolate", "None on functional form; assumes rows are exchangeable", "Trained"],
        ["Histogram gradient boosting (HGB)", "Strong on tabular data; native NaN handling; regularisation options", "More hyper-parameters; sensitive to noise when signal is weak; sequential, harder to explain", "None on functional form", "Trained"],
        ["Support vector machine (RBF)", "Flexible margin classifier", "Needs scaling; probabilities need extra calibration; training cost grows quickly with rows; no native missing handling; kernel choice hard to interpret", "Smooth decision boundary", "Not trained"],
        ["k-nearest neighbours", "Simple, non-parametric", "Distances are dominated by a few scales in 20 mixed features; rows of one country are near-duplicates, so neighbours leak across time; no feature importance", "Meaningful distance", "Not trained"],
        ["Neural network (MLP)", "Flexible function approximator", "About four thousand rows of noisy tabular data is too small to justify it; extra tuning and instability; weak interpretability", "Smooth function; scaled inputs", "Not trained"],
    ]
    return md_table(["Algorithm", "Strengths", "Limitations", "Assumptions", "Status in this project"], rows,
                    ["l", "l", "l", "l", "l"])


def grids(V) -> str:
    rows = []
    for m in M.MODELS:
        for k, v in M.GRIDS[m].items():
            rows.append([MNAME[m], k.split("__")[1], ", ".join("None" if x is None else str(x) for x in v)])
    return md_table(["Model", "Hyper-parameter", "Values in the search space"], rows, ["l", "l", "l"])


def tuning_best(V) -> str:
    cmp = rd("cv_model_comparison.csv").set_index("model")
    rows = []
    for m in M.MODELS:
        rows.append([MNAME[m], cmp.loc[m, "imputer"], cmp.loc[m, "class_weight"], ", ".join(f"{k} = {v}" for k, v in ast.literal_eval(cmp.loc[m, "params"]).items())])
    return md_table(["Model", "Imputation", "Class weight", "Tuned hyper-parameters (stage 2)"], rows, ["l", "l", "l", "l"])


def stage1(V) -> str:
    d = rd("stage1_preprocessing.csv")
    nm = {"median": "median + indicators", "ffill_median": "forward fill + median", "iterative": "iterative (BayesianRidge)", "native": "native NaN"}
    rows = [[MNAME[r["model"]], nm[r["imputer"]], r["class_weight"], f"{r['pr_auc_mean']:.3f}", f"{r['pr_auc_se']:.3f}",
             f"{r['roc_auc_mean']:.3f}", f"{r['brier_mean']:.3f}"] for _, r in d.iterrows()]
    return md_table(["Model", "Imputation", "Class weight", "Mean PR-AUC", "SE", "Mean ROC-AUC", "Mean Brier"], rows,
                    ["l", "l", "l", "r", "r", "r", "r"])


def cv_comparison(V) -> str:
    d = rd("cv_model_comparison.csv")
    order = ["lr", "rf", "hgb", "persistence", "prior"]
    d = d.set_index("model").loc[order]
    rows = [[MNAME[m], f"{r['pr_auc_mean']:.3f}", f"{r['pr_auc_se']:.3f}", f"{r['roc_auc_mean']:.3f}",
             "n/a" if m == "persistence" else f"{r['brier_mean']:.3f}", "n/a" if m == "persistence" else f"{r['ece_mean']:.3f}"]
            for m, r in d.iterrows()]
    return md_table(["Model", "Mean PR-AUC", "SE", "Mean ROC-AUC", "Mean Brier", "Mean ECE"], rows)


def cv_paired(V) -> str:
    d = rd("cv_paired_differences.csv")
    rows = [[f"{MNAME[r['model_a']]} minus {MNAME[r['model_b']]}", f"{r['diff']:+.3f}", f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"]
            for _, r in d.iterrows()]
    return md_table(["Pooled OOF PR-AUC difference", "Difference", "95% country-clustered CI"], rows)


def groupkfold(V) -> str:
    d = rd("cv_groupkfold.csv").set_index("model")
    return md_table(["Model", "Mean PR-AUC, grouped by country", "SE", "Mean ROC-AUC"],
                    [[MNAME[m], f"{r['pr_auc_mean']:.3f}", f"{r['pr_auc_se']:.3f}", f"{r['roc_auc_mean']:.3f}"] for m, r in d.iterrows()])


# --------------------------------------------------------------------------- test tables
ORDER = ["lr", "rf", "hgb", "persistence", "prior"]


def test_scores(V) -> str:
    d = rd("test_metrics.csv").set_index("model").loc[ORDER]
    rows = []
    for m, r in d.iterrows():
        prob = m not in ("persistence",)
        rows.append([MNAME[m], ci(r["pr_auc"], r["pr_auc_lo"], r["pr_auc_hi"]), ci(r["roc_auc"], r["roc_auc_lo"], r["roc_auc_hi"]),
                     ci(r["brier"], r["brier_lo"], r["brier_hi"]) if prob else "n/a",
                     ci(r["ece"], r["ece_lo"], r["ece_hi"]) if prob else "n/a"])
    return md_table(["Model", "PR-AUC [95% CI]", "ROC-AUC [95% CI]", "Brier score [95% CI]", "ECE [95% CI]"], rows)


def test_thresholded(V) -> str:
    d = rd("test_metrics.csv").set_index("model").loc[ORDER]
    rows = []
    for m, r in d.iterrows():
        rows.append([MNAME[m], f(r["threshold"], 3), ci(r["f1"], r["f1_lo"], r["f1_hi"]), ci(r["mcc"], r["mcc_lo"], r["mcc_hi"]),
                     ci(r["precision"], r["precision_lo"], r["precision_hi"]), ci(r["recall"], r["recall_lo"], r["recall_hi"]),
                     ci(r["balanced_accuracy"], r["balanced_accuracy_lo"], r["balanced_accuracy_hi"]),
                     ci(r["accuracy"], r["accuracy_lo"], r["accuracy_hi"])])
    return md_table(["Model", "Threshold", "F1 [95% CI]", "MCC [95% CI]", "Precision [95% CI]", "Recall [95% CI]",
                     "Balanced accuracy [95% CI]", "Accuracy [95% CI]"], rows)


def test_confusion(V) -> str:
    d = rd("test_confusion.csv").set_index("model").loc[ORDER]
    rows = [[MNAME[m], int(r["tn"]), int(r["fp"]), int(r["fn"]), int(r["tp"]), pct(r["tn_rate"], 0), pct(r["fp_rate"], 0),
             pct(r["fn_rate"], 0), pct(r["tp_rate"], 0)] for m, r in d.iterrows()]
    return md_table(["Model", "TN", "FP", "FN", "TP", "TN rate", "FP rate", "FN rate", "TP rate"], rows)


def test_paired(V) -> str:
    d = rd("test_paired_differences.csv")
    rows = [[f"{MNAME[r['model_a']]} minus {MNAME[r['model_b']]}", r["metric"].replace("_", "-").upper(), f"{r['diff']:+.3f}",
             f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"] for _, r in d.iterrows()]
    return md_table(["Comparison on the test set", "Metric", "Difference", "95% country-clustered CI"], rows,
                    ["l", "l", "r", "r"])


def subgroups(V) -> str:
    d = rd("test_subgroups.csv")
    cfgc = json.loads((RES / "frozen_config.json").read_text(encoding="utf-8"))["chosen_model"]
    rows = []
    for g in ("income_group", "region"):
        for _, r in d[d["grouping"] == g].iterrows():
            lo, hi = r.get(f"{cfgc}_pr_auc_lo"), r.get(f"{cfgc}_pr_auc_hi")
            c = ci(r[f"{cfgc}_pr_auc"], lo, hi, 2) if pd.notna(lo) else f(r[f"{cfgc}_pr_auc"], 2)
            rows.append([r["group"], int(r["n"]), int(r["n_countries"]), int(r["n_positive"]), pct(r["base_rate"], 0), c,
                         f(r["persistence_pr_auc"], 2), f(r[f"{cfgc}_roc_auc"], 2)])
    return md_table(["Group", "Rows", "Countries", "Positives", "Prevalence", f"{MNAME[cfgc]} PR-AUC [95% CI]",
                     "Persistence PR-AUC", f"{MNAME[cfgc]} ROC-AUC"], rows, ["l", "r", "r", "r", "r", "r", "r", "r"])


def test_robust(V) -> str:
    d = rd("test_robustness.csv")
    rows = [[r["variant"], MNAME[r["model"]], int(r["n"]), int(r["n_positive"]), ci(r["pr_auc"], r["pr_auc_lo"], r["pr_auc_hi"], 3),
             f(r["roc_auc"], 3), f(r["f1"], 3), f(r["mcc"], 3)] for _, r in d.iterrows()]
    return md_table(["Variant", "Model", "Rows", "Positives", "PR-AUC [95% CI]", "ROC-AUC", "F1", "MCC"], rows,
                    ["l", "l", "r", "r", "r", "r", "r", "r"])


def robust_cv(V) -> str:
    d = rd("robustness_cv.csv")
    rows = [[r["variant"], MNAME[r["model"]], f"{int(r['n_rows']):,}", pct(r["base_rate"]), f(r["pr_auc"]), f(r["roc_auc"])]
            for _, r in d.iterrows()]
    return md_table(["Variant", "Model", "Rows", "Base rate", "PR-AUC", "ROC-AUC"], rows, ["l", "l", "r", "r", "r", "r"])


def ablation(V) -> str:
    cv, te = rd("ablation_cv.csv"), rd("test_ablation.csv")
    rows = []
    for _, r in cv.iterrows():
        t = te[(te["model"] == r["model"]) & (te["feature_set"] == r["feature_set"])].iloc[0]
        rows.append([MNAME[r["model"]], r["feature_set"], int(r["n_features"]), f"{r['pr_auc_mean']:.3f} ({r['pr_auc_se']:.3f})",
                     ci(t["pr_auc"], t["pr_auc_lo"], t["pr_auc_hi"])])
    return md_table(["Model", "Feature set", "Features", "CV mean PR-AUC (SE)", "Test PR-AUC [95% CI]"], rows,
                    ["l", "l", "r", "r", "r"])


def posthoc(V) -> str:
    d = rd("test_posthoc.csv")
    nm = dict(MNAME, group_prior="Income-group prior (pool base rate of the group)")
    rows = [[f"{nm[r['model_a']]} minus {nm[r['model_b']]}", r["metric"].replace("_", "-").upper(), f(r["value_a"]), f(r["value_b"]),
             f"{r['diff']:+.3f}", f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"] for _, r in d.iterrows()]
    return md_table(["Comparison (post hoc)", "Metric", "Model A", "Model B", "Difference", "95% country-clustered CI"], rows,
                    ["l", "l", "r", "r", "r", "r"])


def base_rate_year(V) -> str:
    d = rd("label_base_rate_by_year.csv")
    rows = [[int(r["year"]), r["split"], int(r["n_rows"]), int(r["n_positive"]), pct(r["base_rate"])] for _, r in d.iterrows()]
    return md_table(["Window start year t", "Split", "Rows", "Positives", "Base rate"], rows, ["r", "l", "r", "r", "r"])


def importance(V) -> str:
    cvp = rd("permutation_importance_cv.csv").head(8)
    tp = rd("test_permutation_importance.csv").set_index("feature")
    rows = [[r["feature"], f"{r['importance_mean']:.3f} ({r['importance_sd_over_folds']:.3f})", int(r["folds_positive"]),
             f"{tp.loc[r['feature'], 'importance_mean']:.3f} ({tp.loc[r['feature'], 'importance_sd']:.3f})"] for _, r in cvp.iterrows()]
    return md_table(["Feature (top 8 in CV)", "CV importance (SD over folds)", "Folds with positive importance of 5",
                     "Test importance (SD over 20 shuffles)"], rows)


def coefficients(V) -> str:
    d = rd("lr_standardised_coefficients.csv")
    d = d[~d["feature"].str.startswith("missingindicator")]
    d = d.reindex(d["standardised_coefficient"].abs().sort_values(ascending=False).index)
    def verdict(r):
        exp = F.FEATURE_META.get(r["feature"], ("", "?"))[1]
        if exp not in ("+", "-"):
            return "no prior"
        if r["standardised_coefficient"] == 0:
            return "zeroed"
        return "agrees" if (exp == "+") == (r["standardised_coefficient"] > 0) else "opposite"
    rows = [[r["feature"], f"{r['standardised_coefficient']:+.3f}", F.FEATURE_META[r["feature"]][1], verdict(r)]
            for _, r in d.iterrows()]
    return md_table(["Feature", "Standardised coefficient", "Expected sign (plan)", "Observed versus expected"], rows,
                    ["l", "r", "l", "l"])


# --------------------------------------------------------------------------- metrics, tests, deviations, references
def metric_definitions(V) -> str:
    rows = [
        ["PR-AUC (average precision)", "Area under the precision-recall curve; mean precision weighted by the increase in recall", "Selection metric: focuses on the positive (decoupled) class, so abundant true negatives cannot flatter it"],
        ["ROC-AUC", "Probability that a random positive window is scored above a random negative one", "Threshold-free ranking quality; reported next to PR-AUC"],
        ["Accuracy", "Share of correct predictions", "Reported but flagged as misleading when about three quarters of rows are negatives"],
        ["Balanced accuracy", "Mean of recall and specificity", "Not driven by class sizes"],
        ["Precision", "True positives divided by predicted positives", "How many alarms are correct"],
        ["Recall", "True positives divided by actual positives", "How many decoupling windows are found"],
        ["F1", "Harmonic mean of precision and recall", "Used only for the single threshold rule"],
        ["MCC", "Correlation between predicted and actual classes, from the whole confusion matrix", "Balanced summary that is not inflated by imbalance"],
        ["Brier score", "Mean squared error of the predicted probability", "Quality of the probabilities themselves"],
        ["ECE", "Weighted mean gap between predicted probability and observed frequency over 10 equal-width bins", "Calibration summary; shown with a reliability curve"],
    ]
    return md_table(["Metric", "Definition", "Why it is used here"], rows, ["l", "l", "l"])


def tests_list(V) -> str:
    src = (ROOT / "tests" / "test_pipeline_integrity.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    rows = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_"):
            num = n.name.split("_")[1]
            desc = " ".join(n.name.split("_")[2:]).capitalize()
            rows.append([num, desc, (ast.get_docstring(n) or "").split("\n")[0]])
    return md_table(["No.", "Check", "Detail"], rows, ["r", "l", "l"])


def deviations(V) -> str:
    text = (ROOT / "docs" / "ml_plan.md").read_text(encoding="utf-8")
    dev = text.split("## Deviations")[1]
    items = [re.match(r"(\d+)\. (.*)", ln) for ln in dev.splitlines()]
    rows = [[m.group(1), m.group(2).replace("|", "/")] for m in items if m]
    return md_table(["No.", "Entry appended to docs/ml_plan.md (2026-11-05)"], rows, ["r", "l"])


def references() -> str:
    d = rd("reference_check.csv")
    d = d[d["status"] == "VERIFIED"].copy()
    d["first"] = d["authors"].str.split(";").str[0].str.split(",").str[0]
    d = d.sort_values(["first", "year"])
    out = []
    for _, r in d.iterrows():
        auth = [a.strip() for a in str(r["authors"]).split(";")]
        names = []
        for a in auth:
            fam, _, giv = a.partition(",")
            ini = " ".join(p[0] + "." for p in re.split(r"[ \-]+", giv.strip()) if p)
            names.append(f"{fam.strip()}, {ini}".strip().rstrip(","))
        author_str = ", ".join(names[:-1]) + (", & " if len(names) > 1 else "") + names[-1] if len(names) > 1 else names[0]
        def clean(x):
            x = str(x).strip()
            return int(float(x)) if re.fullmatch(r"\d+(\.0)?", x) else x
        vol = f", {clean(r['volume'])}" if pd.notna(r["volume"]) and str(r["volume"]) != "" else ""
        pages = f", {clean(r['pages'])}" if pd.notna(r["pages"]) and str(r["pages"]) != "" else ""
        title = re.sub(r"\s+", " ", str(r["title"]).strip())
        title = title if title.endswith(("?", "!", ".")) else title + "."
        container = re.sub(r"\s+", " ", str(r["container"]))
        out.append(f"- {author_str} ({int(r['year'])}). {title} *{container}*{vol}{pages}. https://doi.org/{r['crossref_doi']}")
    return "\n".join(out)


REGISTRY = {
    "feature_catalogue": feature_catalogue, "feature_stats": feature_stats, "feature_sets": feature_sets, "vif": vif_table,
    "missingness": missingness, "sample_flow": sample_flow, "cleaning_log": cleaning_log, "cv_folds": cv_folds,
    "algo_comparison": algo_comparison, "grids": grids, "tuning_best": tuning_best, "stage1": stage1,
    "cv_comparison": cv_comparison, "cv_paired": cv_paired, "groupkfold": groupkfold, "test_scores": test_scores,
    "test_thresholded": test_thresholded, "test_confusion": test_confusion, "test_paired": test_paired,
    "subgroups": subgroups, "test_robust": test_robust, "robust_cv": robust_cv, "ablation": ablation,
    "base_rate_year": base_rate_year, "posthoc": posthoc, "importance": importance, "coefficients": coefficients,
    "metric_definitions": metric_definitions, "tests_list": tests_list, "deviations": deviations,
}


def make(name: str, V: dict) -> str:
    if name in REGISTRY:
        return REGISTRY[name](V)
    from risk import risk_register  # written after the results exist
    if name == "risk_register":
        return risk_register(V)
    raise KeyError(f"unknown table: {name}")
