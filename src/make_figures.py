"""Make all 14 report figures from results/*.csv (PNG, 200 dpi, colour-blind-safe Okabe-Ito palette).

Every figure is landscape or squarish (height <= 0.8 x width) and readable at about 16 cm wide.
Run after src/evaluate_cv.py and src/evaluate_test.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, PROC, FIG = ROOT / "results", ROOT / "data" / "processed", ROOT / "figures"

# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, VERM, SKY, PURPLE, YELLOW, GREY, INK = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#F0E442", "#8C8C8C", "#222222")
MCOL = {"lr": BLUE, "rf": GREEN, "hgb": ORANGE, "persistence": VERM, "prior": GREY}
MNAME = {"lr": "Elastic-net LR", "rf": "Random forest", "hgb": "HGB", "persistence": "Persistence rule", "prior": "Prior (base rate)"}
W = 6.4  # inches, about 16 cm

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#555555", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": "#E3E3E3", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name: str) -> None:
    h, w = fig.get_size_inches()[1], fig.get_size_inches()[0]
    assert h <= 0.8 * w + 1e-6, f"{name}: height {h:.2f} exceeds 0.8 x width {w:.2f}"
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("saved", name)


def cfg() -> dict:
    return json.loads((RES / "frozen_config.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- 1 pipeline flowchart
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(W, 3.7))
    ax.set_xlim(-1, 102); ax.set_ylim(0, 58); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, text, fc="#F4F7FB", ec=BLUE, fs=7.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color="#555555", lw=1.0))

    y1, y2, h = 38, 18, 15
    bw = 21
    xs = [1, 27, 53, 79]
    box(xs[0], y1, bw, h, "1. Acquire\nOWID CO$_2$ + World\nBank (cached, sha256)", fs=6.2)
    box(xs[1], y1, bw, h, "2. Clean and join\nno aggregates, ISO3\n+ year, Kosovo, logs", fs=6.2)
    box(xs[2], y1, bw, h, "3. Label, features\ny from t and t+3;\nX from t and earlier", fs=6.2)
    box(xs[3], y1, bw, h, "4. Time-aware split\npool t $\\leq$ 2013, purge\n2014-16, test 2017-19", ec=VERM, fs=6.2)
    for x in xs[:3]:
        arrow(x + bw + 0.4, y1 + h / 2, x + 26 - 0.4, y1 + h / 2)
    arrow(xs[3] + bw / 2, y1 - 0.4, xs[3] + bw / 2, y2 + h + 0.4)
    box(xs[0], y2, bw, h, "8. Report\nnumbers spliced from\nresults/*.csv", ec=GREEN, fs=6.2)
    box(xs[1], y2, bw, h, "7. Single test run\nfrozen model, threshold,\nclustered bootstrap", ec=VERM, fs=6.2)
    box(xs[2], y2, bw, h, "6. Freeze\nmodel, parameters,\nF1 threshold from OOF", ec=ORANGE, fs=6.2)
    box(xs[3], y2, bw, h, "5. Pipeline in CV\nfilter, winsorise, impute,\nscale, classifier", fs=6.2)
    for x in xs[:3]:
        arrow(x + 26 - 0.4, y2 + h / 2, x + bw + 0.4, y2 + h / 2)
    ax.text(50, 6.0, "Rolling-origin CV (5 folds, 3-year purge) selects imputer, class weight, hyper-parameters and model;\n"
                     "the test years are touched only in step 7.", ha="center", va="center", fontsize=6.8, color="#444444")
    save(fig, "fig_pipeline_flowchart")


# --------------------------------------------------------------------------- 2 label + split timeline
def fig_timeline():
    c = cfg()
    fig, ax = plt.subplots(figsize=(W, 3.9))
    ax.grid(False)
    ax.set_xlim(1987, 2023.5); ax.set_ylim(-0.3, 7.7)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xticks(range(1990, 2023, 4))
    rows = [
        (7.0, 1988, 1992, GREY, "look-back"),
        (7.0, 1993, 2013, BLUE, "pool: window start years t = 1993-2013"),
        (6.0, 1996, 2016, SKY, "pool labels (t+3) end by 2016"),
        (5.0, 2014, 2016, "#BBBBBB", "purge"),
        (5.0, 2017, 2019, VERM, "test t"),
        (4.0, 2020, 2022, "#E8A98A", "labels"),
    ]
    for y, a, b, col, txt in rows:
        ax.add_patch(Rectangle((a - 0.5, y - 0.32), b - a + 1, 0.64, fc=col, ec="white", lw=1))
        ax.text((a + b) / 2, y, txt, ha="center", va="center", fontsize=6.8,
                color="white" if col in (BLUE, VERM, GREY) else INK)
    ax.text(1987.2, 7.55, "Data and pool", fontsize=7, color="#555555", va="bottom")
    # example window
    ax.add_patch(Rectangle((2004 - 0.5, 3.1 - 0.28), 1, 0.56, fc=ORANGE, ec="white"))
    ax.add_patch(Rectangle((2007 - 0.5, 3.1 - 0.28), 1, 0.56, fc=GREEN, ec="white"))
    ax.annotate("", xy=(2007, 3.45), xytext=(2004, 3.45), arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
    ax.text(2005.5, 3.75, "h = 3 years", ha="center", fontsize=7)
    ax.text(2003.8, 2.6, "features at t = 2004", ha="right", fontsize=6.8)
    ax.text(2007.7, 2.6, "outcome at t+3 = 2007", ha="left", fontsize=6.8)
    ax.text(1987.2, 3.75, "One window", fontsize=7, color="#555555", va="bottom")
    # CV folds
    from features import fold_blocks, pool_end
    ax.text(1987.2, 1.95, "Rolling-origin validation blocks (train = all earlier t, minus 3-year purge)", fontsize=7,
            color="#555555", va="bottom")
    cols = [SKY, BLUE, GREEN, ORANGE, PURPLE]
    for k, (lo, hi) in enumerate(fold_blocks(3, 5)):
        y = 1.2 - 0.0
        ax.add_patch(Rectangle((lo - 0.5, y - 0.3), hi - lo + 1, 0.6, fc=cols[k], ec="white", lw=1))
        ax.text((lo + hi) / 2, y, f"V{k + 1}", ha="center", va="center", fontsize=6.8, color="white" if k in (1, 2, 4) else INK)
    ax.axvline(2013.5, color="#555555", lw=0.8, ls=":")
    ax.axvline(2016.5, color=VERM, lw=0.8, ls=":")
    ax.set_xlabel("Calendar year")
    save(fig, "fig_label_split_timeline")


# --------------------------------------------------------------------------- 3 base rate by year
def fig_base_rate():
    br = pd.read_csv(RES / "label_base_rate_by_year.csv")
    fig, ax = plt.subplots(figsize=(W, 3.3))
    for split, col, lab in (("pool", BLUE, "pool (t $\\leq$ 2013)"), ("test", VERM, "test (t = 2017-2019)")):
        s = br[br["split"] == split]
        ax.bar(s["year"], 100 * s["base_rate"], color=col, width=0.72, label=lab)
    overall = pd.read_csv(RES / "label_base_rate_overall.csv").set_index("scope")["base_rate"]
    ax.axhline(100 * overall["all eligible pool + test rows"], color=INK, lw=0.9, ls="--")
    ax.text(1993.2, 100 * overall["all eligible pool + test rows"] + 1.2, "overall base rate", fontsize=7)
    ax.set_xlabel("Window start year t"); ax.set_ylabel("Decoupled windows (%)")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.set_xlim(1992, 2020)
    save(fig, "fig_label_base_rate")


# --------------------------------------------------------------------------- 4 missingness
def fig_missingness():
    my = pd.read_csv(RES / "missingness_by_year.csv")
    mi = pd.read_csv(RES / "missingness_by_income.csv")
    order = [f for f in F.FEATURES if my.loc[my["feature"] == f, "share_missing"].max() > 0]
    py = my[my["feature"].isin(order)].pivot(index="feature", columns="year", values="share_missing").loc[order]
    inc_order = ["High income", "Upper middle income", "Lower middle income", "Low income", "Not classified"]
    pi = mi[mi["feature"].isin(order)].pivot(index="feature", columns="income_group", values="share_missing").loc[order]
    pi = pi[[c for c in inc_order if c in pi.columns]]
    n = mi.drop_duplicates("income_group").set_index("income_group")["n_rows"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.9), gridspec_kw={"width_ratios": [2.2, 1]})
    for ax, data, xl in ((axes[0], py, "Window start year t"), (axes[1], pi, "Income group")):
        im = ax.imshow(100 * data.values, aspect="auto", cmap="Blues", vmin=0, vmax=100)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order if ax is axes[0] else [], fontsize=7)
        cols = list(data.columns)
        if ax is axes[0]:
            ax.set_xticks(range(0, len(cols), 4)); ax.set_xticklabels([cols[i] for i in range(0, len(cols), 4)])
        else:
            short = {"High income": "High", "Upper middle income": "Upper", "Lower middle income": "Lower",
                     "Low income": "Low", "Not classified": "n/c"}
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels([f"{short[c]}\n{n[c]:,}" for c in cols], fontsize=6.0)
        ax.set_xlabel(xl); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Missing (% of pool rows)")
    save(fig, "fig_missingness_heatmap")


# --------------------------------------------------------------------------- 5 correlation heatmap
def fig_corr():
    r = pd.read_csv(RES / "feature_spearman.csv").set_index("feature")
    fig, ax = plt.subplots(figsize=(W, 5.0))
    im = ax.imshow(r.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(r))); ax.set_xticklabels(r.columns, rotation=60, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(r))); ax.set_yticklabels(r.index, fontsize=6.5)
    for i in range(len(r)):
        for j in range(len(r)):
            if i != j and abs(r.values[i, j]) >= 0.6:
                ax.text(j, i, f"{r.values[i, j]:.1f}", ha="center", va="center", fontsize=4.8,
                        color="white" if abs(r.values[i, j]) > 0.8 else INK)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Spearman rho (pool)")
    save(fig, "fig_feature_correlation")


# --------------------------------------------------------------------------- 6 imputation comparison
def fig_imputation():
    s = pd.read_csv(RES / "stage1_preprocessing.csv")
    fig, axes = plt.subplots(1, 3, figsize=(W, 3.3), sharey=True)
    names = {"median": "median", "ffill_median": "ffill", "iterative": "iterative", "native": "native"}
    for ax, m in zip(axes, M_ORDER):
        sub = s[s["model"] == m]
        imps = [i for i in ["native", "median", "ffill_median", "iterative"] if i in set(sub["imputer"])]
        for cw, mk, off in (("none", "o", -0.13), ("balanced", "s", 0.13)):
            d = sub[sub["class_weight"] == cw].set_index("imputer").loc[imps]
            ax.errorbar(np.arange(len(imps)) + off, d["pr_auc_mean"], yerr=d["pr_auc_se"], fmt=mk, color=MCOL[m],
                        mfc=MCOL[m] if cw == "none" else "white", mec=MCOL[m], ms=5, capsize=2, lw=1)
        ax.set_xticks(range(len(imps))); ax.set_xticklabels([names[i] for i in imps], fontsize=6.8)
        ax.set_title(MNAME[m], fontsize=8.5)
        ax.set_xlim(-0.6, len(imps) - 0.4)
        if m == "lr":
            ax.set_ylabel("Mean PR-AUC, 5 folds ($\\pm$ 1 SE)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", color=INK, ls="", label="no class weight"),
               Line2D([], [], marker="s", mfc="white", mec=INK, color=INK, ls="", label="class_weight = balanced")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_imputation_comparison")


M_ORDER = ["lr", "rf", "hgb"]


# --------------------------------------------------------------------------- 7 CV model comparison
def fig_cv_models():
    cmp = pd.read_csv(RES / "cv_model_comparison.csv").set_index("model")
    folds = pd.read_csv(RES / "cv_model_folds.csv")
    chosen = cfg()["chosen_model"]
    order = ["lr", "rf", "hgb", "persistence", "prior"]
    fig, ax = plt.subplots(figsize=(W, 3.2))
    for i, m in enumerate(order):
        f = folds[folds["model"] == m]["pr_auc"]
        ax.scatter(np.full(len(f), i) + np.linspace(-0.12, 0.12, len(f)), f, s=10, color=MCOL[m], alpha=0.45, zorder=2)
        ax.errorbar(i, cmp.loc[m, "pr_auc_mean"], yerr=cmp.loc[m, "pr_auc_se"], fmt="D", color=MCOL[m], mec=INK, ms=6,
                    capsize=3, lw=1.4, zorder=3)
        ax.text(i + 0.18, cmp.loc[m, "pr_auc_mean"], f"{cmp.loc[m, 'pr_auc_mean']:.3f}", va="center", fontsize=7)
    ax.axhline(cmp.loc["prior", "pr_auc_mean"], color=GREY, ls=":", lw=1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([MNAME[m] + ("\n(selected)" if m == chosen else "") for m in order])
    ax.set_ylabel("PR-AUC (diamond = mean $\\pm$ 1 SE; dots = folds)")
    ax.set_xlim(-0.5, len(order) - 0.3)
    save(fig, "fig_cv_model_comparison")


# --------------------------------------------------------------------------- test figures
def fig_roc():
    curves = pd.read_csv(RES / "test_curve_roc.csv")
    tm = pd.read_csv(RES / "test_metrics.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    ax.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1, label="chance")
    for m in ["lr", "rf", "hgb"]:
        c = curves[curves["model"] == m]
        ax.plot(c["fpr"], c["tpr"], color=MCOL[m], lw=1.6, label=f"{MNAME[m]} (AUC {tm.loc[m, 'roc_auc']:.2f})")
    p = tm.loc["persistence"]
    ax.scatter([p["fp"] / (p["fp"] + p["tn"])], [p["tp"] / (p["tp"] + p["fn"])], color=VERM, marker="X", s=40, zorder=4,
               label="Persistence rule (operating point)")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(frameon=False, loc="lower right")
    save(fig, "fig_test_roc")


def fig_pr():
    curves = pd.read_csv(RES / "test_curve_pr.csv")
    tm = pd.read_csv(RES / "test_metrics.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    ax.axhline(tm.loc["lr", "n_positive"] / tm.loc["lr", "n"], color=GREY, ls=":", lw=1, label="prevalence (test)")
    for m in ["lr", "rf", "hgb"]:
        c = curves[curves["model"] == m]
        ax.plot(c["recall"], c["precision"], color=MCOL[m], lw=1.6, label=f"{MNAME[m]} (PR-AUC {tm.loc[m, 'pr_auc']:.2f})")
    p = tm.loc["persistence"]
    ax.scatter([p["recall"]], [p["precision"]], color=VERM, marker="X", s=40, zorder=4, label="Persistence rule")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02); ax.legend(frameon=False, loc="upper right")
    save(fig, "fig_test_precision_recall")


def fig_confusion():
    cm = pd.read_csv(RES / "test_confusion.csv").set_index("model")
    chosen = cfg()["chosen_model"]
    models = list(dict.fromkeys([chosen, "lr", "rf", "persistence"]))
    fig, axes = plt.subplots(1, len(models), figsize=(W, 2.5))
    for ax, m in zip(np.atleast_1d(axes), models):
        r = cm.loc[m]
        rate = np.array([[r["tn_rate"], r["fp_rate"]], [r["fn_rate"], r["tp_rate"]]])
        cnt = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(rate, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cnt[i, j])}\n({100 * rate[i, j]:.0f}%)", ha="center", va="center", fontsize=8,
                        color="white" if rate[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred. 0", "pred. 1"]); ax.set_yticks([0, 1])
        ax.set_yticklabels(["actual 0", "actual 1"] if ax is np.atleast_1d(axes)[0] else [])
        ax.set_title(MNAME[m] + (" (selected)" if m == chosen else ""), fontsize=8)
    save(fig, "fig_test_confusion_matrices")


def fig_calibration():
    cal = pd.read_csv(RES / "test_calibration.csv")
    tm = pd.read_csv(RES / "test_metrics.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    ax.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1, label="perfect calibration")
    for m in M_ORDER:
        c = cal[(cal["model"] == m) & (cal["n"] > 0)]
        ax.plot(c["mean_predicted"], c["observed_rate"], "o-", color=MCOL[m], ms=3.5, lw=1.3,
                label=f"{MNAME[m]} (ECE {tm.loc[m, 'ece']:.3f})")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed share of decoupled windows")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_test_calibration")


def fig_importance():
    chosen = cfg()["chosen_model"]
    pi = pd.read_csv(RES / "test_permutation_importance.csv").head(10)[::-1]
    co = pd.read_csv(RES / "lr_standardised_coefficients.csv")
    co = co[~co["feature"].str.startswith("missingindicator")]
    co = co.reindex(co["standardised_coefficient"].abs().sort_values(ascending=False).index).head(10)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.5))
    axes[0].barh(pi["feature"], pi["importance_mean"], xerr=pi["importance_sd"], color=MCOL[chosen], height=0.6,
                 error_kw=dict(lw=0.8, capsize=2))
    axes[0].set_xlabel(f"Drop in test PR-AUC when shuffled\n({MNAME[chosen]}, 20 repeats)")
    axes[0].axvline(0, color="#555555", lw=0.8)
    colors = [BLUE if v > 0 else VERM for v in co["standardised_coefficient"]]
    axes[1].barh(co["feature"], co["standardised_coefficient"], color=colors, height=0.6)
    axes[1].set_xlabel("Standardised logistic coefficient\n(blue: raises odds of decoupling)")
    axes[1].axvline(0, color="#555555", lw=0.8)
    for ax in axes:
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    save(fig, "fig_importance_coefficients")


def fig_subgroups():
    sg = pd.read_csv(RES / "test_subgroups.csv")
    chosen = cfg()["chosen_model"]
    sg = sg[sg["grouping"] == "income_group"].copy()
    order = ["High income", "Upper middle income", "Lower middle income", "Low income"]
    sg = sg.set_index("group").reindex([g for g in order if g in set(sg["group"])]).reset_index()
    fig, ax = plt.subplots(figsize=(W, 3.4))
    x = np.arange(len(sg))
    seen = set()
    for off, m in ((-0.2, chosen), (0.0, "lr"), (0.2, "persistence")):
        if m in seen:
            continue
        seen.add(m)
        if m == chosen and m == "lr":
            off = -0.1
        elif m == "persistence" and chosen == "lr":
            off = 0.1
        y = sg[f"{m}_pr_auc"]
        if f"{m}_pr_auc_lo" in sg:
            err = np.vstack([y - sg[f"{m}_pr_auc_lo"], sg[f"{m}_pr_auc_hi"] - y])
            err = np.nan_to_num(err, nan=0.0)
        else:
            err = None
        ax.errorbar(x + off, y, yerr=err, fmt="o", color=MCOL[m], ms=5, capsize=2.5, lw=1.1, label=MNAME[m])
    ax.scatter(x, sg["base_rate"], marker="_", s=900, color=INK, lw=1.4, zorder=1, label="group prevalence")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\nn={n}, positives={p}" for g, n, p in zip(sg["group"], sg["n"], sg["n_positive"])], fontsize=6.8)
    ax.set_ylabel("Test PR-AUC (95% country-clustered CI)"); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=4, loc="upper center", fontsize=6.8, bbox_to_anchor=(0.5, 1.12))
    save(fig, "fig_subgroup_income")


def fig_ablation():
    cv = pd.read_csv(RES / "ablation_cv.csv")
    te = pd.read_csv(RES / "test_ablation.csv")
    chosen = cfg()["chosen_model"]
    sets = ["A", "B", "C", "C+L1"]
    seen_ab = set()
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.0), sharey=False)
    for ax, (df, mean, lo, hi, ttl) in zip(axes, ((cv, "pr_auc_mean", None, "pr_auc_se", "Rolling-origin CV (pool)"),
                                                  (te, "pr_auc", "pr_auc_lo", "pr_auc_hi", "Test (single run)"))):
        for off, m in ((-0.12, chosen), (0.12, "lr")):
            if m == "lr" and chosen == "lr":
                off = 0.0
                if ax is not axes[0] and False:
                    continue
            if (m, id(ax)) in seen_ab:
                continue
            seen_ab.add((m, id(ax)))
            d = df[df["model"] == m].set_index("feature_set").loc[sets]
            if lo is None:
                err = d[hi]
            else:
                err = np.vstack([d[mean] - d[lo], d[hi] - d[mean]])
            ax.errorbar(np.arange(4) + off, d[mean], yerr=err, fmt="o", color=MCOL[m], capsize=2.5, ms=5, lw=1.1, label=MNAME[m])
        ax.set_xticks(range(4)); ax.set_xticklabels(["A\nmomentum", "B\n+ structure", "C\nfull", "C + L1\nselection"], fontsize=6.8)
        ax.set_title(ttl, fontsize=8.5)
    axes[0].set_ylabel("PR-AUC"); axes[0].legend(frameon=False, loc="best")
    save(fig, "fig_feature_set_ablation")


def main() -> None:
    fig_pipeline(); fig_timeline(); fig_base_rate(); fig_missingness(); fig_corr(); fig_imputation(); fig_cv_models()
    fig_roc(); fig_pr(); fig_confusion(); fig_calibration(); fig_importance(); fig_subgroups(); fig_ablation()


if __name__ == "__main__":
    main()
