"""Integrity tests for the Week 3 ML pipeline. Run with:  pytest -v   (after src/acquire.py, clean.py, evaluate_cv.py, evaluate_test.py).

They check data hygiene (keys, aggregates, joins, checksums), label construction on hand-made examples, absence of
look-ahead in features, the time-aware split design, that imputers / scalers / winsorising limits learn from training
folds only, reproducibility, and the discipline around the single test run.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features as F  # noqa: E402
import metrics as MT  # noqa: E402
import models as M  # noqa: E402

RAW, PROC, RES = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "results"


@pytest.fixture(scope="session")
def panel():
    return pd.read_csv(PROC / "panel.csv")


@pytest.fixture(scope="session")
def dataset(panel):
    return F.build_dataset(panel)


@pytest.fixture(scope="session")
def pool(dataset):
    return F.get_pool(dataset)


def _toy_panel(rows):
    """rows: dict iso3 -> list of (year, co2_pc, gdp_pc). All other columns constant."""
    recs = []
    for iso, series in rows.items():
        for yr, co2, gdp in series:
            recs.append({"iso3": iso, "year": yr, "co2_pc": co2, "gdp_pc": gdp, "population": 5e6, "urban_pct": 50.0,
                         "industry_pct": 25.0, "energy_per_capita": 20000.0, "energy_per_gdp": 1.0,
                         "co2_per_unit_energy": 200.0, "coal_co2": 1.0, "oil_co2": 2.0, "gas_co2": 1.0,
                         "co2_total_mt": 5.0, "wb_name": iso, "region": "R", "income_group": "High income"})
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------------- data hygiene
def test_01_no_duplicate_iso3_year_in_panel_and_dataset(panel, dataset):
    assert not panel.duplicated(["iso3", "year"]).any()
    assert not dataset.duplicated(["iso3", "year"]).any()


def test_02_no_aggregates_in_panel(panel):
    meta = pd.read_csv(RAW / "wb_country_metadata.csv", keep_default_na=False)
    agg = set(meta.loc[meta["region"] == "Aggregates", "iso3"])
    assert not set(panel["iso3"]) & agg
    assert panel["iso3"].str.fullmatch(r"[A-Z]{3}").all()
    assert not panel["iso3"].str.startswith("OWID").any()


def test_03_kosovo_crosswalk_applied(panel):
    k = panel[(panel["iso3"] == "XKX") & panel["co2_pc"].notna()]
    assert len(k) > 0 and panel.loc[panel["iso3"] == "XKX", "gdp_pc"].notna().any()


def test_04_iso3_join_completeness(panel):
    jr = pd.read_csv(RES / "join_report.csv").set_index("category")["country_year_rows"]
    assert jr["both"] + jr["owid_only"] + jr["wb_only"] == jr["total"] == len(panel)
    assert jr["both"] > 0.85 * jr["total"] - 1000          # most country-years exist in both sources
    elig_iso = set(F.build_dataset(panel)["iso3"])
    assert elig_iso <= set(panel["iso3"])
    d = F.build_dataset(panel)
    assert (d["income_group"] != "Not classified").mean() > 0.95


def test_05_raw_files_match_manifest():
    man = json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))
    assert len(man) >= 7
    for name, info in man.items():
        data = (RAW / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == info["sha256"], name


def test_06_core_series_never_imputed(panel, dataset):
    """CO2pc and GDPpc of eligible rows are observed values (strictly positive), never filled."""
    m = dataset.merge(panel[["iso3", "year", "co2_pc", "gdp_pc"]], on=["iso3", "year"], how="left")
    assert (m["co2_pc"] > 0).all() and (m["gdp_pc"] > 0).all()


def test_07_log_guard_and_finite_features(dataset):
    X = dataset[F.FEATURES].to_numpy(float)
    assert not np.isinf(X).any()
    d_ln = dataset[[c for c in F.FEATURES if c.startswith("ln_")]]
    assert d_ln.notna().sum().min() > 0


def test_08_share_features_in_unit_interval(dataset):
    for c in ("coal_share", "oil_share", "gas_share"):
        v = dataset[c].dropna()
        assert v.between(0, 1).all()
    assert dataset["urban_pct"].dropna().between(0, 100).all()
    assert dataset["industry_pct"].dropna().between(0, 100).all()


# ----------------------------------------------------------------------------- label
def test_09_label_hand_made_examples():
    toy = _toy_panel({
        "AAA": [(1990, 10.0, 1000.0), (1993, 8.0, 1200.0)],      # CO2 down, GDP up  -> 1
        "BBB": [(1990, 10.0, 1000.0), (1993, 12.0, 1200.0)],     # CO2 up,   GDP up  -> 0
        "CCC": [(1990, 10.0, 1000.0), (1993, 8.0, 900.0)],       # CO2 down, GDP down-> 0
        "DDD": [(1990, 10.0, 1000.0), (1993, 10.0, 1200.0)],     # CO2 equal         -> 0 (strict fall required)
        "EEE": [(1990, 10.0, 1000.0), (1993, 9.0, 1000.0)],      # GDP equal         -> 0 (strict rise required)
    })
    g = F.to_grid(toy, 1990, 1993)
    lab = F.make_label(g, 3).set_index(["iso3", "year"])["y"]
    assert [int(lab[(c, 1990)]) for c in ["AAA", "BBB", "CCC", "DDD", "EEE"]] == [1, 0, 0, 0, 0]


def test_10_label_uses_only_t_and_t_plus_h():
    toy = _toy_panel({"AAA": [(y, 10.0 - 0.1 * (y - 1990), 1000.0 + 10 * (y - 1990)) for y in range(1990, 1999)]})
    g = F.to_grid(toy, 1990, 1998)
    base = F.make_label(g, 3).set_index("year")["y"]
    g2 = g.copy()
    g2.loc[g2["year"] > 1993, ["co2_pc", "gdp_pc"]] = [99.0, 1.0]       # change everything after t+h for t=1990
    changed = F.make_label(g2, 3).set_index("year")["y"]
    assert changed[1990] == base[1990] == 1                              # window 1990-1993 unaffected by later years
    g3 = g.copy()
    g3.loc[g3["year"] == 1993, "co2_pc"] = 99.0                          # change t+h itself
    assert F.make_label(g3, 3).set_index("year")["y"][1990] == 0


def test_11_persistence_rule_hand_example():
    X = pd.DataFrame({"co2_chg3": [-0.1, 0.1, -0.1, np.nan], "gdp_cagr3": [0.02, 0.02, -0.01, 0.02]})
    assert M.persistence_pred(X).tolist() == [1.0, 0.0, 0.0, 0.0]


def test_12_label_base_rate_within_bounds(pool):
    assert 0.05 <= pool["y"].mean() <= 0.60
    by_year = pool.groupby("year")["y"].mean()
    assert by_year.between(0.0, 0.9).all()


# ----------------------------------------------------------------------------- features: no look-ahead
def test_13_features_do_not_use_data_after_t(panel):
    grid = F.to_grid(panel)
    base = F.compute_features(grid)
    T = 2008
    pert = grid.copy()
    num = pert.select_dtypes("number").columns.difference(["year"])
    mask = pert["year"] > T
    rng = np.random.default_rng(0)
    pert.loc[mask, num] = pert.loc[mask, num] * rng.uniform(0.2, 5.0, size=(mask.sum(), len(num)))
    new = F.compute_features(pert)
    a = base[base["year"] <= T].reset_index(drop=True)
    b = new[new["year"] <= T].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_14_features_hand_computed():
    years = range(1988, 1996)
    toy = _toy_panel({"AAA": [(y, 10.0 * 0.98 ** (y - 1988), 1000.0 * 1.03 ** (y - 1988)) for y in years]})
    f = F.compute_features(F.to_grid(toy, 1988, 1995)).set_index("year")
    assert f.loc[1995, "gdp_g1"] == pytest.approx(np.log(1.03))
    assert f.loc[1995, "gdp_cagr3"] == pytest.approx(np.log(1.03))
    assert f.loc[1995, "co2_chg5"] == pytest.approx(5 * np.log(0.98))
    assert f.loc[1995, "ci_chg3"] == pytest.approx(3 * (np.log(0.98) - np.log(1.03)))
    assert f.loc[1995, "gdp_vol5"] == pytest.approx(0.0, abs=1e-12)
    assert f.loc[1990, "co2_chg5"] != f.loc[1990, "co2_chg5"]            # NaN: no data five years back inside the grid


def test_15_forward_fill_is_past_only_and_limited_to_two_years():
    toy = _toy_panel({"AAA": [(y, 10.0, 1000.0) for y in range(1988, 1996)]})
    toy.loc[toy["year"] == 1990, "urban_pct"] = 40.0
    toy.loc[toy["year"].between(1991, 1994), "urban_pct"] = np.nan
    g = F.forward_fill(F.to_grid(toy, 1988, 1995)).set_index("year")["urban_pct"]
    assert g[1991] == 40.0 and g[1992] == 40.0                           # filled from the past, 2 years
    assert np.isnan(g[1993]) and np.isnan(g[1994])                       # limit reached, no back-filling
    assert g[1995] == 50.0                                               # untouched observed value
    assert g[1989] == 50.0


def test_16_forbidden_features_not_used():
    banned = {"year", "iso3", "income_group", "region", "country", "wb_name"}
    assert not banned & set(F.FEATURES)
    for s in F.FEATURE_SETS.values():
        assert not banned & set(s)
    assert set(F.FEATURE_SETS["A"]) <= set(F.FEATURE_SETS["B"]) <= set(F.FEATURE_SETS["C"])


# ----------------------------------------------------------------------------- split design
def test_17_train_purge_test_periods_do_not_overlap(dataset):
    pool, test = F.get_pool(dataset), F.get_test(dataset)
    h = F.H_DEFAULT
    assert pool["year"].max() == F.pool_end(h) == 2013
    assert set(test["year"]) == {2017, 2018, 2019}
    assert not set(pool["year"]) & set(test["year"])
    assert pool["year"].max() + h < test["year"].min()                   # pool label windows end before test feature years
    purge = set(range(pool["year"].max() + 1, test["year"].min()))
    assert purge == {2014, 2015, 2016}
    assert not purge & (set(pool["year"]) | set(test["year"]))


def test_18_test_years_absent_from_selection_outputs(pool):
    """Selection code never touches the test years: pool, OOF file and source text all clean."""
    assert pool["year"].max() <= 2013
    oof = pd.read_csv(PROC / "oof_predictions.csv")
    assert oof["year"].max() <= 2013
    src = (ROOT / "src" / "evaluate_cv.py").read_text(encoding="utf-8")
    for forbidden in ("get_test(", "evaluate_test", "test_predictions"):
        assert forbidden not in src, forbidden
    # TEST_YEARS may appear once, only to record the year numbers in the frozen config (no rows are loaded)
    uses = [ln for ln in src.splitlines() if "TEST_YEARS" in ln]
    assert len(uses) == 1 and '"test_years"' in uses[0]


def test_19_rolling_origin_folds_have_purge_gap(pool):
    h = F.H_DEFAULT
    folds = F.rolling_origin_folds(pool, h)
    assert len(folds) == 5
    prev_val_end = 0
    for tr, va in folds:
        ytr, yva = pool["year"].to_numpy()[tr], pool["year"].to_numpy()[va]
        assert ytr.max() + h < yva.min()                                 # training label windows end before validation starts
        assert yva.min() - ytr.max() == h + 1                            # exactly h purged years
        assert yva.min() > prev_val_end                                  # validation blocks do not overlap
        prev_val_end = yva.max()
        assert len(set(tr) & set(va)) == 0
    assert prev_val_end == F.pool_end(h)


def test_20_group_folds_never_share_a_country(pool):
    for tr, va in F.group_folds(pool):
        assert not set(pool["iso3"].iloc[tr]) & set(pool["iso3"].iloc[va])


# ----------------------------------------------------------------------------- fitting on training folds only
def _toy_xy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    X.loc[rng.random(n) < 0.2, "b"] = np.nan
    y = (X["a"].fillna(0) + rng.normal(size=n) > 0.5).astype(int).to_numpy()
    return X, y


def test_21_imputer_scaler_and_winsorizer_fitted_on_training_fold_only():
    X, y = _toy_xy()
    tr, va = np.arange(300), np.arange(300, 400)
    X = X.copy()
    X.loc[va, "a"] += 1000.0                                             # validation block is wildly different
    X.loc[va, "b"] = 777.0
    pipe = M.make_pipeline("lr", "median", None, corr_filter=False).fit(X.iloc[tr], y[tr])
    assert pipe.named_steps["winsorise"].hi_[0] == pytest.approx(np.percentile(X.iloc[tr]["a"], 99))
    assert pipe.named_steps["winsorise"].hi_[0] < 100
    win = pipe.named_steps["winsorise"].transform(X.iloc[tr])
    assert pipe.named_steps["impute"].statistics_[1] == pytest.approx(np.nanmedian(win[:, 1]))
    imputed = pipe.named_steps["impute"].transform(win)
    assert pipe.named_steps["scale"].mean_[0] == pytest.approx(imputed[:, 0].mean())
    # predicting on the strange validation block does not change anything that was learned
    before = pipe.named_steps["scale"].mean_.copy()
    pipe.predict_proba(X.iloc[va])
    np.testing.assert_array_equal(before, pipe.named_steps["scale"].mean_)


def test_22_correlation_filter_uses_training_fold_and_drops_duplicates():
    rng = np.random.default_rng(1)
    base = rng.normal(size=300)
    X = pd.DataFrame({"x1": base, "x2": base * 2 + rng.normal(scale=0.01, size=300), "x3": rng.normal(size=300)})
    flt = M.CorrelationFilter(0.9).fit(X)
    assert list(flt.get_feature_names_out(["x1", "x2", "x3"])) == ["x1", "x3"]
    assert flt.transform(X).shape[1] == 2


def test_23_seed_reproducibility():
    X, y = _toy_xy()
    for model in M.MODELS:
        imp = "native" if model == "hgb" else "median"
        a = M.make_pipeline(model, imp).fit(X, y).predict_proba(X)[:, 1]
        b = M.make_pipeline(model, imp).fit(X, y).predict_proba(X)[:, 1]
        np.testing.assert_array_equal(a, b)
    rep = pd.read_csv(RES / "reproducibility.csv")
    assert rep["max_abs_difference_between_two_fits"].iloc[0] == 0.0


def test_24_no_smote_in_source():
    """SMOTE is deliberately not used on time-ordered panel data."""
    for p in (ROOT / "src").glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "imblearn" not in text and "SMOTE(" not in text, p.name


# ----------------------------------------------------------------------------- metrics and bootstrap
def test_25_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert MT.safe_auc(y, p) == pytest.approx(0.75)
    m = MT.threshold_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 0, 2, 1)
    assert m["f1"] == pytest.approx(2 * 1 * 0.5 / 1.5)
    assert MT.ece(np.array([1, 1, 0, 0]), np.array([1.0, 1.0, 0.0, 0.0])) == 0.0
    assert MT.best_f1_threshold(y, p) == pytest.approx(0.35)


def test_26_cluster_bootstrap_resamples_whole_countries():
    groups = np.array(["A"] * 3 + ["B"] * 2 + ["C"] * 4)
    boot = MT.ClusterBootstrap(groups, n_boot=50, seed=42)
    sizes = {"A": 3, "B": 2, "C": 4}
    for ix in boot.draws():
        counts = pd.Series(groups[ix]).value_counts()
        for g, c in counts.items():
            assert c % sizes[g] == 0                                     # a country enters with all of its rows, or not at all


# ----------------------------------------------------------------------------- bookkeeping and the single test run
def test_27_sample_flow_matches_dataset(dataset):
    flow = pd.read_csv(RES / "sample_flow.csv")
    assert flow["n_remaining"].iloc[-1] == len(dataset)
    assert (flow["n_remaining"].diff().dropna() <= 0).all()
    assert flow["n_dropped"].sum() == flow["n_remaining"].iloc[0] - flow["n_remaining"].iloc[-1]


def test_28_frozen_config_complete_and_before_test_run():
    cfg = json.loads((RES / "frozen_config.json").read_text(encoding="utf-8"))
    lock = json.loads((RES / "test_run_lock.json").read_text(encoding="utf-8"))
    assert cfg["chosen_model"] in M.MODELS and cfg["baseline_model"] == "lr"
    assert 0 < cfg["models"][cfg["chosen_model"]]["threshold"] < 1
    assert cfg["frozen_utc"] < lock["run_utc"]                            # frozen before the test was evaluated
    assert lock["config_sha256"] == hashlib.sha256((RES / "frozen_config.json").read_bytes()).hexdigest()


def test_29_test_run_used_frozen_threshold_and_test_rows_only(dataset):
    cfg = json.loads((RES / "frozen_config.json").read_text(encoding="utf-8"))
    pred = pd.read_csv(PROC / "test_predictions.csv")
    assert set(pred["year"]) == {2017, 2018, 2019}
    assert len(pred) == len(F.get_test(dataset))
    chosen = cfg["chosen_model"]
    thr = cfg["models"][chosen]["threshold"]
    tm = pd.read_csv(RES / "test_metrics.csv")
    row = tm[tm["model"] == chosen].iloc[0]
    recomputed = MT.threshold_metrics(pred["y"], pred[f"p_{chosen}"], thr)
    assert row["tp"] == recomputed["tp"] and row["fp"] == recomputed["fp"]
    assert row["threshold"] == pytest.approx(thr)


def test_30_report_has_no_placeholders_and_clean_tables():
    """No placeholder words, LF line endings, and every table row has the same number of cells as its header."""
    path = ROOT / "report" / "report_content.md"
    if not path.exists():
        pytest.skip("report not built yet")
    raw = path.read_bytes()
    assert b"\r" not in raw
    text = raw.decode("utf-8")
    for bad in ("TODO", "TBD", "lorem", "author to enter", "{{", "}}"):
        assert bad.lower() not in text.lower(), bad
    expected = None
    for line in text.splitlines():
        if line.startswith("|"):
            if expected is None:
                expected = line.count("|")
            assert line.count("|") == expected, line[:80]
        else:
            expected = None


def test_31_figures_exist_and_are_not_tall():
    import matplotlib.image as mpimg
    figs = sorted((ROOT / "figures").glob("fig_*.png"))
    assert len(figs) == 14
    for f in figs:
        h, w = mpimg.imread(f).shape[:2]
        assert h <= 0.85 * w, f.name                                     # landscape or squarish (0.8 plus padding)
