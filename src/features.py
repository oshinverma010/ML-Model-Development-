"""Feature engineering, label construction and leakage-safe splits.

Everything here follows docs/ml_plan.md (committed before any model was fitted).

* Raw series are placed on a complete country x year grid before any lag is taken, so "t-3" is always three calendar years.
* Every feature at year t uses values at t, t-1, ... only. Nothing from t+1 onwards enters a feature
  (tests/test_pipeline_integrity.py perturbs the future and checks that features do not change).
* The label looks forward h years: y = 1 if CO2pc(t+h) < CO2pc(t) and GDPpc(t+h) > GDPpc(t).
* Split: pool t <= 2016 - h (labels end by 2016), purge gap, test t in {2017, 2018, 2019}.
  Selection code must call get_pool(); only src/evaluate_test.py may call get_test().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"

GRID_START, GRID_END = 1988, 2022      # 1988-1989 are look-back only
T_MIN = 1993                            # first window start year (5-year features need levels back to 1988)
LAST_LABEL_YEAR = 2022
LABEL_END_BY = 2016                     # pool labels must end by this year
TEST_YEARS = (2017, 2018, 2019)
H_DEFAULT = 3
BLOCK_YEARS = 3                         # length of each rolling-origin validation block
SMALL_POP = 1_000_000

# raw level columns that the optional within-country forward fill may touch (never the core CO2 / GDP series)
FFILL_COLS = ["urban_pct", "industry_pct", "energy_per_capita", "energy_per_gdp", "co2_per_unit_energy",
              "coal_co2", "oil_co2", "gas_co2", "co2_total_mt"]
FFILL_LIMIT = 2

# feature catalogue in the priority order used by the near-duplicate filter
FEATURES = [
    "ln_gdp_pc", "gdp_g1", "gdp_cagr3", "gdp_vol5", "ln_pop", "pop_g3", "urban_pct", "industry_pct",
    "ln_co2_pc", "co2_chg1", "co2_chg3", "co2_chg5", "ln_co2_per_gdp", "ci_chg3",
    "ln_energy_pc", "ln_energy_per_gdp", "co2_per_unit_energy", "coal_share", "oil_share", "gas_share",
]
FEATURE_SETS = {
    "A": ["co2_chg1", "co2_chg3", "co2_chg5", "ci_chg3", "gdp_g1", "gdp_cagr3"],
    "B": ["co2_chg1", "co2_chg3", "co2_chg5", "ci_chg3", "gdp_g1", "gdp_cagr3",
          "ln_gdp_pc", "gdp_vol5", "ln_pop", "pop_g3", "urban_pct", "industry_pct", "ln_co2_pc", "ln_co2_per_gdp"],
    "C": FEATURES,
}
FEATURE_META = {   # group, expected direction of association with P(y = 1)
    "ln_gdp_pc": ("economic", "+"), "gdp_g1": ("economic", "+"), "gdp_cagr3": ("economic", "+"),
    "gdp_vol5": ("economic", "-"), "ln_pop": ("economic", "?"), "pop_g3": ("economic", "-"),
    "urban_pct": ("economic", "?"), "industry_pct": ("economic", "-"),
    "ln_co2_pc": ("emissions", "+"), "co2_chg1": ("emissions", "-"), "co2_chg3": ("emissions", "-"),
    "co2_chg5": ("emissions", "-"), "ln_co2_per_gdp": ("emissions", "-"), "ci_chg3": ("emissions", "-"),
    "ln_energy_pc": ("energy mix", "?"), "ln_energy_per_gdp": ("energy mix", "-"),
    "co2_per_unit_energy": ("energy mix", "-"), "coal_share": ("energy mix", "-"),
    "oil_share": ("energy mix", "?"), "gas_share": ("energy mix", "+"),
}
META_COLS = ["iso3", "year", "wb_name", "region", "income_group", "population", "small_pop"]


# --------------------------------------------------------------------------- helpers
def _ln(s: pd.Series) -> pd.Series:
    """Natural log of strictly positive values; non-positive and missing become NaN (log guard)."""
    return np.log(s.where(s > 0))


def load_panel(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or PROC / "panel.csv")


def to_grid(panel: pd.DataFrame, start: int = GRID_START, end: int = GRID_END) -> pd.DataFrame:
    """Complete iso3 x year grid, sorted, so that shift(k) means k calendar years."""
    idx = pd.MultiIndex.from_product([sorted(panel["iso3"].unique()), range(start, end + 1)], names=["iso3", "year"])
    g = panel.set_index(["iso3", "year"]).reindex(idx).reset_index()
    static = panel.drop_duplicates("iso3").set_index("iso3")[["wb_name", "region", "income_group"]]
    for c in static.columns:
        g[c] = g["iso3"].map(static[c])
    return g.sort_values(["iso3", "year"]).reset_index(drop=True)


def forward_fill(grid: pd.DataFrame, cols: list[str] = FFILL_COLS, limit: int = FFILL_LIMIT) -> pd.DataFrame:
    """Within-country forward fill (past values only), at most `limit` consecutive years."""
    out = grid.copy()
    out[cols] = out.groupby("iso3")[cols].ffill(limit=limit)
    return out


# --------------------------------------------------------------------------- features
def compute_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Feature columns for every (iso3, year) of the grid. Uses only values at year t and earlier."""
    g = grid.sort_values(["iso3", "year"]).reset_index(drop=True)
    by = g["iso3"]
    f = pd.DataFrame({"iso3": g["iso3"], "year": g["year"]})
    ln_gdp, ln_co2, ln_pop = _ln(g["gdp_pc"]), _ln(g["co2_pc"]), _ln(g["population"])

    def lag(s: pd.Series, k: int) -> pd.Series:
        return s.groupby(by).shift(k)

    f["ln_gdp_pc"] = ln_gdp
    f["gdp_g1"] = ln_gdp - lag(ln_gdp, 1)
    f["gdp_cagr3"] = (ln_gdp - lag(ln_gdp, 3)) / 3
    f["gdp_vol5"] = f["gdp_g1"].groupby(by).transform(lambda s: s.rolling(5, min_periods=5).std())
    f["ln_pop"] = ln_pop
    f["pop_g3"] = (ln_pop - lag(ln_pop, 3)) / 3
    f["urban_pct"] = g["urban_pct"]
    f["industry_pct"] = g["industry_pct"]
    f["ln_co2_pc"] = ln_co2
    for k in (1, 3, 5):
        f[f"co2_chg{k}"] = ln_co2 - lag(ln_co2, k)
    ln_ci = ln_co2 - ln_gdp                               # ln of CO2 per unit of GDP (identity: ln CO2pc - ln GDPpc)
    f["ln_co2_per_gdp"] = ln_ci
    f["ci_chg3"] = ln_ci - lag(ln_ci, 3)
    f["ln_energy_pc"] = _ln(g["energy_per_capita"])
    f["ln_energy_per_gdp"] = _ln(g["energy_per_gdp"])
    f["co2_per_unit_energy"] = g["co2_per_unit_energy"].where(g["co2_per_unit_energy"] >= 0)
    tot = g["co2_total_mt"].where(g["co2_total_mt"] > 0)
    for name, col in (("coal_share", "coal_co2"), ("oil_share", "oil_co2"), ("gas_share", "gas_co2")):
        sh = g[col] / tot
        f[name] = sh.where(sh.between(0, 1.0))
    return f[["iso3", "year"] + FEATURES]


# --------------------------------------------------------------------------- label + dataset
def make_label(grid: pd.DataFrame, h: int = H_DEFAULT) -> pd.DataFrame:
    """Label and eligibility for window start year t. Uses values at t and t+h only."""
    g = grid.sort_values(["iso3", "year"]).reset_index(drop=True)
    by = g["iso3"]
    out = g[["iso3", "year"]].copy()
    out["co2_t"], out["gdp_t"] = g["co2_pc"], g["gdp_pc"]
    out["co2_th"] = g["co2_pc"].groupby(by).shift(-h)
    out["gdp_th"] = g["gdp_pc"].groupby(by).shift(-h)
    out["y"] = ((out["co2_th"] < out["co2_t"]) & (out["gdp_th"] > out["gdp_t"])).astype(int)
    return out


def build_dataset(panel: pd.DataFrame | None = None, h: int = H_DEFAULT, ffill: bool = False,
                  write_flow: bool = False) -> pd.DataFrame:
    """Eligible modelling rows (one per country-year window start) with label, features and metadata."""
    panel = load_panel() if panel is None else panel
    grid = to_grid(panel)
    if ffill:
        grid = forward_fill(grid)
    feats = compute_features(grid)
    lab = make_label(grid, h)
    meta = grid[["iso3", "year", "wb_name", "region", "income_group", "population"]].copy()
    meta["small_pop"] = meta["population"] < SMALL_POP

    df = meta.merge(lab, on=["iso3", "year"]).merge(feats, on=["iso3", "year"])
    df = df[(df["year"] >= T_MIN) & (df["year"] <= LAST_LABEL_YEAR - h)].copy()

    stages = [
        ("no CO2 per capita > 0 at t", df["co2_t"] > 0),
        ("no GDP per capita > 0 at t", df["gdp_t"] > 0),
        (f"no CO2 per capita > 0 at t+{h}", df["co2_th"] > 0),
        (f"no GDP per capita > 0 at t+{h}", df["gdp_th"] > 0),
    ]
    alive = pd.Series(True, index=df.index)
    flow = [{"stage": f"all entity-years, t = {T_MIN}-{LAST_LABEL_YEAR - h}", "n_remaining": int(alive.sum()),
             "n_dropped": 0}]
    dropped = []
    for name, ok in stages:
        ok = ok.fillna(False)
        drop = alive & ~ok
        d = df.loc[drop, ["iso3", "year"]].copy()
        d["reason"] = name
        dropped.append(d)
        alive &= ok
        flow.append({"stage": name, "n_remaining": int(alive.sum()), "n_dropped": int(drop.sum())})
    if write_flow:
        RES.mkdir(exist_ok=True)
        PROC.mkdir(exist_ok=True)
        pd.DataFrame(flow).to_csv(RES / "sample_flow.csv", index=False)
        pd.concat(dropped).to_csv(PROC / "dropped_rows.csv", index=False)
    df = df[alive].copy()
    df["income_group"] = df["income_group"].fillna("Not classified")
    df["region"] = df["region"].fillna("Not classified")
    cols = META_COLS + ["y"] + FEATURES
    return df[cols].sort_values(["year", "iso3"]).reset_index(drop=True)


# --------------------------------------------------------------------------- splits
def pool_end(h: int = H_DEFAULT) -> int:
    """Last window start year of the pool: its label window ends by 2016."""
    return LABEL_END_BY - h


def get_pool(df: pd.DataFrame, h: int = H_DEFAULT) -> pd.DataFrame:
    """Training / selection pool. The only accessor selection code may use."""
    return df[df["year"] <= pool_end(h)].reset_index(drop=True)


def get_test(df: pd.DataFrame, h: int = H_DEFAULT) -> pd.DataFrame:
    """Final test rows. Called only by src/evaluate_test.py."""
    ok = [t for t in TEST_YEARS if t + h <= LAST_LABEL_YEAR]
    return df[df["year"].isin(ok)].reset_index(drop=True)


def fold_blocks(h: int = H_DEFAULT, n_folds: int = 5) -> list[tuple[int, int]]:
    """Validation year blocks (3 years each) ending at the pool end, oldest first."""
    end = pool_end(h)
    return [(end - BLOCK_YEARS * (k + 1) + 1, end - BLOCK_YEARS * k) for k in reversed(range(n_folds))]


def rolling_origin_folds(pool: pd.DataFrame, h: int = H_DEFAULT, n_folds: int = 5):
    """Expanding-window folds with a purge gap of h years.

    For a validation block starting at year v the training rows are those with t <= v - h - 1,
    so every training label window (t .. t+h) ends before v.
    Returns a list of (train_positions, validation_positions) into `pool`.
    """
    yr = pool["year"].to_numpy()
    folds = []
    for lo, hi in fold_blocks(h, n_folds):
        tr = np.where(yr <= lo - h - 1)[0]
        va = np.where((yr >= lo) & (yr <= hi))[0]
        folds.append((tr, va))
    return folds


def group_folds(pool: pd.DataFrame, n_splits: int = 5):
    """Secondary check: country-grouped folds (ignores time order)."""
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(n_splits=n_splits).split(pool, groups=pool["iso3"]))


def base_rate_by_year(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("year")["y"].agg(n_rows="size", n_positive="sum", base_rate="mean").reset_index()
    return g


if __name__ == "__main__":
    d = build_dataset(write_flow=True)
    pool = get_pool(d)
    print(f"eligible rows {len(d):,}; pool rows {len(pool):,}; pool base rate {pool['y'].mean():.3f}")
    print(pd.read_csv(RES / "sample_flow.csv").to_string(index=False))
