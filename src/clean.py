"""Clean and merge OWID + World Bank data (cleaning rules carried over from Weeks 1-2).

Rules
-----
1. Remove non-country aggregates: OWID rows with no ISO3 code, World Bank rows whose region is "Aggregates".
2. Join on ISO3 + year only, never on country names. The only name-based step is the Kosovo crosswalk
   (OWID has no ISO code for Kosovo; the World Bank uses XKX).
3. The outcome series (CO2 per capita) and the main series (GDP per capita) are never imputed here.
4. Keep legitimate extremes (for example Qatar); log transforms downstream use strictly positive values only.
5. Shares outside 0-100 are set to missing and logged.
6. Years: 1988-2022 loaded (1990-2022 study window, 1988-1989 are look-back only).
7. Every dropped or changed row is written to data/processed/cleaning_log_rows.csv; step totals to results/cleaning_log.csv.

Outputs: data/processed/panel.csv, results/cleaning_log.csv, results/join_report.csv, results/column_availability.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"

Y0, Y1 = 1988, 2022          # loaded window (study window 1990-2022 plus 2 look-back years)
STUDY_START = 1990
KOSOVO_CROSSWALK = {"Kosovo": "XKX"}
WB_COLS = {
    "NY.GDP.PCAP.KD": "gdp_pc",
    "SP.POP.TOTL": "population_wb",
    "SP.URB.TOTL.IN.ZS": "urban_pct",
    "NV.IND.TOTL.ZS": "industry_pct",
}
# OWID columns the plan asks for. Present ones are loaded; absent ones are recorded in column_availability.csv.
OWID_WANTED = [
    "country", "iso_code", "year", "population", "co2", "co2_per_capita",
    "energy_per_capita", "energy_per_gdp", "co2_per_unit_energy",
    "coal_co2", "oil_co2", "gas_co2", "cement_co2", "flaring_co2",
    "fossil_share_energy", "low_carbon_share_energy", "renewables_share_energy",
]


class CleaningLog:
    """Collects step-level counts and row-level drop/change records."""

    def __init__(self) -> None:
        self.steps: list[dict] = []
        self.rows: list[pd.DataFrame] = []

    def step(self, step: str, rule: str, before: int, after: int, note: str = "") -> None:
        self.steps.append({"step": step, "rule": rule, "rows_before": before, "rows_after": after,
                           "rows_dropped_or_changed": before - after, "note": note})
        print(f"[{step}] {rule}: {before:,} -> {after:,}  {note}")

    def rows_dropped(self, df: pd.DataFrame, source: str, step: str, reason: str, key_cols: list[str]) -> None:
        if len(df) == 0:
            return
        out = df[key_cols].copy()
        out.columns = ["key1", "key2"] if len(key_cols) == 2 else ["key1"]
        if "key2" not in out:
            out["key2"] = ""
        out.insert(0, "source", source)
        out.insert(1, "step", step)
        out["reason"] = reason
        self.rows.append(out)

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows = pd.concat(self.rows, ignore_index=True) if self.rows else pd.DataFrame()
        return pd.DataFrame(self.steps), rows


def column_availability() -> pd.DataFrame:
    """Which of the wanted OWID columns exist in the codebook and in the downloaded data."""
    code = pd.read_csv(RAW / "owid-co2-codebook.csv")
    data_cols = set(pd.read_csv(RAW / "owid-co2-data.csv", nrows=0).columns)
    in_code = set(code["column"])
    rows = [{"column": c, "in_codebook": c in in_code, "in_data": c in data_cols,
             "status": "used" if c in data_cols else "NOT AVAILABLE (dropped from feature list)"}
            for c in OWID_WANTED]
    return pd.DataFrame(rows)


def load_wb_metadata(log: CleaningLog) -> tuple[pd.DataFrame, set[str]]:
    m = pd.read_csv(RAW / "wb_country_metadata.csv", keep_default_na=False)
    agg = m[m["region"] == "Aggregates"]
    countries = m[m["region"] != "Aggregates"].copy()
    log.step("wb_meta", "drop World Bank aggregates (region == Aggregates)", len(m), len(countries))
    log.rows_dropped(agg, "wb_metadata", "wb_meta", "aggregate", ["iso3", "name"])
    return countries[["iso3", "name", "region", "income_group"]].reset_index(drop=True), set(agg["iso3"])


def load_wb_indicators(countries: pd.DataFrame, agg_codes: set[str], log: CleaningLog) -> pd.DataFrame:
    frames = []
    for code, col in WB_COLS.items():
        d = pd.read_csv(RAW / f"wb_{code}.csv", keep_default_na=False)
        d["value"] = pd.to_numeric(d["value"].replace("", np.nan))
        n0 = len(d)
        blank = d[d["iso3"] == ""]
        aggs = d[d["iso3"].isin(agg_codes)]
        keep = d[d["iso3"].isin(set(countries["iso3"]))]
        log.rows_dropped(blank, f"wb_{code}", "wb_ind", "aggregate (blank ISO3)", ["wb_name", "year"])
        log.rows_dropped(aggs, f"wb_{code}", "wb_ind", "aggregate (region == Aggregates)", ["iso3", "year"])
        other = d[~d.index.isin(blank.index) & ~d.index.isin(aggs.index) & ~d.index.isin(keep.index)]
        log.rows_dropped(other, f"wb_{code}", "wb_ind", "ISO3 not in country metadata", ["iso3", "year"])
        log.step(f"wb_ind:{code}", "keep country rows only", n0, len(keep),
                 f"{int(keep['value'].isna().sum()):,} null values kept as missing (not imputed)")
        if keep.duplicated(["iso3", "year"]).any():
            raise ValueError(f"duplicate (iso3, year) keys in World Bank {code}")
        frames.append(keep[["iso3", "year", "value"]].rename(columns={"value": col}))
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["iso3", "year"], how="outer", validate="one_to_one")
    return out


def load_owid(log: CleaningLog) -> pd.DataFrame:
    cols = [c for c in OWID_WANTED if c in pd.read_csv(RAW / "owid-co2-data.csv", nrows=0).columns]
    o = pd.read_csv(RAW / "owid-co2-data.csv", usecols=cols)
    n0 = len(o)
    o = o[(o["year"] >= Y0) & (o["year"] <= Y1)].copy()
    log.step("owid_years", f"restrict to {Y0}-{Y1}", n0, len(o))
    kos = o["iso_code"].isna() & o["country"].isin(KOSOVO_CROSSWALK)
    log.rows_dropped(o[kos], "owid", "owid_crosswalk", "Kosovo: iso_code NaN -> XKX (changed)", ["country", "year"])
    o.loc[kos, "iso_code"] = o.loc[kos, "country"].map(KOSOVO_CROSSWALK)
    log.step("owid_crosswalk", "manual crosswalk Kosovo -> XKX", int(kos.sum()), 0, "rows changed (not dropped)")
    noiso = o["iso_code"].isna()
    log.rows_dropped(o[noiso], "owid", "owid_agg", "aggregate / non-country (no ISO3)", ["country", "year"])
    o2 = o[~noiso].rename(columns={"iso_code": "iso3", "co2": "co2_total_mt", "co2_per_capita": "co2_pc",
                                   "population": "population_owid"})
    log.step("owid_agg", "drop OWID rows with no ISO3 (aggregates, regions, shipping, aviation...)", len(o), len(o2),
             f"{o.loc[noiso, 'country'].nunique()} non-country entities")
    if o2.duplicated(["iso3", "year"]).any():
        raise ValueError("duplicate (iso3, year) keys in OWID")
    return o2.drop(columns="country").reset_index(drop=True)


def build_panel(log: CleaningLog) -> pd.DataFrame:
    countries, agg_codes = load_wb_metadata(log)
    wb = load_wb_indicators(countries, agg_codes, log)
    owid = load_owid(log)

    merged = owid.merge(wb, on=["iso3", "year"], how="outer", indicator="_join", validate="one_to_one")
    jc = merged["_join"].value_counts().to_dict()
    log.step("join", "outer join OWID x World Bank on (iso3, year)", len(owid) + len(wb), len(merged),
             f"both={jc.get('both', 0):,}, owid_only={jc.get('left_only', 0):,}, wb_only={jc.get('right_only', 0):,}")
    owid_only = merged[merged["_join"] == "left_only"].drop_duplicates("iso3")
    log.rows_dropped(owid_only.assign(src="owid_only"), "join", "join",
                     "OWID country without World Bank record (no income group / GDP)", ["iso3", "src"])
    join_report = pd.DataFrame({
        "category": ["both", "owid_only", "wb_only", "total"],
        "country_year_rows": [jc.get("both", 0), jc.get("left_only", 0), jc.get("right_only", 0), len(merged)],
        "countries": [merged.loc[merged["_join"] == k, "iso3"].nunique() for k in ["both", "left_only", "right_only"]]
        + [merged["iso3"].nunique()],
    })
    join_report.to_csv(RES / "join_report.csv", index=False)

    panel = merged.merge(countries.rename(columns={"name": "wb_name"}), on="iso3", how="left", validate="many_to_one")

    for col in ["urban_pct", "industry_pct"]:
        bad = panel[col].notna() & ~panel[col].between(0, 100)
        log.rows_dropped(panel[bad], "panel", "range_check", f"{col} outside 0-100 -> set to NaN", ["iso3", "year"])
        log.step(f"range:{col}", "shares outside 0-100 set to NaN", int(panel[col].notna().sum()),
                 int(panel[col].notna().sum() - bad.sum()))
        panel.loc[bad, col] = np.nan

    for col in ["co2_pc", "gdp_pc"]:
        nonpos = panel[col].notna() & (panel[col] <= 0)
        log.rows_dropped(panel[nonpos], "panel", "log_guard", f"{col} <= 0: excluded from log transforms (value kept)",
                         ["iso3", "year"])
        log.step(f"logguard:{col}", "non-positive values excluded from log transforms", int(panel[col].notna().sum()),
                 int(panel[col].notna().sum() - nonpos.sum()))

    panel["population"] = panel["population_wb"].combine_first(panel["population_owid"])
    panel = panel.drop(columns=["_join", "population_wb", "population_owid"])
    return panel.sort_values(["iso3", "year"]).reset_index(drop=True)


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    avail = column_availability()
    avail.to_csv(RES / "column_availability.csv", index=False)
    print(avail.to_string(index=False))
    log = CleaningLog()
    panel = build_panel(log)
    steps, rows = log.frames()
    panel.to_csv(PROC / "panel.csv", index=False)
    steps.to_csv(RES / "cleaning_log.csv", index=False)
    rows.to_csv(PROC / "cleaning_log_rows.csv", index=False)
    print(f"\npanel {panel.shape}; countries {panel['iso3'].nunique()}; row-level log {len(rows):,} rows")


if __name__ == "__main__":
    main()
