"""Copy the handful of Week 2 numbers quoted in the Introduction into results/week2_context.csv.

Reads the read-only clone reference/week2 (git-ignored). Run once; the CSV is committed so this repo stays self-contained.
Source files: results/hypothesis_results.csv and results/h6_contingency_and_residuals.csv of
https://github.com/KanishkVerma-046/co2-gdp-hypothesis-testing
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
W2 = ROOT / "reference" / "week2" / "results"
h = pd.read_csv(W2 / "hypothesis_results.csv")
h6 = pd.read_csv(W2 / "h6_contingency_and_residuals.csv")
rows = []


def add(key, value, source):
    rows.append({"key": key, "value": value, "source": source})


r = h[(h.hypothesis == "H1") & (h.variant == "primary")].iloc[0]
add("h1_rho", float(r["effect_size"]), "hypothesis_results.csv H1 primary"); add("h1_n", int(r["n"]), "hypothesis_results.csv H1 primary")
r = h[(h.hypothesis == "H3") & (h.variant == "primary")].iloc[0]
add("h3_ratio", float(r["comment"].split("geometric-mean ratio=")[1].split("x")[0]), "hypothesis_results.csv H3 primary comment")
dec = h6[h6["outcome"] == "Decoupled"].set_index("income_group")["row_pct_decoupled"]
for g, v in dec.items():
    add("decoupled_pct_" + g.lower().replace(" ", "_"), float(v), "h6_contingency_and_residuals.csv")
r = h[(h.hypothesis == "H4") & (h.variant == "primary")].iloc[0]
add("h4_p_raw", float(r["p_raw"]), "hypothesis_results.csv H4 primary")
r = h[(h.hypothesis == "H4") & (h.variant == "robustness: excluding population < 1 million")].iloc[0]
add("h4_p_excl_small", float(r["p_raw"]), "hypothesis_results.csv H4 robustness excluding population < 1 million")
pd.DataFrame(rows).to_csv(ROOT / "results" / "week2_context.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
