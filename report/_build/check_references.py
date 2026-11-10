"""Verify every candidate reference against the Crossref API (title, journal, year, volume, pages, DOI).

Writes results/reference_check.csv. A reference is VERIFIED only if Crossref returns a work whose DOI equals the
expected DOI and whose title matches the expected title (case-insensitive, punctuation ignored).
The report cites only references with status VERIFIED.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
HEADERS = {"User-Agent": "week3-ml-coursework-reference-check/1.0 (mailto:)"}

# key, expected DOI, expected title (as the author would cite it)
CANDIDATES = [
    ("Breiman2001", "10.1023/A:1010933404324", "Random Forests"),
    ("Friedman2001", "10.1214/aos/1013203451", "Greedy function approximation: A gradient boosting machine"),
    ("ZouHastie2005", "10.1111/j.1467-9868.2005.00503.x", "Regularization and variable selection via the elastic net"),
    ("Pedregosa2011", "", "Scikit-learn: Machine Learning in Python"),
    ("DavisGoadrich2006", "10.1145/1143844.1143874", "The relationship between Precision-Recall and ROC curves"),
    ("SaitoRehmsmeier2015", "10.1371/journal.pone.0118432",
     "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets"),
    ("Cerqueira2020", "10.1007/s10994-020-05910-7",
     "Evaluating time series forecasting models: an empirical study on performance estimation methods"),
    ("BergmeirBenitez2012", "10.1016/j.ins.2011.12.028", "On the use of cross-validation for time series predictor evaluation"),
    ("NiculescuMizilCaruana2005", "10.1145/1102351.1102430", "Predicting good probabilities with supervised learning"),
    ("ChiccoJurman2020", "10.1186/s12864-019-6413-7",
     "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation"),
    ("Efron1979", "10.1214/aos/1176344552", "Bootstrap Methods: Another Look at the Jackknife"),
    ("Shmueli2010", "10.1214/10-STS330", "To Explain or to Predict?"),
    ("Roberts2017", "10.1111/ecog.02881",
     "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure"),
    ("KapoorNarayanan2023", "10.1016/j.patter.2023.100804",
     "Leakage and the reproducibility crisis in machine-learning-based science"),
    ("Haberl2020", "10.1088/1748-9326/ab842a",
     "A systematic review of the evidence on decoupling of GDP, resource use and GHG emissions, part II: synthesizing the insights"),
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def fetch(doi: str, title: str) -> dict | None:
    if doi:
        r = requests.get(f"https://api.crossref.org/works/{doi}", headers=HEADERS, timeout=60)
        if r.status_code == 200:
            return r.json()["message"]
    r = requests.get("https://api.crossref.org/works", params={"query.bibliographic": title, "rows": 3},
                     headers=HEADERS, timeout=60)
    r.raise_for_status()
    items = r.json()["message"]["items"]
    for it in items:
        t = (it.get("title") or [""])[0]
        if norm(t) == norm(title):
            return it
    return None


def main() -> None:
    rows = []
    for key, doi, title in CANDIDATES:
        try:
            m = fetch(doi, title)
        except requests.RequestException as exc:
            rows.append({"key": key, "status": f"NOT VERIFIED (request failed: {type(exc).__name__})"})
            continue
        if m is None:
            rows.append({"key": key, "expected_doi": doi, "status": "NOT VERIFIED (no matching Crossref record)"})
            continue
        got_title = (m.get("title") or [""])[0]
        authors = "; ".join(f"{a.get('family', '')}, {a.get('given', '')}" for a in m.get("author", []))
        issued = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
        container = (m.get("container-title") or [""])[0]
        title_ok = norm(got_title) == norm(title)
        doi_ok = (not doi) or m.get("DOI", "").lower() == doi.lower()
        rows.append({"key": key, "expected_doi": doi, "crossref_doi": m.get("DOI", ""), "title": got_title,
                     "authors": authors, "container": container, "year": issued, "volume": m.get("volume", ""),
                     "issue": m.get("issue", ""), "pages": m.get("page", m.get("article-number", "")),
                     "title_match": title_ok, "doi_match": doi_ok,
                     "status": "VERIFIED" if (title_ok and doi_ok) else "NOT VERIFIED (title or DOI mismatch)"})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "reference_check.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(out[["key", "status", "crossref_doi", "container", "year", "volume", "pages"]].to_string(index=False))


if __name__ == "__main__":
    main()
