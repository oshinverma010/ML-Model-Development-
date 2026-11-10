"""Check that every external URL used in the repository answers, and save the outcome to results/url_check.csv.

Checked: the data URLs, the API endpoints, the Week 1 and Week 2 repositories, and every reference DOI in the report.
(Only a status code is recorded; the page content is not used.)
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
HEADERS = {"User-Agent": "week3-ml-coursework-url-check/1.0"}
FIXED = [
    ("OWID CO2 data", "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"),
    ("OWID CO2 codebook", "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-codebook.csv"),
    ("World Bank API (GDP per capita)", "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.KD?date=2022&format=json&per_page=1"),
    ("World Bank API (country metadata)", "https://api.worldbank.org/v2/country?format=json&per_page=1"),
    ("Crossref API", "https://api.crossref.org/works/10.1214/aos/1176344552"),
    ("Week 1 repository", "https://github.com/KanishkVerma-046/co2-economy-eda-proposal"),
    ("Week 2 repository", "https://github.com/KanishkVerma-046/co2-gdp-hypothesis-testing"),
]


def status(url: str) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=60, allow_redirects=True, stream=True)
        code = r.status_code
        r.close()
        return f"HTTP {code}"
    except requests.RequestException as exc:
        return f"FAILED ({type(exc).__name__})"


def main() -> None:
    rows = [{"what": w, "url": u} for w, u in FIXED]
    text = (ROOT / "report" / "report_content.md").read_text(encoding="utf-8")
    for doi in sorted(set(re.findall(r"https://doi\.org/\S+", text))):
        rows.append({"what": "reference DOI", "url": doi.rstrip(".")})
    for r in rows:
        r["outcome"] = status(r["url"])
        r["checked_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "url_check.csv", index=False)
    print(df[["what", "outcome"]].to_string(index=False))
    print(f"\n{(df['outcome'] == 'HTTP 200').sum()} of {len(df)} returned HTTP 200")


if __name__ == "__main__":
    main()
