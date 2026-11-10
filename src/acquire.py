"""Download raw data (OWID CO2 + codebook, World Bank indicators and country metadata) into data/raw.

Rules
-----
* Raw files are cached: an existing file is NEVER overwritten (delete it by hand to force a refresh).
* data/raw/manifest.json records source URL, download time (UTC), size and SHA-256 of each file.
* World Bank calls are retried with back-off; every retry is printed.
* Years 1988-2022 are requested: 1990-2022 is the study window, 1988-1989 are look-back only
  (5-year features at t = 1993 need levels back to 1988). See docs/ml_plan.md section 2.

Cleaning rules and the structure of this module are carried over from Week 2
(https://github.com/KanishkVerma-046/co2-gdp-hypothesis-testing).
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "manifest.json"

OWID_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
OWID_CODEBOOK_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-codebook.csv"
WB_BASE = "https://api.worldbank.org/v2"
INDICATORS = {
    "NY.GDP.PCAP.KD": "gdp_pc",
    "SP.POP.TOTL": "population_wb",
    "SP.URB.TOTL.IN.ZS": "urban_pct",
    "NV.IND.TOTL.ZS": "industry_pct",
}
YEAR_RANGE = "1988:2022"
HEADERS = {"User-Agent": "week3-ml-pipeline-coursework/1.0"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def _record(manifest: dict, path: Path, url: str, note: str = "") -> None:
    manifest[path.name] = {
        "url": url,
        "downloaded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "note": note,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _get(url: str, params: dict | None = None, tries: int = 5, timeout: int = 120) -> requests.Response:
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:  # network error, HTTP error, timeout
            wait = 2 ** attempt
            print(f"  [retry {attempt}/{tries}] {url} -> {type(exc).__name__}: {exc}; sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after {tries} attempts")


def fetch_file(url: str, dest: Path, manifest: dict) -> Path:
    if dest.exists():
        print(f"cached: {dest.name}")
        return dest
    print(f"downloading {dest.name} ...")
    dest.write_bytes(_get(url).content)
    _record(manifest, dest, url)
    return dest


def _wb_paginated(url: str, params: dict) -> list[dict]:
    rows: list[dict] = []
    page = 1
    while True:
        payload = _get(url, {**params, "format": "json", "per_page": 20000, "page": page}).json()
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            raise RuntimeError(f"Unexpected World Bank response for {url}: {str(payload)[:300]}")
        meta, data = payload
        rows.extend(data)
        if page >= int(meta["pages"]):
            break
        page += 1
    return rows


def fetch_wb_indicator(code: str, manifest: dict) -> Path:
    dest = RAW / f"wb_{code}.csv"
    if dest.exists():
        print(f"cached: {dest.name}")
        return dest
    print(f"downloading World Bank indicator {code} ...")
    url = f"{WB_BASE}/country/all/indicator/{code}"
    rows = _wb_paginated(url, {"date": YEAR_RANGE})
    df = pd.DataFrame({
        "iso3": [r["countryiso3code"] for r in rows],
        "wb_id": [r["country"]["id"] for r in rows],
        "wb_name": [r["country"]["value"] for r in rows],
        "year": [int(r["date"]) for r in rows],
        "value": [r["value"] for r in rows],
    })
    df.to_csv(dest, index=False)
    _record(manifest, dest, url + f"?date={YEAR_RANGE}", note=f"{len(df)} rows incl. aggregates and nulls")
    return dest


def fetch_wb_countries(manifest: dict) -> Path:
    dest = RAW / "wb_country_metadata.csv"
    if dest.exists():
        print(f"cached: {dest.name}")
        return dest
    print("downloading World Bank country metadata ...")
    url = f"{WB_BASE}/country"
    rows = _wb_paginated(url, {})
    df = pd.DataFrame({
        "wb_id": [r["id"] for r in rows],
        "iso3": [r["id"] for r in rows],
        "iso2": [r["iso2Code"] for r in rows],
        "name": [r["name"] for r in rows],
        "region_id": [r["region"]["id"] for r in rows],
        "region": [r["region"]["value"].strip() for r in rows],
        "income_id": [r["incomeLevel"]["id"] for r in rows],
        "income_group": [r["incomeLevel"]["value"] for r in rows],
    })
    df.to_csv(dest, index=False)
    _record(manifest, dest, url, note=f"{len(df)} rows incl. aggregates")
    return dest


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    fetch_file(OWID_URL, RAW / "owid-co2-data.csv", manifest)
    fetch_file(OWID_CODEBOOK_URL, RAW / "owid-co2-codebook.csv", manifest)
    fetch_wb_countries(manifest)
    for code in INDICATORS:
        fetch_wb_indicator(code, manifest)
    print("\nManifest:")
    for name, info in _load_manifest().items():
        print(f"  {name:32s} {info['downloaded_utc']}  {info['bytes']:>10,d} bytes  sha256={info['sha256'][:12]}...")


if __name__ == "__main__":
    main()
