# Does Wealth Burn Carbon? Week 3: Machine Learning Model Development and Evaluation Plan

Predicting **absolute decoupling** of CO2 per person from GDP per person, three years ahead, on a country-year panel (1990-2022). This repository is the Week 3 task of a four-task data-science virtual internship. It is a *plan first*, backed by a working implementation so the plan can be shown to be executable.

* **Question:** for country *c* and year *t*, will CO2 per person (OWID `co2_per_capita`) **fall** and GDP per person (World Bank `NY.GDP.PCAP.KD`) **rise** over the next three years?
* **Honest framing:** this is predictive association, not causal inference. Results are modest and are reported as such; see `report/report_content.md`, Section 8.

## Provenance: Weeks 1 and 2

The project is cumulative. **The cleaning rules were carried over from Weeks 1 and 2** (aggregate removal, ISO3 + year join, the Kosovo crosswalk, log guards, row-level logging), and the code in `src/acquire.py` and `src/clean.py` is a port of those repositories. Nothing is imported from them; this repository is self-contained.

* Week 1, Data Exploration and Visualization Strategy: https://github.com/KanishkVerma-046/co2-economy-eda-proposal
* Week 2, Statistical Analysis and Hypothesis Testing Framework: https://github.com/KanishkVerma-046/co2-gdp-hypothesis-testing
  (Week 2 numbers quoted in the report's introduction are copied into `results/week2_context.csv` by `report/_build/extract_week2_context.py`.)

## Data sources (downloaded by `src/acquire.py`, cached in `data/raw/`, never overwritten)

* Our World in Data CO2 dataset: `https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv` and its codebook.
* World Bank Indicators API v2: `NY.GDP.PCAP.KD`, `SP.POP.TOTL`, `SP.URB.TOTL.IN.ZS`, `NV.IND.TOTL.ZS`, and the country-metadata endpoint (income group, region).

No API key is needed. `data/raw/manifest.json` records URL, UTC time, size and SHA-256 of each file. Years 1988-2022 are loaded (1990-2022 is the study window; 1988-1989 are look-back only, so that 5-year features exist from t = 1993).

## Project workflow and provenance

The supplied ZIP does not include Git history, so this snapshot cannot independently verify the original commit order. The plan and workflow below are described by the project materials.

1. Review `docs/ml_plan.md`, which records the pre-specification and dated deviations.
2. Acquire and clean the data, then build features and labels with leakage-safe splits.
3. Compare models on the **training pool only** (window start years t = 1993-2013) using rolling-origin cross-validation with a 3-year purge gap; freeze the model, hyper-parameters, and decision threshold in `results/frozen_config.json`.
4. Evaluate once on held-out test years (t = 2017-2019) with `src/evaluate_test.py`; `results/test_run_lock.json` blocks a second evaluation.
5. Run clearly labelled post hoc analysis on the saved test predictions, figures, report, and integrity checks.

## Repository layout

```
README.md, requirements.txt, pytest.ini
docs/ml_plan.md                  analysis pre-specification + dated Deviations
src/acquire.py, clean.py         data download and cleaning (ported from Weeks 1-2)
src/features.py                  feature engineering, label, pool / purge / test split, rolling-origin folds
src/models.py                    scikit-learn pipelines, custom transformers, search grids, baselines
src/metrics.py                   metrics, F1 threshold rule, country-clustered bootstrap
src/evaluate_cv.py               EDA on the pool, CV, tuning, selection rule, freeze
src/evaluate_test.py             the single test run
src/posthoc_test_analysis.py     exploratory comparisons from the saved test predictions
src/make_figures.py              14 figures (PNG, 200 dpi, colour-blind-safe)
tests/test_pipeline_integrity.py pytest integrity checks
notebooks/03_model_pipeline.ipynb  executed walk-through (reads saved test results; does not re-evaluate the test set)
data/raw, data/processed         raw downloads and manifest; cleaned panel, logs, predictions
results/*.csv                    every number used in the report
figures/*.png                    figures used in the report
report/report_content.md         the report (generated; Markdown, LF line endings)
report/report_notes.md           private author notes; excluded from this public repository
report/_build/                   template sections + scripts that splice numbers from results/*.csv into the report
```

## How to reproduce

```
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/acquire.py                  # about 1 minute; cached on later runs
python src/clean.py
python src/features.py
python src/evaluate_cv.py              # about 8 minutes (8 cores); writes results/frozen_config.json
python src/evaluate_test.py            # about 6 minutes; the ONE test run, refuses to run twice
python src/posthoc_test_analysis.py
python src/make_figures.py
pytest                                 # integrity tests
jupyter nbconvert --to notebook --execute --inplace notebooks/03_model_pipeline.ipynb
python report/_build/build_report.py   # regenerates report/report_content.md from results/*.csv
```

Seeds are fixed (`random_state = 42`), so repeated runs give identical tables. The test run is locked: to repeat it on a fresh clone, delete `results/test_run_lock.json` by hand. `python src/evaluate_test.py --rehearsal` runs the same code on a pseudo-split of the pool, for debugging. Developed with Python 3.13.3 on Windows 11; exact package versions are in `requirements.txt`.

## Where to find things

* Plan and deviations: `docs/ml_plan.md`
* Sample flow and cleaning logs: `results/sample_flow.csv`, `results/cleaning_log.csv`, `data/processed/cleaning_log_rows.csv`
* Missingness, correlation and VIF (pool only): `results/missingness_*.csv`, `results/feature_spearman.csv`, `results/vif.csv`
* Model comparison and selection: `results/stage1_*.csv`, `results/tuning_results.csv`, `results/cv_*.csv`, `results/model_selection.csv`
* Test results: `results/test_*.csv`, `results/test_posthoc.csv`
* Reference verification: `results/reference_check.csv` (Crossref), `results/url_check.csv`
