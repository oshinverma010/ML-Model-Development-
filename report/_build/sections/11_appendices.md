## Appendices

### Appendix A. Reproduction steps

All commands run from the repository root on Windows, macOS or Linux (activate the environment with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` elsewhere).

```
python -m venv .venv
pip install -r requirements.txt
python src/acquire.py              # downloads and caches raw data, writes the manifest
python src/clean.py                # cleaning, join report, column availability
python src/features.py             # label, features, sample flow
python src/evaluate_cv.py          # EDA on the pool, CV, selection, freeze (about 8 minutes)
python src/evaluate_test.py        # the single test run (about 6 minutes); refuses to run twice
python src/posthoc_test_analysis.py   # exploratory comparisons from the saved test predictions
python src/make_figures.py         # 14 figures
pytest                             # integrity tests
python report/_build/build_report.py   # regenerates report/report_content.md from results/*.csv
```

The test script is blocked by `results/test_run_lock.json`. To repeat it on a fresh clone, delete that file by hand, which is a deliberate, visible act. `python src/evaluate_test.py --rehearsal` runs the same code on a pseudo-split of the pool, for debugging only.

### Appendix B. Hyper-parameter search spaces

Each search drew 30 random combinations (`RandomizedSearchCV`, `random_state = 42`) and scored them by mean PR-AUC across the five rolling-origin folds. The logistic regression space has 66 combinations, the random forest 120 and HGB 1,728 or more.

{{table:grids}}

### Appendix C. Full feature catalogue

Summary statistics are computed on the training pool only.

{{table:feature_stats}}

### Appendix D. Integrity tests

The suite in `tests/test_pipeline_integrity.py` has {{n_tests|i}} checks, all passing.

{{table:tests_list}}
