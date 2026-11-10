## 7. Implementation

### 7.1 Repository and order of work

`src/acquire.py` and `src/clean.py` build the data, `src/features.py` features, labels and splits, `src/models.py` the pipelines, `src/metrics.py` metrics and bootstrap, `src/evaluate_cv.py` the training-period analysis and selection, `src/evaluate_test.py` the single test run and `src/make_figures.py` the figures. The commit history shows the order: plan first, then acquisition and cleaning, features and splits, cross-validation with the frozen configuration, and last the test run, figures and report. Numbers in this document are not typed by hand: `report/_build/build_report.py` splices them from the result files and fails if a value does not exist.

### 7.2 Pipeline architecture

The factory chains near-duplicate filter, winsorising, imputation, scaling (logistic regression only), an optional L1 selector and the classifier. The one custom selector is fitted on the training fold only.

{{code:src/models.py:make_pipeline}}

{{code:src/models.py:CorrelationFilter}}

### 7.3 Time-aware folds

This generator enforces the purge gap: each validation block keeps only training rows whose three-year label window ends before the block begins.

{{code:src/features.py:rolling_origin_folds}}

### 7.4 Search and freeze

`RandomizedSearchCV` receives the rolling-origin index pairs as its `cv` argument, so tuning uses exactly the planned folds.

```python
# from src/evaluate_cv.py (stage 2, abridged)
search = RandomizedSearchCV(M.make_pipeline(m, imp, cw), M.GRIDS[m], n_iter=M.N_ITER,
                            scoring="average_precision", cv=folds, random_state=M.SEED,
                            n_jobs=N_JOBS, refit=False, error_score="raise")
search.fit(P[F.FEATURES], y)
```

Out-of-fold predictions then feed the selection rule and the F1 threshold, and everything is written to `results/frozen_config.json` with a hash of the out-of-fold file, which the test script checks before it starts.

### 7.5 Integrity tests and deviations

A pytest suite of {{n_tests|i}} checks covers data hygiene, label construction on hand-made examples, absence of look-ahead, the split design, fitting on training folds only, reproducibility and the single-test-run discipline (Appendix D); all pass on the final commit. Changes to the plan are allowed only in a dated section appended to it. Those entries are listed in full below; the first five were made before the test run and none touched the test set.

{{table:deviations}}

The environment was Python {{python_version}} with scikit-learn {{sklearn_version}} and pandas {{pandas_version}} on Windows 11; exact versions are pinned in `requirements.txt`.
