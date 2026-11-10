## 6. Evaluation strategy

### 6.1 Splits: pool, purge and test

The split is by time: a random split would put a country's 2010 row in training and its 2011 row in validation, and since those windows share two years of outcomes the model would be graded partly on what it had seen. The design (Figure 2.1) is: **pool** (training and selection) with t = 1993 to 2013, labels ending by 2016; a **purge gap** of t = 2014 to 2016, unused because those windows would overlap the test period; and **test** with t = 2017, 2018, 2019 (labels ending 2020 to 2022). The test set informs no choice: not features, imputer, winsorising limits, model, hyper-parameters or threshold.

### 6.2 Cross-validation inside the pool

**Cross-validation (CV)** repeatedly splits training data into a fitting part and a scoring part, so choices can be made without the test set. Time-ordered data need blocks that respect time (Bergmeir & Benitez, 2012; Cerqueira et al., 2020), and grouped data such as countries make random splits too optimistic (Roberts et al., 2017). The primary scheme is **rolling-origin (expanding-window) CV** with five folds: validation blocks of three consecutive years (1999-2001 to 2011-2013) and, for a block starting in year v, training rows with t at most v minus 4, leaving a three-year purge gap. Table 6.1 lists the folds; the first has only {{fold_1_n_train|i}} training rows and a noisier score.

**Table 6.1: Rolling-origin folds (years are window start years).**

{{table:cv_folds}}

A **secondary check** uses `GroupKFold`: five folds with no country on both sides, which measures generalisation to unseen countries. It ignores time, so it is optimistic and never used to select.

### 6.3 Metrics

A **confusion matrix** counts true positives (TP), false positives (FP), true negatives (TN) and false negatives (FN) for hard predictions. Table 6.2 defines each metric. The **selection metric** is PR-AUC: the precision-recall curve is more informative than the ROC curve when positives are the minority (Davis & Goadrich, 2006; Saito & Rehmsmeier, 2015), and a model with no skill scores the base rate. Accuracy is reported but flagged as misleading. The Matthews correlation coefficient is preferred to F1 or accuracy as a one-number summary because it uses all four cells of the confusion matrix (Chicco & Jurman, 2020).

**Table 6.2: Metrics, definitions and purpose.**

{{table:metric_definitions}}

### 6.4 Decision threshold

A **threshold** turns a probability into a yes or no. The single rule is: take the value that maximises F1 on the **pooled out-of-fold predictions** of the CV (every pool row predicted by a model that did not train on it), compute it once and freeze it in `results/frozen_config.json` before the test run. The rule has a known weakness: when the model separates classes only weakly, the F1-maximising threshold falls so low that nearly every row is flagged. PR-AUC and ROC-AUC do not depend on the threshold, so they carry the main conclusions.

### 6.5 Calibration

A model is **calibrated** if, among rows given probability 0.3, about 30% are positive; some model families distort this (Niculescu-Mizil & Caruana, 2005). It is assessed with a reliability curve (10 equal-width bins), the **expected calibration error (ECE)** and the Brier score. No recalibration is applied.

### 6.6 Uncertainty

A row-level bootstrap would give intervals that are too narrow for dependent rows. The **bootstrap** re-estimates a statistic on many resamples (Efron, 1979); here whole countries are resampled with replacement, keeping every window of a chosen country, and all 95% intervals are percentile intervals from 2,000 resamples. To compare two models the same resampled countries score both and the **paired difference** in PR-AUC is summarised; an interval excluding zero means the difference is unlikely to be noise.

### 6.7 Subgroups, interpretation and ablation

On the test set, performance is split by World Bank income group (the *current* classification, used only for evaluation) and region, with row and positive counts shown. Interpretation uses permutation importance, standardised logistic coefficients and partial dependence (the average predicted probability as one feature varies while the others keep their observed values). Findings are at country level (an **ecological** reading that says nothing about individuals or regions inside a country), and income level is not independent of GDP, which is a feature. The **ablation** refits the chosen model with feature sets A, B, C and an L1-selected C, with fixed tuned hyper-parameters.

### 6.8 Robustness variants

Three variants are saved as separate files: a five-year horizon (CV with a five-year purge, pool ending at t = 2011, four folds); excluding the window ending in 2020 (t = 2017), which is affected by COVID-19; and excluding countries under one million people. The last two are subsets of the one test run.

### 6.9 Discipline around the test set

Only `src/evaluate_test.py` calls the function that returns the test years. It reads the frozen configuration, refuses to run if `results/test_run_lock.json` exists, and writes the lock when it finishes. Pytest checks fail if the CV code refers to the test accessor, and confirm that the configuration predates the lock and that the stored threshold reproduces the reported confusion matrix.
