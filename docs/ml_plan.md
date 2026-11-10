# ML Plan (pre-specification) - "Does Wealth Burn Carbon?" Week 3

**Status:** written and committed *before any model was fitted and before any data file was loaded into the project*.
The only things done before this commit were: creating the project skeleton, cloning the Week 1 and Week 2 repositories read-only,
installing packages, and reading the OWID *codebook* (column names and units only, no data rows) to learn which energy columns exist.
Later changes go only into the dated "Deviations" section at the end of this file; nothing above that section is rewritten.

Project: predicting absolute decoupling of CO2 per person from GDP per person, country-year panel, 1990-2022 (continuation of Weeks 1-2).

## 1. Problem and label definition

* **Task:** binary classification on a country-year panel.
* **Unit of observation:** country `c` in year `t` (called a window start year).
* **Label** (horizon `h = 3` years): `y(c,t) = 1` if `CO2pc(c,t+3) < CO2pc(c,t)` **and** `GDPpc(c,t+3) > GDPpc(c,t)`; otherwise `0`.
  This is *absolute decoupling*: emissions per person fall while income per person rises.
* **CO2pc** = OWID `co2_per_capita` (fossil fuels and industry, production-based, tonnes per person).
  **GDPpc** = World Bank `NY.GDP.PCAP.KD` (constant 2015 US$).
* **Eligibility of a row:** CO2pc and GDPpc must be observed (and strictly positive, for the log guard) at `t`, and both must be observed at `t+h`. These core series are never imputed. Rows failing this are dropped and counted in `results/sample_flow.csv`.
* **Features** use only information available at the end of year `t` (values at `t`, `t-1`, ... never `t+1` or later).
* **Interpretation:** the aim is to *predict* decoupling and to see which structural features are *associated* with it. This is predictive association, not causal inference. No causal claim is made anywhere in this project.
* **Base rate:** reported overall and per year (pool years in the EDA stage, test years only by the single test run).
* **Success criteria (fixed now):** (1) PR-AUC of the chosen model on the test set clearly above the prevalence line and above the persistence baseline, judged by a country-clustered 95% interval of the paired difference; (2) a pipeline that is leakage-safe by construction (pytest checks); (3) honest reporting if (1) fails. A modest result is acceptable.

## 2. Data

Same sources as Week 2, downloaded by `src/acquire.py` into `data/raw/` (never overwritten; `manifest.json` stores URL, UTC time, bytes, sha256).

* Our World in Data CO2 dataset (`owid-co2-data.csv`) and its codebook (`owid-co2-codebook.csv`).
* World Bank Indicators API v2: `NY.GDP.PCAP.KD`, `SP.POP.TOTL`, `SP.URB.TOTL.IN.ZS`, `NV.IND.TOTL.ZS`, and the country-metadata endpoint (income group, region).
* **Window:** study window 1990-2022. Because 5-year features need five years of history, the loaders also read 1988-1989 as **look-back only**. Modelling rows start at `t = 1993`; the last row with a 3-year label is `t = 2019` (label year 2022).
* **Cleaning rules ported from Weeks 1-2:** remove aggregates (OWID rows without ISO3, World Bank rows with region "Aggregates"); join on ISO3 + year only; a single manual Kosovo crosswalk (OWID "Kosovo" to `XKX`); a join report; logs only of strictly positive values (zeros documented, values kept); share variables outside 0-100 set to missing; every dropped or changed row logged.
* **OWID energy columns, confirmed from the codebook:** present: `energy_per_capita`, `energy_per_gdp`, `co2_per_unit_energy`, `coal_co2`, `oil_co2`, `gas_co2`, `cement_co2`, `flaring_co2`, `primary_energy_consumption`, `co2_per_gdp`. **Not present** in this dataset: `fossil_share_energy`, `low_carbon_share_energy`, `renewables_share_energy`. These three are therefore dropped from the feature list and recorded as unavailable (they live in OWID's separate energy dataset, which is not used).

## 3. Split design (leakage-safe and time-aware)

| Block | Years `t` | Label window ends | Use |
|---|---|---|---|
| Pool (train + selection) | 1993-2013 | 1996-2016 | all fitting, tuning, selection, threshold |
| Purge gap | 2014-2016 | 2017-2019 | unused (labels would overlap test features) |
| **Test** | **2017, 2018, 2019** | 2020, 2021, 2022 | evaluated once, after everything is frozen |

* The test set informs **no** choice: not features, imputer, scaler limits, model, hyperparameters, or threshold.
* **Rolling-origin CV (primary selection):** expanding window, 5 folds on the pool. Validation blocks: 1999-2001, 2002-2004, 2005-2007, 2008-2010, 2011-2013. For a validation block starting at year `v`, training rows are those with `t <= v - 4`, which leaves a 3-year purge gap (`v-3`, `v-2`, `v-1`) so no training label window reaches into the validation period.
* **Secondary check:** `GroupKFold` by country (5 folds) on the pool, to measure generalisation to *unseen countries*. It ignores time order, so it is optimistic about time and is used only as a check, never for selection.
* **COVID:** windows ending in 2020 or later are affected. Primary test is `t = 2017-2019`; a robustness variant excludes `t = 2017` (window ends 2020). The label base rate by year is reported.
* Rows within a country are strongly dependent (consecutive windows share two years). All uncertainty intervals therefore resample **countries**, not rows.

## 4. Features (at year t, past data only)

Raw level series are placed on a complete country x year grid before any lag is taken, so "t-3" always means three calendar years.
Notation: `ln` natural log of strictly positive values only (otherwise missing). Expected direction is the sign of the association with `P(y=1)`; it is a prior, not a result.

| Feature | Definition | Group | Why it matters | Expected direction |
|---|---|---|---|---|
| `ln_gdp_pc` | ln GDPpc(t) | economic | richer economies have more room for structural change | + |
| `gdp_g1` | ln GDPpc(t) - ln GDPpc(t-1) | economic | recent growth: growth must continue for y=1 | + |
| `gdp_cagr3` | [ln GDPpc(t) - ln GDPpc(t-3)] / 3 | economic | medium-run growth trend | + |
| `gdp_vol5` | std of the last five 1-year log growth rates | economic | unstable economies rarely decouple steadily | - |
| `ln_pop` | ln population(t) | economic | scale; weak prior | ? |
| `pop_g3` | [ln pop(t) - ln pop(t-3)] / 3 | economic | fast population growth raises total, and often per-person, energy demand needs | - |
| `urban_pct` | urban population share (%) | economic | urbanisation reflects development stage | ? |
| `industry_pct` | industry value added (% of GDP) | economic | industry-heavy economies are harder to decarbonise | - |
| `ln_co2_pc` | ln CO2pc(t) | emissions | high-emitters have more to cut (and a higher bar to fall from) | + |
| `co2_chg1`, `co2_chg3`, `co2_chg5` | ln CO2pc(t) - ln CO2pc(t-k), k = 1, 3, 5 | emissions (momentum) | recent falls tend to persist | - |
| `ln_co2_per_gdp` | ln [CO2pc(t) / GDPpc(t)] | emissions | carbon intensity of output (Kaya identity term) | - |
| `ci_chg3` | 3-year change of `ln_co2_per_gdp` | emissions | intensity momentum | - |
| `ln_energy_pc` | ln `energy_per_capita` | energy mix | energy use level | ? |
| `ln_energy_per_gdp` | ln `energy_per_gdp` | energy mix | energy intensity of output (Kaya term) | - |
| `co2_per_unit_energy` | OWID `co2_per_unit_energy` (g/kWh) | energy mix | carbon intensity of energy (Kaya term) | - |
| `coal_share`, `oil_share`, `gas_share` | `coal_co2/co2`, `oil_co2/co2`, `gas_co2/co2` | energy mix | which fossil fuel dominates; coal-heavy systems decouple less easily, but switching away from coal is a main route | coal -, gas + |

**Kaya link.** Because CO2pc = GDPpc x (energy / GDP) x (CO2 / energy), the energy-intensity and carbon-intensity features are the Kaya factors that mechanically drive the label. `ln_co2_per_gdp` is by construction `ln_co2_pc - ln_gdp_pc` and `ci_chg3` is `co2_chg3 - 3 * gdp_cagr3`; these exact linear identities are kept for the tree models and will be flagged by the VIF diagnostic (infinite VIF by identity). The elastic-net penalty handles the exact collinearity.
**Unavailable (confirmed absent from the codebook):** `fossil_share_energy`, `low_carbon_share_energy`, `renewables_share_energy`.

**Excluded on purpose:** calendar year (a time index would let a flexible model memorise year-specific base rates and would extrapolate into unseen test years), country identity (turns the task into memorising countries and prevents generalisation to new ones), and World Bank income group (its definition is a threshold on GNI per capita, so it is circular with the economic features; also the current classification is not year-specific, which leaks future development into past rows). Income group and region are used **only** for subgroup evaluation.

**Ablation feature sets:** A = momentum only (`co2_chg1`, `co2_chg3`, `co2_chg5`, `ci_chg3`, `gdp_g1`, `gdp_cagr3`); B = A + economic structure (`ln_gdp_pc`, `gdp_vol5`, `ln_pop`, `pop_g3`, `urban_pct`, `industry_pct`, `ln_co2_pc`, `ln_co2_per_gdp`); C = full set (B + energy and mix features that exist).

**Feature selection to demonstrate**
1. *Filter:* drop near-duplicates greedily in the order of the table above; a feature is dropped when |Spearman| with an already-kept feature exceeds 0.9 (fitted on the training fold inside the pipeline). VIF of the retained set is reported on the pool.
2. *Embedded:* L1 / elastic-net coefficients of the logistic model (zeroed features are "not selected"); also an L1-based `SelectFromModel` variant in the ablation.
3. *Permutation importance* of the chosen model on the validation folds (drop in PR-AUC when one feature is shuffled).
4. *Wrapper (RFE):* considered but not used: it refits the model many times inside every CV fold, it is unstable with correlated features, and it adds selection-bias risk; the embedded and permutation routes give the same information more cheaply.

## 5. Preprocessing (all inside scikit-learn `Pipeline`s, fitted on training folds only)

* **Missingness audit** by feature, by year, and by income group, shown as a heatmap on the **pool only**. Discuss MCAR / MAR / MNAR: small states and conflict countries are likely MNAR (the value is missing *because* of the country's circumstances), which means no imputer can fully repair it. Missing-indicator flags let a model use the *fact* of missingness.
* **Imputation options (compared by CV):**
  (i) median + missing-indicator flags (`SimpleImputer(add_indicator=True)`);
  (ii) within-country forward fill, limit 2 years (applied to raw level series, uses only past values, so it cannot leak), then median;
  (iii) `IterativeImputer` with `BayesianRidge`;
  (iv) native NaN handling in `HistGradientBoostingClassifier` (HGB only).
* **Transforms:** logs for skewed positive variables (already in the feature definitions); **winsorise** at the 1st and 99th percentiles with limits learned on the training fold; `StandardScaler` for the linear model only. Trees are invariant to monotone rescaling, so scaling would change nothing for them.
* **Imbalance:** compare `class_weight=None` and `class_weight="balanced"`. **SMOTE is not used.** On a time-ordered, country-clustered panel it would create synthetic rows that interpolate between different countries and different years, which blurs the temporal structure and can place synthetic information from the future next to the past.
* HGB's internal early stopping is **off** (it draws a random validation split from the training rows, ignoring time order).

## 6. Algorithms

Fixed `random_state = 42` everywhere. scikit-learn only.

**Baselines**
* (a) *Prior / majority:* constant score equal to the training base rate. PR-AUC equals prevalence.
* (b) *Persistence rule:* predict 1 if CO2pc fell **and** GDPpc rose over the previous 3 years (`co2_chg3 < 0` and `gdp_cagr3 > 0`); predict 0 if either is missing.

**Trained candidates**
1. Elastic-net logistic regression (`saga` solver).
2. Random forest.
3. `HistGradientBoostingClassifier`.

**Considered but not trained** (discussed in the report): SVM (RBF), k-nearest neighbours, a neural network (MLP).

**Hyperparameter search:** `RandomizedSearchCV`, `n_iter = 30`, scoring = average precision, run inside the same rolling-origin folds (a pre-split list of index pairs).

| Model | Grid |
|---|---|
| Elastic-net LR | `C` in {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100}; `l1_ratio` in {0.1, 0.3, 0.5, 0.7, 0.9, 1.0} |
| Random forest | `n_estimators` in {300, 500}; `max_depth` in {3, 5, 8, 12, None}; `min_samples_leaf` in {5, 10, 20, 40}; `max_features` in {"sqrt", 0.3, 0.5} |
| HGB | `learning_rate` in {0.01, 0.03, 0.05, 0.1}; `max_depth` in {2, 3, 4, 6}; `max_leaf_nodes` in {8, 15, 31}; `min_samples_leaf` in {20, 50, 100}; `l2_regularization` in {0, 0.1, 1, 10}; `max_iter` in {100, 200, 400} |

**Two-stage procedure**
* *Stage 1 (preprocessing choice, default hyperparameters):* for each model, compare imputation x class weighting (LR 3x2, RF 3x2, HGB 4x2 combinations).
* *Stage 2 (tuning):* randomised search of the grid above with the preprocessing chosen in stage 1.

## 7. Selection rules (fixed in advance)

* **Selection metric:** mean PR-AUC (average precision) across the five rolling-origin folds.
* **Stage-1 rule:** choose the combination with the highest mean PR-AUC; if combinations are within one standard error (SE) of the best, choose the simplest. Complexity order for imputation: native NaN < median + indicators < forward fill + median < iterative; weighting: none < balanced. (SE = fold standard deviation / sqrt(5) of the best combination.)
* **Model rule:** choose the model with the highest mean fold PR-AUC; if a simpler model is within one SE of the best, choose the simpler. Complexity order: elastic-net LR < random forest < HGB.
* **Deliverable:** the chosen model **plus** the interpretable logistic baseline kept for comparison.

## 8. Evaluation

* **Metrics:** PR-AUC (selection), ROC-AUC, accuracy (flagged as misleading under imbalance), balanced accuracy, precision, recall, F1, MCC, Brier score, calibration (reliability curve and expected calibration error, ECE, 10 equal-width bins).
* **Threshold rule (single):** the probability threshold that maximises F1 on the **pooled out-of-fold rolling-origin CV predictions** of the chosen model, computed once and frozen in `results/frozen_config.json` before the test run. It is not changed after seeing the test.
* **Test set (one run by `src/evaluate_test.py`):** confusion matrix (counts and row-normalised), ROC and PR curves with baselines, calibration plot, 95% CIs from a country-clustered bootstrap (2,000 resamples). A lock file stops the script from being run twice.
* **Model comparison uncertainty:** paired country-level bootstrap (2,000 resamples) of the PR-AUC difference on pooled out-of-fold predictions (chosen model vs persistence, vs LR, vs the other candidate models).
* **Subgroups on the test set:** by World Bank income group and by region, with `n` and number of positives shown and wide CIs where groups are small. Income group is the *current* classification, evaluation-only.
* **Interpretation:** permutation importance, standardised logistic coefficients, partial dependence of the top three features. Cautions: associations are ecological and predictive, and income group is circular with GDP.
* **Ablation:** feature sets A, B, C compared by CV (fixed tuned hyperparameters) and, in the single test run, by test metrics.
* **Robustness variants** (each saved to its own CSV):
  1. Horizon `h = 5`: re-run of the CV on the pool with its own 5-year purge (labels end 2016, so training uses `t <= 2011`); frozen hyperparameters. Not evaluated on the test set (only `t = 2017` would have a 5-year label).
  2. Excluding windows that end in 2020 (`t = 2017`): subset of the single test run.
  3. Excluding countries with population under 1 million: subset of the single test run (and CV).
* **Honesty rule:** if the chosen model does not clearly beat the persistence baseline, the report says so.

## 9. Risk register (pre-specified)

| Risk | Mitigation planned |
|---|---|
| Temporal leakage | Features from year `t` and earlier; pool/purge/test split; rolling-origin CV with purge; tests that perturbing the future does not change features |
| Label overlap between adjacent windows | 3-year purge gap; clustered bootstrap |
| Country leakage | Country-grouped secondary CV; clustered bootstrap; no country ID features |
| COVID shift in test windows | Base rate by year; sensitivity excluding `t = 2017`; honest framing |
| Class imbalance | PR-AUC, MCC, balanced accuracy; class-weight comparison; no SMOTE |
| MNAR missingness | Audit; missing-indicator flags; native NaN handling; sensitivity by group |
| Overfitting to CV | Small grids, `n_iter <= 30`, simplicity rule, test untouched |
| Selection bias from comparing many models | Pre-specified rule, 3 candidates, test used once |
| Calibration drift | Reliability curve, ECE, Brier on test; threshold from CV only |
| Small subgroup n | Show `n`; clustered CIs; no conclusions from tiny groups |
| Circular income groups | Evaluation only, never a feature |
| Ecological fallacy | Country-level conclusions only; no claim about individuals or sub-national units |
| Production-based accounting | Documented; trade-embedded emissions not modelled |
| Data revisions / vintage | Raw files hashed in the manifest; results tied to one download date |

## 10. Compute and reproducibility

* All randomness from `random_state = 42`; bootstrap uses `numpy.random.default_rng(42)`.
* Everything runs on CPU; commands are cross-platform.

---

## Deviations (appended after the plan was committed; dated)

**2026-11-05 (before the test run; none of these touched the test years)**

1. `src/evaluate_cv.py` was run twice. The first run wrote duplicate rows into `results/robustness_cv.csv` (the chosen model and the logistic baseline are the same model, so one row was written twice). The loop was fixed to de-duplicate and the script re-run. All fitted numbers, the selection and the frozen threshold were identical in both runs.
2. The saga solver of the elastic-net logistic regression reached `max_iter = 5000` without full convergence for some large-`C` candidates inside the randomised search (a convergence warning, suppressed in the logs). The grid was not changed; the selected configuration (`C = 3`) is reported as fitted.
3. `IterativeImputer` printed convergence warnings with `max_iter = 10` and `BayesianRidge`; they are suppressed and `max_iter` was not changed. The option did not win the pre-specified selection rule for any model.
4. `src/evaluate_test.py` has a `--rehearsal` mode that runs the same code on a pseudo-split carved out of the pool (train `t <= 2005`, pseudo-test `t = 2009-2011`) to debug the script without opening the test labels. Its outputs are git-ignored.
5. The plan describes a "simpler model within one standard error" rule. The standard error is the fold-to-fold standard deviation divided by the square root of five; with five temporally dependent folds this is a rough yardstick. It is used exactly as pre-specified.

**2026-11-05 (after the single test run; the pre-specified choice and threshold were not changed)**

6. The single test run (`src/evaluate_test.py`, lock file written 2026-11-05 12:47:59 UTC) selected nothing and changed nothing. Because the pre-specified rule chose the logistic regression, the script's list of paired test comparisons (chosen versus logistic baseline) was degenerate and omitted LR versus the random forest and HGB. These comparisons were added afterwards as a clearly labelled **post-hoc, exploratory** analysis by `src/posthoc_test_analysis.py`, which only reads the saved test predictions (no model is refitted and the test set is not re-evaluated). The same script scores the test rows with an income-group prior (the pool base rate of each World Bank income group) to show how much of the skill is group-level prevalence; income group is still not a model feature. Results: `results/test_posthoc.csv`.
7. `results/test_robustness.csv` and `results/test_paired_differences.csv` contain duplicated rows for the logistic regression, because the chosen model and the baseline are the same model and the script looped over both. The files were left as written by the single run; the report build removes exact duplicate rows when it reads them.

