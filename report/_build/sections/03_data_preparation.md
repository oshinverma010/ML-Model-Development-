## 3. Data preparation

### 3.1 Cleaning rules carried over from Weeks 1 and 2

`src/clean.py` is a port of the Week 2 code (repositories linked in the README). Aggregates are removed (OWID rows without ISO3, World Bank region "Aggregates"); the join uses ISO3 plus year, never names, with one manual Kosovo crosswalk (`XKX`); legitimate extremes such as Qatar are kept; shares outside 0-100 become missing; logarithms use strictly positive values only; and every dropped or changed row is logged (Table 3.1). A few early CO2 values are zero (for example Kosovo in the 1990s); they stay in the panel, become missing in log-based features, and exclude rows whose label depends on them.

**Table 3.1: Cleaning log (step totals from `results/cleaning_log.csv`).**

{{table:cleaning_log}}

### 3.2 Missingness audit

The audit uses only pool rows. {{mrows_share_rows_with_any_missing|p0}} of pool rows have at least one missing feature, and {{mt_n_features_missing|i}} of the 20 features have some gaps (Figure 3.1, Table 3.2).

![Figure 3.1: Share of pool rows with a missing feature value, by window start year (left) and by World Bank income group (right: High, Upper-middle, Lower-middle, Low, with row counts under each label). Only features with gaps are shown.](../figures/fig_missingness_heatmap.png)

**Table 3.2: Missingness in the pool and its relation to observed country characteristics.**

{{table:missingness}}

Data are **missing completely at random (MCAR)** if missingness is unrelated to anything, **missing at random (MAR)** if it depends only on observed variables such as income group, and **missing not at random (MNAR)** if it depends on the missing value itself or on unobserved circumstances. The gaps here are not MCAR: {{mt_n_features_income_sig|i}} of the {{mt_n_features_missing|i}} affected features differ significantly by income group, and the larger gaps (fuel shares, energy intensity) are concentrated in poorer countries. Small states and conflict-affected countries are probably MNAR, which cannot be tested and which no imputation fully repairs. The response is to give models the fact of missingness (indicator flags) and to show subgroup results (Section 8.7).

### 3.3 Imputation options

**Imputation** fills a missing value with an estimate. Four options were compared by cross-validation: (1) **median plus missing-indicator flags**, using the training-fold median and an extra 0/1 column; (2) **within-country forward fill (at most two years), then median**, which uses only past values and is applied to raw level series other than CO2 and GDP; (3) **iterative imputation** (`IterativeImputer` with `BayesianRidge`), which predicts each gappy feature from the others; and (4) **native NaN handling** in `HistGradientBoostingClassifier`, which learns which branch missing values should follow.

### 3.4 Scaling, transformation and outliers

Skewed positive variables enter as natural logarithms, and growth and momentum features are differences of logarithms. The pipeline **winsorises** each feature, clipping values below the 1st or above the 99th percentile to those percentiles, with limits learned on the training fold only. `StandardScaler` (subtract the mean, divide by the standard deviation) is used for the logistic regression only, because its penalty needs comparable scales. Tree models split on thresholds and are unchanged by monotone rescaling.

### 3.5 Class imbalance

About three rows in four are negatives, so always predicting "no decoupling" scores high accuracy. Two remedies were compared: no weighting, and `class_weight="balanced"`, which weights each row's loss by the inverse frequency of its class. **SMOTE** (synthetic minority oversampling) was deliberately not used: it interpolates between neighbouring minority rows, and on a panel those neighbours come from different countries and years, which blurs the temporal structure and can place information from later years beside earlier ones.

### 3.6 Encoding and the pipeline

All 20 features are numeric, so no encoding is needed; country, income group and region are not features (Section 4.3), and a categorical feature would be one-hot encoded inside the pipeline.

Every step that learns from data (filter, winsorising limits, imputer, scaler, model) belongs to one scikit-learn `Pipeline`, refitted on each training fold, so it never sees validation or test rows. The Winsorizer is representative.

{{code:src/models.py:Winsorizer}}
