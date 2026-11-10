## 5. Algorithm selection

### 5.1 Candidates and baselines

Two baselines set the bar. The **prior (majority) baseline** gives every row the training base rate, so its PR-AUC equals the share of positives. The **persistence rule** predicts decoupling when CO2 per person fell and GDP per person rose over the previous three years (and no decoupling if either is unknown); a learned model is only useful if it beats it. Three algorithms were trained and three more considered but not trained, all from scikit-learn with `random_state = 42` (Table 5.1).

**Table 5.1: Algorithm comparison.**

{{table:algo_comparison}}

### 5.2 Rationale for the trained models

**Elastic-net logistic regression** models the log-odds of decoupling as a weighted sum of features. Its penalty blends an L1 term (which can set weights to zero) and an L2 term (which shares weight among correlated features), a recognised remedy for collinear predictors (Zou & Hastie, 2005). Its standardised coefficients state direction and size and can be compared with the expected signs; its limit is linearity.

**Random forest** averages many decision trees, each grown on a bootstrap sample with a random subset of features at each split (Breiman, 2001). It captures thresholds and interactions without scaling, but its probabilities are usually less well calibrated and it cannot extrapolate.

**Histogram gradient boosting (HGB)** builds trees one after another, each correcting the errors of the earlier ones (Friedman, 2001), on binned feature values. It often does best on tabular data and handles missing values natively, but it has more settings and can fit noise when the signal is weak.

The support vector machine, k-nearest neighbours and neural network were not trained, for the reasons in Table 5.1.

### 5.3 The decision rule

The final choice was made by a rule fixed before looking at results. **Pick the model with the highest mean PR-AUC across the five rolling-origin folds; if a simpler model is within one standard error of the best, choose the simpler one.** The complexity order is logistic regression, random forest, HGB. With few folds and dependent windows, small score differences are hard to tell from noise, and a simpler model is easier to explain. The deliverable is the chosen model plus the interpretable logistic baseline; if they coincide, as they did (Section 8.3), the best non-linear model is still evaluated as a reference.

### 5.4 Hyper-parameters and the two-stage search

Hyper-parameters are settings fixed before fitting, such as penalty strength. The search spaces are small and set in advance (Appendix B). **Stage 1** fixes the preprocessing: for each model every combination of imputation and class weighting is scored with default hyper-parameters, and the rule above picks one, preferring simpler preprocessing (native NaN handling, then median, forward fill, iterative; no weights before balanced weights). **Stage 2** runs `RandomizedSearchCV` with 30 random draws per model, scored by mean PR-AUC on the same folds. HGB's early stopping is switched off, because its internal validation split is random and ignores time order.
