## 8. Results of executing the plan

### 8.1 Sample and base rates

The pool holds {{pool_n_rows|i}} windows from {{pool_n_countries|i}} countries ({{pool_n_positive|i}} decoupling windows, {{pool_base_rate|p1}}). The test set holds {{br_test_n_rows|i}} windows from {{br_test_n_countries|i}} countries ({{br_test_n_positive|i}} decoupling windows, {{br_test_base_rate|p1}}); by year the rate was {{bry_test_2017_base_rate|p1}} (t = 2017), {{bry_test_2018_base_rate|p1}} (2018) and {{bry_test_2019_base_rate|p1}} (2019). A model with no skill would therefore score a PR-AUC of about {{br_test_base_rate|p0}} on the test set.

**Table 8.1: Label base rate by window start year (pool years from the training-period analysis, test years from the single test run).**

{{table:base_rate_year}}

### 8.2 Preprocessing choice (stage 1)

Every combination of imputation and class weighting was scored with default hyper-parameters (Table 8.2, Figure 8.1). Within each model all combinations lay within one standard error of the best, so the rule picked the simplest: {{s1sel_lr_chosen_imputer}} imputation without weights for the logistic regression and the random forest, and {{s1sel_hgb_chosen_imputer}} NaN handling without weights for HGB. The nominally best options (iterative imputation for the logistic regression, {{s1sel_lr_best_pr_auc|3}} against {{s1sel_lr_chosen_pr_auc|3}}; balanced weights for the random forest, {{s1sel_rf_best_pr_auc|3}} against {{s1sel_rf_chosen_pr_auc|3}}) differ by far less than the standard errors ({{s1sel_lr_best_se|3}} to {{s1sel_rf_best_se|3}}): how gaps are filled matters little here.

![Figure 8.1: Mean PR-AUC across the five rolling-origin folds for each imputation option and class weighting (default hyper-parameters). Bars are plus and minus one standard error.](../figures/fig_imputation_comparison.png)

**Table 8.2: Stage 1, imputation by class weighting, rolling-origin CV on the pool.**

{{table:stage1}}

### 8.3 Model comparison and selection in cross-validation

The tuned settings are in Table 8.3, and Table 8.4 and Figure 8.2 compare the models. The random forest had the highest mean PR-AUC ({{sel_best_pr_auc|3}}, standard error {{sel_best_se|3}}). The logistic regression scored {{cv_lr_pr_auc_mean|3}}, above the "within one standard error" cut-off of {{sel_threshold_for_simpler|3}}, so the rule selected the **{{chosen_name}}**. All three beat persistence ({{cv_persistence_pr_auc_mean|3}}) and the prior ({{cv_prior_pr_auc_mean|3}}), but folds vary a lot (fold standard deviation {{cv_rf_pr_auc_sd|3}} for the random forest).

**Table 8.3: Preprocessing and tuned hyper-parameters per model.**

{{table:tuning_best}}

**Table 8.4: Rolling-origin CV results (five folds). Brier and ECE are not shown for the persistence rule, which gives only 0 or 1.**

{{table:cv_comparison}}

![Figure 8.2: PR-AUC in cross-validation. Diamonds are fold means with plus and minus one standard error; faint dots are the five folds; the dotted line is the prior.](../figures/fig_cv_model_comparison.png)

The paired bootstrap on pooled out-of-fold predictions (Table 8.5) is sharper than the fold standard errors. It says the random forest is better than the logistic regression in CV (difference {{pair_lr_rf_diff|3}}, interval {{pair_lr_rf_ci_low|3}} to {{pair_lr_rf_ci_high|3}}) and that every trained model beats persistence. The rule is more conservative than this bootstrap and was not changed afterwards. The country-grouped check (Table 8.6) scores every model higher, partly because it lets years mix, but orders them the same way.

**Table 8.5: Paired differences in pooled out-of-fold PR-AUC.**

{{table:cv_paired}}

**Table 8.6: Secondary check, five folds grouped by country (time ignored).**

{{table:groupkfold}}

The frozen threshold for the {{chosen_name}} was {{cfg_lr_threshold|3}}; on pooled out-of-fold predictions it gave recall {{oof_lr_recall|p0}}, precision {{oof_lr_precision|p0}} and MCC {{oof_lr_mcc|3}}, so for this weakly separating model the F1 rule chose "flag almost everything".

### 8.4 The single test run

The configuration was frozen on {{frozen_utc}} UTC and the test years were evaluated once, on {{lock_utc}} UTC. Table 8.7 gives threshold-free metrics with 95% country-clustered intervals; Table 8.8 gives metrics at the frozen thresholds. Curves, confusion matrices and reliability curves are in Figures 8.3 to 8.6 and Table 8.9.

**Table 8.7: Test results, threshold-free metrics (t = 2017-2019).**

{{table:test_scores}}

**Table 8.8: Test results at the frozen thresholds.**

{{table:test_thresholded}}

![Figure 8.3: ROC curves on the test set. The cross is the persistence rule's single operating point.](../figures/fig_test_roc.png)

![Figure 8.4: Precision-recall curves on the test set. The dotted line is the test prevalence; the cross is the persistence rule.](../figures/fig_test_precision_recall.png)

![Figure 8.5: Test confusion matrices at the frozen thresholds (counts and row-normalised shares).](../figures/fig_test_confusion_matrices.png)

**Table 8.9: Test confusion matrices at the frozen thresholds (rates are shares of actual negatives for TN and FP, and of actual positives for FN and TP).**

{{table:test_confusion}}

![Figure 8.6: Reliability curves on the test set (observed share of decoupling windows against mean predicted probability, 10 bins).](../figures/fig_test_calibration.png)

**Table 8.10: Paired differences on the test set (pre-specified comparisons).**

{{table:test_paired}}

### 8.5 Reading the test results honestly

**The model beats the simple rule.** The {{chosen_name}} reached PR-AUC {{test_lr_pr_auc|3}} (95% interval {{test_lr_pr_auc_lo|3}} to {{test_lr_pr_auc_hi|3}}) and ROC-AUC {{test_lr_roc_auc|3}}. Persistence reached {{test_persistence_pr_auc|3}} and {{test_persistence_roc_auc|3}}, and the prior {{test_prior_pr_auc|3}}. The paired PR-AUC difference from persistence was {{tpair_lr_persistence_pr_auc_diff|3}} (interval {{tpair_lr_persistence_pr_auc_ci_low|3}} to {{tpair_lr_persistence_pr_auc_ci_high|3}}), which excludes zero, so the first success criterion is met for ranking. The gain is modest: a random decoupling window is ranked above a random non-decoupling one about {{test_lr_roc_auc|p0}} of the time.

**As a yes/no classifier at the frozen threshold it is not useful.** It made {{conf_lr_fp|i}} false positive calls against {{conf_lr_tn|i}} true negatives and caught {{conf_lr_tp|i}} of {{br_test_n_positive|i}} positives. Its accuracy ({{test_lr_accuracy|3}}) was below the prior's ({{test_prior_accuracy|3}}), its precision ({{test_lr_precision|3}}) about the test prevalence, and its MCC only {{test_lr_mcc|3}} (interval {{test_lr_mcc_lo|3}} to {{test_lr_mcc_hi|3}}), lower than persistence ({{test_persistence_mcc|3}}). It is useful for ranking, not for hard classification at this threshold.

**Part of the skill is the income gradient.** A post hoc reference (Table 8.11) scores each test row with the pool base rate of its income group, with no model at all. It reached PR-AUC {{ph_lr_group_prior_pr_auc_value_b|3}} and ROC-AUC {{ph_lr_group_prior_roc_auc_value_b|3}}, and the logistic regression was not distinguishable from it (PR-AUC difference {{ph_lr_group_prior_pr_auc_diff|3}}, interval {{ph_lr_group_prior_pr_auc_ci_low|3}} to {{ph_lr_group_prior_pr_auc_ci_high|3}}). What the linear model mostly learned is the Week 2 finding that richer countries decouple more often.

**The non-linear models did better on the test set, but the choice was not changed.** The random forest scored PR-AUC {{test_rf_pr_auc|3}} ({{test_rf_pr_auc_lo|3}} to {{test_rf_pr_auc_hi|3}}) and HGB {{test_hgb_pr_auc|3}} ({{test_hgb_pr_auc_lo|3}} to {{test_hgb_pr_auc_hi|3}}). In a post hoc paired comparison the PR-AUC of the logistic regression minus that of the random forest was {{ph_lr_rf_pr_auc_diff|3}} (interval {{ph_lr_rf_pr_auc_ci_low|3}} to {{ph_lr_rf_pr_auc_ci_high|3}}), but in ROC-AUC only {{ph_lr_rf_roc_auc_diff|3}} (interval {{ph_lr_rf_roc_auc_ci_low|3}} to {{ph_lr_rf_roc_auc_ci_high|3}}, which includes zero). The random forest also beat the income-group reference (difference {{ph_rf_group_prior_pr_auc_diff|3}}, interval {{ph_rf_group_prior_pr_auc_ci_low|3}} to {{ph_rf_group_prior_pr_auc_ci_high|3}}), pointing to some structure beyond income. Swapping models now would use the test set to make a choice, which the protocol forbids; the finding is a hypothesis for the next iteration.

**Table 8.11: Post hoc comparisons from the saved test predictions (exploratory, not pre-specified).**

{{table:posthoc}}

**Calibration** was acceptable, not excellent: test ECE was {{test_lr_ece|3}} for the logistic regression, {{test_rf_ece|3}} for the random forest and {{test_hgb_ece|3}} for HGB, against {{test_prior_ece|3}} for the constant prior. The curves (Figure 8.6) over-predict in the 0.2 to 0.3 band and under-predict above 0.4.

### 8.6 Interpretation

On the test set, shuffling `{{pt1_feature}}` cost {{pt1_imp|3}} of PR-AUC, followed by `{{pt2_feature}}` ({{pt2_imp|3}}), `{{pt3_feature}}` ({{pt3_imp|3}}) and `{{pt4_feature}}` ({{pt4_imp|3}}) (Table 8.12). In CV the leaders were `{{pcv1_feature}}` ({{pcv1_imp|3}}), `{{pcv2_feature}}` ({{pcv2_imp|3}}) and `{{pcv3_feature}}` ({{pcv3_imp|3}}), though the first was positive in only {{pcv1_pos|i}} of five folds.

**Table 8.12: Permutation importance of the selected model (drop in PR-AUC).**

{{table:importance}}

The L1 part of the penalty zeroed only {{coef_n_zeroed|i}} of {{coef_n_total|i}} columns (a missing-indicator column), so the penalty acted mostly as shrinkage (Table 8.13). The largest coefficients are `{{coef1_feature}}` ({{coef1_value|2}}), `{{coef2_feature}}` ({{coef2_value|2}}) and `{{coef3_feature}}` ({{coef3_value|2}}). They are tied by the identity in Section 4.1 and must be read together: high income with low emissions per person raises the odds of decoupling. Only {{sign_agree_n|i}} of the {{sign_defined_n|i}} features with a pre-specified sign had a coefficient of that sign; correlated features make single signs unreliable. Partial dependence moved most for GDP per person (predicted probability from {{pdp_ln_gdp_pc_min|2}} to {{pdp_ln_gdp_pc_max|2}} across its 5th to 95th percentile), then oil share ({{pdp_oil_share_min|2}} to {{pdp_oil_share_max|2}}) and urbanisation ({{pdp_urban_pct_min|2}} to {{pdp_urban_pct_max|2}}). These are predictive associations at country level, not causes.

![Figure 8.7: Left, permutation importance of the selected model on the test set (mean and standard deviation of 20 shuffles). Right, the ten largest standardised logistic coefficients (missing-indicator columns omitted).](../figures/fig_importance_coefficients.png)

**Table 8.13: Standardised logistic coefficients on the pool and agreement with the pre-specified expected signs.**

{{table:coefficients}}

### 8.7 Subgroups

The selected model's PR-AUC was {{sub_income_group_high_income_lr_pr_auc|2}} for high-income countries, whose prevalence was {{sub_income_group_high_income_base_rate|p0}}: only a small improvement where decoupling is most common (Figure 8.8, Table 8.14). For upper-middle income it was {{sub_income_group_upper_middle_income_lr_pr_auc|2}} against {{sub_income_group_upper_middle_income_base_rate|p0}}; for lower-middle income {{sub_income_group_lower_middle_income_lr_pr_auc|2}} against {{sub_income_group_lower_middle_income_base_rate|p0}}, with ROC-AUC {{sub_income_group_lower_middle_income_lr_roc_auc|2}}, below 0.5. Low income had only {{sub_income_group_low_income_n_positive|i}} positive windows, Sub-Saharan Africa a ROC-AUC of {{sub_region_sub_saharan_africa_lr_roc_auc|2}}, and North America ({{sub_region_north_america_n|i}} rows) and South Asia ({{sub_region_south_asia_n|i}} rows) are too small to read. Skill lies in separating income groups from each other; within most groups it is close to chance, and intervals are wide.

![Figure 8.8: Test PR-AUC by income group for the selected model and the persistence rule, with 95% country-clustered intervals. The black bar is each group's prevalence.](../figures/fig_subgroup_income.png)

**Table 8.14: Test performance by income group and region of the selected model.**

{{table:subgroups}}

### 8.8 Ablation and robustness

With momentum features only (set A) the logistic regression scored a test PR-AUC of {{tabl_lr_a_pr_auc|3}}, no better than the prior ({{test_prior_pr_auc|3}}). Adding economic structure (set B) raised it to {{tabl_lr_b_pr_auc|3}}, and energy and fuel mix (set C) to {{tabl_lr_c_pr_auc|3}}; in CV the sets scored {{abl_lr_a_pr_auc_mean|3}}, {{abl_lr_b_pr_auc_mean|3}} and {{abl_lr_c_pr_auc_mean|3}}. L1 selection on top of C made no difference ({{tabl_lr_c_l1_pr_auc|3}}). A country's recent trajectory alone carries almost no information about the next three years, as the persistence rule also shows.

![Figure 8.9: Feature-set ablation for the selected model in cross-validation (left) and the single test run (right). A: momentum only; B: A plus economic structure; C: full set; C + L1: full set with L1 selection.](../figures/fig_feature_set_ablation.png)

**Table 8.15: Feature-set ablation (fixed tuned hyper-parameters).**

{{table:ablation}}

Excluding the window ending in 2020 (t = 2017) raised every score but kept the ordering: PR-AUC {{trob_exclude_t_2017_window_ends_2020_lr_pr_auc|3}} for the logistic regression against {{trob_exclude_t_2017_window_ends_2020_persistence_pr_auc|3}} for persistence. Excluding countries under one million people gave {{trob_exclude_population_1_million_lr_pr_auc|3}} against {{trob_exclude_population_1_million_persistence_pr_auc|3}}. With a five-year horizon in CV, the logistic regression's mean fold PR-AUC was {{rob_h5_lr_pr_auc|3}} (persistence {{rob_h5_persistence_pr_auc|3}}, prior {{rob_h5_prior_pr_auc|3}}), against {{cv_lr_pr_auc_mean|3}} at three years. The selected model stays ahead of persistence in every variant.

**Table 8.16: Test robustness variants (subsets of the single test run).**

{{table:test_robust}}

**Table 8.17: Robustness in cross-validation (pooled out-of-fold subset, and the five-year horizon with fold means).**

{{table:robust_cv}}
