## 10. Limitations and responsible use

* **Predictive association, not causation.** An importance or a coefficient does not say what would happen if a country changed that feature (Shmueli, 2010).
* **Production-based, territorial CO2.** OWID's `co2_per_capita` counts fossil-fuel and industrial CO2 emitted inside a country. A country can "decouple" by importing goods produced elsewhere, and land-use change and other gases are excluded, so absolute decoupling on this definition is not a falling climate footprint (compare Haberl et al., 2020).
* **A narrow, strict label.** A tiny fall in emissions counts the same as a large one, and an unchanged value counts as no decoupling.
* **Small, dependent sample.** The data hold {{br_all_eligible_pool_test_rows_n_rows|i}} windows from {{br_all_eligible_pool_test_rows_n_countries|i}} countries with overlapping windows, and the test set has only {{br_test_n_positive|i}} positive windows.
* **One COVID-shaped test period.** The test windows end in 2020 to 2022 and may not describe calmer periods.
* **Current income groups** are used for subgroups, not the groups in force in each year.
* **The F1 threshold rule failed to give a useful classifier.** It was pre-specified and kept, and its weakness is part of the finding. Likewise the selection rule and the test result disagree about the best model; the post hoc analysis is exploratory and changed nothing.
* **One data vintage.** OWID and World Bank values are revised.
* **Responsible use.** The model should not rate, rank or sanction countries, set targets, or support claims that policies work. It is a teaching pipeline whose value is the process, not the predictions.

## 11. Roadmap beyond Week 3

If the model were to be used: **monitor** the base rate, score distribution and calibration as data arrive; **retrain** yearly with the same rolling-origin design and test discipline; **follow the post hoc lead** by repeating the comparison, with the random forest and boosting given more weight, on a later hold-out period, adding calibration (Platt scaling or isotonic regression fitted inside folds); and **improve the label and features** with a continuous target, consumption-based emissions and OWID's energy-mix shares. Deployment is out of scope.

## 12. Conclusion

The task was to build and evaluate a model predicting absolute decoupling of CO2 per person from GDP per person three years ahead, with a complete, leakage-safe process. The pipeline does that: a pre-specified plan, time-aware splits with a purge gap, everything fitted inside folds, a rule-based model choice, a frozen threshold and one test run; {{n_tests|i}} integrity tests pass.

The result is modest. The selected {{chosen_name}} reached a test PR-AUC of {{test_lr_pr_auc|3}} (95% interval {{test_lr_pr_auc_lo|3}} to {{test_lr_pr_auc_hi|3}}), against {{test_persistence_pr_auc|3}} for the persistence rule and a test prevalence of {{br_test_base_rate|p1}}; the paired difference from persistence ({{tpair_lr_persistence_pr_auc_diff|3}}, interval {{tpair_lr_persistence_pr_auc_ci_low|3}} to {{tpair_lr_persistence_pr_auc_ci_high|3}}) shows that ranking by the model beats extrapolating the recent past. The limits matter as much: at the frozen threshold the model flags almost every window, an income-group prior ranks as well, and a country's recent trajectory alone predicts almost nothing beyond the base rate, while level and structure (above all income per person, then fuel mix and industry) carry the signal. The random forest and boosting scored higher on the test set in a post hoc comparison, but the pre-specified choice was not changed.

For the title question the answer is cautious. Income per person is strongly associated with whether emissions per person fall while income rises, echoing Week 2, but beyond that association a country's next three years are hard to predict from these features.

## 13. References

{{refs}}
