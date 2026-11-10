## 9. Potential pitfalls and mitigation

### 9.1 Risk register

Table 9.1 is the risk register written into the plan, with a column that says what happened in this data. Several risks mattered: the weak signal made the threshold unreliable, the one-standard-error rule and the test set disagreed about the best model, and income-group information accounted for much of the skill.

**Table 9.1: Risk register with the outcome observed in this project.**

{{table:risk_register}}

### 9.2 What mattered most

**Leakage and label overlap** are the usual ways a panel model fools its author. They were handled by construction and by test; the indirect evidence that they did not occur is that scores are modest, which is not what a leak usually produces. **Selection under weak signal:** with about {{pool_n_rows|i}} correlated rows and five folds, differences of a few hundredths in PR-AUC cannot be separated from noise inside CV, which is why the rule prefers a simpler model. The test result suggests this cost some ranking quality, but a rule fixed before seeing results is worth more than one adjusted afterwards; the remedy is more evidence, not a changed rule. **Reading a modest model correctly:** a score above the baselines does not make a model useful for decisions. Here the ranking beats chance and persistence, but thresholded predictions are close to useless and a trivial income-group reference matches the linear model (Section 8.5).
