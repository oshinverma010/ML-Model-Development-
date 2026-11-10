## 4. Feature engineering

### 4.1 Feature extraction

The design follows the **Kaya identity**: CO2 per person equals GDP per person times energy per unit of GDP times CO2 per unit of energy. Because the label concerns CO2 per person and GDP per person, these three factors are the natural drivers of whether it turns on. The 20 features fall into three groups: economic structure and growth, emissions and momentum, and energy and fuel mix. Changes are differences of natural logarithms, so -0.05 means a fall of about 5%, and every feature at year t uses data from t and earlier. Table 4.1 gives the expected signs written in the plan before any model was fitted; Appendix C adds summary statistics.

**Table 4.1: Feature catalogue (all features use data up to year t).**

{{table:feature_catalogue}}

Two features are exact linear combinations of others: log carbon intensity equals log CO2 per person minus log GDP per person, and its 3-year change equals the 3-year CO2 change minus three times the 3-year growth rate. They were kept because they give tree models a ready-made ratio and the elastic-net penalty copes with exact collinearity, but the variance inflation factor flags them.

### 4.2 Feature sets for ablation

Three nested sets were fixed in advance to show where information comes from: A is **momentum only**, B adds economic structure, and C is the full set including energy and fuel mix.

{{table:feature_sets}}

### 4.3 What is deliberately excluded

**Calendar year** would let a flexible model memorise year-specific base rates and then extrapolate to unseen years. **Country identity** would turn the task into memorising countries instead of generalising to new ones. **World Bank income group** is defined by a threshold on national income, so it is circular with GDP per person; the classification used here is also the current one, not the one in force in year t, so it would carry information from the future into past rows. Income group and region are used only to evaluate subgroups.

### 4.4 Feature selection

**Filter: near-duplicates and collinearity.** Inside every training fold, features are visited in the order of Table 4.1 and one is dropped if its absolute Spearman rank correlation with an already kept feature exceeds 0.9. On the whole pool this removed {{cf_dropped_n|i}} feature, {{cf_dropped_names}} (rho = {{cf_ln_energy_pc_abs_rho|2}} with `{{cf_ln_energy_pc_dropped_because_of}}`): countries that emit more per person use more energy per person (Figure 4.1). The **variance inflation factor (VIF)** measures how well each feature is predicted by the others; values above about 10 signal troublesome collinearity (Table 4.2). By the identities above, {{vif_inf_n|i}} features have an infinite VIF ({{vif_inf_names}}), and the 3-year momentum features are large but finite ({{vif_co2_chg3_vif_all_20|1}} for `co2_chg3`, {{vif_ci_chg3_vif_all_20|1}} for `ci_chg3`).

![Figure 4.1: Spearman rank correlations between features on the pool. Cell labels are shown where the absolute correlation is at least 0.6.](../figures/fig_feature_correlation.png)

**Table 4.2: Variance inflation factors on the pool (median-imputed, standardised).**

{{table:vif}}

**Embedded selection.** The L1 part of the elastic-net penalty shrinks weak coefficients to exactly zero, so the fitted model is itself a selector; a variant with an L1 `SelectFromModel` step is compared in the ablation. **Permutation importance** shuffles one feature at a time in held-out data and records the fall in PR-AUC; it is computed on each validation block and, descriptively, on the test set. **Recursive feature elimination (RFE)** was considered but not used: it refits the model repeatedly inside every fold, is unstable with correlated features, and adds one more tuned setting.

### 4.5 Leakage review

Leakage is a widespread cause of unreliable results in machine-learning studies (Kapoor & Narayanan, 2023), so each feature was checked against three questions: does it use data after t, does it use a label-defining quantity from t+1 or later, and does it use a classification that depends on the future? None does. Pytest checks perturb all data after a cut-off and confirm that earlier features do not change (Appendix D, test 13), and that forward fill uses only past values (test 15).
