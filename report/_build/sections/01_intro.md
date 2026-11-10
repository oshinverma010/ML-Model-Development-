# Does Wealth Burn Carbon? A Machine Learning Model Development and Evaluation Plan for Predicting Emissions Decoupling

*Week 3 Task - Machine Learning Model Development and Evaluation Plan (continuation of the Week 1 and Week 2 project)*

---

## 1. Introduction and problem framing

### 1.1 Where the project stands

This project asks whether rising income inevitably brings rising carbon dioxide (CO2) emissions. Week 1 proposed the data and visualisation strategy; Week 2 tested seven pre-specified hypotheses on Our World in Data (OWID) CO2 figures and World Bank indicators for 1990-2022. Its main findings, re-read from the result files, were: GDP per person and CO2 per person were strongly rank-correlated across {{w2_h1_n|i}} countries (Spearman rho = {{w2_h1_rho|2}}); high-income countries emitted about {{w2_h3_ratio|0}} times more per person than lower-middle and low-income countries combined; and {{w2_decoupled_pct_high_income|0}}% of high-income countries achieved absolute decoupling (income per person up, emissions per person down) between 1990 and 2022, against {{w2_decoupled_pct_low_income|0}}% to {{w2_decoupled_pct_upper_middle_income|0}}% in other income groups. The weakest result, a difference in income-emissions slopes by income group, lost significance without small countries (p = {{w2_h4_p_raw|3}} in the main test, {{w2_h4_p_excl_small|3}} without them).

Those tests described the past. Week 3 asks a forward-looking question suited to machine learning: **given what a country looks like in year t, will it decouple over the next three years?** The goal is a complete, honest model development plan, backed by working code.

### 1.2 The prediction task

One country in one year is an **observation** (a row). For country c and year t the **label** y is 1 if CO2 per person (OWID `co2_per_capita`, fossil fuels and industry) falls between t and t+3 **and** GDP per person (World Bank `NY.GDP.PCAP.KD`, constant 2015 US dollars) rises over the same period; otherwise y is 0. This is **absolute decoupling**. The inputs, called **features**, use only information available at the end of year t. A **model** is a function learned from past rows that outputs the probability that y = 1 for a new row. The task is **binary classification** on a **panel** (many countries over many years).

The aim is prediction, not explanation. A model can show which characteristics are *associated* with future decoupling, but not that changing them would *cause* it (Shmueli, 2010).

### 1.3 Success criteria fixed in advance

The pre-specification (`docs/ml_plan.md`, committed before any model was fitted) set three criteria. (1) The selected model should rank decoupling windows better than the base rate and better than a "persistence rule" that assumes the recent past continues; "better" means the 95% interval of the paired PR-AUC difference excludes zero (PR-AUC is explained in Section 6). (2) The pipeline must be free of leakage, meaning information from the test period or the future reaching the model; pytest checks test this (Appendix D). (3) Whatever the result, it is reported honestly: a modest result is acceptable, an inflated one is not.

### 1.4 Organisation

Sections 2 to 6 are the plan, Section 7 the implementation, Section 8 the results, Section 9 the risk register, and Sections 10 to 12 the limitations, roadmap and conclusion (Figure 1.1). The order of work is part of the evidence: the plan was committed first, exploration used only training years, the model was frozen, and the test years were evaluated once.

![Figure 1.1: The end-to-end workflow. Steps 1 to 6 use only the training pool; the test years are touched once, in step 7.](../figures/fig_pipeline_flowchart.png)
