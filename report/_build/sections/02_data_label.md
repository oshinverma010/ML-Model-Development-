## 2. Dataset and label definition

### 2.1 Sources

Both public sources were used in Week 2: the OWID CO2 dataset with its codebook, and the World Bank Indicators API (version 2) for GDP per person, population, urbanisation, industry share and country metadata (income group, region). `src/acquire.py` downloads each file once, never overwrites it, and records URL, UTC time, size and SHA-256 hash in `data/raw/manifest.json`. All files were downloaded on {{download_date}}, so every number is tied to one data vintage.

The study window is 1990-2022. Five-year features need levels from five years earlier, so 1988 and 1989 are loaded as **look-back only**. Modelling rows start at t = 1993, and the last row with a three-year label is t = 2019. The codebook check (`results/column_availability.csv`) shows that {{col_unavailable_n|i}} of the wanted energy columns do not exist in this dataset ({{col_unavailable_names}}); they were dropped from the feature list, not replaced.

### 2.2 From country-years to modelling rows

The cleaned panel has {{join_total_country_year_rows|i}} country-year rows for {{join_total_countries|i}} countries and territories: {{join_both_country_year_rows|i}} present in both sources, {{join_owid_only_country_year_rows|i}} only in OWID and {{join_wb_only_country_year_rows|i}} only in the World Bank file. A row is **eligible** if CO2 per person and GDP per person are observed and strictly positive at t and at t+3. These core series are never imputed, because imputing the quantity that defines the label would put invented values inside the answer. Table 2.1 follows every exclusion; {{elig_n|i}} rows are eligible in total (pool, purge years and test years).

**Table 2.1: Sample flow from all entity-years to eligible rows (h = 3).**

{{table:sample_flow}}

### 2.3 The label

The label uses only values at t and t+h (h = 3). The inequalities are strict: an unchanged value counts as neither falling nor rising.

{{code:src/features.py:make_label}}

Figure 2.1 shows the timeline. A row at t = 2013 already needs outcomes through 2016, which is why the split has a purge gap.

![Figure 2.1: Label construction and the train, purge and test periods. Pool rows have t up to 2013, so their labels end by 2016. The test rows (t = 2017 to 2019) start after a three-year purge.](../figures/fig_label_split_timeline.png)

### 2.4 Base rate

The **base rate** is the share of rows with y = 1. In the training pool ({{pool_n_rows|i}} rows, {{pool_n_countries|i}} countries) it is {{pool_base_rate|p1}}, so about one window in four decouples. The yearly pool rate ranges from {{pool_year_min_rate|p1}} (t = {{pool_year_min_year|i}}) to {{pool_year_max_rate|p1}} (t = {{pool_year_max_year|i}}), moving with the world economy, which is one reason for a time-aware split. Over all eligible rows it is {{br_all_eligible_pool_test_rows_base_rate|p1}} and in the test years {{br_test_base_rate|p1}}. Table 8.1 lists every year and Figure 2.2 plots them. The pre-specified sanity range was 5% to 60%.

![Figure 2.2: Share of windows that decouple, by window start year. Blue bars are pool years; orange-red bars are test years (revealed only by the single test run).](../figures/fig_label_base_rate.png)

### 2.5 Why three years

One-year changes in emissions are noisy, so three years smooths some noise yet still gives many windows per country. The cost is that consecutive windows overlap by two years, so rows from one country are strongly dependent; hence all intervals resample whole countries, and a five-year horizon is a robustness variant (Section 8.8).
