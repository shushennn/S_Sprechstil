# Midlevel descriptors description

## mld.MLD()

### General

* n = 576
* The MLD output dataframe in unified format contains one row per segment, or - in case no segments are available - a single row for the entire file.
* The dataframe contains the following columns (provided that the required content was found in the openSmile table).
* sumStat refers to {mean|var|skewness|kurtosis|diff|slope|start|peak|end} (parametric) or {median|iqr|q_var|q_skewness|q_kurtosis|q_range5|diff|slope|start|peak|end} (non-parametric) summary statistics
* diff, slope: time and feature range normalized to [0, 1]
    * diff: m of first half - m of second half (m is mean or median)
    * slope: 1st order coef of linear fit
* start, peak, end:
    * start: y[0] - median(y)
    * peak:  max(y) - median(y)
    * end:   y[-1] - median(y)
* q_var: median RMS from median
* q_skewness: ((p90-p50)-(p50-p10))/(p90-p10)
* q_kurtosis: (p90-p10)/(p75-p25)
* q_range5: p95-p5
* `*` is to be replaced by all these summary statistics variables.

### Features
* align_corr_*: sumStat of per syllable correlation between f0 and energy
* align_spread_*: sumStat of overall variation of per-syl f0-en corr
* align_spreadzero_*: sumStat of overall variation of per-syl f0-en corr from 0
* align_flux_*: sumStat of f0/en correlation flux
* end: timedelta end of segment (index)
* f0en_granger_causality_F0final_sma: p value of test that energy contour is better predicted by both f0 and energy than by energy alone
* f0en_granger_causality_pcm_RMSenergy_sma: p value of test that f0 contour is better predicted by both f0 and energy than by f0 alone
* f0en_rms: root mean squared deviation between centered+scaled f0 and energy contour
* (f0en_sample_entropy_F0final_sma: forecastability of f0; currently deactivated)
* (f0en_sample_entropy_pcm_RMSenergy_sma: forecastability of energy; currently deactivated)
* f0en_spea: spearman correlation between f0 and energy contour
* f0var_mean_*: sumStat of f0 in variation ratio analysis
* f0var_varRat_*: sumStat of f0 variation ratio mean
* f0var_var_*: sumStat of f0 variance mean in variation ratio analysis
* file: file name (index)
* hld_gapRel: pause proportion in segment
* hld_sylRate: syllables per second
* lld_*_*: sumStat of low level descriptors extracted around syllable nuclei
* reg_ml_intercept: register midline intercept
* reg_ml_mean: register midline mean
* reg_ml_rate: register midline rate
* reg_rng_bl: register range mean divided by baseline mean
* reg_rng_intercept: register range intercept
* reg_rng_mean: register range mean
* reg_rng_rate: register range rate
* rhy_en_mae: mean absolute error between original and IDCT energy contour 
* rhy_en_prop: impact of syllable rate on energy DCT spectrum
* rhy_en_sm1: 1st spectral moment of energy DCT spectrum
* rhy_en_sm2: 2nd spectral moment of energy DCT spectrum
* rhy_en_sm3: 3rd spectral moment of energy DCT spectrum
* rhy_en_sm4: 4th spectral moment of energy DCT spectrum
* rhy_f0_mae: mean absolute error between original and IDCT f0 contour 
* rhy_f0_prop: impact of syllable rate on f0 DCT spectrum
* rhy_f0_sm1: 1st spectral moment of f0 DCT spectrum
* rhy_f0_sm2: 2nd spectral moment of f0 DCT spectrum
* rhy_f0_sm3: 3rd spectral moment of f0 DCT spectrum
* rhy_f0_sm4: 4th spectral moment of f0 DCT spectrum
* rhy_iso_pvi: normalized pairwise variability index of syllable durations
* rhy_iso_varco: variance coefficient of syllable durations
* shape_c0_*: sumStat of local f0 shape, 0th polycoef mean
* shape_c1_*: sumStat of local f0 shape, 1st polycoef mean
* shape_c2_*: sumStat of local f0 shape, 2nd polycoef mean
* shape_c3_*: sumStat of local f0 shape, 3rd polycoef mean
* shape_diff_*: sumStat of difference mean of local f0 shape from register midline
* shape_flux_*: sumStat of shape flux (Euclidean dist. of adjacent shapes)
* shape_mae_*: sumStat of absolute deviation mean between f0 shape and register midline 
* shape_rng_*: sumStat of mean range of local f0 shape
* shape_spread_*: sumStat of mean overall variation of f0 shapes around centroid
* shape_spreadzero_*: sumStat of overall variation of f0 shapes around zeros
* shape_tmax_*: sumStat of normalized time of f0 maximum relative to syllable
* spec_flux_*: sumStat of spectral flux on syllable nuclei
* spec_spread_*: sumStat of spectral variation on syllable nuclei
* start: timedelta start of segment (index)
* inter_flux_*: sumStat of spectral flux between syllable nuclei
* inter_spread_*: sumStat of spectral spread between syllable nuclei
* voi_mean_*: sumStat of 1st spectral moments ...
* voi_variance_*: 2nd spectral moments ...
* voi_skewness_*: 3rd spectral moments ...
* voi_kurtosis_*: 4th spectral moments of voicing distribution around syllable nuclei

## mld.VAD_MLD()

### General
* n = 25
* The MLD_VAD output dataframe in unified format contains one row per file.
* Pauses at file margins are not considered in feature extraction, since they currently depend solely on the auvad configurations.
* `*` is to be replaced by the summary statistics introduced above.

### Features
* pau_prop: proportion of speech pauses [0, 1]
* pau_count: absolute count of pauses (int)
* pau_rate: pause count per utterance duration
* pau_*: pause duration distribution parameters
* vad_*: VAD segment duration distribution parameters


## mld.MLD_SUST()

### General

* n = 85
* The MLD_SUST output dataframe in unified format contains one row per segment, or - in case no segments are available - a single row for the entire file.
* `*` again is to be replaced by the summary statistics introduced above.
* "en/f0" indicates that features are available for both F0 and energy contour (e.g. shape_en/f0_d_end -> shape_en_d_end, shape_f0_d_end).

### Features

* end: timedelta end of segment (index)
* f0en_granger_causality_F0final_sma: p value of test that energy contour is better predicted by both f0 and energy than by energy alone
* f0en_granger_causality_pcm_RMSenergy_sma: p value of test that f0 contour is better predicted by both f0 and energy than by f0 alone
* f0en_rms: root mean squared deviation between centered+scaled f0 and energy contour
* f0en_spea: spearman correlation between f0 and energy contour
* file: file name (index)
* reg_ml_intercept: register midline intercept
* reg_ml_mean: register midline mean
* reg_ml_rate: register midline rate
* reg_rng_intercept: register range intercept
* reg_rng_mean: register range mean
* reg_rng_rate: register range rate
* shape_en/f0_d_end: segment-initial distance to midline
* shape_en/f0_d_peak: maximum distance to midline
* shape_en/f0_d_start: segment-final distance to midline
* shape_en/f0_dlt_*: sumStat over time series deltas
* shape_en/f0_mae: mean absolute deviation between time series and its midline
* shape_en/f0_mcr: midline crossing rate
* shape_en/f0_tpr: turning point rate
* shape_en/f0_y_*: sumStat over time series
* spec_flux_*: sumStat of spectral flux over entire sustained sound
* spec_spread_*: sumStat of spectral variation over entire sustained sound
* start: timedelta start of segment (index)

