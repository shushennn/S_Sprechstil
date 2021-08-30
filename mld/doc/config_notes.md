# notes on configurations

## window lengths

* most llds to be extracted within acoustic steady-state part of syllable nuclei. Thus the analysis window (in sec) shoud be short `lld:win = 0.03`
* some llds capture speaking rate, thus their analysis windows should include transitions between the nucleus and subsequent sounds, e.g. `lld:pcm_fftMag_spectralFlux_sma = 0.11`

## spectral spread

* from mfcc columns (default `"mfcc"`) or from audSpec columns `col_mld_vec = "audSpec"`

## summary statistics

* if "robust", non-parametric distribution descriptions are calculated, which is likely to be more appropriate for small samples and multimodal distributions. "standard" gives a parametric description. 

## distances

* for spectral spread and flux (mld_vec) one might want to choose a distance correcting for different coefficient value ranges `mld_vec:dist = "canberra"`

* for shape spread and flux (mld_shape) polycoefs do not systematically have different ranges. Thus more appropriate is `mld_shape:dist = "euclidean"`

## task-specific configurations

* **language identification:** add all mfcc values to the LLD pool. Choose a longer window for `"voicingFinalUnclipped_sma"`, so that not only voicing characteristics of vowel segments but also the amount of voiced/unvoiced is measured in the surroundings; `lld:voicingFinalUnclipped_sma = 0.15`
