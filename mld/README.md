# mld

* speech chunk and syllable nucleus time stamp extraction
* mid-level-descriptor (MLD) extraction

## Installation

* Create and activate Python virtual environment, e.g.

        $ virtualenv --python=python3 ${HOME}/.envs/mld
        $ source ${HOME}/.envs/mld/bin/activate
        (mld) $ pip install -r requirements.txt

## Usage: extraction of prosodic, voice quality, and articulation MLDs from index

* input:
    * `df_seg`: unified format dataframe of segment or file type, that defines the segments (VAD segments or entire file) within which the MLDs are to be extracted.
    * `df_lld` (optionally): LLD dataframe returned by pyopensmile. Needed only if audiofiles are not available.
* output: `df_mld`, unified format dataframe
    * of same type as `df_seg`
    * file type: one feature vector per file
    * segment type: one feature vector per VAD segment

### Signal files available

```
import sys
sys.path.append("myPathTo/mld/src")
import audb
import midlevel_descriptors as mld

db = audb.load('emodb', version='1.0.1', format='wav')
df_seg = db["emotion"].df
fex_mld = mld.MLD()
df_mld = fex_mld.extract_from_index(index=df_seg)
```

### LLDs available

* scenario: project partner ships only LLDs but no signal files
* currently required LLD set: ComParE_2016
* IMPORTANT: signal- and LLD-based margin removal and syllable nucleus extraction differ slightly! Thus MLD extraction from signals and from LLDs give slightly different results.

```
import sys
sys.path.append("myPathTo/mld/src")
import audb
from opensmile import FeatureExtractor, FeatureSet, FeatureLevel
import midlevel_descriptors as mld

db = audb.load('emodb', version='1.0.1', format='wav')
df_seg = db["emotion"].df

fex_lld = FeatureExtractor(feature_set=FeatureSet.ComParE_2016,
                           feature_level=FeatureLevel.LowLevelDescriptors)
df_lld = fex_lld.extract_from_index(df_emo)

fex_mld = mld.MLD()
df_mld = fex_mld.extract_from_index(index=df_seg, lld=df_lld)
```

## Usage: extraction of prosodic, voice quality, and articulation MLDs from array

* scenario: single file is to be processed

```
import sys
sys.path.append("myPathTo/mld/src")
import audiofile as af
import midlevel_descriptors as mld

fex = mld.MLD()
y, fs = af.read(myAudioFile)
df_mld = fex.extract_from_array(y, fs)
```

## Usage: pause/speech proportion MLDs from index

* input: `df_seg`, unified format dataframe of segment type (VAD segments); gaps between VAD-segments are assumed to be speech pauses.
* output: `df_mld`, unified format dataframe of file type; one feature vector per file
* no audiofiles required

```
import sys
sys.path.append("myPathTo/mld/src")
import midlevel_descriptors as mld

fex = mld.VAD_MLD()
df_mld = fex.extract_from_index(index=df_seg)
```

## Usage: sustained vowel MLDs from index

* input:
    * `df_seg`: unified format dataframe of segment or file type, that defines the segments (VAD segments or entire file) within which the MLDs are to be extracted.
    * `df_lld` (optionally): LLD dataframe returned by pyopensmile. Needed only if audiofiles are not available.
* output: `df_mld`, unified format dataframe
    * of same type as `df_seg`
    * file type: one feature vector per file
    * segment type: one feature vector per VAD segment
* call: same as for MLD(); extraction from signal and from LLDs supported

```
...
fex = mld.MLD_SUST()
df_mld = fex.extract_from_index(index=df_seg, lld=df_lld)
```

## Usage: sustained vowel MLDs from array

* scenario: single file is to be processed

```
import sys
sys.path.append("myPathTo/mld/src")
import audiofile as af
import midlevel_descriptors as mld

fex = mld.MLD_SUST()
y, fs = af.read(myAudioFile)
df_mld = fex.extract_from_array(y, fs)
```


## Configurations for mld.MLD()

* stored in `config/` folder
* documentation in `doc/mld_config.json`
* `mld_emo_from_ComParE_2016.json`: config for emotion detection
* `mld_lng_from_ComParE_2016.json`: config for language detection
* same configurations used for mld.MLD_SUST()
    * some options like window size etc are ignored

## Documentation
* [documented config file](doc/mld_config.json)
* [MLD feature descriptions](doc/features.md)
* [notes on configuration values](doc/config_notes.md)

