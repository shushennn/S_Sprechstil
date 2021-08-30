import os
import sys
import audiofile as af
from audinterface import Feature, Process
import midlevel_descriptors as mld

# test audio files, mono+stereo
f_mono = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "audio", "testaudio_mono.wav")
f_stereo = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "audio", "testaudio_stereo.wav")

# mld extractor
mld_extractor = mld.MLD()

## 1. standalone #######################################
# mono; returns 1-dim np.ndarray
y, fs = af.read(f_mono)
features = mld_extractor(y, fs)
print("mono:", features)

# stereo; returns 2-dim np.ndarray
y, fs = af.read(f_stereo)
features = mld_extractor(y, fs)
print("stereo:", features)

## 2. integration into audinterface #####################
feature_extractor = Feature(feature_names = mld_extractor.feature_names,
                            feature_params = mld_extractor.params,
                            process_func = mld_extractor)

# ... only working with mono; returns dataframe
y, fs = af.read(f_mono)
features = feature_extractor.process_signal(y, fs)
print("integrated:", features)


