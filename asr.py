import os
import shutil
import sys
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import speech_recognition as sr
import audiofile as af


class ASR():

    def __init__(self, language):

        super().__init__()
        self.language = language

        
    def extract_from_index(self, index, margin=0.0,
                           cache_path=None, tmp="/tmp"):

        ''' returns dataFrame with index from index.index and column "asr" 
        Args:
        index: (pd.Dataframe, pd.Index, pd.MultiIndex)
        margin: (float) if index is multiindex, add margins to start and end
                Reason: auvad() tends to cut off voiceless phonemes 
        cache_path: (str) name of cached pkl file
        tmp: (str) temporary working directory
        Returns:
        df (pd.DataFrame) with index as in index and column "asr"
        '''

        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as h:
                return pickle.load(h)

        print("ASR ...")
        
        # convert index to dataframe    
        if isinstance(index, pd.Index) or isinstance(index, pd.MultiIndex):
            index = index.to_frame()

        #index = index.iloc[0:10] #!x
            
        # speech recognizer
        reco = sr.Recognizer()
        
        # asr language mapping
        if self.language == "deu":
            lng = "de-DE"
        elif self.language == "eng":
            lng = "en-EN"
        else:
            sys.exit("{}: not supported".format(self.language))
            
        # sr output
        text = []

        # resampler
        # resample = audsp.Resample(target_rate=16000, quality=audsp.define.ResampleQuality.HIGH)

        #ff = sorted(set(index.index.get_level_values("file")))
        ff = index.index.get_level_values("file")

        fprev = ""
        for i in tqdm(range(len(ff))):
            if ff[i] == fprev:
                continue
            f, fprev = ff[i], ff[i]
            fo = f"{tmp}/audio.wav"
            dfx = index.loc[f]

            # 16 kHz audio cache
            y, fs = af.read(f)
            len_audio = len(y)*fs
            if fs != 16000:
                y = resample(y, fs)
                if len(y.shape) > 1:
                    y = y[0]
                af.write(fo, y, 16000)
            else:
                shutil.copyfile(f, fo)  

            if isinstance(index.index, pd.MultiIndex):

                starts = dfx.index.get_level_values("start").total_seconds().to_numpy()
                ends = dfx.index.get_level_values("end").total_seconds().to_numpy()
      
                for i in range(len(starts)):

                    si, ei = starts[i], ends[i]
                    if margin > 0:
                        si = np.max([0, si-margin])
                        ei = np.min([len_audio, ei+margin])
                        
                    o, d = si, ei - si
                    try:
                        with sr.AudioFile(fo) as h:
                            y = reco.record(h, offset=o, duration=d)  
                        s = reco.recognize_google(y, language=lng)
                    except:
                        s = ""
                    text.append(s)
            else:
                with sr.AudioFile(fo) as h:
                    y = reco.record(h)
                try:
                    s = reco.recognize_google(y, language=lng)
                except:
                    s = ""
                text.append(s)
        
        df = pd.DataFrame({"asr": text})
        df.index = index.index
        
        if cache_path:
            df.to_pickle(cache_path)
                     
        return df
    
