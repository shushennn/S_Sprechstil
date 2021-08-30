#!/usr/bin/env python3

import os
import re
from typing import Union, Sequence, List
from multiprocessing import Pool, cpu_count
import json
import copy

from tqdm import tqdm
import pandas as pd
import numpy as np

import opensmile
#from opensmile import FeatureExtractor, FeatureSet, FeatureLevel

import audiofile as af
import mld_pipeline as mp
import myUtilHL as myl


class MLD(object):
    
    r"""midlevel descriptor extracted from opensmile LLDs"""

    def __init__(self, feature_set: str = "mld_emo_from_ComParE_2016",
                 summary_statistics: str = "robust"):

        ''' initialization 
        feature_set: mld_emo_from_ComParE_2016, mld_lng_from_ComParE_2016
        summary_statistics: robust, standard '''
        
        super().__init__()

        f_params = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "config", "{}.json".format(feature_set))

        with open(f_params, "r") as h:
            self.params = json.load(h)

        self.params["summaryStat"] = summary_statistics

        # feature names
        f_featnames = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "config", "feature_names.json")
        with open(f_featnames, "r") as h:
            fn = json.load(h)
        self.feature_names = fn[feature_set]
        

    #def __call__(self, signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    def extract_from_array(self, signal: np.ndarray, sampling_rate: int,
                           segment: bool=True, filename: str=None):
    
        r"""LLD, syllable nucleus, and MLD extraction
        Args:
        signal: (np.array) acoustic signal. If n-dimensional, each row contains
        a channel
        sampling rate: (int) sample rate
        segment: (bool) if True, the signal is further segmented into
            interpausal units (not by auvad but
            prosodic_structure.pau_detector())
        filename: (str) if a file index should be returned
        Returns:
        df_mld (pd.DataFrame) in unified format;
             if more than one channel, a "channel" column is added
             to dataframe with channel index starting with 0
        """
        
        # openSMILE LLD extractor (fex_v is "old" <=0.11, or "new" >=0.20)
        fex, fex_v = featext_init(self.params["opensmile"])
        
        # container dict to be passed to mp.mld_wrapper()
        mwo = {"samplerate": sampling_rate, "config": self.params,
               "output": None, "segment": None, "noseg": not segment,
               "return_nSyl": True}
        # mono
        if len(signal.shape) == 1 or signal.shape[0] == 1:

            if len(signal.shape) == 1:
                mwo["wav"] = signal
            else:
                mwo["wav"] = signal[0,:]
            if fex_v == "old":
                mwo["opensmile"] = fex.extract_from_array(signal, sampling_rate)
            else:
                mwo["opensmile"] = fex.process_signal(signal, sampling_rate)
            df_mld = mp.mld_wrapper(mwo)

        else:
        
            # multichannel file
            df_mld = pd.DataFrame()
            for i in range(len(signal.shape)):
                mwo["wav"] = signal[i,:]
                if fex_v == "old":
                    mwo["opensmile"] = fex.extract_from_array(signal[i,:], sampling_rate)
                else:
                    mwo["opensmile"] = fex.process_signal(signal[i,:], sampling_rate)
                df = mp.mld_wrapper(mwo)
                df["channel"] = [i] * df.shape[0]
                df_mld = pd.concat([df_mld, df])

        # add file name or omit file index
        df_mld.reset_index(inplace=True)
        if filename is None:
            df_mld.drop(columns = ["file"], inplace=True)
            df_mld.set_index(["start", "end"], inplace=True)
        else:
            df_mld["file"] = [filename] * df_mld.shape[0]
            df_mld.set_index(["file", "start", "end"], inplace=True)
        
        return df_mld

    
    def extract_from_index(self,
                           index: Union[pd.DataFrame, pd.Index,
                                        pd.MultiIndex],
                           lld: pd.DataFrame = None,
                           root: str = None,
                           num_jobs: int = 5,
                           channel: int = None,
                           cache_path: str = None) -> pd.DataFrame:

        ''' extract MLD from pd.Index, pd.Multiindex or pd.DataFrame
        with single (file) or multi (file, start, end) index.
        Temporary pkl files are generated (one mld table per input file)
        in the cache folder.
        Args:
        index: pd.Index, pd.Multiindex or pd.DataFrame with index or multindex
             [file, (start, end)]
        lld: (pd.DataFrame, None) of already extracted LLDs. Required only,
             if signal files are not available. If signal files are available
             the LLDs are extracted from them on the fly. 
        root: (str) root folder to be prefixed to file index entries if
              not already part of the index.
        num_jobs: (int) number of parallel jobs. Defaults to 5
        channel: (int) channel 0 or 1. Just relevant for stereo files.
              channel=None converts stereo files to mono (averaging)
              Processing stereo files: requires pd.DataFrame with column "channel"
              with values 0 (left) and 1 (right). Run extractor twice with
              channel=0 and =1. One cache file is stored per channel.
              (adjust cache_path accordingly for each channel).
        cache_path: (str) file name where to store the resulting data frame
        Returns:
        mld: (pd.DataFrame) with same index as in pd.DataFrame index
        '''

        return efi(index, lld, root, num_jobs, channel, cache_path, self.params)
    

class MLD_SUST(object):
    
    r"""midlevel descriptors extracted from sustained sounds"""

    def __init__(self, feature_set: str = "mld_emo_from_ComParE_2016"):

        ''' initialization
        summary_statistics: robust, standard '''
        
        super().__init__()

        f_params = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "config", "{}.json".format(feature_set))

        with open(f_params, "r") as h:
            self.params = json.load(h)

    def extract_from_array(self, signal: np.ndarray, sampling_rate: int,
                           filename: str=None):
    
        r"""LLD, syllable nucleus, and MLD extraction
        Args:
        signal: (np.array) acoustic signal. If n-dimensional, only first channel is
            considered (signal[0])
        a channel
        sampling rate: (int) sample rate
        filename: (str) if a file index should be returned
        Returns:
        df_mld (pd.DataFrame) in unified format
        """
        
        # openSMILE LLD extractor (fex_v is old 0.11, or new 0.20)
        fex, fex_v = featext_init(self.params["opensmile"])

        # container dict to be passed to mp.mld_wrapper()
        mwo = {"samplerate": sampling_rate, "config": self.params,
               "output": None, "segment": None, "noseg": True,
               "return_nSyl": False, "macro": "sustained"}

        # 1st channel only
        if len(signal.shape) > 1:
            print("stereo input. Processing left channel only.")
            signal = signal[0]
        
        mwo["wav"] = signal
        if fex_v == "old":
            mwo["opensmile"] = fex.extract_from_array(signal, sampling_rate)
        else:
            mwo["opensmile"] = fex.process_signal(signal, sampling_rate)

        df_mld = mp.mld_wrapper(mwo)

        # add file name or omit file index
        df_mld.reset_index(inplace=True)
        if filename is None:
            df_mld.drop(columns=["file"], inplace=True)
            df_mld.set_index(["start", "end"], inplace=True)
        else:
            df_mld["file"] = [filename] * df_mld.shape[0]
            df_mld.set_index(["file", "start", "end"], inplace=True)
        
        return df_mld

            
            
    def extract_from_index(self,
                           index: Union[pd.DataFrame, pd.Index,
                                        pd.MultiIndex],
                           lld: pd.DataFrame = None,
                           root: str = None,
                           num_jobs: int = 5,
                           channel: int = None,
                           cache_path: str = None) -> pd.DataFrame:

        ''' extract sustained sound MLDs from pd.DataFrame with single (file) or multi
        (file, start, end) index. Would need to be run twice for stereo
        files (adjust cache_path accordingly for each channel).
        Temporary pkl files are generated (one mld table per input file)
        in the cache folder.
        Args:
        index: pd.Index, pd.Multiindex or pd.DataFrame with index
               or multindex [file, (start, end)]
        lld: (pd.DataFrame, None) of already extracted LLDs. Required only,
             if signal files are not available. If signal files are available
             the LLDs are extracted from them on the fly. 
        root: (str) root folder to be prefixed to file index entries if
              not already part of the index.
        num_jobs: (int) number of parallel jobs. Defaults to 5
        channel: (int) channel 0 or 1. Just relevant for stereo files.
              channel=None converts stereo files to mono (averaging).
              For stereo file input channel=None leads to taking the average
              over both channels.
              Processing stereo files: requires pd.DataFrame with column "channel"
              with values 0 (left) and 1 (right). Run extractor twice with
              channel=0 and =1. One cache file is stored per channel.
              (adjust cache_path accordingly for each channel).
        cache_path: (str) file name where to store the resulting data frame
        Returns:
        mld: (pd.DataFrame) with same index as in pd.DataFrame index
        '''
        
        return efi(index, lld, root, num_jobs, channel, cache_path, self.params, "sustained")


class VAD_MLD(object):
    r"""midlevel descriptors extracted from VAD segmentation"""

    def __init__(self, summary_statistics: str = "robust"):

        ''' initialization
        summary_statistics: robust, standard '''
        
        super().__init__()
        self.params = {"summaryStat": summary_statistics}
        
    def extract_from_index(self,
                           index: Union[pd.DataFrame, pd.Index,
                                        pd.MultiIndex],
                           num_jobs: int = 5,
                           channel: int = None,
                           cache_path: str = None) -> pd.DataFrame:

        ''' extract VAD-based MLDs from pd.DataFrame of segment type (file, start, end index).
        Args:
          index: pd.Index, pd.Multiindex or pd.DataFrame with multindex
          num_jobs: (int) number of parallel jobs. Defaults to 5
          channel: (int) channel 0 or 1. None: signal is converted to mono
              channel=None converts stereo files to mono (averaging).
              Processing stereo files: requires pd.DataFrame with column "channel"
              with values 0 (left) and 1 (right). Run extractor twice with
              channel=0 and =1. One cache file is stored per channel.
              (adjust cache_path accordingly for each channel).
          cache_path: (str) file name where to store the resulting data frame
        Returns:
          mld: (pd.DataFrame) with file index and VAD-pause summaryStat features
        '''
        
        if cache_path:
            if os.path.exists(cache_path):
                return pd.read_pickle(cache_path)
            else:
                cache_dir, cache_stm, cache_ext = myl.dfe(cache_path)
        else:
            cache_dir, cache_stm, cache_ext = myl.dfe("/tmp/mld_vad/mld_vad.pkl")
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)

        if isinstance(index, pd.Index) or isinstance(index, pd.MultiIndex):
            index = index.to_frame()
        else:
            index = index.reset_index(inplace=False)
            
        # if channel information is provided
        # create subset of dataframe for respective channel
        if channel and "channel" in index.columns:
            index = index.loc[index["columns"] == channel]

        mpo = []
        ffo = []
        foi = 0
        
        ff = sorted(set(index["file"].to_list()))
        for f in ff:
            fo_infx = str(foi).zfill(40)
            foi += 1            
            if channel:
                fo = os.path.join(cache_dir, "{}-{}-{}-mld_vad.pkl".format(cache_stm,
                                                                           fo_infx,
                                                                           channel))
            else:
                fo = os.path.join(cache_dir, "{}-{}-mld_vad.pkl".format(cache_stm,
                                                                        fo_infx))

            ffo.append(fo)
            # warm start
            if os.path.isfile(fo):
                continue
            
            mpo.append({"fo": fo, "index": index,
                        "select": f, "opt": self.params})
            
        # multi- or single-processing (multi: one process per file)
        if not num_jobs:
            # num_jobs = cpu_count()-1
            num_jobs = 5
        if num_jobs > 1:
            pool = Pool(num_jobs)
            list(tqdm(pool.imap(mld_vad_feat_thread, mpo),
                      desc='vad-midlevel descriptors',
                      total=len(mpo)))
        else:
            for i in tqdm(range(len(mpo))):
                mld_vad_feat_thread(mpo[i])

        # concatenate
        df = None
        for fo in ffo:
            if not os.path.isfile(fo):
                continue
            mld = pd.read_pickle(fo)
            if df is None:
                df = copy.deepcopy(mld)
            else:
                df = pd.concat([df, mld])

        # clean up
        for fo in ffo:
            if os.path.isfile(fo):
                os.remove(fo)

        # final output
        if (df is not None) and cache_path:
            df.to_pickle(cache_path)
            
        return df



def mld_vad_feat_thread(o):
    
    # summary statistics keys
    ssk = mp.summaryStatKeys(o["opt"])

    # output df
    mld = {"file": [],
           "pau_prop": np.array([]),
           "pau_count": np.array([]),
           "pau_rate": np.array([])}
    
    for x in ["pau", "vad"]:
        for y in ssk:
            mld["{}_{}".format(x, y)] = np.array([])

    df = o["index"].loc[o["index"]["file"]==o["select"]]
    df = myl.from_timedelta(df)
    starts = df["start"].to_numpy()
    ends = df["end"].to_numpy()
    total = ends[-1] - starts[0]
    vad = ends - starts
    pau = starts[1:len(starts)] - ends[0:len(ends)-1]
    n_pau = df.shape[0]-1
    vad_sumStat = myl.summary_stats(vad, o["opt"]["summaryStat"])
    pau_sumStat = myl.summary_stats(pau, o["opt"]["summaryStat"])

    mld["file"] = np.append(mld["file"], o["select"])
    # time proportion of pauses
    mld["pau_prop"] = np.append(mld["pau_prop"], np.sum(pau)/total)
    # total number of pauses
    mld["pau_count"] = np.append(mld["pau_count"], n_pau)
    # number of pauses per time
    mld["pau_rate"] = np.append(mld["pau_rate"], n_pau/total)
    
    for x in vad_sumStat:
        u = "vad_{}".format(x)
        if u in mld:
            mld[u] = np.append(mld[u], vad_sumStat[x])
                    
    for x in pau_sumStat:
        u = "pau_{}".format(x)
        if u in mld:
            mld[u] = np.append(mld[u], pau_sumStat[x])

    #for x in mld:
    #    print(x, ":", mld[x])
    #myl.stopgo()
        
    mld = pd.DataFrame.from_dict(mld)
    mld.set_index(["file"], inplace=True)
    
    mld.to_pickle(o["fo"])
        
        
def efi(index, lld, root, num_jobs, channel, cache_path, params, macro=None):

    ''' joint extract_from_index() function for MLD() and MLD_SUST() '''
    
    # cache
    if cache_path:
        if os.path.exists(cache_path):
            return pd.read_pickle(cache_path)
        else:
            cache_dir, cache_stm, cache_ext = myl.dfe(cache_path)
    else:
        cache_dir, cache_stm, cache_ext = myl.dfe("/tmp/mld/mld.pkl")
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

    # convert index to dataframe
    if isinstance(index, pd.Index) or isinstance(index, pd.MultiIndex):
        index = index.to_frame()

    # if channel information is provided
    # create subset of dataframe for respective channel
    if channel and "channel" in index.columns:
        index = index.loc[index["columns"] == channel]
        
    # opensmile feature extractor
    fex, fex_v = featext_init(params["opensmile"])

    # unique sorted file list
    files = sorted(set(index.index.get_level_values(0)))
    
    # options for file-wise processing
    mpo = []
    ffo = []
    # output file idx
    foi = 0

    if macro == "sustained":
        fset_infx = "mld_sust"
        descr = "sustained sound midlevel descriptors"
    else:
        fset_infx = "mld"
        descr = "midlevel descriptors"
        
    for f in files:
        if root:
            fi = os.path.join(root, f)
        else:
            fi = f
        if not os.path.isfile(fi) and lld is None:
            print(fi, ": not found. Skipped.")
            continue

        fo_infx = str(foi).zfill(40)
        foi += 1
            
        if channel:
            fo = os.path.join(cache_dir, "{}-{}-{}-{}.pkl".format(cache_stm,
                                                                  fo_infx,
                                                                  channel,
                                                                  fset_infx))
        else:
            fo = os.path.join(cache_dir, "{}-{}-{}.pkl".format(cache_stm,
                                                               fo_infx,
                                                               fset_infx))

        ffo.append(fo)
        # warm start
        if os.path.isfile(fo):
            continue
            
        # if LLDs are provided, select the current file's subset
        if lld is not None:
            lld_f = lld.loc[f]
        else:
            lld_f = None

        mpo.append({"f": fi, "fo": fo, "fex": fex, "fex_v": fex_v, "lld": lld_f,
                    "select": f, "channel": channel, "index": index,
                    "opt": params, "macro": macro})

        
    # multi- or single-processing (multi: one process per file)
    if not num_jobs:
        # num_jobs = cpu_count()-1
        num_jobs = 5
    if num_jobs > 1:
        pool = Pool(num_jobs)
        list(tqdm(pool.imap(mld_feat_thread, mpo),
                  desc=descr,
                  total=len(mpo)))
    else:
        for i in tqdm(range(len(mpo))):
            mld_feat_thread(mpo[i])

    # concatenate
    df = None
    for fo in ffo:
        if not os.path.isfile(fo):
            continue
        mld = pd.read_pickle(fo)
        if df is None:
            df = copy.deepcopy(mld)
        else:
            df = pd.concat([df, mld])

    # clean up
    for fo in ffo:
        if os.path.isfile(fo):
            os.remove(fo)
                
    if (df is not None) and cache_path:
        df.to_pickle(cache_path)

    return df
    
    
def mld_feat_thread(o):

    ''' MLD extraction for single file o["select"] '''

    # extract LLDs (requires AUDIO)
    if o["lld"] is None:
        
        # audio input
        s, sr = af.read(o["f"])
    
        # ... to mono
        if len(s.shape)>1:
            s = remix_mono(s,o["channel"])
        
        # openSmile LLD extraction
        if o["fex_v"] == "old":
            lld = o["fex"].extract_from_array(s, sr)
        else:
            lld = o["fex"].process_signal(s, sr)
    else:
        #try:
        #    s, sr = af.read(o["f"])
        #except:
        s, sr = None, None
        lld = o["lld"]

    lld = add_file_idx(lld,o["select"])
    
    # return index type
    rmi = isinstance(o["index"].index, pd.MultiIndex)

    # MLDs
    mwo = {"wav": s, "opensmile": lld, "config": o["opt"],
           "samplerate": sr, "select": o["select"], "output": None,
           "output_format": "pickle", "file_index": o["select"],
           "segment": o["index"], "f0_nrm_model": None,
           "return_multi_index": rmi, "return_nSyl": True,
           "macro": o["macro"]}
    
    mld = mp.mld_wrapper(mwo)
    if mld is not None:
        mld.to_pickle(o["fo"])
    else:
        print(o["fo"], ": no output")
    
    return True


def remix_mono(s,c):
    ''' remix to mono. c: channel {None, 0, 1} '''
    if c is None:
        s = s.astype(float)
        return (s[0]+s[1])/2
    return s[c]

def add_file_idx(df,f):
    ''' adds file index to openSmile dataframe '''

    df["file"] = [f] * df.shape[0]
    
    if "file" in df.index.names:
        df.set_index("file", inplace=True)
    else:
        if "start" in df.index.names:
            index_col = ["file", "start", "end"]
        else:
            index_col = "file"
        df.reset_index(inplace=True)
        df.set_index(index_col, inplace=True)
        #df.set_index("file", append=True, inplace=True)

    return df


def featext_init(fset: str = "ComParE_2016"):

    r'''init openSmile feature extractor by feature set and level lld, fun;
    for pyopensmile <=0.11.1 or >= 0.20.1
    Args:
    fset (str) feature set
    Returns
    fex (opensmile extractor object)
    fev_v: (str) version "old", "new", relevant for how to apply it
    '''

    # test whether pyopensmile >= 0.20.1 or <=0.11.1 is installed
    try:
        mySmile = opensmile.Smile()
        fex_v = "new"
    except:
        fex_v = "old"
        
    if fex_v == "new":
    
        if fset == "ComParE_2016_Basic":
            return opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016_Basic,
                                   feature_level=opensmile.FeatureLevel.LowLevelDescriptors), fex_v
        if fset == "ComParE_2016":
            return opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016, 
                                   feature_level=opensmile.FeatureLevel.LowLevelDescriptors), fex_v
        if fset == "GeMAPS":
            return opensmile.Smile(feature_set=opensmile.FeatureSet.GeMAPS, 
                                   feature_level=opensmile.FeatureLevel.LowLevelDescriptors), fex_v
        if fset == "eGeMAPS":
            return opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPS, 
                                   feature_level=opensmile.FeatureLevel.LowLevelDescriptors), fex_v
        if fset == "eGeMAPSv01a":
            return opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv01a, 
                                   feature_level=opensmile.FeatureLevel.LowLevelDescriptors), fex_v
        if fset == "eGeMAPSv01b":
            return opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv01b, 
                                   feature_level=opensmile.FeatureLevel.LowLevelDescriptors), fex_v
        if fset == "GeMAPSv01a":
            return opensmile.Smile(feature_set=opensmile.FeatureSet.GeMAPSv01a, 
                                   feature_level=opensmile.FeatureLevel.LowLevelDescriptors), fex_v
        if fset == "GeMAPSv01b":
            return opensmile.Smile(feature_set=opensmile.FeatureSet.GeMAPSv01b, 
                                   feature_level=opensmile.FeatureLevel.LowLevelDescriptors), fex_v
        if fset == "GeMAPSplus_v01":
            return opensmile.Smile(feature_set=opensmile.FeatureSet.GeMAPSplus_v01, 
                                   feature_level=opensmile.FeatureLevel.LowLevelDescriptors), fex_v
        if fset == "GeMAPSplus_v01b_egemaps":
            return opensmile.Smile(feature_set=opensmile.FeatureSet.GeMAPSplus_v01b_egemaps, 
                                   feature_level=opensmile.FeatureLevel.LowLevelDescriptors), fex_v

    else:
        if fset == "ComParE_2016_Basic":
            return opensmile.FeatureExtractor(feature_set=opensmile.FeatureSet.ComParE_2016_Basic,
                                              feature_level=opensmile.FeatureLevel.LowLevelDescriptors), fex_v
        if fset == "ComParE_2016":
            return opensmile.FeatureExtractor(feature_set=opensmile.FeatureSet.ComParE_2016, 
                                              feature_level=opensmile.FeatureLevel.LowLevelDescriptors), fex_v
        if fset == "GeMAPS":
            return opensmile.FeatureExtractor(feature_set=opensmile.FeatureSet.GeMAPS, 
                                              feature_level=opensmile.FeatureLevel.LowLevelDescriptors), fex_v
        if fset == "eGeMAPS":
            return opensmile.FeatureExtractor(feature_set=opensmile.FeatureSet.eGeMAPS, 
                                              feature_level=opensmile.FeatureLevel.LowLevelDescriptors), fex_v
        
    return None



def featext_init_0_11(fset: str = "ComParE_2016"):
    r'''init openSmile feature extractor by feature set and level lld, fun;
    for pyopensmile <= 0.11.x'''

    if fset == "ComParE_2016_Basic":
        return opensmile.FeatureExtractor(feature_set=opensmile.FeatureSet.ComParE_2016_Basic,
                                          feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    if fset == "ComParE_2016":
        return opensmile.FeatureExtractor(feature_set=opensmile.FeatureSet.ComParE_2016, 
                                          feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    if fset == "GeMAPS":
        return opensmile.FeatureExtractor(feature_set=opensmile.FeatureSet.GeMAPS, 
                                          feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    if fset == "eGeMAPS":
        return opensmile.FeatureExtractor(feature_set=opensmile.FeatureSet.eGeMAPS, 
                                          feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    return None


