#!/usr/bin/env python3

import mld_core as mld
import mld_timestamps as ts
import preproc as pp
import myUtilHL as myl
import pandas as pd
import datetime as dt
import sys
import numpy as np
import copy as cp
import argparse
import re
import audiofile as af



def mld_wrapper(args):
    r''' wrapper around mid-level descriptor extraction for single
    file (e.g. selected from pd.DataFrame in unif format).
    Script can be called embedded or as standalone from terminal.
    For standalone call, see: > python3 mld_wrapper.py -h
    Args:
      args (dict) with keys:
         wav (str or np.array or None):
                      if str: name of mono audio file
                      if np.array: one-dim array containing the
                      mono signal (requires the argument "samplerate")
                      if None, syllable nuclei will be extracted based on
                      opensmile pd.DataFrame
         opensmile (str or pd.Dataframe)
                      if str: file name of openSmile LLD table
                      if pd.Dataframe: dataframe returned by
                      opensmile.FeatureExtractor() for single file
         config (str or dict)
                      if str: name of json configuration file;
                         see e.g. ../minex/example.json
                      if dict: configuration dict
         output (str) output file name, if results should be written
                         to file
                      (separated csv file with header containing the
                       MLDs and the indices "start" and "end". One
                          line per VAD segment)
         segment (str or pd.Dataframe or np.array, None)
                      if str: name of a csv file containing a table
                           with columns "start" and "end".
                           This could be an annotation table or the
                           output of the auvad module.
                           In case this table contains more than one
                           file, the "file" argument needs to be specified
                           to select the corresponding subset.
                      if pd.Dataframe: dataframe in unified format with
                           indices "start" and "end".
                           This could be an annotation table or the
                           output of the auvad module.
                           In case this table contains more than one
                           file, the "file" argument needs to be specified
                           to select the corresponding subset.
                      if np.array: two-dimensional array [[start end] ...]
                      if None: if "noseg" flag is set, no chunking into
                           VAD segments is
                        carried out but just one row of MLDs is
                        calculated for the entire file. If "noseg" flag
                        is not set, the segments's time stamps are
                        calculated internally (which is different from
                        openSmile VAD!). time onsets and offsets.
         return_multi_index (boolean): if False, only "file" index is returned
         select (str): name of audio file for corresponding subset selection
                       in segment dataframe
         noseg (flag) if this flag is set, no chunking into VAD segments is
                        carried out but just one row of MLDs is calculated
                        for the entire file
         f0_nrm_model (sklearn object): model for f0 speaker normalization
         samplerate (int): of audio; needed if audio args["wav"] is provided
                        as np.array
         v (flag): if this flag is set, warnings are verbosed
         macro (str, None): macrosetting; "sustained" - feature extraction from
                   sustained sound (i.e. no syllable extraction etc)
         return_nSyl (boolean): <False> if True, number of syllables is
                    outputted as hld_nSyl for evtl. later data filtering
                    (some MLDs only make sense for more than 1 syllable in
                     chunk)
    Returns:
      feat: dataframe in unified format, one row per chunk containing
            midlevel features.
            Indices: "start", "stop", (and "file", if "audfile" is provided).
            If input cannot be processed, feat is None.
    File output: to args["output"] if set
    '''

    # configuration #########################
    args = myl.opt_default(args, {"noseg": False, "v": False,
                                  "output_format": "csv",
                                  "select": None, "file_index": None,
                                  "return_nSyl": False,
                                  "f0_nrm_model": None,
                                  "return_multi_index": True,
                                  "macro": None})
    opt = myl.args2opt(args)
    opt = myl.opt_default(opt, {"col_f0": "F0final_sma",
                                "col_energy": "pcm_LOGenergy_sma",
                                "col_mld_voi": "voicingFinalUnclipped_sma",
                                "f0_nrm": "ml",
                                "summaryStat": "robust",
                                "mld_varRat_fe": opt["mld_varRat"],
                                "mld_voi": {},
                                "verbose": args["v"],
                                "macro": args["macro"]})
    
    # macro settings
    if opt["macro"] == "sustained":
        opt["f0_nrm"] = "no"
        opt["f0_nrm_model"] = None
        opt["return_nSyl"] = False
        
    # map summary stat to all subdicts
    for x in opt:
        if ((type(opt[x]) is dict) and (re.search("[mlh]ld", x))):
            opt[x]["summaryStat"] = opt["summaryStat"]
            
    # print("step 1")
    # distribute summaryStat type to subdicts
    for sd in opt:
        if ((type(opt[sd]) is dict) and (sd not in ["pause", "syl"])):
            opt[sd] = myl.opt_default(
                opt[sd], {"summaryStat": opt["summaryStat"]})
    # time series analyses defaults
    opt = myl.opt_default(opt, {"col_corrDist": [opt["col_f0"],
                                                 opt["col_energy"]],
                                "col_align": [opt["col_f0"],
                                              opt["col_energy"]],
                                "preproc": {"zi": 1, "st": 1}})
    # print("step 2")
    if "mld_corrDist" in opt:
        opt["mld_corrDist"] = myl.opt_default(
            opt["mld_corrDist"], {"drop_zeroF0": True})

    # read oS table #########################
    # print("step 3")
    # from file outputted by openSmile
    if type(args["opensmile"]) is str:
        os_in = pd.read_csv(args["opensmile"], sep=";")
    # from pd.dataframe generated by pyopensmile
    else:
        os_in = args["opensmile"]

    # select rows for current file ##########
    # for current audio file in args[select]
    os = tab_subset(os_in, args)
    # add frameTime column
    os = os_add_frameTime_col(os, args)

    # print("step 4")
    # columns check #########################
    opt, navi, file_verbose = check_columns(os, opt, args)

    # print("step 5")
    # read wav file, DC removal #############
    if type(args["wav"]) is str:
        # fs, s = sio.read(args["wav"])
        # s = myl.wav_int2float(s)
        sig, fs = af.read(args["wav"])
        sig = sig - np.mean(sig)
    else:
        sig = args["wav"]
        if ((sig is not None) and (not args["samplerate"])):
            sys.exit("samplerate is required, if audio is provided as array.")
        fs = args["samplerate"]

    if ((sig is not None) and (len(sig.shape) > 1)):
        print("{} - Only mono signals can"
              "be processed! Return None.".format(file_verbose))
        return None

    opt["pause"]["fs"] = fs
    opt["syl"]["fs"] = fs

    # fallback chunk segments: [[0 fileDuration]]
    if sig is None:
        t_chunks_fallback = np.array([[0, os["frameTime"].iloc[-1]]])
    else:
        t_chunks_fallback = np.array([[0, len(sig)/fs]])

    # print("step 6")
    # read segments ########################
    # no segmentation of audio
    t_chunks_dt = None
    if args["noseg"]:
        t_chunks = t_chunks_fallback
    # segmentation
    elif (("segment" in args) and (args["segment"] is not None)):
        # array
        if myl.of_list_type(args["segment"]):
            t_chunks = args["segment"]
        # csv file, unified-format dataframe
        else:
            if type(args["segment"]) is str:
                tab_in = pd.read_csv(args["segment"], sep=",")
            else:
                tab_in = args["segment"]
            tab = tab_subset(tab_in, args)
            t_chunks, t_chunks_dt = pd_idx2intervals(tab, args)

        if not myl.of_list_type(t_chunks):
            t_chunks = t_chunks_fallback

        # make sure that array is 2-dim
        if len(t_chunks.shape) == 1:
            t_chunks = np.array([t_chunks])
    else:
        t_chunks = myl.ea()

    # print("step 7")
    # prosodic structure ####################
    if len(t_chunks) == 0:
        if sig is None:
            t_chunks = t_chunks_fallback
            if opt["macro"] == "sustained":
                ncl = None
            else:
                ncl = ts.syl_ncl_from_df(os)
        else:
            pc, ncl = ts.prosodic_structure(sig, fs, opt["pause"], opt["syl"])
            t_chunks = pc["tc"]
    else:
        if opt["macro"] == "sustained":
            ncl = None
        else:
            if sig is None:
                ncl = ts.syl_ncl_from_df_wrapper(os, t_chunks)
            else:
                i_chunks = (np.asarray(t_chunks) * fs).astype(int)
                ncl = ts.syl_ncl_wrapper(sig, i_chunks, opt["syl"])

    if opt["macro"] == "sustained":
        t_ncl = None
    else:
        t_ncl = ncl["t"]
    
    # timedelta values for feature table output
    # (if input is audb df, t_chunks_dt already contains the original time stamps)
    if t_chunks_dt is None:
        t_chunks_start = [dt.timedelta(seconds=i) for i in t_chunks[:, 0]]
        t_chunks_end = [dt.timedelta(seconds=i) for i in t_chunks[:, 1]]
        t_chunks_dt = np.column_stack((t_chunks_start, t_chunks_end))

    # print("step 8")
    # F0 processing #########################
    # linear interpolation over zeros and outliers
    # (if f0-related features to be extracted,
    # among which is "mld_register")
    cn_f0 = opt["col_f0"]
    # store orig f0
    f0_orig = cp.deepcopy(os[cn_f0])

    # no speech signal -> return None
    if np.max(f0_orig) <= 0:
        print("{} - probably no speech signal."
              "Returning None".format(file_verbose))
        return None

    # f0 speaker normalization (needs to be done separately from and
    # before pp.preproc_f0, since
    # it's to be done per segment and on Hz values)
    if args["f0_nrm_model"] is not None:
        os[cn_f0] = pp.f0_spk_nrm(os[cn_f0], os["frameTime"],
                                  t_chunks, args["f0_nrm_model"])

    # preprocessing
    # os[cn_f0]: preprocessed f0 which might further undergo midline normalization
    #            (e.g. for later local contour stylization)
    # os["f0_noninterp"]: as os[cn_f0] but without zero-interpolation
    # os["f0_preproc"]: as os[cn_f0]. Frozen state. Won't undergo midline nrm
    #            (for later register feature extraction)
    if navi["mld_register"]:
        i_zero = myl.find(os[cn_f0],"==",0)
        f0_preproc = pp.preproc_f0(os[cn_f0], opt["preproc"])
        os["f0_preproc"] = cp.deepcopy(f0_preproc)
        f0_noninterp = cp.deepcopy(f0_preproc)
        f0_noninterp[i_zero] = 0
        os["f0_noninterp"] = f0_noninterp
        os[cn_f0] = f0_preproc
    else:
        os["f0_preproc"] = cp.deepcopy(os[cn_f0])
        os["f0_noninterp"] = cp.deepcopy(os[cn_f0])
    
    # print("step 9")
    # F0 register removal ###################
    # - fit and subtract f0 register midline separately within each chunk
    #   to keep only local f0 movements relevant for the
    #   f0-related features subsequently extracted by mld.mld_shape() and
    #   mld.mld_corrDist()
    # - copy f0 column, so that mld_register() features can be
    #   calculated separately
    # - set all register normalization to None so that it's not
    #   subtracted twice
    os["f0_preproc"] = cp.deepcopy(os[cn_f0])
    if opt["f0_nrm"] == "ml":
        os[cn_f0] = pp.subtract_register(os[cn_f0], os["frameTime"],
                                         t_chunks, args)
        for x in opt:
            if type(opt[x]) is dict:
                opt[x]["register"] = None

    # print("step 10")
    # tolerances for corrDist sample entropy calculations ############
    tol = {}
    if navi["mld_corrDist"]:
        if opt["mld_corrDist"]["drop_zeroF0"]:
            nzi = myl.find(f0_orig, ">", 0)
        else:
            nzi = myl.idx(f0_orig)
        for c in opt["col_corrDist"]:
            if len(nzi) == 0:
                tol[c] = None
            else:
                tol[c] = 0.2*np.std(os[c][nzi])

    # print("step 11")
    # chunk-wise processing #################
    feat = {}
    # for t_chunk in t_chunks:
    for i in myl.idx(t_chunks):
        # print("\tstep 11x")
        if opt["macro"] == "sustained":
            feat = mld_sust_per_chunk(feat, os, sig, fs, t_chunks[i],
                                      t_chunks_dt[i], tol, opt, navi, args)
        else:
            feat = mld_per_chunk(feat, os, sig, fs, t_chunks[i],
                                 t_chunks_dt[i], t_ncl, tol,
                                 opt, navi, args)
        
    # print("step 12")
    # to DataFrame ######################
    feat = pd.DataFrame.from_dict(feat)

    # print("step 13")
    # output csv (before indexing) ##########
    if args["output"] and args["output_format"] == "csv":
        myl.write_wrapper(feat, args["output"], args["output_format"])
        
    # index generation ######################
    # has no effect after df conversion
    #feat["start"] = np.asarray(feat["start"], dtype='timedelta64[ns]')
    #feat["end"] = np.asarray(feat["end"], dtype='timedelta64[ns]')

    if "file" in feat.columns:
        if args["return_multi_index"]:
            feat.set_index(["file", "start", "end"], inplace=True)
            #feat.index.levels[1].astype('timedelta64')
            #feat.index.levels[2].astype('timedelta64')
            
        else:
            feat.set_index(["file"], inplace=True)
            feat.drop(columns = ["start", "end"], inplace=True)
    else:
        feat.set_index(["start", "end"], inplace=True)

        
    # output ################################
    if args["output"] and args["output_format"] == "pickle":
        myl.write_wrapper(feat, args["output"], args["output_format"])
        
    return feat


def pd_idx2intervals(tab, args):
    ''' return 2x 2-dim time interval np.array in abs seconds and timedeltas
    Args:
    tab: audb df time segment table with segment start and end information
    args
    Returns:
    t_chunks: (2d np.array) [[start end] ...] in absolute seconds
    t_chunks_dt: (2d np.array) [[start end] ...] in timedelta format
    Comments:
    t_chunks is used as input for the feature extractor,
    t_chunks_dt: will be written to feature table so that MLD table
       and segment table are properly time-aligned
    '''
    
    # from column (timedelta or abs sec format)
    if "start" in tab:
        if is_numeric(tab["start"][0]):
            t_start = np.asarray(tab["start"])
            t_end = np.asarray(tab["end"])
            t_start_dt, t_end_dt = None, None
        else:
            t_start = np.asarray([tab["start"].dt.total_seconds()])
            t_end = np.asarray([tab["end"].dt.total_seconds()])
            t_start_dt = tab["start"]
            t_end_dt = tab["end"]
    # from index (always in timedelta format)
    elif "start" in tab.index.names:
        t_start = np.asarray([tab.index.get_level_values('start').total_seconds()])
        try:
            t_end = np.asarray([tab.index.get_level_values('end').total_seconds()])
        except:
            # unspecified end time (as in mixed file/segment-type benchmarks)
            # set to None so that entire file is treated as single segment
            t_start, t_end = None, None
        # catch NaT if not captured right above
        if ((t_end is not None) and len(t_end[0]) == 1 and np.isnan(t_end[0][0])):
            t_start, t_end = None, None
        t_start_dt = tab.index.get_level_values('start')
        t_end_dt = tab.index.get_level_values('end')
    else:
        if args["v"]:
            print("cannot infer segment starts and ends from table."
                  "Entire file is treated as single segment.")
        return None, None

    # chunks in abs sec
    if t_start is None:
        t_chunks = None
    else:
        if not myl.of_list_type(t_start[0]):
            t_start = np.asarray([t_start])
            t_end = np.asarray([t_end])
        t_chunks = np.concatenate((t_start.T, t_end.T), axis=1)

    # chunks in timedelta (fallback: infer from abs sec)
    if t_start_dt is not None:
        t_chunks_dt = np.column_stack((t_start_dt, t_end_dt))
    else:
        t_chunks_start = [dt.timedelta(seconds=i) for i in t_chunks[:, 0]]
        t_chunks_end = [dt.timedelta(seconds=i) for i in t_chunks[:, 1]]
        t_chunks_dt = np.column_stack((t_chunks_start, t_chunks_end))

    return t_chunks, t_chunks_dt


def is_numeric(x):
    if type(x) is float:
        return True
    if isinstance(x, np.float64):
        return True
    if isinstance(x, np.int64):
        return True
    return False


def os_add_frameTime_col(os, args):
    ''' adds frameTime column to openSmile table '''
    if "frameTime" in os:
        return os
    t_chunks, t_chunks_dt = pd_idx2intervals(os, args)
    os["frameTime"] = t_chunks[:, 0]
    return os


def tab_subset(tab, args):
    ''' return table subset for specified audio file in args[select] '''
    # no selection
    if not args["select"]:
        return tab
    # column
    if "file" in tab.columns:
        return tab.loc[tab["file"] == args["select"]]
    # index
    if "file" in tab.index.names:
        ff = tab.index.get_level_values("file")
        return tab.loc[ff == args["select"]]
        # return tab.loc[args["select"]]
    if args["v"]:
        print("table does not contain a 'file' column or index. Not"
              "possible to create subset for {}".format(args["select"]))
    return tab


def check_columns(os, opt, args):
    ''' tests, whether columns in opt["col_*"] are in os table.
    If not, the corresponding features won't be extracted '''

    osk = set(os.keys())
    s_missing = ": column not in openSmile table. Corresponding features" \
                "are not extracted"

    navi = {"hld": True}

    if opt["col_f0"] not in osk:
        if opt["verbose"]:
            print("{}{}".format(opt["col_f0"], s_missing))
        b = False
    else:
        b = True
    navi["mld_shape"] = b
    navi["mld_varRat"] = b
    navi["mld_varRat_fe"] = b
    navi["mld_register"] = b
    navi["mld_rhy_f0"] = b

    if opt["col_energy"] in osk:
        navi["mld_rhy_en"] = True
        navi["mld_shape_en"] = True
    else:
        if opt["verbose"]:
            print("{}{}".format(opt["col_energy"], s_missing))
        navi["mld_rhy_en"] = False
        navi["mld_shape_en"] = False

    if opt["col_mld_voi"] in osk:
        navi["mld_voi"] = True
    else:
        if opt["verbose"]:
            print("{}{}".format(opt["col_mld_voi"], s_missing))
        navi["mld_voi"] = False
        
    navi["mld_rhy_iso"] = True
    
    col_lld = []
    if "col_lld" in opt:
        for c in opt["col_lld"]:
            if c in osk:
                col_lld.append(c)
            elif opt["verbose"]:
                print("{}{}".format(c, s_missing))
    opt["col_lld"] = col_lld
    if len(col_lld) > 0:
        navi["lld"] = True
    else:
        navi["lld"] = False

    navi["mld_corrDist"] = True
    for c in opt["col_corrDist"]:
        if c not in osk:
            if opt["verbose"]:
                print("{}{}".format(c, s_missing))
            navi["mld_corrDist"] = False
            break

    # mld align
    navi["mld_align"] = True
    for c in opt["col_align"]:
        if c not in osk:
            if opt["verbose"]:
                print("{}{}".format(c, s_missing))
            navi["mld_align"] = False
            break

    # mld_vec: default from mfcc
    if "col_mld_vec" not in opt:
        opt["col_mld_vec"] = "mfcc"

    navi["mld_vec"] = True
    navi["mld_inter"] = True

    # str: e.g. mfcc, audSpec -> list of features whoes names contain
    #      this pattern
    # list: explicit list of features
    if type(opt["col_mld_vec"]) is str:
        pat = opt["col_mld_vec"]
        opt["col_mld_vec"] = []
        for c in osk:
            if re.search(pat, c):
                opt["col_mld_vec"].append(c)
        if len(opt["col_mld_vec"]) == 0:
            if opt["verbose"]:
                print("{}{}".format(pat, s_missing))
            navi["mld_vec"] = False
            navi["mld_inter"] = False
        else:
            opt["col_mld_vec"] = col_sort(opt["col_mld_vec"], pat)
    else:
        for c in opt["col_mld_vec"]:
            if c not in osk:
                if opt["verbose"]:
                    print("{}{}".format(c, s_missing))
                navi["mld_vec"] = False
                navi["mld_inter"] = False
                break

    if "col_mld_disp" in opt:
        navi["mld_disp"] = True
        for c in opt["col_mld_disp"]:
            if c not in osk:
                if opt["verbose"]:
                    print("{}{}".format(c, s_missing))
                navi["mld_disp"] = False
                break
    else:
        navi["mld_disp"] = False
        for c in osk:
            if re.search("lspFreq", c):
                navi["mld_disp"] = True
                break
        opt["col_mld_disp"] = None

    # file name to be reported in error messages
    if args["select"] is not None:
        file_verbose = args["select"]
    elif type(args["wav"]) is str:
        file_verbose = args["wav"]
    elif type(args["opensmile"]) is str:
        file_verbose = args["opensmile"]
    else:
        file_verbose = ""

    return opt, navi, file_verbose


def col_sort(cols, pat):
    ''' sorts column names by index (e.g. mfcc[2]<mfcc[10])
    or alphanumerically '''

    if not re.search("(mfcc|audSpec)", pat):
        return sorted(cols)

    for i in range(0, len(cols)-1):
        for j in range(i+1, len(cols)):
            x = cols[i].split("[")
            y = cols[j].split("[")
            x = int(re.sub(r"\]", "", x[-1]))
            y = int(re.sub(r"\]", "", y[-1]))
            if y < x:
                b = cols[i]
                cols[i] = cols[j]
                cols[j] = b

    return cols


def mld_sust_per_chunk(feat, os, sig, fs, t_interval_margin,
                       t_interval_dt, tol, opt, navi, args):

    ''' mld sust set for each chunk in current db-table.
    For sustained vowels etc. '''

    t_interval, t_interval_margin = adjust_interval(t_interval_margin,
                                                    sig, os, fs, opt)
    
    # init feature container
    cc = {}
    for featset in ["mld_vec", "mld_register", "mld_shape",
                    "mld_shape_en", "mld_corrDist"]:
        if navi[featset]:
            cc[featset] = None

    
    # print("\t\t step vec")
    # spectral spread and flux
    if navi["mld_vec"]:
        spec_summary, spec_frm = mld.mld_vec(os, opt["col_mld_vec"],
                                             None, t_interval,
                                             opt["mld_vec"])
        cc["mld_vec"] = spec_summary

    # print("\t\t step reg")
    # Register (on new F0 column which has not yet been register-normalized)
    if navi["mld_register"]:
        cc["mld_register"] = mld.mld_register_interval(os,
                                                       "f0_preproc",
                                                       t_interval,
                                                       opt["mld_register"])
    # print("\t\t step shape")
    # F0 shape
    if navi["mld_shape"]:
        cc["mld_shape_f0"] = mld.mld_shape_sust(os, opt["col_f0"], t_interval,
                                             opt["mld_shape"])
        

    # energy shape
    if navi["mld_shape_en"]:
        cc["mld_shape_en"] = mld.mld_shape_sust(os, opt["col_energy"],
                                                t_interval,
                                                opt["mld_shape"])

        
    # print("\t\t step corrDist")
    # F0-energy correlation and distance
    if navi["mld_corrDist"]:
        cc["mld_corrDist"] = mld.mld_corrDist(os, [opt["col_f0"],
                                                   opt["col_energy"]],
                                              None, t_interval, tol,
                                              opt["mld_corrDist"])

    # update feat
    return mld_upd(feat, cc, t_interval_margin, t_interval_dt, opt, args, "sustained")


def adjust_interval(t_interval_margin, sig, os, fs, opt):

    ''' adjust analysis interval: truncation, nan replacement '''
    
    # replace NaT, NaN
    if pd.isnull(t_interval_margin[0]):
        t_interval_margin[0] = 0
    if pd.isnull(t_interval_margin[1]):
        t_interval_margin[1] = os["frameTime"].iloc[-1]

    # truncate chunk
    # print("\t\t step trunc")
    if ((sig is not None) and opt["truncate"]):
        # print("\t\t step trunc1")
        sig_i = myl.find_interval(
            np.arange(1, len(sig)+1), t_interval_margin*fs)

        if len(sig_i) == 0:
            t_interval = t_interval_margin
        else:
            opau = cp.deepcopy(opt["pause"])
            opau["ons"] = sig_i[0]
            opau["trunc_only"] = True
            pc = ts.pau_detector(sig[sig_i], opau)
            t_interval = pc["tc"]
    else:
        t_interval = t_interval_margin

    return t_interval, t_interval_margin
        
        
    
def mld_per_chunk(feat, os, sig, fs, t_interval_margin,
                  t_interval_dt, t_ncl, tol, opt, navi, args):
    
    ''' mld set for each chunk in current db-table '''

    t_interval, t_interval_margin = adjust_interval(t_interval_margin,
                                                    sig, os, fs, opt)

    # print("\t\t step ncl")
    # syllable nuclei in interval, syllable rate, sum of silent gap
    i_ncl = myl.find_interval(t_ncl, t_interval)

    # init feature container
    cc = {}
    for featset in navi:
        if navi[featset]:
            cc[featset] = None

    if len(i_ncl) == 0:
        return mld_upd(feat, cc, t_interval_margin, t_interval_dt, opt, args)

    # print("\t\t step hld")
    # syllable rate, gaps
    if navi["hld"]:
        cc["hld"] = myHld(t_interval, t_ncl, i_ncl, opt["hld"])

    # print("\t\t step lld")
    # LLDs around timestamps
    if navi["lld"]:
        cc["lld"] = mld.lld_at_timestamps(
            os, opt["col_lld"], t_ncl[i_ncl], opt["lld"])

    # print("\t\t step vec")
    # spectral spread and flux
    if navi["mld_vec"]:
        spec_summary, spec_frm = mld.mld_vec(os, opt["col_mld_vec"],
                                             t_ncl, t_interval,
                                             opt["mld_vec"])
        cc["mld_vec"] = spec_summary

    # inter-nucleus spread and flux (on same columns as spread, flux)
    if navi["mld_inter"]:
        inter_summary, inter_frm = mld.mld_inter(os, opt["col_mld_vec"],
                                                     t_ncl, t_interval,
                                                     opt["mld_inter"])
        cc["mld_inter"] = inter_summary
        
    # print("\t\t step disp")
    # LSP dispersion
    if navi["mld_disp"]:
        disp_summary, disp_frm = mld.mld_disp(os, opt["col_mld_disp"],
                                              t_ncl, t_interval,
                                              opt["mld_disp"])
        cc["mld_disp"] = disp_summary

    # print("\t\t step reg")
    # Register (on new F0 column which has not yet been register-normalized)
    if navi["mld_register"]:
        cc["mld_register"] = mld.mld_register_interval(os,
                                                       "f0_preproc",
                                                       t_interval,
                                                       opt["mld_register"])

    # print("\t\t step shape")
    # F0 shape
    if navi["mld_shape"]:
        shape_summary, shape_frm = mld.mld_shape(os, opt["col_f0"],
                                                 t_ncl, t_interval,
                                                 opt["mld_shape"])
        cc["mld_shape"] = shape_summary

    # print("\t\t step varRat")
    # F0 variance ratio
    if navi["mld_varRat"]:
        varRat_summary, varRat_frm = mld.mld_varRat(os, opt["col_f0"],
                                                    t_ncl, t_interval,
                                                    opt["mld_varRat"])
        cc["mld_varRat"] = varRat_summary

    # F0 variance ratio by fe
    if navi["mld_varRat_fe"]:
        cc["mld_varRat_fe"] = mld.mld_varRat_fe(os, "f0_preproc", "f0_noninterp",
                                                t_ncl, t_interval, opt["mld_varRat_fe"])

    if navi["mld_voi"]:
        cc["mld_voi"] = mld.mld_voi(os, opt["col_mld_voi"], t_ncl, t_interval,
                                    opt["mld_voi"])

        
    # print("\t\t step corrDist")
    # F0-energy correlation and distance
    if navi["mld_corrDist"]:
        cc["mld_corrDist"] = mld.mld_corrDist(os, [opt["col_f0"],
                                                   opt["col_energy"]],
                                              t_ncl, t_interval, tol,
                                              opt["mld_corrDist"])

    if navi["mld_align"]:
        align_summary, align_frm = mld.mld_align(os, [opt["col_f0"],
                                                      opt["col_energy"]],
                                                 t_ncl, t_interval,
                                                 opt["mld_align"])
        cc["mld_align"] = align_summary
    # print("\t\t step rhy_f0")
    # F0 rhythm
    if navi["mld_rhy_f0"]:
        cc["mld_rhy_f0"] = mld.mld_rhy(
            os, opt["col_f0"], t_ncl, t_interval, opt["mld_rhy_f0"])

    # print("\t\t step rhy_en")
    # energy rhythm
    if navi["mld_rhy_en"]:
        cc["mld_rhy_en"] = mld.mld_rhy(
            os, opt["col_f0"], t_ncl, t_interval, opt["mld_rhy_en"])

    # classical isochrony rhythm
    if navi["mld_rhy_iso"]:
        cc["mld_rhy_iso"] = mld.mld_isochrony(t_ncl, t_interval)

    # print("\t\t step upd")
    # update feat
    return mld_upd(feat, cc, t_interval_margin, t_interval_dt, opt, args)


def myHld(t_interval, t_ncl, i_ncl, opt):
    ''' returns syllable rate, proportion of silent/filled gaps (approximated
    by sylncl further away from each other than opt[gap_length]), and number
    of syllables
    Returns:
    hld dict with keys "sylRate", "gapRel", "nSyl"
    '''

    opt = myl.opt_default(opt, {"gap_margins": False, "gap_length": 0.5})

    ti_ncl = t_ncl[i_ncl]
    if len(i_ncl) < 2 or ti_ncl[-1]-ti_ncl[0] == 0:
        sylRate = 0
    else:
        sylRate = len(i_ncl)/(ti_ncl[-1]-ti_ncl[0])
    # +/- adding chunk margins
    if opt["gap_margins"]:
        gapAbs = (ti_ncl[0]-t_interval[0]) + (t_interval[1]-ti_ncl[-1])
    else:
        gapAbs = 0
    for i in range(len(ti_ncl)-1):
        de = ti_ncl[i+1] - ti_ncl[i]
        if de > opt["gap_length"]:
            gapAbs += de
    gapRel = gapAbs/(t_interval[1]-t_interval[0])
    return {"sylRate": sylRate, "gapRel": gapRel, "nSyl": len(i_ncl)}


def mld_upd(feat, cc, t_interval, t_interval_dt, opt, args, macro=None):
    ''' update mld feature dict '''

    if "nan_replace" in opt:
        nanrep = opt["nan_replace"]
    else:
        nanrep = np.nan

    # file
    feat = mld_init(feat, "file")
    if args["select"]:
        feat["file"].append(args["select"])
    elif args["file_index"]:
        feat["file"].append(args["file_index"])
    elif type(args["wav"]) is str:
        feat["file"].append(args["wav"])
    else:
        feat["file"].append(None)
        
    # time
    for x in ["start", "end"]:
        feat = mld_init(feat, x)

    # #!x
    # t1 = dt.timedelta(seconds=t_interval[0])
    # t2 = dt.timedelta(seconds=t_interval[1])
    # print(t1, t_interval_dt[0])
    # print(t2, t_interval_dt[1])
    # myl.stopgo()

    # feat["start"].append(dt.timedelta(seconds=t_interval[0]))
    # feat["end"].append(dt.timedelta(seconds=t_interval[1]))

    feat["start"].append(t_interval_dt[0])
    feat["end"].append(t_interval_dt[1])

    #feat["start"] = np.append(feat["start"], t_interval_dt[0])
    #feat["end"] = np.append(feat["end"], t_interval_dt[1])

    # hlds
    if "hld" in cc:
        if args["return_nSyl"]:
            yy = ["sylRate", "gapRel", "nSyl"]
        else:
            yy = ["sylRate", "gapRel"]
        for y in yy:
            x = "hld_{}".format(y)
            feat = mld_init(feat, x)
            try:
                feat[x].append(cc["hld"][y])
            except:
                feat[x].append(nanrep)

    # llds
    if "lld" in cc:
        sumStats = summaryStatKeys(opt["lld"])
        for y in opt["col_lld"]:
            for m in sumStats:
                x = "lld_{}_{}".format(y, m)
                feat = mld_init(feat, x)
                try:
                    feat[x].append(cc["lld"][y][m])
                except:
                    feat[x].append(nanrep)

    # spectral spread+flux, moments
    if "mld_vec" in cc:
        # start peak end in connected speech too much dependend on vowels
        # only use for sustained sounds
        if macro == "sustained":
            spe = True
        else:
            spe = False
        sumStats = summaryStatKeys(opt["mld_vec"], start_peak_end=spe)
        for y in ["spread", "flux"]:
            for m in sumStats:
                x = "spec_{}_{}".format(y, m)
                feat = mld_init(feat, x)
                try:
                    feat[x].append(cc["mld_vec"][y][m])
                except:
                    feat[x].append(nanrep)
        # for i in sorted(cc["mld_vec"]["specmom"].keys()):
        #    for j in range(5):
        #        x = "spec_{}_sm{}".format(i,j+1)
        #        feat = mld_init(feat,x)
        #        try:
        #            feat[x].append(cc["mld_vec"]["specmom"][i][j])
        #        except:
        #            feat[x].append(nanrep)

    # inter-nucleus spread and flux
    if "mld_inter" in cc:
        sumStats = summaryStatKeys(opt["mld_inter"], start_peak_end=True)
        for y in ["spread", "flux"]:
            for m in sumStats:
                x = "inter_{}_{}".format(y, m)
                feat = mld_init(feat, x)
                try:
                    feat[x].append(cc["mld_inter"][y][m])
                except:
                    feat[x].append(nanrep)
        
    # shape spread(zero)+flux
    if "mld_shape" in cc and macro is None:
        sumStats = summaryStatKeys(opt["mld_shape"])
        for y in ["spread", "spreadzero", "flux", "mae",
                  "diff", "tmax", "rng"]:
            for m in sumStats:
                x = "shape_{}_{}".format(y, m)
                feat = mld_init(feat, x)
                try:
                    feat[x].append(cc["mld_shape"][y][m])
                except:
                    feat[x].append(nanrep)

        z = ["c3", "c2", "c1", "c0"]
        for y in ["coef"]:
            for m in sumStats:
                for o in range(len(z)):
                    x = "shape_{}_{}".format(z[o], m)
                    feat = mld_init(feat, x)
                    try:
                        feat[x].append(cc["mld_shape"][y][m][o])
                    except:
                        feat[x].append(nanrep)

    # shape spread(zero)+flux for sustained vowels (for f0 and energy)
    if macro == "sustained":
        for u in ["mld_shape_f0", "mld_shape_en"]:
            if u not in cc:
                continue
            infx = re.sub("^mld_", "", u)
            for y in ["d_start", "d_peak", "d_end", "mcr", "tpr", "mae"]:
                x = "{}_{}".format(infx, y)
                feat = mld_init(feat, x)
                try:
                    feat[x].append(cc[u][y])
                except:
                    feat[x].append(nanrep)
            # z = ["c3", "c2", "c1", "c0"]
            # for o in range(len(z)):
            #     x = "shape_{}_{}".format(infx, z[o])
            #     feat = mld_init(feat, x)
            #     try:
            #         feat[x].append(cc[u]["coef"][o])
            #     except:
            #         feat[x].append(nanrep)
            for y in ["dlt", "y"]:
                sumStats = summaryStatKeys(opt["mld_shape"])
                for m in sumStats:
                    x = "{}_{}_{}".format(infx, y, m)
                    feat = mld_init(feat, x)
                    try:
                        feat[x].append(cc[u][y][m])
                    except:
                        feat[x].append(nanrep)
        
    # f0-en per syllable sync
    if "mld_align" in cc:
        sumStats = summaryStatKeys(opt["mld_align"])
        for y in ["spread", "spreadzero", "flux", "corr"]:
            for m in sumStats:
                x = "align_{}_{}".format(y, m)
                feat = mld_init(feat, x)
                try:
                    feat[x].append(cc["mld_align"][y][m])
                except:
                    feat[x].append(nanrep)

    # f0 variance ratio
    if "mld_varRat" in cc:
        sumStats = summaryStatKeys(opt["mld_varRat"])
        for y in ["varRat", "var", "mean"]:
            for m in sumStats:
                x = "f0var_{}_{}".format(y, m)
                feat = mld_init(feat, x)
                try:
                    feat[x].append(cc["mld_varRat"][y][m])
                except:
                    feat[x].append(nanrep)

    # f0 variance ratio by fe
    if "mld_varRat_fe" in cc:
        for y in ["intra_inter", "inter_all"]:
            x = "f0varRat_fe_{}".format(y)
            feat = mld_init(feat, x)
            try:
                feat[x].append(cc["mld_varRat_fe"][y])
            except:
                feat[x].append(nanrep)

    if "mld_voi" in cc:
        sumStats = summaryStatKeys(opt["mld_voi"])
        for y in ["mean", "variance", "skewness", "kurtosis"]:
            for m in sumStats:
                x = "voi_{}_{}".format(y, m)
                feat = mld_init(feat, x)
                try:
                    feat[x].append(cc["mld_voi"][y][m])
                except:
                    feat[x].append(nanrep)
                    
    # lsp dispersion
    if "mld_disp" in cc:
        sumStats = summaryStatKeys(opt["mld_disp"])
        for y in ["disp", "dispMin"]:
            for m in sumStats:
                x = "lsp_{}_{}".format(y, m)
                feat = mld_init(feat, x)
                try:
                    feat[x].append(cc["mld_disp"][y][m])
                except:
                    feat[x].append(nanrep)

    # f0-en global sync and forecastability
    if "mld_corrDist" in cc:
        for y in ["spea", "rms"]:
            x = "f0en_{}".format(y)
            feat = mld_init(feat, x)
            try:
                feat[x].append(cc["mld_corrDist"][y])
            except:
                feat[x].append(nanrep)
        # for y in ["sample_entropy", "granger_causality"]:
        for y in ["granger_causality"]:
            for i in myl.idx(opt["col_corrDist"]):
                x = "f0en_{}_{}".format(y, opt["col_corrDist"][i])
                feat = mld_init(feat, x)
                try:
                    feat[x].append(cc["mld_corrDist"][y][i])
                except:
                    feat[x].append(nanrep)

    # register
    if "mld_register" in cc:
        for lev in ["bl", "ml", "rng", "tl"]:
            for y in ["mean", "intercept", "rate"]:
                x = "reg_{}_{}".format(lev, y)
                feat = mld_init(feat, x)
                try:
                    feat[x].append(cc["mld_register"]["{}_{}".format(lev, y)])
                except:
                    feat[x].append(nanrep)
        for lev in ["rng_bl"]:
            x = "reg_{}".format(lev)
            feat = mld_init(feat, x)
            try:
                feat[x].append(cc["mld_register"][lev])
            except:
                feat[x].append(nanrep)
                    
    # f0 and energy rhythm
    for dom in ["f0", "en"]:
        z = "mld_rhy_{}".format(dom)
        if z not in cc:
            continue
        for y in ["mae", "prop", "sm1", "sm2", "sm3", "sm4"]:
            x = "rhy_{}_{}".format(dom, y)
            feat = mld_init(feat, x)
            try:
                feat[x].append(cc[z][y])
            except:
                feat[x].append(nanrep)

    # isochrony
    if "mld_rhy_iso" in cc:
        for y in ["pvi", "varco"]:
            x = "rhy_iso_{}".format(y)
            feat = mld_init(feat, x)
            try:
                feat[x].append(cc["mld_rhy_iso"][y])
            except:
                feat[x].append(nanrep)
                
    return feat


def summaryStatKeys(o, start_peak_end=True):
    ''' returns set of keys to access summaryStat dict '''

    sumStat = {"robust": set(["median", "iqr", "q_var", "q_skewness",
                              "q_kurtosis", "q_range5", "slope", "diff"]),
               "standard": set(["mean", "var", "skewness", "kurtosis",
                                "slope", "diff"]),
               "misc": set(["iqr1-2", "iqr1-3", "iqr2-3", "percentile1",
                            "percentile99", "quartile1", "quartile2",
                            "quartile3", "rqmean", "slope", "diff"])}
    sumStat["extended"] = (sumStat["robust"] | sumStat["standard"] |
                           sumStat["misc"])
    sumStat["standard_extended"] = (sumStat["standard"] | sumStat["misc"])
    sumStat["robust_extended"] = (sumStat["robust"] | sumStat["misc"])

    if start_peak_end:
        for x in sumStat:
            sumStat[x] = (sumStat[x] | set(["start", "peak", "end"]))

    # sorting
    for x in sumStat:
        sumStat[x] = sorted(sumStat[x])
    
    return sumStat[o["summaryStat"]]


def mld_init(feat, x):
    ''' add key to mld feature dict '''

    if x not in feat:
        #if x in ["start", "end"]:
        #    feat[x] = np.array([], dtype='timedelta64[ns]')
        #else:
        feat[x] = []
    return feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mid level descriptors."
                                     "Generates 1 feature vector"
                                     "per VAD segment")
    parser.add_argument('-wav', '--wav',
                        help='mono audio file (.wav) or 1-dim np.array',
                        required=True)
    parser.add_argument('-os', '--opensmile',
                        help='openSmile LLD table (.csv)',
                        required=True)
    parser.add_argument('-c', '--config', help='configuration file (.json)',
                        required=True)
    parser.add_argument('-o', '--output',
                        help="output file; Depending on output_format"
                        "argument either a pickle file or a"
                        ";-separated csv file with header containing"
                        "the MLDs. One line per VAD segment.",
                        required=False)
    parser.add_argument('-oform', '--output_format',
                        help='output format; csv or pickle.',
                        default="csv", required=False)
    parser.add_argument('-seg', '--segment',
                        help="name of a file containing a table"
                        "in unified format with columns"
                        "\"start\" and \"end\"."
                        "This could be an annotation table or the output"
                        "of the auvad module. In case this table contains"
                        "more than one file, the \"select\" argument needs"
                        "to be specified to select the corresponding subset.",
                        required=False)
    parser.add_argument('-sel', '--select',
                        help="name of audio file (as stored in table index)"
                        "for corresponding subset selection in segment"
                        "dataframe",
                        required=False)
    parser.add_argument('-f', '--file_index',
                        help="name of audio file. Used for output dataframe"
                        "indexing if argument select is not provided.",
                        required=False)
    parser.add_argument('-sr', '--samplerate',
                        help="audio sample rate. Needed if audio -wav"
                        "is provided as np.array.",
                        required=False)
    parser.add_argument('-noseg',
                        help="if this flag is set, no chunking into"
                        "VAD segments is carried out but just one row"
                        "of MLDs is calculated for the entire file",
                        required=False, action='store_true')
    parser.add_argument('-v', help="if this flag is set, warnings"
                        "are verbosed.",
                        required=False, action='store_true')
    args = vars(parser.parse_args())
    mld_wrapper(args)
