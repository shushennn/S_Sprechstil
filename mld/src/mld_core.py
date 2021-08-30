#!/usr/bin/env python3

import itertools
import numpy as np
import copy as cp
import re
import sys
import sklearn.preprocessing as sp
import scipy as sc
import myUtilHL as myl
import sklearn.metrics as sm
import scipy.stats as sst
import dct_wrapper as dct
#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

# collection of LLD (at predefined landmarks only), MLD, and HLD extractors
# remarks:
#    if plotting functions *_plot() need to be included for illustrations
#    the 2 import lines import matplotlib, matplotlib.use('agg') need to be
#    commented out in mld.py, preproc.py, and myUtilHL.py !


def lld_at_timestamps(os, col, t, opt={}):
    r''' return openSmile feature statistics for selected time stamps only
    Args:
    os: (dict) openSmile output
    col: (list) of openSmile columns to be processed
    t: (list) 1- or 2-dim of timestamps, resp intervals for which feature
      statistics is to be calculated
    opt: (dict)
      win: (float) length (in sec) of symmetric window centered
           on time stamps in t (default value for all LLDs)
           If 0: no window, values in t must match values in os["frameTime"]
           (2-element list of floats) left and right half of assymmetric
           windows. Only relevant if t is 1-dim.
      myLLDcolumnName: (float or list); window size for specific LLD
    Returns:
      summaryStat: (dict)
        myColumnName: (dict)
        'mean', 'var', 'skewness', 'kurtosis',
        'median', 'iqr', 'q_skewness', 'q_kurtosis': (floats)
    '''
    opt = myl.opt_default(opt, {"win": 0.03, "nanrep": 0})

    if type(col) is str:
        col = [col]

    # indices at which to collect values
    ti = {}
    tf = np.asarray(os["frameTime"])

    # over all required window sizes
    # ti[myCol] = idx; myCol is "win" per default
    for x in opt:
        if x == "win" or x in os:
            ti[x] = lld_win(t, tf, opt[x])

    # summary statistics
    typ = get_statTyp(opt)
    
    summaryStat = {}
    for c in col:
        if c not in os:
            continue
        # indices over which to aggregate
        if c in ti:
            cti = ti[c]
        else:
            cti = ti["win"]
        if len(cti) == 0:
            y = myl.ea()
        else:
            y = np.asarray(os[c])[cti]
        summaryStat[c] = myl.summary_stats(y, typ, nanrep=opt["nanrep"])

    return summaryStat


def lld_win(t, tf, w):
    ti = []
    for x in t:
        # intervals
        if myl.listType(x):
            win = x
        # time points without window
        elif w == 0:
            win = x
        # timepoints with window
        else:
            win = [x-w/2, x+w/2]
        if myl.listType(win):
            wi = myl.find_interval(tf, win)
        else:
            wi = myl.find(tf, "==", x)
        if len(wi) > 0:
            ti.extend(wi)
    return np.unique(ti)


def hld_pauseRate(t_ipu, t_ncl):
    r''' high level features pause ratio and syllable rate.
    Pauses before the first and after the last interpausal unit are not
    included.
    It is assumed that all t_ncl are within one of the intervals in t_ipu.
    Args:
    t_ipu: (2-dim np.array) [[start end] ...] of interpausal units (in sec)
    t_ncl: (1-dim np.array) of syllable nuclei midpoints (in sec)
    Returns:
    y (dict):
      pause_ratio: (float)  time_pause/time_total
      syl_rate: (float) n_syllables/time_ipu_total
      ipu_dur: (float) mean IPU duration
      pau_dur: (float) mean pause duration
      pau_ipu_ratio: (float) pause duration divided by IPU duration
      pau_rate: (float) number of pauses divided by IPU duration
    '''
    time_total = t_ipu[-1, 1] - t_ipu[0, 0]
    n_syl = len(t_ncl)
    pau_dur, ipu_dur = myl.ea(), myl.ea()
    for i in myl.idx(t_ipu):
        ipu_dur = np.append(ipu_dur, t_ipu[i, 1]-t_ipu[i, 0])
        if i > 0:
            pau_dur = np.append(pau_dur, t_ipu[i, 0]-t_ipu[i-1, 1])
    if len(pau_dur) == 0:
        pau_dur = 0
        n_pau = 0
        pau_dur_sum = 0
    else:
        pau_dur_sum = np.sum(pau_dur)
        n_pau = len(pau_dur)
        pau_dur = np.mean(pau_dur)
    if len(ipu_dur) == 0:
        ipu_dur = 0
        ipu_dur_sum = 0
    else:
        ipu_dur_sum = np.sum(ipu_dur)
        ipu_dur = np.mean(ipu_dur)
    y = {"pause_ratio": pau_dur_sum/time_total,
         "syl_rate": n_syl/ipu_dur_sum,
         "ipu_dur": ipu_dur,
         "pau_dur": pau_dur,
         "pau_ipu_ratio": pau_dur_sum/ipu_dur_sum,
         "pau_rate": n_pau/time_total}
    return y


def mld_align(os, col=None, t=None, t_interval=None, opt={}):
    r''' spread and flux of feature pearson correlation similar to mld_corrDist,
    but here corr is calculated separately for each time stamp in t
    (as opposed to be calculated once over entire interval), and spread
    and flux are calculated from these values.
    Args:
    os: (dict) openSmile output
    col: (2-element list) name of openSmile columns to be further processed
      If None: "F0final_sma" and "pcm_LOGenergy_sma" are taken. Further than
      2 elements are ignored. Especially F0 SHOULD already be preprocessed by
      preproc.preproc_f0()
    t: (array-like) of time stamps (e.g. syllable nuclei) around which
      the requested features are to be collected
      If None: each sample index (within t_interval) is processed.
    t_interval: (2 element array) [start end] of interval to be
      examined. If None: entire file is taken
    opt: (dict)
      win: (float) length (in sec) of symmetric window centered on
        time stamps in t.
        If 0: no window, values in t must match values in os["frameTime"]
          Only relevant if t is not None
      register: "ml", None; subtract midline from f0 contour (!! assumed
          to be 1st element in col !!)
    Returns:
    summaryStat: (dict, None)
       'corr': (dict) summary stat over correlation values
       'spread': (dict) mean abs distance of correlation from median corr value
       'spreadzero' (dict) mean abs distance from 0
       'flux': (dict)  abs distance between subsequent frames
    perFrame: (dict, None)
       't': (np.array) time stamps
       'spread': per frame distances underlying spread summary
       'spreadzero': per frame distances underlying spreadzero summary
       'flux': per frame distances underlying flux summary
            (horizontal left extrapolation)
    '''

    opt = myl.opt_default(opt, {"win": 0.2,
                                "register": None,
                                "nanrep": 0})
    if col is None:
        col = ["F0final_sma", "pcm_LOGenergy_sma"]

    tf, a, b = np.asarray(os["frameTime"]), np.asarray(
        os[col[0]]), np.asarray(os[col[1]])
    if t is None:
        t = cp.copy(tf)
    if t_interval is None:
        t_interval = [tf[0], tf[-1]]

    # restrict to analysis interval
    ti = myl.find_interval(t, t_interval)

    if len(ti) == 0:
        return None, None

    t = np.asarray(t)[ti]

    # limit time and y to analyzed interval
    tfi = myl.find_interval(tf, t_interval)
    tf = tf[tfi]
    a = a[tfi]
    b = b[tfi]

    # subtract register (only meaningful if a contains f0 values)
    if opt["register"] == "ml":
        cr = myl.myPolyfit(tfi, a, 1)
        yreg = np.polyval(cr, tfi)
        a -= yreg

    # collect f0, energy correlations
    # over time stamps (e.g. syllable nuclei)
    corrs = myl.ea()
    for x in t:
        # window indices
        win = [x-opt["win"]/2, x+opt["win"]/2]
        wi = myl.find_interval(tf, win)
        # correlation
        if len(wi) == 0 or np.max(a[wi]) == np.min(a[wi]) or np.max(b[wi]) == np.min(b[wi]):
            corrs = np.append(corrs, 0)
        else:
            pe = sst.pearsonr(a[wi], b[wi])
            corrs = np.append(corrs, pe[0])

    # spread and flux per frame
    spreadX = np.abs(corrs-np.mean(corrs))
    spreadzeroX = np.abs(corrs)
    fluxX = np.diff(corrs)

    # summary statistics
    typ = get_statTyp(opt)
    summary = {"corr": myl.summary_stats(corrs, typ, nanrep=opt["nanrep"]),
               "spread": myl.summary_stats(spreadX, typ, nanrep=opt["nanrep"],
                                           exceptVal=0),
               "spreadzero": myl.summary_stats(spreadzeroX, typ,
                                               nanrep=opt["nanrep"],
                                               exceptVal=0),
               "flux": myl.summary_stats(fluxX, typ, nanrep=opt["nanrep"],
                                         exceptVal=0)}

    # per frame
    # horizontal extrapolation of fluxX
    if len(fluxX) == 0:
        fluxX = np.append(fluxX, 0)
    else:
        fluxX = np.insert(fluxX, 0, fluxX[0])
    perFrame = {"corr": corrs,
                "spread": spreadX,
                "spreadzero": spreadzeroX,
                "flux": fluxX}

    return summary, perFrame


def mld_corrDist(os, col=None, t=None, t_interval=None, tol=None, opt={}):
    r''' feature correlation and rmsd
    Args:
    os: (dict) openSmile output
    col: (2-element list) name of openSmile columns to be further processed
      If None: "F0final_sma" and "pcm_LOGenergy_sma" are taken. Further than
      2 elements are ignored. Especially F0 SHOULD already be preprocessed by
      preproc.preproc_f0()
    t: (array-like) of time stamps (e.g. syllable nuclei) around which
      the requested features are to be collected
      If None: each sample index (within t_interval) is processed.
    t_interval: (2 element array) [start end] of interval to be
      examined. If None: entire file is taken
    tol: (dict or None). If dict: keys are items in col, values are
      max distances for sample entropy calculation
    opt: (dict)
      win: (float) length (in sec) of symmetric window centered on
        time stamps in t.
        If 0: no window, values in t must match values in os["frameTime"]
          Only relevant if t is not None
      register: "ml", None; subtract midline from f0 contour (!! assumed
          to be 1st element in col !!)
    Returns:
      r (dict)
       .pear
       .spea
       .rms
       .mae
       .sample_entropy: CURRENTLY NOT OUTPUTTED since too time-consuming
          to calculate!
          2-element list of within forecastability of os[col]s
       .granger_causality: 2-element list of cross forecastability p-values
          (1st: does os[col][0] predict os[col][1]?, 2nd: vice versa;
           if p<0.05 value of y is better forecasted by both x and y than
           by y alone)
    Remarks: if t_interval is None, but register normalization is wanted,
       it must be done beforehand in entire file (separately for each IPU)
       by preproc.subtract_register(),
       !! and opt["register"] must be set to None !!
    '''
    opt = myl.opt_default(opt, {"win": 0.2,
                                "register": None,
                                "nanrep": 0})
    if col is None:
        col = ["F0final_sma", "pcm_LOGenergy_sma"]

    if tol is None:
        tol = {}
        for c in col:
            tol[c] = None

    # restrict to analysis interval
    # ti: row indices in os
    # tt: corresponding time values
    ti, tt = analysis_interval(os, t, t_interval, opt)

    if len(ti) == 0:
        return None

    a = np.asarray(os[col[0]])[ti]
    b = np.asarray(os[col[1]])[ti]

    if len(a) < 2:
        return None
    
    # subtract register (from f0 only)
    if opt["register"] == "ml":
        ai = myl.idx(a)
        cr = myl.myPolyfit(ai, a, 1)
        areg = np.polyval(cr, ai)
        a -= areg

    # centering, scaling
    a_nrm = my_centerScale(a)
    b_nrm = my_centerScale(b)
    
    # within forecastability: sample entropy
    # not in use since very time consuming (calculation of C) !
    # sampEntrop = []
    # sampEntrop.append(sample_entropy(a,2,tol[col[0]]))
    # sampEntrop.append(sample_entropy(b,2,tol[col[1]]))

    # cross forecastability: Granger causality
    # append p values
    max_lag, myTest = 1, "params_ftest"
    gc = []
    # 1. a causes b; 2. b causes a
    gc.append(granger_causality(a, b, max_lag, myTest))
    gc.append(granger_causality(b, a, max_lag, myTest))

    # correlation, distance
    if np.max(a_nrm) == np.min(a_nrm) or np.max(b_nrm) == np.min(b_nrm):
        sp_v, sp_p, pe = 0, 1, [0, 0]
    else:
        sp_v, sp_p = sst.spearmanr(a_nrm, b_nrm)
        pe = sst.pearsonr(a_nrm, b_nrm)

    r = {"spea": sp_v,
         "pear": pe[0],
         "mae": abs(sm.mean_absolute_error(a_nrm, b_nrm)),
         "rms": abs(sm.mean_squared_error(a_nrm, b_nrm)),
         "granger_causality": gc}

    # dismissed: "sample_entropy": sampEntrop,

    return r


def granger_causality(x, y, max_lag=1, test="params_ftest"):
    ''' cf https://en.wikipedia.org/wiki/Granger_causality
    We say that a variable X that evolves over time Granger-causes
    another evolving variable Y if predictions of the value of Y based
    on its own past values and on the past values of X are better than
    predictions of Y based only on its own past values.
    Args:
    x, y: (arrays)
    max_lag: (int)
    test: (string) p value according to which test to be returned
    Returns: p value (if <0.05 y is better predicted by x and y than
         by y alone)
    underlying dict:
    keys are the number of lags. For each lag the values are a tuple,
    with the first element a dictionary with teststatistic, pvalues,
    degrees of freedom, the second element are the OLS estimation results
    for the restricted model, the unrestricted model and the restriction
    (contrast) matrix for the parameter f_test
    '''

    max_lag = np.min([max_lag, len(x)])
    if max_lag == 0:
        return np.nan

    yx = np.column_stack((y, x))

    if np.min(x) == np.max(x) or np.min(y) == np.max(y):
        return 1

    try:
        gc = grangercausalitytests(yx, maxlag=max_lag, verbose=False)
        return gc[max_lag][0][test][1]
    except:
        return 1


def sample_entropy(y, m=2, r=None):
    """Compute Sample entropy
    cf https://www.machinelearningplus.com/time-series/...
    time-series-analysis-python/
    https://en.wikipedia.org/wiki/Sample_entropy
    Args:
    y: (np.array) signal
    m: (int) embedding dimension (subsequence length)
    r: (float) tolerance; max distance; if possible calculate over
       entire file. If not provided, it is calculated as 0.2*sd(y)
    Returns:
    e: (float) sample entropy
    """

    if len(y) == 0:
        return 0

    if r is None:
        r = 0.2*np.std(y)

    nom = se_phi(y, m+1, r)
    denom = se_phi(y, m, r)

    if denom == 0:
        return np.nan

    return -np.log(nom/denom)


def se_maxdist(x_i, x_j):
    return max([abs(ua - va) for ua, va in zip(x_i, x_j)])


def se_phi(y, m, r):
    N = len(y)
    x = [[y[j] for j in range(i, i+m-1+1)] for i in range(N-m+1)]
    C = [len([1 for j in range(len(x)) if i != j and
              se_maxdist(x[i], x[j]) <= r]) for i in range(len(x))]
    return sum(C)


def my_centerScale(x):
    y = np.asarray([x]).T
    obj = sp.StandardScaler().fit(y)
    y = obj.transform(y)
    y = y.T
    return y[0]


def mld_disp(os, col=None, t=None, t_interval=None, opt={}):
    ''' feature vector dispersion mean and variance
    Args:
    os: (dict) openSmile output
    col: (list) name of openSmile columns to be further processed
      If None: all non-delta lspFreq_sma are taken (using re)
    t: (array-like) of time stamps (e.g. syllable nuclei) around which
      the requested features are to be collected
      If None: each sample index (within t_interval) is processed.
    t_interval: (2 element array) [start end] of interval to be
      examined. If None: entire file is taken
    opt: (dict)
      win: (float) length (in sec) of symmetric window centered on
        time stamps in t.
        If 0: no window, values in t must match values in os["frameTime"]
          Only relevant if t is not None
    Returns:
    summaryStat: (dict, None)
       'disp': (dict)
          'mean', 'median', 'var', 'skewness', 'kurtosis', 'iqr':
           (floats) of frequency dispersion sd
           (i.e. sd of pairwise distances of adjacent frequencies)
       'dispMax', 'dispMin': (dicts) same keys as disp
          summary stats of max and min LSP dispersion, respectively
    perFrame: (dict, None)
       't': (np.array) time stamps
       'disp': per frame dispersion sd underlying summaryStat
       'dispMax'
       'dispMin'
    '''

    opt = myl.opt_default(opt, {"win": 0.05, "nanrep": 0})

    # column names
    # default lspFreq columns
    if col is None:
        col, i_max = [], -1
        for i in range(0, 20):
            col.append("")
        for cn in sorted(os.keys()):
            if re.search("_de", cn):
                continue
            m = re.search(r"lspFreq.+?\[(?P<idx>(.+?))\]", cn)
            if m is None:
                continue
            i = int(m.group("idx"))
            col[i] = cn
            i_max = max(i, i_max)
        col = col[0:i_max+1]

    # restrict to analysis interval
    # ti: row indices in os
    # tt: corresponding time values
    ti, tt = analysis_interval(os, t, t_interval, opt)

    if len(ti) == 0:
        return None, None

    # data subset
    d = myl.ea()
    # over selected columns and indices
    for c in col:
        if c not in os:
            print("{}: table does not contain this column. Skipped".format(c))
        d = myl.push(d, np.asarray(os[c])[ti])
    d = d.T

    # dispersion per frame
    dd = np.diff(d, axis=1)
    dispX = np.std(dd, axis=1)
    dispMaxX = np.max(dd, axis=1)
    dispMinX = np.min(dd, axis=1)

    # summary statistics
    typ = get_statTyp(opt)
    summary = {"disp": myl.summary_stats(dispX, typ, nanrep=opt["nanrep"]),
               "dispMax": myl.summary_stats(dispMaxX, typ,
                                            nanrep=opt["nanrep"]),
               "dispMin": myl.summary_stats(dispMinX, typ,
                                            nanrep=opt["nanrep"])}

    # per frame
    perFrame = {"t": tt,
                "disp": dispX,
                "dispMax": dispMaxX,
                "dispMin": dispMinX}

    return summary, perFrame


def analysis_interval(os, t, t_interval, opt, dim=1):
    r''' restrict to analysis interval t_interval,
    and return time windows around stamps in t only
    Args:
      os: (dict) openSmile output
      t: (np.array or None) time stamps
      t_interval: (2-element np.array or None) on- and offset of interval
      opt: (dict)
      dim: return 1 or 2-dim array
    Out:
      ti: (np-array) row indices in os
      tt: (np-array) with corresponding time values
    '''
    tf = np.asarray(os["frameTime"])
    if t_interval is None:
        t_interval = [tf[0], tf[-1]]
    tfi = myl.find_interval(tf, t_interval)
    tf = tf[tfi]
    # time stamps within analysis interval
    if t is None:
        if dim == 2:
            ti = [tfi]
            tt = [tf]
        else:
            ti = tfi
            tt = tf
    else:
        t = t[myl.find_interval(t, t_interval)]
        ti = []
        tt = []
        for x in t:
            if opt["win"] == 0:
                wi = myl.find(tf, "==", x)
            else:
                win = [x-opt["win"]/2, x+opt["win"]/2]
                wi = myl.find_interval(tf, win)
            if len(wi) > 0:
                if dim == 2:
                    ti.append(wi)
                    tt.append(tf[wi])
                else:
                    ti.extend(wi)
                    tt.extend(tf[wi])
        if dim == 1:
            ti = np.unique(ti)
            tt = np.unique(tt)

    return ti, tt

def mld_inter(os, col=None, t=None, t_interval=None, opt={}):
    r''' feature vector spread and flux at syllable transitions.
    For syllable complexity and articulatory precision of consonants.
    Higher spec flux values indicate higher complexity and precision.
    Calculates mean spread and flux in each inter-nucleus interval.
    Then calculates summary statistics over all inter-nucleus intervals.
    Args:
    os: (dict) openSmile output
    col: (list) name of openSmile columns to be further processed
      If None: all non-delta MFCCs >0 are taken (using re)
    t: (array-like) of time stamps (syllable nuclei) between which
      the requested features are to be collected
      If None: each sample index (within t_interval) is processed.
    t_interval: (2 element array) [start end] of interval to be
      examined. If None: entire file is taken
    opt: (dict)
      win: (float) length (in sec) of symmetric window centered on
        time stamps in t.
        If 0: no window, values in t must match values in os["frameTime"]
        Only relevant if t is not None
      dist: (str); {"mahal", "mahal_distrib", <"euclidean">, "canberra"}
        how to calculate spread. "mahal": mahalanobis distance of each
        vector to centroid. "mahal_distrib": mahalanobis distance of
        each vector to vector distribution. "euclidean": euclidean distance
        of each vector to centroid. mahal* is NOT recommended, since
        it normalises away distribution variance differences!
      scale: (boolean); if True, coef matrix is scaled to mean and sd.
    Returns:
    summaryStat: (dict, None)
       'spread': (dict) user-defined distance from centroid
       'spreadzero' (dict) Euclidean distance from zero-vector
       'flux': (dict) Euclidean distance between subsequent frames
    perFrame: (dict, None)
       't': (np.array) time stamps
       'spread': per frame distances underlying spread summary
       'spreadzero': per frame distances underlying spreadzero summary
       'flux': per frame distances underlying flux summary
            (horizontal left extrapolation)
    '''
    
    opt = myl.opt_default(opt, {"win": 0.1,
                                "dist": "euclidean",
                                "scale": False,
                                "nanrep": 0})
    
    # column names
    # default mfcc columns
    if col is None:
        col, i_max = [], -1
        for i in range(0, 20):
            col.append("")
        for cn in sorted(os.keys()):
            if re.search(r"(_de|\[0\])", cn):
                continue
            m = re.search(r"mfcc.+?\[(?P<idx>(.+?))\]", cn)
            if m is None:
                continue
            i = int(m.group("idx"))-1
            col[i] = cn
            i_max = max(i, i_max)
        col = col[0:i_max+1]

    # inter nucleus time stamps
    if ((t is not None) and len(t)>1):
        td = t[1:len(t)] - np.diff(t)/2
    else:
        td = None

        
    if td is None:
        return None, None
    
    # restrict to analysis interval + windowing
    # ti: 2-dim list of array of row indices in os
    # tt: 2-dim list of corresponding time values
    ti, tt = analysis_interval(os, t, t_interval, opt, dim=2)
                
    if len(ti) == 0:
        return None, None

    # summary statistics
    typ = get_statTyp(opt)

    spreadX, spreadzeroX, fluxX = myl.ea(), myl.ea(), myl.ea()

    # over inter-nucleus intervals
    for ii in ti:
        # data subset
        d = myl.ea()
        # over selected columns and indices
        for c in col:
            if c not in os:
                print("{}: table does not contain this column. Skipped".format(c))
            d = myl.push(d, np.asarray(os[c])[ii])
        d = d.T

        # spread and flux per frame
        spreadX_ii, spreadzeroX_ii, fluxX_ii = spread_flux(d, opt["dist"], opt["scale"])

        if len(spreadX_ii) == 0:
            continue

        spreadX_ii = myl.nan_repl(spreadX_ii, opt["nanrep"])
        spreadzeroX_ii = myl.nan_repl(spreadzeroX_ii, opt["nanrep"])
        fluxX_ii = myl.nan_repl(fluxX_ii, opt["nanrep"])
        
        # append mean spread and flux within inter-nucleus interval
        spreadX = np.append(spreadX, np.mean(spreadX_ii))
        spreadzeroX = np.append(spreadzeroX, np.mean(spreadzeroX_ii))
        fluxX = np.append(fluxX, np.mean(fluxX_ii))

    summary = {"spread": myl.summary_stats(spreadX, typ, nanrep=opt["nanrep"]),
               "spreadzero":  myl.summary_stats(spreadzeroX, typ,
                                                nanrep=opt["nanrep"]),
               "flux": myl.summary_stats(fluxX, typ, nanrep=opt["nanrep"])}

    # per frame
    # horizontal extrapolation of fluxX
    if len(fluxX) > 0:
        fluxX = np.insert(fluxX, 0, fluxX[0])
    perFrame = {"t": tt,
                "spread": spreadX,
                "spreadzero": spreadzeroX,
                "flux": fluxX}
    
    return summary, perFrame
    


def mld_vec(os, col=None, t=None, t_interval=None, opt={}):
    r''' feature vector spread and flux. For vowel space size estimation.
    Args:
    os: (dict) openSmile output
    col: (list) name of openSmile columns to be further processed
      If None: all non-delta MFCCs >0 are taken (using re)
    t: (array-like) of time stamps (e.g. syllable nuclei) around which
      the requested features are to be collected
      If None: each sample index (within t_interval) is processed.
    t_interval: (2 element array) [start end] of interval to be
      examined. If None: entire file is taken
    opt: (dict)
      win: (float) length (in sec) of symmetric window centered on
        time stamps in t.
        If 0: no window, values in t must match values in os["frameTime"]
        Only relevant if t is not None
      dist: (str); {"mahal", "mahal_distrib", <"euclidean">, "canberra"}
        how to calculate spread. "mahal": mahalanobis distance of each
        vector to centroid. "mahal_distrib": mahalanobis distance of
        each vector to vector distribution. "euclidean": euclidean distance
        of each vector to centroid. mahal* is NOT recommended, since
        it normalises away distribution variance differences!
      scale: (boolean); if True, coef matrix is scaled to mean and sd.
    Returns:
    summaryStat: (dict, None)
       'spread': (dict) user-defined distance from centroid
       'spreadzero' (dict) Euclidean distance from zero-vector
       'flux': (dict) Euclidean distance between subsequent frames
    perFrame: (dict, None)
       't': (np.array) time stamps
       'spread': per frame distances underlying spread summary
       'spreadzero': per frame distances underlying spreadzero summary
       'flux': per frame distances underlying flux summary
            (horizontal left extrapolation)
    '''

    opt = myl.opt_default(opt, {"win": 0.05,
                                "dist": "euclidean",
                                "scale": False,
                                "nanrep": 0})

    # column names
    # default mfcc columns
    if col is None:
        col, i_max = [], -1
        for i in range(0, 20):
            col.append("")
        for cn in sorted(os.keys()):
            if re.search(r"(_de|\[0\])", cn):
                continue
            m = re.search(r"mfcc.+?\[(?P<idx>(.+?))\]", cn)
            if m is None:
                continue
            i = int(m.group("idx"))-1
            col[i] = cn
            i_max = max(i, i_max)
        col = col[0:i_max+1]
        
    # restrict to analysis interval
    # ti: row indices in os
    # tt: corresponding time values
    ti, tt = analysis_interval(os, t, t_interval, opt)
    
    if len(ti) == 0:
        return None, None

    # data subset
    d = myl.ea()
    # over selected columns and indices
    for c in col:
        if c not in os:
            print("{}: table does not contain this column. Skipped".format(c))
        d = myl.push(d, np.asarray(os[c])[ti])
    d = d.T

    # spread and flux per frame
    spreadX, spreadzeroX, fluxX = spread_flux(d, opt["dist"], opt["scale"])

    if len(spreadX) == 0:
        return None, None

    # summary statistics
    typ = get_statTyp(opt)

    summary = {"spread": myl.summary_stats(spreadX, typ, nanrep=opt["nanrep"]),
               "spreadzero":  myl.summary_stats(spreadzeroX, typ,
                                                nanrep=opt["nanrep"]),
               "flux": myl.summary_stats(fluxX, typ, nanrep=opt["nanrep"])}

    # per frame
    # horizontal extrapolation of fluxX
    if len(fluxX) > 0:
        fluxX = np.insert(fluxX, 0, fluxX[0])
    perFrame = {"t": tt,
                "spread": spreadX,
                "spreadzero": spreadzeroX,
                "flux": fluxX}

    return summary, perFrame


def mld_register(os, col="F0final_sma", t_interval=None, opt={}):
    r''' base-, mid-, and topline fitting and derived features.
    see mld_register_interval()
    Args:
    os: (dict) openSmile output
    col: (string) name of column to be further processed
    t_interval: (2-dim array) [[start end],...] of intervals to be
      examined. If None: entire file is taken
    opt: (dict)
      "min_dur": (float) minimum duration of interval so that it contributes
            to summary statistics
      all other keys: see mld_register_interval()
    Returns:
       reg_summary: summary statistics of values in reg
       reg: (dict of np.arrays) same keys as returned by
          mld_register_interval(). Values collected for all intervals in
                t_intervals longer than opt["min_dur"]. Plus:
          ["index"]: list of indices of those intervals in t_interval that
                were analyized
    '''

    opt = myl.opt_default(opt, {"win": 0.1,
                                "bl": 5,
                                "tl": 95,
                                "rng": [0, 1],
                                "plot": False,
                                "min_dur": 0.5,
                                "nanrep": 0})

    if t_interval is None:
        t_interval = [[os["frameTime"][0], os["frameTime"][-1]]]
    elif not myl.listType(t_interval[0]):
        t_interval = [t_interval]

    reg = {"index": []}
    for i in range(len(t_interval)):
        t = t_interval[i, ]
        if t[1]-t[0] < opt["min_dur"]:
            continue
        regi = mld_register_interval(os, col, t, opt)
        for x in regi:
            if x not in reg:
                reg[x] = np.array([])
            reg[x] = np.append(reg[x], regi[x])
        reg["index"].append(i)

    typ = get_statTyp(opt)
    reg_summary = {}
    for x in reg:
        if x == "index":
            continue
        reg_summary[x] = myl.summary_stats(reg[x], typ, nanrep=opt["nanrep"])

    return reg_summary, reg


def mld_register_interval(os, col="F0final_sma", t_interval=None, opt={}):
    r''' base-, mid-, and topline, as well as range
    (pointwise distance between base- and topline).
    Polynomial fits.
    Args:
    os: (dict) openSmile output
    col: (string) name of column to be further processed
    t_interval: (2 element array) [start end] of interval to be
      examined. If None: entire file is taken
    opt: (dict)
       "win": (float) sliding window length within which
              to calculate percentiles
       "bl": (int) percentile for baseline fitting
       "tl": (int) percentile for topline fitting
       "rng": (2-element list) of start and end of normalized time
            for polyfit
       "plot": (boolean) if True base, mid, and toplinefit is plotted
    Returns:
      reg: (dict)
        {bl|ml|tl|rng}_{slope|intercept|mean|rate}
          base/mid/topline/range line slope, intercept, arit mean and rate
             (slope vs rate: the latter preserves duration information)
        {tl_ml|tl_bl|ml_bl}_cross_{t|f0}: time (normalized) and f0 value
             for top/mid, top/base, mid/baseline intersection
      or None, if segment is shorter than 2 samples
    '''

    opt = myl.opt_default(opt, {"win": 0.1,
                                "bl": 5,
                                "tl": 95,
                                "rng": [0, 1],
                                "plot": False,
                                "nanrep": 0})
    
    # time points, sample rate
    tf = os["frameTime"]
    if len(tf) < 2:
        return None
    sr = int(1/tf[1]-tf[0])

    # restrict to interval
    if t_interval is None:
        ii = myl.idx_a(len(tf))
    else:
        ii = myl.find_interval(tf, t_interval)

    if len(ii) < 2:
        return None

    # time and values in selected interval
    t, y = np.asarray(tf)[ii], np.asarray(os[col])[ii]

    # coefs, fitted lines, normalized time
    coef, fit, tn = register_styl(t, y, sr, opt)

    # temporary dict for quick access
    tmp = {}
    for dim in ["bl", "ml", "tl", "rng"]:
        tmp[dim] = {"c": coef[dim], "y": fit[dim]}

    # init return dict
    reg = {"bl_slope": coef["bl"][0],
           "bl_intercept": coef["bl"][1],
           "ml_slope": coef["ml"][0],
           "ml_intercept": coef["ml"][1],
           "tl_slope": coef["tl"][0],
           "tl_intercept": coef["tl"][1],
           "rng_slope": coef["rng"][0],
           "rng_intercept": coef["rng"][1]}

    # mean and rate features
    dur = t[-1]-t[0]
    for x in register_keys():
        reg["{}_mean".format(x)] = np.mean(tmp[x]["y"])
        if dur == 0:
            reg["{}_rate".format(x)] = 0
        else:
            reg["{}_rate".format(x)] = (tmp[x]["y"][-1]-tmp[x]["y"][0])/dur
            
    # to what extent range is determined by lowering baseline
    if np.isnan(reg["rng_mean"]):
        reg["rng_bl"] = 0
    else:
        if reg["bl_mean"] == 0 or np.isnan(reg["bl_mean"]):
            reg["rng_bl"] = reg["rng_mean"]
        else:
            reg["rng_bl"] = reg["rng_mean"] / reg["bl_mean"]

    # coordinates of register crossing
    for ra in ['tl', 'ml']:
        for rb in ['ml', 'bl']:
            if ra == rb:
                continue
            is_t, is_f0 = line_intersect(tmp[ra]['c'], tmp[rb]['c'])
            reg["{}_{}_cross_t".format(ra, rb)] = is_t
            reg["{}_{}_cross_f0".format(ra, rb)] = is_f0

    # plotting
    # opt["plot"] = True
    if opt["plot"]:
        # base-, mid-, topline
        myl.myPlot({"y": tn, "bl": tn, "ml": tn, "tl": tn},
                   {"y": y, "bl": fit["bl"], "ml": fit["ml"], "tl": fit["tl"]},
                   {"ls": {"y": "-b", "bl": "-g", "ml": "-r", "tl": "-g"}})
        # range
        # myl.myPlot({"1": tn},{"1": rng})

    return reg


def register_styl(t, y, sr, opt):
    r''' register (base, mid, topline and range stylization
    Args:
      t (array): time
      y (array): f0
      sr (int): sample rate (Hz). If opt["win"] contains sample indices
             instead of time vales, set sr=1
      opt (dict): see mld_register()
    Returns:
      coef: (dict)
        keys: 'ml'|'bl'|'tl'|'range'; values: coefs [slope offset]
      fit: (dict)
        keys: 'ml'|'bl'|'tl'|'range'; values: fitted lines
      tn: (array) normalized time
    '''

    tn = myl.nrm_vec(myl.idx_a(len(t)),
                     {"mtd": "minmax", "rng": opt["rng"]})

    # fit midline
    ml_coef = np.polyfit(tn, y, 1)

    # windows for base- and topline fit
    # [[start end], ...] indices for windows
    # in which to collect percentile values
    yw = myl.seq_windowing({'win': int(opt['win']*sr),
                            'rng': [0, len(t)],
                            'align': 'center'})

    # normalized time, one value per element in yw
    tw = myl.nrm_vec(myl.idx_a(len(yw)),
                     {"mtd": "minmax", "rng": opt["rng"]})

    # pre-calculate midline in order to avoid bl and tl crossings over midline
    ml = np.polyval(ml_coef, tw)

    # percentile values and time for base- and topline fitting
    blq, tlq, tq = [], [], []
    for i in range(len(yw)):
        s = y[yw[i, 0]:yw[i, 1]]
        if len(s) == 0:
            continue
        if len(s) <= 2:
            qb, qt = np.min(s), np.max(s)
        else:
            qb, qt = np.percentile(s, [opt['bl'], opt['tl']])

        # avoid midline crossings
        qb = np.min([qb, ml[i]])
        qt = np.max([qt, ml[i]])

        tq.append(tw[i])
        blq.append(qb)
        tlq.append(qt)

    bl_coef = np.polyfit(tq, blq, 1)
    tl_coef = np.polyfit(tq, tlq, 1)

    bl = np.polyval(bl_coef, tn)
    ml = np.polyval(ml_coef, tn)
    tl = np.polyval(tl_coef, tn)

    # range
    tb_diff = tl-bl
    tb_diff[myl.find(tb_diff, "<", 0)] = 0
    rng_coef = np.polyfit(tn, tb_diff, 1)
    rng = np.polyval(rng_coef, tn)

    coef = {"bl": bl_coef, "ml": ml_coef, "tl": tl_coef, "rng": rng_coef}
    fit = {"bl": bl, "ml": ml, "tl": tl, "rng": rng}

    return coef, fit, tn


def register_keys():
    return ["bl", "ml", "tl", "rng"]


def line_intersect(c1, c2):
    r''' x and y coordinates of line intersection
    Args:
    c1: [slope intercept] of line 1
    c2: [slope intercept] of line 2
    Returns:
    x: (float) x value (np.nan if lines are parallel)
    y: (float) y value (np.nan if lines are parallel)
    '''
    a, c = c1[0], c1[1]
    b, d = c2[0], c2[1]
    if a == b:
        return np.nan, np.nan
    x = (d-c)/(a-b)
    y = a*x+c
    return x, y


def mld_voi_sust(os, col="voicingFinalUnclipped_sma", t_interval=None, opt={}):

    ''' voicing distribution in sustained sound '''

    opt = myl.opt_default(opt, {"nanrep": 0})
    # time frames and y values
    tf, yf = np.asarray(os["frameTime"]), np.asarray(os[col])
    t = cp.copy(tf)
    # analysis interval
    if t_interval is None:
        t_interval = [tf[0], tf[-1]]
    # restrict to analysis interval
    ti = myl.find_interval(t, t_interval)

    if len(ti) == 0:
        return None

    # voicing probabilities
    voi = yf[ti]
            
    # normalized time points
    w = myl.nrm_vec(ti, {"mtd": "minmax", "rng": [-1, 1]})

    #voisum = np.sum(voi)
    #if voisum > 0:
    #    voi = voi/voisum
    #sm = dct.specmom(voi, w, 4)
    #print("specmom:", sm)

    try:
        sm = dct.specmom(voi, w, 4)            
    except:
        sm = [opt["nanrep"]] * 4

    return {"mean": sm[0], "variance": sm[1],
            "skewness": sm[2], "kurtosis": sm[3]}


def mld_voi(os, col="voicingFinalUnclipped_sma", t=None, t_interval=None, opt={}):

    ''' voicing distributions '''

    
    opt = myl.opt_default(opt, {"win": 0.15, "nanrep": 0})
    # time frames and y values
    tf, yf = np.asarray(os["frameTime"]), np.asarray(os[col])
    
    # syl ncl time stamps
    if t is None:
        t = cp.copy(tf)

    # analysis interval
    if t_interval is None:
        t_interval = [tf[0], tf[-1]]

    # restrict to analysis interval
    ti = myl.find_interval(t, t_interval)

    if len(ti) == 0:
        return None

    # limit time and f0 to analyzed interval
    t = np.asarray(t)[ti]
    tfi = myl.find_interval(tf, t_interval)
    tf = tf[tfi]
    yf = yf[tfi]

    feat = {"mean": np.array([]),
            "variance": np.array([]),
            "skewness": np.array([]),
            "kurtosis": np.array([])}
    
    # over syllables
    for x in t:
        # window
        win = [x-opt["win"]/2, x+opt["win"]/2]
        wi = myl.find_interval(tf, win)
        if len(wi) == 0:
            continue
        # voicing probabilities
        voi = yf[wi]
            
        
        # normalized time points
        w = myl.nrm_vec(wi, {"mtd": "minmax", "rng": [-1, 1]})

        #voisum = np.sum(voi)
        #if voisum > 0:
        #    voi = voi/voisum
        #sm = dct.specmom(voi, w, 4)
        #print("specmom:", sm)

        # ensure that voicing prob is not rising again across syllable boundaries
        c = int(len(voi)/2)
        for i in range(c, -1, -1):
            voi[i] = np.min([voi[i], voi[i+1]])
        for i in range(c, len(voi)-1):
            voi[i+1] = np.min([voi[i], voi[i+1]])

        try:
            sm = dct.specmom(voi, w, 4)            
        except:
            sm = [opt["nanrep"]] * 4
            
        feat["mean"] = np.append(feat["mean"], sm[0])
        feat["variance"] = np.append(feat["variance"], sm[1])
        feat["skewness"] = np.append(feat["skewness"], sm[2])
        feat["kurtosis"] = np.append(feat["kurtosis"], sm[3])

        #voi_plot(feat, w, voi, c)
        
    typ = get_statTyp(opt)
    for x in feat:
        feat[x] = myl.summary_stats(feat[x], typ, nanrep=opt["nanrep"])
        
    return feat

def voi_plot(feat, w, voi, c):
    print("voicingProb:", voi, "\nmean, variance, skewness:", feat["mean"][-1], feat["variance"][-1], feat["skewness"][-1])
    fig = plt.figure()
    _ = fig.canvas.mpl_connect('button_press_event', onclick_next)
    _ = fig.canvas.mpl_connect('key_press_event', onclick_exit)
    plt.plot(w, voi, '-b', linewidth=4)
    plt.plot([w[c], w[c]], [0, 1], '-k', linewidth=3)
    plt.xlabel("time (nrm)", fontsize=18)
    plt.ylabel("voicing probability", fontsize=18)
    plt.show()

        
def mld_varRat_fe(os, col="F0final_sma", col_orig="F0final_sma",
                  t=None, t_interval=None, opt={}):
    ''' variance ratio as proposed by feyben mail 200723 12:54
    1. compute variance/standard deviation of all non 0 F0 values (v_all)
    2. Find segments s_i of continuous voiced regions (F0 > 0)
    3. Compute mean F0 u_i of each segment s_i
    4. Normalise F0 within each segment s_i by subtracting corresponding u_i
    5. Compute variance / standard deviation of all normalised F0 values (v_intraseg)
    6. Compute variance / standard deviation of all u_i (v_interseg)
    7. Compute ratios v_intraseg/v_interseg or v_interseg/v_intraseg  or
       v_interseg/v_all and use these as additional features.
    Difference: s_i segments are windows around syllable nuclei. 
    Args:
    os (pd.DataFrame) openSmile features
    col (str) f0 column name. wrapper will pass "f0_preproc", which
       contains f0 in semitones and interpolated over voiceless segments
       and outliers
    col_orig (str) f0 column name for orig f0 (cf step 1). wrapper will pass
        "f0_noninterp", which contains f0 in semitones without interpolation
    t: (array-like) of syllable ncl time stamps
    t_interval: (2 element array) [start end] of interval to be
      examined. If None: entire file is taken
    opt: (dict)
      win: window around time stamps in t around which to calculate f0
        variance
    Returns:
    varRat_fe (dict)
      intra_inter: v_intraseg/v_interseg
      inter_all: v_interseg/v_all
    '''
    opt = myl.opt_default(opt, {"win": 0.3, "register": None, "nanrep": 0})

    # time frames and y values
    tf, yf = np.asarray(os["frameTime"]), np.asarray(os[col])
    yf_orig = np.asarray(os[col_orig])

    # syl ncl time stamps
    if t is None:
        t = cp.copy(tf)

    # analysis interval
    if t_interval is None:
        t_interval = [tf[0], tf[-1]]

    # restrict to analysis interval
    ti = myl.find_interval(t, t_interval)

    if len(ti) == 0:
        return None

    # limit time and f0 to analyzed interval
    t = np.asarray(t)[ti]
    tfi = myl.find_interval(tf, t_interval)
    tf = tf[tfi]
    yf = yf[tfi]
    yf_orig = yf_orig[tfi]
    
    # variance of all >0 f0 values
    i_pos = myl.find(yf_orig, ">", 0)
    if len(i_pos) == 0:
        v_all = 0
    else:
        v_all = np.var(yf_orig[i_pos])

    # v_all = np.var(yf_orig[myl.find(yf_orig, ">", 0)])

    # centered f0 values, means
    y_intra, y_inter = myl.ea(), myl.ea()

    # over syllables
    for x in t:
        # window indices
        win = [x-opt["win"]/2, x+opt["win"]/2]
        wi = myl.find_interval(tf, win)
        if len(wi) == 0:
            continue
        v = yf[wi]
        m = np.mean(v)
        y_intra = np.append(y_intra, v - m)
        y_inter = np.append(y_inter, m)
        
    v_intra = np.var(y_intra)
    v_inter = np.var(y_inter)

    # variance ratios
    if v_inter == 0:
        intra_inter = 0
    else:
        intra_inter = v_intra / v_inter
    if v_all == 0:
        inter_all = 0
    else:
        inter_all = v_inter / v_all

    return {"intra_inter": intra_inter,
            "inter_all": inter_all}

def mld_varRat(os, col="F0final_sma", t=None, t_interval=None, opt={}):
    ''' variance ratio: variance within segment
    (each segment normalized to mean 0 within) VS. inter segment variance
    (mean of each segment, variance over all segment means)
    os: (dict) openSmile output
    col: (string) name of column to be further processed
    t: (array-like) of time stamps around which to calculate variability.
      If None: each sample index (within t_interval) is processed
    t_interval: (2 element array) [start end] of interval to be
      examined. If None: entire file is taken
    opt: (dict)
      win: window around time stamps in t around which to calculate f0
        variance
      register: ("ml" or None); if "ml": subtract regression midline
           from os[col]
    Returns:
      summaryStat: (dict, None)
        'varRat': (dict) variance ratio
        'var': (dict) variance
        'mean': (dict) mean
      perFrame: (dict, None)
        'varRat', 'var', 'mean': per frame values underlying the
            resp. summary stats
    Remarks: if t_interval is None, but register normalization is wanted,
       it must be done beforehand in entire file (separately for each IPU)
       by preproc.subtract_register(),
       !! and opt["register"] must be set to None !!
    '''

    opt = myl.opt_default(opt, {"win": 0.3, "register": None, "nanrep": 0})

    # Init time stamps and analysis interval
    # time frames and y values
    tf, yf = np.asarray(os["frameTime"]), np.asarray(os[col])
    if t is None:
        t = cp.copy(tf)
    if t_interval is None:
        t_interval = [tf[0], tf[-1]]

    # restrict to analysis interval
    ti = myl.find_interval(t, t_interval)

    if len(ti) == 0:
        return None, None

    t = np.asarray(t)[ti]

    # limit time and y to analyzed interval
    tfi = myl.find_interval(tf, t_interval)
    tf = tf[tfi]
    yf = yf[tfi]

    # subtract register
    if opt["register"] == "ml":
        cr = myl.myPolyfit(tfi, yf, 1)
        yreg = np.polyval(cr, tfi)
        yf -= yreg

    # collect variances and means vectors
    vv, mm = myl.ea(), myl.ea()
    for x in t:
        # window indices
        win = [x-opt["win"]/2, x+opt["win"]/2]
        wi = myl.find_interval(tf, win)

        if len(wi) == 0:
            continue

        c = yf[wi]
        cz = c - np.mean(c)

        # variance of centered segment
        vv = np.append(vv, np.var(cz))
        # mean of segment
        mm = np.append(mm, np.mean(c))

    # variance of means
    vm = np.var(mm)

    # variance ratios
    if vm == 0:
        vr = np.zeros(len(vv))
    else:
        vr = vv/vm

    # summary statistics
    typ = get_statTyp(opt)
    summary = {"varRat": myl.summary_stats(vr, typ, nanrep=opt["nanrep"]),
               "var": myl.summary_stats(vv, typ, nanrep=opt["nanrep"]),
               "mean": myl.summary_stats(mm, typ, nanrep=opt["nanrep"])}

    perFrame = {"varRat": vr,
                "var": vv,
                "mean": mm}

    return summary, perFrame


def mld_rhy(os, col, t, t_interval=None, opt={}):
    r'''rthythm features. Rate of events occuring at times T,
    and their influence on the contour of OS[COL]
    Args:
    os: (dict) openSmile output
    col: (string) name of column to be further processed (usually an
      f0 or energy column
    t: (array-like) of time stamps at which events (e.g. syllables) occur
    t_interval: (2 element array) [start end] of interval to be
      examined. If None: entire file is taken
    opt: (dict)
      register: ("ml" or None); if "ml": subtract regression midline
           from os[col]
      wintyp: (str, 'kaiser'), any type supported by
                        scipy.signal.get_window()
      winparam: (int/string/list, 1) additionally needed window parameters,
          variable type depends on 'wintyp'
      rmo: (boolean, True) skip first (lowest) cosine (=constant offset)
          in spectral moment calculation <1>|0
      lb: (float, 0.0) lower cutoff frequency for coef truncation
      ub: (float, 8.0) upper cutoff frequency (if 0, no cutoff)
          Recommended e.g. for f0 DCT, so that only influence
          of events with <= ub Hz on f0 contour is considered. Default set
          to 8 Hz which is reasonable for calculating the influence of
          syllable or accent rate)
      rb: (float, 1.0) frequency catch band around event rate to measure its
          influence in DCT
          e.g. for event rate 4, and opt['rb']=1
          the abs DCT coefs of 3,4,5 Hz are summed up
    Returns:
    rr: (dict)
      sm1: 1st spectral moment
      sm2: 2nd spectral moment
      sm3: 3rd spectral moment
      sm4: 4th spectral moment
      mae: mean absolute error between IDCT of coefs around resp rate
           and IDCT of coefs between 'lb' and 'ub'
      prop: proportion of coefs around resp rate relative to coef sum
      rate: event rate in analysed domain
      dgm: dist to glob max in dct
      dlm: dist to loc max in dct
      dur - segment duration (in sec)
      f_max - freq of max amplitude
      n_peak - number of peaks in DCT spectrum
    '''

    opt = cp.deepcopy(opt)
    opt = myl.opt_default(opt, {'register': None, 'wintyp': 'kaiser',
                                'winparam': 1, 'rmo': True, 'nsm': 3,
                                'lb': 0, 'ub': 8, 'rb': 1})

    opt["fs"] = os_fs(os)

    # Init time stamps and analysis interval
    # time frames and y values
    tf, yf = np.asarray(os["frameTime"]), np.asarray(os[col])
    if t is None:
        t = cp.copy(tf)
    if t_interval is None:
        t_interval = [tf[0], tf[-1]]

    # restrict to analysis interval
    ti = myl.find_interval(t, t_interval)

    if len(ti) == 0:
        return None

    t = np.asarray(t)[ti]

    # limit time and y to analyzed interval
    tfi = myl.find_interval(tf, t_interval)
    tf = tf[tfi]
    yf = yf[tfi]

    # subtract register
    if opt["register"] == "ml":
        cr = myl.myPolyfit(tfi, yf, 1)
        yreg = np.polyval(cr, tfi)
        yf -= yreg

    # dct features
    # c_orig: all DCT coefs
    # f_orig: all DCT freqs
    # c: coefs with freq between lb and ub
    # f: freq between lb and ub
    # i: indices of 'c' in 'c_orig'
    # sm: spectral moments 1-3
    # m: weighted coef mean
    # sd: weighted coef std
    # cbin: ndarray, summed abs coefs in freq bins between lb and ub
    # fbin: ndarray, corresponding frequencies
    # mae: mean absolute error between IDCT of coefs around resp rate
    # and IDCT of coefs between 'lb' and 'ub'
    # prop: proportion of coefs around resp rate relative to coef sum
    # rate: event rate in analysed domain
    # dgm: dist to glob max in dct
    # dlm: dist to loc max in dct
    # dur - segment duration (in sec)
    # f_max - freq of max amplitude
    # n_peak - number of peaks in DCT spectrum
    rhy = dct.dct_wrapper(yf, opt)

    # number of local maxima
    rhy['n_peak'] = len(rhy['f_lmax'])

    # duration
    rhy['dur'] = t_interval[-1] - t_interval[0]

    # rate
    rhy['rate'] = len(t)/rhy['dur']

    # domain weight features + ['wgt'][myDomain]['prop'|'mae'|'rate']
    rhy = rhy_sub(yf, rhy, opt)

    #rhy_plot2(tf, yf, rhy, col, opt)
    #rhy_plot(rhy, col, opt)

    # return dict
    rr = {"sm1": rhy["sm"][0],
          "sm2": rhy["sm"][1],
          "sm3": rhy["sm"][2],
          "sm4": rhy["sm"][3]}
    for x in ["mae", "prop", "rate", "dgm", "dlm", "dur", "f_max",
              "n_peak"]:
        rr[x] = rhy[x]

    return rr


def mld_isochrony(t, t_interval=None):
    ''' calculates classical syllable isochrony measures:
    pairwise variability index and varcoef of inter-timestamps intervals
    (syllables) in t
    Args:
    t: (np.array) of timestamps (syllable nuclei)
    Returns:
    rr (dict) "pvi": (float) normalized pairwise variability index of
                     np.diff(t)
              "varco": (float) variation coefficient of np.diff(t)
    '''

    rr = {"pvi": 0, "varco": 0}

    # restrict t to analysis interval
    if t_interval is not None:
        ti = myl.find_interval(t, t_interval)
    if len(ti) == 0:
        return rr
    t = np.asarray(t)[ti]
    if len(t) < 2:
        return rr

    # interval durations
    d = np.diff(t)

    # variation coef
    sd, m = np.std(d), np.mean(d)
    if m > 0:
        rr["varco"] = 100 * sd / m

    if len(d) < 2:
        return rr

    # pairwise variability index
    pvi = 0
    if len(d) > 1:
        for i in range(len(d)-1):
            denom = (d[i]+d[i+1])/2
            if denom > 0:
                pvi += (np.abs(d[i]-d[i+1]) / denom)
        rr["pvi"] = 100 * pvi / (len(d)-1)
    else:
        rr["pvi"] = 0

        
    # print("t:", t,"\nd:",d,"\n-->",rr)
    # myl.stopgo()

    return rr


def rhy_sub(y, rhy, opt):

    if len(y) == 0:
        for x in ["mae", "prop", "dgm", "dlm"]:
            rhy[x] = np.nan
        return rhy

    # sum(abs(coeff)) between opt['lb'] and opt['ub']
    ac = abs(rhy['c'])
    sac = sum(ac)

    # freqs of max values
    gpf, lpf = rhy['f_max'], rhy['f_lmax']

    # IDCT by coefficients with freq between lb and ub
    #   (needed for MAE and prop normalization. Otherwise
    #    weights depend on length of y)
    yr = dct.idct_bp(rhy['c_orig'], rhy['i'])

    # distances to global and nearest local peak
    dg = rhy['rate'] - gpf
    dl = np.exp(10)
    for z in lpf:
        dd = rhy['rate']-z
        if abs(dd) < dl:
            dl = dd

    # define catch band around respective event rate of x
    lb = rhy['rate']-opt['rb']
    ub = rhy['rate']+opt['rb']

    # 'prop': DCT coef abs ampl around rate
    if len(ac) == 0:
        j = myl.ea()
        prp = 0
    else:
        j = myl.intersect(myl.find(rhy['f'], '>=', lb),
                          myl.find(rhy['f'], '<=', ub))
        if len(j) == 0 or sac == 0:
            prp = 0
        else:
            prp = sum(ac[j])/sac

    # 'mae': mean abs error between rhy[c] IDCT and IDCT of
    #        coefs between event-dependent lb and ub
    yrx = dct.idct_bp(rhy['c_orig'], j)
    ae = myl.mae(yr, yrx)

    rhy['mae'] = ae
    rhy['prop'] = prp
    rhy['dlm'] = dl
    rhy['dgm'] = dg

    return rhy

def rhy_plot2(tt, yt, rhy, col, opt):
    ''' rhythm features time + stem-plot '''

    n = 7
    y = running_mean(yt, n)
    t = tt[n-1:len(tt)]
    
    if re.search("F0", col):
        myTitle1 = "F0 (ST)"
        myXlab1 = "time (sec)"
        myTitle = "influence on F0 contour"
        myXlab = "f (Hz)"
    else:
        myTitle1 = "energy"
        myXlab1 = "time (sec)"
        myTitle = "influence on energy contour"
        myXlab = "energy"

    fig, spl = plt.subplots(2, 1, squeeze=False)
    _ = fig.canvas.mpl_connect('button_press_event', onclick_next)
    _ = fig.canvas.mpl_connect('key_press_event', onclick_exit)
    fig.subplots_adjust(hspace=0.5)

    spl[0, 0].plot(t, y, linewidth=3)
    spl[0, 0].title.set_text(myTitle1)
    spl[0, 0].set_xlabel(myXlab1, fontsize=15)
    spl[0, 0].set_ylabel(myTitle1, fontsize=15)

    c_sum = sum(abs(rhy['c']))
    mla, sla, bla = spl[1, 0].stem(rhy['f'], abs(rhy['c'])/c_sum, '-.')
    plt.setp(sla, 'color', 'k', 'linewidth', 3)
    spl[1, 0].title.set_text(myTitle)
    b = [max([0, rhy["rate"]-opt["rb"]]), rhy["rate"]+opt["rb"]]
    w = myl.intersect(myl.find(rhy['f'], '>=', b[0]),
                      myl.find(rhy['f'], '<=', b[1]))
    if len(w) == 0:
        return
    ml, sl, bl = spl[1, 0].stem(rhy['f'][w], abs(rhy['c'][w])/c_sum)
    plt.setp(sl, 'color', 'k', 'linewidth', 6)
    spl[1, 0].set_xlabel(myXlab, fontsize=15)
    spl[1, 0].set_ylabel('|coef|', fontsize=15)

    plt.show()

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

    
def rhy_plot(rhy, col, opt):
    ''' rhythm features stem-plot '''

    if re.search("F0", col):
        myTitle = "influence on F0 contour"
        myXlab = "f (Hz)"
    else:
        myTitle = "influence on energy contour"
        myXlab = "energy"

    fig, spl = plt.subplots(1, 1, squeeze=False)
    _ = fig.canvas.mpl_connect('button_press_event', onclick_next)
    _ = fig.canvas.mpl_connect('key_press_event', onclick_exit)
    fig.subplots_adjust(hspace=0.8)
    c_sum = sum(abs(rhy['c']))
    mla, sla, bla = spl[0, 0].stem(rhy['f'], abs(rhy['c'])/c_sum, '-.')
    plt.setp(sla, 'color', 'k', 'linewidth', 3)
    spl[0, 0].title.set_text(myTitle)
    b = [max([0, rhy["rate"]-opt["rb"]]), rhy["rate"]+opt["rb"]]
    w = myl.intersect(myl.find(rhy['f'], '>=', b[0]),
                      myl.find(rhy['f'], '<=', b[1]))
    if len(w) == 0:
        return
    ml, sl, bl = spl[0, 0].stem(rhy['f'][w], abs(rhy['c'][w])/c_sum)
    plt.setp(sl, 'color', 'k', 'linewidth', 6)
    spl[0, 0].set_xlabel(myXlab, fontsize=18)
    spl[0, 0].set_ylabel('|coef|', fontsize=18)

    plt.show()


def onclick_next(event):
    plt.close()


def onclick_exit(event):
    sys.exit()


def mld_shape_sust(os, col="F0final_sma", t_interval=None, opt={}):
    r''' shape variation of some time series (if it is f0 it SHOULD already
    be preprocessed e.g. by preproc.preproc_f0()!) For sustained sounds.
    Args:
    os: (dict) openSmile output
    col: (string) name of column to be further processed
    t_interval: (2 element array) [start end] of interval to be
      examined. If None: entire file is taken
    opt: (dict)
      ord: polynomial order
      rng: [-1, 1] normalized time range
      register: ("ml" or None); if "ml": subtract regression midline
           from os[col]
    Returns:
      v (dict)
         y: time series summary stats
         dlt: abs delta summary stats
         d_start, d_peak, d_end: intial, peak, final distance to midline
         mae: mae between contour and its midline
         mcr: midline crossing rate
         tpr: turning point rate
    Remark: new features need to be added also to mld_pipeline.mld_upd()
    '''

    opt = myl.opt_default(opt, {"nanrep": 0,
                                "plot": False})


    # Init time stamps and analysis interval
    # time frames and y values
    tf, yf = np.asarray(os["frameTime"]), np.asarray(os[col])
    t = cp.copy(tf)
    if t_interval is None:
        t_interval = [tf[0], tf[-1]]
        
    # restrict to analysis interval
    ti = myl.find_interval(t, t_interval)

    if len(ti) == 0:
        return None

    # limit time and y to analyzed interval
    tfi = myl.find_interval(tf, t_interval)
    tf = tf[tfi]
    y = yf[tfi]

    if len(tf) < 2:
        return None
    
    # normalized time (for comparable ML coefs)
    t_nrm = myl.nrm_vec(tf, {"mtd": "minmax", "rng": [0, 1]})
    
    # midline (to which distances are calculated)
    ml_coef = myl.myPolyfit(t_nrm, y, 1)
    ml = np.polyval(ml_coef, t_nrm)

    # shape: not relevant for the moment 
    #c = myl.myPolyfit(t_nrm, y, 3)
    #if np.isnan(c).any():
    #    return None
    # stylized contour
    #v = np.polyval(c, t_nrm)
    # MAE of contour and polycoefs to zero
    #mae_contour = sm.mean_absolute_error(v, np.zeros(len(v)))
    #mae_coef = sm.mean_absolute_error(c, np.zeros(len(c)))
    # location of stylization maximum (in t_nrm)
    #ima, imi = np.argmax(v), np.argmin(v)
    #rng = np.max(v)-np.min(v)
    #tmax = t_nrm[ima]
    #tmin = t_nrm[imi]

    # y and abs-delta sumstats
    sumstat_y = myl.summary_stats(y, "robust", nanrep=opt["nanrep"])
    sumstat_dlt = myl.summary_stats(np.abs(np.diff(y)),
                                    opt["summaryStat"],
                                    nanrep=opt["nanrep"])

    # f0 residual
    yr = y - ml

    # start, peak, end distance of styl contour to mean
    d_start, d_peak, d_end = yr[0], np.max(yr), yr[-1]

    # mae to ml
    mae = np.mean(np.abs(yr))
    
    # ml crossing rate
    tt = (tf[-1] - tf[0])
    mcr = ((yr[:-1] * yr[1:]) < 0).sum() / tt
    
    # turning point rate
    # remove plateaus
    yu = np.array([k for k,g in itertools.groupby(y)])
    dx = np.diff(yu)
    tpr = np.sum(dx[1:] * dx[:-1] < 0) / tt
    
    if opt["plot"]:
        # midline and f0
        myl.myPlot({"y": t_nrm, "ml": t_nrm},
                   {"y": y, "ml": ml},
                   {"ls": {"y": "-b", "ml": "-r"}})
        
    return {"y": sumstat_y, "dlt": sumstat_dlt,
            "d_start": d_start, "d_peak": d_peak, "d_end": d_end,
            "mae": mae, "mcr": mcr, "tpr": tpr}

def mld_shape(os, col="F0final_sma", t=None, t_interval=None, opt={}):
    r''' shape variation of some time series (which SHOULD already
    be preprocessed e.g. by preproc.preproc_f0()!)
    Args:
    os: (dict) openSmile output
    col: (string) name of column to be further processed
    t: (array-like) of time stamps at which shapes are to be stylized.
      If None: each sample index (within t_interval) is processed
    t_interval: (2 element array) [start end] of interval to be
      examined. If None: entire file is taken
    opt: (dict)
      win: stylization window length (in sec)
      ord: polynomial order
      drop0: ignore zeroth order polycoef in variation measure
      rng: [-1, 1] normalized time range
      register: ("ml" or None); if "ml": subtract regression midline
           from os[col]
      dist: (str); {"mahal", "mahal_distrib", <"euclidean">, "canberra"}
        how to calculate spread. "mahal": mahalanobis distance of each
        vector to centroid. "mahal_distrib": mahalanobis distance of
        each vector to vector distribution. "euclidean": euclidean distance
        of each vector to centroid. "canberra": canberra distance which
        normalizes for abs values in vectors
      scale: (boolean) robust scaling of polycoef matrix to mean and sd.
        True recommended due to coef value range differences.
    Returns:
    summaryStat: (dict, None)
       'spread': (dict) user-defined distance from centroid
       'spreadzero' (dict) Euclidean distance from zero-vector
       'flux': (dict) Euclidean distance between subsequent frames
       'coef': (dict) vectors for polycoef statistics (descending coef order)
                (np.arrays of floats)
       'mae': (dict) root mean absolute deviation between contour and reference
           line (midline if opt[register] is "ml", else zeros)
       'diff': mean difference between contour and reference line (midline
           or zeros)
       'tmax': (nrm) time of stylized F0 maximum
       'tmin': (nrm) time of stylized F0 minimum
       'rng': range (max-min of styl. contour)
    perFrame: (dict, None)
       't': (np.array) time stamps
       'spread': per frame distances underlying spread summary
       'spreadzero': per frame distances underlying spreadzero summary
       'flux': per frame distances underlying flux summary
            (horizontal left extrapolation)
       'coef': (2-dim np-array) of polycoef vectors in descending order
       'mae': per frame mae
       'diff': per frame diff
       'tmax': per fram tmax
       'tmin': per frame tmin
       'rng': per fram range
    Remarks: if t_interval is None, but register normalization is wanted,
       it must be done beforehand in entire file (separately for each IPU)
       by preproc.subtract_register(),
       !! and opt["register"] must be set to None !!
    '''

    opt = myl.opt_default(opt, {"win": 0.3, "ord": 3,
                                "drop0": False,
                                "rng": [-1, 1],
                                "register": None,
                                "dist": "euclidean",
                                "scale": False,
                                "nanrep": 0})

    # Init time stamps and analysis interval
    # time frames and y values
    tf, yf = np.asarray(os["frameTime"]), np.asarray(os[col])
    if t is None:
        t = cp.copy(tf)
    if t_interval is None:
        t_interval = [tf[0], tf[-1]]

    # restrict to analysis interval
    ti = myl.find_interval(t, t_interval)

    if len(ti) == 0:
        return None, None

    t = np.asarray(t)[ti]

    # limit time and y to analyzed interval
    tfi = myl.find_interval(tf, t_interval)
    tf = tf[tfi]
    yf = yf[tfi]

    # subtract register
    if opt["register"] == "ml":
        cr = myl.myPolyfit(tfi, yf, 1)
        yreg = np.polyval(cr, tfi)
        # yo = cp.copy(yf)
        yf -= yreg
        # myl.myPlot({"1": tfi, "2": tfi, "3": tfi},
        #           {"1": yo, "2": yreg, "3": yf})

    # collect coefficient vectors
    # columns: coefs, rows: time
    # over time stamps
    coefs, tt, maes, diffs = myl.ea(), myl.ea(), myl.ea(), myl.ea()
    tmaxs, tmins, rngs = myl.ea(), myl.ea(), myl.ea()
    for x in t:
        # window indices
        win = [x-opt["win"]/2, x+opt["win"]/2]
        wi = myl.find_interval(tf, win)

        if len(wi) <= opt["ord"]+1:
            continue
        t_nrm = myl.nrm_vec(wi, {"mtd": "minmax", "rng": opt["rng"]})

        c = myl.myPolyfit(t_nrm, yf[wi], opt["ord"])
        tt = np.append(tt, x)
        # drop offset coef?
        if opt["drop0"]:
            coefs = myl.push(coefs, c[0:len(c)-1])
        else:
            coefs = myl.push(coefs, c)

        # area under contour, sum of differences
        # area: how much does contour differ from midline
        v = np.polyval(c, t_nrm)
        z = np.zeros(len(v))
        maes = np.append(maes, sm.mean_absolute_error(v, z))

        # if maes[-1] > 100:
        #    myl.myPlot({"1": t_nrm, "2": t_nrm},
        #               {"1": yf[wi], "2": v},
        #               {"ls": {"1": "-b", "2": "-r"}})

        diffs = np.append(diffs, np.mean(v))

        # location of maximum (in t_nrm)
        ima, imi = np.argmax(v), np.argmin(v)
        rngs = np.append(rngs, np.max(v)-np.min(v))
        tmaxs = np.append(tmaxs, t_nrm[ima])
        tmins = np.append(tmins, t_nrm[imi])

    # spread and flux per frame
    spreadX, spreadzeroX, fluxX = spread_flux(coefs, opt["dist"], opt["scale"])

    # summary statistics
    typ = get_statTyp(opt)

    summary = {"coef": myl.summary_stats(coefs, typ, nanrep=opt["nanrep"]),
               "spread": myl.summary_stats(spreadX, typ, nanrep=opt["nanrep"],
                                           exceptVal=0),
               "spreadzero": myl.summary_stats(spreadzeroX, typ,
                                               nanrep=opt["nanrep"],
                                               exceptVal=0),
               "flux": myl.summary_stats(fluxX, typ, nanrep=opt["nanrep"],
                                         exceptVal=0),
               "mae": myl.summary_stats(maes, typ, nanrep=opt["nanrep"]),
               "diff": myl.summary_stats(diffs, typ, nanrep=opt["nanrep"]),
               "tmax": myl.summary_stats(tmaxs, typ, nanrep=opt["nanrep"]),
               "tmin": myl.summary_stats(tmins, typ, nanrep=opt["nanrep"]),
               "rng": myl.summary_stats(rngs, typ, nanrep=opt["nanrep"])}

    # per frame
    # horizontal extrapolation of fluxX
    if len(fluxX) == 0:
        fluxX = np.append(fluxX, 0)
    else:
        fluxX = np.insert(fluxX, 0, fluxX[0])
    perFrame = {"t": tt,
                "coef": coefs,
                "spread": spreadX,
                "spreadzero": spreadzeroX,
                "flux": fluxX,
                "mae": maes,
                "diff": diffs,
                "tmax": tmaxs,
                "tmin": tmins,
                "rng": rngs}

    return summary, perFrame


def mahal(x, y=None, data=None, cov=None):
    r''' calculate spread in terms of mahalanobis distance of each
    coef vector in x to reference vector y (default: centroid of x).
    Underlying distribution is defined by data (external) or x.
    Underlying covariance is defined by cov or calculated from data, resp. x
    Args:
    x: (2-dim np-array) of coef vectors (rows) of size m x n
    y: (None or 1-dim np-array) of reference vector of size 1 x n
    data: (None or 2-dim np-array) of size u x n. Data underlying the
       external distribution
    cov: (None or 2-dim np-array) covariance matrix of size n x n
    Returns:
    d: (1-dim np-array) of Mahalanobis distances for each coef vector
    '''
    d = myl.ea()

    # reference vector
    if y is None:
        y = np.mean(x, 0)

    # inverse covariance matrix
    if not cov:
        if not data:
            data = x
        if len(data) < 2:
            return None
        cov = np.cov(data.T)
        # print(data.T) #!x

    try:
        inv_covmat = sc.linalg.inv(cov)
    except:
        # myl.stopgo("return None") #!x
        return None

    np.seterr(all='raise')  # !x

    # distances
    for z in x:
        try:
            d = np.append(d, sc.spatial.distance.mahalanobis(z, y, inv_covmat))
        except:
            continue
    return d


def mahal_distrib(x, data=None, cov=None):
    r''' calculate spread in terms of mahalanobis distance of each
    coef vector to coef distribution or some external distribution
    introduced by data and/or cov.
    Implementation adapted from:
    https://www.machinelearningplus.com/statistics/mahalanobis-distance/
    Args:
    x: (2-dim np-array) of coef vectors (rows) of size m x n
    data: (None or 2-dim np-array) of size u x n. Data underlying the
       external distribution
    cov: (None or 2-dim np-array) covariance matrix of size n x n
    Returns:
    mahal: (1-dim np-array) of Mahalanobis distances for each coef vector
    '''

    if not data:
        data = x

    x_minus_mu = x - np.mean(data)

    if not cov:
        cov = np.cov(data.T)

    inv_covmat = sc.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()


def spread_flux(c_in, dist="euclidean", scale=False):
    r''' calculate spread (variation) and flux (delta) features.
    Spread: user-defined distance to centroid
    Spreadzero: Euclidean (or mahalanobis) distance to zero-vector
    Flux: user-defined distance between subsequent vectors
    Args:
    c: (2-dim np-array) of coefs of size m x n.
    dist: (str) {"mahal", "mahal_distrib", <"euclidean">, "canberra"}
      Distance metrics. See mld_vec() for details.
    scale: (boolean) if True, coef matrix is scaled.
      Only for spread with "euclidean" dist, and for flux.
    Returns:
    spread: (1-dim np.array) of length m
    spreadzero: (1-dim np.array) of length m
    flux: (1-dim np.array) of length m-1
    Remarks: spreadzero only supports mahalanobis or euclidean
    distance since plenty of other distances (e.g. canberra) would lead to
    constant 1 (distance = 1/normalizationFactor)
    '''
    spread, spreadzero, flux = myl.ea(), myl.ea(), myl.ea()

    # copy needed for cases c is center-scaled plus summary statistics
    # is calculated later on
    c = cp.deepcopy(c_in)

    if len(c) == 0:
        return spread, spreadzero, flux

    cntr_zero = np.zeros(len(c[0]))
    did_spread = False

    if dist == "mahal":
        spread = mahal(c)
        spreadzero = mahal(c, cntr_zero)
        if spread is None:
            return myl.ea(), myl.ea(), myl.ea()
        did_spread = True
    elif dist == "mahal_distrib":
        spread = mahal_distrib(c)
        spreadzero = mahal_distrib(c-np.mean(c, 0))
        did_spread = True

    # scaling, centroid
    if scale:
        obj = sp.StandardScaler().fit(c)
        c = obj.transform(c)
        cntr = np.mean(c, 0)
    else:
        cntr = np.mean(c, 0)

    for i in myl.idx(c):
        if not did_spread:
            dv = dist_wrapper(dist, c[i, :], cntr)
            dvz = dist_wrapper("euclidean", c[i, :], cntr_zero)
            spread = np.append(spread, dv)
            spreadzero = np.append(spreadzero, dvz)
        if i > 0:
            dd = dist_wrapper(dist, c[i, :], c[i-1, :])
            flux = np.append(flux, dd)

    return spread, spreadzero, flux


def dist_wrapper(dist, x, y):
    if dist == "euclidean":
        return myl.dist_eucl(x, y)
    elif dist == "canberra":
        return sc.spatial.distance.canberra(x, y)
    print(dist, "not supported!")
    return np.nan


def get_statTyp(opt):    
    if "summaryStat" in opt:
        return opt["summaryStat"]
    return "robust"


def os_fs(os):
    r'''returns feature extraction sample rate from opensmileTable'''

    if (("frameTime" not in os) or len(os["frameTime"]) < 2):
        print("cannot extract feature sample rate from table. Return None")
        return None

    return int(1/(os["frameTime"][1]-os["frameTime"][0]))
