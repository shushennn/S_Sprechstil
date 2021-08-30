#!/usr/bin/env python3

import sys
import os
import json
import re
import numpy as np
import pickle
import pandas as pd
import scipy.stats as sst
import scipy.cluster.vq as sc
import scipy.io.wavfile as sio
#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import copy as cp
import math
import seaborn as sns
import datetime as dt
import shutil as sh
import scipy.stats as st
import scikit_posthocs as sp

pd.options.mode.chained_assignment = None

# collection of functions of general use

def from_audb(df):
    ''' de-audb-ify df: index to columns, absolute sec time vals '''
    df.reset_index(inplace=True)
    return from_timedelta(df)


def to_audb(df):
    ''' audb-ify df: columns to index, timedelta '''
    df = to_timedelta(df)
    df.set_index(["file", "start", "end"], inplace=True)
    return df


def to_timedelta(df, col=["start","end","sub_start","sub_end"]):
    ''' transforms dataframe columns col to timedelta objects.
    (if start and end are part of the multiindex they need to be
    transformed to columns before using this function) '''
    
    for c in col:
        if c not in df:
            continue
        y = df[c].to_numpy()
        if type(y[0]) is not str:
            y = [dt.timedelta(seconds=i) for i in y]
        df[c] = pd.to_timedelta(y)
        
    return df



def to_timedelta_deprec(df, col=["start", "end"]):
    ''' transforms dataframe columns col to timedelta objects.
    (if start and end are part of the multiindex they need to be
    transformed to columns before using this function) '''

    for c in col:
        if c not in df:
            continue
        y = df[c].to_numpy()
        y = [dt.timedelta(seconds=i) for i in y]
        df[c] = pd.to_timedelta(y)

    return df


def from_timedelta(df, col=["start", "end"]):
    ''' transforms dataframe columns col from string or timedelta to
    absolute seconds.
    (if start and end are part of the multiindex they need to be
    transformed to columns before using this function) '''

    for c in col:
        if c not in df:
            continue

        if df[c].dtype == "object":
            y = pd.to_timedelta(df[c])
            y = y.dt.total_seconds().to_numpy()
        else:
            y = df[c].dt.total_seconds().to_numpy()
        df[c] = y

    return df


def of_list_type(x):
    ''' returns True if x is list or np.array '''
    if (type(x) is list) or (type(x) is np.ndarray):
        return True
    return False


def ndim(x):
    ''' returns number of dimensions of array x '''
    return len(x.shape)


def find_interval(x, iv):
    r''' returns indices of values in 1-dim x that are >= iv[0] and <= iv[1]
    '''
    xi = np.where((x >= iv[0]) & (x <= iv[1]))
    return xi[0]


def find_interval_deprec(x, iv, fs=-1):
    r''' deprecated, very slow.
    returns indices of values in 1-dim x that are >= iv[0] and <= iv[1]
    '''
    xi = sorted(intersect(find(x, '>=', iv[0]),
                          find(x, '<=', iv[1])))
    y = np.asarray(xi).astype(int)
    return y


def str2logic(x):
    ''' returns boolean True, False, or None from TRUE|FALSE|NONE
    string input '''
    if not is_type(x) == 'str':
        return x
    if not re.search('(TRUE|FALSE|NONE)', x):
        return x
    elif x == 'TRUE':
        return True
    elif x == 'FALSE':
        return False
    return None

def int2bool(x):
    ''' maps 0 to False, rest to True '''
    if x == 0:
        return False
    return True

def reversi(x, opt={}):
    r''' reversi operation: replace captured element by surround ones
    Args:
    x: 1-dim list
    opt:
     .infx: <1> infix max length
     .ngb_l: <2> left context min length
     .ngb_r: <2> right context max length
    Returns:
    y: x or all elements replaced by first one
    Examples (default options):
    [a,a,b,a,a] -> [a,a,a,a,a]
    [a,a,b,a]   -> [a,a,b,a] (right context too short)
    [a,a,b,c,c] -> [a,a,b,c,c] (left and right context not the same)
    [a,a,b,c,c] -> [a,a,b,a,c] (4 instead of 3 uniform-element sublists)
    '''

    opt = opt_default(opt, {"infx": 1, "ngb_l": 2, "ngb_r": 2})
    if len(x) < opt["infx"]+opt["ngb_l"]+opt["ngb_r"]:
        return x
    y = seg_list(x)
    if (len(y) == 3 and y[0][0] == y[-1][0] and
        len(y[0]) >= opt["ngb_l"] and
        len(y[-1]) >= opt["ngb_r"] and
            len(y[1]) <= opt["infx"]):
        return [y[0][0]] * len(x)
    return x


def seg_list(x):
    r''' segment list into same element segments
    Args:
    x: 1-dim list
    Returns:
    y: 2-dim list
    example:
    [a,a,b,b,a,c,c] -> [[a,a],[b,b],[a],[c,c]]
    '''
    if len(x) == 0:
        return [[]]
    y = [[x[0]]]
    for i in range(1, len(x)):
        if x[i] == x[i-1]:
            y[-1].append(x[i])
        else:
            y.append([x[i]])
    return y


def nan_repl(y, rep):
    ''' replaces nan, inf by rep in np.array y '''
    y[np.isneginf(y)] = 0
    y[np.isinf(y)] = 0
    y[np.isnan(y)] = 0
    return y


def summary_stats(y, typ="standard", nanrep=np.nan, exceptVal=np.nan,
                  b=95):
    r''' summary statistics of some input vector
    Input:
    y: (np.array) of floats; if 2-dim, column-wise statistics are returned
    typ: (str) "standard" or "robust"
    nanrep: (float) value by which to replace nan and inf values in y
    exceptVal: (numeric or np.nan) uniform value to be returned for
        empty input
    b: confidence interval beta value (in percent)
    Returns:
    s: (dict)
      typ=="standard": 'mean', 'var', 'skewness', 'kurtosis',
           'range', 'ci_lower|upper|range'
      typ=="robust": 'median', 'q_var', 'iqr', 'q_skewness',
            'q_kurtosis', 'q_range10', 'q_range5',
            'q_ci_lower|upper|range' (ci for confidence interval)
      typ=="misc": several openSmile functionals not included in standard
            nor robust
      typ=="standard_extended": "standard" + "misc"
      typ=="robust_extended": "robust" + "misc"
      typ=="extended": "standard" + "robust" + "misc"
      for standard/robust: 'slope': (of linear regression, x and y
                                     minmax-normalized to [0 1])
                           'diff': m of first half minus m of second
                                   half (m is mean or median depending on typ)
                           'start': first_value/median
                           'peak': last_value/median
                           'end': final_value/median
      scalar for 1-dim array; array for 2-dim array
      Remark: slope, diff, start, peak, end only meaningful for time series data
    '''

    # replace inf and nan by nanrep
    y = nan_repl(y, nanrep)

    # standard
    if typ == "standard":
        if len(y) == 0:
            return {"mean": exceptVal,
                    "var": exceptVal,
                    "skewness": exceptVal,
                    "kurtosis": exceptVal,
                    "range": exceptVal,
                    "slope": exceptVal,
                    "diff": exceptVal,
                    "ci_lower": exceptVal,
                    "ci_upper": exceptVal,
                    "ci_range": exceptVal,
                    "start": exceptVal,
                    "peak": exceptVal,
                    "end": exceptVal}

            return z
            
        slope, diff = sumStat_slopeDiff(y, typ, exceptVal)
        myStart, myPeak, myEnd = sumStat_startPeakEnd(y, exceptVal)
        ci = confidence_interval(y, typ, b)

        return {"mean": np.mean(y, 0),
                "var": np.var(y, 0),
                "skewness": sst.skew(y, 0),
                "kurtosis": sst.kurtosis(y, 0),
                "range": np.max(y, 0)-np.min(y, 0),
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                "ci_range": ci[1]-ci[0],
                "slope": slope,
                "diff": diff,
                "start": myStart,
                "peak": myPeak,
                "end": myEnd}

    # robust
    elif typ == "robust":
        if len(y) == 0:
            return {"median": exceptVal,
                    "iqr": exceptVal,
                    "q_var": exceptVal,
                    "q_skewness": exceptVal,
                    "q_kurtosis": exceptVal,
                    "q_range5": exceptVal,
                    "q_range10": exceptVal,
                    "q_ci_lower": exceptVal,
                    "q_ci_upper": exceptVal,
                    "q_ci_range": exceptVal,
                    "slope": exceptVal,
                    "diff": exceptVal,
                    "start": exceptVal,
                    "peak": exceptVal,
                    "end": exceptVal}
        
        myMed, myIqr, myVar, mySkew, myKurt, \
            myRange5, myRange10 = nonparam_description(y)
        slope, diff = sumStat_slopeDiff(y, typ, exceptVal)
        myStart, myPeak, myEnd = sumStat_startPeakEnd(y, exceptVal)
        ci_lower, ci_upper = confidence_interval(y, typ, b)

        return {"median": myMed,
                "iqr": myIqr,
                "q_var": myVar,
                "q_skewness": mySkew,
                "q_kurtosis": myKurt,
                "q_range5": myRange5,
                "q_range10": myRange10,
                "q_ci_lower": ci_lower,
                "q_ci_upper": ci_upper,
                "q_ci_range": ci_upper-ci_lower,
                "slope": slope,
                "diff": diff,
                "start": myStart,
                "peak": myPeak,
                "end": myEnd}

    elif typ == "misc":
        if len(y) == 0:
            return {"iqr1-2": exceptVal,
                    "iqr1-3": exceptVal,
                    "iqr2-3": exceptVal,
                    "percentile1": exceptVal,
                    "percentile99": exceptVal,
                    "quartile1": exceptVal,
                    "quartile2": exceptVal,
                    "quartile3": exceptVal,
                    "rqmean": exceptVal}
        pp = np.percentile(y, [1, 25, 50, 75, 99], 0)
        # 1-dim array
        if len(y.shape) == 1:
            rqm = np.sqrt(np.sum(y**2)/len(y))
        else:
            rqm = np.array([])
            for i in range(y.shape[1]):
                rqm = np.append(rqm, np.sqrt(np.sum(y[:,i]**2)/y.shape[0]))
        return {"iqr1-2": pp[2]-pp[1],
                "iqr1-3": pp[3]-pp[1],
                "iqr2-3": pp[3]-pp[2],
                "percentile1": pp[0],
                "percentile99": pp[4],
                "quartile1": pp[1],
                "quartile2": pp[2],
                "quartile3": pp[3],
                "rqmean": rqm}

    # robust-extended
    elif typ == "robust_extended":
        stat = summary_stats(y, "robust", exceptVal=exceptVal)
        stat_misc = summary_stats(y, "misc", exceptVal=exceptVal)
        for x in stat_misc:
            stat[x] = stat_misc[x]
        return stat
            
    # standard-extended
    elif typ == "standard_extended":
        stat = summary_stats(y, "standard", exceptVal=exceptVal)
        stat_misc = summary_stats(y, "misc", exceptVal=exceptVal)
        for x in stat_misc:
            stat[x] = stat_misc[x]
        return stat
        
    # extended
    stat_standard = summary_stats(y, "standard", exceptVal=exceptVal)
    stat_robust = summary_stats(y, "robust", exceptVal=exceptVal)
    stat_misc = summary_stats(y, "misc", exceptVal=exceptVal)
    stat = cp.deepcopy(stat_standard)
    for x in stat_robust:
        stat[x] = stat_robust[x]
    for x in stat_misc:
        stat[x] = stat_misc[x]
    return stat


def confidence_interval(y, typ="standard", b=95):
    ''' calculates confidence interval for the mean,
    resp. median based on parametric (typ="standard") or
    non-parametric (typ="robust") distribution of y.
    Returns ci_lower, and ci_upper, which are scalars for
    1-dim input and np.arrays (one lement per column) for
    2-dim input.
    '''

    # 1-dim array
    if len(y.shape) == 1:
        ci = ssci(y, typ, b)
        return ci[0], ci[1]

    # 2-dim array
    ci_lower, ci_upper = ea(), ea()
    for i in idx(y[0]):
        ci = ssci(y[:, i], typ, b)
        ci_lower = np.append(ci_lower, ci[0])
        ci_upper = np.append(ci_upper, ci[1])

    return ci_lower, ci_upper


def ssci(y, typ, b=95):

    if b == 95:
        f = 1.96
    elif b == 99:
        f = 2.58
    else:
        sys.exit("myUtilHL.ssci(): b needs to be 95 or 99.")

    if len(y) == 0:
        return np.array([np.nan, np.nan])

    n = len(y)

    # non-parametric
    if typ == "robust":
        ys = np.sort(y)
        m = np.median(y)
        w = f*np.sqrt(n)/2
        i0 = n/2-w
        i1 = 1+n/2+w
        i0 = np.max([np.int(1), np.int(np.round(i0))]) - 1
        i1 = np.min([n, np.int(np.round(i1))]) - 1
        return np.array([ys[i0], ys[i1]])

    # parametric
    m = np.mean(y)
    ci0 = m - f * np.std(y)/np.sqrt(n)
    ci1 = m + f * np.std(y)/np.sqrt(n)
    return np.array([ci0, ci1])


def sumStat_startPeakEnd(y, exceptVal):

    ''' return start, peak, end, i.e. how much first, max, and final value
    stick out relative to the median.
    Args:
    y (np.array)
    exceptVal (float)
    Returns:
    start (float or array - one value per y-column) initValue/median
    peak (float or array) maxValue/median
    end (float or array) finalValue/median
    '''

    if len(y) == 0:
        return exceptVal, exceptVal, exceptVal

    # 1-dim
    if len(y.shape) == 1:
        b = np.median(y)
               
        return y[0]-b, np.max(y)-b, y[-1]-b

    # 2-dim: over columns
    else:
        starts, peaks, ends = ea(), ea(), ea()
        for i in idx(y[0]):
            b = np.median(y[:, i])
            starts = np.append(starts, y[0, i]-b)
            peaks = np.append(peaks, np.max(y[:, i])-b)
            ends = np.append(ends, y[-1, i]-b)
            
        return starts, peaks, ends

    
def sumStat_startPeakEnd_deprec(y, exceptVal):

    # not used anymore, not robust, led to very high values for b near 0
    
    ''' return start, peak, end, i.e. how much first, max, and final value
    stick out relative to the median. y is normalized to [0 1] to ensure
    positive values only.
    Args:
    y (np.array)
    exceptVal (float)
    Returns:
    start (float or array - one value per y-column) initValue/median
    peak (float or array) maxValue/median
    end (float or array) finalValue/median
    '''
    
    if len(y) == 0:
        return exceptVal, exceptVal, exceptVal

    # 1-dim
    if len(y.shape) == 1:
        # normalize to [0, 1] to ensure positive values only
        yn, b = spe_nrm_bv(y)
        if b == 0:
            return exceptVal, exceptVal, exceptVal

        #!x
        if np.max([yn[0]/b, np.max(yn)/b, yn[-1]/b]) > 10:
            print("y:", y)
            print("yn:", yn)
            print("b:", b)
            print("s:", yn[0]/b, "\np:", np.max(yn)/b, "\ne:", yn[-1]/b)
            input()
        #!x
        
        return yn[0]/b, np.max(yn)/b, yn[-1]/b
    
    # 2-dim: over columns
    else:
        starts, peaks, ends = ea(), ea(), ea()
        for i in idx(y[0]):
            yn, b = spe_nrm_bv(y[:, i])
            if b == 0:
                s, p, e = exceptVal, exceptVal, exceptVal
            else:
                s, p, e = yn[0]/b, np.max(yn)/b, yn[-1]/b      
            starts = np.append(starts, s)
            peaks = np.append(peaks, p)
            ends = np.append(ends, e)
            
        return starts, peaks, ends
        
            

def spe_nrm_bv_deprec(y):
    ''' normalize y and robust base value extraction 
    Args:
    y: (np.array)
    Returns:
    yn (np.array) y range-normalized to [0, 1]
    b (float) base value
    '''
    
    yn = nrm_vec(y)
    b = np.median(yn)
    if b == 0:
        yp = yn[yn>0]
        if len(yp) == 0:
            return yn, b
        b = np.median(yp)

    return yn, b
            
                

    

def sumStat_slopeDiff(y, typ, exceptVal):

    if len(y) == 0:
        return exceptVal, exceptVal

    x = nrm_vec(np.arange(len(y)))

    # 1-dim array
    if len(y.shape) == 1:
        return sssd(x, y, typ)
    # 2-dim array
    slopes, diffs = ea(), ea()
    for i in idx(y[0]):
        s, d = sssd(x, y[:, i], typ)
        slopes = np.append(slopes, s)
        diffs = np.append(diffs, d)

    return slopes, diffs


def sssd(x, y, typ):
    ''' returns trend (slope of linear fir) and diff 2nd-1st half
    for time-value seq x,y. y values are minmax normalized before '''

    if len(y) == 1:
        return 0, 0

    # slope
    yn = nrm_vec(y)
    c = myPolyfit(x, yn, 1)
    # diff
    mi = int(len(yn)/2)
    y_left = yn[0:mi]
    y_right = yn[mi:len(yn)]
    if typ == "standard":
        diff = np.mean(y_left) - np.mean(y_right)
    else:
        diff = np.median(y_left) - np.median(y_right)

    # print("x", x,"\ny",yn,"\nmm",np.min(y),np.max(y),"\nc",c,"\nd",diff)
    # stopgo()

    return c[0], diff


def nonparam_description(y, p_skew=[10, 50, 90], p_kurt=[10, 25, 75, 90],
                         exceptVal=np.nan):
    ''' non-parametric distribution description
    skewness and kurtosis based on quantiles
    cf https://blogs.sas.com/content/iml/2017/07/19/quantile-skewness.html
    https://stats.stackexchange.com/questions/178987/...
    interquantile-based-kurtosis-measure
    Args:
      y: array-like
      p_skew: 3 skewness percentiles
      p_kurt: 4 kurtosis percentiles
      exceptVal
    Returns
      median, iqr, median_rms_from_median, skewness, kurtosis,
      range(5-95), range(10-90)
    '''

    # median, iqr
    myMed = np.median(y, 0)
    myIqr = sst.iqr(y, 0)

    # variance
    if ndim(y) == 1:
        myVar = np.median((y-myMed)**2)
    else:
        myVar = np.array([])
        for i in range(y.shape[1]):
            myVar = np.append(myVar, np.median((y[:, i]-myMed[i])**2))

    # range
    pp = np.percentile(y, [5, 10, 90, 95], 0)
    range5, range10 = pp[3]-pp[0], pp[2]-pp[1]

    # skewness
    p1, p2, p3 = np.percentile(y, p_skew, 0)
    if ndim(y) == 1:
        if p3 == p1:
            skew = exceptVal
        else:
            skew = ((p3-p2)-(p2-p1))/(p3-p1)
    else:
        # item-wise to catch p3==p1
        #    (otherwise above formula could be applied)
        skew = ea()
        for i in idx(p3):
            if p3[i] == p1[i]:
                skew = np.append(skew, 0)
            else:
                skew = np.append(
                    skew, ((p3[i]-p2[i])-(p2[i]-p1[i]))/(p3[i]-p1[i]))

    # kurtosis
    p1, p2, p3, p4 = np.percentile(y, p_kurt, 0)
    if ndim(y) == 1:
        if p3 == p2:
            kurt = exceptVal
        else:
            kurt = (p4-p1)/(p3-p2)
    else:
        kurt = ea()
        for i in idx(p3):
            if p3[i] == p2[i]:
                kurt = np.append(kurt, exceptVal)
            else:
                kurt = np.append(kurt, (p4[i]-p1[i])/(p3[i]-p2[i]))
    return myMed, myIqr, myVar, skew, kurt, range5, range10


def dist_eucl(x, y, w=[]):
    r''' (weighted) euclidean distance between x and y
    Args:
    x: (np.array)
    y: (np.array)
    w: (np.array) of weights
    Returns:
    d: (float) Euclidean distance of x and y
    '''
    if len(w) == 0:
        w = np.ones(len(x))
    q = x-y
    return np.sqrt((w*q*q).sum())


def myPolyfit(x, y, o=1):
    r''' robust wrapper around polyfit to
    capture too short inputs
    Args:
    x: (np.array)
    y: (np.array)
    o: (int) <1> order
    Returns:
    c: (np.array) of coefficients (highest order first)
    '''

    if len(x) == 0:
        return np.zeros(o+1)
    if len(x) <= o:
        return push(np.zeros(o), np.mean(y))
    return np.polyfit(x, y, o)


def nrm_vec(x, opt={}):
    r''' normalizes vector x according to opt[mtd|rng|max|min]
    mtd: 'minmax'|'zscore'|'std'
    'minmax' - normalize to opt.rng
    'zscore' - z-transform
    'std' - divided by std (whitening)
    Args:
    x: (np.array)
    opt: (dict) 'mtd'|'rng'|'max'|'min'
    Returns:
    x: (np.array) normalized
    '''
    opt = opt_default(opt, {"mtd": "minmax", "rng": [0, 1]})
    if opt['mtd'] == 'minmax':
        r = opt['rng']
        if 'max' in opt:
            ma = opt['max']
        else:
            ma = max(x)
        if 'min' in opt:
            mi = opt['min']
        else:
            mi = min(x)
        if ma > mi:
            x = (x-mi)/(ma-mi)
            x = r[0] + x*(r[1]-r[0])
    elif opt['mtd'] == 'zscore':
        x = sst.zscore(x)
    elif opt['mtd'] == 'std':
        x = sc.whiten(x)
    return x


def nrm(x, opt):
    r''' normalizes scalar to range opt.min|max set to opt.rng
    supports minmax only
    Args:
    x: (float) value to be normalized
    opt: (dict) specifying min, max, and range (rng)
    Returns:
    x: (float) normalized
    '''
    if opt['mtd'] == 'minmax':
        mi = opt['min']
        ma = opt['max']
        r = opt['rng']
        if ma > mi:
            x = (x-mi)/(ma-mi)
            x = r[0] + x*(r[1]-r[0])
    return x


def concat_df_from_dir(pth, ext="pickle"):
    ''' concatenates df content of alphanum-sorted
    files with extension ext into joint df
    Args:
    pth: (str) directory
    ext: (str) extension "csv", "pkl", "pickle"
    Returns:
    df: (pd.dataframe) concatenated
    '''
    df = None
    for f in file_collector(pth, ext):
        dfc = read_wrapper(f, ext)
        if df is None:
            df = cp.deepcopy(dfc)
        else:
            df = pd.concat([df, dfc])
    return df


def myPlot(xin=[], yin=[], opt={}):
    r''' plots x against y(s)
    Args:
    x: array or dict of x values
    y: array or dict of y values
    opt: <{}>
      bw: boolean; False; black-white
      nrm_x: boolean; False; minmax normalize all x values
      nrm_y: boolean; False; minmax normalize all y values
      ls: dict; {}; linespecs (e.g. '-k'), keys as in y
      lw: dict; {}; linewidths, keys as in y
         ls and lw for y as dict input
      legend_order; []; order of keys in x and y to appear in legend
      legend_lab; {}; mapping of keys to labels in dict
    Returns:
    True
    Comments:
    if y is dict:
      myKey1 -> [myYvals]
      myKey2 -> [myYvals] ...
    if x is array: same x values for all y.myKey*
                   if empty, indices 1:len(y) taken
    if x is dict: same keys as for y
    opt[ls|lw] can only be used if y is passed as dict
    '''

    opt = opt_default(opt, {'nrm_x': False,
                            'nrm_y': False,
                            'bw': False,
                            'ls': {},
                            'lw': {},
                            'legend_order': [],
                            'legend_lab': {},
                            'legend_loc': 'best',
                            'fs_legend': 40,
                            'fs': (25, 15),
                            'fs_ylab': 20,
                            'fs_xlab': 20,
                            'fs_title': 30,
                            'title': '',
                            'xlab': '',
                            'ylab': ''})
    xin, yin = cp.deepcopy(xin), cp.deepcopy(yin)
    # uniform dict input (default key 'y')
    if type(yin) is dict:
        y = yin
    else:
        y = {'y': yin}
    if type(xin) is dict:
        x = xin
    else:
        x = {}
        for lab in y:
            xx = xin
            if len(xin) == 0:
                xx = np.arange(0, len(y[lab]))
            x[lab] = xx
    # minmax normalization
    nopt = {'mtd': 'minmax', 'rng': [0, 1]}
    if opt['nrm_x']:
        for lab in x:
            x[lab] = nrm_vec(x[lab], nopt)
    if opt['nrm_y']:
        for lab in y:
            y[lab] = nrm_vec(y[lab], nopt)
    # plotting
    fig = newfig(fs=opt["fs"])
    ax = fig.add_subplot(111)
    if len(opt["title"]) > 0:
        ax.set_title(opt["title"], fontsize=opt["fs_title"])
    if len(opt["xlab"]) > 0:
        ax.set_xlabel(opt["xlab"], fontsize=opt["fs_xlab"])
    if len(opt["ylab"]) > 0:
        ax.set_ylabel(opt["ylab"], fontsize=opt["fs_ylab"])

    # line specs/widths
    # defaults
    if opt['bw']:
        lsd = ['-k', '-k', '-k', '-k', '-k', '-k', '-k']
    else:
        lsd = ['-b', '-g', '-r', '-c', '-m', '-k', '-y']
    while len(lsd) < len(y.keys()):
        lsd.extend(lsd)
    lwd = 4
    i = 0

    leg = []
    if len(opt["legend_order"]) > 0:
        labelKeys = opt["legend_order"]
        for lk in labelKeys:
            leg.append(opt["legend_lab"][lk])
    else:
        labelKeys = sorted(y.keys())

    # plot per label
    for lab in labelKeys:
        if lab in opt['ls']:
            cls = opt['ls'][lab]
        else:
            cls = opt['ls'][lab] = lsd[i]
            i += 1
        if lab in opt['lw']:
            clw = opt['lw'][lab]
        else:
            clw = lwd
        plt.plot(x[lab], y[lab], cls, linewidth=clw)

    if len(leg) > 0:
        plt.legend(leg,
                   fontsize=opt['fs_legend'],
                   loc=opt['legend_loc'])

    plt.show()


def newfig(fs=()):
    r''' init new figure with onclick->next, keypress->exit
    figsize can be customized
    Args:
    fs tuple <()>
    Returns:
    figureHandle
    '''
    if len(fs) == 0:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=fs)
    _ = fig.canvas.mpl_connect('button_press_event', fig_next)
    _ = fig.canvas.mpl_connect('key_press_event', fig_key)
    return fig


def fig_next(event):
    plt.close()


def fig_key(event):
    sys.exit()


# files, directories

def rm_file(x):
    ''' remove file x '''
    if is_file(x):
        os.remove(x)
        return True
    return False


def rm_dir(x):
    ''' remove directory x '''
    if is_dir(x):
        sh.rmtree(x)
        return True
    return False


def is_file(x):
    ''' returns true if x is file, else false '''
    if os.path.isfile(x):
        return True
    return False


def is_dir(x):
    ''' returns true if x is dir, else false '''
    if os.path.isdir(x):
        return True
    return False


def is_mac_dir(x):
    ''' returns true if x is mac index dir '''
    if re.search("__MACOSX", x):
        return True
    return False


def make_dir_rec(x):
    ''' recursively create folders/subfolders '''
    os.makedirs(x, exist_ok=True)


def make_dir(x, purge=False):
    ''' create directory '''
    if purge and is_dir(x):
        sh.rmtree(x)
    if not is_dir(x):
        os.mkdir(x)


def cp_file(x, y):
    ''' copy file x to y (y needs to be file name, too) '''
    if is_file(x):
        sh.copyfile(x, y)
        return True
    return False


def mv_file(x, y):
    ''' copy file x to y (y needs to be file name, too) '''
    if is_file(x):
        sh.move(x, y)
        return True
    return False


def args2opt(args, dflt=None):
    r"""wrapper for flexible command line vs embedded call of some function
    Args:
      args: (dict) from command line by parser module
            needs to contain the key "config" that has as value either
            the name of a json config file (string) or a option dict.
            The former is useful for command line calls, the latter for
            embedded calls.
      dflt: (string) of default json file
    Returns:
       opt: (dict) json file content or value of config key
    """
    if (('config' in args) and (args['config'] is not None)):
        if type(args['config']) is dict:
            opt = cp.deepcopy(args['config'])
        else:
            opt = read_wrapper(args['config'], "json")
        return opt
    if dflt is not None:
        return read_wrapper(dflt, "json")
    sys.exit("cannot return option dict")


def is_type(x):
    '''quick hack to return variable type str|int|float|list (incl np array)'''
    tx = str(type(x))
    for t in ['str', 'int', 'float', 'list']:
        if re.search(t, tx):
            return t
    if re.search('ndarray', tx):
        return 'list'
    return 'unk'


def read_wrapper(f, frm, opt={}):
    r"""wrapper around several reading functions
    Args:
      f: (string) file name
      frm: (string) format "list"|"json"|"string"|"csv"|"warriner"
      opt: (dict) with keys
        "sep": column separator; relevant for frm=csv only
        ... more to come with other output formats
    Returns:
      variable type depends on frm:
      "list": read text file into list of row strings
      "json": read json file content into dict
      "str": read text file into string
      "csv", "warriner": read csv table with header into dictionary
    """
    opt = opt_default(opt, {"sep": ","})
    if frm in ["pickle", "pkl"]:
        m = "rb"
    else:
        m = "r"

    f = replHome(f)

    if frm == "list":
        with open(f, encoding='utf-8') as h:
            lst = [x.strip() for x in h]
            h.close()
        return lst
    with open(f, m) as h:
        if frm == "json":
            return json.load(h)
        if frm == "string":
            return h.read()
        if frm in ["pickle", "pkl"]:
            try:
                return pickle.load(h)
            except:
                print("{} is probably empty. Returning None.".format(f))
                return None
        if frm == "csv":
            o = pd.read_csv(f, sep=opt['sep'])  # ,engine='python')
            return o.to_dict('list')
        if frm == "warriner":
            o = pd.read_csv(f, sep=opt['sep'], dtype={"Word": str})
            return o.to_dict('list')

    sys.exit("Format {} unknown. Cannot read file {}".format(frm, f))


def write_wrapper(v, f, frm, opt={}):
    r"""wrapper around several writing functions
    Args:
      v: (variable type) some variable to be outputted
      f: (string) file name
      frm: (string) format "csv"|"pickle"|"list"; "csv" requires a dict
      opt: (dict) with keys
        "sep": (string) <","> column separator; relevant for frm=csv only
        "header": (boolean) <True> +/- output csv header

    Returns: --
    File output into F
    """
    opt = opt_default(opt, {"sep": ",", "header": True})

    f = replHome(f)

    if frm in ['pickle', 'pkl']:
        m = 'wb'
    else:
        m = 'w'

    if frm in ['pickle', 'pkl']:
        with open(f, m) as h:
            pickle.dump(v, h)
    elif frm == 'csv':
        pd.DataFrame(v).to_csv("{}".format(f), na_rep='NA', index_label=False,
                               index=False, sep=opt['sep'],
                               header=opt['header'])
    elif frm == 'list':
        x = "\n".join(v)
        x += "\n"
        with open(f, m) as h:
            h.write(x)
    else:
        sys.exit("Format {} unknown. Cannot write to file {}".format(frm, f))


def ext_false(d, k):
    r"""extended False of a dict key
    Arguments:
      d: (dict)
      k: (str) key
    Returns:
      boolean. True if (key not in dict) OR (dict.key is False)
    """
    if ((k not in d) or (not d[k])):
        return True
    return False


def numkeys(x):
    ''' returns sorted list of numeric (more general: same data-type) keys '''
    return sorted(list(x.keys()))


def sorted_keys(x):
    return sorted(list(x.keys()))


def inf2nan(x):
    x = np.asarray(x)
    i = find(x, 'is', 'inf')
    if len(i) > 0:
        x[i] = np.nan
    return x


def ext_true(d, k):
    r"""extended True of a dict key
    Arguments:
      d: (dict)
      k: (str) key
    Returns:
      boolean. True if (key in dict) AND (dict.key is True)
    """
    if (k in d) and d[k]:
        return True
    return False


def idx(lst):
    r"""returns index iterable of list L"""
    return range(len(lst))


def opt_default(c, d):
    r"""recursively adds default fields of dict d to dict c
      if not yet specified in c
    Arguments:
      c: (dict) someDict
      d: (dict) defaultDict
    Returns:
      c: (dict) mergedDict; defaults added to c
    """
    for x in d:
        if x not in c:
            c[x] = d[x]
        elif type(c[x]) is dict:
            c[x] = opt_default(c[x], d[x])
    return c


def ea():
    r""" returns empty np.array """
    return np.array([])


def stopgo(s=""):
    _ = input(s)
    return True


def replHome(f):
    r''' replaces $HOME placeholder by home dir
    in file or dir name strings '''
    f = re.sub(r"^\$HOME", os.getenv("HOME"), f)
    return f


def intersect(a, b):
    r''' returns intersection list of two 1-dim lists'''
    return list(set(a) & set(b))


def find(x_in, op, v):
    r"""wrapper around FINF functionality
    Arguments:
      x_in: 1-dim array-line, sumeric
      op: (string)
      v: (float or string {"max", "min", "nan", "inf", "infimum",
         "supremum"}) what is searched for
    Returns:
      xi: 1-dim array of hit indices
        or
          float (for is infimum|supremum)
    """
    x = np.asarray(x_in)
    if op == '==':
        xi = np.asarray((x == v).nonzero())[0, :]
    elif op == '!=':
        xi = np.asarray((x != v).nonzero())[0, :]
    elif op == '>':
        xi = np.asarray((x > v).nonzero())[0, :]
    elif op == '>=':
        xi = np.asarray((x >= v).nonzero())[0, :]
    elif op == '<':
        xi = np.asarray((x < v).nonzero())[0, :]
    elif op == '<=':
        xi = np.asarray((x <= v).nonzero())[0, :]
    elif (op == 'is' and v == 'nan'):
        xi = np.asarray(np.isnan(x).nonzero())[0, :]
    elif (op == 'is' and v == 'finite'):
        xi = np.asarray(np.isfinite(x).nonzero())[0, :]
    elif (op == 'is' and v == 'inf'):
        xi = np.asarray(np.isinf(x).nonzero())[0, :]
    elif (op == 'is' and v == 'max'):
        return find(x, '==', np.max(x))
    elif (op == 'is' and v == 'min'):
        return find(x, '==', np.min(x))
    elif (op == 'isinfimum'):
        xi = np.asarray((x < v).nonzero())[0, :]
        if len(xi) == 0:
            return np.nan
        return int(xi[-1])
    elif (op == 'issupremum'):
        xi = np.asarray((x > v).nonzero())[0, :]
        if len(xi) == 0:
            return np.nan
        return int(xi[0])
    return xi.astype(int)


def push(x, y, a=0):
    r''' pushes 1 additional element y to array x (default: row-wise)
    if x is not empty, i.e. not []: yDim must be xDim-1, e.g.
    if x 1-dim: y must be scalar
    if x 2-dim: y must 1-dim
    if x is empty, i.e. [], the dimension of the output is yDim+1
    Differences to np.append:
    append flattens arrays if dimension of x,y differ, push does not
    REMARK: cmat() might be more appropriate if 2-dim is to be returned
    Args:
       x: (array) can be empty
       y: (array) (if x not empty, then one dimension less than x)
       a: (int) {0,1}, axis (0: push row, 1: push column)
    Returns:
       [x y]
    '''
    if (listType(y) and len(y) == 0):
        return x
    if len(x) == 0:
        return np.array([y])
    return np.concatenate((x, [y]), axis=a)


def file_collector(d, e=''):
    r''' returns files incl full path as list (recursive dir walk)
    Args:
      d: (string) directory; or (dict) containing fields 'dir' and 'ext'
      e: (string), extension; only used if type(d) is str
    Returns:
      ff: (list) of fullPath-filenames
    '''

    d = replHome(d)

    if type(d) is dict:
        pth = d['dir']
        ext = d['ext']
    else:
        pth = d
        ext = e

    ff = []
    for root, dirs, files in os.walk(pth):
        files.sort()
        for f in files:
            if f.endswith(ext):
                ff.append(os.path.join(root, f))
    return sorted(ff)


def str_standard(x):
    r''' standardizes strings:
    removes initial, final, and multiple whitespaces
    Args:
      x: (string)
    Returns:
      x: (string) standardized
    '''
    x = re.sub(r'^\s+', '', x)
    x = re.sub(r'\s+$', '', x)
    x = re.sub(r'\s+', ' ', x)
    return x


def listType(y):
    r''' returns True if input is numpy array or list; else False
    Args:
      someVariable
    Returns:
      boolean
    '''
    if (type(y) == np.ndarray or type(y) == list):
        return True
    return False


def uniq(x):
    r''' returns sorted + unique element list
    Args:
      x: (list)
    Returns:
      x: (list) sorted and uniq
    '''
    return sorted(list(set(x)))


def dist2sim(x):
    r''' transforms distances to [0 1] normalized similarity
    Args:
      x: (arraylike) of distance values
    Returns:
      x: (np.array) of similarity values
    '''

    x = np.asarray(x)
    return 1 - x/np.sum(x)


def stm(f):
    r''' returns file name stem
    Args:
      f: (string) pth/stm.ext
    Returns:
      s: (string) stm
    '''
    s = os.path.splitext(os.path.basename(f))[0]
    return s


def dfe(x):
    ''' returns directory, file stem and extension '''
    dd = os.path.split(x)
    d = dd[0]
    s = os.path.splitext(os.path.basename(dd[1]))
    e = s[1]
    e = re.sub(r'\.', '', e)
    return d, s[0], e


def idx_a(lng, sts=1):
    ''' returns index array for vector of length len() lng
    thus highest idx is lng-1'''
    return np.arange(0, lng, sts)


def idx_seg(on, off, sts=1):
    ''' returns index array between on and off (both included) '''
    return np.arange(on, off+1, sts)


def wavread(f):
    ''' reads wav file, returns signal and sample rate '''
    fs, s = sio.read(f)
    s = wav_int2float(s)
    s = s-np.mean(s)
    return s, fs


def wav_int2float(s):
    ''' maps integers from -32768 to 32767 to interval [-1 1] '''
    s = s/32768
    s[find(s, '<', -1)] = -1
    s[find(s, '>', 1)] = 1
    return s


def seq_windowing(s):
    r''' vectorized version of windowing
    Args:
       s (dict)
         "win": (int) window length in samples
         "rng": (2-element list) [on, off] range of indices to be windowed
         "align: (string) <"center">|"left"|"right"
    # Returns:
    [[on off] ...]
    '''
    s = opt_default(s, {"align": "center"})
    if s["align"] == "center":
        vecwin = np.vectorize(windowing)
    elif s["align"] == "right":
        vecwin = np.vectorize(windowing_rightAligned)
    elif s["align"] == "left":
        vecwin = np.vectorize(windowing_leftAligned)
    r = s['rng']
    ww = np.asarray(vecwin(range(r[0], r[1]), s))
    return ww.T


def windowing_idx1(i, s):
    on, off = windowing(i, s)
    return np.arange(on, off+1, 1)


def windowing(i, s):
    r''' window of length wl on and offset around single index in range [on off]
    vectorized version: seq_windowing
    Args:
       i: (int) current index
       s: (dict)
          "win": (int) window length
          "rng": (2-element list) [on, off] range of indices to be windowed
    Returns:
       on: (int) start index of window around i
       off: (int) end index of window around i
    '''
    # half window
    wl = max([1, math.floor(s['win']/2)])
    r = s['rng']
    on = max([r[0], i-wl])
    off = min([i+wl, r[1]])
    # extend window
    d = (2*wl-1) - (off-on)
    if d > 0:
        if on > r[0]:
            on = max([r[0], on-d])
        elif off < r[1]:
            off = min([off+d, r[1]])
    return on, off

# window around each sample so that it is at the right end (no look-ahead)


def windowing_rightAligned(i, s):
    wl, r = int(s['win']), s['rng']
    on = max([r[0], i-wl])
    off = min([i, r[1]])
    # extend window (left only)
    d = wl - (off-on)
    if d > 0:
        if on > r[0]:
            on = max([r[0], on-d])
    # relax 0,0 case (zero length win)
    if off == on:
        off += 1
    return on, off

# window around each sample so that it is at the left end (no looking back)


def windowing_leftAligned(i, s):
    wl, r = int(s['win']), s['rng']
    on = max([r[0], i])
    off = min([i+wl, r[1]])
    # extend window (right only)
    d = wl - (off-on)
    if d > 0:
        if off < r[1]:
            off = min([r[1], off+d])
    # relax -1, -1 case (zero length win)
    if on == off:
        on -= 1
    return on, off


def ncol(x):
    ''' returns number of columns of numpy array '''
    if np.ndim(x) == 1:
        return 1
    return len(x[0, :])


# violin plots
# IN:
#   df: pandas dataframe
#      (df = pd.read_csv("*.csv"))
#   opt:
#     "x", "y", "hue" as arguments in violinplot
#     + facultatively "order" + "hue_order" (need both to be present or absent)
#     "show": show <False>
#     "save" if to be saved <"">
#     "title" <"">
def violin_plot(df, opt):
    # print("X", opt["x"])
    # print("Y", opt["y"])
    # print(df[opt["x"]])
    # print(df[opt["y"]])
    # print(df[opt["hue"]])
    # stopgo()
    opt = opt_default(opt, {"show": False, "save": "", "title": "",
                            "fontsize": 14, "labelsize": 12})
    sns.set(style="whitegrid")

    if "palette" not in opt:
        pal = None
    else:
        pal = opt["palette"]
    
    if "hue" in opt:
        if "order" in opt:
            ax = sns.violinplot(x=opt["x"], y=opt["y"], hue=opt["hue"],
                                order=opt["order"], hue_order=opt["hue_order"],
                                palette=pal, data=df)
        else:
            ax = sns.violinplot(x=opt["x"], y=opt["y"], palette=pal,
                                hue=opt["hue"], data=df)
    else:
        if "order" in opt:
            ax = sns.violinplot(x=opt["x"], y=opt["y"], palette=pal,
                                order=opt["order"], data=df, cut=0)
            #ax = sns.boxplot(x=opt["x"], y=opt["y"], data=df)
            
        else:
            ax = sns.violinplot(x=opt["x"], y=opt["y"], palette=pal, data=df)

    if len(opt["title"]) > 0:
        plt.title(opt["title"], fontsize=opt["fontsize"])

    if opt["ylim"] is not None:
        ax.set_ylim(opt["ylim"])
        
    ax.xaxis.label.set_size(opt["fontsize"])
    ax.yaxis.label.set_size(opt["fontsize"])    
    ax.tick_params(axis='both', which='major',
                   labelsize=opt["labelsize"])
            
        
    if opt["show"]:
        plt.show()

    if len(opt["save"]) > 0:
        fig = ax.get_figure()
        fig.savefig(opt["save"], dpi=300)
    plt.clf()


# regression plot (same input as to violin_plot())
def regress_plot(df, opt):
    opt = opt_default(opt, {"show": False, "save": "", "title": "", "xlab": "",
                            "ylab": "", "fontsize": 14, "labelsize": 12})
    ax = sns.regplot(x=opt["x"], y=opt["y"], data=df, fit_reg=True,
                     robust=True, x_ci="ci", scatter=True)

    if len(opt["title"]) > 0:
        plt.title(opt["title"], fontsize=opt["fontsize"])

    if len(opt["xlab"]) > 0:
        plt.xlabel(opt["xlab"], fontsize=opt["fontsize"])

    if len(opt["ylab"]) > 0:
        plt.ylabel(opt["ylab"], fontsize=opt["fontsize"])

    if opt["ylim"] is not None:
        ax.set_ylim(opt["ylim"])
        
    ax.tick_params(axis='both', which='major',
                   labelsize=opt["labelsize"])
        
    if opt["show"]:
        plt.show()

    if len(opt["save"]) > 0:
        fig = ax.get_figure()
        fig.savefig(opt["save"], dpi=300)
    plt.clf()


def list2prob(x, pm=None):
    r''' wrapper around counter() and count2prob()
    alternatively to be called with unigram language model
    -> assigns MLE without smoothing
    Args:
    x - list or set of strings
    pm <None> unigram language model
    Returns:
    p - dict
      myType -> myProb
    '''

    if type(pm) is None:
        c, n = counter(x)
        return count2prob(c, n)
    p = {}
    for z in x:
        p[z] = prob(z, pm)
    return p


def prob(x, pm):
    if x in pm:
        return pm[x]
    return 0


def count2prob(c, n):
    ''' MLE probs from counter() output '''
    p = {}
    for x in c:
        p[x] = c[x]/n
    return p


def counter(x):
    c = {}
    for y in x:
        if y not in c:
            c[y] = 1
        else:
            c[y] += 1
    return c, len(x)


def trunc2(n):
    ''' truncates float n to precision 2 '''
    return float('%.2f' % (n))


def trunc4(n):
    ''' truncates float n to precision 4 '''
    return float('%.4f' % (n))


def trunc6(n):
    ''' truncates float n to precision 6 '''
    return float('%.6f' % (n))


def mae(x, y=[]):
    ''' mean absoulte error '''
    if len(y) == 0:
        y = np.zeros(len(x))
    x = np.asarray(x)
    return np.mean(abs(x-y))


#############################

# statistics ######################################################

def mld_stats(df,y=[],yc=[],nSyl=0,alpha=0.05):
    ''' statistics for all mld-s in df. Correlated with numeric (y) and
    grouped by categorical (yc) targets. For yc only 2-10 levels are
    currently supported (hard-code more if needed).
    Args:
    df: dataframe with mid level feature columns
    y: numeric target (correlation)
    yc: class target (mannwhitney, kruskalwallis)
    nSyl: minimum number of syllables in vad segment (only used as filter if hld_nSyl is df column)
    Returns:
    stats: (dict) assigning to each mld
          if yc:
             "stat_class": statistics of Mann-Whitney resp Kruskal-Wallis
             "p_class": p value
             "m_class": class mean values (order as in class_levels)
             "levels": yc-levels alphanumerically sorted
             "posthoc": pd.Dataframe. Sample idx starts with 1
                  e.g. z[1][2]: p value for comparing s[0] and s[1]
             "cohen_d": effect size 2-dim np.array [i, j] between levels i and j
             "cohen_d_level": effect size level from 0 (None) to 6 (Huge) calculated from
                  np.max(cohen_d)
          if y:
            "stat_corr": pearson r
            "p_corr": p-value
    col: (dict) with keys "class" and "corr" to each of which a list of column names
        is assigned for which test on yc was significant on alpha level
    '''


    df, flt = nsyl_filter(df, {"y": y, "yc": yc}, nSyl)
    y, yc = flt["y"], flt["yc"]
    
    # imputation
    df.fillna(0, inplace=True)
    
    # targets
    levels = sorted(set(yc))
    yc = np.asarray(yc)

    stats, col, col_corr, n_all, n_sig, n_sig_corr = {}, [], [], 0, 0, 0
    # over features
    for c in df.columns:
        n_all += 1
        stat_class, m_class, p_class, stat_corr, p_corr, ph = None, None, None, None, None, None
        efs, efs_level = None, None
        v = df[c].to_numpy()
        
        ## class
        if len(yc)>0:
            if len(yc) != len(v):
                sys.exit("stats: length of variable and grouping vectors differ! {} vs {}. Exit.".format(len(v),len(yc)))
            s = None
            sx = {}
            for i in range(len(levels)):
                if s:
                    s.append(v[yc==levels[i]])
                else:
                    s = [v[yc==levels[i]]]
                sx[i] = v[yc==levels[i]]
            stat_class, p_class, m_class = stattest_wrapper(sx)
            if np.isnan(stat_class):
                print("stat error: feature {} cannot be tested".format(c))
            if p_class <= alpha:
                col.append(c)
                n_sig += 1
                # posthoc
                ph = posthoc(s)
            else:
                ph = None
            efs, efs_level = cohen_d(s)
                
        ## correlation
        if len(y)>0:
            if len(y) != len(v):
                sys.exit("stats: length of variable and target vectors differ! {} vs {}. Exit.".format(len(v),len(y)))
            stat_corr, p_corr = st.pearsonr(v, y)
            if p_corr <= 0.05:
                col_corr.append(c)
                n_sig_corr += 1
            
        stats[c] = {"stat_class": stat_class, "p_class": p_class, "m_class": m_class,
                    "levels": levels, "posthoc": ph, "cohen_d": efs, "cohen_d_level": efs_level,
                    "stat_corr": stat_corr, "p_corr": p_corr}

    if len(yc)>0:
        print("significant features (classification):", n_sig, "out of", n_all)
        
    if len(y)>0:
        n_best = np.min([30,n_sig_corr])
        print("significant features (correlation):", n_sig_corr, "out of", n_all)
        print("{} best correlations:".format(n_best))
        cc = sorted(stats.keys())
        for i in range(len(cc)-1):
            for j in range(1, len(cc)):
                if stats[cc[j]]["stat_corr"] > stats[cc[i]]["stat_corr"]:
                    b = cc[j]
                    cc[j] = cc[i]
                    cc[i] = b
        for i in range(n_best):
            print("\t", cc[i], stats[cc[i]]["stat_corr"])

    return stats, {"class": col, "corr": col_corr}

def stattest_wrapper(s):
    ''' returns stat, p, m (sample median vector) '''
    if type(s) is dict:
        num_levels = len(sorted(s.keys()))
    else:
        num_levels = len(s)
        
    m = None
    try:
        if num_levels == 2:
            stat, p = st.mannwhitneyu(s[0], s[1])
        elif num_levels == 3:
            stat, p = st.kruskal(s[0], s[1], s[2])
        elif num_levels == 4:
            stat, p = st.kruskal(s[0], s[1], s[2], s[3])
        elif num_levels == 5:
            stat, p = st.kruskal(s[0], s[1], s[2], s[3], s[4])  
        elif num_levels == 6:
            stat, p = st.kruskal(s[0], s[1], s[2], s[3], s[4], s[5])
        elif num_levels == 7:
            stat, p = st.kruskal(s[0], s[1], s[2], s[3], s[4], s[5], s[6])
        elif num_levels == 8:
            stat, p = st.kruskal(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7])
        elif num_levels == 9:
            stat, p = st.kruskal(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8])
        elif num_levels == 10:
            stat, p = st.kruskal(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9])
        else:
            print("number of target levels >10. Returning nan, 1")
            stat, p, m = np.nan, 1, np.nan
        if m is None:
            m = np.array([])
            for x in s:
                m = np.append(m, np.median(s[x]))
    except:
        stat, p, m = np.nan, 1, np.nan
    return stat, p, m


def cohen_d(s):

    ''' calculates effect sizes (abs Cohen's d) between pairs of samples.
    Interpretation: Very small 0.01, Small 0.20, Medium 0.50, Large 0.80,
    Very large 1.20, Huge 2.0 (see https://en.wikipedia.org/wiki/Effect_size)
    Args:
    s: 2-dim list of samples
    Returns:
    d: 2-dim np.array [i, j]==[j, i]==effect size between s[i] and s[j]
    d_level: cohen d level:
      0 - none
      1 - very_small
      2 - small
      3 - medium
      4 - large
      5 - very_large
      6 - huge
    '''

    n = len(s)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i==j or d[i, j] != 0:
                continue
            x1, x2 = s[i], s[j]
            n1, n2 = len(x1), len(x2)
            m1, m2 = np.mean(x1), np.mean(x2)
            v1, v2 = np.var(x1), np.var(x2)
            psd = np.sqrt(((n1-1)*v1 + (n2-1)*v2)/(n1+n2-2))
            if psd == 0 or np.isnan(psd):
                d[i, j] = 0
            else:
                d[i, j] = np.abs((m1-m2) / psd)
            

    m = np.max(d)
    if m < 0.01:
        d_level = 0
    elif m < 0.2:
        d_level = 1
    elif m < 0.5:
        d_level = 2
    elif m < 0.8:
        d_level = 3
    elif m < 1.2:
        d_level = 4
    elif m < 2:
        d_level = 5
    else:
        d_level = 6
    
    return d, d_level
            
    
def posthoc(s, method="dunn", p_adjust="fdr_bh"):
    ''' returns pd.Dataframe. Sample idx starts with 1
    e.g. z[1][2]: p value for comparing s[0] and s[1]
    '''
    
    z = None
    if method == "dunn":
        z = sp.posthoc_dunn(s, p_adjust=p_adjust)
    return z

def stats_summary_class(stats, cols, min_efs=3, verbose_level=2):
    ''' prints to STDOUT number of significant features and number of those
    with at least medium effect size, based on output od mld_stats.
    For classification tasks.
    Args:
    stats: dict returned by mld_stats(),
    cols: column dict returned by mld_stats()
    Returns:
    cols_efs (list of features from cols[class] with effect size >= min_efs)
    '''

    N, cols_efs = 0, []
    
    for feat in stats:
        
        N += 1

        if feat not in cols["class"]:
            continue
        
        if stats[feat]["cohen_d_level"] >= min_efs:
            cols_efs.append(feat)
            if verbose_level == 2:
                print(feat, "effect size level:",
                      stats[feat]["cohen_d_level"])

    if verbose_level >= 1:
        print("significant:", len(cols["class"]), "out of", N)
        print("sig + medium or higher effect size:", len(cols_efs), "out of", N)

    return cols_efs
    

def stats_compare(s1, s2):
    ''' compares 2 results from stats() whether significant results are the same
    and sample means have the same order 
    Returns:
    dict
      feat_hit: list of hit features (same sig + tendency)
      feat_tendency: list of same tendency features
      p_hit, p_tendency: corresp proportions to all features common among s1 and s2
    '''

    # n(allCommonFeat), feat(same sig+same order), feat(same order), feat(diff)
    n, feat_hit, feat_tendency, feat_diff = 0, [], [], []
    alpha = 0.05
    
    # over features
    for x in sorted(s1.keys()):
        if x not in s2.keys():
            continue
        n += 1

        #print("feat:", x)
        #print("train:", s1[x])
        #print("test:", s2[x])

        # same tendency
        i1 = np.argsort(s1[x]["m_class"])
        i2 = np.argsort(s2[x]["m_class"])
        if any(i1 != i2):
            same_tendency = False
        else:
            same_tendency = True

        if same_tendency:
            feat_tendency.append(x)
            
        # same significance (both < or > alpha)
        if (s1[x]["p_class"]-alpha)*(s1[x]["p_class"]-alpha)>0:
            if same_tendency:
                feat_hit.append(x)

    #print("hit features:", feat_hit)
    print("number of statistically robust features:", len(feat_hit)/n)
    print("number of tendency-robust features:", len(feat_tendency)/n)

    return {"hit": feat_hit,
            "tendency": feat_tendency,
            "p_hit": len(feat_hit)/n,
            "p_tendency": len(feat_tendency)/n}


def nsyl_filter(df, dep_vars=None, nSyl=0):
    ''' removes entries in df with num of syllables lower than nSyl
    Args:
    df: mld df
    dep_vars: None or dict. Each key contains dataframe (! key name needs to start with "df_" !)
        or listlike object. All objects are filtered like df
    nSyl: min number of syllables
    Returns:
    df: filtered
    ret_vars: same dict as dep_vars with filtered objects
    '''
    
    
    if (nSyl == 0 or ("hld_nSyl" not in df.columns)):
        return df, dep_vars

    # indices to remain
    ii = find(df["hld_nSyl"].to_numpy(),">=",nSyl)
    dfo = df.iloc[ii]

    if dep_vars is None:
        return dfo, None

    ret_vars = {}
    for x in dep_vars:
        if re.search("^df", x):
            if dep_vars[x] is None:
                ret_vars[x] = None
            else:
                ret_vars[x] = dep_vars[x].iloc[ii]
        else:
            if dep_vars[x] is None:
                ret_vars[x] = None
            elif len(dep_vars[x]) == 0:
                ret_vars[x] = dep_vars[x]
            else:
                ret_vars[x] = np.asarray(dep_vars[x])[ii]
    
    return dfo, ret_vars

# plotting #########################################

def mld_plot(typ, df, tar, col=[], odir="/tmp", xlab="tar", order=None,
             prefix=None, hue=None, hue_lab=None, hue_order=None,
             col_map=None, fontsize=14, labelsize=12, ylim=None,
             palette=None):
    ''' plotting all variables against numeric or categorical targets
    Args:
    typ: (str) "class", "regress" -> violin or regression plots
    df: (dataFrame) with features
    tar: (array-like) target values
    col: (array-like or dict) list of selected columns. If empty all are selected.
        if dict, col is assumed to be the output COL of mld_stats() where the columns are
        stored in col[typ]
    odir: (str) output directory for pngs
    xlab: (str) x-label
    order: (list or None) of xtick labels
    prefix: (str or None) file stem prefix
    hue: (array-like) hue variable values
    hue_lab: (str) hue-label
    col_map: (dict) column name mapping (maps ylabels to e.g. non-expert titles)
    fontsize, labelsize, ylim, palette: plotting params
    '''

    # imputation
    df.fillna(0,inplace=True)

    # targets, hue
    tar = np.asarray(tar)
    if hue is not None:
        hue = np.asarray(hue)
        if hue_order is None:
            hue_order = sorted(set(hue))
    
    # columns
    if type(col) is dict:
        col = cp.deepcopy(col[typ])
    elif len(col)==0:
        col = list(df.columns)
 
    for c in col:

        if type(col_map) is dict and c in col_map:
            title = col_map[c]
        else:
            title = c
        
        v = df[c].to_numpy()
        #io = outl_idx(v,{"m": "mean", "zi": False, "f": 2})
        io = outl_idx(v,{"m": "median", "zi": False, "f": 2})
        if type(io) is tuple:
            io = io[0]
        i = sorted(set(range(v.size)) - set(list(io)))
        if len(i) < 5:
            i = np.arange(0, len(tar))
        if hue is not None:
            dfc = pd.DataFrame.from_dict({xlab: tar[i],
                                          hue_lab: hue[i],
                                          c: v[i]})
        else:
            dfc = pd.DataFrame.from_dict({xlab: tar[i],
                                          c: v[i]})

        if prefix is not None:
            fo = "{}/{}_{}_{}.png".format(odir,prefix,c,typ)
        else:
            fo = "{}/{}_{}.png".format(odir,c,typ)
        
        if typ=="class":
            o = {"x": xlab, "y": c, "show": False,
                 "save": fo, "title": title,
                 "fontsize": fontsize,
                 "labelsize": labelsize,
                 "ylim": ylim,
                 "palette": palette}
            if order is not None:
                o["order"] = order
                o["hue_order"] = hue_order
            if hue is not None:
                o["hue"] = hue_lab
                
            violin_plot(dfc, o)
        else:
            regress_plot(dfc, {"x": xlab, "y": c, "show": False,
                               "save": fo, "title": title,
                               "fontsize": fontsize,
                               "labelsize": labelsize,
                               "ylim": ylim})


def outl_idx(y, opt={}):

    opt = opt_default(opt, {"zi": False,
                            "f": 2,
                            "m": "mean"})

    r''' marks outliers in arrayreturns idx of outliers
    Args:
    y: (np.array) of F0
    opt: (dict)
      'f': factor of min deviation
      'm': from 'mean', 'median' or 'fence'
            (mean: m +/- f*sd,
             median: med +/- f*iqr,
             fence: q1-f*iqr, q3+f*iqr)
      'zi': (boolean) - ignore zeros in m and outlier calculation
    Returns:
    io: (np.array) indices of outliers
    '''

    if opt['zi']:
        # i = (y!=0).nonzero()
        i = (y != 0 & np.isfinite(y)).nonzero()
    else:
        # i = range(np.size(y))
        i = (np.isfinite(y)).nonzero()
    if type(i) is tuple:
        i = i[0]

    f = opt['f']

    if np.size(i) == 0:
        return np.array([])

    # getting lower and upper boundary lb, ub
    if opt['m'] == 'mean':
        # mean +/- f*sd
        m = np.nanmean(y.take(i))
        r = np.nanstd(y.take(i))
        lb, ub = m-f*r, m+f*r
    else:
        m = np.nanmedian(y.take(i))
        q1, q3 = np.nanpercentile(y.take(i), [25, 75])
        r = q3 - q1
        if opt['m'] == 'median':
            # median +/- f*iqr
            lb, ub = m-f*r, m+f*r
        else:
            # Tukey's fences: q1-f*iqr , q3+f*iqr
            lb, ub = q1-f*r, q3+f*r
   # declaring also nan-s as outliers
    yy = cp.deepcopy(y)
    io_nan = (np.isnan(yy) | np.isinf(yy)).nonzero()
    if type(io_nan) is tuple:
        io_nan = io_nan[0]
    if len(io_nan) > 0:
        yy[io_nan] = ub+1

    if not opt['zi']:
        # io = (np.isfinite(y) & ((y>ub) | (y<lb))).nonzero()
        # io = (np.isnan(y) | np.isinf(y) | (y>ub) | (y<lb)).nonzero()
        io = ((yy > ub) | (yy < lb)).nonzero()
    else:
        # io = (np.isfinite(y) & ((y>0) & ((y>ub) | (y<lb)))).nonzero()
        # io = (np.isnan(y) | np.isinf(y) | ((y>0) & ((y>ub) | \
        # (y<lb)))).nonzero()
        io = ((yy > 0) & ((yy > ub) | (yy < lb))).nonzero()

    if type(io) is tuple:
        io = io[0]

    return io
