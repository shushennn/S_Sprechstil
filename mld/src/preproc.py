#!/usr/bin/env python3

import numpy as np
#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import myUtilHL as myl
import copy as cp
import sys
import pandas as pd


def preproc_f0(y, opt={}):
    r''' f0 preprocessing: outlier removal, linear zero interpolation '''
    y = np.asarray(y)
    # outlier removal
    # ignore zeros
    opt = myl.opt_default(opt, {'zi': True,
                                'f': 3,
                                'm': 'mean',
                                'st': True,
                                'bv': 1})
    io = outl_idx(y, opt)
    if np.size(io) > 0:
        y[io] = 0
    # linear interpolation
    y = linZeroInterp(y)

    # semitone conversion
    if opt["st"]:
        y = semton(y, opt["bv"])

    return y


def semton(y, b=1):
    yi = myl.find(y, '>', 0)
    y[yi] = 12*np.log2(y[yi]/b)
    return y


def st2hz(x):
    y = cp.copy(x)
    yi = myl.find(y, '>', 0)
    y[yi] = 2**(y[yi]/12)
    return y


def f0_nrm_pp(y, opt):
    # set outliers to zero
    io = outl_idx(y, opt)
    if np.size(io) > 0:
        y[io] = 0
    # collect only non-zero values
    return y[myl.find(y, ">", 0)]


def f0_feat(y):
    ''' returns dataframe of summary statistics robust + misc
    w/o slope and diff '''
    sumStat = myl.summary_stats(y, typ="robust", nanrep=0, exceptVal=0)
    sumStat_misc = myl.summary_stats(y, typ="misc", nanrep=0, exceptVal=0)

    for x in sumStat_misc:
        sumStat[x] = sumStat_misc[x]
    for x in ["slope", "diff"]:
        del sumStat[x]
    for x in sumStat:
        sumStat[x] = np.array([sumStat[x]])

    return pd.DataFrame(sumStat, columns=sorted(sumStat.keys()))


def f0_spk_nrm(y, tf, tt, mod):
    ''' speaker normalization of f0
    Args:
    y: f0 column content
    tf: frametime column content
    tt: [[on off]...] of speech chunks
    mod: f0 normalization model
    Returns:
    y: speaker-nrmd f0
    '''

    tf, y = np.asarray(tf), np.asarray(y)
    for t in tt:
        # print("### t:", tt)
        ti = myl.find_interval(tf, t)
        if len(ti) == 0:
            continue
        ys = y[ti]
        # y without outliers and zeros
        # print("in:", ys)
        yp = f0_nrm_pp(ys, {"m": "mean", "f": 3, "zi": True})
        if len(yp) <= 3:
            continue
        # print("yp:", yp)
        # features
        df = f0_feat(yp)
        # print("df:", df)
        X = df.to_numpy()
        # print("X:", X)
        # predict normalization factor
        f = mod.predict(X)
        # print("f:", f)
        # underlying f0 median
        med_ref = np.median(yp) * f[0]
        # print("med:", np.median(yp))
        # print("med_ref:", med_ref)
        # center f0 on this median
        # print("y1:", y)
        ys[ys > 0] -= med_ref
        # print("--> y2:", ys)
        # myl.stopgo()
        y[ti] = ys
    return y


def subtract_register(y, tf, tt, opt={}):
    r''' subtract from y within each interval from tt the regression midline
    Args:
    y: (np.array) e.g. F0 contour
    tf: (np.array) of same length as y with corresponding time stamps
    tt: (2-dim np.array) of intervals within which register midline is fitted
        and subtracted from y
    Returns:
    y: (np.array) after register subtraction
    '''
    tf, y = np.asarray(tf), np.asarray(y)
    for t in tt:
        ti = myl.find_interval(tf, t)
        if len(ti) > 0:
            c = myl.myPolyfit(ti, y[ti], 1)
            r = np.polyval(c, ti)
            # yo = cp.copy(y[ti])
            y[ti] -= r
            # myl.myPlot({"1": ti, "2": ti, "3": ti},
            #           {"1": yo, "2": r, "3": y[ti]})
    return y


def outl_idx(y, opt={}):

    opt = myl.opt_default(opt, {"zi": False,
                                "f": 3,
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
        return myl.ea()

    # getting lower and upper boundary lb, ub

    yi = y.take(i)
    if ((len(yi) == 0) or (not np.isfinite(yi).any())):
        return myl.ea()
    
    if opt['m'] == 'mean':
        # mean +/- f*sd
        m = np.nanmean(yi)
        r = np.nanstd(yi)
        lb, ub = m-f*r, m+f*r
    else:
        m = np.nanmedian(yi)
        q1, q3 = np.nanpercentile(yi, [25, 75])
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


def linZeroInterp(y):
    r''' linear interpolation across zeros
    Args:
    y: (np.array of floats)
    Returns:
    y: (np.array) with zeros replaced by linear interpolation
    '''

    # yo = cp.copy(y)
    xi = myl.find(y, '==', 0)
    xp = myl.find(y, '!=', 0)
    if len(xp) == 0:
        return None
    yp = y[xp]
    yi = np.interp(xi, xp, yp)
    y[xi] = yi

    # t = np.linspace(0,len(y),len(y))
    # fig, spl = plt.subplots(1,1,squeeze=False)
    # cid1 = fig.canvas.mpl_connect('button_press_event', onclick_next)
    # cid2 = fig.canvas.mpl_connect('key_press_event', onclick_exit)
    # spl[0,0].plot(t,y)
    # spl[0,0].plot(t,yo)
    # plt.show()
    return y

# klick on plot -> next one


def onclick_next(event):
    plt.close()

# press key -> exit


def onclick_exit(event):
    sys.exit()
