#!/usr/bin/env python3

import numpy as np
import myUtilHL as myl


def discont(x, ts=[], opt={}):
    r''' measures delta and linear fit discontinuities between
    adjacent array elements in terms of:
    - delta
    - reset of regression lines
    - root mean squared deviation between overall regression line and
      -- preceding segment's regression line
      -- following segment's regression line
      -- both, preceding and following, regression lines
    - extrapolation rmsd between following regression line
      and following regression line, extrapolated by regression
      on preceding segment
    Args:
      x: (nx2 np.array) [[time val] ...]
           OR
         (nx1 np.array) [val ...]
            for the latter indices are taken as time stamps
      ts: (nx1 np.array) [time ...] of time stamps (or indices for size(x)=nx1)
        at which to calculate discontinuity; if empty, discontinuity is
        calculated at each point in time. If size(x)=nx1 ts MUST contain
        indices
           OR
          (nx2 np.array) [[t_off t_on] ...] to additionally account for pauses
      opt: (dict)
        win: (string) <'glob'>|'loc' calculate discontinuity over
          entire sequence or within window
        l: (int) <3> if win==loc, length of window in sec or idx
            (splitpoint - .l : splitpoint + .l)
        do_plot: (boolean) <False> plots orig contour and linear stylization
        plot: (dict) <{}> with plotting options; cf. discont_seg()
    Returns:
      d: (dict)
         ("s1": pre-bnd segment [i-l,i[,
          "s2": post-bnd segment [i,i+l]
          "sc": joint segment [i-l,i+l])
        dlt: delta
        res: reset
        ry1: s1, rmsd between joint vs pre-bnd fit
        ry2: s2, rmsd between joint vs post-bnd fit
        ryc: sc, rmsd between joint vs pre+post-bnd fit
        ry2e: s2: rmsd between pre-bnd fit extrapolated to s2 and post-bnd fit
        rx1: s1, rmsd between joint fit and pre-boundary x-values
        rx2: s2, rmsd between joint fit and post-boundary x-values
        rxc: sc, rmsd between joint fit and pre+post-boundary x-values
        rr1: s1, ratio rmse(joint_fit)/rmse(pre-bnd_fit)
        rr2: s2, ratio rmse(joint_fit)/rmse(post-bnd_fit)
        rrc: sc, ratio rmse(joint_fit)/rmse(pre+post-bnd_fit)

        ra1: c1-rate s1
        ra2: c1-rate s2
        dlt_ra: ra2-ra1
        s1_c3: cubic fitting coefs of s1
        s1_c2
        s1_c1
        s1_c0
        s2_c3: cubic fitting coefs of s2
        s2_c2
        s2_c1
        s2_c0
        dlt_c3: s2_c3-s1_c3
        dlt_c2: s2_c2-s1_c2
        dlt_c1: s2_c1-s1_c1
        dlt_c0: s2_c0-s1_c0
        eucl_c: euclDist(s1_c*,s2_c*)
        corr_c: corr(s1_c*,s2_c*)
        v1: variance in s1
        v2: variance in s2
        vc: variance in sc
        vr: variance ratio (mean(v1,v2))/vc
        dlt_v: v2-v1
        m1: mean in s1
        m2: mean in s2
        dlt_m: m2-m1
        p: pause length (in sec or idx depending on numcol(x);
                         always 0, if t is empty or 1-dim)

      i in each list refers to discontinuity between x[i-1] and x[i]
      dimension of each list:
         if len(ts)==0: n-1 array (first x-element skipped)
         else: mx6; m is number of ts-elements in range of x[:,0],
               resp. in index range of x[1:-1]
    Comments:
    for all variables but corr_c and vr higher values indicate higher
    discontinuity. For corr_c and vr lower values indicate higher
    discontinuity.
    Variables:
    x1: original f0 contour for s1
    x2: original f0 contour for s2
    xc: original f0 contour for sc
    y1: line fitted on segment a
    y2: line fitted on segment b
    yc: line fitted on segments a+b
    yc1: yc part for x1
    yc2: yc part for x2
    ye: x1/y1-fitted line for x2
    cu1: cubic fit coefs of time-nrmd s1
    cu2: cubic fit coefs of time-nrmd s2
    yu1: polyval(cu1)
    yu2: polyval(cu2); yu1 and yu2 are cut to same length
    '''

    # time: first column or indices
    if np.ndim(x) == 1:
        t = np.arange(0, len(x))
        x = np.asarray(x)
    else:
        t = x[:, 0]
        x = x[:, 1]

    # tsi: index pairs in x for which to derive discont values
    #      [[infimum supremum]...] s1 right-aligned to infimum,
    #      s2 left-aligne to supremum
    #      for 1-dim ts both values are adjacent [[i-1, i]...]
    # zp: zero pause True for 1-dim ts input, False for 2-dim
    tsi, zp = discont_tsi(t, ts)

    # opt init
    opt = myl.opt_default(opt, {'win': 'glob', 'l': 3, 'do_plot': False,
                                'plot': {}})

    # output
    d = discont_init()

    # linear fits
    # over time stamp pairs
    for ii in tsi:

        # delta
        d['dlt'].append(x[ii[1]]-x[ii[0]])

        # segments (x, y values of pre-, post, joint segments)
        t1, t2, tc, x1, x2, xc, y1, y2, yc, yc1, yc2, ye, \
            cu1, cu2, yu1, yu2 = discont_seg(t, x, ii, opt)
        d = discont_feat(d, t1, t2, tc, x1, x2, xc, y1, y2, yc,
                         yc1, yc2, ye, cu1, cu2, yu1, yu2, zp)

    # to np.array
    for x in d:
        d[x] = np.asarray(d[x])

    return d


def discont_tsi(t, ts):

    ''' indices in t for which to derive discont values
    Args:
    t: all time stamps/indices
    ts: selected time stamps/indices, can be empty
    Returns:
    ii
    ==t-indices (from 1), if ts empty
    ==indices of supremum t-elements for ts stamps, else
    REMARKS:
    ii will be unique and sorted without 0
    '''

    if len(ts) == 0:
        return np.arange(1, len(t))
    ii = []
    for x in ts:
        s = myl.find(t, '>=', x)
        if len(s) == 0:
            continue
        ii.append(s[0])
    # sort/unique, remove 0
    ii = sortuniq(ii)
    if ii[0] == 0:
        ii = ii[1:len(ii)]
    return ii


def sortuniq(x):

    ''' sort+uniq on a list x '''

    return sorted(set(x))


def discont_init():

    ''' init discont dict '''

    return {"dlt": [],
            "res": [],
            "ry1": [],
            "ry2": [],
            "ryc": [],
            "ry2e": [],
            "rx1": [],
            "rx2": [],
            "rxc": [],
            "rr1": [],
            "rr2": [],
            "rrc": [],
            "ra1": [],
            "ra2": [],
            "dlt_ra": [],
            "s1_c3": [],
            "s1_c2": [],
            "s1_c1": [],
            "s1_c0": [],
            "s2_c3": [],
            "s2_c2": [],
            "s2_c1": [],
            "s2_c0": [],
            "dlt_c3": [],
            "dlt_c2": [],
            "dlt_c1": [],
            "dlt_c0": [],
            "eucl_c": [],
            "corr_c": [],
            "eucl_y": [],
            "corr_y": [],
            "v1": [],
            "v2": [],
            "vc": [],
            "vr": [],
            "dlt_v": [],
            "m1": [],
            "m2": [],
            "dlt_m": [],
            "p": []}

# pre/post-boundary and joint segments


def discont_seg(t, x, ii, opt):
    # preceding, following segment indices
    i1, i2 = discont_idx(t, ii, opt)
    # print(ii,"\n-> ", i1,"\n-> ", i2) #!v
    # myl.stopgo() #!v

    t1, t2, x1, x2 = t[i1], t[i2], x[i1], x[i2]
    tc = np.concatenate((t1, t2))
    xc = np.concatenate((x1, x2))

    # normalized time (only needed for reported polycoefs, not
    # for output lines
    tn1 = myl.nrm_vec(t1, {'mtd': 'minmax',
                           'rng': [-1, 1]})
    tn2 = myl.nrm_vec(t2, {'mtd': 'minmax',
                           'rng': [-1, 1]})

    # linear fit coefs
    c1 = myl.myPolyfit(t1, x1, 1)
    c2 = myl.myPolyfit(t2, x2, 1)
    cc = myl.myPolyfit(tc, xc, 1)

    # cubic fit coefs (for later shape comparison)
    cu1 = myl.myPolyfit(tn1, x1, 3)
    cu2 = myl.myPolyfit(tn2, x2, 3)
    yu1 = np.polyval(cu1, tn1)
    yu2 = np.polyval(cu2, tn2)

    # cut to same length (from boundary)
    ld = len(yu1)-len(yu2)
    if ld > 0:
        yu1 = yu1[ld:len(yu1)]
    elif ld < 0:
        yu2 = yu2[0:ld]
    # robust treatment
    while len(yu2) < len(yu1):
        yu2 = np.append(yu2, yu2[-1])
    while len(yu1) < len(yu2):
        yu1 = np.append(yu1, yu1[-1])

    # fit values
    y1 = np.polyval(c1, t1)
    y2 = np.polyval(c2, t2)
    yc = np.polyval(cc, tc)
    # distrib yc over t1 and t2
    yc1, yc2 = yc[0:len(y1)], yc[len(y1):len(yc)]
    # linear extrapolation
    ye = np.polyval(c1, t2)

    # legend_loc: 'upper left'

    # plotting linear fits
    # segment boundary
    xb = []
    xb.extend(yu1)
    xb.extend(yu2)
    xb.extend(ye)
    xb.extend(x1)
    xb.extend(x2)
    xb = np.asarray(xb)
    if opt['do_plot'] and len(xb) > 0:
        lw1, lw2 = 5, 3
        yb = [np.min(xb), np.max(xb)]
        tb = [t1[-1], t1[-1]]
        po = opt["plot"]
        po = myl.opt_default(po, {"legend_loc": "best",
                                  "fs_legend": 35,
                                  "fs": (20, 12),
                                  "fs_title": 40,
                                  "fs_ylab": 30,
                                  "fs_xlab": 30,
                                  "title": "",
                                  "xlab": "time",
                                  "ylab": ""})
        po["ls"] = {"o": "--k", "b": "-k", "s1": "-g", "s2": "-g",
                    "sc": "-r", "se": "-c"}
        po["lw"] = {"o": lw2, "b": lw2, "s1": lw1,
                    "s2": lw1, "sc": lw1, "se": lw2}
        po["legend_order"] = ["o", "b", "s1", "s2", "sc", "se"]
        po["legend_lab"] = {"o": "orig", "b": "bnd", "s1": "fit s1",
                            "s2": "fit s2", "sc": "fit joint",
                            "se": "pred s2"}
        myl.myPlot({"o": tc, "b": tb, "s1": t1, "s2": t2, "sc": tc, "se": t2},
                   {"o": xc, "b": yb, "s1": y1, "s2": y2, "sc": yc, "se": ye},
                   po)
    return t1, t2, tc, x1, x2, xc, y1, y2, yc, yc1, yc2, ye, cu1, cu2, yu1, yu2


# features
def discont_feat(d, t1, t2, tc, x1, x2, xc, y1, y2, yc, yc1, yc2, ye,
                 cu1, cu2, yu1, yu2, zp):
    # reset
    d["res"].append(y2[0]-y1[-1])
    # y-RMSD between regression lines: 1-pre, 2-post, c-all
    d["ry1"].append(myl.rmsd(yc1, y1))
    d["ry2"].append(myl.rmsd(yc2, y2))
    d["ryc"].append(myl.rmsd(yc, np.concatenate((y1, y2))))
    # extrapolation y-RMSD
    d["ry2e"].append(myl.rmsd(y2, ye))
    # xy-RMSD between regression lines and input values: 1-pre, 2-post, c-all
    rx1 = myl.rmsd(yc1, x1)
    rx2 = myl.rmsd(yc2, x2)
    rxc = myl.rmsd(yc, xc)
    d["rx1"].append(rx1)
    d["rx2"].append(rx2)
    d["rxc"].append(rxc)
    # xy-RMSD ratios of joint fit divided by single fits RMSD
    # (the higher, the more discontinuity)
    d["rr1"].append(myl.robust_div(rx1, myl.rmsd(y1, x1)))
    d["rr2"].append(myl.robust_div(rx2, myl.rmsd(y2, x2)))
    d["rrc"].append(myl.robust_div(
        rxc, myl.rmsd(np.concatenate((y1, y2)), xc)))
    # rates
    d["ra1"].append(drate(t1, y1))
    d["ra2"].append(drate(t2, y2))
    d["dlt_ra"].append(d["ra2"][-1]-d["ra1"][-1])
    # means
    d["m1"].append(np.mean(x1))
    d["m2"].append(np.mean(x2))
    d["dlt_m"].append(d["m2"][-1]-d["m1"][-1])
    # variances
    d["v1"].append(np.var(x1))
    d["v2"].append(np.var(x2))
    d["vc"].append(np.var(xc))
    d["vr"].append(np.mean([d["v1"][-1], d["v2"][-1]])/d["vc"][-1])
    d["dlt_v"].append(d["v2"][-1]-d["v1"][-1])
    # shapes
    d["s1_c3"].append(cu1[0])
    d["s1_c2"].append(cu1[1])
    d["s1_c1"].append(cu1[2])
    d["s1_c0"].append(cu1[3])
    d["s2_c3"].append(cu2[0])
    d["s2_c2"].append(cu2[1])
    d["s2_c1"].append(cu2[2])
    d["s2_c0"].append(cu2[3])
    d["eucl_c"].append(myl.dist_eucl(cu1, cu2))
    rr = np.corrcoef(cu1, cu2)
    d["corr_c"].append(rr[0, 1])
    d["dlt_c3"].append(d["s2_c3"][-1]-d["s1_c3"][-1])
    d["dlt_c2"].append(d["s2_c2"][-1]-d["s1_c2"][-1])
    d["dlt_c1"].append(d["s2_c1"][-1]-d["s1_c1"][-1])
    d["dlt_c0"].append(d["s2_c0"][-1]-d["s1_c0"][-1])
    d["eucl_y"].append(myl.dist_eucl(yu1, yu2))
    rry = np.corrcoef(yu1, yu2)
    d["corr_y"].append(rry[0, 1])
    # pause
    if zp:
        d["p"].append(0)
    else:
        d["p"].append(t2[0]-t1[-1])

    return d


def drate(t, y):
    if t[-1] == t[0]:
        return 0
    return (y[-1] - y[0]) / (t[-1] - t[0])


def discont_idx(t, i, opt):
    r''' preceding, following segment indices around t[i]
    defined by opt[win|l]
    Args:
    t: (array) time vector
    i: (int) current idx in t
    opt: (dict) passed on by discont()
    Returns:
    i1, i2: pre/post boundary index arrays
    REMARK: i is part of i2
    '''
    lx = len(t)
    # glob: preceding, following segment from start/till end
    if opt['win'] == 'glob':
        return np.arange(0, i), np.arange(i, lx)
    # rng = [t[i]-opt['l'], t[i]+opt['l']]
    i1 = myl.find_interval(t, [t[i-1]-opt['l'], t[i-1]])
    i2 = myl.find_interval(t, [t[i], t[i]+opt['l']])
    return i1, i2
