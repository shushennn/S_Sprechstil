#!/usr/bin/env python3

import sys
import re
import copy as cp
import numpy as np
import scipy.io.wavfile as sio
import scipy.signal as sis
import math
import myUtilHL as myl


#########################################
# prosodic structure wrapper ############
#########################################

def prosodic_structure(s, fs=None, opt_pau={}, opt_syl={}):
    r''' wrapper around pau_detector() and syl_ncl_wrapper()
    Args:
    s: (np.array) signal or (str) wav file name
    fs: (int) sample rate (obligatory if s in np.array!)
    opt_pau: (dict) see opt argument in pau_detector()
    opt_syl: (dict) see opt argument in syl_ncl()
    Returns:
    pc: (dict) see pau_detector() output
    ncl: (dict) see syl_ncl_wrapper() output
    '''

    # wavread
    if type(s) is str:
        fs, s = sio.read(s)
        s = myl.wav_int2float(s)
        s = s-np.mean(s)

    opt_pau["fs"] = fs
    opt_syl["fs"] = fs

    pc = pau_detector(s, opt_pau)
    ncl = syl_ncl_wrapper(s, pc["tci"], opt_syl)

    return pc, ncl


##########################################################
# syllable nucleus extraction ############################
##########################################################

def syl_ncl_wrapper(s, i_chunks, opt={}):
    r''' calls syllable nucleus extraction for several chunks.
    Args:
    s: (np.array) signal
    i_chunks: (np.array) [[start endIndex] ...] of each chunk (index in signal)
       corresponds to pc["tci"] from pc=pau_detector()
    opt: (dict) "fs" sample rate is obligatory; rest see syl_ncl()
    Returns:
    ncl['t'] - vector of syl ncl time stamps (in sec)
          ['ti'] - corresponding vector idx in s
          ['e_ratio'] - corresponding energy ratios
                   (analysisWindow/referenceWindow)
    '''

    ncl = {"t": myl.ea(),
           "ti": myl.ea(),
           "e_ratio": myl.ea()}

    for c in i_chunks:
        # signal segment
        ii = np.arange(c[0], c[1], 1)
        if len(ii) == 0:
            continue
        opt["ons"] = ii[0]
        while (len(ii) > 0 and ii[-1] >= len(s)):
            ii = ii[0:len(ii)-1]
        if len(ii) == 0:
            continue
        nclC = syl_ncl(s[ii], opt)
        for x in ncl:
            ncl[x] = np.append(ncl[x], nclC[x])

    return ncl


def syl_ncl(s, opt={}, en=None):
    r''' syllable nucleus detection
    Args:
       s - mono signal
       opt['fs'] - sample frequency (obligatory)
          ['ons'] - onset sample index <0> (to be added to time output)
          ['flt']['f']     - filter options, boundary frequencies in Hz
                             (2 values for btype 'band', else 1):
                             <np.asarray([200,4000])>
                 ['btype'] - <'band'>|'high'|'low'
                 ['ord']   - butterworth order <5>
                 ['fs']    - (internally copied)
          ['l']     - analysis window length
          ['l_ref'] - reference window length
          ['d_min'] - min distance between subsequent nuclei (in sec)
          ['e_min'] - min energy required for nucleus as a proportion to
                      max energy <0.16>
          ['e_rel'] - min energy quotient analysisWindow/referenceWindow
          ['e_val'] - quotient, how sagged the energy valley between two
                      nucleus candidates should be. Measured relative to
                      the lower energy candidate. The lower, the deeper
                      the required valley between two peaks. Meaningful
                      range ]0, 1]. Recommended range: [0.9 1[
          ['center'] - boolean; subtract mean energy
          ['sts'] - stepsize for energy calculation <0.03>

    Returns:
       ncl['t'] - vector of syl ncl time stamps (in sec)
          ['ti'] - corresponding vector idx in s
          ['e_ratio'] - corresponding energy ratios
                   (analysisWindow/referenceWindow)
    '''

    # options
    if 'fs' not in opt:
        sys.exit('syl_ncl: opt does not contain key fs.')
    dflt = {'flt': {'f': np.array([200, 3000]), 'btype': 'band', 'ord': 5},
            'e_rel': 1.05, 'l': 0.05, 'l_ref': 0.15, 'd_min': 0.12,
            'e_min': 0.05, 'ons': 0, 'e_val': 1, 'center': False,
            'sts': 0.03}
    opt = myl.opt_default(opt, dflt)
    opt['flt']['fs'] = opt['fs']
    opt['flt']['f'] = np.asarray(opt['flt']['f'])

    # signal too short
    if len(s)/opt['fs'] < 0.1:
        t = np.asarray([round(len(s)/2+opt['ons'])])
        ncl = {'ti': t, 't': idx2sec(t, opt['fs']), 'e_ratio': [0]}
        return ncl

    # reference window length
    rws = math.floor(opt['l_ref']*opt['fs'])

    # energy win length
    ml = math.floor(opt['l']*opt['fs'])

    # stepsize (unit: sample index; 0.03 sec)
    sts = max([1, math.floor(opt["sts"]*opt['fs'])])

    # minimum distance between subsequent nuclei
    # (unit: sample index)
    md = math.floor(opt['d_min']*opt['fs'])

    # band or lowpass filtering
    flt = fu_filt(s, opt['flt'])
    y = flt['y']

    # signal length
    ls = len(y)

    # energy contour (RMSD)
    e_y = np.array([])
    i_steps = np.arange(0, ls, sts)
    for i in i_steps:
        yi = np.arange(i, min([ls, i+ml]), 1)
        e_y = np.append(e_y, rmsd(y[yi]))

    # centering
    if bool(opt['center']):
        e_y -= np.mean(e_y)

    # minimum energy e_min as proportion opt['e_min'] of found
    # energy maximum
    e_min = opt['e_min']*max(e_y)

    # output vectors
    # time stamps of syl nuclei
    t = np.array([])

    # indices of energy window midpoints in x
    all_i = np.array([])
    # corresponding energy values from analysis windows
    all_e = np.array([])
    # corresponding energy values from reference windows
    all_r = np.array([])

    # energy calculation in analysis and reference windows
    # option dicts for windowing_idx()
    wopt_en = {'win': ml, 'rng': [0, ls]}
    wopt_ref = {'win': rws, 'rng': [0, ls]}
    for i in i_steps:
        # signal segment in analysis window
        yi = windowing_idx(i, wopt_en)
        ys = y[yi]
        e_y = rmsd(ys)
        # signal segment in reference window
        ri = windowing_idx(i, wopt_ref)
        rs = y[ri]
        e_rw = rmsd(rs)
        # updates
        all_i = np.append(all_i, i)
        all_e = np.append(all_e, e_y)
        all_r = np.append(all_r, e_rw)

    # for debugging, model comparison only #!xs
    if en is not None:
        if len(en["e"]) == len(all_i):
            all_e = en["e"]
            all_r = en["r"]
        else:
            print(len(en["e"]), len(all_i), "-> no replacement")
            myl.stopgo()

    # getting initial syllable ncl candidates: local energy maxima
    # (do not use min duration md for order arg, since local
    #  maximum might be obscured already by energy increase
    #  towards a neighboring peak which is further away than md,
    #  and not only by peaks closer than md)
    idx = sis.argrelmax(all_e, order=1)

    # candidate reduction
    # a) energy constraints
    # timestamps (idx)
    tx = np.array([])
    # energy ratios
    e_ratiox = np.array([])
    # idx in all_i
    tix = np.array([]).astype(int)
    for i in idx[0]:

        # valley between this and previous peak deep enough?
        if len(tix) > 0:
            ie = all_e[tix[-1]:i]
            # if len(ie)<3: #!xs
            #    continue  #!xs
            valley = np.min(ie)
            nclmin = np.min([ie[0], all_e[i]])
            if valley >= opt['e_val'] * nclmin:
                continue

        # peak prominent enough?
        if ((all_e[i] >= all_r[i]*opt['e_rel']) and (all_e[i] > e_min)):
            tx = np.append(tx, all_i[i])
            tix = np.append(tix, i)
            e_ratiox = np.append(e_ratiox, all_e[i]/all_r[i])

    # no candidate left?
    if len(tx) == 0:
        return {'ti': np.array([]),
                't': np.array([]),
                'e_ratio': np.array([])}

    # b) min duration constraints
    # init by first found ncl
    t = np.array([tx[0]])
    e_ratio = np.array([e_ratiox[0]])
    ti = np.array([tix[0]]).astype(int)
    for i in range(1, len(tx)):
        # ncl too close
        if np.abs(tx[i]-t[-1]) < md:
            # current ncl with higher energy: replace last stored one
            if e_ratiox[i] > e_ratio[-1]:
                t[-1] = tx[i]
                ti[-1] = tix[i]
                e_ratio[-1] = e_ratiox[i]
        else:
            t = np.append(t, tx[i])
            ti = np.append(ti, tix[i])
            e_ratio = np.append(e_ratio, e_ratiox[i])

    # add onset
    t = t+opt['ons']

    # output dict
    ncl = {'ti': t, 't': idx2sec(t, opt['fs']), 'e_ratio': e_ratio}

    return ncl


def syl_ncl_from_df_wrapper(os, t_chunks):
    r''' calls syllable nucleus extraction on opensmile LLDs
    for several chunks.
    Args:
    os: (pd.DataFrame) opensmile LLDs
    t_chunks: (np.array) [[start endTime] ...] of each chunk
    Returns:
    ncl['t'] - vector of syl ncl time stamps (in sec)
       ['ti'] - corresponding vector idx in s
       ['e_ratio'] - corresponding energy ratios
                   (analysisWindow/referenceWindow)
    '''

    ncl = {"t": myl.ea(),
           "ti": myl.ea(),
           "e_ratio": myl.ea()}

    for c in t_chunks:
                
        nclC = syl_ncl_from_df(os, t_interval=c)
        for x in ncl:
            ncl[x] = np.append(ncl[x], nclC[x])

    return ncl


def syl_ncl_from_df(dfs, col = "pcm_RMSenergy_sma",
                    col_voi = "voicingFinalUnclipped_sma",
                    t_interval = None, opt={}):
    r''' syllable nucleus detection from openSMILE dataFrame
    Args:
       dfs: (pd.DataFrame) opensmile dataframe: should contain entries
          of single file only!
       col: (str) energy column name
       col_voi: (str) voicing probability column name
       t_interval: (array-like) on and offset of segment in which to
            locate syllables (in sec) 
       opt: (dict)
          ['ons'] - onset time <0> (sec, to be added to time output)
          ['d_min'] - min distance between subsequent nuclei (in sec)
          ['e_min'] - min energy required for nucleus as a proportion to
                      max energy <0.16>
          ['e_rel'] - min energy quotient analysisWindow/referenceWindow
          ['e_val'] - quotient, how sagged the energy valley between two
                      nucleus candidates should be. Measured relative to
                      the lower energy candidate. The lower, the deeper
                      the required valley between two peaks. Meaningful
                      range ]0, 1]. Recommended range: [0.9 1[
          ['voi_min'] - voicing probability threshold (0.5)
                      for syllable nucleus candidates at least this value
                      must be reached in dfs[col_voi]
    Returns:
       ncl['t'] - vector of syl ncl time stamps (in sec)
          ['ti'] - corresponding vector idx in s
          ['e_ratio'] - corresponding energy ratios
                   (analysisWindow/referenceWindow)
    '''
    
    # optimized on /home/ureichel/tmp/sylref_200519 to get as close
    # as possible to signal-based syllabification
    dflt = {'e_rel': 1, 'd_min': 0.09, 'voi_min': 0.5,
            'e_min': 0.08, 'ons': 0, 'e_val': 0.7}

    opt = myl.opt_default(opt, dflt)

    # time stamps
    ts = (dfs.index.get_level_values("start").total_seconds().to_numpy() +
          dfs.index.get_level_values("end").total_seconds().to_numpy()) / 2
    
    if t_interval is not None:
        df = dfs.loc[(ts > t_interval[0]) & (ts < t_interval[1])]
        ts = ts[(ts > t_interval[0]) & (ts < t_interval[1])]
    else:
        df = dfs
        
    # minimum distance between subsequent nuclei
    # (unit: sample index)
    md = opt['d_min']

    # entire energy contour 
    all_e = df[col].to_numpy()
    
    # minimum energy e_min as proportion opt['e_min'] of found
    # energy maximum
    e_min = opt['e_min']*max(all_e)

    # output vectors
    # time stamps of syl nuclei
    t = np.array([])

    # corresponding energy values from reference windows
    first_mean = np.mean([all_e[0], all_e[1]])
    last_mean = np.mean([all_e[-2], all_e[-1]])
    all_r = np.convolve(all_e, np.ones((3,))/3, mode='valid')
    all_r = np.concatenate(([first_mean], all_r, [last_mean]))

    # voicing prob
    if col_voi and col_voi in dfs.columns:
        all_voi = dfs[col_voi].to_numpy()
    else:
        all_voi = np.ones(len(all_r))

    # getting initial syllable ncl candidates: local energy maxima
    # (do not use min duration md for order arg, since local
    #  maximum might be obscured already by energy increase
    #  towards a neighboring peak which is further away than md,
    #  and not only by peaks closer than md)
    idx = sis.argrelmax(all_e, order=1)
    
    # candidate reduction
    # a) energy constraints
    # timestamps (sec)
    tx = np.array([]).astype(int)
    # energy ratios
    e_ratiox = np.array([])
    # idx in all_e
    tix = np.array([]).astype(int)
    for i in idx[0]:

        # not enough voicing?
        if all_voi[i] < opt["voi_min"]:
            continue
        
        # valley between this and previous peak deep enough?
        if len(tix) > 0 and opt['e_val'] < 1:
            ie = all_e[tix[-1]:i]
            valley = np.min(ie)
            nclmin = np.min([ie[0], all_e[i]])
            if valley >= opt['e_val'] * nclmin:
                continue

        # peak prominent enough?
        if ((all_e[i] >= all_r[i]*opt['e_rel']) and (all_e[i] > e_min)):
            tx = np.append(tx, ts[i])
            tix = np.append(tix, i)
            e_ratiox = np.append(e_ratiox, all_e[i]/all_r[i])
            
    # no candidate left?
    if len(tx) == 0:
        return {'ti': np.array([]),
                't': np.array([]),
                'e_ratio': np.array([])}

    # b) min duration constraints
    # init by first found ncl (sec)
    t = np.array([tx[0]])
    e_ratio = np.array([e_ratiox[0]])
    ti = np.array([tix[0]]).astype(int)
    
    for i in range(1, len(tx)):
        # ncl too close
        if np.abs(tx[i]-t[-1]) < md:
            # current ncl with higher energy: replace last stored one
            if e_ratiox[i] > e_ratio[-1]:
                t[-1] = tx[i]
                ti[-1] = tix[i]
                e_ratio[-1] = e_ratiox[i]
        else:
            t = np.append(t, tx[i])
            ti = np.append(ti, tix[i])
            e_ratio = np.append(e_ratio, e_ratiox[i])

    # add onset (not needed since t already contains the
    # correct time stamps)
    #t = t+opt['ons']

    # output dict
    ncl = {'ti': ti, 't': t, 'e_ratio': e_ratio}

    return ncl




#################################################################
# speech chunk and pause detection ##############################
#################################################################

# speech chunk and pause detection
# IN:
#   s - mono signal
#   opt['fs']  - sample frequency
#      ['ons'] - sample index onset <0> (to be added to time output)
#      ['flt']['f']     - filter options, boundary frequencies in Hz
#                         (2 values for btype 'band', else 1): <8000>
#                        (evtl. lowered by fu_filt())
#             ['btype'] - <'band'>|'high'|<'low'>
#             ['ord']   - butterworth order <5>
#             ['fs']    - (internally copied)
#      ['l']     - analysis window length (in sec)
#      ['l_ref'] - reference window length (in sec)
#      ['e_rel'] - min energy quotient analysisWindow/referenceWindow
#      ['fbnd']  - True|<False> assume pause at beginning and end of file
#      ['n']     - <-1> extract exactly n pauses (if > -1)
#      ['min_pau_l'] - min pause length <0.5> sec
#      ['min_chunk_l'] - min inter-pausal chunk length <0.2> sec
#      ['force_chunk'] - <False>, if True, pause-only is replaced by
#                        chunk-only
#      ['margin'] - <0> time to reduce pause on both sides (sec;
#                  if chunks need init and final silence)
#      ['trunc_only'] - <False> if True only use for truncation
#                       (cutting of initial and final pause)
# OUT:
#    pau['tp'] 2-dim array of pause [on off] (in sec)
#       ['tpi'] 2-dim array of pause [on off] (indices in s = sampleIdx-1)
#       ['tc'] 2-dim array of speech chunks [on off] (i.e. non-pause, in sec)
#       ['tci'] 2-dim array of speech chunks [on off] (indices)
#       ['e_ratio'] - energy ratios corresponding to pauses in ['tp']
#                 (analysisWindow/referenceWindow)
# for trunc_only=True only a reduced dict for chunk between first and last
#      pause is returned:
#        (time stamps are start and end of segment in case no pause occurs)
#     pau['tci']: [start_idx, end_idx]
#        ['tc']:  [start_time, end_time] in sec
def pau_detector(s, opt={}):
    if 'fs' not in opt:
        sys.exit('pau_detector: opt does not contain key fs.')
    dflt = {'e_rel': 0.075, 'l': 0.155, 'l_ref': 5, 'n': -1,
            'fbnd': False, 'ons': 0, 'force_chunk': False,
            'min_pau_l': 0.4, 'min_chunk_l': 0.2, 'margin': 0,
            'trunc_only': False,
            'flt': {'btype': 'low', 'f': np.asarray([8000]), 'ord': 5}}
    opt = myl.opt_default(opt, dflt)
    opt['flt']['fs'] = opt['fs']

    # removing DC, low-pass filtering
    flt = fu_filt(s, opt['flt'])
    y = flt['y']

    # pause detection for >=n pauses
    t, e_ratio = pau_detector_sub(y, opt)

    # truncation only
    if opt["trunc_only"]:
        return pau_trunc(t, len(y), opt)

    if len(t) > 0:

        # extending 1st and last pause to file boundaries
        if opt['fbnd']:
            t[0, 0] = 0
            t[-1, -1] = len(y)-1

        # merging pauses across too short chunks
        # merging chunks across too small pauses
        if (opt['min_pau_l'] > 0 or opt['min_chunk_l'] > 0):
            t, e_ratio = pau_detector_merge(t, e_ratio, opt)

        # too many pauses?
        # -> subsequently remove the ones with highest e-ratio
        if (opt['n'] > 0 and len(t) > opt['n']):
            t, e_ratio = pau_detector_red(t, e_ratio, opt)

    # speech chunks
    tc = pau2chunk(t, len(y))

    # pause-only -> chunk-only
    if (opt['force_chunk'] and len(tc) == 0):
        tc = cp.deepcopy(t)
        t = np.asarray([])
        e_ratio = np.asarray([])

    # add onset
    t = t+opt['ons']
    tc = tc+opt['ons']

    # return dict
    # incl fields with indices to seconds (index+1=sampleIndex)
    pau = {'tpi': t, 'tci': tc, 'e_ratio': e_ratio}
    pau['tp'] = idx2sec(t, opt['fs'])
    pau['tc'] = idx2sec(tc, opt['fs'])

    # print(pau)

    return pau


def pau_trunc(t, ly, opt):

    # chunks as ipu-s
    tci = pau2chunk(t, ly)

    # return reduced dict
    # tci: [start_idx end_idx] (+onset)
    # tc: [start_time end_time]
    tci = np.asarray([tci[0][0]+opt['ons'], tci[-1][1]+opt['ons']])
    return {'tci': tci, 'tc': idx2sec(tci, opt['fs'])}


# merging pauses across too short chunks
# merging chunks across too small pauses
# IN:
#   t [[on off]...] of pauses
#   e [e_rat ...]
# OUT:
#   t [[on off]...] merged
#   e [e_rat ...] merged (simply mean of merged segments taken)
def pau_detector_merge(t, e, opt):
    # min pause and chunk length in samples
    mpl = sec2smp(opt['min_pau_l'], opt['fs'])
    mcl = sec2smp(opt['min_chunk_l'], opt['fs'])

    # merging chunks across short pauses
    tm = np.asarray([])
    em = np.asarray([])
    for i in myl.idx_a(len(t)):
        if ((t[i, 1]-t[i, 0] >= mpl) or
                (opt['fbnd'] and (i == 0 or i == len(t)-1))):
            tm = myl.push(tm, t[i, :])
            em = myl.push(em, e[i])

    # nothing done in previous step?
    if len(tm) == 0:
        tm = cp.deepcopy(t)
        em = cp.deepcopy(e)
    if len(tm) == 0:
        return t, e

    # merging pauses across short chunks
    tn = np.asarray([tm[0, :]])
    en = np.asarray([em[0]])
    if (tn[0, 0] < mcl):
        tn[0, 0] = 0
    for i in np.arange(1, len(tm), 1):
        if (tm[i, 0] - tn[-1, 1] < mcl):
            tn[-1, 1] = tm[i, 1]
            en[-1] = np.mean([en[-1], em[i]])
        else:
            tn = myl.push(tn, tm[i, :])
            en = myl.push(en, em[i])

    # print("t:\n", t, "\ntm:\n", tm, "\ntn:\n", tn) #!v
    return tn, en


# pause to chunk intervals
# IN:
#    t [[on off]] of pause segments (indices in signal)
#    lng length of signal vector
# OUT:
#    tc [[on off]] of speech chunks
def pau2chunk(t, lng):
    if len(t) == 0:
        return np.asarray([[0, lng-1]])
    if t[0, 0] > 0:
        tc = np.asarray([[0, t[0, 0]-1]])
    else:
        tc = np.asarray([])
    for i in np.arange(0, len(t)-1, 1):
        if t[i, 1] < t[i+1, 0]-1:
            tc = myl.push(tc, [t[i, 1]+1, t[i+1, 0]-1])
    if t[-1, 1] < lng-1:
        tc = myl.push(tc, [t[-1, 1]+1, lng-1])
    if len(tc) == 0:
        return np.asarray([[0, lng-1]])
    return tc

# called by pau_detector
# IN:
#    as for pau_detector
# OUT:
#    t [on off]
#    e_ratio


def pau_detector_sub(y, opt):

    # settings
    # reference window span
    rl = math.floor(opt['l_ref']*opt['fs'])
    # signal length
    ls = len(y)
    # min pause length
    ml = opt['l']*opt['fs']
    # global rmse and pause threshold
    e_rel = cp.deepcopy(opt['e_rel'])

    # global rmse
    # as fallback in case reference window is likely to be pause
    # almost-zeros excluded (cf percentile) since otherwise pauses
    # show a too high influence, i.e. lower the reference too much
    # so that too few pauses detected
    # e_glob = rmsd(y)
    ya = abs(y)
    qq = np.percentile(ya, [50])
    e_glob = rmsd(ya[ya > qq[0]])

    t_glob = opt['e_rel']*e_glob

    # stepsize
    sts = max([1, math.floor(0.05*opt['fs'])])
    # energy calculation in analysis and reference windows
    wopt_en = {'win': ml, 'rng': [0, ls]}
    wopt_ref = {'win': rl, 'rng': [0, ls]}
    # loop until opt.n criterion is fulfilled
    # increasing energy threshold up to 1
    while e_rel < 1:
        # pause [on off], pause index
        t = np.asarray([])
        j = 0
        # [e_y/e_rw] indices as in t
        e_ratio = np.asarray([])
        i_steps = np.arange(1, ls, sts)
        for i in i_steps:
            # window
            yi = windowing_idx(i, wopt_en)
            e_y = rmsd(y[yi])
            # energy in reference window
            e_r = rmsd(y[windowing_idx(i, wopt_ref)])
            # take overall energy as reference if reference window is pause
            if (e_r <= t_glob):
                e_r = e_glob
            # if rmse in window below threshold
            if e_y <= e_r*e_rel:
                yis = yi[0]
                yie = yi[-1]
                if len(t)-1 == j:
                    # values belong to already detected pause
                    if len(t) > 0 and yis < t[j, 1]:
                        t[j, 1] = yie
                        # evtl. needed to throw away superfluous
                        # pauses with high e_ratio
                        if e_r > 0:
                            e_ratio[j] = np.mean([e_ratio[j], e_y/e_r])
                    else:
                        t = myl.push(t, [yis, yie])
                        e_ratio = myl.push(e_ratio, e_y/e_r)
                        j = j+1
                else:
                    t = myl.push(t, [yis, yie])
                    e_ratio = myl.push(e_ratio, e_y/e_r)
        # (more than) enough pauses detected?
        if len(t) >= opt['n']:
            break
        e_rel = e_rel+0.1

    if opt['margin'] == 0 or len(t) == 0:
        return t, e_ratio

    # shorten pauses by margins
    mar = int(opt['margin']*opt['fs'])
    tm, erm = np.array([]), np.array([])
    for i in myl.idx_a(len(t)):
        # only slim non-init and -fin pauses
        if i > 0:
            ts = t[i, 0]+mar
        else:
            ts = t[i, 0]
        if i < len(t)-1:
            te = t[i, 1]-mar
        else:
            te = t[i, 1]

        # pause disappeared
        if te <= ts:
            # ... but needs to be kept
            if opt['n'] > 0:
                tm = myl.push(tm, [t[i, 0], t[i, 1]])
                erm = myl.push(erm, e_ratio[i])
            continue
        # pause still there
        tm = myl.push(tm, [ts, te])
        erm = myl.push(erm, e_ratio[i])

    return tm, erm


def pau_detector_red(t, e_ratio, opt):
    # keep boundary pauses
    if opt['fbnd']:
        n = opt['n']-2
        # bp = [t[0,],t[-1,]]
        bp = np.concatenate((np.array([t[0, ]]), np.array([t[-1, ]])), axis=0)
        ii = np.arange(1, len(t)-1, 1)
        t = t[ii, ]
        e_ratio = e_ratio[ii]
    else:
        n = opt['n']
        bp = np.asarray([])

    if n == 0:
        t = []

    # remove pause with highest e_ratio
    while len(t) > n:
        i = myl.find(e_ratio, 'is', 'max')
        j = myl.find(np.arange(1, len(e_ratio), 1), '!=', i[0])
        t = t[j, ]
        e_ratio = e_ratio[j]

    # re-add boundary pauses if removed
    if opt['fbnd']:
        if len(t) == 0:
            t = np.concatenate(
                (np.array([bp[0, ]]), np.array([bp[1, ]])), axis=0)
        else:
            t = np.concatenate(
                (np.array([bp[0, ]]), np.array([t]),
                 np.array([bp[1, ]])), axis=0)

    return t, e_ratio


###################################################
# signal processing functions #####################
###################################################

def fu_filt(y, opt):
    r''' wrapper around Butterworth filter
    Args:
       1-dim vector
       opt['fs'] - sample rate
          ['f']  - scalar (high/low) or 2-element vector (band)
             of boundary freqs
          ['order'] - integer for order
          ['btype'] - string "band"|"low"|"high"
     Returns:
       flt['y'] - filtered signal
          ['b'] - b coefs
          ['a'] - a coefs
    '''

    # do nothing
    if not re.search('^(high|low|band)$', opt['btype']):
        return {'y': y, 'b': np.array([]), 'a': np.array([])}
    # check f<fs/2
    if (opt['btype'] == 'low' and opt['f'] >= opt['fs']/2):
        opt['f'] = opt['fs']/2-100
    elif (opt['btype'] == 'band' and opt['f'][1] >= opt['fs']/2):
        opt['f'][1] = opt['fs']/2-100
    fn = opt['f']/(opt['fs']/2)
    b, a = sis.butter(opt['ord'], fn, btype=opt['btype'])
    try:
        yf = sis.filtfilt(b, a, y)
    except:
        yf = y
    return {'y': yf, 'b': b, 'a': a}


def idx2sec(i, fs, ons=0):
    r''' transforms numpy array indices (=samplesIdx-1) to seconds
     Args:
       i: index
       fs: sample rate
       ons: <0> onset index to be added
     Returns:
       seconds
    '''
    return (i+1+ons)/fs


def sec2smp(i, fs, ons=0):
    ''' transforms seconds to sample indices (arrayIdx+1)'''
    return np.round(i*fs+ons).astype(int)


def rmsd(x, y=[]):
    r''' returns RMSD of two vectors or of one vector and zeros
     Args:
      x: 1-dim array
      y: 1-dim array <zeros(len(x))>
     Returns:
      root mean squared dev between x and y
    '''

    if len(y) == 0:
        y = np.zeros(len(x))
    x = np.array(x)
    return np.sqrt(np.mean((x-y)**2))


def windowing_idx(i, s):
    r''' returning all indices from onset to offset
    Args:
       i: (int) current index
       s: (dict)
        ["win"] (int) window length
        ["rng"] (2-element list) [on, off] range of indices to be windowed
     Returns:
      [on:1:off] in window around i
    '''

    on, off = windowing(i, s)
    return np.arange(on, off, 1)


def windowing(i, s):
    r''' window of length wl on and offset around single index in range [on off]
    vectorized version: seq_windowing
    Args:
       i: (int) current index
       s: (dict)
        ["win"] (int) window length
        ["rng"] (2-element list) [on, off] range of indices to be windowed
    Returns:
      on, off of window around i
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


def idx_seg(on, off, sts=1):
    '''returns index array between on and off (both included)'''

    return np.arange(on, off+1, sts)
