import scipy.signal as sis
import numpy as np
import myUtilHL as myl
import copy as cp
import scipy.fftpack as sf
import math


def dct_wrapper(y, opt):
    dflt = {'wintyp': 'kaiser', 'winparam': 1, 'rmo': True,
            'lb': 0, 'ub': 0, 'peak_prct': 80}
    opt = myl.opt_default(opt, dflt)
    nsm = 4

    if len(y) == 0:
        sm = myl.ea()
        while len(sm) <= nsm:
            sm = np.append(sm, np.nan)
        return {'c_orig': myl.ea(), 'f_orig': myl.ea(), 'c': myl.ea(),
                'f': myl.ea(), 'i': [], 'sm': sm, 'opt': opt,
                'm': np.nan, 'sd': np.nan, 'cbin': myl.ea(),
                'fbin': myl.ea(), 'f_max': np.nan, 'f_lmax': myl.ea(),
                'c_cog': np.nan}

    # weight window
    w = sig_window(opt['wintyp'], len(y), opt['winparam'])
    y = y*w

    # centralize
    y = y-np.mean(y)

    # DCT coefs
    c = sf.dct(y, norm='ortho')

    # indices (starting with 0)
    ly = len(y)
    ci = myl.idx_a(ly)

    # corresponding cos frequencies
    f = ci+1 * (opt['fs']/(ly*2))

    # band pass truncation of coefs
    # indices of coefs with lb <= freq <= ub
    i = dct_trunc(f, ci, opt)

    # analysis segment too short -> DCT freqs above ub
    if len(i) == 0:
        sm = myl.ea()
        while len(sm) <= nsm:
            sm = np.append(sm, np.nan)
        return {'c_orig': c, 'f_orig': f, 'c': myl.ea(), 'f': myl.ea(),
                'i': [], 'sm': sm, 'opt': opt, 'm': np.nan, 'sd': np.nan,
                'cbin': myl.ea(), 'fbin': myl.ea(), 'f_max': np.nan,
                'f_lmax': myl.ea(), 'c_cog': np.nan}

    # mean abs error from band-limited IDCT
    # mae = dct_mae(c,i,y)

    # remove constant offset with index 0
    # already removed by dct_trunc in case lb>0. Thus checked for i[0]==0
    #  (i[0] indeed represents constant offset; tested by
    #   cr = np.zeros(ly); cr[0]=c[0]; yr = sf.idct(cr); print(yr)
    if opt['rmo'] and len(i) > 1 and i[0] == 0:
        j = i[1:len(i)]
    else:
        j = i

    if type(j) is not list:
        j = [j]

    # coefs and their frequencies between lb and ub
    #   (+ constant offset removed)
    fi = f[j]
    ci = c[j]

    # spectral moments
    if len(j) > 0:
        sm = specmom(ci, fi, nsm)
    else:
        sm = np.zeros(nsm)

    # frequency bins
    fbin, cbin = dct_fbin(fi, ci, opt)

    # frequencies of global and local maxima in DCT spectrum
    f_max, f_lmax, px = dct_peak(ci, fi, sm[0], opt)

    # return
    return {'c_orig': c, 'f_orig': f, 'c': ci, 'f': fi, 'i': j,
            'sm': sm, 'opt': opt, 'm': np.mean(y), 'sd': np.std(y),
            'cbin': cbin, 'fbin': fbin, 'f_max': f_max,
            'f_lmax': f_lmax, 'c_cog': px}


# returns local and max peak frequencies
# IN:
#  x: array of abs coef amplitudes
#  f: corresponding frequencies
#  cog: center of gravity
# OUT:
#  f_gm: freq of global maximu
#  f_lm: array of freq of local maxima
#  px: threshold to be superseeded (derived from prct specs)
def dct_peak(x, f, cog, opt):

    x = abs(cp.deepcopy(x))

    # global maximum
    i = myl.find(x, 'is', 'max')
    if len(i) > 1:
        i = int(np.mean(i))

    f_gm = float(f[i])

    # local maxima
    # threshold to be superseeded
    px = dct_px(x, f, cog, opt)
    idx = myl.find(x, '>=', px)
    # 2d array of neighboring+1 indices
    # e.g. [[0,1,2],[5,6],[9,10]]
    ii = []
    # min freq distance between maxima
    fd_min = 1
    for i in myl.idx(idx):
        if len(ii) == 0:
            ii.append([idx[i]])
        elif idx[i] > ii[-1][-1]+1:
            xi = x[ii[-1]]
            fi = f[ii[-1]]
            j = myl.find(xi, 'is', 'max')
            # print('xi',xi,'fi',fi,'f',f[idx[i]])
            if len(j) > 0 and f[idx[i]] > fi[j[0]]+fd_min:
                # print('->1')
                ii.append([idx[i]])
            else:
                # print('->2')
                ii[-1].append(idx[i])
            # myl.stopgo() #!c
        else:
            ii[-1].append(idx[i])

    # get index of x maximum within each subsegment
    # and return corresponding frequencies
    f_lm = []
    for si in ii:
        zi = myl.find(x[si], 'is', 'max')
        if len(zi) > 1:
            zi = int(np.mean(zi))
        else:
            zi = zi[0]
        i = si[zi]
        if not np.isnan(i):
            f_lm.append(f[i])

    # print('px',px)
    # print('i',ii)
    # print('x',x)
    # print('f',f)
    # print('m',f_gm,f_lm)
    # myl.stopgo()

    return f_gm, f_lm, px


def dct_px(x, f, cog, opt):
    r''' return center-of-gravity related amplitude
    Args:
    x: array of coefs
    f: corresponding freqs
    cog: center of gravity freq
    opt
    Returns:
    coef amplitude related to cog
    '''

    x = abs(cp.deepcopy(x))

    # cog outside freq range
    if cog <= f[0]:
        return x[0]
    elif cog >= f[-1]:
        return x[-1]
    # find f-indices adjacent to cog
    for i in range(len(f)-1):
        if f[i] == cog:
            return x[i]
        elif f[i+1] == cog:
            return x[i+1]
        elif f[i] < cog and f[i+1] > cog:
            # interpolate
            # xi = np.interp(cog,f[i:i+2],x[i:i+2])
            # print('cog:',cog,'xi',f[i:i+2],x[i:i+2],'->',xi)
            return np.interp(cog, f[i:i+2], x[i:i+2])
    return np.percentile(x, opt['peak_prct'])


def idct_bp(c, i=myl.ea()):
    ''' IDCT within bandpass '''
    if len(i) == 0:
        return sf.idct(c, norm='ortho')
    cr = np.zeros(len(c))
    cr[i] = c[i]
    return sf.idct(cr)


def dct_mae(c, i, y):
    ''' mean abs error from IDCT '''
    cr = np.zeros(len(c))
    cr[i] = c[i]
    yr = sf.idct(cr)
    return myl.mae(yr, y)


def dct_trunc(f, ci, opt):
    r''' indices to truncate DCT output to freq band
    Args:
    f - ndarray, all frequencies
    ci - all indices of coef ndarray
    opt['lb'] - lower cutoff freq
       ['ub'] - upper cutoff freq
    Returns:
    i - ndarray, indices in F of elements to be kept
    '''

    if opt['lb'] > 0:
        ihp = myl.find(f, '>=', opt['lb'])
    else:
        ihp = ci
    if opt['ub'] > 0:
        ilp = myl.find(f, '<=', opt['ub'])
    else:
        ilp = ci
    return myl.intersect(ihp, ilp)


def dct_fbin(f, c, opt):
    r''' frequency bins: symmetric 2-Hz windows around freq integers
    in bandpass overlapped by 1 Hz
    Args:
    f - ndarray frequencies
    c - ndarray coefs
    opt['lb'] - lower and upper truncation freqs
       ['ub']
    Returns:
    fbin - ndarray, lower bnd of freq bins
    cbin - ndarray, summed abs coef values in these bins
    '''

    fb = myl.idx_seg(math.floor(opt['lb']), math.ceil(opt['ub']))
    cbin = np.zeros(len(fb)-1)
    for j in myl.idx_a(len(fb)-1):
        k = myl.intersect(myl.find(f, '>=', fb[j]),
                          myl.find(f, '<=', fb[j+1]))
        cbin[j] = sum(abs(c[k]))

    fbin = fb[myl.idx_a(len(fb)-1)]
    return fbin, cbin


def sig_window(typ, lng=1, par=''):
    ''' wrapper around windows
    Args:
    typ: any type supported by scipy.signal.get_window()
    lng: <1> length
    par: <''> additional parameters as string, scalar, list etc
    Returns:
    window array
    '''

    if typ == 'none' or typ == 'const':
        return np.ones(lng)
    if ((type(par) is str) and (len(par) == 0)):
        return sis.get_window(typ, lng)
    return sis.get_window((typ, par), lng)


def specmom(c, f=[], n=4):
    r''' spectral moments
    Args:
    c - ndarray, coefficients
    f - ndarray, related frequencies <1:len(c)>
    n - number of spectral moments <4>
    Returns:
    m - ndarray moments (increasing)
    '''
    if len(f) == 0:
        f = myl.idx_a(len(c))+1
    c = abs(c)
    s = sum(c)
    k = 0
    m = np.asarray([])
    for i in myl.idx_seg(1, n):
        if s == 0:
            m = np.append(m, 0)
        else:
            m = myl.push(m, sum(c*((f-k)**i))/s)
        k = m[-1]

    return m


def specmom_vec(d, nsm=4):
    ''' spectral moments 1-nsm for each column in matrix d
    Args:
    d: 2-dim np.array
    nsm: number of spectral moments
    Returns:
    sm: dict
      keys: column indices
      values: np.array of nsm spectral moments
    '''

    if len(d) == 0:
        return None

    sm = {}
    for i in myl.idx(np.size(d, 1)):
        sm[i] = specmom(d[:, i], [], nsm)
    return sm
