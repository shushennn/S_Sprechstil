import myUtilHL as myl
import re
import copy as cp

# TextGrid output of dict read in by i_tg()
# (appended if file exists, else from scratch)
# IN:
#   tg dict
#   f fileName
# OUT:
#   intoFile


def o_tg(tg, fil):
    h = open(fil, mode='w', encoding='utf-8')
    idt = '    '
    fld = tg_fields()
    # head
    if tg['format'] == 'long':
        h.write("File type = \"ooTextFile\"\nObject class = \"TextGrid\"\n")
        h.write("xmin = {}\n".format(tgv(tg['head']['xmin'], 'xmin')))
        h.write("xmax = {}\n".format(tgv(tg['head']['xmax'], 'xmax')))
        h.write("tiers? <exists>\n")
        h.write("size = {}\n".format(tgv(tg['head']['size'], 'size')))
    else:
        h.write("File type = \"ooTextFile\"\nObject class = \"TextGrid\"\n")
        h.write("{}\n".format(tgv(tg['head']['xmin'], 'xmin')))
        h.write("{}\n".format(tgv(tg['head']['xmax'], 'xmax')))
        h.write("<exists>\n")
        h.write("{}\n".format(tgv(tg['head']['size'], 'size')))

    # item
    if (tg['format'] == 'long'):
        h.write("item []:\n")

    for i in myl.numkeys(tg['item']):
        # subkey := intervals or points?
        if re.search(tg['item'][i]['class'], 'texttier', re.I):
            subkey = 'points'
        else:
            subkey = 'intervals'
        if tg['format'] == 'long':
            h.write("{}item [{}]:\n".format(idt, i))
        for f in fld['item']:
            if tg['format'] == 'long':
                if f == 'size':
                    h.write("{}{}{}: size = {}\n".format(
                        idt, idt, subkey, tgv(tg['item'][i]['size'], 'size')))
                else:
                    h.write("{}{}{} = {}\n".format(
                        idt, idt, f, tgv(tg['item'][i][f], f)))
            else:
                h.write("{}\n".format(tgv(tg['item'][i][f], f)))

        # empty tier
        if subkey not in tg['item'][i]:
            continue
        for j in myl.numkeys(tg['item'][i][subkey]):
            if tg['format'] == 'long':
                h.write("{}{}{} [{}]:\n".format(idt, idt, subkey, j))
            for f in fld[subkey]:
                if (tg['format'] == 'long'):
                    myv = tgv(tg['item'][i][subkey][j][f], f)
                    h.write(
                        "{}{}{}{} = {}\n".format(idt, idt, idt, f, myv))
                else:
                    myv = tgv(tg['item'][i][subkey][j][f], f)
                    h.write("{}\n".format(myv))
    h.close()


# returns field names of TextGrid head and items
# OUT:
#   hol fieldNames
def tg_fields():
    return {'head': ['xmin', 'xmax', 'size'],
            'item': ['class', 'name', 'xmin', 'xmax', 'size'],
            'points': ['time', 'mark'],
            'intervals': ['xmin', 'xmax', 'text']}


# rendering of TextGrid values
# IN:
#   s value
#   s attributeName
# OUT:
#   s renderedValue
def tgv(v, a):
    if re.search('(xmin|xmax|time|size)', a):
        return v
    else:
        return "\"{}\"".format(v)

# returns tier subdict from TextGrid
# IN:
#   tg: dict by i_tg()
#   tn: name of tier
# OUT:
#   t: dict tier (deepcopy)


def tg_tier(tg, tn):
    if tn not in tg['item_name']:
        return {}
    return cp.deepcopy(tg['item'][tg['item_name'][tn]])

# returns list of TextGrid tier names
# IN:
#   tg: textgrid dict
# OUT:
#   tn: sorted list of tiernames


def tg_tn(tg):
    return sorted(list(tg['item_name'].keys()))

# returns tier type
# IN:
#   t: tg tier (by tg_tier())
# OUT:
#   typ: 'points'|'intervals'|''


def tg_tierType(t):
    for x in ['points', 'intervals']:
        if x in t:
            return x
    return ''

# returns text field name according to tier type
# IN:
#   typ: tier type returned by tg_tierType(myTier)
# OUT:
#   'points'|<'text'>


def tg_txtField(typ):
    if typ == 'points':
        return 'mark'
    return 'text'

# transforms TextGrid tier to 2 arrays
# point -> 1 dim + lab
# interval -> 2 dim (one row per segment) + lab
# IN:
#   t: tg tier (by tg_tier())
#   opt dict
#       .skip <""> regular expression for labels of items to be skipped
#             if empty, only empty items will be skipped
# OUT:
#   x: 1- or 2-dim array of time stamps
#   lab: corresponding labels
# REMARK:
#   empty intervals are skipped


def tg_tier2tab(t, opt={}):
    opt = myl.opt_default(opt, {"skip": ""})
    if len(opt["skip"]) > 0:
        do_skip = True
    else:
        do_skip = False
    x = myl.ea()
    lab = []
    if 'intervals' in t:
        for i in myl.numkeys(t['intervals']):
            z = t['intervals'][i]
            if len(z['text']) == 0:
                continue
            if do_skip and re.search(opt["skip"], z["text"]):
                continue

            x = myl.push(x, [z['xmin'], z['xmax']])
            lab.append(z['text'])
    else:
        for i in myl.numkeys(t['points']):
            z = t['points'][i]
            if do_skip and re.search(opt["skip"], z["mark"]):
                continue
            x = myl.push(x, z['time'])
            lab.append(z['mark'])
    return x, lab


# transforms table to TextGrid tier
# IN:
#    t - numpy 1- or 2-dim array with time info
#    lab - list of labels <[]>
#    specs['class'] <'IntervalTier' for 2-dim, 'TextTier' for 1-dim>
#         ['name']
#         ['xmin'] <0>
#         ['xmax'] <max tab>
#         ['size'] - will be determined automatically
#         ['lab_pau'] - <''>
# OUT:
#    dict tg tier (see i_tg() subdict below myItemIdx)
# for 'interval' tiers gaps between subsequent intervals will be bridged
# by lab_pau
def tg_tab2tier(t, lab, specs):
    tt = {'name': specs['name']}
    nd = myl.ndim(t)
    # 2dim array with 1 col
    if nd == 2:
        nd = myl.ncol(t)
    # tier class
    if nd == 1:
        tt['class'] = 'TextTier'
        tt['points'] = {}
    else:
        tt['class'] = 'IntervalTier'
        tt['intervals'] = {}
        # pause label for gaps between intervals
        if 'lab_pau' in specs:
            lp = specs['lab_pau']
        else:
            lp = ''
    # xmin, xmax
    if 'xmin' not in specs:
        tt['xmin'] = 0
    else:
        tt['xmin'] = specs['xmin']
    if 'xmax' not in specs:
        if nd == 1:
            tt['xmax'] = t[-1]
        else:
            tt['xmax'] = t[-1, 1]
    else:
        tt['xmax'] = specs['xmax']
    # point tier content
    if nd == 1:
        for i in myl.idx_a(len(t)):
            # point tier content might be read as [[x],[x],[x],...]
            # or [x,x,x,...]
            if myl.listType(t[i]):
                z = t[i, 0]
            else:
                z = t[i]
            if len(lab) == 0:
                myMark = "x"
            else:
                myMark = lab[i]
            tt['points'][i+1] = {'time': z, 'mark': myMark}
        tt['size'] = len(t)
    # interval tier content
    else:
        j = 1
        # initial pause
        if t[0, 0] > tt['xmin']:
            tt['intervals'][j] = {'xmin': tt['xmin'],
                                  'xmax': t[0, 0], 'text': lp}
            j += 1
        for i in myl.idx_a(len(t)):
            # pause insertions
            if ((j-1 in tt['intervals']) and
                    t[i, 0] > tt['intervals'][j-1]['xmax']):
                tt['intervals'][j] = {'xmin': tt['intervals'][j-1]['xmax'],
                                      'xmax': t[i, 0], 'text': lp}
                j += 1
            if len(lab) == 0:
                myMark = "x"
            else:
                myMark = lab[i]
            tt['intervals'][j] = {'xmin': t[i, 0],
                                  'xmax': t[i, 1], 'text': myMark}
            j += 1
        # final pause
        if tt['intervals'][j-1]['xmax'] < tt['xmax']:
            tt['intervals'][j] = {'xmin': tt['intervals'][j-1]['xmax'],
                                  'xmax': tt['xmax'], 'text': lp}
            j += 1  # so that uniform 1 subtraction for size
        # size
        tt['size'] = j-1
    return tt

# add tier to TextGrid
# IN:
#   tg dict from i_tg(); can be empty dict
#   tier subdict to be added:
#       same dict form as in i_tg() output, below 'myItemIdx'
#   opt
#      ['repl'] <True> - replace tier of same name
# OUT:
#   tg updated
# REMARK:
#   !if generated from scratch head xmin and xmax are taken over from the tier
#    which might need to be corrected afterwards!


def tg_add(tg, tier, opt={'repl': True}):

    # from scratch
    if 'item_name' not in tg:
        fromScratch = True
        tg = {'name': '', 'format': 'long', 'item_name': {}, 'item': {},
              'head': {'size': 0, 'xmin': 0, 'xmax': 0, 'type': 'ooTextFile'}}
    else:
        fromScratch = False

    # tier already contained?
    if (opt['repl'] and (tier['name'] in tg['item_name'])):
        i = tg['item_name'][tier['name']]
        tg['item'][i] = tier
    else:
        # item index
        ii = myl.numkeys(tg['item'])
        if len(ii) == 0:
            i = 1
        else:
            i = ii[-1]+1
        tg['item_name'][tier['name']] = i
        tg['item'][i] = tier
        tg['head']['size'] += 1

    if fromScratch and 'xmin' in tier:
        for x in ['xmin', 'xmax']:
            tg['head'][x] = tier[x]

    return tg
