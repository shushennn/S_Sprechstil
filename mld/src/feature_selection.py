#!/usr/bin/env python3

from collections import defaultdict
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import mylib as myl
import sys
sys.path.insert(0, '/homes/reichelu/repos/repo_pl/src/copasul_py/')


def select_uncorrelated_features(df, opt={}, do_plot=True):
    ''' clustering of correlated features.
    Per cluster only the feature with the highest variance is kept
    Args:
    df: audb dataframe
    opt: (dict)
      plot: (str) of plot output file
      depth: (int) cluster tree depth; the higher the fewer clusters
            and returned features
    Returns:
      selected_features: (list) of selected column names
          to be applied outside of functions by df_sel = df[selected_features]
    '''

    opt = myl.opt_default(opt, {"plot": "/tmp/featclust.png", "depth": 1})

    df.fillna(0, inplace=True)
    cols = df.columns.to_list()
    X = df.to_numpy()
    var = np.var(X, axis=0)

    corr = spearmanr(X).correlation
    corr_linkage = hierarchy.ward(corr)

    # do_plot=True

    if do_plot:
        fig, ax1 = plt.subplots(1, 1, figsize=(18, 12))
        # formerly: dendro =
        _ = hierarchy.dendrogram(corr_linkage,
                                 labels=cols, ax=ax1,
                                 leaf_rotation=90)
        # dendro_idx = np.arange(0, len(dendro['ivl']))
        fig.tight_layout()
        fig.savefig(opt["plot"])
        print("plot written to", opt["plot"])
        plt.show()

    # the higher the depth value, the fewer clusters
    depth = opt["depth"]
    cluster_ids = hierarchy.fcluster(corr_linkage, depth,
                                     criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)

    # f.a. cluster: keep feature with max variance
    selected_features = []

    for clst in cluster_id_to_feature_ids:
        # print(clst)
        # for i in cluster_id_to_feature_ids[clst]:
        #    print("\t{}".format(cols[i]))
        ii = cluster_id_to_feature_ids[clst]
        i_best = ii[np.argmax(var[ii])]
        # print("\t{}".format(cols[i_best]))
        # selected_features_i.append(i_best)
        selected_features.append(cols[i_best])

    return selected_features
