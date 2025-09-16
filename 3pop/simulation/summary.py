import pandas as pd
import numpy as np
import pyslim
import tskit
import utils
import random
import pickle
import os
from statistics import mean
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import iqr
from scipy.spatial import procrustes

def summary_stats(ts):
    """
    Caclulates lots of (mostly) traditional statistics,
    that are summaries of the site frequency spectrum.

    Arguments
    ---------
    ts: tskit TreeSequence

    Returns
    ---------
    DataFrame of statistics
    """
    pop_dict = {"domestic": 0, "scot": 1, "captive": 2}

    stats = {
        "diversity": {},
        "segregating_sites": {},
        "tajimas_d": {},
        "divergence": {},
        "relatedness": {},
        "fst": {},
        "f2": {},
        "f3": {},
        "y2": {},
        "y3": {},
        "pc1_median": {},
        "pc2_median": {},
        "pc1_iqr": {},
        "pc2_iqr": {},
        "pc1_dist": {},
        "pc2_dist": {}
    }

    # One way statistics
    for pop_name, pop_num in pop_dict.items():
        stats["diversity"][pop_name] = ts.diversity(sample_sets=ts.samples(population=pop_num))
        stats["segregating_sites"][pop_name] = ts.segregating_sites(sample_sets=ts.samples(population=pop_num))
        stats["tajimas_d"][pop_name] = ts.Tajimas_D(sample_sets=ts.samples(population=pop_num))

    # Two way statistics
    for comparison in ["domestic_scot", "domestic_captive", "scot_captive"]:
        p = comparison.split("_")
        stats["divergence"][comparison] = ts.divergence(sample_sets=[ts.samples(population=pop_dict[p[0]]),
                                                                     ts.samples(population=pop_dict[p[1]])])
        stats["relatedness"][comparison] = ts.genetic_relatedness(sample_sets=[ts.samples(population=pop_dict[p[0]]),
                                                                               ts.samples(population=pop_dict[p[1]])])
        stats["fst"][comparison] = ts.Fst(sample_sets=[ts.samples(population=pop_dict[p[0]]),
                                                       ts.samples(population=pop_dict[p[1]])])
        stats["f2"][comparison] = ts.f2(sample_sets=[ts.samples(population=pop_dict[p[0]]),
                                                     ts.samples(population=pop_dict[p[1]])])
        stats["y2"][comparison] = ts.Y2(sample_sets=[ts.samples(population=pop_dict[p[0]]),
                                                     ts.samples(population=pop_dict[p[1]])])

    # Three way statistics (3 unique triplets)
    triplets = [
        ("domestic", "scot", "captive"),
        ("scot", "domestic", "captive"),
        ("captive", "domestic", "scot"),
    ]

    for p in triplets:
        key = "_".join(p)
        stats["f3"][key] = ts.f3(sample_sets=[ts.samples(population=pop_dict[p[0]]),
                                              ts.samples(population=pop_dict[p[1]]),
                                              ts.samples(population=pop_dict[p[2]])])
        stats["y3"][key] = ts.Y3(sample_sets=[ts.samples(population=pop_dict[p[0]]),
                                              ts.samples(population=pop_dict[p[1]]),
                                              ts.samples(population=pop_dict[p[2]])])

    ##### PCA summary stats #####
    genotype = ts.genotype_matrix()

    # convert from 01 to 012
    samples = np.shape(genotype)[1]
    ones = []
    twos = []

    for col in range(0, samples):
        if col % 2 == 0:
            ones.append(col)
        else:
            twos.append(col)

    matrix_012 = np.add(genotype[:, ones], genotype[:, twos])

    # scaled PCA
    standardizedData = StandardScaler().fit_transform(matrix_012.T)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X=standardizedData)

    pca_df = pd.DataFrame(principalComponents, columns=['pc1', 'pc2'])
    
    pops = ["domestic"] * 6 + ["scot"] * 63 + ["captive"] * 21
    pca_df["pop"] = pops

    # PCA stats
    pop_names = ["domestic", "scot", "captive"]

    stats_df = pca_df.groupby("pop").agg((np.median, iqr))
    stats_dict = stats_df.to_dict()

    # individual median and iqr
    for pop in pop_names:
        stats["pc1_median"][pop] = stats_dict[('pc1', 'median')][pop]
        stats["pc2_median"][pop] = stats_dict[('pc2', 'median')][pop]
        stats["pc1_iqr"][pop] = stats_dict[('pc1', 'iqr')][pop]
        stats["pc2_iqr"][pop] = stats_dict[('pc2', 'iqr')][pop]

    # pairwise median comparisons
    for comparison in ["domestic_scot", "domestic_captive", "scot_captive"]:
        p = comparison.split("_")
        stats["pc1_dist"][comparison] = abs(stats_dict[('pc1', 'median')][p[0]] - stats_dict[('pc1', 'median')][p[1]])
        stats["pc2_dist"][comparison] = abs(stats_dict[('pc2', 'median')][p[0]] - stats_dict[('pc2', 'median')][p[1]])

    ref_table = pd.DataFrame(utils.flatten_dict(stats), index=['i', ])

    array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    filename = "./output/stats/sumstats%s.pickle" % array_id
    print(filename)

    with open(filename, 'wb') as handle:
        pickle.dump(ref_table, handle, protocol=pickle.DEFAULT_PROTOCOL)

    return ref_table
