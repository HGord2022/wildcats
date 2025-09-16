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
import itertools

def summary_stats(ts):
    """
    Calculates lots of (mostly) traditional statistics,
    that are summaries of the site frequency spectrum.

    Arguments
    ---------
    ts: tskit.TreeSequence

    Returns
    ---------
    Nested dictionary of statistics
    """
    pop_dict = {"domestic":0, "scot":1, "captive":2, "eu":3, "lyb":4}

    stats = {
        "diversity": {},
        "segregating_sites": {},
        "tajimas_d": {},
        "divergence": {},
        "relatedness": {},
        "fst": {},
        "f2": {},
        "f3": {},
        "f4": {},
        "y2": {},
        "y3": {},
        "pc1_median":{},
        "pc2_median":{},
        "pc1_iqr":{},
        "pc2_iqr":{},
        "pc1_dist":{},
        "pc2_dist":{}
    }

    # One-way statistics
    for pop_name, pop_num in pop_dict.items():
        stats["diversity"][pop_name] = ts.diversity(sample_sets=ts.samples(population=pop_num))
        stats["segregating_sites"][pop_name] = ts.segregating_sites(sample_sets=ts.samples(population=pop_num))
        stats["tajimas_d"][pop_name] = ts.Tajimas_D(sample_sets=ts.samples(population=pop_num))

    # Two-way statistics (order matters for f2, y2)
    for a, b in itertools.permutations(pop_dict.keys(), 2):
        key = f"{a}_{b}"
        stats["divergence"][key] = ts.divergence(sample_sets=[ts.samples(population=pop_dict[a]),
                                                              ts.samples(population=pop_dict[b])])
        stats["relatedness"][key] = ts.genetic_relatedness(sample_sets=[ts.samples(population=pop_dict[a]),
                                                                        ts.samples(population=pop_dict[b])])
        stats["fst"][key] = ts.Fst(sample_sets=[ts.samples(population=pop_dict[a]),
                                                ts.samples(population=pop_dict[b])])
        stats["f2"][key] = ts.f2(sample_sets=[ts.samples(population=pop_dict[a]),
                                              ts.samples(population=pop_dict[b])])
        stats["y2"][key] = ts.Y2(sample_sets=[ts.samples(population=pop_dict[a]),
                                              ts.samples(population=pop_dict[b])])

    # Three-way statistics
    for a, b, c in itertools.permutations(pop_dict.keys(), 3):
        key = f"{a}_{b}_{c}"
        stats["f3"][key] = ts.f3(sample_sets=[ts.samples(population=pop_dict[a]),
                                              ts.samples(population=pop_dict[b]),
                                              ts.samples(population=pop_dict[c])])
        stats["y3"][key] = ts.Y3(sample_sets=[ts.samples(population=pop_dict[a]),
                                              ts.samples(population=pop_dict[b]),
                                              ts.samples(population=pop_dict[c])])

    # Four-way statistics
    for a, b, c, d in itertools.permutations(pop_dict.keys(), 4):
        key = f"{a}_{b}_{c}_{d}"
        stats["f4"][key] = ts.f4(sample_sets=[ts.samples(population=pop_dict[a]),
                                              ts.samples(population=pop_dict[b]),
                                              ts.samples(population=pop_dict[c]),
                                              ts.samples(population=pop_dict[d])])

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

    pops = ["domestic"] * 6 + ["scot"] * 63 + ["captive"] * 22 + ["eu"] * 15 + ["lyb"] * 4
    pca_df["pop"] = pops

    # PCA stats
    pop_names = ["domestic", "scot", "captive", "eu", "lyb"]

    stats_df = pca_df.groupby("pop").agg((np.median, iqr))
    stats_dict = stats_df.to_dict()

    # individual median and iqr
    for pop in pop_names:
        stats["pc1_median"][pop] = stats_dict[('pc1', 'median')][pop]
        stats["pc2_median"][pop] = stats_dict[('pc2', 'median')][pop]
        stats["pc1_iqr"][pop] = stats_dict[('pc1', 'iqr')][pop]
        stats["pc2_iqr"][pop] = stats_dict[('pc2', 'iqr')][pop]

    # pairwise median comparisons
    for comparison in ["domestic_scot", "domestic_captive", "domestic_eu", "domestic_lyb",
                       "scot_captive", "scot_eu", "scot_lyb", "captive_eu", "captive_lyb", "eu_lyb"]:
        p = comparison.split("_")
        stats["pc1_dist"][comparison] = abs(stats_dict[('pc1', 'median')][p[0]] - stats_dict[('pc1', 'median')][p[1]])
        stats["pc2_dist"][comparison] = abs(stats_dict[('pc2', 'median')][p[0]] - stats_dict[('pc2', 'median')][p[1]])

    ref_table = pd.DataFrame(utils.flatten_dict(stats), index=['i', ])

    array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    filename = f"./output/stats/sumstats{array_id}.pickle"
    print(filename)

    with open(filename, 'wb') as handle:
        pickle.dump(ref_table, handle, protocol=pickle.DEFAULT_PROTOCOL)

    return ref_table
