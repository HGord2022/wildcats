"""
Module contains assortment of functions for helping handle simulated data, files, tests etc.
"""

import pandas as pd
import numpy as np
import msprime
from pyarrow.lib import ArrowIOError
import tskit
import random


def flatten_dict(d, sep='_'):
    """
    Recursively flattens a nested dictionary, concatenating the outer and inner keys.

    Arguments
    -----------
    stats_dict: A nested dictionary of statistics
    sep: seperator for keys

    Returns
    ------------
    dict
    """

    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value).items():
                    yield key + sep + subkey, subvalue
            else:
                yield key, value

    return dict(items())
    

def mac_filter(ts, count):
    '''
    Minor allele count filter for tree sequence.
    ts: tree sequence
    count: minimum minor allele count for site to be kept
    returns: list of site IDs to remove
    '''
    variant = tskit.Variant(ts)
    mac_remove = []

    for site_id in range(ts.num_sites):
        variant.decode(site_id)
        if sum(variant.genotypes) not in range(count,(len(variant.genotypes)-count)):
            mac_remove.append(site_id)

    return mac_remove


def thinning(ts, window):
    '''
    Carries out thinning of sites, keeping one site per window size.
    ts: tree sequence
    window: window size
    returns: list of site IDs to remove
    '''
    window = 2000
    ids = []
    batch = []
    positions = zip(range(1,ts.num_sites),ts.tables.sites.position)
    for position in positions:
        if position[1] < window:
            batch.append(position[0])
        elif len(batch) == 0:
            window = window+window
        else:
            window = window+window
            ids.append(random.choice(batch))
            batch = []
    return list(set(range(0,ts.num_sites))-set(ids))
