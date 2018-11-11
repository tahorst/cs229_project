'''
Implementation of Gaussian mixture models using sklearn

Usage:
    python src/gmm.py
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

from util import load_genome, load_wigs, load_region_reads, get_region_bounds, plot_reads

if __name__ == '__main__':
    reads = load_wigs()
    genes, _, starts, ends = load_genome()
    window = 21

    ### TODO: create for loop over all regions
    # Information for region to be analyzed
    region = 1
    fwd_strand = True
    n_levels = 6
    x = load_region_reads(reads, region, fwd_strand, ma_window=window, expanded=True)
    start, end = get_region_bounds(region, fwd_strand)

    # Fit model
    gmm = GaussianMixture(n_components=n_levels)
    gmm.fit(x)
    labels = gmm.predict(x)
    means = gmm.means_.mean(axis=1)

    # Output levels
    levels = np.array([means[x] for x in labels])

    # Plot output
    plot_reads(start, end, genes, starts, ends, reads, fit=levels)
