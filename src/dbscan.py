'''
Implementation of DBSCAN for clustering and outlier detection using sklearn.

Usage:
    python src/dbscan.py

Output:
    Saves estimates for transcription unit assignments in output/hmm_assignments/
'''

import os

from sklearn.cluster import DBSCAN
import numpy as np

from util import load_genome, load_wigs, load_region_reads, get_region_info, plot_reads, OUTPUT_DIR


if __name__ == '__main__':
    reads = load_wigs()
    genes, _, starts, ends = load_genome()
    window = 1
    expanded = False
    total = True

     ### TODO: create for loop over all regions
    # Information for region to be analyzed
    region = 1
    fwd_strand = True
    print('\nRegion: {}'.format(region))
    x = load_region_reads(reads, region, fwd_strand, ma_window=window, expanded=expanded, total=total)
    start, end, region_genes, region_starts, region_ends = get_region_info(
        region, fwd_strand, genes, starts, ends)

    # Model parameters
    for eps in [0.1, 0.2, 0.4, 0.7, 1, 1.2, 1.5, 2, 2.5, 3, 3.5, 4, 5]:
        print('  eps: {}'.format(eps))
        for min_samples in [2, 3, 5, 7, 10, 12, 15]:
            print('    min_samples: {}'.format(min_samples))
            # Fit model
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
            labels = clustering.labels_

            # Plot output
            ## Path setup
            out_dir = os.path.join(OUTPUT_DIR, 'dbscan_assignments')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            ## Raw level assignments
            out = os.path.join(out_dir, '{}_cluster_{}_{}.png'.format(region, eps, min_samples))
            plot_reads(start, end, genes, starts, ends, reads, fit=clustering.labels_+2, path=out)
