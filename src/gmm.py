'''
Implementation of Gaussian mixture models using sklearn

Usage:
    python src/gmm.py

Output:
    Saves raw and MLE per gene estimates for transcription unit assignments in output/gmm_assignments/
'''

import os

import numpy as np
from sklearn.mixture import GaussianMixture

from util import load_genome, load_wigs, load_region_reads, get_region_info, get_n_regions, plot_reads, OUTPUT_DIR


if __name__ == '__main__':
    reads = load_wigs()
    genes, _, starts, ends = load_genome()
    window = 21

    fwd_strand = True
    for region in range(get_n_regions(fwd_strand)):
        print(region)

        # Information for region to be analyzed
        x = load_region_reads(reads, region, fwd_strand, ma_window=window, expanded=True)
        start, end, region_genes, region_starts, region_ends = get_region_info(
            region, fwd_strand, genes, starts, ends)
        n_levels = len(region_genes) + 2  # Extra level for 0 reads and spiked reads

        # Fit model
        gmm = GaussianMixture(n_components=n_levels)
        gmm.fit(x)
        labels = gmm.predict(x)
        log_prob = np.log(gmm.predict_proba(x))
        means = gmm.means_.mean(axis=1)

        # Output levels
        levels = np.array([means[l] for l in labels])
        mle_gene_levels = np.zeros_like(levels)
        for s, e in zip(region_starts, region_ends):
            s = s - start
            e = e - start
            label = np.argmax(np.sum(log_prob[s:e, :], axis=0))
            mle_gene_levels[s:e] = means[label]

        # Plot output
        ## Path setup
        out_dir = os.path.join(OUTPUT_DIR, 'gmm_assignments')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        ## Raw level assignments
        out = os.path.join(out_dir, '{}_raw.png'.format(region))
        plot_reads(start, end, genes, starts, ends, reads, fit=levels, path=out)

        ## MLE per gene level assignments
        out = os.path.join(out_dir, '{}_mle.png'.format(region))
        plot_reads(start, end, genes, starts, ends, reads, fit=mle_gene_levels, path=out)
