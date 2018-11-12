'''
Implementation of a hidden Markov model using hmmlearn.

Usage:
    python src/hmm.py

Output:
    Saves estimates for transcription unit assignments in output/hmm_assignments/
'''

from __future__ import division

import os

from hmmlearn.hmm import GaussianHMM
import numpy as np

from util import load_genome, load_wigs, load_region_reads, get_region_info, plot_reads, OUTPUT_DIR


VERBOSE = True


if __name__ == '__main__':
    reads = load_wigs()
    genes, _, starts, ends = load_genome()
    window = 21
    n_iters = 10

     ### TODO: create for loop over all regions
    # Information for region to be analyzed
    region = 1
    fwd_strand = True
    x = load_region_reads(reads, region, fwd_strand, ma_window=window, expanded=False)
    start, end, region_genes, region_starts, region_ends = get_region_info(
        region, fwd_strand, genes, starts, ends)
    n_levels = 2*len(region_genes) + 2  # Extra level for 0 reads and spiked reads

    # Model parameters
    start_prob = np.zeros(n_levels)
    start_prob[0] = 1
    trans_prob_up = n_levels / (end - start)
    trans_prob_down = 0.5
    trans_mat = np.zeros((n_levels, n_levels))
    for n in range(n_levels-1):
        if n % 2:
            trans_mat[n, n:n+2] = [1-trans_prob_down, trans_prob_down]
        else:
            trans_mat[n, n:n+2] = [1-trans_prob_up, trans_prob_up]
    trans_mat[-1, -1] = 1

    for it in range(n_iters):
        # Fit model
        hmm = GaussianHMM(n_components=n_levels, init_params='mc')
        hmm.startprob_ = start_prob
        hmm.transmat_ = trans_mat
        hmm.fit(x)
        labels = hmm.predict(x)
        # log_prob = np.log(gmm.predict_proba(x))
        means = hmm.means_.mean(axis=1)
        score = hmm.score(x)

        if VERBOSE:
            if hmm.monitor_.converged:
                print('Converged with score: {:.2f}'.format(score))
            else:
                print('** Did not converge **')

        # Output levels
        levels = np.array([means[l] for l in labels])

        # Plot output
        ## Path setup
        out_dir = os.path.join(OUTPUT_DIR, 'hmm_assignments')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        ## Raw level assignments
        out = os.path.join(out_dir, '{}_iter_{}.png'.format(region, it))
        plot_reads(start, end, genes, starts, ends, reads, fit=levels, path=out)
