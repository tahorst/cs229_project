'''
Implementation of a hidden Markov model using hmmlearn.

Usage:
    python src/hmm.py

Output:
    Saves estimates for transcription unit assignments in output/hmm_assignments/

Parameters to search:
    n_levels (int): number of levels to assign, varies depending on genes in a region
    trans_prob_up (float): value between 0 and 1 for the transition probability to a
        spiked level
    trans_prob_down (float): value between 0 and 1 for the transition probability to
        a gene level
    window (int): moving average window for smoothing inputs
    expanded (bool): True if using wider feature set (within window surrounding point,
        False if only the single point
    n_iter (int): number of iterations for the GaussianHMM method

TODO:
'''

import csv
from datetime import datetime as dt
import os

from hmmlearn.hmm import GaussianHMM
import numpy as np

import util


VERBOSE = True
SUMMARY_FILE = os.path.join(util.OUTPUT_DIR, 'hmm_summary_{}.csv'.format(
    dt.strftime(dt.now(), '%Y%m%d-%H%M%S')))


def get_spikes(levels, reads, tol=8):
    '''
    Identifies spikes from level assignments from the HMM.

    Args:
        levels (array of float): array of mean of assigned hidden state at each position,
            can be None if algorithm did not converge or had errors
        reads (array of float): 2D array of reads for the 3' and 5' strand,
            dims: (2 x region size)
        tol (int): maximum length of a spike to label it a spike

    Returns:
        initiations (array of int): positions where transcription initiation occurs
        terminations (array of int): positions where transcription termination occurs
    '''

    # Arrays to return
    initiations = []
    terminations = []

    # Arrays for identifying spike regions
    spike_starts = []
    spikes_ends = []
    current_spike = False

    if levels is not None:
        changes = np.where(levels[1:] - levels[:-1] != 0)[0]
        if len(changes) > 0:
            diff = [changes[i+1] - changes[i] for i in range(len(changes) - 1)]
            diff.append(len(levels) - changes[-1])

            # Identify ranges that contain a spike
            for c, d in zip(changes, diff):
                if d < tol and not current_spike:
                    spike_starts.append(c)
                    current_spike = True
                elif d > tol and current_spike:
                    spikes_ends.append(c)
                    current_spike = False

            # Identify one position from each group
            for start, stop in zip(spike_starts, spikes_ends):
                data = reads[:, start:stop]
                strand, true_spike = np.where(data == data.max())

                # Add 2, 1 for 0 indexing and 1 for shift with changes calculation
                position = start + true_spike[0] + 2
                if strand[0]:
                    initiations.append(position)
                else:
                    terminations.append(position)

    return np.array(initiations), np.array(terminations)


if __name__ == '__main__':
    reads = util.load_wigs()
    genes, _, starts, ends = util.load_genome()
    test = True

    idx_3p = util.WIG_STRANDS.index('3f')
    idx_5p = util.WIG_STRANDS.index('5f')
    three_prime = reads[idx_3p, :]
    five_prime = reads[idx_5p, :]
    fwd_reads = np.vstack((three_prime, five_prime))

    # Parameters
    test_accuracy = True
    total = True
    n_seeds = 10
    n_iter = 20
    tol = 5

    series_list = [range(33)]  # Validation set of regions

    # Evaluate validation and test sets with optimal parameters
    if test:
        expanded_list = [False]
        window_list = [1]
        series_list.append(range(33, 53))
    # Search for optimal parameters with validation set
    else:
        expanded_list = [False, True]
        window_list = [1, 3, 5, 9, 15, 21]

    with open(SUMMARY_FILE, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['MA window', 'Expanded', 'Accuracy (%)', 'False Positives (%)',
            'Correct', 'Annotated', 'Wrong', 'Identified', util.get_git_hash()])

        fwd_strand = True
        for series in series_list:
            for expanded in expanded_list:
                for window in window_list:
                    total_correct = 0
                    total_wrong = 0
                    total_annotated = 0
                    total_identified = 0

                    for region in series:
                        initiations_val, terminations_val = util.get_labeled_spikes(region, fwd_strand)

                        # Skip if only testing region with annotations
                        if test_accuracy and len(initiations_val) == 0 and len(terminations_val) == 0:
                            continue

                        print('\nRegion: {}'.format(region))

                        # Information for region to be analyzed
                        x = util.load_region_reads(reads, region, fwd_strand, ma_window=window, expanded=expanded, total=total)
                        start, end, region_genes, region_starts, region_ends = util.get_region_info(
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

                        best_score = -np.inf
                        best_fit = None
                        for it in range(n_seeds):
                            try:
                                # Fit model
                                hmm = GaussianHMM(n_components=n_levels, init_params='mc', n_iter=n_iter)
                                hmm.startprob_ = start_prob
                                hmm.transmat_ = trans_mat
                                hmm.fit(x)
                                labels = hmm.predict(x)
                                means = hmm.means_.mean(axis=1)
                                score = hmm.score(x)

                                if VERBOSE:
                                    if hmm.monitor_.converged:
                                        print('Iteration {:2d} converged in {:2d} steps with score: {:.2f}'.format(
                                            it, hmm.monitor_.iter, score))
                                    else:
                                        print('** Did not converge **')

                                # Output levels
                                levels = np.array([means[l] for l in labels])
                                if score > best_score:
                                    best_score = score
                                    best_fit = levels

                                # Plot output
                                ## Path setup
                                out_dir = os.path.join(util.OUTPUT_DIR, 'hmm_assignments')
                                if not os.path.exists(out_dir):
                                    os.makedirs(out_dir)

                                ## Raw level assignments
                                out = os.path.join(out_dir, '{}_iter_{}.png'.format(region, it))
                                util.plot_reads(start, end, genes, starts, ends, reads, fit=levels, path=out)
                            except Exception as e:
                                print(e)
                                continue

                        initiations, terminations = get_spikes(best_fit, fwd_reads[:, start:end])
                        initiations += start
                        terminations += start

                        n_val, n_test, correct, wrong, accuracy, false_positives = util.get_match_statistics(
                            initiations, terminations, initiations_val, terminations_val, tol
                            )

                        # Region statistics
                        print('\tIdentified: {}   {}'.format(initiations, terminations))
                        print('\tValidation: {}   {}'.format(initiations_val, terminations_val))
                        print('\tAccuracy: {}/{} ({:.1f}%)'.format(correct, n_val, accuracy))
                        print('\tFalse positives: {}/{} ({:.1f}%)'.format(wrong, n_test, false_positives))

                        total_annotated += n_val
                        total_identified += n_test
                        total_correct += correct
                        total_wrong += wrong

                    # Summary
                    accuracy = total_correct / total_annotated * 100
                    false_positives = total_wrong / total_identified * 100
                    print('Overall accuracy for method: {}/{} ({:.1f}%)'.format(
                        total_correct, total_annotated, accuracy)
                        )
                    print('Overall false positives for method: {}/{} ({:.1f}%)'.format(
                        total_wrong, total_identified, false_positives)
                        )

                    writer.writerow([window, expanded, accuracy, false_positives,
                        total_correct, total_annotated, total_wrong, total_identified])
