'''
Implementation of DBSCAN for clustering and outlier detection using sklearn.

Usage:
    python src/dbscan.py

Output:
    Saves estimates for transcription unit assignments in output/dbscan_assignments/
    Saves summary for parameters in output/dbscan_summary_yyyymmdd-hhmmss.csv
'''

import csv
from datetime import datetime as dt
import os
import time

import numpy as np
from sklearn.cluster import DBSCAN

import util


SUMMARY_FILE = os.path.join(util.OUTPUT_DIR, 'dbscan_summary_{}.csv'.format(
    dt.strftime(dt.now(), '%Y%m%d-%H%M%S')))


def get_spikes(outliers, reads, tol=8):
    '''
    Identifies spikes from outliers from DBSCAN.

    Args:
        outliers (array of int): array of positions of outliers
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

    if len(outliers) > 0:
        diff = [outliers[i+1] - outliers[i] for i in range(len(outliers) - 1)]
        diff.append(len(reads) - outliers[-1])

        # Identify ranges that contain a spike
        for c, d in zip(outliers, diff):
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

            # Add 1 for 0 indexing
            position = start + true_spike[0] + 1
            if strand[0]:
                initiations.append(position)
            else:
                terminations.append(position)

    return np.array(initiations), np.array(terminations)


if __name__ == '__main__':
    start_time = time.time()
    reads = util.load_wigs()
    genes, _, starts, ends = util.load_genome()
    test = True

    window = 1
    tol = 5
    expanded = False
    total = True
    test_accuracy = True

    fwd_strand = True

    series_list = [range(33)]  # Validation set of regions

    # Evaluate validation and test sets with optimal parameters
    if test:
        eps_list = [2]
        min_samples_list = [15]
        series_list.append(range(33, 53))
    # Search for optimal parameters with validation set
    else:
        eps_list = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5]
        min_samples_list = [2, 3, 5, 7, 10, 15]

    for series in series_list:
        stats = {}
        for region in series:
            initiations_val, terminations_val = util.get_labeled_spikes(region, fwd_strand)

            # Skip if only testing region with annotations
            if test_accuracy and len(initiations_val) == 0 and len(terminations_val) == 0:
                continue

            print('\nRegion: {}'.format(region))
            x = util.load_region_reads(reads, region, fwd_strand, ma_window=window, expanded=expanded, total=total)
            start, end, region_genes, region_starts, region_ends = util.get_region_info(
                region, fwd_strand, genes, starts, ends)

            # Model parameters
            for eps in eps_list:
                print('  eps: {}'.format(eps))
                for min_samples in min_samples_list:
                    pair = (eps, min_samples)
                    if pair not in stats:
                        stats[pair] = {
                            'annotated': 0,
                            'identified': 0,
                            'correct': 0,
                            'wrong': 0,
                            }
                    print('    min_samples: {}'.format(min_samples))
                    # Fit model
                    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
                    labels = clustering.labels_

                    # Plot output
                    ## Path setup
                    out_dir = os.path.join(util.OUTPUT_DIR, 'dbscan_assignments')
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)

                    ## Raw level assignments
                    out = os.path.join(out_dir, '{}_cluster_{}_{}.png'.format(region, eps, min_samples))
                    util.plot_reads(start, end, genes, starts, ends, reads, fit=clustering.labels_+2, path=out)

                    initiations, terminations = get_spikes(np.where(labels == -1)[0], reads[(0,2), start:end])
                    initiations += start
                    terminations += start

                    n_val, n_test, correct, wrong, accuracy, false_positives = util.get_match_statistics(
                        initiations, terminations, initiations_val, terminations_val, tol
                    )
                    stats[pair]['annotated'] += n_val
                    stats[pair]['identified'] += n_test
                    stats[pair]['correct'] += correct
                    stats[pair]['wrong'] += wrong

                    # Region statistics
                    print('\tIdentified: {}   {}'.format(initiations, terminations))
                    print('\tValidation: {}   {}'.format(initiations_val, terminations_val))
                    print('\tAccuracy: {}/{} ({:.1f}%)'.format(correct, n_val, accuracy))
                    print('\tFalse positives: {}/{} ({:.1f}%)'.format(wrong, n_test, false_positives))

        # Save summary output statistics
        with open(SUMMARY_FILE, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['eps', 'min samples', 'Accuracy (%)', 'False Positives (%)',
                'Correct', 'Annotated', 'Wrong', 'Identified', util.get_git_hash()])\

            for (eps, min_samples), d in stats.items():
                correct = d['correct']
                wrong = d['wrong']
                identified = d['identified']
                annotated = d['annotated']
                accuracy = '{:.1f}'.format(correct / annotated * 100)
                if identified > 0:
                    false_positive_percent = '{:.1f}'.format(wrong / identified * 100)
                else:
                    false_positive_percent = 0

                print('Overall Accuracy: {}/{} ({}%)'.format(correct, annotated, accuracy))
                print('Overall False positives: {}/{} ({}%)'.format(wrong, identified, false_positive_percent))
                writer.writerow([eps, min_samples, accuracy, false_positive_percent,
                    correct, annotated, wrong, identified])

    print('Completed in {:.1f} min'.format((time.time() - start_time) / 60))
