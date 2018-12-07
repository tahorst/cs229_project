'''
Implementation of logistic regression model using sklearn

Usage:
    python src/logreg.py

Output:
    Saves expected initiations and terminations in output/logreg_assignments/
    Saves statistics in output/logreg_summary_yyyymmdd-hhmmss.csv
'''

import csv
from datetime import datetime as dt
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

import util


LABELS = 3
SUMMARY_FILE = os.path.join(util.OUTPUT_DIR, 'logreg_summary_{}.csv'.format(
    dt.strftime(dt.now(), '%Y%m%d-%H%M%S')))


def get_data(reads, window, regions, pad=0, down_sample=False, training=False, normalize=False):
    '''
    Generates training or test data with different options.

    Args:
        reads (2D array of float): reads for each strand at each position
            dims (n_features x genome length)
        window (int): size of sliding window
        regions (iterable): the regions to include in training data
        pad (int): size of pad inside the window of positive samples to not include as training
        down_sample (bool): if True, down samples the no spike case because of
            class imbalance
        training (bool): if True, skips regions that would have 2 labels for better training
        normalize (bool) if True, data is normalized by mean and stdev

    Returns:
        array of float: 2D array of read data, dims (m samples x n features)
        array of int: 1D array of labels, dims (m samples)
    '''

    data = []
    labels = []
    fwd_strand = True

    for region in regions:
        start, end = util.get_region_bounds(region, fwd_strand)
        initiations, terminations = util.get_labeled_spikes(region, fwd_strand)

        length = end - start
        n_splits = length - window + 1

        denom = np.std(reads[:, start:end])

        for i in range(n_splits):
            s = start + i
            e = s + window

            label = 0
            if np.any((initiations >= s) & (initiations < e)):
                if np.any((initiations >= s + pad) & (initiations < e - pad)):
                    label = 1
                else:
                    continue
            if np.any((terminations >= s) & (terminations < e)):
                if np.any((terminations >= s + pad) & (terminations < e - pad)):
                    if label == 1:
                        # Exclude regions that have both an initiation and termination from training
                        if training:
                            continue
                        else:
                            print('*** both peaks ***')
                    label = 2
                else:
                    continue

            # Down sample the cases that do not have a spike because of class imbalance
            if down_sample:
                if not (np.any((s > terminations) & (s < terminations + window*100))
                        or np.any((e < initiations) & (e > initiations - window*100))
                        or label != 0):
                    continue

            labels.append(label)
            d = reads[:, s-1:e-1]
            if normalize:
                offset = np.mean(d, axis=1).reshape(-1, 1)
                d = (d - offset) / denom
            data.append(d.reshape(-1))

    return np.array(data), np.array(labels)

def get_spikes(labels, reads, gap=3):
    '''
    Identifies the initiation and termination spikes from the model output.

    Args:
        labels (array of int): array labels (0 or 1) for each sample (position)
        reads (array of float): array of reads for the strand of interest in a given region
            (5' for initiations, 3' for terminations)
        gap (int): minimum genome position gap between identified positions to call a distinct group

    Returns:
        spikes (array of int): positions of spikes
    '''

    def group_spikes(spike_locations, reads):
        true_spikes = []
        n_spike_locations = len(spike_locations)

        # Identify groups of near consecutive positions that are above the cutoff
        if n_spike_locations == 0:
            groups = []
        elif n_spike_locations == 1:
            groups = [np.array([spike_locations[0]])]
        else:
            groups = []
            current_group = [spike_locations[0]]
            for loc in spike_locations[1:]:
                if loc - current_group[-1] > gap:
                    groups.append(np.array(current_group))
                    current_group = []
                current_group.append(loc)
            groups.append(np.array(current_group))

        # Select one point from each group to be the true initiation or termination
        for group in groups:
            data = reads[group]
            true_spike = np.where(data == data.max())[0]

            position = group[true_spike[0]]
            true_spikes.append(position)

        return true_spikes

    locations = np.where(labels == 1)[0]

    # Add 1 to locations for actual genome location because of 0 indexing
    spikes = np.array(group_spikes(locations, reads)) + 1

    return spikes

def train_model(reads, window, pad, training_range, weighted, normalize, oversample):
    '''
    Builds a logistic regression model for initiations and terminations classes.

    Args:
        reads (array of float): 2D array of processed reads, dims (n_features x genome size)
        window (int): size of window used for training data generation
        pad (int): size of pad inside the window of positive samples to not include as training
        training_range (iterable of int): regions to use as training data
        weighted (bool): if True, classes are weighted based on samples to address class
            imbalance of positive samples
        normalize (bool): if True, data is normalized by mean and stdev
        oversample (float): if positive, minority classes are oversampled by this factor with SMOTE

    Returns:
        LogisticRegression object: fit logistic regression model for different classes of data
            (initiations and terminations)
    '''

    x_train, y_train = get_data(reads, window, training_range, pad=pad, training=True, normalize=normalize)

    # Oversample minority for class imbalance
    if oversample > 0:
        x_train, y_train = util.oversample(x_train, y_train, factor=oversample)

    # Weight classes differently for class imbalance
    if weighted:
        weights = {i: count for i, count in enumerate(np.bincount(y_train))}
    else:
        weights = None

    logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight=weights)
    return logreg.fit(x_train, y_train)

def test_model(model, raw_reads, reads, window, all_reads, genes, starts, ends, tol, normalize, plot_desc=None, gap=3):
    '''
    Assesses the model performance against test data.  Outputs two plots for identified
    initiation and termination peaks for each region overlayed on read data to
    output/logreg_assignments.  Displays statistics for each region and overall performance.

    Args:
        model (LogisticRegression object): fit model object
        raw_reads (array of float): 2D array of raw reads, dims (2 x genome size)
        reads (array of float): 2D array of processed reads, dims (n_features x genome size)
        window (int): size of window used for training data generation
        all_reads (2D array of float): reads for each strand at each position
            dims (strands x genome length)
        genes (array of str): names of genes
        starts (array of int): start position for each gene
        ends (array of int): end position for each gene
        tol (int): distance assigned peak can be from labeled peak to call correct
        normalize (bool) if True, data is normalized by mean and stdev
        plot_desc (str): if None, will not output the plot to files, otherwise creates file names
            starting with this string, can be buggy if used in multiprocessing
        gap (int): gap between unique spike identifications

    Returns:
        total_correct (int): total number of correctly identified labeled peaks
        total_wrong (int): total number of incorrectly identified peaks
        total_annotated (int): total number of labeled peaks
        total_identified (int): total number of identified peaks
    '''

    test_accuracy = True
    pad = (window - 1) // 2
    fwd_strand = True
    idx_3p = 0
    idx_5p = 1

    total_correct = 0
    total_wrong = 0
    total_annotated = 0
    total_identified = 0

    for region in range(16, util.get_n_regions(fwd_strand)):
        initiations_val, terminations_val = util.get_labeled_spikes(region, fwd_strand)

        # Skip if only testing region with annotations
        if test_accuracy and len(initiations_val) == 0 and len(terminations_val) == 0:
            continue

        print('\nRegion: {}'.format(region))

        start, end = util.get_region_bounds(region, fwd_strand)

        # Test trained model
        x_test, y_test = get_data(reads, window, [region], normalize=normalize)

        decision = model.decision_function(x_test)
        pad_pred = np.zeros((pad, LABELS))
        pad_pred[:, 0] = 1
        decision = np.vstack((pad_pred, decision, pad_pred))

        y_pred = np.argmax(decision, axis=1)
        init_pred = np.array(y_pred == 1, int)
        term_pred = np.array(y_pred == 2, int)

        initiations = get_spikes(init_pred, raw_reads[idx_5p, start:end], gap) + start
        terminations = get_spikes(term_pred, raw_reads[idx_3p, start:end], gap) + start

        # Plot outputs
        if plot_desc:
            # Create directory
            out_dir = os.path.join(util.OUTPUT_DIR, 'logreg_assignments')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # Plot softmax values with reads
            normalized = -(decision / np.min(decision)) + 1
            desc = '{}_{}'.format(region, plot_desc)
            out = os.path.join(out_dir, '{}_init.png'.format(desc))
            util.plot_reads(start, end, genes, starts, ends, all_reads, fit=normalized[:, 1], path=out)
            out = os.path.join(out_dir, '{}_term.png'.format(desc))
            util.plot_reads(start, end, genes, starts, ends, all_reads, fit=normalized[:, 2], path=out)

        n_val, n_test, correct, wrong, accuracy, false_positives = util.get_match_statistics(
            initiations, terminations, initiations_val, terminations_val, tol
            )
        total_annotated += n_val
        total_identified += n_test
        total_correct += correct
        total_wrong += wrong

        # Region statistics
        print('\tIdentified: {}   {}'.format(initiations, terminations))
        print('\tValidation: {}   {}'.format(initiations_val, terminations_val))
        print('\tAccuracy: {}/{} ({:.1f}%)'.format(correct, n_val, accuracy))
        print('\tFalse positives: {}/{} ({:.1f}%)'.format(wrong, n_test, false_positives))

    return total_correct, total_wrong, total_annotated, total_identified

def summarize(ma_window, window, pad, correct, wrong, annotated, identified):
    '''
    Print string to stdout and save results in a file.

    Args:
        ma_window (int): size of window for moving average of reads
        window (int): size of window for feature selection
        pad (int): size of samples to ignore when labeling peaks in training data
        correct (int): number of correctly identified peaks
        wrong (int): number of incorrectly identified peaks
        annotated (int): number of peaks that have been annotated
        identified (int): number of peaks that were identified by the algorithm
    '''

    accuracy = '{:.1f}'.format(correct / annotated * 100)
    if identified > 0:
        false_positive_percent = '{:.1f}'.format(wrong / identified * 100)
    else:
        false_positive_percent = 0

    # Standard out
    print('\nMA window: {}  window: {}  pad: {}'.format(ma_window, window, pad))
    print('Overall accuracy for method: {}/{} ({}%)'.format(
        correct, annotated, accuracy)
        )
    print('Overall false positives for method: {}/{} ({}%)'.format(
        wrong, identified, false_positive_percent)
        )

    # Save in summary file
    with open(SUMMARY_FILE, 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([ma_window, window, pad, accuracy, false_positive_percent,
            correct, annotated, wrong, identified])

def main(reads, ma_reads, all_reads, ma_window, window, pad, training_range, class_weighted,
        oversample, normalize, tol, genes, starts, ends, plot):
    '''
    Main function to allow for parallel evaluation of models.
    '''

    # Train model on training regions
    logreg = train_model(ma_reads, window, pad, training_range, class_weighted, normalize, oversample)

    # Test model on other regions
    if plot:
        desc = '{}_{}_{}'.format(ma_window, window, pad)
    else:
        desc = None
    correct, wrong, annotated, identified = test_model(logreg, reads, ma_reads,
        window, all_reads, genes, starts, ends, tol, normalize, plot_desc=desc)

    # Print out summary statistics
    summarize(ma_window, window, pad, correct, wrong, annotated, identified)


if __name__ == '__main__':
    reads = util.load_wigs()
    genes, _, starts, ends = util.load_genome()

    # Hyperparameters
    class_weighted = True
    oversample = 10
    normalize = False
    read_mode = 0
    tol = 5
    plot = True
    training_range = range(16)

    fwd_strand = True

    # Write summary headers
    with open(SUMMARY_FILE, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['MA window', 'Window', 'Pad', 'Accuracy (%)', 'False Positives (%)',
            'Correct', 'Annotated', 'Wrong', 'Identified', util.get_git_hash()])

    for ma_window in [1, 3, 5, 7, 11, 15, 21]:
        fwd_reads, fwd_reads_ma, n_features = util.get_fwd_reads(reads, ma_window, mode=read_mode)

        for window in [3, 5, 7, 11, 15, 21]:
            for pad in range(window // 2 + 1):
                main(fwd_reads, fwd_reads_ma, reads, ma_window, window, pad, training_range,
                    class_weighted, oversample, normalize, tol, genes, starts, ends, plot)
