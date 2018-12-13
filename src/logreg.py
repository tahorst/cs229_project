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
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

import util


LABELS = 3
SUMMARY_FILE = os.path.join(util.OUTPUT_DIR, 'logreg_summary_{}.csv'.format(
    dt.strftime(dt.now(), '%Y%m%d-%H%M%S')))


def get_data(reads, window, initiations, terminations, neg_samples, pad=0,
        down_sample=False, training=False, normalize=False):
    '''
    Generates training, validation or test data with different options.

    Args:
        reads (2D array of float): reads for each strand at each position
            dims (n_features x genome length)
        window (int): size of sliding window
        initiations (array of int): the initiation locations to include in the dataset
        terminations (array of int): the termination locations to include in the dataset
        neg_samples (int): the number of negative samples to include on either side of the spike
        pad (int): size of pad inside the window of positive samples to not include as training
        down_sample (bool): if True, down samples the no spike case because of
            class imbalance
        training (bool): if True, skips regions that would have 2 labels for better training
        normalize (bool) if True, data is normalized by mean and stdev

    Returns:
        array of float: 2D array of read data, dims (m samples x n features)
        array of int: 1D array of labels, dims (m samples)
        array of int: 1D array of genome positions, dims (m samples)
    '''

    def get_region(data, start, end, matrix=False):
        start = max(start, 0)
        end = min(end, util.GENOME_SIZE)
        if matrix:
            return data[:, start:end]
        else:
            return data[start:end]

    data = []
    fwd_strand = True

    spikes = np.sort(np.hstack((initiations, terminations)))
    initiations = set(initiations)
    labels = -np.ones(spikes[-1] + window + neg_samples, dtype=int)

    if not training:
        region = get_region(labels, spikes[0] - window, spikes[-1] + window)
        region[:] = 0

    for spike in spikes:
        if spike in initiations:
            label = 1
        else:
            label = 2

        # Only include base class on either side of spike if not already assigned
        region = get_region(labels, spike - window - neg_samples, spike - window)
        region[region == -1] = 0
        region = get_region(labels, spike, spike + neg_samples)
        region[region == -1] = 0

        # Always exclude padded region around spike
        region = get_region(labels, spike - window, spike - window + pad)
        region[:] = -2
        region = get_region(labels, spike - pad, spike)
        region[:] = -2

        # Exclude if two spikes in the same region for training data
        region = get_region(labels, spike - window + pad, spike - pad)
        if training:
            region[(region > 0) & (region != label)] = -2
            region[region != -2] = label
        else:
            region[region != -2] = label

    positions = np.where(labels[:-window] > -1)[0]
    labels = labels[:-window][labels[:-window] > -1]

    for pos in positions:
        d = reads[:, pos:pos+window]
        if normalize:
            offset = np.mean(d, axis=1).reshape(-1, 1)
            denom = max(np.std(get_region(reads, pos - 50, pos + window + 50, matrix=True)), 1)
            d = (d - offset) / denom
        data.append(d.reshape(-1))

    return np.array(data), labels, positions

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

def train_model(reads, spikes, window, pad, weighted, normalize, oversample):
    '''
    Builds a logistic regression model for initiations and terminations classes.

    Args:
        reads (array of float): 2D array of processed reads, dims (n_features x genome size)
        spikes (tuple array of int): the initiation and termination locations for training
        window (int): size of window used for training data generation
        pad (int): size of pad inside the window of positive samples to not include as training
        weighted (bool): if True, classes are weighted based on samples to address class
            imbalance of positive samples
        normalize (bool): if True, data is normalized by mean and stdev
        oversample (float): if positive, minority classes are oversampled by this factor with SMOTE

    Returns:
        LogisticRegression object: fit logistic regression model for different classes of data
            (initiations and terminations)
    '''

    neg_samples = 500
    x_train, y_train, _ = get_data(reads, window, spikes[0], spikes[1], neg_samples, pad=pad, training=True, normalize=normalize)

    # Oversample minority for class imbalance
    if oversample > 0:
        x_train, y_train = util.oversample(x_train, y_train, factor=oversample)

    # Weight classes differently for class imbalance
    if weighted:
        weights = {i: count for i, count in enumerate(np.bincount(y_train))}
    else:
        weights = None

    logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight=weights, max_iter=200)
    return logreg.fit(x_train, y_train)

def validate_model(model, raw_reads, reads, spikes, window, all_reads, genes, starts, ends, tol, normalize, plot_desc=None, gap=3):
    '''
    Assesses the model performance against validation data.  Outputs two plots for identified
    initiation and termination peaks for each region overlayed on read data to
    output/logreg_assignments.  Displays statistics for each region and overall performance.

    Args:
        model (LogisticRegression object): fit model object
        raw_reads (array of float): 2D array of raw reads, dims (2 x genome size)
        reads (array of float): 2D array of processed reads, dims (n_features x genome size)
        spikes (tuple array of int): the initiation and termination locations for validation
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
        correct (int): total number of correctly identified labeled peaks
        wrong (int): total number of incorrectly identified peaks
        n_annotated (int): total number of labeled peaks
        n_identified (int): total number of identified peaks
    '''

    test_accuracy = True
    neg_samples = 1000
    pad = (window - 1) // 2
    fwd_strand = True
    idx_3p = 0
    idx_5p = 1

    initiations_val, terminations_val = spikes

    # Validate trained model
    x_val, y_val, positions = get_data(reads, window, initiations_val, terminations_val, neg_samples, normalize=normalize)
    start = positions[0]
    end = positions[-1] + 1

    decision = model.decision_function(x_val)
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
        out = os.path.join(out_dir, '{}_init.png'.format(plot_desc))
        util.plot_reads(start, end, genes, starts, ends, all_reads, fit=normalized[:, 1], path=out)
        out = os.path.join(out_dir, '{}_term.png'.format(plot_desc))
        util.plot_reads(start, end, genes, starts, ends, all_reads, fit=normalized[:, 2], path=out)

    n_annotated, n_identified, correct, wrong, recall, precision = util.get_match_statistics(
        initiations, terminations, initiations_val, terminations_val, tol
        )

    print('\nIdentified init: {}'.format(initiations))
    print('Validation init: {}'.format(initiations_val))
    print('Identified term: {}'.format(terminations))
    print('Validation term: {}'.format(terminations_val))

    return correct, wrong, n_annotated, n_identified

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

    recall = '{:.1f}'.format(correct / annotated * 100)
    if identified > 0:
        precision = '{:.1f}'.format(correct / identified * 100)
    else:
        precision = 0

    # Standard out
    print('\nMA window: {}  window: {}  pad: {}'.format(ma_window, window, pad))
    print('Overall recall for method: {}/{} ({}%)'.format(
        correct, annotated, recall)
        )
    print('Overall precision for method: {}/{} ({}%)'.format(
        correct, identified, precision)
        )

    # Save in summary file
    with open(SUMMARY_FILE, 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([ma_window, window, pad, recall, precision,
            correct, annotated, wrong, identified])

def main(reads, ma_reads, all_reads, ma_window, window, pad, spikes, class_weighted,
        oversample, normalize, tol, genes, starts, ends, plot):
    '''
    Main function to allow for parallel evaluation of models.
    '''

    # Train model on training regions
    logreg = train_model(ma_reads, spikes['train'], window, pad, class_weighted, normalize, oversample)

    # Validate model on other regions
    if plot:
        desc = '{}_{}_{}'.format(ma_window, window, pad)
    else:
        desc = None
    correct, wrong, annotated, identified = validate_model(logreg, reads, ma_reads,
        spikes['validation'], window, all_reads, genes, starts, ends, tol, normalize, plot_desc=desc)

    # Print out summary statistics
    summarize(ma_window, window, pad, correct, wrong, annotated, identified)


if __name__ == '__main__':
    start_time = time.time()
    reads = util.load_wigs()
    genes, _, starts, ends = util.load_genome()
    test = True

    # Hyperparameters
    class_weighted = False
    oversample = 0
    normalize = True
    read_mode = 0
    tol = 5
    plot = True

    fwd_strand = True

    spikes = {
        'train': util.get_all_spikes(util.TRAINING),
        'validation': util.get_all_spikes(util.VALIDATION),
        'test': util.get_all_spikes(util.TEST),
        }

    # Test optimal hyperparameters
    # TODO: functionalize
    if test:
        ma_window = 1
        window = 7
        pad = 3
        fwd_reads, fwd_reads_ma, n_features = util.get_fwd_reads(reads, ma_window, mode=read_mode)

        # Train model
        logreg = train_model(fwd_reads_ma, spikes['train'], window, pad, class_weighted, normalize, oversample)

        # Validation data
        desc = 'validation'
        correct, wrong, annotated, identified = validate_model(logreg, fwd_reads, fwd_reads_ma,
            spikes['validation'], window, reads, genes, starts, ends, tol, normalize, plot_desc=desc)

        recall = '{:.1f}'.format(correct / annotated * 100)
        if identified > 0:
            precision = '{:.1f}'.format(correct / identified * 100)
        else:
            precision = 0

        # Standard out
        print('\nMA window: {}  window: {}  pad: {}'.format(ma_window, window, pad))
        print('Overall recall for method: {}/{} ({}%)'.format(
            correct, annotated, recall)
            )
        print('Overall precision for method: {}/{} ({}%)'.format(
            correct, identified, precision)
            )

        # Test data
        desc = 'test'
        correct, wrong, annotated, identified = validate_model(logreg, fwd_reads, fwd_reads_ma,
            spikes['test'], window, reads, genes, starts, ends, tol, normalize, plot_desc=desc)

        recall = '{:.1f}'.format(correct / annotated * 100)
        if identified > 0:
            precision = '{:.1f}'.format(correct / identified * 100)
        else:
            precision = 0

        # Standard out
        print('\nMA window: {}  window: {}  pad: {}'.format(ma_window, window, pad))
        print('Overall recall for method: {}/{} ({}%)'.format(
            correct, annotated, recall)
            )
        print('Overall precision for method: {}/{} ({}%)'.format(
            correct, identified, precision)
            )

    # Search for optimal hyperparameters
    else:
        # Write summary headers
        with open(SUMMARY_FILE, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['MA window', 'Window', 'Pad', 'Recall (%)', 'Precision (%)',
                'Correct', 'Annotated', 'Wrong', 'Identified', util.get_git_hash()])

        for ma_window in [1, 3, 5, 7, 11, 15, 21]:
            fwd_reads, fwd_reads_ma, n_features = util.get_fwd_reads(reads, ma_window, mode=read_mode)

            for window in [3, 5, 7, 11, 15, 21]:
                for pad in range(window // 2 + 1):
                    main(fwd_reads, fwd_reads_ma, reads, ma_window, window, pad, spikes,
                        class_weighted, oversample, normalize, tol, genes, starts, ends, plot)

    print('Completed in {:.1f} min'.format((time.time() - start_time) / 60))
