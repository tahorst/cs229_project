'''
Implementation of a neural net using keras.

Usage:
    python src/nn.py

Output:
    Saves estimates for transcription unit assignments in output/nn_assignments/
    Saves statistics in output/nn_summary_yyyymmdd-hhmmss.csv

Parameters to search:

TODO:
- LOOCV
'''

import csv
from datetime import datetime as dt
import multiprocessing as mp
import os

import keras
import numpy as np

import util


LABELS = 3  # number of labels for output (no spike, initiation, termination)
SUMMARY_FILE = os.path.join(util.OUTPUT_DIR, 'nn_summary_{}.csv'.format(
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
        array of int: 2D array of 1-hot class labels, dims (m samples x n classes)
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
            if np.any((initiations >= s) & (initiations <= e)):
                if np.any((initiations >= s + pad) & (initiations <= e - pad)):
                    label = 1
                else:
                    continue
            if np.any((terminations >= s) & (terminations <= e)):
                if np.any((terminations >= s + pad) & (terminations <= e - pad)):
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

    labels = keras.utils.np_utils.to_categorical(np.array(labels), num_classes=LABELS)
    return np.array(data), labels

def get_spikes(prob, reads, gap=3):
    '''
    Identifies the initiation and termination spikes from the model output.

    Args:
        prob (array of float): 2D array of probabilities for each category,
            dims (m samples x LABELS)
        reads (array of float): 2D array of reads for the 3' and 5' strand
            dims: (2 x region size)
        gap (int): minimum genome position gap between identified positions to call a distinct group

    Returns:
        initiations (array of int): positions where transcription initiation occurs
        terminations (array of int): positions where transcription termination occurs

    TODO:
        use cutoff instead of argmax?
    '''

    def group_spikes(spike_locations, strand):
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
            data = reads[strand, group]
            true_spike = np.where(data == data.max())[0]

            position = group[true_spike[0]]
            true_spikes.append(position)

        return true_spikes

    labels = np.argmax(prob, axis=1)
    init_locations = np.where(labels == 1)[0]
    term_locations = np.where(labels == 2)[0]

    # Add 1 to locations for actual genome location because of 0 indexing
    initiations = np.array(group_spikes(init_locations, 1)) + 1
    terminations = np.array(group_spikes(term_locations, 0)) + 1

    return initiations, terminations

def build_model(input_dim, hidden_nodes, activation):
    '''
    Builds a neural net model

    Args:
        input_dim (int): number of dimensions of the input data
        hidden_nodes (array of int): number of nodes at each hidden layer
        activation (str): activation type

    Returns:
        model (keras.Sequential object): compiled model object
    '''

    model = keras.Sequential()
    model.add(keras.layers.Dense(hidden_nodes[0], input_dim=input_dim, activation=activation))
    for nodes in hidden_nodes[1:]:
        model.add(keras.layers.Dense(nodes, activation=activation))
    model.add(keras.layers.Dense(LABELS, activation='softmax'))
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def test_model(model, raw_reads, reads, window, all_reads, genes, starts, ends, tol, normalize, plot_desc=None):
    '''
    Assesses the model performance against test data.  Outputs two plots of probabilities for
    initiation and termination peaks for each region overlayed on read data to
    output/nn_assignments.  Displays statistics for each region and overall performance.

    Args:
        model (keras.Sequential object): compiled model object
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

    Returns:
        total_correct (int): total number of correctly identified labeled peaks
        total_wrong (int): total number of incorrectly identified peaks
        total_annotated (int): total number of labeled peaks
        total_identified (int): total number of identified peaks
    '''

    test_accuracy = True
    pad = (window - 1) // 2
    fwd_strand = True

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
        prediction = model.predict(x_test)
        pad_pred = np.zeros((pad, LABELS))
        pad_pred[:, 0] = 1
        prediction = np.vstack((pad_pred, prediction, pad_pred))

        # Plot outputs
        if plot_desc:
            # Create directory
            out_dir = os.path.join(util.OUTPUT_DIR, 'nn_assignments')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # Plot softmax values with reads
            desc = '{}_{}'.format(region, plot_desc)
            out = os.path.join(out_dir, '{}_init.png'.format(desc))
            util.plot_reads(start, end, genes, starts, ends, all_reads, fit=prediction[:, 1], path=out)
            out = os.path.join(out_dir, '{}_term.png'.format(desc))
            util.plot_reads(start, end, genes, starts, ends, all_reads, fit=prediction[:, 2], path=out)

        initiations, terminations = get_spikes(prediction, raw_reads[:, start:end])

        initiations += start
        terminations += start

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

def summarize(ma_window, window, pad, nodes, oversample, correct, wrong, annotated, identified):
    '''
    Print string to stdout and save results in a file.

    Args:
        ma_window (int): size of window for moving average of reads
        window (int): size of window for feature selection
        pad (int): size of samples to ignore when labeling peaks in training data
        nodes (array of int): number of nodes at each hidden layer
        oversample (float): if positive, minority classes are oversampled by this factor with SMOTE
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
    print('\nMA window: {}  window: {}  pad: {}  nodes: {}  oversample: {}'.format(
        ma_window, window, pad, nodes, oversample)
        )
    print('Overall accuracy for method: {}/{} ({}%)'.format(
        correct, annotated, accuracy)
        )
    print('Overall false positives for method: {}/{} ({}%)'.format(
        wrong, identified, false_positive_percent)
        )

    # Save in summary file
    with open(SUMMARY_FILE, 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([ma_window, window, pad, nodes, oversample, accuracy, false_positive_percent,
            correct, annotated, wrong, identified])

def main(input_dim, hidden_nodes, activation, training_data, training_labels, fwd_reads,
        fwd_reads_ma, window, reads, genes, starts, ends, tol, ma_window, oversample, normalize,
        pad, plot=False):
    '''
    Main function to allow for parallel evaluation of models.
    '''

    # Build neural net
    model = build_model(input_dim, hidden_nodes, activation)

    # Train neural net
    model.fit(training_data, training_labels, epochs=3)

    # Test model on each region
    if plot:
        plot_desc = '{}_{}_{}'.format(hidden_nodes, window, ma_window)
    else:
        plot_desc = None
    correct, wrong, annotated, identified = test_model(
        model, fwd_reads, fwd_reads_ma, window, reads, genes, starts, ends, tol, normalize, plot_desc)

    # Overall statistics
    summarize(ma_window, window, pad, hidden_nodes, oversample, correct, wrong, annotated, identified)


if __name__ == '__main__':
    reads = util.load_wigs()
    genes, _, starts, ends = util.load_genome()

    tol = 5
    fwd_strand = True
    plot = False
    down_sample = False
    normalize = False
    read_mode = 0
    parallel = True

    models = np.array([
        [30, 20, 20, 10],
        [30, 20, 10, 5],
        [20, 30, 20],
        [32, 16, 8],
        [30, 10, 10],
        [10, 20, 10],
        [20, 10, 5],
        [10, 10, 10],
        [10, 10],
        [5, 5],
        ])

    # Write summary headers
    with open(SUMMARY_FILE, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['MA window', 'Window', 'Pad', 'Hidden nodes', 'Oversample', 'Accuracy (%)',
            'False Positives (%)', 'Correct', 'Annotated', 'Wrong', 'Identified', util.get_git_hash()])

    for ma_window in [1, 3, 5, 7, 11, 15, 21]:
        fwd_reads, fwd_reads_ma, n_features = util.get_fwd_reads(reads, ma_window, mode=read_mode)

        for window in [5, 7, 11, 15, 21, 31]:
            input_dim = n_features * window
            activation = 'sigmoid'

            for pad in range(window // 2 + 1):
                x_train, y_train = get_data(fwd_reads_ma, window, range(16),
                    pad=pad, down_sample=down_sample, training=True, normalize=normalize)

                # Oversample minority for class imbalance
                for oversample in [0, 2, 5, 10, 20]:
                    if oversample > 0:
                        x_train, y_train = util.oversample(x_train, np.argmax(y_train, axis=1), factor=oversample)
                        y_train = keras.utils.np_utils.to_categorical(np.array(y_train), num_classes=LABELS)

                    if parallel:
                        pool = mp.Pool(processes=mp.cpu_count())
                        results = [pool.apply_async(main,
                            (input_dim, hidden_nodes, activation, x_train, y_train, fwd_reads,
                            fwd_reads_ma, window, reads, genes, starts, ends, tol, ma_window,
                            oversample, normalize, pad, plot))
                            for hidden_nodes in models]

                        pool.close()
                        pool.join()

                        for result in results:
                            if not result.successful():
                                print('*** Exception in multiprocessing ***')
                    else:
                        for hidden_nodes in models:
                            main(input_dim, hidden_nodes, activation, x_train, y_train,
                                fwd_reads, fwd_reads_ma, window, reads, genes, starts, ends, tol,
                                ma_window, oversample, normalize, pad, plot=True)
