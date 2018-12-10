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
- confusion matrix
'''

import csv
from datetime import datetime as dt
import multiprocessing as mp
import os
import time

import keras
import numpy as np

import util


LABELS = 3  # number of labels for output (no spike, initiation, termination)
SUMMARY_FILE = os.path.join(util.OUTPUT_DIR, 'nn_summary_{}.csv'.format(
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
        array of int: 2D array of 1-hot class labels, dims (m samples x n classes)
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
    labels = keras.utils.np_utils.to_categorical(labels, num_classes=LABELS)

    for pos in positions:
        d = reads[:, pos:pos+window]
        if normalize:
            offset = np.mean(d, axis=1).reshape(-1, 1)
            denom = max(np.std(get_region(reads, pos - 50, pos + window + 50, matrix=True)), 1)
            d = (d - offset) / denom
        data.append(d.reshape(-1))

    return np.array(data), labels, positions

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

def validate_model(model, raw_reads, reads, spikes, window, all_reads, genes, starts, ends, tol, normalize, plot_desc=None):
    '''
    Assesses the model performance against validation data.  Outputs two plots of probabilities for
    initiation and termination peaks for each region overlayed on read data to
    output/nn_assignments.  Displays statistics for each region and overall performance.

    Args:
        model (keras.Sequential object): compiled model object
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

    initiations_val, terminations_val = spikes

    # Validate trained model
    x_val, y_val, positions = get_data(reads, window, initiations_val, terminations_val, neg_samples, normalize=normalize)
    start = positions[0]
    end = positions[-1] + 1

    prediction = model.predict(x_val)

    # Plot outputs
    if plot_desc:
        # Create directory
        out_dir = os.path.join(util.OUTPUT_DIR, 'nn_assignments')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Plot softmax values with reads
        out = os.path.join(out_dir, '{}_init.png'.format(plot_desc))
        util.plot_reads(start, end, genes, starts, ends, all_reads, fit=prediction[:, 1], path=out)
        out = os.path.join(out_dir, '{}_term.png'.format(plot_desc))
        util.plot_reads(start, end, genes, starts, ends, all_reads, fit=prediction[:, 2], path=out)

    initiations, terminations = get_spikes(prediction, raw_reads[:, start:end])

    initiations += start
    terminations += start

    n_annotated, n_identified, correct, wrong, accuracy, false_positives = util.get_match_statistics(
        initiations, terminations, initiations_val, terminations_val, tol
        )

    print('\nIdentified init: {}'.format(initiations))
    print('Validation init: {}'.format(initiations_val))
    print('Identified term: {}'.format(terminations))
    print('Validation term: {}'.format(terminations_val))

    return correct, wrong, n_annotated, n_identified

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

def main(input_dim, hidden_nodes, activation, x_train, y_train, fwd_reads,
        fwd_reads_ma, spikes_val, window, reads, genes, starts, ends, tol, ma_window, oversample, normalize,
        pad, plot=False):
    '''
    Main function to allow for parallel evaluation of models.
    '''

    # Build neural net
    model = build_model(input_dim, hidden_nodes, activation)

    # Train neural net
    model.fit(x_train, y_train, epochs=3)

    # Validate model on each region
    if plot:
        plot_desc = '{}_{}_{}'.format(hidden_nodes, window, ma_window)
    else:
        plot_desc = None
    correct, wrong, annotated, identified = validate_model(
        model, fwd_reads, fwd_reads_ma, spikes_val, window, reads, genes, starts, ends, tol, normalize, plot_desc)

    # Overall statistics
    summarize(ma_window, window, pad, hidden_nodes, oversample, correct, wrong, annotated, identified)


if __name__ == '__main__':
    start_time = time.time()
    reads = util.load_wigs()
    genes, _, starts, ends = util.load_genome()
    test = True

    tol = 5
    fwd_strand = True
    plot = False
    down_sample = False
    normalize = False
    read_mode = 1
    parallel = True
    neg_samples = 500

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

    spikes_train = util.get_all_spikes(util.TRAINING)
    spikes_val = util.get_all_spikes(util.VALIDATION)
    spikes_test = util.get_all_spikes(util.TEST)

    # Test optimal hyperparameters
    # TODO: functionalize
    if test:
        ma_window = 15
        window = 5
        pad = 0
        oversample = 2
        hidden_nodes = [20, 30, 20]
        activation = 'sigmoid'

        # Get data
        fwd_reads, fwd_reads_ma, n_features = util.get_fwd_reads(reads, ma_window, mode=read_mode)
        input_dim = n_features * window
        x_train, y_train, _ = get_data(fwd_reads_ma, window, spikes_train[0], spikes_train[1], neg_samples,
            pad=pad, down_sample=down_sample, training=True, normalize=normalize)
        if oversample > 0:
            x_train, y_train = util.oversample(x_train, np.argmax(y_train, axis=1), factor=oversample)
            y_train = keras.utils.np_utils.to_categorical(np.array(y_train), num_classes=LABELS)

        # Build neural net
        model = build_model(input_dim, hidden_nodes, activation)

        # Train neural net
        model.fit(x_train, y_train, epochs=3)

        # Validate model on each region
        plot_desc = 'validation'
        correct, wrong, annotated, identified = validate_model(
            model, fwd_reads, fwd_reads_ma, spikes_val, window, reads, genes, starts, ends, tol, normalize, plot_desc)

        # Print summary
        accuracy = '{:.1f}'.format(correct / annotated * 100)
        if identified > 0:
            false_positive_percent = '{:.1f}'.format(wrong / identified * 100)
        else:
            false_positive_percent = 0

        print('\nMA window: {}  window: {}  pad: {}  nodes: {}  oversample: {}'.format(
            ma_window, window, pad, hidden_nodes, oversample)
            )
        print('Overall accuracy for method: {}/{} ({}%)'.format(
            correct, annotated, accuracy)
            )
        print('Overall false positives for method: {}/{} ({}%)'.format(
            wrong, identified, false_positive_percent)
            )

        # Test model on each region
        plot_desc = 'test'
        correct, wrong, annotated, identified = validate_model(
            model, fwd_reads, fwd_reads_ma, spikes_test, window, reads, genes, starts, ends, tol, normalize, plot_desc)

        # Print summary
        accuracy = '{:.1f}'.format(correct / annotated * 100)
        if identified > 0:
            false_positive_percent = '{:.1f}'.format(wrong / identified * 100)
        else:
            false_positive_percent = 0

        print('\nMA window: {}  window: {}  pad: {}  nodes: {}  oversample: {}'.format(
            ma_window, window, pad, hidden_nodes, oversample)
            )
        print('Overall accuracy for method: {}/{} ({}%)'.format(
            correct, annotated, accuracy)
            )
        print('Overall false positives for method: {}/{} ({}%)'.format(
            wrong, identified, false_positive_percent)
            )

    # Search for optimal hyperparameters
    else:
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
                    x_train, y_train, _ = get_data(fwd_reads_ma, window, spikes_train[0], spikes_train[1], neg_samples,
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
                                fwd_reads_ma, spikes_val, window, reads, genes, starts, ends, tol, ma_window,
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
                                    fwd_reads, fwd_reads_ma, spikes_val, window, reads, genes, starts, ends, tol,
                                    ma_window, oversample, normalize, pad, plot=True)

    print('Completed in {:.1f} min'.format((time.time() - start_time) / 60))
