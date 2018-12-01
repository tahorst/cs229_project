'''
Implementation of a neural net using keras.

Usage:
    python src/nn.py

Output:
    Saves estimates for transcription unit assignments in output/nn_assignments/

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


def get_data(reads, window, regions, pad=0):
    '''
    Uses process_reads to generate training_data for genes that have been
    identified to not have spikes.

    Args:
        reads (2D array of float): reads for each strand at each position
            dims (n_features x genome length)
        window (int): size of sliding window
        regions (iterable): the regions to include in training data

    Returns:
        array of float: 2D array of read data, dims (m samples x n features)
        array of int: 1D array of labels, dims (m samples)
    '''

    data = []
    labels = []

    for region in regions:
        start, end = util.get_region_bounds(region, True)
        initiations, terminations = util.get_labeled_spikes(region, True)

        length = end - start
        n_splits = length - window + 1

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
                if label == 1:
                    print('*** both peaks ***')
                if np.any((terminations >= s + pad) & (terminations <= e - pad)):
                    label = 2
                else:
                    continue

            labels.append(label)
            data.append(reads[:, s:e].reshape(-1))

    labels = keras.utils.np_utils.to_categorical(np.array(labels))
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

def get_fwd_reads(reads, ma_window, mode=1):
    '''
    Get read data for just the forward strand.

    Args:
        reads (2D array of float): reads for each strand at each position
            dims (strands x genome length)
        ma_window (int): window size for taking the moving average
        mode (int): mode of creating data, possible values 0-1

    Returns:
        fwd_reads (array of float): 2D array of raw reads, dims (n_features x genome size)
        fwd_reads_ma (array of float): 2D array of averaged reads, dims (n_features x genome size)
        n_features (int): number of features assembled
    '''

    idx_3p = util.WIG_STRANDS.index('3f')
    idx_5p = util.WIG_STRANDS.index('5f')

    three_prime = reads[idx_3p, :]
    five_prime = reads[idx_5p, :]
    fwd_reads = np.vstack((three_prime, five_prime))

    if mode == 0:
        n_features = 2

        convolution = np.ones((ma_window,)) / ma_window
        fwd_reads_ma = np.vstack((np.convolve(three_prime, convolution, 'same'),
            np.convolve(five_prime, convolution, 'same'))
            )
    elif mode == 1:
        n_features = 4

        pad = (ma_window - 1) // 2
        convolution_back = np.ones((ma_window,)) / (pad + 1)
        convolution_back[-pad:] = 0
        convolution_forward = np.ones((ma_window,)) / (pad + 1)
        convolution_forward[:pad] = 0

        fwd_reads_ma = np.vstack((
            np.convolve(three_prime, convolution_back, 'same'),
            np.convolve(three_prime, convolution_forward, 'same'),
            np.convolve(five_prime, convolution_back, 'same'),
            np.convolve(five_prime, convolution_forward, 'same'),
            ))

    return fwd_reads, fwd_reads_ma, n_features

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

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model

def test_model(model, raw_reads, reads, window, all_reads, genes, starts, ends, tol):
    '''
    Assesses the model performance against test data.  Outputs two plot of probabilities for
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
        test_data, test_labels = get_data(reads, window, [region])
        prediction = model.predict(test_data)
        pad_pred = np.zeros((pad, LABELS))
        pad_pred[:, 0] = 1
        prediction = np.vstack((pad_pred, prediction, pad_pred))

        # Plot outputs
        ## Create directory
        out_dir = os.path.join(util.OUTPUT_DIR, 'nn_assignments')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        ## Plot MSE values with reads
        out = os.path.join(out_dir, '{}_init.png'.format(region))
        util.plot_reads(start, end, genes, starts, ends, all_reads, fit=prediction[:, 1], path=out)
        out = os.path.join(out_dir, '{}_term.png'.format(region))
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
        print('\tInitiations: {}'.format(initiations))
        print('\tTerminations: {}'.format(terminations))
        print('\tAccuracy: {}/{} ({:.1f}%)'.format(correct, n_val, accuracy))
        print('\tFalse positives: {}/{} ({:.1f}%)'.format(wrong, n_test, false_positives))

    return total_correct, total_wrong, total_annotated, total_identified

def summarize(ma_window, window, pad, nodes, correct, wrong, annotated, identified):
    '''
    Print string to stdout and save results in a file.

    Args:
        ma_window (int): size of window for moving average of reads
        window (int): size of window for feature selection
        pad (int): size of samples to ignore when labeling peaks in training data
        nodes (array of int): number of nodes at each hidden layer
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
    print('\nMA window: {}  window: {}  pad: {}  nodes: {}'.format(
        ma_window, window, pad, nodes)
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
        writer.writerow([ma_window, window, pad, nodes, accuracy, false_positive_percent,
            correct, annotated, wrong, identified])

def main(input_dim, hidden_nodes, activation, training_data, training_labels, fwd_reads,
        fwd_reads_ma, window, reads, genes, starts, ends, tol, ma_window, pad):
    '''
    Main function to allow for parallel evaluation of models.
    '''

    # Build neural net
    model = build_model(input_dim, hidden_nodes, activation)

    # Train neural net
    model.fit(training_data, training_labels, epochs=5)

    # Test model on each region
    correct, wrong, annotated, identified = test_model(
        model, fwd_reads, fwd_reads_ma, window, reads, genes, starts, ends, tol)

    # Overall statistics
    summarize(ma_window, window, pad, hidden_nodes, correct, wrong, annotated, identified)


if __name__ == '__main__':
    reads = util.load_wigs()
    genes, _, starts, ends = util.load_genome()

    tol = 5
    fwd_strand = True

    models = np.array([
        [10, 10],
        [10, 10, 10],
        [20, 10, 5],
        [30, 10, 10],
        [3, 10, 5],
        [5, 5],
        [3, 3],
        [32, 16, 8],
        [10, 20, 10],
        [20, 30, 20],
        [30, 20, 10, 5],
        [30, 20, 20, 10],
        ])

    # Write summary headers
    with open(SUMMARY_FILE, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['MA window', 'Window', 'Pad', 'Hidden nodes', 'Accuracy (%)',
            'False Positives (%)', 'Correct', 'Annotated', 'Wrong', 'Identified'])

    for ma_window in [1, 3, 5, 7, 11, 15, 21]:
        fwd_reads, fwd_reads_ma, n_features = get_fwd_reads(reads, ma_window)

        for window in [5, 7, 11, 15, 21, 31]:
            input_dim = n_features * window
            activation = 'sigmoid'

            for pad in range(window // 2 + 1):
                training_data, training_labels = get_data(fwd_reads_ma, window, range(16), pad=pad)

                pool = mp.Pool(processes=mp.cpu_count())
                results = [pool.apply_async(main,
                    (input_dim, hidden_nodes, activation, training_data, training_labels, fwd_reads,
                    fwd_reads_ma, window, reads, genes, starts, ends, tol, ma_window, pad))
                    for hidden_nodes in models]

                pool.close()
                pool.join()

                for result in results:
                    if not result.successful():
                        print('*** Exception in multiprocessing ***')
