'''
Implementation of a autoencoder neural net using keras for outlier detection.

Usage:
    python src/ae.py

Output:
    Saves estimates for transcription unit assignments in output/ae_assignments/

Parameters to search:
'''

import csv
from datetime import datetime as dt
import os

import keras
import numpy as np

import util


SUMMARY_FILE = os.path.join(util.OUTPUT_DIR, 'ae_summary_{}.csv'.format(
    dt.strftime(dt.now(), '%Y%m%d-%H%M%S')))


def process_reads(reads, start, end, window):
    '''
    Processes the reads to feed to the neural network.
    Stacks the 3' and 5' reads and will slide a window to create new samples
    for every section in the region of interest.  Within each section, the
    samples are normalized to be between 0 and 1.

    Args:
        reads (array of float): 2D array of reads for the 3' and 5' strand
            dims: (2 x genome size)
        start (int): start position in the genome
        end (int): end position in the genome
        window (int): size of sliding window

    Returns:
        list of arrays of float: each array will be a sample with processed reads
    '''

    data = []

    length = end - start
    n_splits = length - window + 1

    for i in range(n_splits):
        s = start + i
        e = s + window
        data.append(np.hstack((reads[0, s:e], reads[1, s:e])))
        max_read = data[-1].max()
        if max_read < 10:
            max_read = 10
        data[-1] /= max_read

    return data

def get_training_data(reads, genes, starts, ends, window):
    '''
    Uses process_reads to generate training_data for genes that have been
    identified to not have spikes.

    Args:
        reads (2D array of float): reads for each strand at each position
            dims (strands x genome length)
        genes (array of str): names of genes
        starts (array of int): start position for each gene
        ends (array of int): end position for each gene
        window (int): size of sliding window

    Returns:
        array of float: 2D array of read data, dims (m samples x n features)
    '''

    stop_gene = 'cdaR'
    excluded = {'yaaY', 'ribF', 'ileS', 'folA', 'djlA', 'yabP', 'mraZ', 'ftsI', 'ftsW', 'ddlB', 'ftsA', 'yadD', 'hrpB'}

    data = []

    for gene, start, end in zip(genes, starts, ends):
        if gene in excluded:
            continue

        start += 20
        end -= 20
        data += process_reads(reads, start, end, window)

        if gene == stop_gene:
            break

    return np.array(data)

def get_spikes(mse, reads, cutoff, gap=3):
    '''
    Identifies the initiation and termination spikes from the model output.

    Args:
        mse (array of float): mean square error for each position in the region
        reads (array of float): 2D array of reads for the 3' and 5' strand
            dims: (2 x region size)
        cutoff (float): MSE value cutoff to identify as a peak
        gap (int): minimum genome position gap between regions of MSE above the
            cutoff to call a distinct group

    Returns:
        initiations (array of int): positions where transcription initiation occurs
        terminations (array of int): positions where transcription termination occurs
    '''

    initiations = []
    terminations = []

    spike_locations = np.where(mse > cutoff)[0]
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
        data = reads[:, group]
        strand, true_spike = np.where(data == data.max())

        position = group[true_spike[0]]
        if strand[0]:
            initiations.append(position)
        else:
            terminations.append(position)

    # Add 1 to locations for actual genome location because of 0 indexing
    return np.array(initiations) + 1, np.array(terminations) + 1

def get_fwd_reads(reads, ma_window):
    '''
    Get read data for just the forward strand.

    Args:
        reads (2D array of float): reads for each strand at each position
            dims (strands x genome length)
        ma_window (int): window size for taking the moving average

    Returns:
        fwd_reads (array of float): 2D array of raw reads, dims (n_features x genome size)
        fwd_reads_ma (array of float): 2D array of averaged reads, dims (n_features x genome size)
        n_features (int): number of features assembled
    '''

    n_features = 2

    convolution = np.ones((ma_window,)) / ma_window
    idx_3p = util.WIG_STRANDS.index('3f')
    idx_5p = util.WIG_STRANDS.index('5f')

    three_prime = reads[idx_3p, :]
    five_prime = reads[idx_5p, :]
    fwd_reads = np.vstack((three_prime, five_prime))
    fwd_reads_ma = np.vstack((np.convolve(three_prime, convolution, 'same'),
        np.convolve(five_prime, convolution, 'same'))
        )

    return fwd_reads, fwd_reads_ma, n_features

def build_model(input_dim, hidden_nodes, activation):
    '''
    Builds an autoencoder neural net model

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
    model.add(keras.layers.Dense(input_dim, activation='sigmoid'))
    model.summary()

    model.compile(optimizer='adam', loss='mse')

    return model

def test_model(model, ma_reads, reads, window, all_reads, genes, starts, ends, tol, cutoff):
    '''
    Assesses the model performance against test data.  Outputs a plot of mean square error within
    each region overlayed on read data to output/ae_assignments.  Displays statistics for each
    region and overall performance.

    Args:
        model (keras.Sequential object): compiled model object
        ma_reads (array of float): 2D array of averaged reads, dims (n_features x genome size)
        reads (array of float): 2D array of raw reads, dims (n_features x genome size)
        window (int): size of window used for training data generation
        all_reads (2D array of float): reads for each strand at each position
            dims (strands x genome length)
        genes (array of str): names of genes
        starts (array of int): start position for each gene
        ends (array of int): end position for each gene
        tol (int): distance assigned peak can be from labeled peak to call correct
        cutoff (float): cutoff of MSE value for labelling a peak
    '''

    pad = (window - 1) // 2
    fwd_strand = True

    total_correct = 0
    total_wrong = 0
    total_annotated = 0
    total_identified = 0

    for region in range(util.get_n_regions(fwd_strand)):
        initiations_val, terminations_val = util.get_labeled_spikes(region, fwd_strand)

        # Skip if only testing region with annotations
        if test_accuracy and len(initiations_val) == 0 and len(terminations_val) == 0:
            continue

        print('\nRegion: {}'.format(region))

        start, end = util.get_region_bounds(region, fwd_strand)

        # Test trained model
        test_data = np.array(process_reads(ma_reads, start, end, window))
        prediction = model.predict(test_data)
        mse = np.mean((test_data - prediction)**2, axis=1)
        mse = np.hstack((np.zeros(pad), mse, np.zeros(pad)))

        # Plot outputs
        ## Create directory
        out_dir = os.path.join(util.OUTPUT_DIR, 'ae_assignments')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        ## Plot MSE values with reads
        out = os.path.join(out_dir, '{}.png'.format(region))
        util.plot_reads(start, end, genes, starts, ends, all_reads, fit=mse/cutoff/np.e, path=out)

        initiations, terminations = get_spikes(mse, reads[:, start:end], cutoff)

        initiations += start
        terminations += start

        # Determine accuracy of peak identification
        # TODO: functionalize in util
        n_val = len(initiations_val) + len(terminations_val)
        n_test = len(initiations) + len(terminations)
        correct = 0
        for val in initiations_val:
            for test in initiations:
                if np.abs(val-test) < tol:
                    correct += 1
                    break
        for val in terminations_val:
            for test in terminations:
                if np.abs(val-test) < tol:
                    correct += 1
                    break
        wrong = n_test - correct

        total_annotated += n_val
        total_identified += n_test
        total_correct += correct
        total_wrong += wrong

        if n_val > 0:
            accuracy = correct / n_val * 100
        else:
            accuracy = 0

        if n_test > 0:
            false_positives = wrong / n_test * 100
        else:
            false_positives = 0

        # Region statistics
        print('\tInitiations: {}'.format(initiations))
        print('\tTerminations: {}'.format(terminations))
        print('\tAccuracy: {}/{} ({:.1f}%)'.format(correct, n_val, accuracy))
        print('\tFalse positives: {}/{} ({:.1f}%)'.format(wrong, n_test, false_positives))

    return total_correct, total_wrong, total_annotated, total_identified

def summarize(ma_window, window, cutoff, nodes, correct, wrong, annotated, identified):
    '''
    Print string to stdout and save results in a file.

    Args:
        ma_window (int): size of window for moving average of reads
        window (int): size of window for feature selection
        cutoff (float): cutoff of MSE value for labelling a peak
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
    print('\nMA window: {}  window: {}  cutoff: {}'.format(ma_window, window, cutoff))
    print('Overall accuracy for method: {}/{} ({}%)'.format(
        correct, annotated, accuracy)
    )
    print('Overall false positives for method: {}/{} ({}%)'.format(
        wrong, identified, false_positive_percent)
    )

    # Save in summary file
    with open(SUMMARY_FILE, 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([ma_window, window, cutoff, nodes, accuracy, false_positive_percent,
            correct, annotated, wrong, identified])


if __name__ == '__main__':
    reads = util.load_wigs()
    genes, _, starts, ends = util.load_genome()
    forward = starts > 0

    test_accuracy = True
    tol = 5

    models = np.array([
        [12, 6, 3, 2, 3, 6, 12],
        [12, 6, 3, 6, 12],
        [16, 8, 4, 2, 4, 8, 16],
        [16, 8, 4, 3, 4, 8, 16],
        [16, 12, 8, 4, 2, 4, 8, 12, 16],
        [8, 2, 8],
        [8, 3, 8],
        [8, 4, 8],
        [8, 4, 3, 4, 8],
        [8, 4, 2, 4, 8],
        ])

    # Clear summary file
    with open(SUMMARY_FILE, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['MA window', 'Window', 'Cutoff', 'Hidden nodes',
            'Accuracy (%)', 'False Positives (%)', 'Correct', 'Annotated', 'Wrong', 'Identified'])

    # Process data
    for ma_window in [1, 5, 11, 15, 21, 25, 31]:
        fwd_reads, fwd_reads_ma, n_features = get_fwd_reads(ma_window)

        for window in [1, 3, 5, 7, 11, 15, 21]:
            # Metaparameters
            input_dim = n_features * window
            activation = 'sigmoid'

            training_data = get_training_data(fwd_reads_ma, genes[forward], starts[forward], ends[forward], window)

            for hidden_nodes in models:
                # Build neural net
                model = build_model(input_dim, hidden_nodes, activation)

                # Train neural net
                model.fit(training_data, training_data, epochs=3)

                for cutoff in [0.05, 0.08, 0.1, 0.12, 0.15]:
                    # Test model on each region
                    correct, wrong, annotated, identified = test_model(
                        model, fwd_reads_ma, fwd_reads, window, reads, genes, starts, ends, tol, cutoff)

                    # Overall statistics
                    summarize(ma_window, window, cutoff, hidden_nodes, correct, wrong, annotated, identified)
