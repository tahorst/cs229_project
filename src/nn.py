'''
Implementation of a neural net using keras.

Usage:
    python src/nn.py

Output:
    Saves estimates for transcription unit assignments in output/nn_assignments/

Parameters to search:

TODO:
'''

import os

import keras
import numpy as np

import util


def get_training_data(reads, window, regions):
    '''
    Uses process_reads to generate training_data for genes that have been
    identified to not have spikes.

    Args:
        reads (2D array of float): reads for each strand at each position
            dims (strands x genome length)
        window (int): size of sliding window
        regions (iterable): the regions to include in training data

    Returns:
        array of float: 2D array of read data, dims (m samples x n features)
        array of int: 1D array of labels, dims (m samples)
    '''

    data = []
    labels = []
    pad = 3

    for region in regions:
        start, end = util.get_region_bounds(region, True)
        initiations, terminations = util.get_labeled_spikes(region, True)

        length = end - start
        n_splits = length - window + 1

        for i in range(n_splits):
            s = start + i
            e = s + window

            label = 0
            if np.any((initiations > s) & (initiations < e)):
                if np.any((initiations > s + pad) & (initiations < e - pad)):
                    label = 1
                else:
                    continue
            if np.any((terminations > s) & (terminations < e)):
                if label == 1:
                    print('*** both peaks ***')
                if np.any((terminations > s + pad) & (terminations < e - pad)):
                    label = 2
                else:
                    continue

            labels.append(label)
            data.append(np.hstack((reads[0, s:e], reads[1, s:e])))

    labels = keras.utils.np_utils.to_categorical(np.array(labels))
    return np.array(data), labels

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
    model.add(keras.layers.Dense(3, activation='softmax'))
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy')

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
        test_data, test_labels = get_training_data(reads, window, [region])
        prediction = model.predict(test_data)
        print(np.where(prediction[:,1] > 0.1))
        print(np.where(prediction[:,2] > 0.1))
        import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    reads = util.load_wigs()
    genes, _, starts, ends = util.load_genome()

    tol = 5
    fwd_strand = True

    models = np.array([[10, 10]])

    for ma_window in [1]:
        fwd_reads, fwd_reads_ma, n_features = get_fwd_reads(reads, ma_window)

        for window in [11]:
            # Metaparameters
            input_dim = n_features * window
            activation = 'sigmoid'

            training_data, training_labels = get_training_data(fwd_reads, window, range(16))

            for hidden_nodes in models:
                # Build neural net
                model = build_model(input_dim, hidden_nodes, activation)

                # Train neural net
                model.fit(training_data, training_labels, epochs=5)

                for cutoff in [0.05, 0.08, 0.1, 0.12, 0.15]:
                    # Test model on each region
                    correct, wrong, annotated, identified = test_model(
                        model, fwd_reads_ma, fwd_reads, window, reads, genes, starts, ends, tol, cutoff)

                    # Overall statistics
                    summarize(ma_window, window, cutoff, hidden_nodes, correct, wrong, annotated, identified)
