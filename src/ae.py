'''
Implementation of a autoencoder neural net using keras for outlier detection.

Usage:
    python src/ae.py

Output:
    Saves estimates for transcription unit assignments in output/ae_assignments/

Parameters to search:

TODO:
'''

import os

import keras
import numpy as np

import util


def process_reads(reads, start, end, window):
    '''
    Processes the reads to feed to the neural network.
    Stacks the 3' and 5' reads and will slide a window to create new samples
    for every section in the region of interest.  Within each section, the
    samples are normalized to be between 0 and 1.

    Args:
        reads (array of floats): 2D array of reads for the 3' and 5' strand
            dims: (2 x genome size)
        start (int): start position in the genome
        end (int): end position in the genome
        window (int): size of sliding window

    Returns:
        list of arrays of floats: each array will be a sample with processed reads
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
        array of floats: 2D array of read data, dims (m samples x n features)
    '''

    stop_gene = 'cdaR'
    excluded = {'yaaY', 'ribF', 'ileS', 'folA', 'djlA', 'yabP', 'mraZ', 'ftsI', 'ftsW', 'ddlB', 'ftsA', 'yadD', 'hrpB'}

    data = []

    for gene, start, end in zip(genes, starts, ends):
        if gene in excluded:
            continue

        data += process_reads(reads, start, end, window)

        if gene == stop_gene:
            break

    return np.array(data)


if __name__ == '__main__':
    reads = util.load_wigs()
    genes, _, starts, ends = util.load_genome()
    forward = starts > 0

    ma_window = 21
    convolution = np.ones((ma_window,)) / ma_window
    idx_3p = util.WIG_STRANDS.index('3f')
    idx_5p = util.WIG_STRANDS.index('5f')
    fwd_reads = np.vstack((np.convolve(reads[idx_3p, :], convolution, 'same'),
        np.convolve(reads[idx_5p, :], convolution, 'same')))

    region = 0
    fwd_strand = True
    start, end = util.get_region_bounds(region, fwd_strand)

    # Metaparameters
    n_features = 2
    window = 7
    pad = (window - 1) // 2
    input_nodes = n_features * window
    hidden_nodes = 8
    activation = 'sigmoid'

    # Build neural net
    model = keras.Sequential()
    model.add(keras.layers.Dense(hidden_nodes, input_dim=input_nodes, activation=activation))
    model.add(keras.layers.Dense(hidden_nodes // 2, activation=activation))
    # model.add(keras.layers.Dense(hidden_nodes // 4, activation=activation))
    model.add(keras.layers.Dense(2, activation=activation))
    # model.add(keras.layers.Dense(hidden_nodes // 4, activation=activation))
    model.add(keras.layers.Dense(hidden_nodes // 2, activation=activation))
    model.add(keras.layers.Dense(hidden_nodes, activation=activation))
    model.add(keras.layers.Dense(input_nodes, activation='sigmoid'))
    model.summary()

    model.compile(optimizer='adam', loss='mse')

    # Train neural net
    training_data = get_training_data(fwd_reads, genes[forward], starts[forward], ends[forward], window)
    model.fit(training_data, training_data, epochs=5, validation_split=0.1)

    # Test trained model
    test_data = np.array(process_reads(fwd_reads, start, end, window))
    prediction = model.predict(test_data)
    mse = np.mean((test_data - prediction)**2, axis=1)
    mse = np.hstack((np.zeros(pad), mse, np.zeros(pad)))

    # Plot outputs
    ## Create directory
    out_dir = os.path.join(util.OUTPUT_DIR, 'ae_assignments')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ## Raw level assignments
    out = os.path.join(out_dir, '{}.png'.format(region))
    util.plot_reads(start, end, genes, starts, ends, reads, fit=mse, path=out)

    import ipdb; ipdb.set_trace()
