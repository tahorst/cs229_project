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

    return np.array(initiations), np.array(terminations)


if __name__ == '__main__':
    reads = util.load_wigs()
    genes, _, starts, ends = util.load_genome()
    forward = starts > 0

    ma_window = 21
    convolution = np.ones((ma_window,)) / ma_window
    idx_3p = util.WIG_STRANDS.index('3f')
    idx_5p = util.WIG_STRANDS.index('5f')
    fwd_reads = np.vstack((reads[idx_3p, :], reads[idx_5p, :]))
    fwd_reads_ma = np.vstack((np.convolve(reads[idx_3p, :], convolution, 'same'),
        np.convolve(reads[idx_5p, :], convolution, 'same')))

    # Metaparameters
    n_features = 2
    window = 7
    pad = (window - 1) // 2
    input_nodes = n_features * window
    hidden_nodes = 8
    activation = 'sigmoid'
    cutoff = 0.1

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
    training_data = get_training_data(fwd_reads_ma, genes[forward], starts[forward], ends[forward], window)
    model.fit(training_data, training_data, epochs=3)

    # Test model on each region
    fwd_strand = True
    for region in range(util.get_n_regions(fwd_strand)):
        print('\nRegion: {}'.format(region))

        start, end = util.get_region_bounds(region, fwd_strand)

        # Test trained model
        test_data = np.array(process_reads(fwd_reads_ma, start, end, window))
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
        util.plot_reads(start, end, genes, starts, ends, reads, fit=mse, path=out)

        initiations, terminations = get_spikes(mse, fwd_reads[:, start:end], cutoff)
        print('\tInitiations: {}'.format(start + initiations))
        print('\tTerminations: {}'.format(start + terminations))
