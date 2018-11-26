'''
Implementation of a convolutional neural net using keras.

Usage:
    python src/cnn.py

Output:
    Saves estimates for transcription unit assignments in output/cnn_assignments/

Parameters to search:

TODO:
'''

import os

import keras
import numpy as np

import util


if __name__ == '__main__':
    reads = util.load_wigs()
    genes, _, starts, ends = util.load_genome()

    window = 1
    expanded = False
    total = False
    kernel_size = 21
    pad_size = (kernel_size - 1) // 2
    fwd_strand = True
    region = 0

    # TODO: loop over each region
    print('\nRegion: {}'.format(region))
    x_train = util.load_region_reads(reads, region, fwd_strand, ma_window=window, expanded=expanded, total=total)
    spikes = util.get_labeled_spikes(region, fwd_strand)
    train_length, n_features = x_train.shape
    x_train = x_train.reshape(1, train_length, n_features)

    test_region = 1
    test_fwd_strand = True
    x_test = util.load_region_reads(reads, test_region, test_fwd_strand)
    test_spikes = util.get_labeled_spikes(test_region, test_fwd_strand)
    test_length = x_test.shape[0]
    x_test = x_test.reshape(1, test_length, n_features)

    y_train = np.zeros(train_length - kernel_size + 1)
    y_train[spikes - pad_size - 1] = 1
    y_train = keras.utils.to_categorical([y_train])

    y_test = np.zeros(test_length - kernel_size + 1)
    y_test[test_spikes - pad_size - 1] = 1
    y_test = keras.utils.to_categorical([y_test])

    # Create model
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(filters=2, kernel_size=kernel_size, input_shape=(None, n_features)))
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))
    model.summary()

    # Use model
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy')
    model.fit(x_train, y_train)
    model.evaluate(x_test, y_test)
    y = model.predict(x_test)

    pad = np.array([[0, 0]]*pad_size)
    y = np.vstack((pad, y.squeeze(), pad))

    ## Path setup
    out_dir = os.path.join(util.OUTPUT_DIR, 'cnn_assignments')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out = os.path.join(out_dir, '{}.png'.format(test_region))
    start, end, region_genes, region_starts, region_ends = util.get_region_info(
            test_region, fwd_strand, genes, starts, ends)
    util.plot_reads(start, end, genes, starts, ends, reads, fit=y[:, 0], path=out)
