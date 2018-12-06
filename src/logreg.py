'''
Implementation of logistic regression model using sklearn

Usage:
    python src/logreg.py

Output:
    Saves expected initiations and terminations in output/logreg_assignments/
'''

import os

import numpy as np
from sklearn.linear_model import LogisticRegression

import util


def get_data(reads, window, regions, pad=0, down_sample=False, training=False):
    '''
    Uses process_reads to generate training_data for genes that have been
    identified to not have spikes.

    Args:
        reads (2D array of float): reads for each strand at each position
            dims (n_features x genome length)
        window (int): size of sliding window
        regions (iterable): the regions to include in training data
        down_sample (bool): if True, down samples the no spike case because of
            class imbalance
        training (bool): if True, skips regions that would have 2 labels for better training

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
            data.append(reads[:, s:e].reshape(-1))

    return np.array(data), np.array(labels)

def get_spikes(labels, reads, gap=3):
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

    locations = np.where(labels == 1)[0]

    # Add 1 to locations for actual genome location because of 0 indexing
    spikes = np.array(group_spikes(locations, 1))

    return spikes

def train_models(reads, window, training_range):
    '''

    '''

    x_train, y_train = get_data(reads, window, training_range, training=True)

    # Generate model for classes 1 and 2 (initiations and terminations)
    for i in range(1,3):
        y = (y_train == i).astype(int)

        logreg = LogisticRegression()
        yield logreg.fit(x_train, y)

def test_models(init_model, term_model):
    '''

    '''

    return correct, wrong, annotated, identified

def main(reads, window, training_range):
    '''

    '''

    init_model, term_model = train_models(reads, window, training_range)

    correct, wrong, annotated, identified = test_models(init_model, term_model)

if __name__ == '__main__':
    reads = util.load_wigs()
    genes, _, starts, ends = util.load_genome()
    window = 15
    expanded = False
    total = True

    training_range = range(16)

    fwd_strand = True

    main(reads, window, training_range)

        x_test, y_test = get_data(reads, window, range(16,33))
        y_train = np.array([1 if y == i else 0 for y in y_train])
        y_test = set(np.where([1 if y == i else 0 for y in y_test])[0])

        logreg = LogisticRegression()
        logreg.fit(x_train, y_train)

        y_pred = logreg.predict(x_test)
        spikes = get_spikes(y_pred, reads)

        correct = np.sum([1 if y in y_test else 0 for y in np.where(y_pred)[0]])
        print('{}/{}'.format(correct, np.sum(y_pred))


    import ipdb; ipdb.set_trace()

    for region in range(util.get_n_regions(fwd_strand)):
        print('\nRegion: {}'.format(region))

        # Information for region to be analyzed
        x = util.load_region_reads(reads, region, fwd_strand, ma_window=window, expanded=expanded, total=total)
        start, end, region_genes, region_starts, region_ends = util.get_region_info(
            region, fwd_strand, genes, starts, ends)
        n_levels = len(region_genes) + 3  # Extra level for 0 reads and spiked reads on each strand

        # Fit model
        gmm = method(n_components=n_levels)
        gmm.fit(x)
        labels = gmm.predict(x)
        with np.errstate(divide='ignore'):
            log_prob = np.log(gmm.predict_proba(x))
        means = gmm.means_.mean(axis=1)

        # Output levels
        levels = np.array([means[l] for l in labels])
        mle_gene_levels = np.zeros_like(levels)
        for s, e in zip(region_starts, region_ends):
            s = s - start
            e = e - start
            label = np.argmax(np.sum(log_prob[s:e, :], axis=0))
            mle_gene_levels[s:e] = means[label]

        # Plot output
        ## Path setup
        out_dir = os.path.join(util.OUTPUT_DIR, 'gmm_assignments')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        ## Raw level assignments
        out = os.path.join(out_dir, '{}_raw.png'.format(region))
        util.plot_reads(start, end, genes, starts, ends, reads, fit=levels, path=out)

        ## MLE per gene level assignments
        out = os.path.join(out_dir, '{}_mle.png'.format(region))
        util.plot_reads(start, end, genes, starts, ends, reads, fit=mle_gene_levels, path=out)
