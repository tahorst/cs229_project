'''
Script to generate analysis insights from the data.

Usage (run from project directory):
    python src/analysis.py
'''

import os

import numpy as np

from util import load_wigs, load_genome, plot_distribution, OUTPUT_DIR


def plot_read_distributions(genes, starts, ends, reads, desc):
    '''
    Plot read distributions for each gene.

    Args:
        genes (array of str): names of genes
        starts (array of int): start position for each gene
        ends (array of int): end position for each gene
        reads (2D array of float): reads for each strand at each position
            dims (strands x genome length)
        desc (str): description of the data, becomes directory for output
    '''

    # Check for output directory
    out_dir = os.path.join(OUTPUT_DIR, desc)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Save a distribution for each gene in the genome
    for gene, start, end in zip(genes, starts, ends):
        out = os.path.join(out_dir, '{}.png'.format(gene))
        plot_distribution(gene, start, end, ma, out)


if __name__ == '__main__':
    # Load sequencing and genome data
    reads = load_wigs()
    genes, _, starts, ends = load_genome()

    # Raw read distributions
    plot_read_distributions(genes, starts, ends, reads, 'distributions')

    # Moving average distributions - becomes more Gaussian like
    ma = np.zeros_like(reads)
    window = 11
    for i, strand in enumerate(reads):
        ma[i, (window-1)//2:-(window-1)//2] = np.convolve(strand, np.ones((window,))/window, 'valid')
    plot_read_distributions(genes, starts, ends, ma, 'ma_distributions')
