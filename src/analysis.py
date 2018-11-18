'''
Script to generate analysis insights from the data.

Usage (run from project directory):
    python src/analysis.py
'''

import os

import numpy as np

from util import load_wigs, load_genome, get_region_info, plot_distribution, plot_reads, OUTPUT_DIR


def plot_region(region, genes, starts, ends, reads):
    '''
    Plot reads and genes for a gene region.

    Args:
        region (int): region to plot, if < 0, defaults to hardcoded region
        genes (array of str): names of genes
        starts (array of int): start position for each gene
        ends (array of int): end position for each gene
        reads (2D array of float): reads for each strand at each position
            dims (strands x genome length)
    '''

    # Check for output directory
    out_dir = os.path.join(OUTPUT_DIR, 'segmentation')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Get start and end positions to plot
    if region >= 0:
        start, end, _, _, _ = get_region_info(region, True, genes, starts, ends)
    else:
        region = 'both'
        start = 1
        end = 12000

    # Save read information
    out = os.path.join(out_dir, '{}.png'.format(region))
    plot_reads(start, end, genes, starts, ends, reads, path=out)

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
        os.makedirs(out_dir)

    # Save a distribution for each gene in the genome
    for gene, start, end in zip(genes, starts, ends):
        out = os.path.join(out_dir, '{}.png'.format(gene))
        plot_distribution(gene, start, end, reads, path=out)


if __name__ == '__main__':
    # Load sequencing and genome data
    reads = load_wigs()
    genes, _, starts, ends = load_genome()

    # Illustrate region segmentation
    for region in range(-1, 2):
        plot_region(region, genes, starts, ends, reads)

    # Raw read distributions
    plot_read_distributions(genes, starts, ends, reads, 'distributions')

    # Moving average distributions - becomes more Gaussian like
    ma = np.zeros_like(reads)
    window = 11
    for i, strand in enumerate(reads):
        ma[i, (window-1)//2:-(window-1)//2] = np.convolve(strand, np.ones((window,))/window, 'valid')
    plot_read_distributions(genes, starts, ends, ma, 'ma_distributions')
