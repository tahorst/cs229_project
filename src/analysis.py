'''
Script to generate analysis insights from the data.

Usage (run from project directory):
    python src/analysis.py
'''

import os

from util import load_wigs, load_genome, plot_distribution, OUTPUT_DIR


def plot_read_distributions(genes, starts, ends, reads):
    '''
    Plot read distributions for each gene.

    Args:
        genes (array of str): names of genes
        starts (array of int): start position for each gene
        ends (array of int): end position for each gene
        reads (2D array of float): reads for each strand at each position
            dims (strands x genome length)
    '''

    # Check for output directory
    out_dir = os.path.join(OUTPUT_DIR, 'distributions')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Save a distribution for each gene in the genome
    for gene, start, end in zip(genes, starts, ends):
        out = os.path.join(out_dir, '{}.png'.format(gene))
        plot_distribution(gene, start, end, reads, out)


if __name__ == '__main__':
    # Load sequencing and genome data
    reads = load_wigs()
    genes, _, starts, ends = load_genome()

    plot_read_distributions(genes, starts, ends, reads)