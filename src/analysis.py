'''
Script to generate analysis insights from the data.

Usage (run from project directory):
    python src/analysis.py
'''

import os

import numpy as np
import plotly.offline as py
import plotly.graph_objs as go

import util


def plot_labeled_spikes(reads, genes, starts, ends):
    '''
    Overlays labeled spikes with read data for each labeled region with plot
    saved in output/labels.

    Args:
        genes (array of str): names of genes
        starts (array of int): start position for each gene
        ends (array of int): end position for each gene
        reads (2D array of float): reads for each strand at each position
            dims (strands x genome length)
    '''

    # Check for output directory
    out_dir = os.path.join(util.OUTPUT_DIR, 'labels')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fwd_strand = True
    for region in range(util.get_n_regions(fwd_strand)):
        initiations, terminations = util.get_labeled_spikes(region, fwd_strand)

        if len(initiations) == 0 and len(terminations) == 0:
            continue

        start, end, _, _, _ = util.get_region_info(region, fwd_strand, genes, starts, ends)
        fit = np.ones(end-start)
        fit[initiations-start] = 2
        fit[terminations-start] = 0.5

        out = os.path.join(out_dir, '{}.png'.format(region))
        util.plot_reads(start, end, genes, starts, ends, reads, fit=fit, path=out)

def plot_interactive_reads(reads, start=0, end=100000):
    '''
    Plot reads genes for the whole genome in an interactive plot saved in
    the output directory as interactive_reads.html.

    Args:
        reads (2D array of float): reads for each strand at each position
            dims (strands x genome length)
        start (int): starting index of reads to plot
        end (int): ending index of reads to plot
    '''

    reads += 0.1  # to display 0 counts after taking log
    x = list(range(start, end))
    with np.errstate(divide='ignore'):
        three_prime = go.Scatter(
            x=x,
            y=np.log(reads[util.WIG_STRANDS.index('3f'), start:end]),
            name="3'",
            )
        five_prime = go.Scatter(
            x=x,
            y=np.log(reads[util.WIG_STRANDS.index('5f'), start:end]),
            name="5'",
            )

    data = [three_prime, five_prime]
    layout = dict(
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                ),
            ),
        )

    fig = dict(data=data, layout=layout)

    if not os.path.exists(util.OUTPUT_DIR):
        os.makedirs(util.OUTPUT_DIR)

    out = os.path.join(util.OUTPUT_DIR, 'interactive_reads.html')
    py.plot(fig, filename=out, auto_open=False)

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
    out_dir = os.path.join(util.OUTPUT_DIR, 'segmentation')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Get start and end positions to plot
    if region >= 0:
        start, end, _, _, _ = util.get_region_info(region, True, genes, starts, ends)
    else:
        region = 'both'
        start = 1
        end = 12000

    # Save read information
    out = os.path.join(out_dir, '{}.png'.format(region))
    util.plot_reads(start, end, genes, starts, ends, reads, path=out)

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
    out_dir = os.path.join(util.OUTPUT_DIR, desc)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save a distribution for each gene in the genome
    for gene, start, end in zip(genes, starts, ends):
        out = os.path.join(out_dir, '{}.png'.format(gene))
        util.plot_distribution(gene, start, end, reads, path=out)


if __name__ == '__main__':
    # Load sequencing and genome data
    reads = util.load_wigs()
    genes, _, starts, ends = util.load_genome()

    # Visualize labeled data for confirmation of correct labels
    print('Plotting labeled spikes...')
    plot_labeled_spikes(reads, genes, starts, ends)

    # Generate interactive plotly plot for identifying labeled peaks
    print('Generating interactive plot...')
    plot_interactive_reads(reads)

    # Illustrate region segmentation
    print('Plotting example segmentation...')
    for region in range(-1, 2):
        plot_region(region, genes, starts, ends, reads)

    # Raw read distributions
    print('Plotting raw distributions...')
    plot_read_distributions(genes, starts, ends, reads, 'distributions')

    # Moving average distributions - becomes more Gaussian like
    print('Plotting moving average distributions...')
    ma = np.zeros_like(reads)
    window = 11
    for i, strand in enumerate(reads):
        ma[i, (window-1)//2:-(window-1)//2] = np.convolve(strand, np.ones((window,))/window, 'valid')
    plot_read_distributions(genes, starts, ends, ma, 'ma_distributions')
