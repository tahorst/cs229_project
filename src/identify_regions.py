'''
Script to identify clusters of genes within the genome to analyze separately from
the entire genome.

Usage (run from project directory):
    python src/identify_regions.py
'''

import json
import os

import numpy as np

from util import load_wigs, load_genome, plot_reads, GENOME_SIZE, WIG_STRANDS, OUTPUT_DIR, REGIONS_DIR


def adjust_for_strand(genes, starts, ends, reads, rev=False):
    '''
    Selects data for specified strand and performs manipulations needed to align
    starts, ends and reads.

    Args:
        genes (array of str): names of genes
        starts (array of int): start position for each gene
        ends (array of int): end position for each gene
        reads (2D array of float): reads for each strand at each position
            dims (strands x genome length)
        rev (bool): True if reverse strand, False if forward

    Returns:
        genes (array of str): names of genes on fwd/rev strand
        starts (array of int): start position for each gene on fwd/rev strand
        ends (array of int): end position for each gene on fwd/rev strand
        threeprime (array of float): reads from the 3' fwd/rev strand
        fiveprime (array of float): reads from the 5' fwd/rev strand

    TODO: handle the rev strand
    '''

    mask = starts > 0
    genes = genes[mask]
    starts = starts[mask]
    ends = ends[mask]

    threeprime = reads[WIG_STRANDS.index('3f'), :]
    fiveprime = reads[WIG_STRANDS.index('5f'), :]

    return genes, starts, ends, threeprime, fiveprime

def identify_regions(starts, ends, threeprime, fiveprime):
    '''
    Identify regions of genes that are separate from others with no/minimal reads
    in between.

    Args:
        starts (array of int): start position for each gene
        ends (array of int): end position for each gene
        threeprime (array of float): reads for 3' strand at each position
        fiveprime (array of float): reads for 5' strand at each position

    Returns:
        real_starts (array of int): start positions for each region
        real_ends (array of int): end positions for each region

    TODO: handle the rev strand
    '''

    window = 11
    threshold = 0.5
    gene_pad = 100

    # Convert raw read data into binary data based on threshold in a region
    ma = np.convolve(threeprime + fiveprime, np.ones(window), 'same')
    reads = np.zeros_like(ma)
    reads[ma > threshold] = 1

    # Initialize for loops
    real_starts = []
    real_ends = []
    gene = 0
    pos = 0

    shifts = np.where(reads == 0)[0]
    n_genes = len(starts)
    n_shifts = len(shifts)

    # Loop through each gene to assign to a region
    while gene < n_genes:
        # Start new region to include gene
        if gene == 0 and starts[gene] < shifts[pos]:
            real_starts += [0]
        else:
            real_starts += [shifts[pos]]

        # Expand region until no reads where a gene does not exist
        while gene < n_genes - 1 and pos < n_shifts - 1:
            if ends[gene] > shifts[pos] + window:
                pos += 1
            elif shifts[pos] > starts[gene + 1] - gene_pad:
                gene += 1
            else:
                temp_pos = pos
                next_start = starts[gene + 1]
                while temp_pos < n_shifts - 1 and shifts[temp_pos + 1] < next_start:
                    temp_pos += 1

                real_ends += [shifts[temp_pos]]
                break

        gene += 1

    # Last shift is before end of last gene so set end at end of genome
    if len(real_starts) != len(real_ends):
        real_ends += [GENOME_SIZE]

    return np.array(real_starts), np.array(real_ends)

def save_region(genes, starts, ends, start, end, prefix):
    '''
    Save data for region.  Outputs a JSON object to the regions data directory
    and plots for the reads in each region to the regions output directory.

    Args:
        genes (array of str): names of genes
        starts (array of int): start position for each gene
        ends (array of int): end position for each gene
        start (int): start position of region
        end (int): end position of region
        prefix (str): filename prefix for regions identified
    '''

    # Save numpy array for reads in region
    ## Check for output directory
    if not os.path.exists(REGIONS_DIR):
        os.makedirs(REGIONS_DIR)

    ## Save region info
    out = os.path.join(REGIONS_DIR, '{}.json'.format(prefix))
    data = [float(start), float(end)]
    with open(out, 'w') as f:
        json.dump(data, f)

    # Save image of region
    ## Check for output directory
    out_dir = os.path.join(OUTPUT_DIR, 'regions')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ## Save image
    out = os.path.join(out_dir, '{}.png'.format(prefix))
    plot_reads(start, end, genes, starts, ends, reads, path=out)


if __name__ == '__main__':
    # Load sequencing and genome data
    reads = load_wigs()
    genes, _, starts, ends = load_genome()

    # Forward strand
    f_genes, f_starts, f_ends, f_3, f_5 = adjust_for_strand(genes, starts, ends, reads, False)
    real_starts, real_ends = identify_regions(f_starts, f_ends, f_3, f_5)
    for i, (s, e) in enumerate(zip(real_starts+1, real_ends+1)):
        save_region(f_genes, f_starts, f_ends, s, e, 'f{}'.format(i))

    # Reverse strand
    # TODO: implement handling for reverse strand
    # r_genes, r_starts, r_ends, r_3, r_5 = adjust_for_strand(genes, starts, ends, reads, True)
    # real_starts, real_ends = identify_regions(r_starts, r_ends, r_3, r_5)
    # for i, (s, e) in enumerate(zip(real_starts+1, real_ends+1)):
    #     save_region(f_genes, f_starts, f_ends, s, e, 'f{}'.format(i))
