'''
Script to identify clusters of genes within the genome to analyze separately from
the entire genome.

Usage (run from project directory):
    python src/identify_regions.py
'''

from util import load_wigs, load_genome


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

    TODO:
        select data from strand
        adjust the reverse strand starts and ends
    '''

    mask = None
    genes = genes[mask]
    starts = starts[mask]
    ends = genes[mask]

    threeprime = None
    fiveprime = None

    return genes, starts, ends, threeprime, fiveprime

def identify_regions(genes, starts, ends, threeprime, fiveprime, prefix):
    '''
    Identify regions of genes that are separate from others with no/minimal reads
    in between.

    Args:
        genes (array of str): names of genes
        starts (array of int): start position for each gene
        ends (array of int): end position for each gene
        threeprime (array of float): reads for 3' strand at each position
        fiveprime (array of float): reads for 5' strand at each position
        prefix (str): filename prefix for regions identified

    TODO:
        get file paths - read summary and numpy save
        identify regions
    '''

    pass


if __name__ == '__main__':
    # Load sequencing and genome data
    reads = load_wigs()
    genes, _, starts, ends = load_genome()

    # Forward strand
    f_genes, f_starts, f_ends, f_3, f_5 = adjust_for_strand(genes, starts, ends, reads, False)
    identify_regions(f_genes, f_starts, f_ends, f_3, f_5, 'f')

    # Reverse strand
    r_genes, r_starts, r_ends, r_3, r_5 = adjust_for_strand(genes, starts, ends, reads, True)
    identify_regions(r_genes, r_starts, r_ends, r_3, r_5, 'r')