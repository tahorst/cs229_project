'''
Useful functions and constants for analysis.
'''

import csv
import json
import os
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
REGIONS_DIR = os.path.join(DATA_DIR, 'regions')

# .wig files from Lalanne et al
WIG_FILE = os.path.join(RAW_DIR, 'GSM2971252_Escherichia_coli_WT_Rend_seq_5_exo_MOPS_comp_25s_frag_pooled_{}_no_shadow.wig')
PROCESSED_FILE = os.path.join(DATA_DIR, 'GSM2971252_reads.npy')
WIG_STRANDS = ['3f', '3r', '5f', '5r']

# Spike annotations
ANNOTATED_SPIKES_FILE = os.path.join(DATA_DIR, 'validation', 'annotated_spikes.json')
ALL = 0  # Label for all spikes
SMALL = 1  # Label for spikes in regions with small operons
LARGE = 2  # Label for spikes in regions with large operons

# Genome information
GENOME_SIZE = 4639675
ANNOTATION_FILE = os.path.join(RAW_DIR, 'U00096.2.faa')


def load_wigs(cached=None, wig_file=None, strands=None, genome_size=None):
    '''
    Loads read data from .wig files

    Inputs:
        cached (str): path to cached data, if None, defaults to PROCESSED_FILE.
            If file exists, it will be read, otherwise data will be loaded from
            the raw data and written to file.
        wig_file (str): template for the wig files to be read (gets formatted for each strand),
            if None, defaults to WIG_FILE
        strands (list of str): each strand name that gets inserted into the wig_file string
            (order determines order of first dimension of returned array), if None, defaults
            to WIG_STRANDS
        genome_size (int): base pair length of the genome, if None, defaults to GENOME_SIZE

    Returns 2D array of floats (4 x genome_size) for reads on each strand
    '''

    # Check for cached results and return if exists
    if cached is None:
        cached = PROCESSED_FILE
    if os.path.exists(cached):
        print('Loaded cached read data')
        return np.load(cached)

    # Defaults in file for E. coli if none given
    if wig_file is None:
        wig_file = WIG_FILE
    if strands is None:
        strands = WIG_STRANDS
    if genome_size is None:
        genome_size = GENOME_SIZE

    data = np.zeros((len(strands), genome_size))
    for i, strand in enumerate(strands):
        with open(wig_file.format(strand), 'r') as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)

            # skip lines of text at top of the file
            d = []
            while len(d) == 0:
                try:
                    d = np.array(list(reader))
                except Exception as exc:
                    print(exc)

            index = np.array(d[:, 0], dtype=int) - 1
            reads = np.array(d[:, 1], dtype=float)

            data[i, index] = reads

    np.save(cached, data)

    return data

def load_genome(path=None):
    '''
    Loads information about the genome, specifically genes and their locations.

    Args:
        path (str): path to genome file, if None, defaults to ANNOTATION_FILE

    Returns:
        genes (array of str): gene names
        locus_tags (array of str): locus tag names
        starts (array of int): genome position of start of gene
        ends (array of int): genome position of end of gene
    '''

    # Set defaults
    if path is None:
        path = ANNOTATION_FILE

    # Read in file
    with open(path) as f:
        data = f.read()

    # Initialize data structures to return
    genes = []
    locus_tags = []
    starts = []
    ends = []

    # Read in each gene and pull out relevant information
    for gene in data.split('\n>'):
        locus = re.findall('\[locus_tag=(.*?)\]', gene)[0]
        gene_name = re.findall('\[gene=(.*?)\]', gene)[0]

        location = re.findall('\[location=(.*?)\]', gene)[0]
        # Handle complement strand
        if location[0] == 'c':
            dir = -1
        else:
            dir = 1
        start, end = (dir * int(pos) for pos in re.findall('([0-9]*)\.\.([0-9]*)', location)[0])

        genes.append(gene_name)
        locus_tags.append(locus)
        starts.append(start)
        ends.append(end)

    genes = np.array(genes)
    locus_tags = np.array(locus_tags)
    starts = np.array(starts)
    ends = np.array(ends)

    sort_idx = np.argsort(starts)

    return genes[sort_idx], locus_tags[sort_idx], starts[sort_idx], ends[sort_idx]

def load_region_reads(reads, region, fwd_strand, ma_window=1, expanded=False, total=False):
    '''
    Selects the read data for a region of interest.

    Args:
        reads (2D array of float): reads for each strand at each position
            dims (strands x genome length)
        region (int): index of region
        fwd_strand (bool): True if region is on fwd strand, False if rev
        ma_window (int): Moving avergae taken if > 1, should be an odd number
            for best handling
        expanded (bool): If True, the features are expanded to include reads in
            the ma_window
        total (bool): If True, only the total reads are returned with a second
            feature representing the difference between 5' and 3' reads, does
            not work with expanded set to True

    Returns:
        array of float: 2D array (region length, features): reads for the region
            of interest at each location
    '''

    def ma(strand, reads, convolution, clipped_index):
        '''
        Calculates the moving average of reads on a given strand.

        Args:
            strand (str): strand from WIG_STRANDS (eg '3f')
            reads (array of float): 2D array (# strands x genome size) of read counts
            convolution (array of float): convolution array to apply to reads
            clipped_index (int): indexes that get clipped from taking the moving average

        Returns:
            array of float: moving average applied to the reads on the given strand
        '''

        idx = WIG_STRANDS.index(strand)
        return np.convolve(reads[idx, :], convolution, 'full')

    if expanded and total:
        print('Warning: both expanded and total not supported together for load_region_reads')

    clipped_index = (ma_window-1) // 2  # For proper indexing since data is lost with ma
    convolution = np.ones((ma_window,)) / ma_window

    start, end = get_region_bounds(region, fwd_strand)

    if fwd_strand:
        three_prime = ma('3f', reads, convolution, clipped_index)
        five_prime = ma('5f', reads, convolution, clipped_index)
    else:
        three_prime = ma('3r', reads, convolution, clipped_index)
        five_prime = ma('5r', reads, convolution, clipped_index)

    # Assemble desired features
    if expanded:
        tp = (three_prime[start+i+clipped_index:end+i+clipped_index] for i in range(-clipped_index, clipped_index+1))
        fp = (five_prime[start+i+clipped_index:end+i+clipped_index] for i in range(-clipped_index, clipped_index+1))
        x = np.hstack((np.vstack(tp).T, np.vstack(fp).T))
    elif total:
        tp = three_prime[start:end] + 1
        fp = five_prime[start:end] + 1
        summed_reads = tp + fp - 2
        diff = np.fmax(tp / fp, fp / tp)
        x = np.vstack((summed_reads, diff)).T
    else:
        x = np.vstack((three_prime[start:end], five_prime[start:end])).T

    return x

def get_region_bounds(region, fwd_strand):
    '''
    Gets the start and end of a genome sequence.

    Args:
        region (int): index of region
        fwd_strand (bool): True if region is on fwd strand, False if rev

    Returns:
        start (int): starting position of region
        end (int): ending position of region
    '''

    file = os.path.join(REGIONS_DIR, '{}{}.json'.format('f' if fwd_strand else 'r', region))
    if not os.path.exists(file):
        print('Region information does not exist for region {}. Try running identify_regions.py'.format(region))
        return -1, -1

    with open(file) as f:
        start, end = json.load(f)

    return int(start), int(end)

def get_n_regions(fwd_strand):
    '''
    Gives the number of regions for a given strand for easy looping.

    Args:
        fwd_strand (bool): True if forward strand, False if reverse

    Returns:
        int: number of regions for strand

    TODO:
        handle rev strand
        better representation of data to not have this function?
    '''

    return 993

def get_region_info(region, fwd_strand, genes, starts, ends):
    '''
    Gets information about a region.

    Args:
        region (int): index of region
        fwd_strand (bool): True if region is on fwd strand, False if rev
        genes (array of str): names of genes
        starts (array of int): start position for each gene
        ends (array of int): end position for each gene

    Returns:
        start (int): starting position of region
        end (int): ending position of region
        region_genes (array of str): names of genes in region
        region_starts (array of int): start position for each gene in region
        region_ends (array of int): end position for each gene in region
    '''

    start, end = get_region_bounds(region, fwd_strand)

    if fwd_strand:
        mask = (ends > start) & (starts < end)
    else:
        mask = (-ends > -start) & (-starts < -end)

    region_genes = genes[mask]
    region_starts = starts[mask]
    region_ends = ends[mask]

    return start, end, region_genes, region_starts, region_ends

def get_labeled_spikes(region, fwd_strand, tag=ALL):
    '''
    Gets locations of annotated spikes within a specific region.

    Args:
        region (int): index of region
        fwd_strand (bool): True if region is on fwd strand, False if rev
        tag (int): which set of annotations to return (ALL, SMALL, LARGE)

    Returns:
        initiations (arrays of int): positions within a region where initiation occurs
        terminations (arrays of int): positions within a region where termination occurs
    '''

    start, end = get_region_bounds(region, fwd_strand)

    with open(ANNOTATED_SPIKES_FILE) as f:
        data = json.load(f)

    starts = np.array(data['starts'])
    ends = np.array(data['ends'])

    initiations = starts[(starts >= start) & (starts <= end)]
    terminations = ends[(ends >= start) & (ends <= end)]

    return initiations, terminations

def get_match_statistics(initiations, terminations, initiations_val, terminations_val, tol):
    '''
    Evaluates the performance of identified initiation and termination location with
    labeled data.

    Args:
        initiations (array of int): positions of identified initiation sites
        terminations (array of int): positions of identified termination sites
        initiations_val (array of int): positions of labeled initiation sites
        terminations_val (array of int): positions of labeled termination sites
        tol (int): distance assigned peak can be from labeled peak to call correct

    Returns:
        n_val (int): number of labeled peaks
        n_test (int): number of identified peaks
        correct (int): number of labeled peaks identified (true positives)
        wrong (int): number of identified peaks that are not labeled (false positives)
        accuracy (float): percentage of accuracy
        false_positives (float): percentage of false positives in identified peaks

    Notes:
        False negatives will be n_val - correct
        True negatives are essentially ignored but would be the total positions in the region - n_test
    '''

    n_val = len(initiations_val) + len(terminations_val)
    n_test = len(initiations) + len(terminations)
    correct = 0
    for val in initiations_val:
        for test in initiations:
            if np.abs(val - test) < tol:
                correct += 1
                break
    for val in terminations_val:
        for test in terminations:
            if np.abs(val - test) < tol:
                correct += 1
                break
    wrong = n_test - correct

    if n_val > 0:
        accuracy = correct / n_val * 100
    else:
        accuracy = 0

    if n_test > 0:
        false_positives = wrong / n_test * 100
    else:
        false_positives = 0

    return n_val, n_test, correct, wrong, accuracy, false_positives

def plot_reads(start, end, genes, starts, ends, reads, fit=None, path=None):
    '''
    Plots the reads of the 3' and 5' data on the given strand.  Also shows any
    genes that start or finish within the specified region.

    Args:
        start (int): start position in the genome
        end (int): end position in the genome
        genes (array of str): names of genes
        starts (array of int): start position for each gene
        ends (array of int): end position for each gene
        reads (2D array of float): reads for each strand at each position
            dims (strands x genome length)
        path (str): path to save image, if None, just displays image to screen
    '''

    # Set forward (0) or reverse (1) strand to correspond to data order
    if start < 0:
        strand = 1
    else:
        strand = 0

    # Identify genes in region and get reads
    # Need to subtract 1 from index for reads since starts at 0 not 1 like genome position
    if strand:
        # Reverse strand needs indices adjusted
        mask = (-ends > -start) & (-starts < -end)
        loc = np.arange(end, start)
        with np.errstate(divide='ignore'):
            three_prime = np.log(reads[strand, int(-start-1):int(-end-1)][::-1])
            five_prime = np.log(reads[2+strand, int(-start-1):int(-end-1)][::-1])
    else:
        # Forward strand
        mask = (ends > start) & (starts < end)
        loc = np.arange(start, end)
        with np.errstate(divide='ignore'):
            three_prime = np.log(reads[strand, int(start-1):int(end-1)])
            five_prime = np.log(reads[2+strand, int(start-1):int(end-1)])

    genes = genes[mask]
    starts = starts[mask]
    ends = ends[mask]

    # Adjust for 0 reads from log
    three_prime[three_prime < 0] = -1
    five_prime[five_prime < 0] = -1

    plt.figure()
    plt.step(loc, np.vstack((three_prime, five_prime)).T, linewidth=0.25)

    if fit is not None:
        with np.errstate(divide='ignore'):
            plt.step(loc, np.log(fit), color='k')

    gene_line = -1.5
    gene_offset = 0.1
    for gene, s, e in zip(genes, starts, ends):
        plt.plot([s, e], [gene_line, gene_line], 'k')
        plt.plot([s, s], [gene_line-gene_offset, gene_line+gene_offset], 'k')
        plt.plot([e, e], [gene_line-gene_offset, gene_line+gene_offset], 'k')
        plt.text((s+e)/2, gene_line-3*gene_offset, gene, ha='center', fontsize=6)
    plt.xlim([loc[0], loc[-1]])

    plt.xlabel('Genome Location')
    plt.ylabel('Reads (log)')

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close('all')

def plot_gene_distribution(gene, genes, starts, ends, reads, path=None):
    '''
    Plot read distributions for the length of the specified gene.

    Args:
        gene (str): name of the gene to get distribution for
        genes (array of str): names of genes
        starts (array of int): start position for each gene
        ends (array of int): end position for each gene
        reads (2D array of float): reads for each strand at each position
            dims (strands x genome length)
        path (str): path to save image, if None, just displays image to screen
    '''

    idx = np.where(genes == gene)
    start = starts[idx]
    end = ends[idx]

    plot_distribution(gene, start, end, reads, path)

def plot_distribution(label, start, end, reads, path=None):
    '''
    Plot read distributions for the specified length of the genome.

    Args:
        label (str): name of the gene to get distribution for
        start (int): starting genome position
        end (int): ending genome position
        reads (2D array of float): reads for each strand at each position
            dims (strands x genome length)
        path (str): path to save image, if None, just displays image to screen
    '''

    # Get read data
    if start < 0:
        three_prime = reads[1, int(-start-1):int(-end-1)]
        five_prime = reads[3, int(-start-1):int(-end-1)]
    else:
        three_prime = reads[0, int(start-1):int(end-1)]
        five_prime = reads[2, int(start-1):int(end-1)]

    ave_reads = (three_prime + five_prime) / 2

    n_subplots = 3
    bins = range(int(np.max((five_prime, three_prime))) + 1)
    plt.figure()
    ax = plt.subplot(n_subplots,1,1)
    plt.title(label)
    ax.hist(three_prime, bins=bins)
    ax.set_ylabel("3' Counts")
    ax = plt.subplot(n_subplots,1,2)
    ax.hist(five_prime, bins=bins)
    ax.set_ylabel("5' Counts")
    ax = plt.subplot(n_subplots,1,3)
    ax.hist(ave_reads, bins=bins)
    ax.set_ylabel("Average Counts")

    plt.xlabel('Reads')

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close('all')

def get_git_hash():
    '''
    Returns:
        str: git hash
    '''

    return subprocess.check_output('git rev-parse HEAD'.split()).rstrip()
