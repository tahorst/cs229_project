'''
Useful functions and constants for analysis.
'''

import csv
import os
import re

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')

# .wig files from Lalanne et al
WIG_FILE = os.path.join(RAW_DIR, 'GSM2971252_Escherichia_coli_WT_Rend_seq_5_exo_MOPS_comp_25s_frag_pooled_{}_no_shadow.wig')
WIG_STRANDS = ['3f', '3r', '5f', '5r']

# Genome information
GENOME_SIZE = 4639675
ANNOTATION_FILE = os.path.join(RAW_DIR, 'U00096.2.faa')

def load_wigs(wig_file=None, strands=None, genome_size=None):
    '''
    Loads read data from .wig files.

    Inputs:
        wig_file (str): template for the wig files to be read (gets formatted for each strand),
            if None, defaults to WIG_FILE
        strands (list of str): each strand name that gets inserted into the wig_file string
            (order determines order of first dimension of returned array), if None, defaults
            to WIG_STRANDS
        genome_size (int): base pair length of the genome, if None, defaults to GENOME_SIZE

    Returns 2D array of floats (4 x genome_size) for reads on each strand
    '''

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

    return data

def load_genome(path=None):
    '''
    Loads information about the genome, specifically genes and their locations.

    Args:
        path (str): path to genome file, if None, defaults to ANNOTATION_FILE

    Returns:
        dict (str: dict): keys are gene names, subkeys are {locus: str, start: int, end: int}
    '''

    # Set defaults
    if path is None:
        path = ANNOTATION_FILE

    # Read in file
    with open(path) as f:
        data = f.read()

    genome = {}
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

        d = {}
        d['locus'] = locus
        d['start'] = start
        d['end'] = end

        genome[gene_name] = d

    return genome
