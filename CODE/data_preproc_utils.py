import os
import re
import string
import random
import numpy as np
import pandas as pd
from pybedtools import BedTool
from Bio import SeqIO
import warnings
import logging.config

warnings.filterwarnings("ignore")

## Intialize logger
logging.config.fileConfig('logging.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def intersect_peak_regdna(peak_bed, regdna_bed, gene_df):
    """Intersect peaks and regulatory DNA to obtain signals within the 
    regulatory regions. The relative start and end of each peak is based
    off gene start (if + strand) or gene stop (if - strand). Negative value
    represents upstream of the gene; negative value represents downstream.
    Args:
        peak_bed    - Feature peaks in bed object
        regdna_bed  - Regulatory DNA (region) in bed object
        gene_df    - Gene annotation
    Returns:
        Dataframe of peaks assigned to regulatory regions
    """
    INTER_COLS = [
        'peak_chr', 'peak_start', 'peak_end', 'peak_name', 
        'peak_score', 'reg_chr', 'reg_start', 'reg_end', 
        'gene', 'reg_score', 'strand']
    OUT_COLS = [
        'peak_chr', 'peak_start', 'peak_end', 'peak_score', 
        'gene', 'strand', 'rel_start', 'rel_end']

    inter_bed = peak_bed.intersect(regdna_bed, wb=True)
    inter_df = pd.DataFrame(data=inter_bed.to_dataframe().values, columns=INTER_COLS)
    gene_df = gene_df.rename(columns={
        'name': 'gene', 'start': 'gene_start', 'end': 'gene_end'})
    inter_df = inter_df.merge(
        gene_df[['gene', 'gene_start', 'gene_end']], how='left', on='gene')

    ## Calculate the relative positions of peaks to the gene start. 
    out_df = pd.DataFrame(columns=OUT_COLS)
    for strand, subdf in inter_df.groupby('strand'):
        if strand == '+':
            subdf['rel_start'] = subdf['peak_start'] - subdf['gene_start']
            subdf['rel_end'] = subdf['peak_end'] - subdf['gene_start']
        elif strand == '-':
            subdf['rel_start'] = subdf['gene_end'] - subdf['peak_end'] 
            subdf['rel_end'] = subdf['gene_end'] - subdf['peak_start']
        else:
            logger.warning('WARNING: {} does not have strand info. Skipped.'.format(
                np.unique(subdf['gene'])))
            continue
        out_df = out_df.append(subdf[OUT_COLS], ignore_index=True)
    return out_df


def create_regdna(gene_bed, genome_fa, reg_bound=(500, 500)):
    """Create regulatory region(s) for each gene.
    Args:
        gene_bed    - Gene annotation in bed object
        reg_bound   - Tuple for the boundary of regulatory region, i.e.
                        (upstream distance, downstream distance)
        genome_fa   - Genome fasta filepath
    Returns:
        Regulatory DNA (regions) in bed object
    """
    upstream_bound, downstream_bound = reg_bound
    chrom_size_dict = {x.id: len(x) for x in load_fasta(genome_fa)}
    gene_df = gene_bed.to_dataframe()
    bed_cols = gene_df.columns

    ## Calculate relative start and end, grouped by strand
    regdna_df = pd.DataFrame(columns=bed_cols)
    for chrom, subdf in gene_df.groupby('chrom'):
        for strand, subdf2 in subdf.groupby('strand'):
            if strand == '+':
                subdf2['reg_start'] = subdf2['start'] - upstream_bound
                subdf2['reg_end'] = subdf2['start'] + downstream_bound
            elif strand == '-':
                subdf2['reg_start'] = subdf2['end'] - downstream_bound
                subdf2['reg_end'] = subdf2['end'] + upstream_bound  
            else:
                logger.warning('WARNING: {} does not have strand info. Skipped.'.format(
                    np.unique(subdf2['gene'])))
                continue

            ## Reformat dataframe
            subdf2 = subdf2.drop(columns=['start', 'end'])
            subdf2 = subdf2.rename(columns={'reg_start': 'start', 'reg_end': 'end'})
            subdf2.loc[subdf2['start'] < 1, 'start'] = 1
            subdf2.loc[subdf2['end'] < 1, 'end'] = 1
            subdf2.loc[
                subdf2['start'] >= chrom_size_dict[chrom], 'start'] = \
                chrom_size_dict[chrom] - 1
            subdf2.loc[
                subdf2['end'] >= chrom_size_dict[chrom], 'end'] = \
                chrom_size_dict[chrom] - 1
            regdna_df = regdna_df.append(subdf2, ignore_index=True)
    return BedTool.from_dataframe(regdna_df[bed_cols]).sort()


def calculate_matrix_position(df, upstream_bound):
    """Create new columns for the positions of features in feature matrix. 
    """
    df['mtx_start'] = df['rel_start'] + upstream_bound
    df['mtx_end'] = df['rel_end'] + upstream_bound
    return df


def get_onehot_dna_sequence(regdna_bed, genome_fa, reg_bound, genes):
    """Get DNA sequence from regulatory DNA bed object.
    """
    FEAT_COL = ['gene_idx', 'mtx_start', 'mtx_end', 'peak_score', 'alphabet']
    dna_df_list = []

    regdna_df = regdna_bed.to_dataframe()
    regdna_fa = regdna_bed.getfasta(fi=genome_fa, name=True)

    for x in load_fasta(regdna_fa.seqfn):
        ## Keep upstream at left
        strand = regdna_df.loc[regdna_df['name'] == x.id, 'strand'].values[0]
        seq = str(x.seq) if strand == '+' else str(x.seq)[::-1]
        ## Convert one-hot encoding
        dummy_seq = pd.get_dummies(list(seq))
        shift = reg_bound[0] + reg_bound[1] - len(seq)
        if shift > 0:  ## if regulatory region is shorter than queried region
            blank_seq = pd.DataFrame(
                data=np.zeros((shift, 4), dtype=int), 
                columns=dummy_seq.columns)
            dummy_seq = blank_seq.append(dummy_seq, ignore_index=False)

        ## Convert to row and col index of the entries of ones
        coord_idx, alphabet_idx = np.where(dummy_seq == 1)
        gene_idx = genes.index(x.id)
        tmp_df = pd.DataFrame({
            'mtx_start': coord_idx,
            'mtx_end': coord_idx + 1,
            'alphabet': alphabet_idx})
        tmp_df['gene_idx'] = gene_idx
        tmp_df['peak_score'] = 1
        dna_df_list.append(tmp_df)
    return pd.concat(dna_df_list, ignore_index=True)[FEAT_COL]


def get_onehot_dna_sequence_slim(regdna_bed, genome_fa, tss_df):
    """Get DNA sequence from regulatory DNA bed object.
    """
    FEAT_COL = ['gene_idx', 'rel_dist', 'alphabet']
    dna_df_list = []

    regdna_df = regdna_bed.to_dataframe()
    regdna_fa = regdna_bed.getfasta(fi=genome_fa, name=True)
    genes = tss_df['name'].tolist()

    print('==> get_onehot_dna_sequence_slim <==')
    for i, s in enumerate(load_fasta(regdna_fa.seqfn)):
        ## Keep upstream at left
        seq_info = regdna_df.iloc[i]
        strand = seq_info['strand']
        start_pos = seq_info['start']
        end_pos = seq_info['end']

        name = s.id.split(':')[0]
        chrom = s.id.split(':')[2]
        tmp = seq_info['name']
        tss_pos = tss_df.loc[tss_df['name'] == name, 'start'].iloc[0]
        # tss_pos = tss_df.loc[tss_df['name'] == s.id, 'start'].iloc[0]
        
        if strand == '+':
            rel_dists = np.arange(start_pos, end_pos, dtype=int) - tss_pos
            seq = str(s.seq)
        else: 
            rel_dists = tss_pos - np.arange(start_pos, end_pos, dtype=int)
            seq = str(s.seq)[::-1]
        ## Convert one-hot encoding
        dummy_seq = pd.get_dummies(list(seq))

        ## Convert to row and col index of the entries of ones
        _, alphabet_idx = np.where(dummy_seq == 1)

        if len(rel_dists) != len(alphabet_idx):
            print(s.id)

        gene_idx = genes.index(name)
        # gene_idx = genes.index(s.id)
        tmp_df = pd.DataFrame({
            'gene_idx': [gene_idx] * len(rel_dists),
            'rel_dist': rel_dists,
            'alphabet': alphabet_idx})
        dna_df_list.append(tmp_df)
    return pd.concat(dna_df_list, ignore_index=True)[FEAT_COL]


def get_nt_frequency(regdna_bed, genome_fa, genes):
    """Get di-nucleotide frequences of regualtory DNA.
    """
    DINUCLEOTIDES = [
        ['AA', 'TT'], ['AC', 'GT'], ['AG', 'CT'],
        ['CA', 'TG'], ['CC', 'GG'], ['GA', 'TC'],
        ['AT'], ['CG'], ['GC'], ['TA']]

    ## Initialize freq dict for dints
    nt_dict = {}
    for dints in DINUCLEOTIDES:
        for dint in dints:
            nt_dict[dint] = np.zeros(len(genes))
    len_dict = {i: 0 for i in range(len(genes))}

    regdna_bed = regdna_bed.getfasta(fi=genome_fa, name=True)

    for s in load_fasta(regdna_bed.seqfn):
        gene_idx = genes.index(s.id)
        seq_len = len(s.seq)
        seq = str(s.seq)
        
        ## Add dint count and sequence length for each region 
        for k in range(seq_len - 1):
            dint = seq[k: k + 2]
            if 'N' not in dint:
                nt_dict[dint][gene_idx] += 1
        len_dict[gene_idx] += seq_len

    ## Calcualte frequency
    for dint in nt_dict.keys():
        for gene_idx, seq_len in len_dict.items():
            nt_dict[dint][gene_idx] /= seq_len

    ## Combine reverse complement
    for dints in DINUCLEOTIDES:
        if len(dints) > 1:
            nt_dict[dints[0]] = nt_dict[dints[0]] + nt_dict[dints[1]]
            del nt_dict[dints[1]]
    return nt_dict


def convert_gnashy_to_bed(filename, binarize_peak_score=False):
    """Convert 3-column gnashy file to bed obejct.
    Args:
        filename                - Gnashy filename
        binarize_peak_score     - Boolean flag
    Returns: 
        Output bed object
    """
    with open(filename, "r") as f:
        lines = f.readlines()
        bed = []
        for i, line in enumerate(lines):
            chrm, pos, score = line.strip().split("\t")
            if binarize_peak_score:
                score = 1
            bed.append(["chr" + chrm, int(pos), int(pos) + 1, ".", score])
    return BedTool.from_dataframe(pd.DataFrame(bed))


def convert_orf_to_bed(filename):
    """Convert yeast orf fasta headers to orf coordinate in bed format.
    Args:
        filename         - Fastq filename
    Returns:
        Gene annotation in bed object
    """
    CH_DICT = {
        'Chr I': 'chr1', 'Chr II': 'chr2', 'Chr III': 'chr3', 
        'Chr IV': 'chr4', 'Chr V': 'chr5', 'Chr VI': 'chr6', 
        'Chr VII': 'chr7', 'Chr VIII': 'chr8', 'Chr IX': 'chr9', 
        'Chr X': 'chr10', 'Chr XI': 'chr11', 'Chr XII': 'chr12', 
        'Chr XIII': 'chr13', 'Chr XIV': 'chr14', 'Chr XV': 'chr15', 
        'Chr XVI': 'chr16', 'Chr Mito': 'chrm'}

    with open(filename, 'r') as f:
        lines = f.readlines()

    bed = []
    for i in range(len(lines)):
        if lines[i].startswith('>'):
            x = lines[i].strip().strip('>').split(', ')
            g = x[0].split(' ')[0]
            s = '+' if g.split('-')[0].endswith('W') else '-'

            if x[1].startswith('2-micron plasmid'):  ## deal with plasmid 
                y = re.split('-|,| from ', x[1].strip('2-micron '))
                ch = y[0]
            else:  ## deal with other chromsomes
                y = re.split('-|,| from ', x[1])
                ch = CH_DICT[y[0]]
            pos = np.array(y[1:], dtype=int)
            g_start, g_end = np.min(pos), np.max(pos)
            bed.append([ch, g_start, g_end, g, '.', s])
    return BedTool.from_dataframe(pd.DataFrame(bed)).sort()


def liftover_bed(in_bed, chain_filename):
    """Liftover bed object from one reference genome to another using chain file.
    Args:
        in_bed          - Bed object
        chain_filename  - Chain filename for liftover
    Returns:
        Output bed object
    """
    in_filename = '/tmp/{}.bed'.format(
        ''.join(random.choice(string.ascii_lowercase) for x in range(10)))
    out_filename = '/tmp/{}.bed'.format(
        ''.join(random.choice(string.ascii_lowercase) for x in range(10)))
    unmapped_filename = '/tmp/{}.bed'.format(
        ''.join(random.choice(string.ascii_lowercase) for x in range(10)))

    in_bed.saveas(in_filename)
    os.system('liftOver {} {} {} {}'.format(
        in_filename, chain_filename, out_filename, unmapped_filename))
    return BedTool(out_filename)


def load_fasta(filepath):
    return SeqIO.parse(filepath, 'fasta')
