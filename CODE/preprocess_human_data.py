import sys
import os.path
import argparse
import h5py
import numpy as np
import pandas as pd
from pybedtools import BedTool
import logging.config

from data_preproc_utils import create_regdna, intersect_peak_regdna, \
    calculate_matrix_position, get_onehot_dna_sequence_slim, get_nt_frequency

## Intialize logger
logging.config.fileConfig('logging.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def parse_args(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-o', '--output_h5', required=True,
        help='Output hdf5 file.')
    parser.add_argument(
        '-a', '--gene_annot', required=True,
        help='Bed file for ORF or TSS annotation data.')
    parser.add_argument(
        '-r', '--regdna_bed', required=True,
        help='Bed file for regulatory DNA (promoters + enhancers).')
    parser.add_argument(
        '-f', '--genome_fa', required=True,
        help='Fasta file for genome sequence.')
    parser.add_argument(
        '--tf_bind', nargs='*',
        help='Bed file(s) for TF binding locations data (delimited by single space). Wildcard in filename (*) is accpetable.')
    parser.add_argument(
        '--hist_mod', nargs='*',
        help='Bed file(s) for histone modifications data (delimited by single space). Wildcard in filename (*) is accpetable.')
    parser.add_argument(
        '--chrom_acc', nargs='*',
        help='Bed file(s) for chromatin accessibility data (delimited by single space). Wildcard in filename (*) is accpetable.')
    parser.add_argument(
        '--gene_expr', nargs='*',
        help='Csv file for pre-perturbation gene expression data.')
    parser.add_argument(
        '--gene_var', nargs='*',
        help='Csv file for gene expression variation data.')
    parsed = parser.parse_args(argv[1:])
    return parsed


def parse_feature_dictionary(X):
    return {os.path.splitext(os.path.basename(x))[0]: x for x in X}


def parse_features(args):
    """Parse feature in args into structured dictionary with first
    key as feature type, second key as feature name, and value as
    filepath.
    """ 
    D = {
        'dna_sequence': args.genome_fa}
    if args.tf_bind:
        D.update({
            'tf_binding': parse_feature_dictionary(args.tf_bind)})
    if args.hist_mod:
        D.update({
            'histone_modifications': parse_feature_dictionary(args.hist_mod)})
    if args.chrom_acc:
        D.update({
            'chromatin_accessibility': parse_feature_dictionary(args.chrom_acc)})
    if args.gene_expr:
        D.update({
            'gene_expression': parse_feature_dictionary(args.gene_expr)})
    if args.gene_var:
        D.update({
            'gene_variation': parse_feature_dictionary(args.gene_var)})
    return D


def generate_features(h5, tss, regdna, feat_dict):
    """Generate datasets for all features in hdf5 file.
    Args:
        h5          - h5 filename
        tss         - TSSs or ORF starts of genes
        regdna      - Regulatory DNA bed filename
        feat_dict   - Dictionary of feature types and feature names
    Returns:
        NULL
    """
    DNA_COL = ['gene_idx', 'rel_dist']
    EPIG_COL = ['gene_idx', 'rel_start', 'rel_end', 'peak_score']

    ## Parse regulator DNA and gene list
    regdna_bed = BedTool(regdna)
    genes = sorted(pd.unique(regdna_bed.to_dataframe()['name']))
    tss_df = BedTool(tss).to_dataframe()
    tss_df = tss_df[tss_df['name'].isin(genes)]

    ## Write grouped datasets into hdf5
    with h5py.File(h5, 'w') as f:
        ## Gene list 
        f.create_dataset(
            'genes', 
            data=np.array(genes, dtype='S'), 
            compression='gzip')

        for k1, v1 in feat_dict.items(): 
            ## Regulatory DNA sequence
            if k1 == 'dna_sequence': 
                logger.info('Working on {} {}'.format(k1, v1))
                ## Store one-hot encode sequence in A, C, G, T channels
                # logger.debug('... one-hot DNA')
                regdna_df = get_onehot_dna_sequence_slim(regdna_bed, v1, tss_df)
                g = f.require_group(k1)
                for i, alphabet in enumerate(['A', 'C', 'G', 'T']):
                    g.create_dataset(
                        alphabet, 
                        data=regdna_df.loc[
                            regdna_df['alphabet'] == i, DNA_COL].values.astype(int), 
                        compression='gzip')
                # Store single and di-nucleotide frequencies
                # logger.debug('... di-nucleotide freqs')
                nt_freq_dict = get_nt_frequency(regdna_bed, v1, genes)
                g = f.require_group(k1 + '_nt_freq')
                for nt, freq_arr in nt_freq_dict.items():
                    g.create_dataset(
                        nt,
                        data=np.array(freq_arr, dtype='float16'),
                        compression='gzip')

            ## Gene expression level features
            elif k1 == 'gene_expression' or k1 == 'gene_variation': 
                for k2, v2 in v1.items():
                    logger.info('Working on {} {}'.format(k1, k2))
                    g = f.require_group('gene_expression')
                    ## Load gene expression matrix and sort genes in the 
                    ## same dimension as other features
                    expr_df = pd.read_csv(v2, index_col=0)
                    expr_df = expr_df.reindex(genes)
                    if k1 == 'gene_variation':
                        g.create_dataset(
                            'variation',
                            data=expr_df.values[:, 0].astype(float),
                            compression='gzip')
                    else:
                        g.create_dataset(
                            'median_level',
                            data=expr_df.values[:, 0].astype(float),
                            compression='gzip')

            ## Epigenetic features, e.g. TF binding data
            else: 
                g = f.require_group(k1)
                for k2, v2 in v1.items():  ## Feature name, e.g. TF name
                    logger.info('Working on {} {}'.format(k1, k2))
                    v2_bed = BedTool(v2)
                    ## Intersect peak features with regulatory region
                    feat_df = intersect_peak_regdna(v2_bed, regdna_bed, tss_df)
                    feat_df['gene_idx'] = [genes.index(x) for x in feat_df['gene']]
                    g.create_dataset(
                        k2, 
                        data=feat_df[EPIG_COL].values.astype(float), 
                        compression='gzip')


def main(argv):
    args = parse_args(argv)
    feature_dict = parse_features(args)
    gene_annot = args.gene_annot
    regdna_bed = args.regdna_bed
    output_h5 = args.output_h5

    generate_features(output_h5, gene_annot, regdna_bed, feature_dict)


if __name__ == "__main__":
    main(sys.argv)
