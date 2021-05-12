import sys
import numpy as np
import pandas as pd
import configparser
import logging.config
import h5py
import scipy.sparse as sps
import multiprocess as mp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pybedtools import BedTool


## Intialize logger
logging.config.fileConfig('logging.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

## Load default configuration
config = configparser.ConfigParser()
config.read('config.ini')
RAND_NUM = int(config['DEFAULT']['rand_num'])


def construct_expanded_input(filepath_dict, feat_info_dict):
    """Construct 2D feature matrix and label dataframe for a given TF. The 2D feature 
    matrix = (gene x feature), in which feature values defined by enhancer and promoter regions along the genomic coordinate are concatenated horizontally .
    Args:
        filepath_dict   - Filepath dictionary for h5 features and response labels
        feat_info_dict  - Feature information dictionary for feature types,
                        quantized length, and quantized bins
    Returns:
        A tuple of feature matrix (in numpy.ndarray), corresponding features, 
        and label dataframe
    """
    ## Get common genes between feature matrix and label matrix
    tfs = feat_info_dict['tfs']
    h5_filepath = filepath_dict['feat_h5']
    label_filepath = filepath_dict['resp_label']
    genes, gene_map, _ = create_gene_index_map(
        h5_filepath, label_filepath, True, 'gene_ensg')

    ## Create model label and feature matrix
    labels_dict = {tf: create_model_label(
        label_filepath, tf, genes, True, 'tf_ensg', 'gene_ensg') for tf in tfs}

    ## Create feature name lists for tf related and tf unrelated feature types and names
    tf_features, nontf_features = get_h5_features(
        h5_filepath, feat_info_dict['feat_types'], tfs)
    tf_feat_types = sorted(set([x[0] for x in tf_features]))

    ## Create feature matrix for each feautre in parallel
    tf_mp_dict = create_feat_mtx_parallel(
        tf_features, h5_filepath, gene_map,
        is_fixed_input=False,
        promo_bound=feat_info_dict['promo_bound'],
        enhan_bound=feat_info_dict['enhan_bound'],
        promo_width=feat_info_dict['promo_width'],
        enhan_min_width=feat_info_dict['enhan_min_width'])

    nontf_mp_dict = create_feat_mtx_parallel(
        nontf_features, h5_filepath, gene_map,
        is_fixed_input=False,
        promo_bound=feat_info_dict['promo_bound'],
        enhan_bound=feat_info_dict['enhan_bound'],
        promo_width=feat_info_dict['promo_width'],
        enhan_min_width=feat_info_dict['enhan_min_width'])

    ## Concatenate feature matrices in order
    col_idx = 0
    tf_feat_mtx_dict = {}
    feat_details = []

    for i, tf in enumerate(tfs):
        tf_feat_mtx = sps.csc_matrix((len(gene_map), 0))

        for feat_type in tf_feat_types:
            mtx = tf_mp_dict[(feat_type, tf)]
            mtx_bins = mtx.shape[1]
            tf_feat_mtx = sps.hstack((tf_feat_mtx, mtx))

            if i == 0:  ## Only increment for the first TF
                feat_details.append((feat_type, 'TF') + (col_idx, col_idx + mtx_bins))
                col_idx += mtx_bins

        tf_feat_mtx_dict[tf] = tf_feat_mtx.toarray()

    ## Concatenate feature matrices in order for tf unrelated features 
    nontf_feat_mtx = sps.csc_matrix((len(gene_map), 0))

    for feat_tuple in nontf_features:
        mtx = nontf_mp_dict[feat_tuple]
        mtx_bins = mtx.shape[1]
        nontf_feat_mtx = sps.hstack((nontf_feat_mtx, mtx))

        feat_details.append(feat_tuple + (col_idx, col_idx + mtx_bins))
        col_idx += mtx_bins
    
    nontf_feat_mtx = nontf_feat_mtx.toarray()

    return tf_feat_mtx_dict, nontf_feat_mtx, feat_details, labels_dict


def construct_fixed_input(filepath_dict, feat_info_dict):
    """Construct 2D feature matrix and label dataframe for a given TF. The 2D feature 
    matrix = (gene x feature), in which feature values defined by fixed range for
    promoter regions along the genomic coordinate are concatenated horizontally.
    Args:
        filepath_dict   - Filepath dictionary for h5 features and response labels
        feat_info_dict  - Feature information dictionary for feature types,
                        quantized length, and quantized bins
    Returns:
        A tuple of feature matrix (in numpy.ndarray), corresponding features, 
        and label dataframe
    """
    ## Get common genes between feature matrix and label matrix
    tfs = feat_info_dict['tfs']
    h5_filepath = filepath_dict['feat_h5']
    label_filepath = filepath_dict['resp_label']
    genes, gene_map, _ = create_gene_index_map(h5_filepath, label_filepath)

    ## Create model label in dict
    labels_dict = {tf: create_model_label(label_filepath, tf, genes) for tf in tfs}

    ## Create feature name lists for tf related and tf unrelated feature types and names
    tf_features, nontf_features = get_h5_features(
        h5_filepath, feat_info_dict['feat_types'], tfs)
    tf_feat_types = sorted(set([x[0] for x in tf_features]))

    ## Create feature matrix for each feautre in parallel
    tf_mp_dict = create_feat_mtx_parallel(
        tf_features, h5_filepath, gene_map, 
        feat_length=feat_info_dict['feat_length'], 
        feat_bins=feat_info_dict['feat_bins'])

    nontf_mp_dict = create_feat_mtx_parallel(
        nontf_features, h5_filepath, gene_map, 
        feat_length=feat_info_dict['feat_length'], 
        feat_bins=feat_info_dict['feat_bins'])

    ## Concatenate feature matrices in order for tf related features
    col_idx = 0
    tf_feat_mtx_dict = {}
    feat_details = []

    for i, tf in enumerate(tfs):
        tf_feat_mtx = sps.csc_matrix((len(gene_map), 0))

        for feat_type in tf_feat_types:
            mtx = tf_mp_dict[(feat_type, tf)]
            mtx_bins = mtx.shape[1]
            tf_feat_mtx = sps.hstack((tf_feat_mtx, mtx))

            if i == 0:  ## Only increment for the first TF
                feat_details.append((feat_type, 'TF') + (col_idx, col_idx + mtx_bins))
                col_idx += mtx_bins

        tf_feat_mtx_dict[tf] = tf_feat_mtx.toarray()

    ## Concatenate feature matrices in order for tf unrelated features 
    nontf_feat_mtx = sps.csc_matrix((len(gene_map), 0))

    for feat_tuple in nontf_features:
        mtx = nontf_mp_dict[feat_tuple]
        mtx_bins = mtx.shape[1]
        nontf_feat_mtx = sps.hstack((nontf_feat_mtx, mtx))

        feat_details.append(feat_tuple + (col_idx, col_idx + mtx_bins))
        col_idx += mtx_bins

    nontf_feat_mtx = nontf_feat_mtx.toarray()

    return tf_feat_mtx_dict, nontf_feat_mtx, feat_details, labels_dict


def create_feat_mtx_parallel(features, h5_filepath, gene_map, is_fixed_input=True, **kwargs):
    """Create feature matrix for each feautre in parallel
    """
    mp_dict = mp.Manager().dict()
    mp_jobs = []
    ## Define parallel jobs
    for feat_tuple in features:
        mp_job = mp.Process(
            target=create_feat_mtx_wrapper, 
            args=(mp_dict, feat_tuple, h5_filepath, gene_map, is_fixed_input),
            kwargs=kwargs
        )
        mp_jobs.append(mp_job)
    ## Execute parallel jobs
    _ = [p.start() for p in mp_jobs]
    _ = [p.join() for p in mp_jobs]
    return mp_dict


def create_feat_mtx_wrapper(D, k, h5_filepath, gene_map, is_fixed_input, **kwargs):
    """Wrapper for create_feat_mtx.
    """
    if is_fixed_input:
        D[k] = create_fixed_feat_mtx(h5_filepath, k, gene_map, **kwargs)
    else:
        D[k] = create_expanded_feat_mtx(h5_filepath, k, gene_map, **kwargs)


def create_fixed_feat_mtx(filepath, feat_tuple, gene_map, **kwargs):
    """Create feature matrix for a feature defined by fixed regulatory regions
    (single promoters).
    """
    feat_type, feat_name = feat_tuple
    feat_length = kwargs.get('feat_length', None)
    feat_bins = kwargs.get('feat_bins', None)

    logger.info('Calculating feature: {} > {}'.format(feat_type, feat_name))

    mtx = load_h5_mtx(filepath, feat_type, feat_name)
    if feat_type == 'gene_expression' or feat_type == 'dna_sequence_nt_freq':
        ## Build genomic location independent feature
        mtx = create_expr_vector(mtx)
        mtx = map_feature_mtx_gene_index(mtx, gene_map)
        feat_width = 1
    else:
        ## Build genomic location dependent feature
        mtx = map_feature_mtx_gene_index(mtx, gene_map)
        mtx = convert_cont_adjmtx(mtx)
        ## Quantize feature matrix
        if feat_bins is not None:
            bin_width = feat_length / feat_bins
            mtx = quantize_feature_mtx(mtx, bin_width)
            feat_width = feat_bins
        else:
            feat_width = feat_length
    mtx = convert_adjmtx_to_sparsemtx(mtx, len(gene_map), feat_width)
    return mtx


def create_expanded_feat_mtx(filepath, feat_tuple, gene_map, **kwargs):
    """Create feature matrix for a feature defined by expaned regulatory regions
    (promoters + enhancers).
    """
    feat_type, feat_name = feat_tuple
    pbound = kwargs.get('promo_bound', 0)
    pwidth = kwargs.get('promo_width', 0)
    ebound = kwargs.get('enhan_bound', None)
    ewidth = kwargs.get('enhan_min_width', None)

    logger.info('Calculating feature: {} > {}'.format(feat_type, feat_name))

    mtx = load_h5_mtx(filepath, feat_type, feat_name)
    if feat_type == 'gene_expression' or feat_type == 'dna_sequence_nt_freq':
        ## Build genomic location independent feature
        mtx = create_expr_vector(mtx)
        mtx = map_feature_mtx_gene_index(mtx, gene_map)
        feat_width = 1
    else:
        ## Create bin regions
        bins = create_ext_bins(pbound, ebound, pwidth, ewidth)
        ## Build coordinate dependent feature
        mtx = map_feature_mtx_gene_index(mtx, gene_map)
        ## Map features into bins
        mtx = map_feature_mtx_to_bins(mtx, bins)
        feat_width = len(bins)
    mtx = convert_adjmtx_to_sparsemtx(mtx, len(gene_map), feat_width)
    return mtx


def create_ext_bins(pbound, ebound, pwidth, ewidth):
    """Create bins for promoters and enhancers into concatenated intervals (bins).
    The first n bins are the intervals ranging from upstream to downstream promoter.
    The next m bins represent the upstream enhancers, followed by another m bins
    representing the downstream enhancers.
    Note: the bins for each promoter/enhancer are sorted by the relative distance
    to TSS, i.e. from upstream (negative dist) to downstream (postive dist).
    """
    ## Create bins for pomoters
    bins = [[-i, -i + pwidth] for i in np.arange(pbound[0], 0, -pwidth, dtype=int)]
    bins += [[i, i + pwidth] for i in np.arange(0, pbound[1], pwidth, dtype=int)]
    if ebound is None:
        return np.array(bins)
    ## Create bins for enhancers
    if ewidth is not None:
        up_enhan_bins = create_exponential_bins(
            pbound[0], ebound[0], ewidth, is_upstream=True)
        down_enhan_bins = create_exponential_bins(
            pbound[1], ebound[1], ewidth, is_upstream=False)
    else:
        up_enhan_bins = [[-ebound[0], -pbound[0]]]
        down_enhan_bins = [[pbound[1], ebound[1]]]
    bins += up_enhan_bins + down_enhan_bins
    return np.array(bins)


def create_exponential_bins(start, stop, w, is_upstream=False):
    """Create bins with exponetial widths.
    """
    ticks = [start]
    i = 1
    while ticks[-1] <= stop:
        x = i * w + ticks[-1]
        ticks.append(int(x))
        i += 1
    ticks[-1] = stop
    
    if is_upstream:
        bins = [[-ticks[j], -ticks[j - 1]] for j in range(len(ticks) - 1, 0, -1)]
    else:
        bins = [[ticks[j], ticks[j + 1]] for j in range(len(ticks) - 1)]
    return bins


def map_feature_mtx_to_bins(mtx, bins, shift=10 ** 7):
    """Map coordinate dependent features into bins of regulatory regions.
    """
    ## Convert feature matrix and bin vector into bed
    f_len = mtx.shape[0]
    b_len = len(bins)

    m = BedTool.from_dataframe(
        pd.DataFrame({
            'chrom': ['.'] * f_len,
            'start': mtx[:, 1].astype(int) + shift, 
            'end': mtx[:, 2].astype(int) + shift,
            'name': mtx[:, 0].astype(int), 
            'score': mtx[:, 3]
        }))
    b = BedTool.from_dataframe(
        pd.DataFrame({
            'chrom': ['.'] * b_len,
            'start': bins[:, 0].astype(int) + shift, 
            'end': bins[:, 1].astype(int) + shift,
            'name': ['.'] * b_len,
            'score': list(range(b_len))
        }))

    ## Intersect m and b
    mb = m.intersect(b, wa=True, wb=True).to_dataframe()
    mb = mb.rename(columns={
        'name': 'gene', 'blockCount': 'bin',
        'thickStart': 'bStart', 'thickEnd': 'bEnd'
    })

    ## Divide the score if a feature is wider than a bin
    mb['segStart'] = np.max(mb[['start', 'bStart']], axis=1)
    mb['segEnd'] = np.min(mb[['end', 'bEnd']], axis=1)
    seg_frac = (mb['segEnd'] - mb['segStart']) / (mb['end'] - mb['start'])
    mb['segScore'] = mb['score'] * seg_frac

    ## Sum scores within each bin
    mb = mb.groupby(['gene', 'bin'])['segScore'].sum().reset_index()
    return mb[['gene', 'bin', 'segScore']].values


def create_model_label(filepath, tf, genes, is_long_csv=False, tf_col=None, gene_col=None):
    """Create label vector matching gene list.
    Args:
        filepath        - Filepath of label csv matrix
        tf              - Column (Transcription factor) of interest
        genes           - Gene list, by which the label is ordered
        is_long_csv     - csv file in long (True) or wide (False) format
        tf_col          - Column name for TFs
        gene_col        - Column name for genes
    Returns:
        Label dataframe
    """
    #TOOD: Unify label csv as long format.
    if is_long_csv:
        df = pd.read_csv(filepath)
        if tf not in df[tf_col].unique():
            logger.error('TF {} not found in label file. ==> Aborted <=='.format(tf))
            sys.exit(1)

        cols = [gene_col, 'log2FoldChange'] 
        if 'padj' in df.columns:
            cols.append('padj')
        label_df = df.loc[df[tf_col] == tf, cols]
        label_df = label_df.set_index(gene_col)
    
    else:
        df = pd.read_csv(filepath, index_col=0)
        if tf not in df.columns:
            logger.error('TF {} not found in label file. ==> Aborted <=='.format(tf))
            sys.exit(1)

        label_df = df[tf]
    label_df = map_label_gene_index(label_df, genes)
    return label_df


def get_h5_features(h5_filepath, feat_types, tfs):
    """Get all features in the h5 file based on feature types.
    Args:
        h5_filepath     - h5 filepath
        feat_types      - List of feature types
        tf              - TF name
    Returns:
        List of tuple (feature type, feature name)
    """
    tf_features = set()
    nontf_features = set()
    for x in feat_types:
        if x in ['tf_binding', 'binding_potential']:
            for tf in tfs:
                tf_features.add((x, tf))
        elif x == 'gene_expression':
            for tf in tfs:
                ys = list_h5_datasets(h5_filepath, x)
                if tf in ys:
                    tf_features.add((x, tf))
                elif 'median_level' in ys:
                    nontf_features.add((x, 'median_level'))
        elif x == 'gene_variation':
            nontf_features.add(('gene_expression', 'variation'))
        else:
            nontf_features |= {(x, y) for y in list_h5_datasets(h5_filepath, x)}
    return sorted(tf_features), sorted(nontf_features)


def load_h5_genes(filepath):
    """Load genes list from h5 file.
    """
    with h5py.File(filepath, 'r') as f:
        genes = f['genes'][:].tolist()
        return [x.decode('utf-8') for x in genes]


def load_h5_mtx(filepath, feat_type, feat_name):
    """Load h5 feature matrix based on the type and name of the feature.
    """
    with h5py.File(filepath, 'r') as f:
        return f['{}/{}'.format(feat_type, feat_name)][:]


def list_h5_datasets(filepath, feat_type):
    """List datasets under a h5 group (feature type).
    """
    with h5py.File(filepath, 'r') as f:
        return list(f[feat_type].keys())


def load_csv_genes(filepath, is_long_csv, gene_col):
    """Load gene list from matrix.
    """
    if is_long_csv:
        df = pd.read_csv(filepath)
        genes = sorted(df[gene_col].unique())
    else:
        df = pd.read_csv(filepath, usecols=[0], index_col=0)
        genes = df.index.tolist()
    return genes


def create_gene_index_map(h5_filepath, csv_filepath, is_long_csv=False, gene_col=None):
    """Map gene index of h5 and gene index of csv to the common genes. 
    Args:
        h5_filepath     - h5 file for signals in regulatory regions 
        csv_filepath    - csv file for label signals
        is_long_csv     - csv file in long (True) or wide (False) format
        gene_col        - Column name for genes
    Return: 
        Tuple (common_genes, h_map, c_map), representing the common genes in the 
        intersection, and index dictionaries for mapping h5 genes and csv genes 
        to common genes respectively.
    """
    h_genes = load_h5_genes(h5_filepath)
    c_genes = load_csv_genes(csv_filepath, is_long_csv, gene_col)
    common_genes = sorted(set(h_genes) & set(c_genes))
    h_map = {h_genes.index(x): i for i, x in enumerate(common_genes)}
    c_map = {c_genes.index(x): i for i, x in enumerate(common_genes)}
    return (common_genes, h_map, c_map)


def map_feature_mtx_gene_index(mtx, D):
    """Filter and map the gene index (the first column of feature matrix)
    based on index mapping dictionary.
    Args:
        mtx     - Numpy matrix
        D       - Index mapping dictionary
    Return:
        Numpy matrix
    """
    ## Fitler current gene index
    idx = np.argwhere(np.isin(mtx[:, 0], sorted(D.keys())))
    mtx = mtx[idx.squeeze(), :]
    ## Map to new gene index 
    mtx[:, 0] = list(map(D.get, mtx[:, 0]))
    return mtx


def map_label_gene_index(df, genes):
    """Filter and map the gene index of label dataframe.
    """
    return df.reindex(genes)


def convert_cont_adjmtx(mtx):
    """Transform feature matrix to continuous adjacency matrix, where the feature 
    value is assigned to single base position.
    """
    mtx2 = []
    for row in mtx:
        i, start, end, val = row
        i, start, end = int(i), int(start), int(end)
        for j in range(start, end):
            mtx2.append([i, j, val])
    return np.array(mtx2)


def quantize_feature_mtx(mtx, width):
    """Bin features along genomic position, and aggregate values within each bin.
    Args:
        mtx     - 4-col numpy matrix, whose row is (gene index, strat, end, value)
        width   - Width of bin
    Returns:
        3-col numpy matrix, whose row is (gene index, bin index, value)
    """
    mtx2 = []
    df = pd.DataFrame(mtx)
    for i, subdf in df.groupby(0):
        subdf2 = subdf.groupby(subdf[1] // width)[2].sum()
        m1 = i * np.ones(subdf2.shape[0]).reshape(-1, 1)
        m2 = subdf2.reset_index().values
        mtx2 += np.hstack((m1, m2)).tolist()
    return np.array(mtx2)


def convert_adjmtx_to_sparsemtx(mtx, gene_num, feat_len):
    """Convert adjacency matrix to csc matrix.
    Args:
        mtx         - 3-col adjacency matrix
        gene_num    - Number of genes
        feat_len    - Length of feature
    Returns:
        Sparse csc matrix
    """
    csc_shape = (gene_num, feat_len)
    return sps.csc_matrix((mtx[:, 2], (mtx[:, 0], mtx[:, 1])), shape=csc_shape)


def create_expr_vector(mtx):
    """Create expression profile as a feature vector.
    """
    n = len(mtx)
    idx = np.hstack(
        (
            np.arange(n, dtype=int).reshape(-1, 1), 
            np.zeros(n, dtype=int).reshape(-1, 1)
        )
    )
    mtx = np.hstack((idx, mtx.reshape(-1, 1)))
    return mtx


def standardize_feat_mtx(X_tr, X_te, method='zscore'):
    """Standardize feature matrix in 2D and 3D using Z-score or min-max transformation.
    Args:
        X_tr        - Feature matrix for training
        X_te        - Feature matrix for test
        method      - Standardization method. Choose from `zscore` and `minmax`.
    Returns:
        Standarized feature matrix
    """
    if X_tr.shape[1] == 0:
        return X_tr, X_te
    if len(X_tr.shape) == 3:
        X_tr_xform, X_te_xform = X_tr.copy(), X_te.copy()
        for i in range(X_tr.shape[1]):
            X_tr_i, X_te_i = standardize_2d_feat_mtx(
                X_tr[:, i, :], X_te[:, i, :], method)
            X_tr_xform[:, i, :] = X_tr_i
            X_te_xform[:, i, :] = X_te_i
        return X_tr_xform, X_te_xform
    else:
        return standardize_2d_feat_mtx(X_tr, X_te, method)


def standardize_2d_feat_mtx(X_tr, X_te, method):
    """Standardize 2D feature matrix using Z-score (zero mean and std dev). 
    """
    if method.lower() == 'zscore':
        scaler = StandardScaler()
    elif method.lower() == 'minmax':
        scaler = MinMaxScaler()
    scaler.fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te) if X_te is not None else None


def binarize_label(y, lfc_cutoff=None, p_cutoff=None):
    """Binarize the label of absolute response level based on cutoff.
    """
    if lfc_cutoff is None and p_cutoff is None:
        return y
    y2 = pd.Series(index=y.index, data=np.zeros(y.shape[0], dtype=int))
    if p_cutoff is None:
        y2[np.abs(y) > lfc_cutoff] = 1
    else:
        y2[(np.abs(y['log2FoldChange']) > lfc_cutoff) & (y['padj'] < p_cutoff)] = 1
    return y2


def find_sequence_complement_indx(D):
    """ Find mapping from indx to complement index.
    """
    compl_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    D2 = {}
    for k1, v1 in D.items():
        for k2, v2 in compl_dict.items():
            if k1 == k2:
                D2[v1] = D[v2]
                break
    return D2


def create_random_input(X, y, n_samples=1000):
    """Create random input for model interpretation by randomly sampling feature values.
    """
    np.random.seed(RAND_NUM)

    n_feats = X.shape[1]
    X_rand = np.empty((n_samples, n_feats))
    for j in range(n_feats):
        X_rand[:, j] = np.random.choice(X[:, j], size=n_samples, replace=True)

    genes = np.array(['rand_' + str(i) for i in range(n_samples)])
    cv_folds = pd.unique(y['cv'])
    n_per_fold = int(n_samples / len(cv_folds))
    y_rand = pd.DataFrame({
        'gene': genes,
        'cv': np.ndarray.flatten(np.array(
            [[k] * n_per_fold for k in cv_folds]))})
    return X_rand, y_rand, genes


def compile_mp_results(mp_dicts):
    """Compile the array of dicts into one dict.
    """
    result_dict = {d: [] for d in ['preds', 'stats', 'models']}
    for k in sorted(mp_dicts.keys()):
        for d in result_dict.keys():
            result_dict[d].append(mp_dicts[k].get()[d])
    return result_dict
