import numpy as np
import pandas as pd
import os.path
from glob import glob
import scipy.stats as ss
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score


COLORS = {
    'orange': '#f0593e', 
    'dark_red': '#7c2712', 
    'red': '#ed1d25',
    'yellow': '#ed9f22', 
    'light_green': '#67bec5', 
    'dark_green': '#018a84',
    'light_blue': '#00abe5', 
    'dark_blue': '#01526e', 
    'grey': '#a8a8a8'
}

DINUCLEOTIDES = {
    'AA': 'AA/TT', 'AC': 'AC/GT', 'AG': 'AG/CT',
    'CA': 'CA/TG', 'CC': 'CC/GG', 'GA': 'GA/TC'
}

FEATURE_NAME_DICT = {
    'yeast': {
        'tf_binding:TF': 'TF binding', 
        'histone_modifications:h3k27ac_tp1_0_merged': 'H3K27ac',
        'histone_modifications:h3k36me3_tp1_0_merged': 'H3K36me3',
        'histone_modifications:h3k4me3_tp1_0_merged': 'H3K4me3',
        'histone_modifications:h3k4me_tp1_0_merged': 'H3K4me1',
        'histone_modifications:h3k79me_tp1_0_merged': 'H3K79me1',
        'histone_modifications:h4k16ac_tp1_0_merged': 'H4K16ac',
        'chromatin_accessibility:BY4741_ypd_osm_0min.occ': 'Chrom acc',
        'gene_expression:TF': 'GEX level', 
        'gene_expression:variation': 'GEX var',
        'dna_sequence:nt_freq_agg': 'Dinucleotides'
    },
    'human_k562': {
        'tf_binding:TF': 'TF binding', 
        'histone_modifications:K562_H3K27ac': 'H3K27ac',
        'histone_modifications:K562_H3K27me3': 'H3K27me3',
        'histone_modifications:K562_H3K36me3': 'H3K36me3',
        'histone_modifications:K562_H3K4me1': 'H3K4me1',
        'histone_modifications:K562_H3K4me3': 'H3K4me3',
        'histone_modifications:K562_H3K9me3': 'H3K9me3',
        'chromatin_accessibility:K562_atac': 'Chrom acc',
        'gene_expression:median_level': 'GEX level', 
        'gene_expression:variation': 'GEX var',
        'dna_sequence:nt_freq_agg': 'DNA sequence'
    },
    'human_hek293': {
        'tf_binding:TF': 'TF binding', 
        'histone_modifications:HEK293_H3K27ac': 'H3K27ac',
        'histone_modifications:HEK293_H3K27me3': 'H3K27me3',
        'histone_modifications:HEK293_H3K36me3': 'H3K36me3',
        'histone_modifications:HEK293_H3K4me1': 'H3K4me1',
        'histone_modifications:HEK293_H3K4me3': 'H3K4me3',
        'histone_modifications:HEK293_H3K9me3': 'H3K9me3',
        'chromatin_accessibility:HEK293T_dnase': 'Chrom acc',
        'gene_expression:median_level': 'GEX level', 
        'gene_expression:variation': 'GEX var',
        'dna_sequence:nt_freq_agg': 'DNA sequence'
    },
    'human_h1': {
        'tf_binding:TF': 'TF binding', 
        'histone_modifications:H3K27ac': 'H3K27ac',
        'histone_modifications:H3K27me3': 'H3K27me3',
        'histone_modifications:H3K36me3': 'H3K36me3',
        'histone_modifications:H3K4me1': 'H3K4me1',
        'histone_modifications:H3K4me3': 'H3K4me3',
        'histone_modifications:H3K9me3': 'H3K9me3',
        'chromatin_accessibility:H1_ChromAcc_intersect': 'Chrom acc',
        'gene_expression:median_level': 'GEX level', 
        'gene_expression:variation': 'GEX var',
        'dna_sequence:nt_freq_agg': 'DNA sequence'
    }
}


def parse_classifier_stats(dirpath, algorithm, feat_types, sys2com_dict=None):
    out_df = pd.DataFrame(
        columns=['tf', 'chance', 'feat_type', 'cv', 'auroc', 'auprc'])
    
    for feat_type in feat_types:
        print('... working on', feat_type)
        subdirs = glob('{}/{}/{}/*'.format(dirpath, feat_type, algorithm))
        
        for subdir in subdirs:
            tf = os.path.basename(subdir)
            filename = glob('{}/stats.csv*'.format(subdir))[0]
            stats_df = pd.read_csv(filename)
            filename = glob('{}/preds.csv*'.format(subdir))[0]
            preds_df = pd.read_csv(filename)
            stats_df['feat_type'] = feat_type
            
            if sys2com_dict is not None:
                tf_com = sys2com_dict[tf] if tf in sys2com_dict else tf
                stats_df['tf'] = '{} ({})'.format(tf, tf_com)
                stats_df['tf_com'] = tf_com
            else:
                stats_df['tf'] = tf
            
            stats_df['chance'] = np.sum(preds_df['label'] == 1) / preds_df.shape[0]
            out_df = out_df.append(stats_df, ignore_index=True)
    return out_df


def compare_model_stats(df, metric, comp_groups):
    stats_df = pd.DataFrame(columns=['tf', 'comp_group', 'p_score'])
    
    for tf, df2 in df.groupby('tf'):
        for (f1, f2) in comp_groups:
            x1 = df2.loc[df2['feat_type'] == f1, metric]
            x2 = df2.loc[df2['feat_type'] == f2, metric]
            _, p = ss.ttest_rel(x1, x2)
            sign = '+' if np.median(x2) > np.median(x1) else '-'
           
            stats_row = pd.Series({
                'tf': tf, 
                'comp_group': '{} vs {}'.format(f1, f2), 
                'p_score': -np.log10(p),
                'sign': sign})
            stats_df = stats_df.append(stats_row, ignore_index=True)
    return stats_df


def get_feature_indices(df, organism):
    """Parse feature indices for visualization.
    """
    feat_dict = FEATURE_NAME_DICT[organism]

    idx_df = pd.DataFrame()
    for _, row in df.iterrows():
        if row['feat_type'] == 'dna_sequence_nt_freq':
            type_name = 'dna_sequence:nt_freq_agg'
        else:
            type_name = row['feat_type'] + ':' + row['feat_name']
        type_name2 = feat_dict[type_name]
        for i in range(row['start'], row['end']):
            idx_df = idx_df.append(pd.Series({'feat_type_name': type_name2, 'feat_idx': i}), ignore_index=True)
    return idx_df


def calculate_resp_and_unresp_signed_shap_sum(data_dir, tfs=None, organism='yeast', sum_over_type='tf'):
    """Calculate the sum of SHAP values within responsive and unresponsive genes respectively.
    """
    print('Loading feature data ...', end=' ')
    shap_subdf_list = []
    for i, shap_subdf in enumerate(pd.read_csv('{}/feat_shap_wbg.csv.gz'.format(data_dir), chunksize=10 ** 7, low_memory=False)):
        print(i, end=' ')
        shap_subdf = shap_subdf.rename(columns={'gene': 'tf:gene', 'feat': 'shap'})
        shap_subdf['tf'] = shap_subdf['tf:gene'].apply(lambda x: x.split(':')[0])
        if tfs is not None:
            shap_subdf = shap_subdf[shap_subdf['tf'].isin(tfs)]
        shap_subdf_list.append(shap_subdf)
    print()
    shap_df = pd.concat(shap_subdf_list)
    del shap_subdf_list
    
    feats_df = pd.read_csv('{}/feats.csv.gz'.format(data_dir), names=['feat_type', 'feat_name', 'start', 'end'])

    preds_df = pd.read_csv('{}/preds.csv.gz'.format(data_dir))
    if tfs is not None:
        preds_df = preds_df[preds_df['tf'].isin(tfs)]

    feat_idx_df = get_feature_indices(feats_df, organism)
    
    ## Parse out shap+ and shap- values
    print('Parsing signed shap values ...')
    shap_df = shap_df.merge(preds_df[['tf:gene', 'label', 'gene']], how='left', on='tf:gene')
    shap_df['shap+'] = shap_df['shap'].clip(lower=0)
    shap_df['shap-'] = shap_df['shap'].clip(upper=0)

    ## Sum across reg region for each feature and each tf:gene, and then take 
    ## the mean among responsive targets and repeat for non-responsive targets.
    print('Summing shap ...')
    shap_df = shap_df.merge(feat_idx_df[['feat_type_name', 'feat_idx']], on='feat_idx')
    sum_shap = shap_df.groupby(['tf', 'gene', 'label', 'feat_type_name'])[['shap+', 'shap-']].sum().reset_index()
    sum_shap = sum_shap.groupby([sum_over_type, 'label', 'feat_type_name'])[['shap+', 'shap-']].mean().reset_index()
    sum_shap['label_name'] = sum_shap['label'].apply(lambda x: 'Responsive' if x == 1 else 'Non-responsive')
    sum_shap['label_name'] = pd.Categorical(
        sum_shap['label_name'], ordered=True, categories=['Responsive', 'Non-responsive'])

    sum_shap_pos = sum_shap[[sum_over_type, 'label_name', 'feat_type_name', 'shap+']].copy().rename(columns={'shap+': 'shap'})
    sum_shap_pos['shap_dir'] = 'SHAP > 0'
    sum_shap_neg = sum_shap[[sum_over_type, 'label_name', 'feat_type_name', 'shap-']].copy().rename(columns={'shap-': 'shap'})
    sum_shap_neg['shap_dir'] = 'SHAP < 0'
    
    sum_signed_shap = pd.concat([sum_shap_pos, sum_shap_neg])
    sum_signed_shap['shap_dir'] = pd.Categorical(sum_signed_shap['shap_dir'], categories=['SHAP > 0', 'SHAP < 0'])
    return sum_signed_shap


def get_best_yeast_model(data_dirs, tf_name):
    tf1_dir = '{}/{}'.format(data_dirs[0], tf_name)
    tf2_dir = '{}/{}'.format(data_dirs[1], tf_name)
    if (not os.path.exists(tf1_dir)) and (not os.path.exists(tf2_dir)):
        return None
    elif (os.path.exists(tf1_dir)) and (os.path.exists(tf2_dir)):
        acc1 = pd.read_csv('{}/stats.csv.gz'.format(tf1_dir))['auprc'].median()
        acc2 = pd.read_csv('{}/stats.csv.gz'.format(tf2_dir))['auprc'].median()
        tf_dir = tf1_dir if acc1 >= acc2 else tf2_dir
        is_cc = True if acc1 >= acc2 else False
        acc = max(acc1, acc2)
    else:
        tf_dir = tf1_dir if os.path.exists(tf1_dir) else tf2_dir
        is_cc = True if os.path.exists(tf1_dir) else False
        acc = pd.read_csv('{}/stats.csv.gz'.format(tf_dir))['auprc'].median()
    return (tf_dir, is_cc, acc)
    

def calculate_shap_net_influence(df, sum_over_type='tf'):
    """Calculate net influence of each feature type from SHAP values.
    """
    df2 = pd.DataFrame()
    if sum_over_type == 'tf':
        for (label_name, feat_type_name, tf), subdf in df.groupby(['label_name', 'feat_type_name', 'tf']):
            shap_diff = subdf.loc[subdf['shap_dir'] == 'SHAP > 0', 'shap'].iloc[0] - \
                        np.abs(subdf.loc[subdf['shap_dir'] == 'SHAP < 0', 'shap'].iloc[0])
            df2 = df2.append(pd.Series({
                'label_name': label_name, 'feat_type_name': feat_type_name,
                'tf': tf, 'shap_diff': shap_diff
                }), ignore_index=True)
    elif sum_over_type == 'gene':
        for (label_name, feat_type_name, gene), subdf in df.groupby(['label_name', 'feat_type_name', 'gene']):
            shap_diff = subdf.loc[subdf['shap_dir'] == 'SHAP > 0', 'shap'].iloc[0] - \
                        np.abs(subdf.loc[subdf['shap_dir'] == 'SHAP < 0', 'shap'].iloc[0])
            df2 = df2.append(pd.Series({
                'label_name': label_name, 'feat_type_name': feat_type_name,
                'gene': gene, 'shap_diff': shap_diff
                }), ignore_index=True)
    return df2
    

def parse_gene_shap_mtx(data_dir, tf, gene, width=15, use_abs=False, agg_dna=True):
    """Parse SHAP matrix for a list of target genes.
    """
    shap_subdf_list = []
    for i, shap_subdf in enumerate(pd.read_csv('{}/feat_shap_wbg.csv.gz'.format(data_dir), chunksize=10 ** 7, low_memory=False)):
        shap_subdf = shap_subdf.rename(columns={'gene': 'tf:gene', 'feat': 'shap'})
        shap_subdf['tf'] = shap_subdf['tf:gene'].apply(lambda x: x.split(':')[0])
        shap_subdf['gene'] = shap_subdf['tf:gene'].apply(lambda x: x.split(':')[1])
        shap_subdf = shap_subdf[(shap_subdf['tf'] == tf) & (shap_subdf['gene'] == gene)]
        shap_subdf_list.append(shap_subdf)
        
    target_df = pd.concat(shap_subdf_list)
    
    feats_df = pd.read_csv(data_dir + '/feats.csv.gz', names=['feat_type', 'feat_name', 'start', 'end'])

    ## Add feature info
    for _, row in feats_df.iterrows():
        f_type, f_name, start, end = row
        target_df.loc[(target_df['feat_idx'] >= start) & (target_df['feat_idx'] < end), 'feat_type'] = f_type
        target_df.loc[(target_df['feat_idx'] >= start) & (target_df['feat_idx'] < end), 'feat_name'] = f_name
    
    ## Aggregate DNA sequence and drop nt freq rows
    if agg_dna:
        nt_freq_agg = target_df.loc[target_df['feat_type'] == 'dna_sequence_nt_freq', 'shap']
        target_df = target_df.append(pd.Series({
            'gene': gene,
            'feat_idx': min(target_df.loc[target_df['feat_type'] == 'dna_sequence_nt_freq', 'feat_idx']),
            'feat_type': 'dna_sequence',
            'feat_name': 'nt_freq_agg',
            'shap': sum(np.abs(nt_freq_agg)) if use_abs else sum(nt_freq_agg)
            }),
            ignore_index=True)
        target_df = target_df[target_df['feat_type'] != 'dna_sequence_nt_freq']

    ## Add feature position in visualization
    start_idx = target_df.groupby(['feat_type', 'feat_name'])['feat_idx'].min().reset_index()
    
    for i, row in target_df.iterrows():
        diff_idx = start_idx.loc[(start_idx['feat_type'] == row['feat_type']) & (start_idx['feat_name'] == row['feat_name']), 'feat_idx'].values[0]
        target_df.loc[i, 'pos'] = row['feat_idx'] - diff_idx

    target_df['feat_type_name'] = target_df['feat_type'] + ':' + target_df['feat_name']
    if use_abs:
        target_df['shap'] = np.abs(target_df['shap'])
    return target_df


def annotate_resp_type(resp_df, feat_mtx, feats_df, tf, three_resp_dir=True, bound_targets=None):
    """Annotate the responsiveness type of each gene.
    """
    tfb_idx = feats_df.loc[(feats_df['feat_type'] == 'tf_binding'), ['start', 'end']].iloc[0]
    tfb_sum = feat_mtx[range(tfb_idx['start'], tfb_idx['end'])].sum(axis=1).to_frame().rename(columns={0: 'tfb'})
    
    resp_df = resp_df[tf].to_frame().rename(columns={tf: 'resp'})
    resp_df = resp_df.merge(tfb_sum, left_index=True, right_on='gene')
    
    resp_df['is_bound'] = 'Unbound'
    if bound_targets is None:
        resp_df.loc[resp_df['tfb'] > 0, 'is_bound'] = 'Bound'
    else:
        resp_df.loc[bound_targets, 'is_bound'] = 'Bound'
    
    if three_resp_dir:
        resp_df['resp_dir'] = 'Non-responsive'
        resp_df.loc[resp_df['resp'] > 0, 'resp_dir'] = 'Activated'
        resp_df.loc[resp_df['resp'] < 0, 'resp_dir'] = 'Repressed'
    else:
        resp_df['resp_dir'] = 'Non-responsive'
        resp_df.loc[resp_df['resp'] != 0, 'resp_dir'] = 'Responsive'
    return resp_df[['is_bound', 'resp_dir']]


def link_shap_to_coord_feats(feat_type, tfs, data_dir, resp_filepath, **kwargs):
    """Link SHAP values to features at coordinate resolution.
    """
    is_tf_dependent = kwargs.get('is_tf_dependent', False)
    feat_name = kwargs.get('feat_name', 'TF')
    coord_offset = kwargs.get('coord_offset', None)
    bin_width = kwargs.get('bin_width', None)
    cc_dir = kwargs.get('cc_dir', None)
    is_resp_format_long = kwargs.get('is_resp_format_long', False)
    
#     shap_df = pd.read_csv('{}/feat_shap_wbg.csv.gz'.format(data_dir))
#     shap_df = shap_df.rename(columns={'gene': 'tf:gene', 'feat': 'shap'})
    print('Loading feature data ...', end=' ')
    shap_subdf_list = []
    for i, shap_subdf in enumerate(pd.read_csv('{}/feat_shap_wbg.csv.gz'.format(data_dir), chunksize=10 ** 7, low_memory=False)):
        print(i, end=' ')
        shap_subdf = shap_subdf.rename(columns={'gene': 'tf:gene', 'feat': 'shap'})
        shap_subdf['tf'] = shap_subdf['tf:gene'].apply(lambda x: x.split(':')[0])
        if tfs is not None:
            shap_subdf = shap_subdf[shap_subdf['tf'].isin(tfs)]
        shap_subdf_list.append(shap_subdf[['tf:gene', 'feat_idx', 'shap']])
    print()
    shap_df = pd.concat(shap_subdf_list)
    del shap_subdf_list
    
    feats_df = pd.read_csv('{}/feats.csv.gz'.format(data_dir), names=['feat_type', 'feat_name', 'start', 'end'])
    
    preds_df = pd.read_csv('{}/preds.csv.gz'.format(data_dir))
    preds_df = preds_df[preds_df['tf'].isin(tfs)]
    
    tf_gene_pairs = np.loadtxt('{}/tf_gene_pairs.csv.gz'.format(data_dir), dtype=str)
    feat_mtx_tfs = set([x.split(':')[0] for x in tf_gene_pairs])
    if is_tf_dependent:
        feat_mtx = np.loadtxt('{}/feat_mtx_tf.csv.gz'.format(data_dir), delimiter=',')
    else:
        feat_mtx_0 = np.loadtxt('{}/feat_mtx_nontf.csv.gz'.format(data_dir), delimiter=',')
        feat_mtx = np.concatenate([feat_mtx_0] * len(feat_mtx_tfs))
    feat_mtx = pd.DataFrame(data=feat_mtx, index=tf_gene_pairs)
    feat_mtx = feat_mtx.reset_index().rename(columns={'index': 'tf:gene'})
    feat_mtx['tf'] = feat_mtx['tf:gene'].apply(lambda x: x.split(':')[0])
    feat_mtx['gene'] = feat_mtx['tf:gene'].apply(lambda x: x.split(':')[1])
    
    if is_resp_format_long:
        resp_df = pd.read_csv(resp_filepath, index_col=None)
        resp_df['has_sig_lfc'] = (resp_df['log2FoldChange'].abs() > 0.5).astype(int)
        resp_df['has_sig_p'] = (resp_df['padj'] < 0.05).astype(int)
        resp_df['is_resp'] = resp_df['has_sig_lfc'] * resp_df['has_sig_p']
        resp_df = resp_df.pivot(index='gene_ensg', columns='tf_ensg', values='is_resp')
    else:
        resp_df = pd.read_csv(resp_filepath, index_col='GeneName')
    resp_df.index.name = 'gene'
    
    annot_genes_df = pd.DataFrame()
    
    for tf in tfs:
        if cc_dir is not None:
            tfb_sig = pd.read_csv('{}/{}.sig_prom.txt'.format(cc_dir, tf), sep='\t')
            tfb_targets = tfb_sig.loc[tfb_sig['Poisson pvalue'] < 10**-3, 'Systematic Name']
            tfb_targets = np.intersect1d(tfb_targets, resp_df.index)
        else:
            tfb_targets = None

        annot_subdf = annotate_resp_type(
            resp_df, feat_mtx[feat_mtx['tf'] == tf].set_index('gene'), feats_df, tf, 
            three_resp_dir=False, bound_targets=tfb_targets
        )
        annot_subdf = annot_subdf.reset_index().rename(columns={'index': 'gene'})
        annot_subdf['tf'] = tf
        annot_subdf['tf:gene'] = annot_subdf['tf'] + ':' + annot_subdf['gene']
        annot_genes_df = annot_genes_df.append(annot_subdf)

    feat_row = feats_df[(feats_df['feat_type'] == feat_type) & (feats_df['feat_name'] == feat_name)].iloc[0]
    feat_mtx_idx_offset = 0 if is_tf_dependent else feats_df.loc[feats_df['feat_name'] == 'TF', 'end'].max()
    
    comb_df_list = []
    
    for feat_idx in range(feat_row['start'], feat_row['end']):
        comb_df = preds_df[['label', 'pred', 'tf:gene']].merge(
            shap_df.loc[shap_df['feat_idx'] == feat_idx], how='left', on='tf:gene'
        )
        comb_df = comb_df.merge(
            feat_mtx[['tf:gene', feat_idx - feat_mtx_idx_offset]], on='tf:gene'
        )
        comb_df = comb_df.merge(
            annot_genes_df, how='left', on='tf:gene'
        )

        comb_df = comb_df.rename(columns={feat_idx - feat_mtx_idx_offset: 'input'})
        comb_df['label'] = comb_df['label'].astype(int).astype(str)
        if bin_width is not None and coord_offset is not None:
            comb_df['coord'] = (feat_idx - feat_row['start']) * bin_width - coord_offset
        comb_df_list.append(comb_df)

    return pd.concat(comb_df_list)

    
def resp_ratio(x):
    return sum(x == 1) / len(x)


def sigmoid(x, a ,b, k, c):
    return a / (1 + np.exp(-k * (x - b))) + c
