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
    'grey': '#a8a8a8'}

DINUCLEOTIDES = {
    'AA': 'AA/TT', 'AC': 'AC/GT', 'AG': 'AG/CT',
    'CA': 'CA/TG', 'CC': 'CC/GG', 'GA': 'GA/TC'
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
    if organism == 'yeast':
        feat_dict = {
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
            'dna_sequence:nt_freq_agg': 'Dinucleotides'}
    elif organism == 'human_k562':
        feat_dict = {
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
            'dna_sequence:nt_freq_agg': 'DNA sequence'}
    elif organism == 'human_hek293':
        feat_dict = {
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
            'dna_sequence:nt_freq_agg': 'DNA sequence'}
    elif organism == 'human_h1':
        feat_dict = {
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
            'dna_sequence:nt_freq_agg': 'DNA sequence'}

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
    # TODO: update shap csv header
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
                'tf': tf, 'auprc': subdf['auprc'].iloc[0], 'shap_diff': shap_diff
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

    
def resp_ratio(x):
    return sum(x == 1) / len(x)