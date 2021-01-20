import numpy as np
import pandas as pd
from os.path import basename
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
            tf = basename(subdir)
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
