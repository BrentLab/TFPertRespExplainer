import sys
import configparser
import logging.config
from copy import deepcopy

import numpy as np
import pandas as pd
import multiprocess as mp
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score
import xgboost as xgb
import shap

from modeling_utils import *

## Intialize logger
logging.config.fileConfig('logging.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

## Load default configuration
config = configparser.ConfigParser()
config.read('config.ini')
RAND_NUM = int(config['DEFAULT']['rand_num'])
np.random.seed(RAND_NUM)
MAX_RECURSION = int(config['DEFAULT']['max_recursion'])
sys.setrecursionlimit(MAX_RECURSION)
MAX_CV_FOLDS = int(config['DEFAULT']['max_cv_folds'])
BG_GENE_NUM = 500


class TFPRExplainer:
    def __init__(self, tf_feat_mtx_dict, nontf_feat_mtx, features, label_df_dict):
        self.tfs = np.sort(list(label_df_dict.keys()))
        self.genes = label_df_dict[self.tfs[0]].index.values
        self.feats = features
        self.n_tfs = len(self.tfs)
        self.n_genes = len(self.genes)
        self.k_folds = min(MAX_CV_FOLDS, len(self.tfs))
        
        tf_X = np.vstack([tf_feat_mtx_dict[tf] for tf in self.tfs])
        nontf_X = np.vstack([nontf_feat_mtx for i in range(len(self.tfs))])
        self.tg_pairs = [tf + ':' + gene for tf in self.tfs for gene in self.genes]
        self.X = np.hstack([tf_X, nontf_X])
        self.y = np.hstack([label_df_dict[tf].values for tf in self.tfs])

    def cross_validate(self):
        """Cross valdiate a classifier or regressor using multiprocessing.
        """
        with mp.Pool(processes=self.k_folds) as pool:
            mp_results = {}
            tf_pseudo_X = np.empty((len(self.tfs), 0))
            
            kfolds = KFold(n_splits=self.k_folds, shuffle=True, random_state=RAND_NUM)

            for k, (tf_tr_idx, tf_te_idx) in enumerate(kfolds.split(tf_pseudo_X)):
                tr_idx = expand_tf2gene_index(tf_tr_idx, self.n_genes)
                te_idx = expand_tf2gene_index(tf_te_idx, self.n_genes)
                
                y_tr, y_te = self.y[tr_idx], self.y[te_idx]
                X_tr, X_te = self.X[tr_idx], self.X[te_idx]
                X_tr, X_te = standardize_feat_mtx(X_tr, X_te, 'zscore')

                mp_results[k] = pool.apply_async(
                    train_and_predict,
                    args=(k, (X_tr, y_tr), (X_te, y_te), self.tfs[tf_te_idx], self.genes))

            self.cv_results = compile_mp_results(mp_results)

    def explain(self):
        """Use SHAP values to features' contributions to predict the 
        responsiveness of a gene.
        """
        with mp.Pool(processes=self.k_folds) as pool:
            mp_results = {}

            for k, y_te in enumerate(self.cv_results['preds']):
                y_te['tf:gene'] = y_te['tf'] + ':' + y_te['gene']
                te_tg_pairs = y_te['tf:gene'].values
                te_idx = [self.tg_pairs.index(tg_pair) for tg_pair in te_tg_pairs]
                
                tr_idx = sorted(set(range(len(self.tg_pairs))) - set(te_idx))
                logger.info('Explaining {} genes in fold {}'.format(len(te_idx), k))

                X_tr, X_te = self.X[tr_idx], self.X[te_idx]
                X_tr, X_te = standardize_feat_mtx(X_tr, X_te, 'zscore')

                bg_idx = np.random.choice(
                    range(X_tr.shape[0]), BG_GENE_NUM, replace=False)
                mp_results[k] = pool.apply_async(
                    calculate_tree_shap,
                    args=(
                        self.cv_results['models'][k], 
                        X_te, te_tg_pairs, X_tr[bg_idx],))
            
            self.shap_vals = [mp_results[k].get() for k in sorted(mp_results.keys())]

    def save(self, dirpath):
        """Save output data.
        """
        pd.concat(self.cv_results['preds']).to_csv(
            '{}/preds.csv.gz'.format(dirpath), 
            index=False, compression='gzip')

        pd.concat(self.cv_results['stats']).to_csv(
            '{}/stats.csv.gz'.format(dirpath), 
            index=False, compression='gzip')

        np.savetxt(
            '{}/feats.csv.gz'.format(dirpath), np.array(self.feats),
            fmt='%s', delimiter=',')

        np.savetxt(
            '{}/tf_gene_pairs.csv.gz'.format(dirpath), np.array(self.tg_pairs),
            fmt='%s', delimiter=',')
    
        np.savetxt(
            '{}/feat_mtx.csv.gz'.format(dirpath), self.X,
            fmt='%.8f', delimiter=',')

        # TODO
        for k, df in enumerate(self.shap_vals):
            df['cv'] = k
            self.shap_vals[k] = df
        pd.concat(self.shap_vals).to_csv(
            '{}/feat_shap_wbg.csv.gz'.format(dirpath),
            index=False, compression='gzip')


def train_and_predict(k, D_tr, D_te, tfs, genes):
    """Train classifier and predict gene responses. 
    """
    logger.info('Cross validating fold {}'.format(k))

    X_tr, y_tr = D_tr
    X_te, y_te = D_te
       
    ## Train classifier and test
    model = train_classifier(X_tr, y_tr)

    y_pred = pd.DataFrame(
        data=model.predict_proba(X_te), 
        columns=model.classes_)[1].values

    ## Calculate AUC for each TF
    stats_df = pd.DataFrame()
    preds_df = pd.DataFrame()
    n_genes = len(genes)

    for i, tf in enumerate(tfs):
        idx = list(range(i * n_genes, (i + 1) * n_genes))
        auprc = average_precision_score(y_te[idx], y_pred[idx])
        auroc = roc_auc_score(y_te[idx], y_pred[idx])

        preds_df = preds_df.append(pd.DataFrame(
            {'gene': genes, 'tf': [tf] * n_genes, 'label': y_te[idx], 'pred': y_pred[idx]}),
            ignore_index=True)
        stats_df = stats_df.append(pd.DataFrame(
            {'cv': [k], 'tf': [tf], 'auroc': [auroc], 'auprc': [auprc]}),
            ignore_index=True)
    
        logger.info('Cross-validation fold {}, TF {} > AUPRC={:.3f}'.format(k, tf, auprc))

    return {'preds': preds_df, 'stats': stats_df, 'models': model}


def train_classifier(X, y):
    """Train a XGBoost classifier.
    """
    model = xgb.XGBClassifier(
        n_estimators=2500,
        learning_rate=.01,
        booster='gbtree',
        gamma=5,
        colsample_bytree=.8,
        subsample=.8,
        n_jobs=-1,
        random_state=RAND_NUM
    )
    model.fit(X, y)
    return model


def train_regressor(X, y):
    """Train a XGBoost regressor.
    """
    model = xgb.XGBRegressor(
        n_estimators=2500,
        learning_rate=.01,
        objective='reg:squarederror',
        booster='gbtree',
        n_jobs=-1,
        random_state=RAND_NUM
    )
    model.fit(X, y)
    return model


def calculate_tree_shap(model, X, genes, X_bg):
    """Calcualte SHAP values for tree-based model.
    """
    n_genes, n_feats = X.shape
    
    ## Calculate SHAP values
    explainer = shap.TreeExplainer(model, X_bg)
    shap_mtx = explainer.shap_values(X, approximate=False, check_additivity=False)
    
    ## Convert wide to long format
    shap_df = pd.DataFrame(
        data=shap_mtx,
        index=genes,
        columns=['feat' + str(i) for i in range(n_feats)])
    shap_df.index.name = 'gene'
    shap_df = shap_df.reset_index()
    shap_df = pd.wide_to_long(shap_df, 'feat', 'gene', 'feat_idx').reset_index()
    return shap_df


def expand_tf2gene_index(t, n):
    g = []
    for i in t:
        g += list(range(i * n, (i + 1) * n))
    return g
