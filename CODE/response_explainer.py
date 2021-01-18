import sys
import configparser
import logging.config
from copy import deepcopy

import numpy as np
import pandas as pd
import multiprocess as mp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
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
BG_GENE_NUM = 500


class TFPRExplainer:
    def __init__(self, feat_mtx, features, label_df):
        self.X = feat_mtx
        self.y = label_df
        self.feats = features
        self.genes = label_df.index.values
        self.k_folds = 10

    def cross_validate(self):
        """Cross valdiate a classifier using multiprocessing.
        """
        with mp.Pool(processes=self.k_folds) as pool:
            mp_results = {}
            kfolds = StratifiedKFold(
                n_splits=self.k_folds, shuffle=True, random_state=RAND_NUM)

            for k, (tr_idx, te_idx) in enumerate(kfolds.split(self.X, self.y)):
                y_tr, y_te = self.y[tr_idx], self.y[te_idx]
                X_tr, X_te = self.X[tr_idx], self.X[te_idx]
                X_tr, X_te = standardize_feat_mtx(X_tr, X_te, 'zscore')

                mp_results[k] = pool.apply_async(
                    train_and_predict,
                    args=(k, (X_tr, y_tr), (X_te, y_te),))

            self.cv_results = compile_mp_results(mp_results)

    def explain(self):
        """Use SHAP values to features' contributions to predict the 
        responsiveness of a gene.
        """
        with mp.Pool(processes=self.k_folds) as pool:
            mp_results = {}

            for k, y_te in self.y.groupby('cv'):
                te_idx = [i for i, g in enumerate(self.genes) if g in y_te['gene'].values]
                te_genes = self.genes[te_idx]
                tr_idx = sorted(set(range(len(self.genes))) - set(te_idx))
                logger.debug('Explainer for fold {} uses {} genes'.format(k, len(te_idx)))

                X_tr, X_te = self.X[tr_idx], self.X[te_idx]
                X_tr, X_te = standardize_feat_mtx(X_tr, X_te, 'zscore')

                bg_idx = np.random.choice(
                    range(X_tr.shape[0]), BG_GENE_NUM, replace=False)
                mp_results[k] = pool.apply_async(
                    calculate_tree_shap,
                    args=(
                        self.cv_results['models'][k], 
                        X_te, te_genes, X_tr[bg_idx],))
            
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
            '{}/feats.csv.gz'.format(dirpath), np.array(self.features),
            fmt='%s', delimiter=',')

        np.savetxt(
            '{}/genes.csv.gz'.format(dirpath), self.genes,
            fmt='%s', delimiter=',')
    
        np.savetxt(
            '{}/feat_mtx.csv.gz'.format(dirpath), self.X,
            fmt='%.8f', delimiter=',')
    
        # for k, model in enumerate(self.cv_results['models']):
        #     pickle.dump(model, open('{}/cv{}.pkl'.format(dirpath, k), 'wb'))

        for k, df in enumerate(self.shap_vals):
            df['cv'] = k
            self.shap_vals[k] = df
        pd.concat(self.shap_vals).to_csv(
            '{}/feat_shap_wbg.csv.gz'.format(dirpath),
            index=False, compression='gzip')


def train_and_predict(k, D_tr, D_te):
    """Train classifier and predict gene responses. 
    """
    X_tr, y_tr = D_tr
    X_te, y_te = D_te
    n_te_samples, n_feats = X_te.shape

    ## Train and test model
    model = train_classifier(X_tr, y_tr)

    ## Evaluate model performance
    y_pred = pd.DataFrame(
        data=model.predict_proba(X_te), 
        columns=model.classes_)[1].values
    auprc = average_precision_score(y_te, y_pred)
    auroc = roc_auc_score(y_te, y_pred)
    logger.debug('pr={:.3f}'.format(auprc))

    return {
        'preds': pd.DataFrame({
            'gene': y_te.index.values,
            'cv': [k] * n_te_samples, 
            'label': y_te.values,
            'pred': y_pred}),
        'stats': pd.DataFrame({
            'cv': [k],
            'auroc': [auroc],
            'auprc': [auprc]}),
        'models': model}


def train_classifier(X, y):
    """Train a XGBoost classifier.
    """
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=.01,
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
    logger.debug('SHAP expected value = {}'.format(explainer.expected_value))
    shap_mtx = explainer.shap_values(X, approximate=False, check_additivity=False)
    logger.debug('SHAP matrix shape = {}'.format(shap_mtx.shape))
    
    ## Convert wide to long format
    shap_df = pd.DataFrame(
        data=shap_mtx,
        index=genes,
        columns=['feat' + str(i) for i in range(n_feats)])
    shap_df.index.name = 'gene'
    shap_df = shap_df.reset_index()
    shap_df = pd.wide_to_long(shap_df, 'feat', 'gene', 'feat_idx').reset_index()
    return shap_df
