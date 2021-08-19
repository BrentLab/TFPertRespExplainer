import sys
from copy import deepcopy
import numpy as np
import pandas as pd
import multiprocess as mp
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score
import xgboost as xgb
import shap
from sklearn.model_selection import cross_val_predict
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import config
from logger import logger
from modeling_utils import *


RAND_NUM = config.rand_num
np.random.seed(RAND_NUM)

MAX_RECURSION = config.max_recursion
sys.setrecursionlimit(MAX_RECURSION)

MAX_CV_FOLDS = config.max_cv_folds
BG_GENE_NUM = 1000
MAX_TUNING_ITRS = 2


class TFPRExplainer:
    def __init__(self, tf_feat_mtx_dict, nontf_feat_mtx, features, label_df_dict):
        self.tfs = np.sort(list(label_df_dict.keys()))
        self.genes = label_df_dict[self.tfs[0]].index.values
        self.feats = features
        self.n_genes = len(self.genes)
        self.tf_X_dict = tf_feat_mtx_dict
        self.nontf_X = nontf_feat_mtx
        self.y_dict = label_df_dict
        
        self.model_hyparams = {
            'n_estimators': 500,  #TODO: update to 2500 trees
            'learning_rate': .01,
            'gamma': 5,
            'colsample_bytree': .8,
            'subsample': .8,
        }

    def data_setup(self):
        self.k_folds = min(MAX_CV_FOLDS, len(self.tfs))
        self.tg_pairs = [tf + ':' + gene for tf in self.tfs for gene in self.genes]
        self.tf_X = np.vstack([self.tf_X_dict[tf] for tf in self.tfs])
        self.y = np.hstack([self.y_dict[tf].values for tf in self.tfs])

    def cross_validate(self):
        """Cross valdiate a classifier or regressor using multiprocessing.
        """
        self.data_setup()

        print(len(self.tfs), self.tfs)
        print(self.tf_X.shape, self.nontf_X.shape, self.y.shape)
        sys.exit()

        with mp.Pool(processes=self.k_folds) as pool:
            mp_results = {}
            tf_pseudo_X = np.empty((len(self.tfs), 0))
            
            kfolds = KFold(n_splits=self.k_folds, shuffle=True, random_state=RAND_NUM)

            for k, (tf_tr_idx, tf_te_idx) in enumerate(kfolds.split(tf_pseudo_X)):
                tr_idx = expand_tf2gene_index(tf_tr_idx, self.n_genes)
                te_idx = expand_tf2gene_index(tf_te_idx, self.n_genes)
                
                y_tr, y_te = self.y[tr_idx], self.y[te_idx]
                tf_X_tr, tf_X_te = self.tf_X[tr_idx], self.tf_X[te_idx]

                tf_X_tr, tf_X_te = standardize_feat_mtx(tf_X_tr, tf_X_te, 'zscore')
                nontf_X, _ = standardize_feat_mtx(self.nontf_X, None, 'zscore')

                mp_results[k] = pool.apply_async(
                    train_and_predict,
                    args=(
                        k, 
                        (tf_X_tr, y_tr), 
                        (tf_X_te, y_te),
                        nontf_X, 
                        (self.tfs[tf_tr_idx], self.tfs[tf_te_idx]), 
                        self.genes,
                        self.model_hyparams))

            self.cv_results = compile_mp_results(mp_results)

    def tune_hyparams(self):
        """Tune hyperparameters of a classifier or regressor using a subset of 
        """
        def model_objective(params):
            model = xgb.XGBClassifier(
                n_estimators=self.model_hyparams['n_estimators'],
                booster='gbtree',
                scale_pos_weight=1,
                n_jobs=-1,
                **params)
            y_pred = cross_val_predict(model, tuning_X, tuning_y, cv=5, n_jobs=-1)
            loss = -average_precision_score(tuning_y, y_pred)  ## AuPRC as loss
            return {'loss': loss, 'status': STATUS_OK}

        ## Randomly split a fraction of tfs for tuning and rest for cv
        n_tuning_tfs = int(np.ceil(len(self.tfs) * .1))
        tuning_tf_idx = np.random.choice(range(len(self.tfs)), size=n_tuning_tfs, replace=False)
        tuning_tfs = self.tfs[tuning_tf_idx]
        logger.info('TFs for model tuning: {}'.format(tuning_tfs))

        sys.exit()

        tuning_X = np.hstack([
            np.vstack([self.tf_X_dict[tf] for tf in tuning_tfs]),
            np.vstack([self.nontf_X for i in range(len(tuning_tfs))])
        ])
        tuning_y = np.hstack([self.y_dict[tf].values for tf in tuning_tfs])

        ## Define search space
        hyparam_choice = {
            # 'learning_rate': hp.loguniform('learning_rate', .001, .1),
            'learning_rate': hp.uniform('learning_rate', .005, .02),
            'gamma': hp.uniform('gamma', 0, 10),
            'colsample_bytree': hp.uniform('colsample_bytree', .5, 1.),
            'subsample': hp.uniform('subsample', .5, 1.)
        }
        search_space = hp.choice('model', [hyparam_choice])

        ## Optimize hyparams
        trials = Trials()
        best_hps = fmin(
            model_objective, search_space, 
            algo=tpe.suggest, max_evals=MAX_TUNING_ITRS, trials=trials,
            rstate=np.random.RandomState(RAND_NUM)
        )
        logger.info('Best model hyperparameters: {}'.format(best_hps))

        self.model_hyparams.update(best_hps)
        self.tfs = np.sort(np.setdiff1d(self.tfs, tuning_tfs))

    def explain(self):
        """Use SHAP values to features' contributions to predict the 
        responsiveness of a gene.
        """
        with mp.Pool(processes=self.k_folds) as pool:
            mp_results = {}

            for k, y_te in enumerate(self.cv_results['preds']):
                n_tfs_te = len(y_te['tf'].unique())
                n_tfs_tr = len(self.tfs) - n_tfs_te

                y_te['tf:gene'] = y_te['tf'] + ':' + y_te['gene']
                te_tg_pairs = y_te['tf:gene'].values
                te_idx = [self.tg_pairs.index(tg_pair) for tg_pair in te_tg_pairs]
                
                tr_idx = sorted(set(range(len(self.tg_pairs))) - set(te_idx))
                logger.info('Explaining {} genes in fold {}'.format(len(te_idx), k))

                tf_X_tr, tf_X_te = self.tf_X[tr_idx], self.tf_X[te_idx]

                tf_X_tr, tf_X_te = standardize_feat_mtx(tf_X_tr, tf_X_te, 'zscore')
                nontf_X, _ = standardize_feat_mtx(self.nontf_X, None, 'zscore')

                X_tr = np.hstack([tf_X_tr, np.vstack([nontf_X for i in range(n_tfs_tr)])])
                X_te = np.hstack([tf_X_te, np.vstack([nontf_X for i in range(n_tfs_te)])])

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
            '{}/feat_mtx_tf.csv.gz'.format(dirpath), self.tf_X,
            fmt='%.8f', delimiter=',')
        np.savetxt(
            '{}/feat_mtx_nontf.csv.gz'.format(dirpath), self.nontf_X,
            fmt='%.8f', delimiter=',')

        # TODO
        for k, df in enumerate(self.shap_vals):
            df['cv'] = k
            self.shap_vals[k] = df
        pd.concat(self.shap_vals).to_csv(
            '{}/feat_shap_wbg.csv.gz'.format(dirpath),
            index=False, compression='gzip')


def train_and_predict(k, D_tr, D_te, nontf_X, tfs, genes, model_hyparams):
    """Train classifier and predict gene responses. 
    """
    logger.info('Cross validating fold {}'.format(k))

    tf_X_tr, y_tr = D_tr
    tf_X_te, y_te = D_te
    tfs_tr, tfs_te = tfs

    X_tr = np.hstack([tf_X_tr, np.vstack([nontf_X for i in range(len(tfs_tr))])])
    X_te = np.hstack([tf_X_te, np.vstack([nontf_X for i in range(len(tfs_te))])])

    ## Train classifier and test
    model = train_classifier(X_tr, y_tr, model_hyparams)

    y_pred = pd.DataFrame(
        data=model.predict_proba(X_te), 
        columns=model.classes_)[1].values

    ## Calculate AUC for each TF
    stats_df = pd.DataFrame()
    preds_df = pd.DataFrame()
    n_genes = len(genes)

    for i, tf in enumerate(tfs_te):
        idx = list(range(i * n_genes, (i + 1) * n_genes))
        auprc = average_precision_score(y_te[idx], y_pred[idx])
        auroc = roc_auc_score(y_te[idx], y_pred[idx])

        preds_df = preds_df.append(pd.DataFrame(
            {'gene': genes, 'tf': [tf] * n_genes, 'label': y_te[idx], 'pred': y_pred[idx]}),
            ignore_index=True)
        stats_df = stats_df.append(pd.DataFrame(
            {'cv': [k], 'tf': [tf], 'auroc': [auroc], 'auprc': [auprc]}),
            ignore_index=True)
    
        logger.info('CV performance for TF {} in fold {}: AUPRC={:.3f}'.format(tf, k, auprc))

    return {'preds': preds_df, 'stats': stats_df, 'models': model}


def train_classifier(X, y, model_hyparams):
    """Train a XGBoost classifier.
    """
    # model = xgb.XGBClassifier(
    #     n_estimators=2500,
    #     learning_rate=.01,
    #     booster='gbtree',
    #     gamma=5,
    #     colsample_bytree=.8,
    #     subsample=.8,
    #     n_jobs=-1,
    #     random_state=RAND_NUM
    # )
    # model.fit(X, y)
    # return model

    # Tuning Notes:
    # - Resonable class weight helps (not as high as neg/pos ratio)
    # - Gamma helps as regularizer
    # - Number of features per tree helps  
    # - Subsample helps

    # neg_pos_ratio = sum(y == 0) / sum(y == 1)
    # neg_pos_ratio = 5
    neg_pos_ratio = 1

    model = xgb.XGBClassifier(
        n_estimators=model_hyparams['n_estimators'],
        booster='gbtree',
        scale_pos_weight=neg_pos_ratio,
        n_jobs=-1,
        random_state=RAND_NUM,
        learning_rate=model_hyparams['learning_rate'],
        gamma=model_hyparams['gamma'],
        colsample_bytree=model_hyparams['colsample_bytree'],
        subsample=model_hyparams['subsample']
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
