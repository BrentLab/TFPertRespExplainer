import sys
import os.path
import argparse
import warnings

import config
from logger import logger
from modeling_utils import *
from response_explainer import TFPRExplainer


warnings.filterwarnings("ignore")

RAND_NUM = config.rand_num
np.random.seed(RAND_NUM)


def parse_args(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-i', '--tfs', required=True, nargs='*',
        help='Perturbed TFs.')
    parser.add_argument(
        '-f', '--feature_types', required=True, nargs='*',
        help='Feature type(s) to be included in feature matrix (delimited by single space).')
    parser.add_argument(
        '-x', '--feature_h5', required=True,
        help='h5 file for input features.')
    parser.add_argument(
        '-y', '--response_label', required=True,
        help='csv file for perturbation response label.')
    parser.add_argument(
        '-o', '--output_dir', required=True,
        help='Output directory path.')
    parsed = parser.parse_args(argv[1:])
    return parsed


def main(argv):
    ## Parse arguments
    args = parse_args(argv)
    logger.info('Input arguments: {}'.format(args))
    filepath_dict = {
        'feat_h5': args.feature_h5,
        'resp_label': args.response_label,
        'output_dir': args.output_dir}
    feat_info_dict = {
        'tfs': args.tfs,
        'feat_types': args.feature_types,
        'promo_bound': (config.human_promoter_upstream_bound, config.human_promoter_downstream_bound),
        'promo_width': config.human_promoter_bin_width,
        'enhan_bound': (config.human_enhancer_upstream_bound, config.human_enhancer_downstream_bound),
        'enhan_min_width': config.human_enhancer_closest_bin_width if config.human_enhancer_bin_type == 'binned' else None}

    ## Construct input feature matrix and labels
    logger.info('==> Constructing labels and feature matrix <==')

    tf_feat_mtx_dict, nontf_feat_mtx, features, label_df_dict = \
        construct_expanded_input(filepath_dict, feat_info_dict)
    label_df_dict = {tf: binarize_label(
        ldf, config.human_min_response_lfc, config.human_max_response_p
        ) for tf, ldf in label_df_dict.items()}

    logger.info('Per TF, label dim={}, TF-related feat dim={}, TF-unrelated feat dim={}'.format(
        label_df_dict[feat_info_dict['tfs'][0]].shape, 
        tf_feat_mtx_dict[feat_info_dict['tfs'][0]].shape,
        nontf_feat_mtx.shape))

    ## Model prediction and explanation
    tfpr_explainer = TFPRExplainer(tf_feat_mtx_dict, nontf_feat_mtx, features, label_df_dict)
    
    if args.model_tuning:
        logger.info('==> Tuning model hyperparameters <==')
        tfpr_explainer.tune_hyparams()

    logger.info('==> Cross validating response prediction model <==')
    tfpr_explainer.cross_validate()

    logger.info('==> Analyzing feature contributions <==')
    tfpr_explainer.explain()
    
    logger.info('==> Saving output data <==')
    if not os.path.exists(filepath_dict['output_dir']):
        os.makedirs(filepath_dict['output_dir'])
    tfpr_explainer.save(filepath_dict['output_dir'])
    
    logger.info('==> Completed <==')


if __name__ == "__main__":
    main(sys.argv)
