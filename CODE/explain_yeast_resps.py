import sys
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
    parser.add_argument(
        '--model_tuning', action='store_true',
        help='Enable model turning.')
    parser.add_argument(
        '--model_config', default='MODEL_CONFIG/yeast_default_config.json',
        help='Json file for model hyperparameters.')
    parsed = parser.parse_args(argv[1:])
    return parsed


def main(argv):
    ## Parse arguments
    args = parse_args(argv)
    logger.info('Input arguments: {}'.format(args))
    filepath_dict = {
        'feat_h5': args.feature_h5,
        'resp_label': args.response_label,
        'output_dir': args.output_dir
    }
    feat_info_dict = {
        'tfs': args.tfs,
        'feat_types': args.feature_types,
        'feat_bins': config.yeast_promoter_bins,
        'feat_length': config.yeast_promoter_upstream_bound + config.yeast_promoter_downstream_bound
    }
    model_hyparams = load_json(args.model_config)

    ## Construct input feature matrix and labels
    logger.info('==> Constructing labels and feature matrix <==')
    
    tf_feat_mtx_dict, nontf_feat_mtx, features, label_df_dict = \
        construct_fixed_input(filepath_dict, feat_info_dict)
    label_df_dict = {tf: binarize_label(
        ldf, config.yeast_min_response_lfc
    ) for tf, ldf in label_df_dict.items()}
    
    logger.info('Per TF, label dim={}, TF-related feat dim={}, TF-unrelated feat dim={}'.format(
        label_df_dict[feat_info_dict['tfs'][0]].shape, 
        tf_feat_mtx_dict[feat_info_dict['tfs'][0]].shape,
        nontf_feat_mtx.shape))

    # # TODO: delete data pickling
    # import pickle

    # with open(filepath_dict['output_dir'] + '/input_data.pkl', 'wb') as f: 
    #     pickle.dump([tf_feat_mtx_dict, nontf_feat_mtx, features, label_df_dict], f)

    # with open(filepath_dict['output_dir'] + '/input_data.pkl', 'rb') as f: 
    #     tf_feat_mtx_dict, nontf_feat_mtx, features, label_df_dict = pickle.load(f)
    # # END OF TODO

    ## Model prediction and explanation
    tfpr_explainer = TFPRExplainer(
        tf_feat_mtx_dict, nontf_feat_mtx, features, 
        label_df_dict, filepath_dict['output_dir'], model_hyparams
    )
    
    if args.model_tuning:
        logger.info('==> Tuning model hyperparameters <==')
        tfpr_explainer.tune_hyparams()

    logger.info('==> Cross validating response prediction model <==')
    tfpr_explainer.cross_validate()

    logger.info('==> Analyzing feature contributions <==')
    tfpr_explainer.explain()
    
    logger.info('==> Saving output data <==')
    tfpr_explainer.save()
    
    logger.info('==> Completed <==')


if __name__ == "__main__":
    main(sys.argv)
