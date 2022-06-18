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
    parser.add_argument(
        '--model_tuning', action='store_true',
        help='Enable model tuning.')
    parser.add_argument(
        '--model_config', default='MODEL_CONFIG/human_default_config.json',
        help='Json file for pretrained model hyperparameters.')
    parser.add_argument(
        '--permutations', action='store_true',
        help="Enable permutations.")
    parser.add_argument(
        '-N', '--number_of_permutations', type=int, nargs='?', const=5, choices=range(1,100),
        help="Number of permutation runs (default 5).")
    parsed = parser.parse_args(argv[1:])
    return parsed


def run_tfpr(tf_feat_mtx_dict, nontf_feat_mtx, features, label_df_dict, output_dir, model_hyparams, model_tuning, permutations, number_of_permutations):

    last_run_num = 0

    if permutations: 
        logger.info('==> Setting up permutation runs <==')

        logger.info('# checking for existing permutations runs')
        last_run_num = check_last_run(output_dir)

        logger.info('### Scheduling runs: %s-%s ###', last_run_num+1, last_run_num+number_of_permutations)
    else:
        number_of_permutations=1

    for run_num in range(last_run_num+1, last_run_num+number_of_permutations+1):
        if permutations: 
            output_subdir = os.path.join(output_dir, "perm{}".format(run_num))

        ## Model prediction and explanation
        tfpr_explainer = TFPRExplainer(
            tf_feat_mtx_dict, nontf_feat_mtx, features, 
            label_df_dict, output_subdir, model_hyparams
        )

        if model_tuning:
            logger.info('==> Tuning model hyperparameters <==')
            tfpr_explainer.tune_hyparams()

        logger.info('==> Cross validating response prediction model <==')
        tfpr_explainer.cross_validate(permute=permutations)

        if not permutations:
            logger.info('==> Analyzing feature contributions <==')
            tfpr_explainer.explain()

        logger.info('==> Saving output data <==')
        tfpr_explainer.save()

        if permutations:
            logger.info('==> Completed RUN %s <==', run_num)

    logger.info('==> Completed <==')



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
        'promo_bound': (config.human_promoter_upstream_bound, config.human_promoter_downstream_bound),
        'promo_width': config.human_promoter_bin_width,
        'enhan_bound': (config.human_enhancer_upstream_bound, config.human_enhancer_downstream_bound),
        'enhan_min_width': config.human_enhancer_closest_bin_width if config.human_enhancer_bin_type == 'binned' else None
    }
    model_hyparams = load_model_config(args.model_config)

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

    run_tfpr(tf_feat_mtx_dict, nontf_feat_mtx, features, label_df_dict, filepath_dict['output_dir'], model_hyparams, args.model_tuning, args.permutations, args.number_of_permutations)


if __name__ == "__main__":
    main(sys.argv)
