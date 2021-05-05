import sys
import os.path
import argparse
import configparser
import warnings
import logging.config

from modeling_utils import *
from response_explainer import TFPRExplainer


warnings.filterwarnings("ignore")

## Initialize logger
logging.config.fileConfig('logging.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

## Load default configuration
config = configparser.ConfigParser()
config.read('config.ini')

RAND_NUM = int(config['DEFAULT']['rand_num'])
np.random.seed(RAND_NUM)

PROMOTER_BINS = int(config['YEAST']['promoter_bins'])
PROMOTER_UPSTREAM_BOUND = int(config['YEAST']['promoter_upstream_bound'])
PROMOTER_DOWNSTREAM_BOUND = int(config['YEAST']['promoter_downstream_bound'])
MIN_RESP_LFC = float(config['YEAST']['min_response_lfc'])


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
        'feat_bins': PROMOTER_BINS,
        'feat_length': PROMOTER_UPSTREAM_BOUND + PROMOTER_DOWNSTREAM_BOUND}

    ## Construct input feature matrix and labels
    logger.info('==> Constructing labels and feature matrix <==')
    
    tf_feat_mtx_dict, nontf_feat_mtx, features, label_df_dict = \
        construct_fixed_input(filepath_dict, feat_info_dict)
    label_df_dict = {tf: binarize_label(ldf, MIN_RESP_LFC) for tf, ldf in label_df_dict.items()}
    
    logger.info('Per TF, label dim={}, TF-related feat dim={}, TF-unrelated feat dim={}'.format(
        label_df_dict[feat_info_dict['tfs'][0]].shape, 
        tf_feat_mtx_dict[feat_info_dict['tfs'][0]].shape,
        nontf_feat_mtx.shape))

    # TODO: delete data pickling
    # import pickle
    # if not os.path.exists(filepath_dict['output_dir']):
    #     os.makedirs(filepath_dict['output_dir'])
    # with open(filepath_dict['output_dir'] + '/input_data.pkl', 'wb') as f: 
    #     pickle.dump([tf_feat_mtx_dict, nontf_feat_mtx, features, label_df_dict], f)

    # with open(filepath_dict['output_dir'] + '/input_data.pkl', 'rb') as f: 
    #     tf_feat_mtx_dict, nontf_feat_mtx, features, label_df_dict = pickle.load(f)

    ## Model prediction and explanation
    tfpr_explainer = TFPRExplainer(tf_feat_mtx_dict, nontf_feat_mtx, features, label_df_dict)
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
