import sys
import os.path
import argparse
import configparser
import warnings
import logging.config

from modeling_utils import *
from response_explainer import TFPRExplainer


warnings.filterwarnings("ignore")

## Intialize logger
logging.config.fileConfig('logging.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

## Load default configuration
config = configparser.ConfigParser()
config.read('config.ini')

RAND_NUM = int(config['DEFAULT']['rand_num'])
np.random.seed(RAND_NUM)

FEAT_BINS = int(config['YEAST']['feat_bins'])
FEAT_UPSTREAM_BOUND = int(config['YEAST']['feat_upstream_bound'])
FEAT_DOWNSTREAM_BOUND = int(config['YEAST']['feat_downstream_bound'])
MIN_RESP_LFC = float(config['YEAST']['min_response_lfc'])


def parse_args(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-i', '--tf', required=True,
        help='Perturbed TF.')
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
        '--is_regressor', action='store_true',
        help='Classifier (default) or regressor.')
    parser.add_argument(
        '--aux_tfs', nargs='*', default=None,
        help='Auxiliary TFs, whose binding data are included.')
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
        'tf': args.tf,
        'feat_types': args.feature_types,
        'feat_bins': FEAT_BINS,
        'feat_length': FEAT_UPSTREAM_BOUND + FEAT_DOWNSTREAM_BOUND,
        'aux_tfs': args.aux_tfs}
    is_regressor = args.is_regressor
    
    ## Make output directory 
    child_dir = feat_info_dict['tf']
    tf_output_dir = '{}/{}'.format(filepath_dict['output_dir'], child_dir)
    if not os.path.exists(tf_output_dir):
        os.makedirs(tf_output_dir)

    ## Construct input feature matrix and labels
    logger.info('==> Constructing labels and feature matrix <==')
    feat_mtx, features, label_df = construct_fixed_input(filepath_dict, feat_info_dict)
    label_df = label_df.abs() if is_regressor else binarize_label(label_df, MIN_RESP_LFC)
    logger.info('Label dim={}, feat mtx dim={}'.format(label_df.shape, feat_mtx.shape))

    ## Model prediction and explanation
    tfpr_explainer = TFPRExplainer(feat_mtx, features, label_df)
    logger.info('==> Cross validating response prediction model <==')
    tfpr_explainer.cross_validate(is_regressor)

    logger.info('==> Analyzing feature contributions <==')
    tfpr_explainer.explain()
    
    logger.info('==> Saving output data <==')
    tfpr_explainer.save(tf_output_dir)
    
    logger.info('==> Completed <==')


if __name__ == "__main__":
    main(sys.argv)
