import configparser


config = configparser.ConfigParser()
config.read('config.ini')

default_config = config['DEFAULT']
rand_num = int(default_config['rand_num'])
tmp_path = default_config['tmp_path']
max_recursion = int(default_config['max_recursion'])
max_cv_folds = int(default_config['max_cv_folds'])

yeast_config = config['YEAST']
yeast_min_response_lfc = int(yeast_config['min_response_lfc'])
yeast_promoter_upstream_bound = int(yeast_config['promoter_upstream_bound'])
yeast_promoter_downstream_bound = int(yeast_config['promoter_downstream_bound'])
yeast_promoter_bins = int(yeast_config['promoter_bins'])

human_config = config['HUMAN']
human_min_response_lfc = float(human_config['min_response_lfc'])
human_max_response_p = float(human_config['max_response_p'])
human_promoter_upstream_bound = float(human_config['promoter_upstream_bound'])
human_promoter_downstream_bound = float(human_config['promoter_downstream_bound'])
human_enhancer_upstream_bound = float(human_config['enhancer_upstream_bound'])
human_enhancer_downstream_bound = float(human_config['enhancer_downstream_bound'])
human_promoter_bin_width = float(human_config['promoter_bin_width'])
human_enhancer_bin_type = human_config['enhancer_bin_type']
human_enhancer_closest_bin_width = int(human_config['enhancer_closest_bin_width'])
