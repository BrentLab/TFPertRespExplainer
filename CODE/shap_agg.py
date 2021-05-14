import pandas as pd
import numpy as np

from visualization_utils import *

import warnings
warnings.filterwarnings('ignore')


organism = 'human'
k562_dir = 'OUTPUT/human_42tfs_k562.10_cv_folds/'
k562_tfs = np.loadtxt('../Pert_Response_Modeling/JOB_SCRIPTS/Human_ENCODE_K562_TFs.txt', dtype=str)[:, 0]

k562_sss_df = calculate_resp_and_unresp_signed_shap_sum(k562_dir, k562_tfs, organism)
k562_sss_df.to_csv('OUTPUT/human_42tfs_k562.10_cv_folds/signed_shap_sum.csv', index=False)
