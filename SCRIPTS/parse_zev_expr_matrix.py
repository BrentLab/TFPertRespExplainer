import sys
import pandas as pd
import numpy as np
import time
from utils import com2sys

"""Parse ZEV expression data in tall format into expression matrix: 
gene x tf (or sample). 
Args:
    dirpath     - Path to ZEV data directory.
    time_point  - Time point of which ZEV induction was measured.
    fc_type     - Type of fold change data. Choose from ['cleaned', 'shrunken', 'prepert']

Example:
## Shrunken data at 15min 
python3 HELPER_SCRIPTS/parse_zev_expr_matrix.py RESOURCES/Yeast_ZEV_IDEA/ 15 shrunken RESOURCES/Yeast_genome/orf_name_conversion.tab

## Pre-perturbation data at 0min (log2 ratio of red/green channels)
python3 HELPER_SCRIPTS/parse_zev_expr_matrix.py RESOURCES/Yeast_ZEV_IDEA/ 0 prepert RESOURCES/Yeast_genome/orf_name_conversion.tab

## Pre-perturbation data at 0min (Red channel)
python3 HELPER_SCRIPTS/parse_zev_expr_matrix.py RESOURCES/Yeast_ZEV_IDEA/ 0 prepertRed RESOURCES/Yeast_genome/orf_name_conversion.tab
"""


TF_BLACKLIST = ['Z3EV']
RESTRICTION = 'P'
FC_DICT = {
    'cleaned': ([10], 'ratio'), 
    'shrunken': ([14], 'timecourses'), 
    'prepert': [[7, 8]],
    'prepertRed': [[8]]}

## Input args
dirpath = sys.argv[1]
time_point = int(sys.argv[2]) 
fc_type = sys.argv[3] 
if fc_type not in FC_DICT:
    sys.exit('{} not in {}'.format(fc_type, FC_DICT))
gene_table_filepath = sys.argv[4] if len(sys.argv) > 4 else None

## Load dataframe
t0 = time.time()
df = pd.read_csv(
    '{}/idea_tall_expression_data.tsv'.format(dirpath), sep='\t',
    usecols=list(range(7)) + FC_DICT[fc_type][0])
t1 = time.time()

print('Elapsed loading time = {}'.format(t1 - t0))
print('Loaded dataframe = {}'.format(df.shape))

## Gene name conversion
convert_gene_names = False
if gene_table_filepath is not None: 
    com2sys_dict = com2sys(gene_table_filepath)
    convert_gene_names = True
    ## Allow some other common names
    com2sys_dict.update(
        {'PHO88': 'YBR106W', 'FRA2': 'YGL220W', 'PET10': 'YKR046C', 'OSW5': 'YMR148W'})

## Query time point
df = df.loc[(~df['TF'].isin(TF_BLACKLIST)) & \
            (df['restriction'] == RESTRICTION) & \
            (df['time'] == time_point)]

## Create a output df
if fc_type == 'prepert':
    expr_col = 'log2_r_g_ratio'
    df[expr_col] = np.log2(df['red_median'] / df['green_median'])
elif fc_type == 'prepertRed':
    expr_col = 'red_median'
    df[expr_col] = df[expr_col]
else:
    expr_col = 'log2_{}_{}'.format(fc_type, FC_DICT[fc_type][1])

out_df = pd.DataFrame(index=pd.unique(df['GeneName']))
for tf, tf_df in df.groupby('TF'):
    ## Take geometric mean of FCs if assayed multiple times
    expr_df = tf_df.groupby('GeneName')[expr_col].mean()
    expr_df = expr_df.to_frame().rename(columns={expr_col: tf})
    out_df = out_df.merge(expr_df, left_index=True, right_index=True)

if convert_gene_names:
    out_df = out_df.rename(columns=com2sys_dict, index=com2sys_dict)

## Save output
out_filepath = '{}/ZEV_{}min_{}Data.csv'.format(dirpath, time_point, fc_type)
out_df.to_csv(out_filepath, index_label='GeneName')
print('Saved to {}'.format(out_filepath))
