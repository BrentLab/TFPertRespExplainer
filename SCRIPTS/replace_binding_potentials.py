import numpy as np
import pandas as pd
import h5py
import glob
from shutil import copyfile

"""Replace binding data with binding potential data.
"""

src_filepath = 'OUTPUT/h5_data/yeast_dna_cc_hm_atac_tss1000to500b_expr_var.h5'
dst_filepath = 'OUTPUT/h5_data/yeast_dna_bp_hm_atac_tss1000to500b_expr_var.h5'
copyfile(src_filepath, dst_filepath)

with h5py.File(dst_filepath, 'r') as f:
    genes = f['genes'][:]
    genes = [x.decode('utf-8') for x in genes]
idx_dict = {x: i for i, x in enumerate(genes)}

with h5py.File(dst_filepath, 'a') as f:
    del f['tf_binding']

with h5py.File(dst_filepath, 'a') as f:
    g = f.create_group('binding_potential')

    for filepath in glob.glob('OUTPUT/Yeast_bp_tss1000to500b/*/fimo.txt'):
        tf = filepath.split('/')[2]
        print('... working on', tf)

        df = pd.read_csv(filepath, sep='\t')[['sequence name', 'start', 'stop', 'score']]
        df['idx'] = [idx_dict[x] for x in df['sequence name']]
        g.create_dataset(tf, data=df[['idx', 'start', 'stop', 'score']].values.astype(float), compression='gzip')
