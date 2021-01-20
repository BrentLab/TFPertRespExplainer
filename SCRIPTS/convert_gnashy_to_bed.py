import sys
import os.path
import glob

sys.path.insert(0, 'CODE/')
from data_preproc_utils import convert_gnashy_to_bed, liftover_bed
from utils import com2sys


"""Liftover transposon calling cards gnashy file mapped to sacCer2 to sacCer3,
then convert it to bed file. Also optionally convert common gene name to 
systematic name. 
Args:
    in_dirpath              - Path to input gnashy data directory.
    out_dirpath             - Path to output bed data directory.
    liftover_filepath       - Liftover file.
    gene_table_filepath     - Gene name lookup table file.

Example:
python3 HELPER_SCRIPTS/convert_gnashy_to_bed.py RESOURCES/Yeast_CallingCards/gnashy/ RESOURCES/Yeast_CallingCards/ RESOURCES/Yeast_genome/V61_2008_06_05_V64_2011_02_03_ChromModified.over.chain RESOURCES/Yeast_genome/orf_name_conversion.tab
"""

## Input args
in_dirpath = sys.argv[1]
out_dirpath = sys.argv[2]
liftover_filepath = sys.argv[3]
gene_table_filepath = sys.argv[4] if len(sys.argv) > 4 else None

## Gene name conversion
if gene_table_filepath is not None: 
    com2sys_dict = com2sys(gene_table_filepath)

## Convert gnashy to bed
filepaths = glob.glob('{}/*.gnashy'.format(in_dirpath))
for filepath in filepaths:
    tf = com2sys_dict[os.path.splitext(os.path.basename(filepath))[0]]
    print('.. working on {}'.format(tf))
    cc_bed2 = convert_gnashy_to_bed(filepath, binarize_peak_score=True)
    cc_bed3 = liftover_bed(cc_bed2, liftover_filepath)
    cc_bed3.saveas('{}/{}.bed'.format(out_dirpath, tf))

## Convert No-TF control gnashy to bed
bg_name = 'NOTF_Control'
filepath = '{}/{}.gnashy'.format(in_dirpath, bg_name)
cc_bed2 = convert_gnashy_to_bed(filepath, binarize_peak_score=True)
cc_bed3 = liftover_bed(cc_bed2, liftover_filepath)
cc_bed3.saveas('{}/{}.bed'.format(out_dirpath, bg_name))
