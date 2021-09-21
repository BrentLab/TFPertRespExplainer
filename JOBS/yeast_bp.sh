#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH -D .
#SBATCH -J tfpr_bp
#SBATCH -o LOG/yeast_bp_%A.out
#SBATCH -e LOG/yeast_bp_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiming.kang@wustl.edu

IDS=$( echo $( cut -f1 RESOURCES/TF_list/Yeast_CC_CE_TFs.txt ))

python3 -u CODE/explain_yeast_resps.py \
	-i $IDS \
	-f binding_potential histone_modifications chromatin_accessibility dna_sequence_nt_freq gene_expression gene_variation \
	-x RESOURCES/h5_data/yeast_dna_bp_hm_atac_tss1000to500b_expr_var.h5 \
	-y RESOURCES/Yeast_ZEV_IDEA/ZEV_15min_shrunkenData.csv \
	-o OUTPUT/yeast_bp/all_feats/
