#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH -D .
#SBATCH -J tfpr_chipexo
#SBATCH -o LOG/yeast_chipexo_%A.out
#SBATCH -e LOG/yeast_chipexo_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiming.kang@wustl.edu

IDS=$( echo $( cut -f1 /scratch/mblab/yiming.kang/Pert_Response_Modeling/RESOURCES/TF_list/Yeast_ChIPexo_TFs.txt ))

python3 -u CODE/explain_yeast_resps.py \
	-i $IDS \
	-f tf_binding histone_modifications chromatin_accessibility dna_sequence_nt_freq gene_expression gene_variation \
	-x /scratch/mblab/yiming.kang/Pert_Response_Modeling/OUTPUT/h5_data/yeast_dna_chipexo_hm_atac_tss1000to500b_expr_var.h5 \
	-y /scratch/mblab/yiming.kang/Pert_Response_Modeling/RESOURCES/Yeast_ZEV_IDEA/ZEV_15min_shrunkenData.csv \
	-o OUTPUT/yeast_chipexo
