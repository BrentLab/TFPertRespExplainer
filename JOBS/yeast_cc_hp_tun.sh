#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH -D .
#SBATCH -J tfpr_cc
#SBATCH -o LOG/yeast_cc_hp_tun_%A.out
#SBATCH -e LOG/yeast_cc_hp_tun_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiming.kang@wustl.edu

python3 -u CODE/explain_yeast_resps.py \
	--model_tuning \
	-i YDR034C YEL009C YIL036W YJL056C YJR060W YKL038W YLR403W YLR451W YMR182C YNL199C YOL108C YOR344C YPL075W YPL248C \
	-f tf_binding histone_modifications chromatin_accessibility dna_sequence_nt_freq gene_expression gene_variation \
	-x /scratch/mblab/yiming.kang/Pert_Response_Modeling/OUTPUT/h5_data/yeast_dna_cc_hm_atac_tss1000to500b_expr_var.h5 \
	-y /scratch/mblab/yiming.kang/Pert_Response_Modeling/RESOURCES/Yeast_ZEV_IDEA/ZEV_15min_shrunkenData.csv \
	-o OUTPUT/yeast_cc_hp_tun

python3 -u CODE/explain_yeast_resps.py \
	-i YDR034C YEL009C YIL036W YJR060W YKL038W YLR403W YLR451W YNL199C YOL108C YOR344C YPL075W YPL248C \
	-f tf_binding histone_modifications chromatin_accessibility dna_sequence_nt_freq gene_expression gene_variation \
	-x /scratch/mblab/yiming.kang/Pert_Response_Modeling/OUTPUT/h5_data/yeast_dna_cc_hm_atac_tss1000to500b_expr_var.h5 \
	-y /scratch/mblab/yiming.kang/Pert_Response_Modeling/RESOURCES/Yeast_ZEV_IDEA/ZEV_15min_shrunkenData.csv \
	-o OUTPUT/yeast_cc_hp_base

