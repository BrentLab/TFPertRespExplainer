#!/bin/bash
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=10
#SBATCH -D .
#SBATCH -J tfpr_yeast
#SBATCH -o LOG/yeast_hp_tun_%A_%a.out
#SBATCH -e LOG/yeast_hp_tun_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiming.kang@wustl.edu

#python3 -u CODE/explain_yeast_resps.py \
#	--model_tuning \
#	-i YAL051W YBL103C YBR239C YEL009C YGL162W YHR178W YJL089W YJR060W YKL038W YLR256W YLR451W YMR280C YNL199C YOL067C YOL108C YOR344C YOR363C YPL075W YPL133C \
#	-f tf_binding histone_modifications chromatin_accessibility dna_sequence_nt_freq gene_expression \
#	-x /scratch/mblab/yiming.kang/Pert_Response_Modeling/OUTPUT/h5_data/yeast_dna_chipexo_hm_atac_tss1000to500b_expr_var.h5 \
#	-y /scratch/mblab/yiming.kang/Pert_Response_Modeling/RESOURCES/Yeast_ZEV_IDEA/ZEV_15min_shrunkenData.csv \
#	-o OUTPUT/yeast_19tfs_chipexo_hp_tun

python3 -u CODE/explain_yeast_resps.py \
	-i YAL051W YBL103C YBR239C YEL009C YGL162W YHR178W YJL089W YJR060W YKL038W YLR256W YLR451W YMR280C YOL067C YOL108C YOR344C YPL075W YPL133C \
	-f tf_binding histone_modifications chromatin_accessibility dna_sequence_nt_freq gene_expression \
	-x /scratch/mblab/yiming.kang/Pert_Response_Modeling/OUTPUT/h5_data/yeast_dna_chipexo_hm_atac_tss1000to500b_expr_var.h5 \
	-y /scratch/mblab/yiming.kang/Pert_Response_Modeling/RESOURCES/Yeast_ZEV_IDEA/ZEV_15min_shrunkenData.csv \
	-o OUTPUT/yeast_19tfs_chipexo_hp_base

