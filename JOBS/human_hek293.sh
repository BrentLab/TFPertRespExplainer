#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH -D .
#SBATCH -J tfpr_hek293
#SBATCH -o LOG/human_hek293_%A.out
#SBATCH -e LOG/human_hek293_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiming.kang@wustl.edu

IDS=$( echo $( cut -f1 /scratch/mblab/yiming.kang/Pert_Response_Modeling/RESOURCES/TF_list/Human_HEK293_TFs.txt ))

python3 -u CODE/explain_human_resps.py \
	-i $IDS \
	-f tf_binding histone_modifications chromatin_accessibility dna_sequence_nt_freq gene_expression gene_variation \
	-x /scratch/mblab/yiming.kang/Pert_Response_Modeling/OUTPUT/h5_data/human_hek293_enhan_alltss_2kbto2kb_promo.h5 \
	-y /scratch/mblab/yiming.kang/Pert_Response_Modeling/RESOURCES/Human_TFPert_Schmitges2016/GSE76495_OE_log2FC_long.csv \
	-o OUTPUT/human_hek293
