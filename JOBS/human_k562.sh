#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH -D .
#SBATCH -J tfpr_k562
#SBATCH -o LOG/human_k562_%A.out
#SBATCH -e LOG/human_k562_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiming.kang@wustl.edu

IDS=$( echo $( cut -f1 RESOURCES/TF_list/Human_ENCODE_K562_Valid_TFs.txt ))

python3 -u CODE/explain_human_resps.py \
	-i $IDS \
	-f tf_binding histone_modifications chromatin_accessibility dna_sequence_nt_freq gene_expression gene_variation \
	-x /scratch/mblab/yiming.kang/Pert_Response_Modeling/OUTPUT/h5_data/human_encode_enhan_alltss_2kbto2kb_promo.h5 \
	-y /scratch/mblab/yiming.kang/Pert_Response_Modeling/RESOURCES/Human_TFPert_ENCODE/K562_pertResp_DESeq2_long.csv \
	-o OUTPUT/human_k562/all_feats
